
"""
vigil/gait.py

Gait analysis module.
Consumes a stream of PoseFrames and computes clinical gait metrics
every stride cycle.

Metrics computed:
  - Gait speed       (pixels/sec → normalized to personal baseline)
  - Stride length    (pixels → normalized)
  - Cadence          (steps/minute)
  - Asymmetry        (% difference left vs right stride timing)
  - Stance width     (hip-ankle lateral distance — balance proxy)

All metrics are pixel-relative. Absolute real-world values require
a depth camera or scene calibration (Phase 4). For longitudinal
anomaly detection, pixel-relative change over time is sufficient.

Usage:
    gait = GaitModule()
    pipeline.add_module(gait)

    # Access latest metrics
    metrics = gait.latest_metrics
"""

import time
import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from data_types import PoseFrame, KP

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Output data structures
# ─────────────────────────────────────────────

@dataclass
class StrideEvent:
    """
    A single detected stride (one full gait cycle for one foot).
    Detected when the ankle reaches its lowest vertical point
    (heel strike approximation in 2D top-down or side-view).
    """
    timestamp: float
    foot: str                    # "left" or "right"
    stride_length_px: float      # pixel distance from last same-foot strike
    stride_duration_sec: float   # time since last same-foot strike
    ankle_x: float               # x position at strike
    ankle_y: float               # y position at strike


@dataclass
class GaitMetrics:
    """
    Computed gait metrics over the most recent analysis window.
    Published after every stride event.
    """
    timestamp: float

    # Core metrics
    speed_px_per_sec: float         # pixels/sec of center-of-mass movement
    stride_length_px: float         # avg stride length in pixels
    cadence_spm: float              # steps per minute
    asymmetry_pct: float            # 0-100%, higher = more asymmetric

    # Balance proxy
    stance_width_px: float          # lateral distance between ankles during stance

    # Confidence — how complete the keypoint data was
    confidence: float               # 0-1, based on keypoint visibility

    # Flags
    is_walking: bool                # True if motion detected in last 2s
    is_shuffling: bool              # True if stride length < 40% of baseline
    slow_gait: bool                 # True if speed < 70% of personal baseline

    # Raw history for debugging
    recent_stride_lengths: list = field(default_factory=list)
    recent_cadences: list = field(default_factory=list)


# ─────────────────────────────────────────────
# Main module
# ─────────────────────────────────────────────

class GaitModule:
    """
    Stateful gait analyzer. Maintains rolling buffers of pose data
    and emits GaitMetrics whenever a stride cycle completes.

    Attach to pipeline:
        gait = GaitModule()
        pipeline.add_module(gait)
    """

    # Tuning constants
    BUFFER_SECONDS = 3.0          # how much pose history to keep
    MIN_FRAMES_FOR_ANALYSIS = 15  # need at least this many frames to compute
    STRIDE_COOLDOWN_SEC = 0.25    # minimum time between detected strides (debounce)
    WALKING_MOTION_THRESH = 8.0   # pixels/frame CoM movement to count as walking
    STILLNESS_TIMEOUT = 2.0       # seconds of no motion before marking not-walking

    def __init__(self, baseline: Optional[dict] = None):
        """
        Args:
            baseline: Optional pre-loaded personal baseline dict with keys:
                      'speed_px_per_sec', 'stride_length_px', 'cadence_spm'
                      If None, baseline is learned from first ~60 seconds of data.
        """
        # Rolling frame buffer — stores (timestamp, keypoints_dict)
        self._frame_buffer: deque = deque()

        # Stride event history
        self._strides: deque = deque(maxlen=20)

        # Per-foot ankle tracking for stride detection
        self._left_ankle_history: deque = deque(maxlen=60)
        self._right_ankle_history: deque = deque(maxlen=60)
        self._last_left_strike: Optional[StrideEvent] = None
        self._last_right_strike: Optional[StrideEvent] = None

        # Center of mass history for speed calculation
        self._com_history: deque = deque(maxlen=90)  # ~3s at 30fps

        # Personal baseline (learned or provided)
        self._baseline = baseline or {}
        self._baseline_samples: list = []
        self._baseline_locked = baseline is not None
        self._baseline_lock_after_n_strides = 20

        # Latest published metrics
        self.latest_metrics: Optional[GaitMetrics] = None

        # State
        self._last_motion_time = time.monotonic()
        self._frame_count = 0

        logger.info("GaitModule initialized" + (" with provided baseline" if baseline else " — will learn baseline"))

    # ─────────────────────────────────────────
    # Pipeline interface
    # ─────────────────────────────────────────

    def update(self, pose: PoseFrame):
        """
        Called by pipeline for every valid PoseFrame.
        This is the main entry point.
        """
        self._frame_count += 1

        # Extract the keypoints we care about
        left_ankle  = pose.get_xy(KP.LEFT_ANKLE)
        right_ankle = pose.get_xy(KP.RIGHT_ANKLE)
        left_hip    = pose.get_xy(KP.LEFT_HIP)
        right_hip   = pose.get_xy(KP.RIGHT_HIP)
        left_knee   = pose.get_xy(KP.LEFT_KNEE)
        right_knee  = pose.get_xy(KP.RIGHT_KNEE)

        # Center of mass approximation — midpoint of hips
        com = pose.midpoint(KP.LEFT_HIP, KP.RIGHT_HIP)

        if com is not None:
            self._com_history.append((pose.timestamp, com))

        # Track ankles for stride detection
        if left_ankle is not None:
            self._left_ankle_history.append((pose.timestamp, left_ankle))
        if right_ankle is not None:
            self._right_ankle_history.append((pose.timestamp, right_ankle))

        # Prune old frames from buffer
        cutoff = pose.timestamp - self.BUFFER_SECONDS
        while self._frame_buffer and self._frame_buffer[0][0] < cutoff:
            self._frame_buffer.popleft()
        self._frame_buffer.append((pose.timestamp, pose))

        # Need enough history before computing
        if self._frame_count < self.MIN_FRAMES_FOR_ANALYSIS:
            return

        # Detect stride events
        self._detect_strides(pose.timestamp)

        # Compute and publish metrics if we have enough strides
        if len(self._strides) >= 2:
            metrics = self._compute_metrics(pose)
            self.latest_metrics = metrics
            self._update_baseline(metrics)
            self._log_metrics(metrics)

    # ─────────────────────────────────────────
    # Stride detection
    # ─────────────────────────────────────────

    def _detect_strides(self, now: float):
        """
        Detect heel strikes using ankle vertical velocity.

        In a side-on or slightly elevated camera view, the ankle
        reaches a local maximum Y value (lowest on screen = highest Y)
        at heel strike. We detect this as a local peak in Y position.

        For a top-down view, we detect the moment the ankle is furthest
        from the body center (max distance from CoM).
        """
        self._detect_foot_strike("left",  self._left_ankle_history,  self._last_left_strike,  now)
        self._detect_foot_strike("right", self._right_ankle_history, self._last_right_strike, now)

    def _detect_foot_strike(self, foot: str, history: deque,
                             last_strike: Optional[StrideEvent], now: float):
        """Detect a strike event for one foot using local peak detection."""
        if len(history) < 5:
            return

        times = [h[0] for h in history]
        positions = [h[1] for h in history]

        # Use Y coordinate for strike detection (works for side view)
        # Use distance from CoM for top-down view
        # We use Y as primary — works for most home camera placements
        y_vals = np.array([p[1] for p in positions])

        # Detect local maximum in Y (lowest point on screen = heel strike)
        # Check if the middle of the recent window is a local max
        mid = len(y_vals) // 2
        if mid < 2 or mid >= len(y_vals) - 2:
            return

        is_peak = (
            y_vals[mid] > y_vals[mid - 1] and
            y_vals[mid] > y_vals[mid - 2] and
            y_vals[mid] > y_vals[mid + 1] and
            y_vals[mid] > y_vals[mid + 2]
        )

        if not is_peak:
            return

        strike_time = times[mid]
        strike_pos = positions[mid]

        # Debounce — ignore if too soon after last strike
        if last_strike and (strike_time - last_strike.timestamp) < self.STRIDE_COOLDOWN_SEC:
            return

        # Ignore strikes too far in the past
        if (now - strike_time) > 1.0:
            return

        # Compute stride length from last same-foot strike
        stride_length = 0.0
        stride_duration = 0.0
        if last_strike:
            dx = strike_pos[0] - last_strike.ankle_x
            dy = strike_pos[1] - last_strike.ankle_y
            stride_length = float(np.sqrt(dx**2 + dy**2))
            stride_duration = strike_time - last_strike.timestamp

            # Sanity check — reject implausible strides
            if stride_duration < 0.3 or stride_duration > 3.0:
                return
            if stride_length < 5.0:  # sub-5px movement is noise
                return

        event = StrideEvent(
            timestamp=strike_time,
            foot=foot,
            stride_length_px=stride_length,
            stride_duration_sec=stride_duration,
            ankle_x=float(strike_pos[0]),
            ankle_y=float(strike_pos[1]),
        )

        self._strides.append(event)

        # Update last strike reference
        if foot == "left":
            self._last_left_strike = event
        else:
            self._last_right_strike = event

        logger.debug(f"Stride detected: {foot} foot | length={stride_length:.1f}px | duration={stride_duration:.2f}s")

    # ─────────────────────────────────────────
    # Metric computation
    # ─────────────────────────────────────────

    def _compute_metrics(self, pose: PoseFrame) -> GaitMetrics:
        now = pose.timestamp

        # ── Gait speed from CoM movement ──
        speed = self._compute_speed()

        # ── Stride length — average of recent strides with valid lengths ──
        valid_strides = [s for s in self._strides if s.stride_length_px > 0]
        stride_length = float(np.median([s.stride_length_px for s in valid_strides])) if valid_strides else 0.0

        # ── Cadence — steps per minute from recent stride durations ──
        cadence = self._compute_cadence()

        # ── Asymmetry — compare left vs right stride timing ──
        asymmetry = self._compute_asymmetry()

        # ── Stance width — lateral distance between ankles ──
        stance_width = self._compute_stance_width(pose)

        # ── Walking detection ──
        is_walking = speed > self.WALKING_MOTION_THRESH

        if is_walking:
            self._last_motion_time = now

        # ── Baseline-relative flags ──
        baseline_speed = self._baseline.get("speed_px_per_sec", None)
        baseline_stride = self._baseline.get("stride_length_px", None)

        slow_gait = (
            baseline_speed is not None and
            speed < (baseline_speed * 0.70) and
            is_walking
        )
        is_shuffling = (
            baseline_stride is not None and
            stride_length > 0 and
            stride_length < (baseline_stride * 0.40)
        )

        # ── Confidence — fraction of key lower-body keypoints visible ──
        lower_body_kps = [KP.LEFT_HIP, KP.RIGHT_HIP, KP.LEFT_KNEE,
                          KP.RIGHT_KNEE, KP.LEFT_ANKLE, KP.RIGHT_ANKLE]
        visible = sum(1 for k in lower_body_kps if pose.get_xy(k) is not None)
        confidence = visible / len(lower_body_kps)

        return GaitMetrics(
            timestamp=now,
            speed_px_per_sec=round(speed, 2),
            stride_length_px=round(stride_length, 2),
            cadence_spm=round(cadence, 1),
            asymmetry_pct=round(asymmetry, 1),
            stance_width_px=round(stance_width, 2),
            confidence=round(confidence, 2),
            is_walking=is_walking,
            is_shuffling=is_shuffling,
            slow_gait=slow_gait,
            recent_stride_lengths=[round(s.stride_length_px, 1) for s in list(self._strides)[-5:]],
            recent_cadences=[],
        )

    def _compute_speed(self) -> float:
        """CoM displacement over the last 1 second."""
        if len(self._com_history) < 2:
            return 0.0

        now = self._com_history[-1][0]
        cutoff = now - 1.0

        recent = [(t, p) for t, p in self._com_history if t >= cutoff]
        if len(recent) < 2:
            return 0.0

        # Total path length over the window
        total_dist = 0.0
        for i in range(1, len(recent)):
            dp = recent[i][1] - recent[i-1][1]
            total_dist += float(np.linalg.norm(dp))

        elapsed = recent[-1][0] - recent[0][0]
        return total_dist / elapsed if elapsed > 0 else 0.0

    def _compute_cadence(self) -> float:
        """Steps per minute from recent stride durations."""
        recent = [s for s in self._strides
                  if s.stride_duration_sec > 0 and
                  (self._strides[-1].timestamp - s.timestamp) < 5.0]
        if not recent:
            return 0.0

        avg_stride_duration = float(np.mean([s.stride_duration_sec for s in recent]))
        # Each stride is one step — cadence = steps/min
        return 60.0 / avg_stride_duration if avg_stride_duration > 0 else 0.0

    def _compute_asymmetry(self) -> float:
        """
        Asymmetry between left and right stride timing.
        Uses the Neurocom asymmetry index: |L-R| / (0.5*(L+R)) * 100
        """
        left_strides  = [s for s in self._strides if s.foot == "left"  and s.stride_duration_sec > 0]
        right_strides = [s for s in self._strides if s.foot == "right" and s.stride_duration_sec > 0]

        if not left_strides or not right_strides:
            return 0.0

        avg_left  = float(np.mean([s.stride_duration_sec for s in left_strides[-3:]]))
        avg_right = float(np.mean([s.stride_duration_sec for s in right_strides[-3:]]))

        denom = 0.5 * (avg_left + avg_right)
        if denom == 0:
            return 0.0

        return abs(avg_left - avg_right) / denom * 100.0

    def _compute_stance_width(self, pose: PoseFrame) -> float:
        """Lateral distance between ankles during stance phase."""
        left  = pose.get_xy(KP.LEFT_ANKLE)
        right = pose.get_xy(KP.RIGHT_ANKLE)
        if left is None or right is None:
            return 0.0
        # Horizontal distance only
        return abs(float(left[0]) - float(right[0]))

    # ─────────────────────────────────────────
    # Baseline learning
    # ─────────────────────────────────────────

    def _update_baseline(self, metrics: GaitMetrics):
        """
        Build personal baseline from first N strides.
        Uses median to be robust against outliers.
        """
        if self._baseline_locked:
            return
        if not metrics.is_walking:
            return

        self._baseline_samples.append({
            "speed": metrics.speed_px_per_sec,
            "stride": metrics.stride_length_px,
            "cadence": metrics.cadence_spm,
        })

        if len(self._baseline_samples) >= self._baseline_lock_after_n_strides:
            self._baseline = {
                "speed_px_per_sec":  float(np.median([s["speed"]   for s in self._baseline_samples])),
                "stride_length_px":  float(np.median([s["stride"]  for s in self._baseline_samples])),
                "cadence_spm":       float(np.median([s["cadence"] for s in self._baseline_samples])),
            }
            self._baseline_locked = True
            logger.info(
                f"Baseline locked after {len(self._baseline_samples)} samples: "
                f"speed={self._baseline['speed_px_per_sec']:.1f}px/s "
                f"stride={self._baseline['stride_length_px']:.1f}px "
                f"cadence={self._baseline['cadence_spm']:.1f}spm"
            )

    # ─────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────

    def _log_metrics(self, m: GaitMetrics):
        flags = []
        if m.slow_gait:    flags.append("SLOW_GAIT")
        if m.is_shuffling: flags.append("SHUFFLING")
        if not m.is_walking: flags.append("STATIONARY")

        flag_str = " ".join(flags) if flags else "nominal"

        logger.info(
            f"GAIT | "
            f"speed={m.speed_px_per_sec:.1f}px/s | "
            f"stride={m.stride_length_px:.1f}px | "
            f"cadence={m.cadence_spm:.1f}spm | "
            f"asym={m.asymmetry_pct:.1f}% | "
            f"conf={m.confidence:.2f} | "
            f"{flag_str}"
        )

    # ─────────────────────────────────────────
    # Accessors
    # ─────────────────────────────────────────

    @property
    def baseline(self) -> dict:
        return self._baseline.copy()

    @property
    def baseline_locked(self) -> bool:
        return self._baseline_locked

    @property
    def stride_count(self) -> int:
        return len(self._strides)