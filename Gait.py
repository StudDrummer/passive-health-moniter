"""
vigil/gait.py  — v2

Robust gait analysis module.

v1 problems solved:
  - Single-axis Y peak detection failed for non-side-view cameras
  - Too sensitive to noisy keypoint readings
  - No smoothing — single bad frame broke detection

v2 approach:
  - Camera-agnostic stride detection using ankle VELOCITY not position
    Heel strike = ankle decelerates to near-zero after a swing phase
    This works for side view, angled view, and front/back view
  - Savitzky-Golay smoothing on ankle trajectories before peak detection
  - Confidence gating — low confidence keypoints filtered, not discarded
  - Adaptive stride threshold based on observed body scale (hip-ankle distance)
  - Automatically detects camera orientation (side/front/angled)
  - Step regularity metric via autocorrelation (Parkinson's/freezing marker)
"""

import time
import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from scipy.signal import find_peaks, savgol_filter

from data_types import PoseFrame, KP

logger = logging.getLogger(__name__)


@dataclass
class StrideEvent:
    timestamp: float
    foot: str
    stride_length_px: float
    stride_length_normalized: float  # stride / body_scale
    stride_duration_sec: float
    ankle_x: float
    ankle_y: float
    peak_velocity: float


@dataclass
class GaitMetrics:
    timestamp: float
    speed_px_per_sec: float
    speed_normalized: float
    stride_length_px: float
    stride_length_normalized: float
    cadence_spm: float
    asymmetry_pct: float
    stance_width_px: float
    step_regularity: float
    body_scale_px: float
    camera_mode: str
    keypoint_confidence: float
    is_walking: bool
    is_shuffling: bool
    slow_gait: bool
    high_asymmetry: bool
    recent_stride_lengths: list = field(default_factory=list)


class GaitModule:
    """
    Camera-agnostic gait analyzer using ankle velocity profiles.
    Automatically adapts to side, front, and angled camera placements.
    """

    HISTORY_SECONDS = 4.0
    MIN_FRAMES = 20
    STRIDE_COOLDOWN_SEC = 0.25
    MIN_SWING_VELOCITY = 30.0     # px/sec minimum ankle speed to count as swing
    SMOOTHING_WINDOW = 7
    SMOOTHING_POLY = 2
    BASELINE_STRIDES = 15

    def __init__(self, baseline: Optional[dict] = None):
        self._left_ankle:  deque = deque(maxlen=120)
        self._right_ankle: deque = deque(maxlen=120)
        self._com:         deque = deque(maxlen=120)
        self._body_scale:  deque = deque(maxlen=30)

        self._strides:     deque = deque(maxlen=30)
        self._last_left:   Optional[StrideEvent] = None
        self._last_right:  Optional[StrideEvent] = None

        self._y_range_history: deque = deque(maxlen=60)
        self._x_range_history: deque = deque(maxlen=60)
        self._camera_mode = "unknown"

        self._baseline = baseline or {}
        self._baseline_samples: list = []
        self._baseline_locked = baseline is not None

        self.latest_metrics: Optional[GaitMetrics] = None
        self._frame_count = 0

        logger.info(f"GaitModule v2 | baseline={'provided' if baseline else 'learning'}")

    # ─────────────────────────────────────────
    # Pipeline entry point
    # ─────────────────────────────────────────

    def update(self, pose: PoseFrame):
        self._frame_count += 1
        now = pose.timestamp

        la = pose.get_xy(KP.LEFT_ANKLE)
        ra = pose.get_xy(KP.RIGHT_ANKLE)
        lh = pose.get_xy(KP.LEFT_HIP)
        rh = pose.get_xy(KP.RIGHT_HIP)

        la_conf = pose.get(KP.LEFT_ANKLE).confidence
        ra_conf = pose.get(KP.RIGHT_ANKLE).confidence

        # Body scale
        if lh is not None and la is not None:
            scale = float(np.linalg.norm(la - lh))
            if scale > 20:
                self._body_scale.append(scale)

        body_scale = float(np.median(self._body_scale)) if self._body_scale else 200.0

        # CoM
        com = pose.midpoint(KP.LEFT_HIP, KP.RIGHT_HIP)
        if com is not None:
            self._com.append((now, com))

        # Ankle history — accept lower confidence than before (0.15 vs 0.3)
        if la is not None and la_conf > 0.15:
            self._left_ankle.append((now, la[0], la[1], la_conf))
        if ra is not None and ra_conf > 0.15:
            self._right_ankle.append((now, ra[0], ra[1], ra_conf))

        if self._frame_count < self.MIN_FRAMES:
            return

        self._update_camera_mode()
        self._detect_strides(now, body_scale)

        if len(self._strides) >= 1:
            metrics = self._compute_metrics(pose, body_scale)
            self.latest_metrics = metrics
            self._update_baseline(metrics)
            self._log_metrics(metrics)

    # ─────────────────────────────────────────
    # Camera orientation detection
    # ─────────────────────────────────────────

    def _update_camera_mode(self):
        if len(self._left_ankle) < 10:
            return
        recent = list(self._left_ankle)[-20:]
        y_range = float(np.ptp([r[2] for r in recent]))
        x_range = float(np.ptp([r[1] for r in recent]))
        self._y_range_history.append(y_range)
        self._x_range_history.append(x_range)
        avg_y = float(np.mean(self._y_range_history))
        avg_x = float(np.mean(self._x_range_history))

        if avg_y > avg_x * 1.5:
            self._camera_mode = "side"
        elif avg_x > avg_y * 1.5:
            self._camera_mode = "front"
        else:
            self._camera_mode = "angled"

    # ─────────────────────────────────────────
    # Velocity-based stride detection
    # ─────────────────────────────────────────

    def _detect_strides(self, now: float, body_scale: float):
        self._process_foot("left",  self._left_ankle,  self._last_left,  now, body_scale)
        self._process_foot("right", self._right_ankle, self._last_right, now, body_scale)

    def _process_foot(self, foot: str, history: deque,
                      last: Optional[StrideEvent], now: float, body_scale: float):
        if len(history) < self.SMOOTHING_WINDOW + 4:
            return

        data  = list(history)
        times = np.array([d[0] for d in data])
        xs    = np.array([d[1] for d in data])
        ys    = np.array([d[2] for d in data])

        # Smooth
        win = min(self.SMOOTHING_WINDOW, len(data) - 1)
        if win % 2 == 0:
            win -= 1
        if win < 3:
            return
        try:
            xs_s = savgol_filter(xs, win, self.SMOOTHING_POLY)
            ys_s = savgol_filter(ys, win, self.SMOOTHING_POLY)
        except Exception:
            xs_s, ys_s = xs, ys

        # Velocity
        dt = np.diff(times)
        dt = np.where(dt < 1e-6, 1e-6, dt)
        vx = np.diff(xs_s) / dt
        vy = np.diff(ys_s) / dt
        speed = np.sqrt(vx**2 + vy**2)

        # Choose signal axis based on camera mode
        if self._camera_mode == "side":
            signal = np.abs(vy)
        elif self._camera_mode == "front":
            signal = np.abs(vx)
        else:
            signal = speed

        if len(signal) < 5:
            return

        avg_dt = float(np.mean(dt))
        min_dist = max(3, int(self.STRIDE_COOLDOWN_SEC / avg_dt))

        peaks, _ = find_peaks(
            signal,
            height=self.MIN_SWING_VELOCITY,
            distance=min_dist,
            prominence=10.0,
        )

        for peak_idx in peaks:
            # Find heel strike = first trough after velocity peak
            post = signal[peak_idx:]
            troughs = np.where(post < self.MIN_SWING_VELOCITY * 0.3)[0]
            strike_idx = peak_idx + (troughs[0] if len(troughs) > 0 else 0)
            strike_idx = min(strike_idx, len(times) - 1)

            strike_time = times[strike_idx]
            strike_x    = xs_s[strike_idx] if strike_idx < len(xs_s) else xs_s[-1]
            strike_y    = ys_s[strike_idx] if strike_idx < len(ys_s) else ys_s[-1]

            # Debounce
            if last and (strike_time - last.timestamp) < self.STRIDE_COOLDOWN_SEC:
                continue
            if (now - strike_time) > 2.0:
                continue
            if self._strides and abs(self._strides[-1].timestamp - strike_time) < self.STRIDE_COOLDOWN_SEC:
                continue

            # Stride metrics
            stride_px = stride_norm = stride_dur = 0.0
            if last:
                dx = strike_x - last.ankle_x
                dy = strike_y - last.ankle_y
                stride_px   = float(np.sqrt(dx**2 + dy**2))
                stride_norm = stride_px / body_scale if body_scale > 0 else 0
                stride_dur  = strike_time - last.timestamp
                if stride_dur < 0.25 or stride_dur > 4.0:
                    continue
                if stride_px < body_scale * 0.05:
                    continue

            event = StrideEvent(
                timestamp=strike_time,
                foot=foot,
                stride_length_px=stride_px,
                stride_length_normalized=stride_norm,
                stride_duration_sec=stride_dur,
                ankle_x=float(strike_x),
                ankle_y=float(strike_y),
                peak_velocity=float(signal[peak_idx]),
            )
            self._strides.append(event)

            if foot == "left":
                self._last_left = event
            else:
                self._last_right = event

            logger.debug(
                f"Stride: {foot} | "
                f"len={stride_px:.0f}px ({stride_norm:.2f}x) | "
                f"dur={stride_dur:.2f}s | "
                f"vel={signal[peak_idx]:.0f}px/s | "
                f"cam={self._camera_mode}"
            )

    # ─────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────

    def _compute_metrics(self, pose: PoseFrame, body_scale: float) -> GaitMetrics:
        now = pose.timestamp
        speed = self._compute_speed()
        speed_norm = speed / body_scale if body_scale > 0 else 0

        valid = [s for s in self._strides if s.stride_length_px > 0]
        stride_px   = float(np.median([s.stride_length_px for s in valid])) if valid else 0.0
        stride_norm = float(np.median([s.stride_length_normalized for s in valid])) if valid else 0.0

        cadence    = self._compute_cadence()
        asymmetry  = self._compute_asymmetry()
        stance     = self._compute_stance_width(pose)
        regularity = self._compute_regularity()
        is_walking = speed > 15.0

        bl_speed  = self._baseline.get("speed_normalized")
        bl_stride = self._baseline.get("stride_length_normalized")
        slow_gait    = bl_speed  is not None and speed_norm  < bl_speed  * 0.70 and is_walking
        is_shuffling = bl_stride is not None and stride_norm > 0 and stride_norm < bl_stride * 0.40
        high_asym    = asymmetry > 25.0

        lower_kps = [KP.LEFT_HIP, KP.RIGHT_HIP, KP.LEFT_KNEE, KP.RIGHT_KNEE, KP.LEFT_ANKLE, KP.RIGHT_ANKLE]
        visible = sum(1 for k in lower_kps if pose.get(k).confidence > 0.3)
        kp_conf = visible / len(lower_kps)

        return GaitMetrics(
            timestamp=now,
            speed_px_per_sec=round(speed, 1),
            speed_normalized=round(speed_norm, 3),
            stride_length_px=round(stride_px, 1),
            stride_length_normalized=round(stride_norm, 3),
            cadence_spm=round(cadence, 1),
            asymmetry_pct=round(asymmetry, 1),
            stance_width_px=round(stance, 1),
            step_regularity=round(regularity, 3),
            body_scale_px=round(body_scale, 1),
            camera_mode=self._camera_mode,
            keypoint_confidence=round(kp_conf, 2),
            is_walking=is_walking,
            is_shuffling=is_shuffling,
            slow_gait=slow_gait,
            high_asymmetry=high_asym,
            recent_stride_lengths=[round(s.stride_length_normalized, 3) for s in list(self._strides)[-6:]],
        )

    def _compute_speed(self) -> float:
        if len(self._com) < 2:
            return 0.0
        now = self._com[-1][0]
        w = [(t, p) for t, p in self._com if t >= now - 1.5]
        if len(w) < 2:
            return 0.0
        dist = sum(float(np.linalg.norm(w[i][1] - w[i-1][1])) for i in range(1, len(w)))
        elapsed = w[-1][0] - w[0][0]
        return dist / elapsed if elapsed > 0 else 0.0

    def _compute_cadence(self) -> float:
        recent = [s for s in self._strides
                  if s.stride_duration_sec > 0 and
                  (self._strides[-1].timestamp - s.timestamp) < 8.0]
        if not recent:
            return 0.0
        avg = float(np.mean([s.stride_duration_sec for s in recent]))
        return 60.0 / avg if avg > 0 else 0.0

    def _compute_asymmetry(self) -> float:
        left  = [s for s in self._strides if s.foot == "left"  and s.stride_duration_sec > 0]
        right = [s for s in self._strides if s.foot == "right" and s.stride_duration_sec > 0]
        if not left or not right:
            return 0.0
        l = float(np.mean([s.stride_duration_sec for s in left[-4:]]))
        r = float(np.mean([s.stride_duration_sec for s in right[-4:]]))
        denom = 0.5 * (l + r)
        return abs(l - r) / denom * 100.0 if denom > 0 else 0.0

    def _compute_stance_width(self, pose: PoseFrame) -> float:
        la = pose.get_xy(KP.LEFT_ANKLE)
        ra = pose.get_xy(KP.RIGHT_ANKLE)
        if la is None or ra is None:
            return 0.0
        return abs(float(la[0]) - float(ra[0]))

    def _compute_regularity(self) -> float:
        durs = [s.stride_duration_sec for s in self._strides if s.stride_duration_sec > 0]
        if len(durs) < 4:
            return 1.0
        d = np.array(durs) - np.mean(durs)
        if d.std() < 1e-6:
            return 1.0
        ac = float(np.corrcoef(d[:-1], d[1:])[0, 1])
        return max(0.0, min(1.0, (ac + 1.0) / 2.0))

    # ─────────────────────────────────────────
    # Baseline
    # ─────────────────────────────────────────

    def _update_baseline(self, m: GaitMetrics):
        if self._baseline_locked or not m.is_walking:
            return
        if m.speed_normalized <= 0 or m.stride_length_normalized <= 0:
            return
        self._baseline_samples.append({
            "speed_normalized":  m.speed_normalized,
            "stride_normalized": m.stride_length_normalized,
            "cadence":           m.cadence_spm,
        })
        if len(self._baseline_samples) >= self.BASELINE_STRIDES:
            self._baseline = {
                "speed_normalized":         float(np.median([s["speed_normalized"]  for s in self._baseline_samples])),
                "stride_length_normalized": float(np.median([s["stride_normalized"] for s in self._baseline_samples])),
                "cadence_spm":              float(np.median([s["cadence"]           for s in self._baseline_samples])),
            }
            self._baseline_locked = True
            logger.info(f"Baseline locked | {self._baseline}")

    def _log_metrics(self, m: GaitMetrics):
        flags = []
        if m.slow_gait:      flags.append("SLOW_GAIT")
        if m.is_shuffling:   flags.append("SHUFFLING")
        if m.high_asymmetry: flags.append("HIGH_ASYMMETRY")
        if not m.is_walking: flags.append("STATIONARY")
        logger.info(
            f"GAIT | cam={m.camera_mode} | "
            f"speed={m.speed_px_per_sec:.0f}px/s ({m.speed_normalized:.2f}x) | "
            f"stride={m.stride_length_normalized:.2f}x | "
            f"cadence={m.cadence_spm:.0f}spm | "
            f"asym={m.asymmetry_pct:.0f}% | "
            f"reg={m.step_regularity:.2f} | "
            f"strides={self.stride_count} | "
            f"{' '.join(flags) if flags else 'nominal'}"
        )

    @property
    def baseline(self) -> dict:
        return self._baseline.copy()

    @property
    def baseline_locked(self) -> bool:
        return self._baseline_locked

    @property
    def stride_count(self) -> int:
        return len(self._strides)

    @property
    def camera_mode(self) -> str:
        return self._camera_mode