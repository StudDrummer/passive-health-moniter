"""
VIGIL Camera Pipeline — New Gait Metric Additions
Add these sections into activity_monitor.py alongside the existing
cadence, asymmetry, and posture computations.

4 new metrics required by riskScores.ts:
  1. strideVariability     — CoV % of stride-to-stride timing (Parkinson's, Dementia, Neuropathy)
  2. armSwingAsymmetry     — % L/R arm swing amplitude difference (Parkinson's)
  3. stepWidth             — normalized lateral distance between feet in cm (OA, Neuropathy)
  4. cadenceVariability    — CoV % of instantaneous cadence within a session (PD, MS)

Prerequisites: existing YOLOv8-nano-pose pipeline with COCO 17-keypoint output.
COCO keypoint indices used here:
  5/6   = left/right shoulder
  7/8   = left/right elbow
  9/10  = left/right wrist
  11/12 = left/right hip
  15/16 = left/right ankle
"""

from collections import deque
import numpy as np

STRIDE_WINDOW_SEC = 20
ARM_WINDOW_FRAMES = 60
STEP_WIDTH_WINDOW = 30
CADENCE_VAR_WINDOW_SEC = 30


class GaitMetricsState:
    """
    Attach an instance of this to your ActivityMonitor class:
        self.gait = GaitMetricsState(fps=15)
    Then call gait.update(keypoints, frame_idx) each processed frame.
    """
    def __init__(self, fps: int = 15):
        self.fps = fps
        self._stride_window = int(STRIDE_WINDOW_SEC * fps)
        self._left_ankle_y_history: deque = deque(maxlen=self._stride_window)
        self._right_ankle_y_history: deque = deque(maxlen=self._stride_window)
        self._left_contact_frames: list = []
        self._right_contact_frames: list = []
        self._left_in_contact = False
        self._right_in_contact = False
        self._stride_times_left: deque = deque(maxlen=20)
        self._stride_times_right: deque = deque(maxlen=20)
        self._arm_window = ARM_WINDOW_FRAMES
        self._left_wrist_y: deque = deque(maxlen=self._arm_window)
        self._right_wrist_y: deque = deque(maxlen=self._arm_window)
        self._step_widths: deque = deque(maxlen=STEP_WIDTH_WINDOW)
        self._cadence_window = int(CADENCE_VAR_WINDOW_SEC * fps)
        self._cadence_history: deque = deque(maxlen=self._cadence_window)
        self.stride_variability: float = 0.0
        self.arm_swing_asymmetry: float = 0.0
        self.step_width: float = 0.0
        self.cadence_variability: float = 0.0

    def _coeff_of_variation(self, values) -> float:
        arr = list(values)
        if len(arr) < 3:
            return 0.0
        m = np.mean(arr)
        return float((np.std(arr, ddof=1) / m) * 100) if m > 0 else 0.0

    def update(self, keypoints: np.ndarray, frame_idx: int,
               current_cadence: float, frame_height: int):
        CONF_THRESHOLD = 0.4

        def kp(idx):
            if keypoints[idx, 2] >= CONF_THRESHOLD:
                return keypoints[idx, 0], keypoints[idx, 1]
            return None

        la = kp(15)
        ra = kp(16)
        lh = kp(11)
        rh = kp(12)

        if la and lh:
            hip_ankle_dist = abs(la[1] - lh[1])
            is_left_contact = la[1] > lh[1] + hip_ankle_dist * 0.85
            if is_left_contact and not self._left_in_contact:
                self._left_contact_frames.append(frame_idx)
                if len(self._left_contact_frames) >= 2:
                    stride_sec = (self._left_contact_frames[-1] - self._left_contact_frames[-2]) / self.fps
                    if 0.5 <= stride_sec <= 2.5:
                        self._stride_times_left.append(stride_sec)
            self._left_in_contact = is_left_contact

        if ra and rh:
            hip_ankle_dist = abs(ra[1] - rh[1])
            is_right_contact = ra[1] > rh[1] + hip_ankle_dist * 0.85
            if is_right_contact and not self._right_in_contact:
                self._right_contact_frames.append(frame_idx)
                if len(self._right_contact_frames) >= 2:
                    stride_sec = (self._right_contact_frames[-1] - self._right_contact_frames[-2]) / self.fps
                    if 0.5 <= stride_sec <= 2.5:
                        self._stride_times_right.append(stride_sec)
            self._right_in_contact = is_right_contact

        all_strides = list(self._stride_times_left) + list(self._stride_times_right)
        if len(all_strides) >= 6:
            self.stride_variability = self._coeff_of_variation(all_strides)

        lw = kp(9)
        rw = kp(10)
        le = kp(7)
        re = kp(8)

        if lw and le:
            self._left_wrist_y.append(lw[1] - le[1])
        if rw and re:
            self._right_wrist_y.append(rw[1] - re[1])

        if len(self._left_wrist_y) >= 20 and len(self._right_wrist_y) >= 20:
            left_amp = float(np.ptp(list(self._left_wrist_y)))
            right_amp = float(np.ptp(list(self._right_wrist_y)))
            total = left_amp + right_amp
            if total > 0:
                self.arm_swing_asymmetry = abs(left_amp - right_amp) / total * 100

        if la and ra and frame_height > 0:
            lateral_px = abs(la[0] - ra[0])
            lateral_normalized = (lateral_px / frame_height) * 100
            if 2.0 <= lateral_normalized <= 40.0:
                self._step_widths.append(lateral_normalized)
                if len(self._step_widths) >= 5:
                    self.step_width = float(np.median(list(self._step_widths)))

        if current_cadence > 0:
            self._cadence_history.append(current_cadence)
        if len(self._cadence_history) >= 30:
            self.cadence_variability = self._coeff_of_variation(self._cadence_history)

    @property
    def as_dict(self) -> dict:
        return {
            "strideVariability": round(self.stride_variability, 2),
            "armSwingAsymmetry": round(self.arm_swing_asymmetry, 2),
            "stepWidth": round(self.step_width, 2),
            "cadenceVariability": round(self.cadence_variability, 2),
        }


# Integration:
# 1. from camera_metrics_additions import GaitMetricsState
# 2. In ActivityMonitor.__init__(): self.gait = GaitMetricsState(fps=self.fps)
# 3. In processing loop: self.gait.update(keypoints, frame_idx, current_cadence, frame.shape[0])
# 4. In metrics payload: payload.update(self.gait.as_dict)
