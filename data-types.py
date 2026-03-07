"""
vigil/core/types.py

Central data types for the entire pipeline.
Every module communicates using these structures — never raw numpy arrays.
This ensures the signal extraction, baseline, and anomaly layers
are all working with the same schema.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum
import time
import numpy as np


# ─────────────────────────────────────────────
# COCO 17-keypoint skeleton definition
# YOLOv8-Pose outputs keypoints in this order
# ─────────────────────────────────────────────

class KP(IntEnum):
    """COCO keypoint indices. Use these names everywhere — never raw integers."""
    NOSE            = 0
    LEFT_EYE        = 1
    RIGHT_EYE       = 2
    LEFT_EAR        = 3
    RIGHT_EAR       = 4
    LEFT_SHOULDER   = 5
    RIGHT_SHOULDER  = 6
    LEFT_ELBOW      = 7
    RIGHT_ELBOW     = 8
    LEFT_WRIST      = 9
    RIGHT_WRIST     = 10
    LEFT_HIP        = 11
    RIGHT_HIP       = 12
    LEFT_KNEE       = 13
    RIGHT_KNEE      = 14
    LEFT_ANKLE      = 15
    RIGHT_ANKLE     = 16


# Skeleton edges for visualization
SKELETON_EDGES = [
    (KP.LEFT_SHOULDER,  KP.RIGHT_SHOULDER),
    (KP.LEFT_SHOULDER,  KP.LEFT_ELBOW),
    (KP.LEFT_ELBOW,     KP.LEFT_WRIST),
    (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW),
    (KP.RIGHT_ELBOW,    KP.RIGHT_WRIST),
    (KP.LEFT_SHOULDER,  KP.LEFT_HIP),
    (KP.RIGHT_SHOULDER, KP.RIGHT_HIP),
    (KP.LEFT_HIP,       KP.RIGHT_HIP),
    (KP.LEFT_HIP,       KP.LEFT_KNEE),
    (KP.LEFT_KNEE,      KP.LEFT_ANKLE),
    (KP.RIGHT_HIP,      KP.RIGHT_KNEE),
    (KP.RIGHT_KNEE,     KP.RIGHT_ANKLE),
    (KP.NOSE,           KP.LEFT_EYE),
    (KP.NOSE,           KP.RIGHT_EYE),
]


# ─────────────────────────────────────────────
# Core data structures
# ─────────────────────────────────────────────

@dataclass
class Keypoint:
    """
    Single keypoint from pose estimation.
    x, y are pixel coordinates (0..frame_width, 0..frame_height).
    confidence is the model's visibility score (0..1).
    """
    x: float
    y: float
    confidence: float

    def is_valid(self, min_confidence: float = 0.3) -> bool:
        return self.confidence >= min_confidence

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class PoseFrame:
    """
    A single pose estimation result for one frame.
    This is the PRIMARY output of the ingestion layer.
    Downstream modules consume streams of PoseFrames.

    NOTE: No image data is stored here. Only numbers.
    """
    # Monotonic timestamp in seconds (time.monotonic())
    timestamp: float

    # Wall clock time for logging/storage
    wall_time: float

    # Frame index since pipeline start
    frame_index: int

    # 17 keypoints in COCO order (indexed by KP enum)
    keypoints: list[Keypoint]

    # Bounding box of detected person [x1, y1, x2, y2] in pixels
    bbox: Optional[tuple[float, float, float, float]] = None

    # Overall detection confidence
    detection_confidence: float = 0.0

    # Person tracking ID (stable across frames)
    track_id: Optional[int] = None

    # Frame dimensions (needed to normalize coordinates downstream)
    frame_width: int = 1280
    frame_height: int = 720

    def get(self, keypoint: KP) -> Keypoint:
        """Retrieve a keypoint by name. e.g. frame.get(KP.LEFT_ANKLE)"""
        return self.keypoints[int(keypoint)]

    def get_xy(self, keypoint: KP) -> Optional[np.ndarray]:
        """Return (x, y) array if keypoint is valid, else None."""
        kp = self.get(keypoint)
        return kp.as_array() if kp.is_valid() else None

    def midpoint(self, kp_a: KP, kp_b: KP) -> Optional[np.ndarray]:
        """Midpoint between two keypoints. Returns None if either is invalid."""
        a = self.get_xy(kp_a)
        b = self.get_xy(kp_b)
        if a is None or b is None:
            return None
        return (a + b) / 2.0

    def normalized(self) -> "PoseFrame":
        """
        Return a copy with keypoints normalized to [0, 1] range.
        Useful for model input and cross-resolution comparison.
        """
        norm_kps = [
            Keypoint(
                x=kp.x / self.frame_width,
                y=kp.y / self.frame_height,
                confidence=kp.confidence,
            )
            for kp in self.keypoints
        ]
        import copy
        f = copy.copy(self)
        f.keypoints = norm_kps
        return f

    def visible_keypoint_count(self, min_confidence: float = 0.3) -> int:
        return sum(1 for kp in self.keypoints if kp.is_valid(min_confidence))

    def is_good_quality(self, min_visible: int = 10) -> bool:
        """Quick quality gate — reject partial detections."""
        return (
            self.detection_confidence > 0.4
            and self.visible_keypoint_count() >= min_visible
        )


@dataclass
class PipelineStats:
    """Rolling performance stats — logged periodically."""
    frames_processed: int = 0
    frames_dropped: int = 0
    inference_count: int = 0
    avg_inference_ms: float = 0.0
    avg_fps: float = 0.0
    last_detection_time: float = field(default_factory=time.monotonic)
    person_detected: bool = False
