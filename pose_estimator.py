"""
vigil/core/pose_estimator.py

YOLOv8-Pose wrapper.
Takes raw BGR frames, returns PoseFrame objects with 17 COCO keypoints.

Key design decisions:
  - Runs inference every N frames (configurable) for compute efficiency
  - Interpolates keypoints between inference frames for smooth downstream signals
  - Uses ByteTrack (built into Ultralytics) for stable person tracking IDs
  - Never stores or forwards image data — only numbers leave this module
  - Falls back gracefully when no person is detected
"""

import time
import logging
import numpy as np
from typing import Optional
from collections import deque

from core.types import PoseFrame, Keypoint, KP, PipelineStats
from core.ingestion import CapturedFrame

logger = logging.getLogger(__name__)


class PoseEstimator:
    """
    Wraps YOLOv8-Pose for the Vigil pipeline.

    Usage:
        estimator = PoseEstimator(config)
        estimator.load()

        for raw_frame in ingestion.frames():
            pose_frame = estimator.process(raw_frame)
            if pose_frame:
                downstream_modules(pose_frame)
    """

    NUM_KEYPOINTS = 17  # COCO skeleton

    def __init__(self, config: dict):
        self.config = config
        self.pose_cfg = config["pose"]
        self.inf_cfg = config["inference"]
        self.privacy_cfg = config["privacy"]

        self._model = None
        self._loaded = False

        # Inference throttling
        self._infer_every = self.inf_cfg.get("inference_every_n_frames", 2)
        self._frame_count = 0

        # Last good pose result — used for interpolation between inference frames
        self._last_pose: Optional[PoseFrame] = None
        self._prev_pose: Optional[PoseFrame] = None

        # Rolling inference time tracking
        self._inference_times: deque = deque(maxlen=30)

        # Stats
        self.stats = PipelineStats()

    # ──────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────

    def load(self) -> "PoseEstimator":
        """
        Load YOLOv8-Pose model. Downloads automatically on first run (~6MB for nano).
        Call once before the processing loop.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed.\n"
                "Run: pip install ultralytics"
            )

        model_path = self.pose_cfg["model"]
        device = self.pose_cfg["device"]

        logger.info(f"Loading YOLOv8-Pose model: {model_path} on {device}")
        t0 = time.monotonic()

        self._model = YOLO(model_path)

        # Warmup pass — first inference is always slow due to CUDA JIT
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.predict(
            dummy,
            device=device,
            verbose=False,
            half=self.inf_cfg.get("half_precision", True),
        )

        elapsed = (time.monotonic() - t0) * 1000
        logger.info(f"Model loaded and warmed up in {elapsed:.0f}ms")
        self._loaded = True
        return self

    # ──────────────────────────────────────────
    # Main processing entry point
    # ──────────────────────────────────────────

    def process(self, frame: CapturedFrame) -> Optional[PoseFrame]:
        """
        Process one captured frame.

        Returns:
            PoseFrame if a person was detected, None otherwise.
            On non-inference frames, returns interpolated pose for smooth signals.
        """
        if not self._loaded:
            raise RuntimeError("Call .load() before .process()")

        self._frame_count += 1
        self.stats.frames_processed += 1

        should_infer = (self._frame_count % self._infer_every == 0)

        if should_infer:
            pose = self._run_inference(frame)
            self._prev_pose = self._last_pose
            self._last_pose = pose
            return pose
        else:
            # Return interpolated pose between last two inference results
            return self._interpolate(frame)

    # ──────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────

    def _run_inference(self, frame: CapturedFrame) -> Optional[PoseFrame]:
        t0 = time.monotonic()

        results = self._model.track(
            source=frame.bgr,
            device=self.pose_cfg["device"],
            conf=self.pose_cfg["confidence_threshold"],
            iou=self.pose_cfg["iou_threshold"],
            imgsz=self.inf_cfg["input_size"],
            half=self.inf_cfg.get("half_precision", True),
            persist=True,        # maintain track IDs across frames
            verbose=False,
            classes=[0],         # person class only
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._inference_times.append(elapsed_ms)
        self.stats.inference_count += 1
        self.stats.avg_inference_ms = float(np.mean(self._inference_times))

        logger.debug(f"Inference: {elapsed_ms:.1f}ms | avg: {self.stats.avg_inference_ms:.1f}ms")

        pose = self._parse_results(results, frame)

        if pose:
            self.stats.person_detected = True
            self.stats.last_detection_time = frame.timestamp
        else:
            # If no detection for >5s, flag as absent
            if (frame.timestamp - self.stats.last_detection_time) > 5.0:
                self.stats.person_detected = False

        return pose

    def _parse_results(self, results, frame: CapturedFrame) -> Optional[PoseFrame]:
        """
        Extract the primary person's pose from YOLO results.
        Selects the largest bounding box (assumed to be the resident).
        """
        if not results or len(results) == 0:
            return None

        result = results[0]

        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        # Pick the primary person — largest bounding box area
        best_idx = self._select_primary_person(result)
        if best_idx is None:
            return None

        # ── Keypoints ──
        kp_data = result.keypoints.data[best_idx].cpu().numpy()  # shape: (17, 3) — x, y, conf
        kp_conf_min = self.pose_cfg["keypoint_confidence_min"]

        keypoints = []
        for i in range(self.NUM_KEYPOINTS):
            x, y, conf = float(kp_data[i, 0]), float(kp_data[i, 1]), float(kp_data[i, 2])
            # Suppress low-confidence keypoints to (0, 0, 0) so callers can gate on is_valid()
            if conf < kp_conf_min:
                keypoints.append(Keypoint(x=0.0, y=0.0, confidence=0.0))
            else:
                keypoints.append(Keypoint(x=x, y=y, confidence=conf))

        # ── Bounding box ──
        bbox = None
        det_conf = 0.0
        if result.boxes is not None and len(result.boxes) > best_idx:
            box = result.boxes[best_idx]
            xyxy = box.xyxy[0].cpu().numpy()
            bbox = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
            det_conf = float(box.conf[0])

        # ── Track ID ──
        track_id = None
        if result.boxes is not None and result.boxes.id is not None:
            try:
                track_id = int(result.boxes.id[best_idx])
            except (IndexError, TypeError):
                pass

        pose = PoseFrame(
            timestamp=frame.timestamp,
            wall_time=frame.wall_time,
            frame_index=frame.frame_index,
            keypoints=keypoints,
            bbox=bbox,
            detection_confidence=det_conf,
            track_id=track_id,
            frame_width=frame.width,
            frame_height=frame.height,
        )

        return pose if pose.is_good_quality() else None

    def _select_primary_person(self, result) -> Optional[int]:
        """
        When multiple people are in frame, select the primary resident.
        Strategy: largest bounding box area (closest to camera).
        Future: add room zone filtering using depth camera.
        """
        if result.boxes is None or len(result.boxes) == 0:
            return None

        best_idx = 0
        best_area = -1.0

        for i, box in enumerate(result.boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            area = float(w * h)
            if area > best_area:
                best_area = area
                best_idx = i

        return best_idx

    # ──────────────────────────────────────────
    # Interpolation
    # ──────────────────────────────────────────

    def _interpolate(self, frame: CapturedFrame) -> Optional[PoseFrame]:
        """
        Linear interpolation between the two most recent inference poses.
        Produces smooth signals at full frame rate without running inference every frame.
        """
        if self._last_pose is None:
            return None
        if self._prev_pose is None:
            return self._last_pose

        # Alpha: 0 = prev, 1 = last
        dt_total = self._last_pose.timestamp - self._prev_pose.timestamp
        if dt_total <= 0:
            return self._last_pose

        dt_current = frame.timestamp - self._prev_pose.timestamp
        alpha = min(max(dt_current / dt_total, 0.0), 1.0)

        interp_kps = []
        for i in range(self.NUM_KEYPOINTS):
            kp_prev = self._prev_pose.keypoints[i]
            kp_last = self._last_pose.keypoints[i]

            # Only interpolate if both endpoints are valid
            if kp_prev.is_valid() and kp_last.is_valid():
                interp_kps.append(Keypoint(
                    x=kp_prev.x + alpha * (kp_last.x - kp_prev.x),
                    y=kp_prev.y + alpha * (kp_last.y - kp_prev.y),
                    confidence=min(kp_prev.confidence, kp_last.confidence),
                ))
            else:
                interp_kps.append(kp_last if kp_last.is_valid() else kp_prev)

        return PoseFrame(
            timestamp=frame.timestamp,
            wall_time=frame.wall_time,
            frame_index=frame.frame_index,
            keypoints=interp_kps,
            bbox=self._last_pose.bbox,
            detection_confidence=self._last_pose.detection_confidence,
            track_id=self._last_pose.track_id,
            frame_width=frame.width,
            frame_height=frame.height,
        )

    # ──────────────────────────────────────────
    # Debug
    # ──────────────────────────────────────────

    def draw_debug(self, bgr: np.ndarray, pose: Optional[PoseFrame]) -> np.ndarray:
        """
        Draw skeleton overlay on a BGR frame for debug visualization.
        Only used when show_debug_window=true in config.
        NEVER called in production — exists only for development.
        """
        import cv2
        from core.types import SKELETON_EDGES

        if pose is None:
            cv2.putText(bgr, "NO DETECTION", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return bgr

        # Optionally blur face region for privacy even in debug
        if self.privacy_cfg.get("blur_face_in_debug", True):
            bgr = self._blur_face_region(bgr, pose)

        # Draw edges
        for kp_a, kp_b in SKELETON_EDGES:
            pt_a = pose.get_xy(kp_a)
            pt_b = pose.get_xy(kp_b)
            if pt_a is not None and pt_b is not None:
                cv2.line(bgr,
                         (int(pt_a[0]), int(pt_a[1])),
                         (int(pt_b[0]), int(pt_b[1])),
                         (0, 220, 150), 2)

        # Draw keypoints
        for kp in pose.keypoints:
            if kp.is_valid():
                cv2.circle(bgr, (int(kp.x), int(kp.y)), 4, (0, 255, 200), -1)

        # Bounding box
        if pose.bbox and self.config["output"].get("debug_draw_bbox", True):
            x1, y1, x2, y2 = [int(v) for v in pose.bbox]
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 180, 100), 1)

        # Stats overlay
        stats_text = [
            f"FPS: {self.stats.avg_fps:.1f}",
            f"Infer: {self.stats.avg_inference_ms:.0f}ms",
            f"Visible KPs: {pose.visible_keypoint_count()}/17",
            f"Track ID: {pose.track_id}",
        ]
        for i, txt in enumerate(stats_text):
            cv2.putText(bgr, txt, (10, 30 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        return bgr

    def _blur_face_region(self, bgr: np.ndarray, pose: PoseFrame) -> np.ndarray:
        """Blur the face region. Uses nose + ear keypoints to estimate face bounds."""
        import cv2
        face_kps = [KP.NOSE, KP.LEFT_EYE, KP.RIGHT_EYE, KP.LEFT_EAR, KP.RIGHT_EAR]
        pts = [pose.get_xy(k) for k in face_kps]
        valid = [p for p in pts if p is not None]
        if len(valid) < 2:
            return bgr

        xs = [p[0] for p in valid]
        ys = [p[1] for p in valid]
        pad = 30
        x1, y1 = max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad)
        x2, y2 = min(bgr.shape[1], int(max(xs)) + pad), min(bgr.shape[0], int(max(ys)) + pad)

        if x2 > x1 and y2 > y1:
            face_roi = bgr[y1:y2, x1:x2]
            bgr[y1:y2, x1:x2] = cv2.GaussianBlur(face_roi, (51, 51), 0)

        return bgr
