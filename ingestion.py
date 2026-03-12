"""
vigil/ingestion.py

Frame ingestion — supports:
  - CSI camera via GStreamer (Jetson)
  - USB camera (index integer)
  - Video file (.mp4, .mov, .avi) — for testing
  - Synthetic test pattern ("test")
"""

import cv2
import time
import logging
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Generator
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CapturedFrame:
    bgr: np.ndarray
    timestamp: float
    wall_time: float
    frame_index: int
    width: int
    height: int


class FrameIngestion:
    """
    Manages camera/video capture lifecycle.

    Source types (set via config camera.source):
        0, 1, 2          → USB camera index
        "test"           → synthetic moving pattern
        "rtsp://..."     → IP camera
        "file.mp4"       → video file (any string ending in video extension)

    GStreamer (for Jetson CSI camera):
        Set use_gstreamer: true in config.
        The gstreamer_pipeline string is used directly.
    """

    VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.m4v', '.hevc')

    def __init__(self, config: dict):
        self.config = config
        self.source = config["camera"]["source"]
        self.target_w = config["camera"]["width"]
        self.target_h = config["camera"]["height"]
        self.target_fps = config["camera"]["fps"]
        self.use_gstreamer = config["camera"].get("use_gstreamer", False)

        self._is_video_file = self._check_is_video_file()

        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._running = False
        self._frame_index = 0

        self._fps_counter = 0
        self._fps_start = time.monotonic()
        self._current_fps = 0.0

    def _check_is_video_file(self) -> bool:
        if isinstance(self.source, str):
            src_lower = self.source.lower()
            return any(src_lower.endswith(ext) for ext in self.VIDEO_EXTENSIONS)
        return False

    def start(self) -> "FrameIngestion":
        self._cap = self._open_capture()
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop if not self._is_video_file else self._video_loop,
            daemon=True,
            name="vigil-capture"
        )
        self._thread.start()
        logger.info(f"Ingestion started | source={self.source} | {self.target_w}x{self.target_h} @ {self.target_fps}fps")
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
        logger.info("Ingestion stopped.")

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()

    def frames(self, timeout: float = 2.0) -> Generator[CapturedFrame, None, None]:
        while self._running:
            try:
                frame = self._queue.get(timeout=timeout)
                yield frame
            except queue.Empty:
                if not self._running:
                    break
                logger.warning("Frame queue empty.")

    @property
    def fps(self) -> float:
        return self._current_fps

    @property
    def is_running(self) -> bool:
        return self._running

    def _open_capture(self) -> cv2.VideoCapture:
        # Synthetic test source
        if self.source == "test":
            logger.info("Using synthetic test source")
            return _TestCapture(self.target_w, self.target_h, self.target_fps)

        # Video file
        if self._is_video_file:
            logger.info(f"Opening video file: {self.source}")
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {self.source}")
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Video: {actual_w}x{actual_h} @ {fps:.1f}fps | {total} frames ({total/fps:.1f}s)")
            return cap

        # GStreamer CSI camera
        if self.use_gstreamer:
            pipeline = self.config["camera"]["gstreamer_pipeline"]
            logger.info(f"Opening GStreamer pipeline...")
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            # USB camera
            logger.info(f"Opening USB camera: {self.source}")
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_h)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera source: {self.source}\n"
                f"  CSI camera: set use_gstreamer: true in config\n"
                f"  No hardware: set source: 'test'"
            )

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened: {actual_w}x{actual_h} @ {cap.get(cv2.CAP_PROP_FPS):.1f}fps")
        return cap

    def _capture_loop(self):
        """Live camera loop — drops stale frames, always keeps freshest."""
        failures = 0
        while self._running:
            ret, bgr = self._cap.read()
            if not ret:
                failures += 1
                if failures >= 30:
                    logger.error("Too many frame failures. Stopping.")
                    self._running = False
                    break
                time.sleep(0.05)
                continue

            failures = 0
            h, w = bgr.shape[:2]
            if w != self.target_w or h != self.target_h:
                bgr = cv2.resize(bgr, (self.target_w, self.target_h))

            now = time.monotonic()
            frame = CapturedFrame(
                bgr=bgr, timestamp=now, wall_time=time.time(),
                frame_index=self._frame_index,
                width=self.target_w, height=self.target_h,
            )
            self._frame_index += 1

            # Drop oldest if full — always keep freshest
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)
            self._update_fps(now)

    def _video_loop(self):
        """
        Video file loop — plays at real-time speed based on file FPS.
        Resizes frames to target resolution.
        Loops video if it ends (for testing).
        """
        file_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if file_fps <= 0:
            file_fps = 30.0
        frame_delay = 1.0 / file_fps

        logger.info(f"Video playback at {file_fps:.1f}fps (frame delay: {frame_delay*1000:.1f}ms)")

        while self._running:
            t_start = time.monotonic()
            ret, bgr = self._cap.read()

            if not ret:
                # End of video — loop back
                logger.info("Video ended — looping")
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Resize to target resolution
            h, w = bgr.shape[:2]
            if w != self.target_w or h != self.target_h:
                # Preserve aspect ratio with letterboxing
                bgr = self._letterbox(bgr, self.target_w, self.target_h)

            now = time.monotonic()
            frame = CapturedFrame(
                bgr=bgr, timestamp=now, wall_time=time.time(),
                frame_index=self._frame_index,
                width=self.target_w, height=self.target_h,
            )
            self._frame_index += 1
            self._queue.put(frame)
            self._update_fps(now)

            # Pace to real-time
            elapsed = time.monotonic() - t_start
            sleep = frame_delay - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _letterbox(self, bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Resize preserving aspect ratio, pad with black bars."""
        h, w = bgr.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(bgr, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_off = (target_w - new_w) // 2
        y_off = (target_h - new_h) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        return canvas

    def _update_fps(self, now: float):
        self._fps_counter += 1
        elapsed = now - self._fps_start
        if elapsed >= 2.0:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_start = now


class _TestCapture:
    """Synthetic moving pattern — no hardware needed."""
    def __init__(self, w, h, fps):
        self.width, self.height, self.fps = w, h, fps
        self._t = 0.0
        self._delay = 1.0 / fps

    def isOpened(self): return True

    def read(self):
        time.sleep(self._delay)
        self._t += self._delay
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cx = int(self.width/2 + np.sin(self._t) * 150)
        cy = int(self.height/2 + np.cos(self._t*0.7) * 80)
        cv2.circle(frame, (cx, cy), 40, (0, 200, 150), -1)
        cv2.putText(frame, "VIGIL TEST SOURCE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 1)
        return True, frame

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:  return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT: return float(self.height)
        if prop == cv2.CAP_PROP_FPS:          return float(self.fps)
        return 0.0

    def set(self, *_): pass
    def release(self): pass