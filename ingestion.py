"""
vigil/core/ingestion.py

Frame ingestion layer.
Handles camera capture from:
  - USB camera
  - CSI camera via GStreamer (Jetson)
  - RTSP IP camera
  - Test pattern (no hardware)

Produces a clean stream of BGR frames for the pose estimator.
Runs in its own thread with a queue so the pose module is never
blocked waiting on I/O.
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
    """Raw frame container. Short-lived — consumed immediately by pose module."""
    bgr: np.ndarray          # H x W x 3 uint8. Never persisted.
    timestamp: float         # time.monotonic()
    wall_time: float         # time.time()
    frame_index: int
    width: int
    height: int


class FrameIngestion:
    """
    Manages camera capture lifecycle.

    Usage:
        ingestion = FrameIngestion(config)
        ingestion.start()
        for frame in ingestion.frames():
            process(frame)
        ingestion.stop()

    Or as a context manager:
        with FrameIngestion(config) as ingestion:
            for frame in ingestion.frames():
                ...
    """

    def __init__(self, config: dict):
        self.config = config
        self.source = config["camera"]["source"]
        self.target_w = config["camera"]["width"]
        self.target_h = config["camera"]["height"]
        self.target_fps = config["camera"]["fps"]
        self.use_gstreamer = config["camera"].get("use_gstreamer", False)

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._queue: queue.Queue = queue.Queue(maxsize=4)  # small buffer — we want fresh frames
        self._running = False
        self._frame_index = 0
        self._lock = threading.Lock()

        # Stats
        self._fps_counter = 0
        self._fps_start = time.monotonic()
        self._current_fps = 0.0

    # ──────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────

    def start(self) -> "FrameIngestion":
        """Open the camera and start the capture thread."""
        self._cap = self._open_capture()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="vigil-capture")
        self._thread.start()
        logger.info(f"Ingestion started — source={self.source} resolution={self.target_w}x{self.target_h} fps={self.target_fps}")
        return self

    def stop(self):
        """Stop capture and release resources."""
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

    # ──────────────────────────────────────────
    # Frame access
    # ──────────────────────────────────────────

    def frames(self, timeout: float = 2.0) -> Generator[CapturedFrame, None, None]:
        """
        Generator that yields frames as they arrive.
        Blocks up to `timeout` seconds waiting for each frame.
        Raises StopIteration cleanly when pipeline stops.
        """
        while self._running:
            try:
                frame = self._queue.get(timeout=timeout)
                yield frame
            except queue.Empty:
                if not self._running:
                    break
                logger.warning("Frame queue empty — camera may have stalled.")

    def get_frame_nowait(self) -> Optional[CapturedFrame]:
        """Non-blocking: return latest frame or None."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def fps(self) -> float:
        return self._current_fps

    @property
    def is_running(self) -> bool:
        return self._running

    # ──────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────

    def _open_capture(self) -> cv2.VideoCapture:
        if self.source == "test":
            logger.info("Using synthetic test source (no camera hardware)")
            return _TestCapture(self.target_w, self.target_h, self.target_fps)

        if self.use_gstreamer:
            pipeline = self.config["camera"]["gstreamer_pipeline"]
            logger.info(f"Opening GStreamer pipeline: {pipeline[:80]}...")
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            logger.info(f"Opening camera source: {self.source}")
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_h)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            # Minimize internal buffer to keep frames fresh
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera source: {self.source}\n"
                f"  If using Jetson CSI camera, set use_gstreamer: true in config.\n"
                f"  If testing without hardware, set source: 'test' in config."
            )

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")
        return cap

    def _capture_loop(self):
        """Runs in background thread. Reads frames and puts them in the queue."""
        consecutive_failures = 0
        max_failures = 30

        while self._running:
            ret, bgr = self._cap.read()

            if not ret:
                consecutive_failures += 1
                logger.warning(f"Frame read failed ({consecutive_failures}/{max_failures})")
                if consecutive_failures >= max_failures:
                    logger.error("Too many consecutive frame failures. Stopping ingestion.")
                    self._running = False
                    break
                time.sleep(0.05)
                continue

            consecutive_failures = 0

            # Resize if camera returned different resolution than requested
            h, w = bgr.shape[:2]
            if w != self.target_w or h != self.target_h:
                bgr = cv2.resize(bgr, (self.target_w, self.target_h))

            now = time.monotonic()
            frame = CapturedFrame(
                bgr=bgr,
                timestamp=now,
                wall_time=time.time(),
                frame_index=self._frame_index,
                width=self.target_w,
                height=self.target_h,
            )
            self._frame_index += 1

            # Drop oldest frame if queue is full (always prefer fresh frames)
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

            self._queue.put(frame)

            # FPS tracking
            self._fps_counter += 1
            elapsed = now - self._fps_start
            if elapsed >= 2.0:
                self._current_fps = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_start = now


class _TestCapture:
    """
    Synthetic camera that generates moving skeleton test frames.
    No hardware required — useful for unit testing and CI.
    """

    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps
        self._frame_time = 1.0 / fps
        self._t = 0.0
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        time.sleep(self._frame_time)
        self._t += self._frame_time
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Draw a simple moving circle so the frame isn't completely static
        cx = int(self.width / 2 + np.sin(self._t) * 100)
        cy = int(self.height / 2 + np.cos(self._t * 0.7) * 60)
        cv2.circle(frame, (cx, cy), 30, (0, 200, 150), -1)
        cv2.putText(frame, "VIGIL TEST SOURCE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)
        return True, frame

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def set(self, *_):
        pass

    def release(self):
        self._opened = False
