"""
vigil/core/pipeline.py

Main pipeline orchestrator.
Ties frame ingestion → pose estimation → signal modules together.
This is the entry point for the camera process.

Run with:
    python pipeline.py
    python pipeline.py --config config/camera_config.yaml
    python pipeline.py --source test    # no hardware needed
"""

import cv2
import time
import logging
import argparse
import signal
import sys
from pathlib import Path
from typing import Optional

import yaml

from ingestion import FrameIngestion
from pose_estimator import PoseEstimator
from data_types import PoseFrame

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
    handlers = [logging.StreamHandler()]

    if config["logging"].get("log_to_file"):
        log_path = Path(config["logging"]["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers,
    )


class VIGILPipeline:
    """
    Top-level pipeline. Wires all components together.

    Extend this class to add signal modules:
        def _on_pose(self, pose: PoseFrame):
            self.gait_module.update(pose)
            self.posture_module.update(pose)
            self.emergency_module.update(pose)
    """

    def __init__(self, config: dict):
        self.config = config
        self._running = False

        # Components
        self.ingestion = FrameIngestion(config)
        self.pose = PoseEstimator(config)

        # Signal modules will attach here in future phases
        self._signal_modules = []

        # Stats
        self._frame_count = 0
        self._pose_count = 0
        self._start_time = 0.0

        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def add_module(self, module):
        """Register a signal module to receive pose frames."""
        self._signal_modules.append(module)
        logger.info(f"Module registered: {module.__class__.__name__}")

    def run(self):
        """Start the pipeline. Blocks until stopped."""
        logger.info("=" * 60)
        logger.info("VIGIL Camera Pipeline starting")
        logger.info("=" * 60)

        # Load model
        self.pose.load()

        # Open camera and start capture thread
        self.ingestion.start()
        self._running = True
        self._start_time = time.monotonic()

        show_debug = self.config["output"].get("show_debug_window", False)

        logger.info("Pipeline running. Press Ctrl+C to stop.")

        try:
            for raw_frame in self.ingestion.frames():
                if not self._running:
                    break

                # Pose estimation
                pose_frame = self.pose.process(raw_frame)
                self._frame_count += 1

                if pose_frame:
                    self._pose_count += 1
                    self._on_pose(pose_frame)

                # Debug window
                if show_debug:
                    display = raw_frame.bgr.copy()
                    display = self.pose.draw_debug(display, pose_frame)
                    self._draw_pipeline_stats(display)
                    cv2.imshow("VIGIL Debug", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Log stats every 5s
                if self._frame_count % (30 * 5) == 0:
                    self._log_stats()

        finally:
            self._shutdown()

    def _on_pose(self, pose: PoseFrame):
        """
        Called for every valid pose frame.
        Dispatches to all registered signal modules.
        Phase 2 will add: gait, posture, emergency, face modules here.
        """
        for module in self._signal_modules:
            try:
                module.update(pose)
            except Exception as e:
                logger.error(f"Module {module.__class__.__name__} error: {e}", exc_info=True)

        # For now — log keypoint summary at DEBUG level
        logger.debug(
            f"Frame {pose.frame_index:06d} | "
            f"track={pose.track_id} | "
            f"visible_kps={pose.visible_keypoint_count()} | "
            f"conf={pose.detection_confidence:.2f}"
        )

    def _shutdown(self):
        logger.info("Shutting down pipeline...")
        self._running = False
        self.ingestion.stop()
        cv2.destroyAllWindows()
        self._log_stats(final=True)

    def _handle_shutdown(self, *_):
        logger.info("Shutdown signal received.")
        self._running = False

    def _log_stats(self, final: bool = False):
        elapsed = time.monotonic() - self._start_time
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        detection_rate = self._pose_count / self._frame_count if self._frame_count > 0 else 0
        prefix = "FINAL" if final else "STATS"
        logger.info(
            f"[{prefix}] "
            f"frames={self._frame_count} | "
            f"fps={fps:.1f} | "
            f"detection_rate={detection_rate:.1%} | "
            f"avg_infer={self.pose.stats.avg_inference_ms:.1f}ms | "
            f"uptime={elapsed:.0f}s"
        )

    def _draw_pipeline_stats(self, bgr):
        import numpy as np
        elapsed = time.monotonic() - self._start_time
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(bgr, f"Pipeline FPS: {fps:.1f}", (bgr.shape[1] - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)


# ──────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VIGIL Camera Pipeline")
    parser.add_argument("--config", default="config/camera_config.yaml")
    parser.add_argument("--source", help="Override camera source (e.g. 'test', '0', 'rtsp://...')")
    parser.add_argument("--debug", action="store_true", help="Enable debug window")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    # CLI overrides
    if args.source:
        config["camera"]["source"] = int(args.source) if args.source.isdigit() else args.source
    if args.debug:
        config["output"]["show_debug_window"] = True
    if args.no_gpu:
        config["pose"]["device"] = "cpu"
        config["inference"]["half_precision"] = False

    pipeline = VIGILPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()