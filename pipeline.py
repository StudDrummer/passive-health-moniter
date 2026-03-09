"""
vigil/pipeline.py  (flat structure — all files in same directory)

Main pipeline orchestrator.
Run with:
    python3 pipeline.py --config camera_config.yaml --debug
    python3 pipeline.py --config camera_config.yaml --source test
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
from gait import GaitModule

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

    def __init__(self, config: dict):
        self.config = config
        self._running = False

        self.ingestion = FrameIngestion(config)
        self.pose = PoseEstimator(config)

        # Signal modules
        self.gait = GaitModule()
        self._signal_modules = [self.gait]

        self._frame_count = 0
        self._pose_count = 0
        self._start_time = 0.0

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def add_module(self, module):
        self._signal_modules.append(module)
        logger.info(f"Module registered: {module.__class__.__name__}")

    def run(self):
        logger.info("=" * 60)
        logger.info("VIGIL Camera Pipeline starting")
        logger.info("=" * 60)

        self.pose.load()
        self.ingestion.start()
        self._running = True
        self._start_time = time.monotonic()

        show_debug = self.config["output"].get("show_debug_window", False)
        logger.info("Pipeline running. Press Ctrl+C to stop.")

        try:
            for raw_frame in self.ingestion.frames():
                if not self._running:
                    break

                pose_frame = self.pose.process(raw_frame)
                self._frame_count += 1

                if pose_frame:
                    self._pose_count += 1
                    self._on_pose(pose_frame)

                if show_debug:
                    display = raw_frame.bgr.copy()
                    display = self.pose.draw_debug(display, pose_frame)
                    self._draw_gait_overlay(display)
                    self._draw_pipeline_stats(display)
                    cv2.imshow("VIGIL Debug", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if self._frame_count % 150 == 0:
                    self._log_stats()

        finally:
            self._shutdown()

    def _on_pose(self, pose: PoseFrame):
        for module in self._signal_modules:
            try:
                module.update(pose)
            except Exception as e:
                logger.error(f"Module {module.__class__.__name__} error: {e}", exc_info=True)

    def _draw_gait_overlay(self, bgr):
        """Draw live gait metrics on the debug window."""
        m = self.gait.latest_metrics
        if m is None:
            cv2.putText(bgr, "GAIT: calibrating...", (10, bgr.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
            return

        lines = [
            f"Speed:   {m.speed_px_per_sec:.1f} px/s",
            f"Stride:  {m.stride_length_px:.1f} px",
            f"Cadence: {m.cadence_spm:.1f} spm",
            f"Asym:    {m.asymmetry_pct:.1f}%",
            f"Conf:    {m.confidence:.2f}",
        ]

        # Background panel
        panel_x, panel_y = 10, bgr.shape[0] - 160
        cv2.rectangle(bgr, (panel_x - 5, panel_y - 15),
                      (panel_x + 200, panel_y + len(lines) * 22 + 5),
                      (0, 0, 0), -1)

        for i, line in enumerate(lines):
            # Flag slow gait in red
            color = (0, 80, 255) if (i == 0 and m.slow_gait) else (0, 220, 140)
            cv2.putText(bgr, line, (panel_x, panel_y + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

        # Baseline status
        bl_text = "Baseline: LOCKED" if self.gait.baseline_locked else f"Baseline: learning ({self.gait.stride_count} strides)"
        cv2.putText(bgr, bl_text, (10, bgr.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 220, 140) if self.gait.baseline_locked else (0, 180, 255), 1)

    def _shutdown(self):
        logger.info("Shutting down pipeline...")
        self._running = False
        self.ingestion.stop()
        cv2.destroyAllWindows()
        self._log_stats(final=True)

        if self.gait.baseline_locked:
            logger.info(f"Final baseline: {self.gait.baseline}")

    def _handle_shutdown(self, *_):
        logger.info("Shutdown signal received.")
        self._running = False

    def _log_stats(self, final: bool = False):
        elapsed = time.monotonic() - self._start_time
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        detection_rate = self._pose_count / self._frame_count if self._frame_count > 0 else 0
        prefix = "FINAL" if final else "STATS"
        logger.info(
            f"[{prefix}] frames={self._frame_count} | fps={fps:.1f} | "
            f"detection_rate={detection_rate:.1%} | "
            f"infer={self.pose.stats.avg_inference_ms:.1f}ms | "
            f"strides={self.gait.stride_count}"
        )

    def _draw_pipeline_stats(self, bgr):
        elapsed = time.monotonic() - self._start_time
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(bgr, f"FPS: {fps:.1f} | Infer: {self.pose.stats.avg_inference_ms:.0f}ms",
                    (bgr.shape[1] - 240, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 150, 150), 1)


def main():
    parser = argparse.ArgumentParser(description="VIGIL Camera Pipeline")
    parser.add_argument("--config", default="camera_config.yaml")
    parser.add_argument("--source", help="Override camera source ('test', '0', 'rtsp://...')")
    parser.add_argument("--debug", action="store_true", help="Enable debug window")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

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