"""
vigil/pipeline.py

Main pipeline. Supports live camera and video file input.

Run:
    python3 pipeline.py --config camera_config.yaml --debug
    python3 pipeline.py --config camera_config.yaml --source stride_detection.mp4 --debug
    python3 pipeline.py --config camera_config.yaml --source test --debug
"""

import cv2
import time
import logging
import argparse
import signal
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
        self.gait = GaitModule()
        self._modules = [self.gait]

        self._frame_count = 0
        self._pose_count = 0
        self._start_time = 0.0

        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def run(self):
        logger.info("=" * 60)
        logger.info("VIGIL Pipeline starting")
        logger.info("=" * 60)

        self.pose.load()
        self.ingestion.start()
        self._running = True
        self._start_time = time.monotonic()

        show_debug = self.config["output"].get("show_debug_window", False)
        logger.info("Running. Ctrl+C to stop.")

        try:
            for raw_frame in self.ingestion.frames():
                if not self._running:
                    break

                pose_frame = self.pose.process(raw_frame)
                self._frame_count += 1

                if pose_frame:
                    self._pose_count += 1
                    for module in self._modules:
                        try:
                            module.update(pose_frame)
                        except Exception as e:
                            logger.error(f"{module.__class__.__name__} error: {e}", exc_info=True)

                if show_debug:
                    display = raw_frame.bgr.copy()
                    display = self.pose.draw_debug(display, pose_frame)
                    self._draw_gait_overlay(display)
                    self._draw_stats(display)
                    cv2.imshow("VIGIL", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if self._frame_count % 150 == 0:
                    self._log_stats()

        finally:
            self._shutdown()

    def _draw_gait_overlay(self, bgr):
        m = self.gait.latest_metrics
        h = bgr.shape[0]

        if m is None:
            cv2.putText(bgr, f"GAIT: calibrating... ({self.gait.stride_count} strides)",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            return

        lines = [
            f"Speed:    {m.speed_px_per_sec:.0f}px/s  ({m.speed_normalized:.2f}x body)",
            f"Stride:   {m.stride_length_normalized:.2f}x body  ({m.stride_length_px:.0f}px)",
            f"Cadence:  {m.cadence_spm:.0f} spm",
            f"Asym:     {m.asymmetry_pct:.0f}%",
            f"Regularity: {m.step_regularity:.2f}",
            f"Camera:   {m.camera_mode}",
            f"Strides:  {self.gait.stride_count}",
        ]

        panel_y = h - (len(lines) * 22 + 30)
        cv2.rectangle(bgr, (5, panel_y - 5), (260, h - 5), (0, 0, 0), -1)

        for i, line in enumerate(lines):
            color = (0, 80, 255) if (i == 0 and m.slow_gait) else (0, 220, 140)
            cv2.putText(bgr, line, (10, panel_y + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

        bl = "Baseline: LOCKED" if self.gait.baseline_locked else f"Baseline: learning ({self.gait.stride_count}/{self.gait.BASELINE_STRIDES})"
        cv2.putText(bgr, bl, (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 220, 140) if self.gait.baseline_locked else (0, 180, 255), 1)

    def _draw_stats(self, bgr):
        elapsed = time.monotonic() - self._start_time
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(bgr, f"FPS:{fps:.0f} Infer:{self.pose.stats.avg_inference_ms:.0f}ms",
                    (bgr.shape[1] - 200, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1)

    def _log_stats(self):
        elapsed = time.monotonic() - self._start_time
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        det = self._pose_count / self._frame_count if self._frame_count > 0 else 0
        logger.info(
            f"[STATS] frames={self._frame_count} | fps={fps:.1f} | "
            f"detection={det:.1%} | infer={self.pose.stats.avg_inference_ms:.1f}ms | "
            f"strides={self.gait.stride_count} | cam={self.gait.camera_mode}"
        )

    def _shutdown(self):
        self._running = False
        self.ingestion.stop()
        cv2.destroyAllWindows()
        self._log_stats()
        if self.gait.baseline_locked:
            logger.info(f"Final baseline: {self.gait.baseline}")

    def _handle_shutdown(self, *_):
        self._running = False


def main():
    parser = argparse.ArgumentParser(description="VIGIL Pipeline")
    parser.add_argument("--config", default="camera_config.yaml")
    parser.add_argument("--source", help="Override source: 'test', '0', 'file.mp4', 'rtsp://...'")
    parser.add_argument("--debug",  action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    if args.source:
        # Detect if it's a camera index or string source
        if args.source.isdigit():
            config["camera"]["source"] = int(args.source)
            config["camera"]["use_gstreamer"] = False
        else:
            config["camera"]["source"] = args.source
            # Video files don't use GStreamer
            VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv', '.m4v')
            if any(args.source.lower().endswith(e) for e in VIDEO_EXTS):
                config["camera"]["use_gstreamer"] = False

    if args.debug:
        config["output"]["show_debug_window"] = True
    if args.no_gpu:
        config["pose"]["device"] = "cpu"
        config["inference"]["half_precision"] = False

    VIGILPipeline(config).run()


if __name__ == "__main__":
    main()