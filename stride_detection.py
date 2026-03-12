# run this directly on the jetson terminal

# python3 - << 'EOF'
# import sys, cv2
# sys.path.insert(0, '.')
# from ultralytics import YOLO

# cap = cv2.VideoCapture('stride_detection.mp4', cv2.CAP_FFMPEG)
# model = YOLO('yolov8n-pose.pt')
# frame_num = 0

# while frame_num < 150:
#     ret, frame = cap.read()
#     if not ret: break
#     frame_num += 1
#     if frame_num % 3 != 0: continue

#     frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#     frame = cv2.resize(frame, (1280, 720))

#     results = model.predict(frame, verbose=False, conf=0.3)
#     if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
#         kps = results[0].keypoints.data[0].cpu().numpy()
#         la, ra = kps[15], kps[16]
#         lh = kps[11]
#         print(f'f={frame_num:03d} LA_y={la[1]:.0f}(c={la[2]:.2f}) RA_y={ra[1]:.0f}(c={ra[2]:.2f}) hip_y={lh[1]:.0f} dy={abs(la[1]-lh[1]):.0f}')
#     else:
#         print(f'f={frame_num:03d} NO DETECTION')
# cap.release()
# EOF


# real code for stride detection
"""
stride_detection.py

Standalone gait analysis on a video file.
No camera, no config, no pipeline needed.

Usage:
    python3 stride_detection.py stride_detection.mp4
    python3 stride_detection.py stride_detection.mp4 --show
    python3 stride_detection.py stride_detection.mp4 --show --rotate 90
"""

import sys
import cv2
import time
import argparse
import logging
import numpy as np
from collections import deque
from scipy.signal import find_peaks, savgol_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Minimal self-contained gait analyzer
# (no imports from other vigil files — works standalone)
# ─────────────────────────────────────────────────────────────

class StandaloneGait:
    """
    Stripped-down version of GaitModule that works with raw keypoint arrays.
    Input: YOLOv8 keypoint numpy arrays directly.
    Output: prints stride events and summary stats.
    """

    # COCO keypoint indices
    NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16

    MIN_CONF = 0.15
    SMOOTHING_WIN = 9
    MIN_VELOCITY = 25.0        # px/sec — minimum swing velocity
    COOLDOWN = 0.35            # sec between strides (slightly longer to reduce duplicates)

    def __init__(self):
        self._left  = deque(maxlen=150)   # (time, x, y)
        self._right = deque(maxlen=150)
        self._com   = deque(maxlen=150)
        self._scale = deque(maxlen=30)    # body scale samples

        self._strides = []
        self._last_left  = None
        self._last_right = None

        # Track the latest timestamp we've already emitted per foot
        # This prevents re-detecting old peaks from the history buffer
        self._last_emitted_left  = -1.0
        self._last_emitted_right = -1.0

        self._cam_mode = "unknown"
        self._y_ranges = deque(maxlen=40)
        self._x_ranges = deque(maxlen=40)

        self._frame_count = 0

    def process_keypoints(self, kps: np.ndarray, timestamp: float):
        """
        kps: shape (17, 3) — x, y, confidence for each COCO keypoint
        timestamp: seconds (monotonic)
        """
        self._frame_count += 1

        la_x, la_y, la_c = kps[self.L_ANKLE]
        ra_x, ra_y, ra_c = kps[self.R_ANKLE]
        lh_x, lh_y, lh_c = kps[self.L_HIP]
        rh_x, rh_y, rh_c = kps[self.R_HIP]

        # Body scale
        if lh_c > 0.3 and la_c > 0.15:
            scale = float(np.sqrt((la_x - lh_x)**2 + (la_y - lh_y)**2))
            if scale > 20:
                self._scale.append(scale)

        body_scale = float(np.median(self._scale)) if self._scale else 200.0

        # CoM
        if lh_c > 0.2 and rh_c > 0.2:
            cx = (lh_x + rh_x) / 2
            cy = (lh_y + rh_y) / 2
            self._com.append((timestamp, cx, cy))

        # Ankle history
        if la_c > self.MIN_CONF:
            self._left.append((timestamp, la_x, la_y))
        if ra_c > self.MIN_CONF:
            self._right.append((timestamp, ra_x, ra_y))

        if self._frame_count < 15:
            return None

        # Camera mode
        self._update_camera_mode()

        # Detect strides
        new_strides = []
        new_strides += self._process_foot("left",  self._left,  self._last_left,  timestamp, body_scale, self._last_emitted_left)
        new_strides += self._process_foot("right", self._right, self._last_right, timestamp, body_scale, self._last_emitted_right)

        for s in new_strides:
            self._strides.append(s)
            if s["foot"] == "left":
                self._last_left = s
                self._last_emitted_left = s["time"]
            else:
                self._last_right = s
                self._last_emitted_right = s["time"]

        return new_strides if new_strides else None

    def _update_camera_mode(self):
        if len(self._left) < 10:
            return
        recent = list(self._left)[-20:]
        y_range = float(np.ptp([r[2] for r in recent]))
        x_range = float(np.ptp([r[1] for r in recent]))
        self._y_ranges.append(y_range)
        self._x_ranges.append(x_range)
        avg_y = float(np.mean(self._y_ranges))
        avg_x = float(np.mean(self._x_ranges))
        if avg_y > avg_x * 1.5:
            self._cam_mode = "side"
        elif avg_x > avg_y * 1.5:
            self._cam_mode = "front"
        else:
            self._cam_mode = "angled"

    def _process_foot(self, foot, history, last, now, body_scale, last_emitted_time):
        if len(history) < self.SMOOTHING_WIN + 4:
            return []

        data  = list(history)
        times = np.array([d[0] for d in data])
        xs    = np.array([d[1] for d in data])
        ys    = np.array([d[2] for d in data])

        win = min(self.SMOOTHING_WIN, len(data) - 1)
        if win % 2 == 0:
            win -= 1
        if win < 3:
            return []

        try:
            xs_s = savgol_filter(xs, win, 2)
            ys_s = savgol_filter(ys, win, 2)
        except Exception:
            xs_s, ys_s = xs, ys

        dt = np.diff(times)
        dt = np.where(dt < 1e-6, 1e-6, dt)
        vx = np.diff(xs_s) / dt
        vy = np.diff(ys_s) / dt
        speed = np.sqrt(vx**2 + vy**2)

        # Pick signal axis based on camera
        if self._cam_mode == "side":
            signal = np.abs(vy)
        elif self._cam_mode == "front":
            signal = np.abs(vx)
        else:
            signal = speed

        if len(signal) < 5:
            return []

        avg_dt = float(np.mean(dt))
        min_dist = max(3, int(self.COOLDOWN / avg_dt))

        peaks, _ = find_peaks(
            signal,
            height=self.MIN_VELOCITY,
            distance=min_dist,
            prominence=12.0,
        )

        results = []
        emitted_this_call = last_emitted_time  # track within this call too

        for peak_idx in peaks:
            post = signal[peak_idx:]
            troughs = np.where(post < self.MIN_VELOCITY * 0.3)[0]
            strike_idx = peak_idx + (troughs[0] if len(troughs) > 0 else 0)
            strike_idx = min(strike_idx, len(times) - 1)

            strike_time = times[strike_idx]
            strike_x = xs_s[min(strike_idx, len(xs_s)-1)]
            strike_y = ys_s[min(strike_idx, len(ys_s)-1)]

            # Must be strictly newer than last emission + cooldown
            if strike_time <= emitted_this_call + self.COOLDOWN:
                continue
            # Must be recent — don't process old history
            if (now - strike_time) > 0.8:
                continue
            # Must be close to the current frame (within 2 * cooldown window)
            if strike_time < now - self.COOLDOWN * 2:
                continue

            stride_px = stride_norm = stride_dur = 0.0
            if last:
                dx = strike_x - last["x"]
                dy = strike_y - last["y"]
                stride_px   = float(np.sqrt(dx**2 + dy**2))
                stride_norm = stride_px / body_scale if body_scale > 0 else 0
                stride_dur  = strike_time - last["time"]
                if stride_dur < 0.25 or stride_dur > 3.0:
                    continue
                if stride_px < body_scale * 0.15:
                    continue
            else:
                pass

            event = {
                "foot":        foot,
                "time":        strike_time,
                "x":           float(strike_x),
                "y":           float(strike_y),
                "stride_px":   stride_px,
                "stride_norm": stride_norm,
                "stride_dur":  stride_dur,
                "peak_vel":    float(signal[peak_idx]),
                "body_scale":  body_scale,
                "cam_mode":    self._cam_mode,
            }
            results.append(event)
            emitted_this_call = strike_time  # block any further peaks in this call

            event = {
                "foot":       foot,
                "time":       strike_time,
                "x":          float(strike_x),
                "y":          float(strike_y),
                "stride_px":  stride_px,
                "stride_norm": stride_norm,
                "stride_dur": stride_dur,
                "peak_vel":   float(signal[peak_idx]),
                "body_scale": body_scale,
                "cam_mode":   self._cam_mode,
            }
            results.append(event)

        return results

    def summary(self):
        if not self._strides:
            return None

        valid = [s for s in self._strides if s["stride_dur"] > 0]
        if not valid:
            return None

        durs   = [s["stride_dur"]  for s in valid]
        norms  = [s["stride_norm"] for s in valid]
        vels   = [s["peak_vel"]    for s in valid]
        left   = [s for s in valid if s["foot"] == "left"]
        right  = [s for s in valid if s["foot"] == "right"]

        cadence = 60.0 / float(np.mean(durs)) if durs else 0

        asym = 0.0
        if left and right:
            l = float(np.mean([s["stride_dur"] for s in left]))
            r = float(np.mean([s["stride_dur"] for s in right]))
            denom = 0.5 * (l + r)
            asym = abs(l - r) / denom * 100.0 if denom > 0 else 0

        return {
            "total_strides":       len(self._strides),
            "valid_strides":       len(valid),
            "left_strides":        len(left),
            "right_strides":       len(right),
            "cadence_spm":         round(cadence, 1),
            "avg_stride_norm":     round(float(np.mean(norms)), 3),
            "std_stride_norm":     round(float(np.std(norms)), 3),
            "avg_stride_dur_sec":  round(float(np.mean(durs)), 3),
            "asymmetry_pct":       round(asym, 1),
            "avg_peak_velocity":   round(float(np.mean(vels)), 1),
            "camera_mode":         self._cam_mode,
            "body_scale_px":       round(float(np.median(self._scale)) if self._scale else 0, 1),
        }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vigil stride detection on video file")
    parser.add_argument("video", help="Path to video file (mp4, mov, avi)")
    parser.add_argument("--show",   action="store_true", help="Show annotated video window")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                        help="Rotate frames before processing (degrees clockwise)")
    parser.add_argument("--conf",   type=float, default=0.3, help="YOLO confidence threshold (default 0.3)")
    parser.add_argument("--slow",   action="store_true", help="Process at 1/2 speed (easier to watch)")
    args = parser.parse_args()

    # ── Load model ──
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip3 install ultralytics")
        sys.exit(1)

    logger.info("Loading YOLOv8-Pose model...")
    model = YOLO("yolov8n-pose.pt")
    logger.info("Model ready.")

    # ── Open video ──
    cap = cv2.VideoCapture(args.video, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error(f"Could not open video: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    file_fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / file_fps

    logger.info(f"Video: {args.video}")
    logger.info(f"  Resolution: {w}x{h} | FPS: {file_fps:.1f} | Frames: {total_frames} | Duration: {duration:.1f}s")
    if args.rotate:
        logger.info(f"  Rotating: {args.rotate}° clockwise")

    gait = StandaloneGait()
    frame_num = 0
    stride_count = 0
    t0 = time.monotonic()

    ROTATE_MAP = {
        90:  cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    logger.info("Processing... (press Q in window to quit)")
    logger.info("-" * 70)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # Rotate if needed
        if args.rotate and args.rotate in ROTATE_MAP:
            frame = cv2.rotate(frame, ROTATE_MAP[args.rotate])

        # Resize to 1280x720 for consistent processing
        fh, fw = frame.shape[:2]
        if fw != 1280 or fh != 720:
            # Letterbox to preserve aspect ratio
            scale = min(1280 / fw, 720 / fh)
            new_w, new_h = int(fw * scale), int(fh * scale)
            resized = cv2.resize(frame, (new_w, new_h))
            canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
            x_off = (1280 - new_w) // 2
            y_off = (720  - new_h) // 2
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            frame = canvas

        timestamp = frame_num / file_fps  # use video time not wall time

        # ── Pose inference ──
        results = model.predict(frame, verbose=False, conf=args.conf, device="cuda")

        kps_raw = None
        if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
            kps_raw = results[0].keypoints.data[0].cpu().numpy()  # (17, 3)

        # ── Gait processing ──
        new_strides = None
        if kps_raw is not None and kps_raw.shape[0] == 17:
            new_strides = gait.process_keypoints(kps_raw, timestamp)

        # ── Log new strides ──
        if new_strides:
            for s in new_strides:
                stride_count += 1
                logger.info(
                    f"  STRIDE #{stride_count:03d} | {s['foot']:5s} | "
                    f"t={s['time']:.2f}s | "
                    f"len={s['stride_norm']:.2f}x body ({s['stride_px']:.0f}px) | "
                    f"dur={s['stride_dur']:.2f}s | "
                    f"vel={s['peak_vel']:.0f}px/s | "
                    f"cam={s['cam_mode']}"
                )

        # ── Progress every 5 seconds of video ──
        if frame_num % int(file_fps * 5) == 0:
            pct = frame_num / total_frames * 100
            elapsed = time.monotonic() - t0
            logger.info(f"  Progress: {pct:.0f}% | frame={frame_num}/{total_frames} | strides_so_far={stride_count} | cam={gait._cam_mode}")

        # ── Debug window ──
        if args.show:
            display = frame.copy()

            # Draw skeleton if keypoints available
            if kps_raw is not None:
                EDGES = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),
                         (11,12),(11,13),(13,15),(12,14),(14,16)]
                for a, b in EDGES:
                    xa, ya, ca = kps_raw[a]
                    xb, yb, cb = kps_raw[b]
                    if ca > 0.2 and cb > 0.2:
                        cv2.line(display, (int(xa), int(ya)), (int(xb), int(yb)), (0, 200, 100), 2)
                for i in range(17):
                    x, y, c = kps_raw[i]
                    if c > 0.2:
                        color = (0, 80, 255) if i in [15, 16] else (0, 220, 140)
                        cv2.circle(display, (int(x), int(y)), 4, color, -1)

            # HUD
            hud = [
                f"Frame: {frame_num}/{total_frames}  t={timestamp:.1f}s",
                f"Camera: {gait._cam_mode}",
                f"Strides: {stride_count}",
            ]
            if gait._strides:
                last = gait._strides[-1]
                hud.append(f"Last stride: {last['foot']} | {last['stride_norm']:.2f}x | {last['stride_dur']:.2f}s")

            for i, line in enumerate(hud):
                pos = (10, 25 + i * 22)
                #outline
                cv2.putText(display, line, pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
                #main text
                cv2.putText(display, line, pos,
                          cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

            # Flash green on new stride
            if new_strides:
                cv2.rectangle(display, (0, 0), (display.shape[1], display.shape[0]), (0, 255, 0), 6)

            cv2.imshow("Vigil Stride Detection", display)
            delay = int(1000 / file_fps * (2 if args.slow else 1))
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    # ── Final summary ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)

    summary = gait.summary()
    if summary:
        logger.info(f"  Total strides detected:  {summary['total_strides']}")
        logger.info(f"  Left / Right:            {summary['left_strides']} / {summary['right_strides']}")
        logger.info(f"  Cadence:                 {summary['cadence_spm']} steps/min")
        logger.info(f"  Avg stride (normalized): {summary['avg_stride_norm']}x body scale")
        logger.info(f"  Stride variability:      {summary['std_stride_norm']}x (lower = more consistent)")
        logger.info(f"  Avg stride duration:     {summary['avg_stride_dur_sec']}s")
        logger.info(f"  Asymmetry:               {summary['asymmetry_pct']}%")
        logger.info(f"  Avg peak ankle velocity: {summary['avg_peak_velocity']} px/s")
        logger.info(f"  Body scale:              {summary['body_scale_px']}px (hip-to-ankle)")
        logger.info(f"  Camera mode detected:    {summary['camera_mode']}")
        logger.info("")

        # Clinical interpretation
        logger.info("INTERPRETATION:")
        if summary['cadence_spm'] > 0:
            if summary['cadence_spm'] < 80:
                logger.info("  ⚠ Cadence LOW (<80 spm) — slow walking speed")
            elif summary['cadence_spm'] > 130:
                logger.info("  ⚠ Cadence HIGH (>130 spm) — fast or shuffling gait")
            else:
                logger.info(f"  ✓ Cadence normal ({summary['cadence_spm']} spm)")

        if summary['asymmetry_pct'] > 25:
            logger.info(f"  ⚠ High asymmetry ({summary['asymmetry_pct']}%) — possible compensation pattern")
        elif summary['asymmetry_pct'] > 0:
            logger.info(f"  ✓ Asymmetry normal ({summary['asymmetry_pct']}%)")

        if summary['std_stride_norm'] > 0.15:
            logger.info(f"  ⚠ High stride variability — inconsistent gait pattern")
        elif summary['std_stride_norm'] > 0:
            logger.info(f"  ✓ Stride variability normal")

        if summary['total_strides'] < 5:
            logger.info("")
            logger.info("  NOTE: Low stride count — walk longer for reliable metrics.")
            logger.info("  Aim for 20+ strides (about 30 seconds of walking).")
    else:
        logger.info("  No strides detected.")
        logger.info("")
        logger.info("  Troubleshooting:")
        logger.info("  1. Make sure full body (head to ankles) is visible")
        logger.info("  2. Walk across the frame (not toward camera)")
        logger.info("  3. Try --rotate 90 if video is portrait orientation")
        logger.info("  4. Try --conf 0.2 to lower detection threshold")

    wall_time = time.monotonic() - t0
    logger.info(f"\nProcessed {frame_num} frames in {wall_time:.1f}s ({frame_num/wall_time:.1f}fps)")


if __name__ == "__main__":
    main()









