"""
VIGIL — Camera Calibration Tool
calibrate.py

Run once after mounting the camera. Person walks a known distance while
the system records. Computes pixel-to-meter conversion factor and saves
it to calibration.json. activity_monitor.py reads this file on startup.

Usage:
    python3 calibrate.py --source 0                      # USB camera
    python3 calibrate.py --source rtsp://192.168.1.42:554/h264Preview_01_main
    python3 calibrate.py --source test_walk.mp4

Protocol:
    1. Mark start and end points on floor exactly WALK_DISTANCE_M apart
    2. Person stands at start, press SPACE to begin recording
    3. Person walks normally to the end mark
    4. Press SPACE to stop — script computes and saves calibration
"""

import argparse
import json
import os
import sys
import time
import cv2
import numpy as np

CALIB_PATH      = os.path.expanduser("~/passive-health-moniter/calibration.json")
MODEL_PATH      = os.path.expanduser("~/passive-health-moniter/yolov8n-pose.pt")
WALK_DISTANCE_M = 4.0     # person walks exactly 4 meters for calibration
MIN_CONF        = 0.5
KP_CONF         = 0.35
KP_LEFT_HIP     = 11
KP_RIGHT_HIP    = 12
KP_LEFT_ANKLE   = 15
KP_RIGHT_ANKLE  = 16


def get_hip_x(kps):
    """Return midpoint x of hips if both visible, else None."""
    lh = kps[KP_LEFT_HIP];  rh = kps[KP_RIGHT_HIP]
    if lh[2] > KP_CONF and rh[2] > KP_CONF:
        return (lh[0] + rh[0]) / 2.0
    if lh[2] > KP_CONF: return lh[0]
    if rh[2] > KP_CONF: return rh[0]
    return None


def get_body_scale_px(kps):
    """
    Approximate body height in pixels using shoulder-to-ankle distance.
    Used to normalise measurements independent of camera distance.
    """
    from activity_monitor import KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER
    ls = kps[KP_LEFT_SHOULDER]; rs = kps[KP_RIGHT_SHOULDER]
    la = kps[KP_LEFT_ANKLE];   ra = kps[KP_RIGHT_ANKLE]

    shoulder_y = None
    if ls[2] > KP_CONF and rs[2] > KP_CONF:
        shoulder_y = (ls[1] + rs[1]) / 2.0
    elif ls[2] > KP_CONF:
        shoulder_y = ls[1]
    elif rs[2] > KP_CONF:
        shoulder_y = rs[1]

    ankle_y = None
    if la[2] > KP_CONF and ra[2] > KP_CONF:
        ankle_y = (la[1] + ra[1]) / 2.0
    elif la[2] > KP_CONF:
        ankle_y = la[1]
    elif ra[2] > KP_CONF:
        ankle_y = ra[1]

    if shoulder_y is not None and ankle_y is not None:
        return abs(ankle_y - shoulder_y)
    return None


def calibrate(source, walk_distance_m=WALK_DISTANCE_M):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.")
        sys.exit(1)

    print(f"\n{'='*55}")
    print("  VIGIL Camera Calibration")
    print(f"{'='*55}")
    print(f"  Walk distance: {walk_distance_m:.1f} m")
    print(f"  Calibration will be saved to: {CALIB_PATH}")
    print(f"{'='*55}\n")

    model = YOLO(MODEL_PATH)
    print("[OK] YOLOv8 pose model loaded")

    if str(source).isdigit():
        cap = cv2.VideoCapture(int(source))
    elif str(source).startswith("rtsp://"):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Camera opened — {fps:.0f} fps")
    print("\nInstructions:")
    print("  1. Stand at the START mark (e.g. a piece of tape on the floor)")
    print("  2. Press SPACE to begin recording your walk")
    print("  3. Walk normally to the END mark ({:.1f}m away)".format(walk_distance_m))
    print("  4. Press SPACE to stop\n")

    recording   = False
    hip_x_start = None
    hip_x_end   = None
    body_scales = []
    hip_xs      = []
    frame_count = 0

    cv2.namedWindow("VIGIL Calibration", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] End of stream")
            break

        frame_count += 1

        # Run inference every 3 frames (10fps at 30fps camera)
        if frame_count % 3 == 0:
            results = model(frame, conf=MIN_CONF, verbose=False)

            best_kps  = None
            best_area = 0

            for result in results:
                if result.keypoints is None:
                    continue
                boxes   = result.boxes
                kps_all = result.keypoints.data.cpu().numpy()
                for i, kp in enumerate(kps_all):
                    if boxes is not None and i < len(boxes):
                        box  = boxes.xyxy[i].cpu().numpy()
                        area = (box[2]-box[0]) * (box[3]-box[1])
                        if area > best_area:
                            best_area = area
                            best_kps  = kp

            if best_kps is not None:
                hx = get_hip_x(best_kps)
                bs = get_body_scale_px(best_kps)

                if hx is not None:
                    # Draw hip midpoint
                    cv2.circle(frame, (int(hx), int(frame.shape[0]*0.5)), 12, (0, 220, 130), -1)
                    cv2.putText(frame, f"hip_x = {hx:.0f}px", (int(hx)+15, int(frame.shape[0]*0.5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,130), 1)

                    if recording:
                        hip_xs.append(hx)
                        if hip_x_start is None:
                            hip_x_start = hx
                            print(f"[CAL] Recording started — hip_x = {hx:.1f}px")

                if bs is not None and recording:
                    body_scales.append(bs)

        # UI overlay
        status  = "RECORDING" if recording else "READY — Press SPACE to start"
        color   = (0, 0, 220) if recording else (0, 220, 130)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (10,10,10), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "VIGIL Calibration", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,130), 1)
        cv2.putText(frame, status, (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if recording and hip_xs:
            span_px = abs(hip_xs[-1] - hip_x_start) if hip_x_start else 0
            cv2.putText(frame, f"Span so far: {span_px:.0f}px", (12, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

        cv2.imshow("VIGIL Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if not recording:
                recording = True
                hip_xs    = []
                body_scales = []
                hip_x_start = None
                print("[CAL] Recording STARTED")
            else:
                # Stop recording
                if len(hip_xs) < 5:
                    print("[WARN] Too few frames captured — try again")
                    recording = False
                    continue

                hip_x_end = hip_xs[-1]
                total_px  = abs(hip_x_end - hip_x_start)
                px_per_m  = total_px / walk_distance_m

                # Body scale: use median for robustness
                median_body_scale = float(np.median(body_scales)) if body_scales else None

                print(f"\n[CAL] Walk complete!")
                print(f"  Start hip_x:    {hip_x_start:.1f} px")
                print(f"  End hip_x:      {hip_x_end:.1f} px")
                print(f"  Pixel span:     {total_px:.1f} px")
                print(f"  Real distance:  {walk_distance_m:.2f} m")
                print(f"  px_per_m:       {px_per_m:.2f}")
                if median_body_scale:
                    print(f"  Body scale:     {median_body_scale:.1f} px")

                if px_per_m < 10:
                    print("[WARN] px_per_m is very low — did the person walk far enough?")
                elif px_per_m > 2000:
                    print("[WARN] px_per_m is very high — camera may be too close")
                else:
                    # Save calibration
                    calib = {
                        "px_per_m":            round(px_per_m, 2),
                        "m_per_px":            round(walk_distance_m / total_px, 5),
                        "walk_distance_m":     walk_distance_m,
                        "pixel_span":          round(total_px, 1),
                        "body_scale_px":       round(median_body_scale, 1) if median_body_scale else None,
                        "calibrated_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "source":              str(source),
                        "frames_captured":     len(hip_xs),
                    }
                    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
                    with open(CALIB_PATH, "w") as f:
                        json.dump(calib, f, indent=2)
                    print(f"\n[OK] Calibration saved to {CALIB_PATH}")
                    print("     Run activity_monitor.py — it will load this automatically.\n")
                    break

                recording = False

        elif key == ord('q'):
            print("[INFO] Cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()



# CALIBRATION LOADER (imported by activity_monitor.py)


def load_calibration():
    """
    Load saved calibration. Returns dict with px_per_m and m_per_px.
    Returns None if not calibrated yet.
    """
    if not os.path.exists(CALIB_PATH):
        return None
    try:
        with open(CALIB_PATH) as f:
            c = json.load(f)
        print(f"[CAL] Loaded calibration: {c['px_per_m']:.1f} px/m "
              f"(calibrated {c.get('calibrated_at','?')})")
        return c
    except Exception as e:
        print(f"[WARN] Could not load calibration: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIGIL Camera Calibration")
    parser.add_argument("--source",   required=True, help="Camera source (0, rtsp://, or video file)")
    parser.add_argument("--distance", type=float, default=4.0,
                        help="Known walk distance in meters (default: 4.0)")
    args = parser.parse_args()
    calibrate(source=args.source, walk_distance_m=args.distance)