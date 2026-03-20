import argparse
import sqlite3
import os
import sys
import time
import math
from collections import deque
from datetime import datetime
 
import cv2
import numpy as np


#config

DB_PATH = os.path.expanduser("~/passive-health-moniter/vigil.db")
MODEL_PATH = os.path.expanduser("~/passive-health-moniter/yolov8n-pose.pt")
SESSION_DURATION = 30 #seconds per snap vid
INFER_EVERY_N = 2 #run pose estimatetion every N frames
MIN_CONF = 0.5 #minimum person detection confidence
KP_CONF = 0.3 #minimum keypoint confidence
SMOOTHING_WIN = 7 #median smoothing window for metric time series


#real world calibration
#average human hip to ankle distance is about 0.9m
REAL_HIP_ANKLE_M = 0.9
#use this to convert pixel distances to real world meters

#keypoint indices - coco format - yolov8 standard

KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

#database

def get_db(): 
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def write_snapshot(metrics: dict):
    """Write one session's gait metrics to camera_metrics table."""
    conn = get_db()
    conn.execute("""
        INSERT INTO camera_metrics (
            session_date, gait_speed_px, cadence_spm, asymmetry_pct,
            stride_norm, step_regularity, body_scale_px, camera_mode, stride_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d"),
        metrics.get("walking_speed_px"),
        metrics.get("cadence_spm"),
        metrics.get("asymmetry_pct"),
        metrics.get("lateral_sway_norm"),
        metrics.get("posture_score"),
        metrics.get("body_scale_px"),
        metrics.get("camera_mode"),
        metrics.get("frame_count"),
    ))
    conn.commit()
    conn.close()
    print(f"[DB] Snapshot written for {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

#geometry help

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
 
 
def angle_from_vertical(p_bottom, p_top):
    #angle between vertical line and line from p_bottom to p_top
    dx = p_top[0] - p_bottom[0]
    dy = p_bottom[1] - p_top[1]  # Y is inverted in image coords
    angle = math.degrees(math.atan2(dx, dy))
    return angle

def median_smooth(values: list, window: int) -> float:
    """Return median of last `window` values."""
    if not values:
        return 0.0
    tail = values[-window:]
    return float(np.median(tail))

#camera mode detection - determine if camera is front facing or side
#side if person moves in x axis
#front if person moves in z axis (grow/shrink)

def detect_camera_mode(hip_positions: list) -> str:
    """Detect camera mode based on hip keypoint movement.
    hip_positions: list of (x, y) tuples
    """
    if len(hip_positions) < 10:
        return "unknown"
 
    xs = [p[0] for p in hip_positions]
    ys = [p[1] for p in hip_positions]
 
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
 
    # Also check if bounding box size changes a lot (front view)
    if x_range > y_range * 2.5:
        return "side"
    elif y_range > x_range * 1.5:
        return "front"
    else:
        return "angled"
    
#pixel to meter conversion - sketchy

def compute_px_per_meter(kps) -> float:
    """Compute pixels per meter using hip to ankle distance."""
    left_hip   = kps[KP_LEFT_HIP]
    right_hip  = kps[KP_RIGHT_HIP]
    left_ankle = kps[KP_LEFT_ANKLE]
    right_ankle = kps[KP_RIGHT_ANKLE]
 
    #use whichever side has higher confidence
    left_conf  = min(left_hip[2], left_ankle[2])
    right_conf = min(right_hip[2], right_ankle[2])
 
    if left_conf >= right_conf and left_conf > KP_CONF:
        dist_px = distance(left_hip[:2], left_ankle[:2])
    elif right_conf > KP_CONF:
        dist_px = distance(right_hip[:2], right_ankle[:2])
    else:
        return 0.0
 
    if dist_px < 10:
        return 0.0
 
    return dist_px / REAL_HIP_ANKLE_M
 


#per frame metric extraction

def extract_frame_metrics(kps, prev_hip_x: float, fps: float, px_per_m: float) -> dict:
    """ extract all metrics from single keypoints"""
    metrics = {}

    metrics["px_per_m"] = px_per_m
    metrics["body_scale"]  = px_per_m * REAL_HIP_ANKLE_M  # hip-ankle in px

    #hip centroid

    lh = kps[KP_LEFT_HIP]
    rh = kps[KP_RIGHT_HIP]
    if lh[2] > KP_CONF and rh[2] > KP_CONF:
        hip_x = (lh[0] + rh[0]) / 2
        hip_y = (lh[1] + rh[1]) / 2
        metrics["hip_x"] = hip_x
        metrics["hip_y"] = hip_y

        #walking speed - pixel displacement

        if prev_hip_x is not None and px_per_m > 0:
            dx_px = abs(hip_x - prev_hip_x)
            dx_m = dx_px / px_per_m
            speed_ms = dx_m * fps / INFER_EVERY_N
            metrics["walking_speed_ms"] = speed_ms
        else:
            metrics["hip_x"] = None
            metrics["hip_y"] = None
        
        #body lean angle

        ls = kps[KP_LEFT_SHOULDER]
        rs = kps[KP_RIGHT_SHOULDER]
        if (ls[2] > KP_CONF and rs[2] > KP_CONF and
            lh[2] > KP_CONF and rh[2] > KP_CONF):
            shoulder_mid = midpoint(ls[:2], rs[:2])
            hip_mid      = midpoint(lh[:2], rh[:2])
            lean         = angle_from_vertical(hip_mid, shoulder_mid)
            metrics["body_lean_deg"] = lean
        else:
            metrics["body_lean_deg"] = None
        

        #lateral sway
        #stored as hip_x sway is computed as std dev over session
        #stored per frame - aggregated at session end

        #step sym

        la = kps[KP_LEFT_ANKLE]
        ra = kps[KP_RIGHT_ANKLE]
        if (lh[2] > KP_CONF and rh[2] > KP_CONF and
                la[2] > KP_CONF and ra[2] > KP_CONF):
            left_stride  = distance(lh[:2], la[:2])
            right_stride = distance(rh[:2], ra[:2])
            if max(left_stride, right_stride) > 0:
                symmetry = min(left_stride, right_stride) / max(left_stride, right_stride)
                metrics["step_symmetry"] = symmetry
            else:
                metrics["step_symmetry"] = None
        else:
            metrics["step_symmetry"] = None
    
        return metrics
    
#session aggregation

def aggergate_session(frame_metrices: list, camera_mode: str) -> dict:
    """ aggregate frame level metrices into single session snapshot
    uses median for robustness against outliers."""

    def med(key):
        vals = [m[key] for m in frame_metrices if m.get(key) is not None]
        return float(np.median(vals)) if vals else None
 
    def std(key):
        vals = [m[key] for m in frame_metrices if m.get(key) is not None]
        return float(np.std(vals)) if len(vals) > 2 else None
 
    walking_speed_ms = med("walking_speed_ms")
    body_lean_deg    = med("body_lean_deg")
    step_symmetry    = med("step_symmetry")
    body_scale_px    = med("body_scale")
    px_per_m         = med("px_per_m")

    #lateral sway = std dev of hip X position - normalized to body height
    hip_xs = [m["hip_x"] for m in frame_metrices if m.get("hip_x") is not None]
    sway_px = float(np.std(hip_xs)) if len(hip_xs) > 2 else None
    sway_norm = (sway_px / body_scale_px) if (sway_px and body_scale_px) else None

    #asymmetry percentage (0 = perfect symmetry, 100 = completely asymmetric)
    asymmetry_pct = ((1 - step_symmetry) * 100) if step_symmetry is not None else None

    #Posture score (0-100, higher = better posture)
    #Penalizes forward lean and lateral sway
    posture_score = None
    if body_lean_deg is not None and sway_norm is not None:
        lean_penalty  = min(abs(body_lean_deg) * 2, 50)   # max 50 points off for lean
        sway_penalty  = min(sway_norm * 200, 50)           # max 50 points off for sway
        posture_score = max(0, 100 - lean_penalty - sway_penalty)

    snapshot = {
        "camera_mode":       camera_mode,
        "walking_speed_ms":  walking_speed_ms,
        "walking_speed_px":  (walking_speed_ms * px_per_m) if (walking_speed_ms and px_per_m) else None,
        "body_lean_deg":     body_lean_deg,
        "lateral_sway_norm": sway_norm,
        "step_symmetry":     step_symmetry,
        "asymmetry_pct":     asymmetry_pct,
        "posture_score":     posture_score,
        "body_scale_px":     body_scale_px,
        "px_per_m":          px_per_m,
        "frame_count":       len(frame_metrices),
        "cadence_spm":       None,   # populated by stride detector if available
    }
    return snapshot
 
 #debug overlay
def draw_overlay(frame, kps, metrics: dict, session_metrics: dict):
    """Draw keypoints, skeleton, and metric values on frame."""
    h, w = frame.shape[:2]
 
    # Draw keypoints
    for i, kp in enumerate(kps):
        x, y, conf = int(kp[0]), int(kp[1]), kp[2]
        if conf > KP_CONF:
            cv2.circle(frame, (x, y), 4, (0, 220, 130), -1)
 
    # Draw skeleton — key connections only
    connections = [
        (KP_LEFT_SHOULDER,  KP_RIGHT_SHOULDER),
        (KP_LEFT_SHOULDER,  KP_LEFT_HIP),
        (KP_RIGHT_SHOULDER, KP_RIGHT_HIP),
        (KP_LEFT_HIP,       KP_RIGHT_HIP),
        (KP_LEFT_HIP,       KP_LEFT_KNEE),
        (KP_RIGHT_HIP,      KP_RIGHT_KNEE),
        (KP_LEFT_KNEE,      KP_LEFT_ANKLE),
        (KP_RIGHT_KNEE,     KP_RIGHT_ANKLE),
        (KP_LEFT_SHOULDER,  KP_LEFT_ELBOW),
        (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW),
    ]
    for a, b in connections:
        if kps[a][2] > KP_CONF and kps[b][2] > KP_CONF:
            cv2.line(frame,
                     (int(kps[a][0]), int(kps[a][1])),
                     (int(kps[b][0]), int(kps[b][1])),
                     (0, 180, 100), 2)
 
    # Metric overlay — top left
    overlay_lines = [
        f"VIGIL — Gait Snapshot",
        f"Mode:     {session_metrics.get('camera_mode', '...')}",
        f"Speed:    {f\"{metrics.get('walking_speed_ms', 0):.2f} m/s\" if metrics.get('walking_speed_ms') else '—'}",
        f"Lean:     {f\"{metrics.get('body_lean_deg', 0):.1f} deg\" if metrics.get('body_lean_deg') is not None else '—'}",
        f"Symmetry: {f\"{metrics.get('step_symmetry', 0):.2f}\" if metrics.get('step_symmetry') is not None else '—'}",
        f"Frames:   {session_metrics.get('frame_count', 0)}",
    ]
    for i, line in enumerate(overlay_lines):
        y_pos = 24 + i * 22
        cv2.putText(frame, line, (12, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 130) if i == 0 else (200, 200, 200),
                    1, cv2.LINE_AA)
 
    return frame

#main pipeline --
def run(source, debug: bool = False, write_db: bool = True):
    #load YOLOv8-pose model
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip3 install ultralytics")
        sys.exit(1)
    print(f"[INFO] Loading YOLOv8-Pose from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded")

    #open video source - i used a ytube vid for testing before I got the cam --
    
    if str(source).isdigit():
        #csi/usb camera
        cap = cv2.VideoCapture(int(source))
    elif str(source).startswith("rtsp://"):
        #rtsp stream
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        #video file
        cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"[Error] Could not open video source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if total_frames > 0 else 0

    print(f"[INFO] Source opened: fps={fps:.1f}  frames={total_frames}  duration={duration_s:.1f}s")
    print(f"[INFO] Session window: {SESSION_DURATION}s  debug={debug}")

    #session state
    frame_num = 0
    session_start = time.time()
    frame_metrics = []
    hip_positions = []
    prev_hip_x = None
    session_count = 0

    while True: 
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream")
            break

        frame_num += 1
        
        #only run inference every N frames - saving compute
        if frame_num % INFER_EVERY_N != 0:
            continue

        #pose estimation
        results = model(
            frame,
            conf=MIN_CONF,
            verbose=False,
            device="cuda"
        )

        #get largest most confident person detected - will need to change once we have multi person data
        best_kps = None
        best_conf = 0  
        best_box_area = 0

        for result in results:
            if result.keypoints is None:
                continue
            boxes = result.boxes
            kps_all = result.keypoints.data.cpu().numpy()
 
            for i, kp in enumerate(kps_all):
                if boxes is not None and i < len(boxes):
                    conf = float(boxes.conf[i])
                    box  = boxes.xyxy[i].cpu().numpy()
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if conf > best_conf and area > best_box_area:
                        best_conf     = conf
                        best_kps      = kp          # shape (17, 3)
                        best_box_area = area
            
            if best_kps is None:
                continue

            #pixel to meter
            px_per_m = compute_px_per_meter(best_kps)

            #extract frame metrics
            fm = extract_frame_metrics(best_kps, prev_hip_x, fps, px_per_m)
            frame_metrics.append(fm)
            
            # Track hip positions for camera mode detection
            if fm.get("hip_x") is not None:
                hip_positions.append((fm["hip_x"], fm["hip_y"]))
                prev_hip_x = fm["hip_x"]

            #debug overlay
            if debug:
                session_so_far = {"camera_mode": detect_camera_mode(hip_positions),
                              "frame_count": len(frame_metrics)}
                frame = draw_overlay(frame, best_kps, fm, session_so_far)
    
                # Resize for display if too large
                dh, dw = frame.shape[:2]
                if dw > 1280:
                    scale = 1280 / dw
                    frame = cv2.resize(frame, (1280, int(dh * scale)))
    
                cv2.imshow("VIGIL — Gait Snapshot", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    
            # ── Session window complete ──────────────────────────
            elapsed = time.time() - session_start
            if elapsed >= SESSION_DURATION:
                camera_mode = detect_camera_mode(hip_positions)
                snapshot    = aggregate_session(frame_metrics, camera_mode)
                session_count += 1
    
                print(f"\n{'='*52}")
                print(f"  SESSION {session_count} COMPLETE — {datetime.now().strftime('%H:%M:%S')}")\
                print(f"{'='*52}")
                print(f"  Camera mode:    {snapshot['camera_mode']}")
                print(f"  Walking speed:  {f\"{snapshot['walking_speed_ms']:.3f} m/s\" if snapshot['walking_speed_ms'] else '—'}")
                print(f"  Body lean:      {f\"{snapshot['body_lean_deg']:.1f} deg\" if snapshot['body_lean_deg'] is not None else '—'}")
                print(f"  Lateral sway:   {f\"{snapshot['lateral_sway_norm']:.3f} norm\" if snapshot['lateral_sway_norm'] else '—'}")
                print(f"  Step symmetry:  {f\"{snapshot['step_symmetry']:.3f}\" if snapshot['step_symmetry'] is not None else '—'}")
                print(f"  Asymmetry:      {f\"{snapshot['asymmetry_pct']:.1f}%\" if snapshot['asymmetry_pct'] is not None else '—'}")
                print(f"  Posture score:  {f\"{snapshot['posture_score']:.1f}/100\" if snapshot['posture_score'] is not None else '—'}")
                print(f"  Body scale:     {f\"{snapshot['body_scale_px']:.1f} px\" if snapshot['body_scale_px'] else '—'}")
                print(f"  Frames used:    {snapshot['frame_count']}")
                print(f"{'='*52}\n")

                if write_db:
                write_snapshot(snapshot)
 
                # Reset for next session window
                frame_metrics = []
                hip_positions = []
                prev_hip_x    = None
                session_start = time.time()

        #end of stream
        if len(frame_metrics) >= 10:
            camera_mode = detect_camera_mode(hip_positions)
            snapshot    = aggregate_session(frame_metrics, camera_mode)
            session_count += 1
    
            print(f"\n{'='*52}")
            print(f"  FINAL SESSION {session_count} — {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*52}")
            print(f"  Camera mode:    {snapshot['camera_mode']}")
            print(f"  Walking speed:  {f\"{snapshot['walking_speed_ms']:.3f} m/s\" if snapshot['walking_speed_ms'] else '—'}")
            print(f"  Body lean:      {f\"{snapshot['body_lean_deg']:.1f} deg\" if snapshot['body_lean_deg'] is not None else '—'}")
            print(f"  Lateral sway:   {f\"{snapshot['lateral_sway_norm']:.3f} norm\" if snapshot['lateral_sway_norm'] else '—'}")
            print(f"  Step symmetry:  {f\"{snapshot['step_symmetry']:.3f}\" if snapshot['step_symmetry'] is not None else '—'}")
            print(f"  Asymmetry:      {f\"{snapshot['asymmetry_pct']:.1f}%\" if snapshot['asymmetry_pct'] is not None else '—'}")
            print(f"  Posture score:  {f\"{snapshot['posture_score']:.1f}/100\" if snapshot['posture_score'] is not None else '—'}")
            print(f"  Frames used:    {snapshot['frame_count']}")
            print(f"{'='*52}\n")

            if write_db:
            write_snapshot(snapshot)
 
        cap.release()
        if debug:
            cv2.destroyAllWindows()
    
        print(f"[INFO] Done — {session_count} sessions processed")

#entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIGIL Gait Snapshot")
    parser.add_argument(
        "--source", required=True,
        help="Video file path, RTSP URL, or camera index (0)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Show debug window with skeleton overlay"
    )
    parser.add_argument(
        "--no-db", action="store_true",
        help="Don't write results to database (dry run)"
    )
    args = parser.parse_args()
 
    run(
        source   = args.source,
        debug    = args.debug,
        write_db = not args.no_db,
    )
                