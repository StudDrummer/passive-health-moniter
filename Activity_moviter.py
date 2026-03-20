import argparse
import sqlite3
import os
import sys
import time
import math
from datetime import datetime
 
import cv2
import numpy as np

#config

DB_PATH = os.path.expanduser("~/passive-health-moniter/vigil.db")
MODEL_PATH = os.path.expanduser("~/passive-health-moniter/yolov8n-pose.pt")
INFER_EVERY_N  = 30    # run inference every 30 frames (~1/sec at 30fps)
MIN_CONF = 0.5
KP_CONF = 0.3
 
# State must persist for this many seconds before being logged as a transition
TRANSITION_CONFIRM_S = 3.0
 
# Walking detection — hip must move this many px/sec to count as walking
WALKING_SPEED_THRESH_PX = 15.0
 
# Sitting detection
# When sitting, knees are roughly at hip height or above.
# Ratio of (hip_y - knee_y) / body_scale — negative or near zero = sitting.
SITTING_HIP_KNEE_RATIO = 0.05
 
# Lying detection — torso angle from vertical > this = lying down
LYING_TORSO_ANGLE_DEG = 45.0
 
# Stillness / freeze detection
# If no significant movement for this many seconds during waking hours → flag
STILLNESS_THRESHOLD_S  = 60.0
STILLNESS_MOVEMENT_PX  = 8.0    # min hip movement px to reset stillness timer
WAKING_HOURS_START     = 7      # 7am
WAKING_HOURS_END       = 22     # 10pm
 
# Keypoint indices (COCO)
KP_NOSE=0; KP_LEFT_EYE=1; KP_RIGHT_EYE=2
KP_LEFT_EAR=3; KP_RIGHT_EAR=4
KP_LEFT_SHOULDER=5; KP_RIGHT_SHOULDER=6
KP_LEFT_ELBOW=7; KP_RIGHT_ELBOW=8
KP_LEFT_WRIST=9; KP_RIGHT_WRIST=10
KP_LEFT_HIP=11; KP_RIGHT_HIP=12
KP_LEFT_KNEE=13; KP_RIGHT_KNEE=14
KP_LEFT_ANKLE=15; KP_RIGHT_ANKLE=16
 
ACTIVITY_COLORS = {
    "walking":    (0, 220, 130),
    "standing":   (200, 200, 200),
    "sitting":    (0, 180, 255),
    "lying_down": (0, 100, 255),
    "unknown":    (80, 80, 80),
}

#database

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
 
 
def init_tables():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS activity_events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            activity     TEXT NOT NULL,
            started_at   TEXT NOT NULL,
            ended_at     TEXT,
            duration_s   REAL,
            notes        TEXT
        );
 
        CREATE TABLE IF NOT EXISTS stillness_events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at  TEXT NOT NULL,
            duration_s   REAL,
            hour_of_day  INTEGER,
            alerted      INTEGER DEFAULT 0
        );
 
        CREATE INDEX IF NOT EXISTS idx_activity_start
            ON activity_events(started_at);
    """)
    conn.commit()
    conn.close()
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
 
 
def init_tables():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS activity_events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            activity     TEXT NOT NULL,
            started_at   TEXT NOT NULL,
            ended_at     TEXT,
            duration_s   REAL,
            notes        TEXT
        );
 
        CREATE TABLE IF NOT EXISTS stillness_events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at  TEXT NOT NULL,
            duration_s   REAL,
            hour_of_day  INTEGER,
            alerted      INTEGER DEFAULT 0
        );
 
        CREATE INDEX IF NOT EXISTS idx_activity_start
            ON activity_events(started_at);
    """)
    conn.commit()
    conn.close()

def log_activity_start(activity):
    conn = get_db()
    conn.execute("""
        INSERT INTO activity_events (activity, started_at)
        VALUES (?, ?)
    """, (activity, datetime.now().isoformat()))
    row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()
    return row_id
 
 
def log_activity_end(row_id, started_at_iso):
    now      = datetime.now()
    started  = datetime.fromisoformat(started_at_iso)
    duration = (now - started).total_seconds()
    conn     = get_db()
    conn.execute("""
        UPDATE activity_events
        SET ended_at = ?, duration_s = ?
        WHERE id = ?
    """, (now.isoformat(), round(duration, 1), row_id))
    conn.commit()
    conn.close()
    return duration
 
 
def log_stillness(duration_s):
    conn = get_db()
    conn.execute("""
        INSERT INTO stillness_events (detected_at, duration_s, hour_of_day)
        VALUES (?, ?, ?)
    """, (datetime.now().isoformat(),
          round(duration_s, 1),
          datetime.now().hour))
    conn.commit()
    conn.close()
    print("[STILLNESS] Logged: " + str(round(duration_s, 0)) + "s")

#geometry for the keypoints

def midpoint(p1, p2):
    return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
 
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
 
def angle_from_vertical(p_bottom, p_top):
    dx = p_top[0]  - p_bottom[0]
    dy = p_bottom[1] - p_top[1]
    return math.degrees(math.atan2(dx, dy))
 
def compute_body_scale(kps):
    lh = kps[KP_LEFT_HIP];  la = kps[KP_LEFT_ANKLE]
    rh = kps[KP_RIGHT_HIP]; ra = kps[KP_RIGHT_ANKLE]
    left_conf  = min(lh[2], la[2])
    right_conf = min(rh[2], ra[2])
    if left_conf >= right_conf and left_conf > KP_CONF:
        return distance(lh[:2], la[:2])
    elif right_conf > KP_CONF:
        return distance(rh[:2], ra[:2])
    return 0.0
 
 #the actuall activity classifier

 #lying down - torso horizontal
 #sitting - knees near hip height
 #walking - hip moving horizontly
 #standing - default upright position

def classify_activity(kps, prev_hip_x, body_scale, fps):
    """
    Returns activity string: walking | sitting | standing | lying_down | unknown
    """
    lh = kps[KP_LEFT_HIP];  rh = kps[KP_RIGHT_HIP]
    ls = kps[KP_LEFT_SHOULDER]; rs = kps[KP_RIGHT_SHOULDER]
    lk = kps[KP_LEFT_KNEE]; rk = kps[KP_RIGHT_KNEE]
 
    # Need at least hips to classify
    if lh[2] < KP_CONF and rh[2] < KP_CONF:
        return "unknown", None
 
    hip_x = (lh[0]+rh[0]) / 2
    hip_y = (lh[1]+rh[1]) / 2


    #lying down
    if ls[2] > KP_CONF and rs[2] > KP_CONF and lh[2] > KP_CONF and rh[2] > KP_CONF:
        shoulder_mid = midpoint(ls[:2], rs[:2])
        hip_mid      = midpoint(lh[:2], rh[:2])
        torso_angle  = abs(angle_from_vertical(hip_mid, shoulder_mid))
        if torso_angle > LYING_TORSO_ANGLE_DEG:
            return "lying_down", hip_x
    #sitting
    # (higher Y value = lower in image, so sitting: knee_y <= hip_y + threshold)
    knee_confs = []
    knee_y_vals = []
    if lk[2] > KP_CONF:
        knee_confs.append(lk[2])
        knee_y_vals.append(lk[1])
    if rk[2] > KP_CONF:
        knee_confs.append(rk[2])
        knee_y_vals.append(rk[1])
 
    if knee_y_vals and body_scale > 0:
        avg_knee_y = sum(knee_y_vals) / len(knee_y_vals)
        # hip_knee_ratio: negative = knees above hips (sitting)
        # near zero or small positive = standing
        hip_knee_ratio = (hip_y - avg_knee_y) / body_scale
        if hip_knee_ratio < SITTING_HIP_KNEE_RATIO:
            return "sitting", hip_x
    #walkingif prev_hip_x is not None and body_scale > 0:
        hip_speed_px = abs(hip_x - prev_hip_x) * fps / INFER_EVERY_N
        if hip_speed_px > WALKING_SPEED_THRESH_PX:
            return "walking", hip_x
 
    #standing - default position
    return "standing", hip_x

#state manner

class ActivityStateManager:
    def __init__(self):
        self.current_activity    = None
        self.current_row_id      = None
        self.current_started_at  = None
 
        self.candidate_activity  = None
        self.candidate_since     = None
 
        # Stillness tracking
        self.last_movement_time  = time.time()
        self.last_hip_x          = None
        self.stillness_alerted   = False
        
    def update(self, activity, hip_x, current_time):
        logs = []
 
        # stillness detection
        if hip_x is not None:
            if self.last_hip_x is not None:
                movement = abs(hip_x - self.last_hip_x)
                if movement > STILLNESS_MOVEMENT_PX:
                    self.last_movement_time = current_time
                    self.stillness_alerted  = False
            self.last_hip_x = hip_x
 
        hour = datetime.now().hour
        in_waking_hours = WAKING_HOURS_START <= hour < WAKING_HOURS_END
        time_still = current_time - self.last_movement_time

        if (in_waking_hours and
                time_still >= STILLNESS_THRESHOLD_S and
                not self.stillness_alerted and
                activity != "lying_down"):
            self.stillness_alerted = True
            log_stillness(time_still)
            logs.append("[STILLNESS] No movement for " + str(round(time_still)) + "s")
 
        # activity transition handling
        if activity != self.current_activity:
            if activity == self.candidate_activity:
                # Same candidate — check if confirmed long enough
                if current_time - self.candidate_since >= TRANSITION_CONFIRM_S:
                    # Confirmed — close current, open new
                    if self.current_activity is not None and self.current_row_id is not None:
                        duration = log_activity_end(self.current_row_id, self.current_started_at)
                        logs.append("[ACTIVITY] " + str(self.current_activity) +
                                    " ended  duration=" + str(round(duration, 0)) + "s")
                    
                    self.current_activity   = activity
                    self.current_row_id     = log_activity_start(activity)
                    self.current_started_at = datetime.now().isoformat()
                    logs.append("[ACTIVITY] " + str(activity) + " started")
                    self.candidate_activity = None
                    self.candidate_since    = None
            else:
                # New candidate
                self.candidate_activity = activity
                self.candidate_since    = current_time
        else:
            # Same as current — reset candidate
            self.candidate_activity = None
            self.candidate_since    = None
 
        return logs
    def close(self):
        """Call when shutting down to close the last open activity."""
        if self.current_activity is not None and self.current_row_id is not None:
            duration = log_activity_end(self.current_row_id, self.current_started_at)
            print("[ACTIVITY] " + str(self.current_activity) +
                  " closed on exit  duration=" + str(round(duration, 0)) + "s")
            

#debug overlay
def draw_overlay(frame, kps, activity, stats):
    connections = [
        (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
        (KP_LEFT_SHOULDER, KP_LEFT_HIP),
        (KP_RIGHT_SHOULDER, KP_RIGHT_HIP),
        (KP_LEFT_HIP, KP_RIGHT_HIP),
        (KP_LEFT_HIP, KP_LEFT_KNEE),
        (KP_RIGHT_HIP, KP_RIGHT_KNEE),
        (KP_LEFT_KNEE, KP_LEFT_ANKLE),
        (KP_RIGHT_KNEE, KP_RIGHT_ANKLE),
    ]
    color = ACTIVITY_COLORS.get(activity, (80,80,80))
    for a, b in connections:
        if kps[a][2] > KP_CONF and kps[b][2] > KP_CONF:
            cv2.line(frame,
                     (int(kps[a][0]), int(kps[a][1])),
                     (int(kps[b][0]), int(kps[b][1])),
                     color, 2)
    for kp in kps:
        if kp[2] > KP_CONF:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0,220,130), -1)
 
    # Activity banner
    cv2.rectangle(frame, (0,0), (frame.shape[1], 50), (10,10,10), -1)
    cv2.putText(frame, "VIGIL — Activity Monitor", (12, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,130), 1, cv2.LINE_AA)
    cv2.putText(frame, "State: " + str(activity).upper(), (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    #stats sidebar
    lines = [
        "Frame: "   + str(stats.get("frame_num", 0)),
        "Walking: " + str(stats.get("walking_s",  0)) + "s",
        "Standing:"+ str(stats.get("standing_s", 0)) + "s",
        "Sitting: " + str(stats.get("sitting_s",  0)) + "s",
        "Lying:   " + str(stats.get("lying_s",    0)) + "s",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (12, 70 + i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1, cv2.LINE_AA)
    return frame
#main

def run(source, debug=False, write_db=True):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.")
        sys.exit(1)
 
    if write_db:
        init_tables()
 
    print("[INFO] Loading YOLOv8-Pose...")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded")
 
    if str(source).isdigit():
        cap = cv2.VideoCapture(int(source))
    elif str(source).startswith("rtsp://"):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
 
    if not cap.isOpened():
        print("[ERROR] Could not open source: " + str(source))
        sys.exit(1)
 
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("[INFO] fps=" + str(round(fps,1)) + "  frames=" + str(total_frames))
    print("[INFO] Inference every " + str(INFER_EVERY_N) + " frames (~1/sec)")
 
    state_mgr   = ActivityStateManager()
    prev_hip_x  = None
    frame_num   = 0
 
    # Session totals for debug display
    time_in_state = {"walking": 0, "standing": 0, "sitting": 0, "lying_down": 0}
    last_activity = None
    last_log_time = time.time()
 
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
 
            frame_num   += 1
            current_time = time.time()
 
            if frame_num % INFER_EVERY_N != 0:
                if debug:
                    cv2.imshow("VIGIL Activity", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue
 
            results = model(frame, conf=MIN_CONF, verbose=False, device="cuda")
 
            best_kps      = None
            best_conf_val = 0.0
            best_area     = 0
 
            for result in results:
                if result.keypoints is None:
                    continue
                boxes   = result.boxes
                kps_all = result.keypoints.data.cpu().numpy()
                for i, kp in enumerate(kps_all):
                    if boxes is not None and i < len(boxes):
                        c    = float(boxes.conf[i])
                        box  = boxes.xyxy[i].cpu().numpy()
                        area = (box[2]-box[0]) * (box[3]-box[1])
                        if c > best_conf_val and area > best_area:
                            best_conf_val = c
                            best_kps      = kp
                            best_area     = area
 
            if best_kps is None:
                continue
 
            body_scale = compute_body_scale(best_kps)
            activity, hip_x = classify_activity(
                best_kps, prev_hip_x, body_scale, fps
            )
            prev_hip_x = hip_x
 
            # Update state manager
            if write_db:
                logs = state_mgr.update(activity, hip_x, current_time)
                for log_line in logs:
                    print(log_line)
            else:
                # Dry run — still print transitions
                if activity != last_activity:
                    print("[ACTIVITY] " + str(last_activity) + " → " + str(activity))
                    last_activity = activity
 
            # Periodic console summary (every 10s)
            if current_time - last_log_time >= 10:
                print("[STATUS] " + datetime.now().strftime("%H:%M:%S") +
                      "  activity=" + str(activity) +
                      "  frame=" + str(frame_num))
                last_log_time = current_time
 
            # Track time in state for debug overlay
            if activity in time_in_state:
                time_in_state[activity] += INFER_EVERY_N / fps
 
            if debug:
                stats = {
                    "frame_num":  frame_num,
                    "walking_s":  int(time_in_state["walking"]),
                    "standing_s": int(time_in_state["standing"]),
                    "sitting_s":  int(time_in_state["sitting"]),
                    "lying_s":    int(time_in_state["lying_down"]),
                }
                frame = draw_overlay(frame, best_kps, activity, stats)
                dh, dw = frame.shape[:2]
                if dw > 1280:
                    scale = 1280/dw
                    frame = cv2.resize(frame, (1280, int(dh*scale)))
                cv2.imshow("VIGIL Activity", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
 
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")
    finally:
        if write_db:
            state_mgr.close()
        cap.release()
        if debug:
            cv2.destroyAllWindows()
 
    # Final summary
    print("")
    print("=" * 40)
    print("  ACTIVITY SUMMARY")
    print("=" * 40)
    for act, secs in time_in_state.items():
        mins = int(secs // 60)
        s    = int(secs % 60)
        print("  " + act.ljust(12) + str(mins) + "m " + str(s) + "s")
    print("=" * 40)

#entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIGIL Activity Monitor")
    parser.add_argument("--source",  required=True)
    parser.add_argument("--debug",   action="store_true")
    parser.add_argument("--no-db",   action="store_true")
    args = parser.parse_args()
    run(source=args.source, debug=args.debug, write_db=not args.no_db)