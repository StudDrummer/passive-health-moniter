
import argparse
import sqlite3
import os
import sys
import time
import math
import requests
from collections import deque
from datetime import datetime

import cv2
import numpy as np


# config

DB_PATH          = os.path.expanduser("~/passive-health-moniter/vigil.db")
MODEL_PATH       = os.path.expanduser("~/passive-health-moniter/yolov8n-pose.pt")
FLASK_URL        = "http://127.0.0.1:5001"
SESSION_DURATION = 30       # seconds per snapshot window
INFER_EVERY_N    = 2        # run pose estimation every N frames
MIN_CONF         = 0.5      # minimum person detection confidence
KP_CONF          = 0.3      # minimum keypoint confidence

# real world calibration
# average human hip to ankle distance is ~0.9m
# used to convert pixel measurements to real world meters
REAL_HIP_ANKLE_M = 0.90

# fall detection thresholds
FALL_DROP_RATIO      = 0.40   # hip must drop 40% of body height
FALL_TIME_WINDOW_S   = 0.5    # drop must happen within 0.5 seconds
FALL_TORSO_ANGLE_DEG = 45     # torso must be > 45 deg from vertical to confirm fall
FALL_CONFIRM_S       = 1.0    # must stay fallen 1s before alerting
FALL_COOLDOWN_S      = 30     # minimum seconds between fall alerts

# posture warning thresholds
POSTURE_HEAD_FORWARD_WARN = 0.15   # head forward > 15% of body scale = warning
POSTURE_SHOULDER_SYM_WARN = 0.90   # shoulder symmetry < 0.90 = warning
POSTURE_LEAN_WARN         = 10.0   # lean > 10 degrees = warning

# keypoint indices - coco format - yolov8 standard
KP_NOSE=0; KP_LEFT_EYE=1; KP_RIGHT_EYE=2
KP_LEFT_EAR=3; KP_RIGHT_EAR=4
KP_LEFT_SHOULDER=5; KP_RIGHT_SHOULDER=6
KP_LEFT_ELBOW=7; KP_RIGHT_ELBOW=8
KP_LEFT_WRIST=9; KP_RIGHT_WRIST=10
KP_LEFT_HIP=11; KP_RIGHT_HIP=12
KP_LEFT_KNEE=13; KP_RIGHT_KNEE=14
KP_LEFT_ANKLE=15; KP_RIGHT_ANKLE=16


# database

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_tables():
    """create all tables needed if they dont exist yet"""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS fall_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at   TEXT NOT NULL,
            hip_drop_pct  REAL,
            torso_angle   REAL,
            body_scale_px REAL,
            confirmed     INTEGER DEFAULT 1,
            alerted       INTEGER DEFAULT 0,
            notes         TEXT
        );

        CREATE TABLE IF NOT EXISTS posture_metrics (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date        TEXT NOT NULL,
            head_forward_norm   REAL,
            shoulder_sym        REAL,
            body_lean_deg       REAL,
            neck_angle_deg      REAL,
            posture_score       REAL,
            posture_flag        INTEGER DEFAULT 0,
            recorded_at         TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


def log_fall(hip_drop_pct, torso_angle, body_scale_px):
    conn = get_db()
    conn.execute("""
        INSERT INTO fall_events
            (detected_at, hip_drop_pct, torso_angle, body_scale_px, alerted)
        VALUES (?, ?, ?, ?, 1)
    """, (
        datetime.now().isoformat(),
        round(hip_drop_pct, 3),
        round(torso_angle, 1),
        round(body_scale_px, 1),
    ))
    conn.commit()
    conn.close()
    print("[FALL] Logged to database")


def write_gait_snapshot(metrics):
    """write one sessions gait metrics to camera_metrics table"""
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
    print("[DB] Gait snapshot written")


def write_posture_snapshot(metrics):
    """write posture metrics - set flag if any thresholds exceeded"""
    flag = int(
        (metrics.get("head_forward_norm") or 0) > POSTURE_HEAD_FORWARD_WARN or
        (metrics.get("shoulder_sym") or 1)      < POSTURE_SHOULDER_SYM_WARN or
        abs(metrics.get("body_lean_deg") or 0)  > POSTURE_LEAN_WARN
    )
    conn = get_db()
    conn.execute("""
        INSERT INTO posture_metrics (
            session_date, head_forward_norm, shoulder_sym,
            body_lean_deg, neck_angle_deg, posture_score, posture_flag
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d"),
        metrics.get("head_forward_norm"),
        metrics.get("shoulder_sym"),
        metrics.get("body_lean_deg"),
        metrics.get("neck_angle_deg"),
        metrics.get("posture_score"),
        flag,
    ))
    conn.commit()
    conn.close()
    print("[DB] Posture snapshot written  flag=" + str(flag))


# alerts

def send_fall_alert(hip_drop_pct, torso_angle):
    """post fall alert to flask server - app polls /alerts/recent every 10s"""
    try:
        requests.post(
            FLASK_URL + "/alert/fall",
            json={
                "detected_at":  datetime.now().isoformat(),
                "hip_drop_pct": round(hip_drop_pct, 3),
                "torso_angle":  round(torso_angle, 1),
                "message":      "Fall detected — please check on the person.",
            },
            timeout=3,
        )
        print("[ALERT] Fall alert sent to Flask server")
    except Exception as e:
        print("[ALERT] Could not reach Flask server: " + str(e))


# geometry helpers

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def angle_from_vertical(p_bottom, p_top):
    """angle between vertical axis and the vector from p_bottom to p_top
    0 = perfectly upright, positive = leaning forward"""
    dx = p_top[0]    - p_bottom[0]
    dy = p_bottom[1] - p_top[1]    # Y is inverted in image coords
    return math.degrees(math.atan2(dx, dy))

def fv(val, fmt, suffix=""):
    """format value for printing - returns dash if None"""
    if val is None:
        return "—"
    return format(val, fmt) + suffix


# fall detector class
# keeps a short history of hip Y positions to detect rapid downward displacement

class FallDetector:
    def __init__(self, fps):
        history_frames       = max(5, int(fps * FALL_TIME_WINDOW_S))
        self.hip_y_history   = deque(maxlen=history_frames)
        self.time_history    = deque(maxlen=history_frames)
        self.fallen_since    = None
        self.last_alert_time = 0
        self.state           = "upright"   # upright | fallen

    def update(self, kps, body_scale_px, current_time):
        """call every frame - returns ("fall", drop_pct, angle) if confirmed fall
        otherwise returns (None, None, None)"""
        lh = kps[KP_LEFT_HIP];  rh = kps[KP_RIGHT_HIP]
        ls = kps[KP_LEFT_SHOULDER]; rs = kps[KP_RIGHT_SHOULDER]

        # need hip keypoints to do anything
        if lh[2] < KP_CONF or rh[2] < KP_CONF:
            return None, None, None

        hip_y = (lh[1] + rh[1]) / 2
        self.hip_y_history.append(hip_y)
        self.time_history.append(current_time)

        if body_scale_px <= 0:
            return None, None, None

        # how far did the hip drop compared to its highest point in the window
        # in image coords Y increases downward so a fall = Y increases
        hip_drop_pct = 0.0
        if len(self.hip_y_history) >= 2:
            hip_drop_pct = (hip_y - min(self.hip_y_history)) / body_scale_px

        # torso angle from vertical - high angle = person is sideways = fallen
        torso_angle = 0.0
        if ls[2] > KP_CONF and rs[2] > KP_CONF:
            torso_angle = abs(angle_from_vertical(
                midpoint(lh[:2], rh[:2]),
                midpoint(ls[:2], rs[:2])
            ))

        is_dropped  = hip_drop_pct >= FALL_DROP_RATIO
        is_sideways = torso_angle  >= FALL_TORSO_ANGLE_DEG

        if self.state == "upright":
            if is_dropped and is_sideways:
                self.state        = "fallen"
                self.fallen_since = current_time
                print("[FALL] Fall state entered  drop=" +
                      str(round(hip_drop_pct * 100, 1)) + "%  angle=" +
                      str(round(torso_angle, 1)) + "deg")

        elif self.state == "fallen":
            if not is_dropped and not is_sideways:
                # person got back up
                self.state        = "upright"
                self.fallen_since = None
                print("[FALL] Person upright again")
            else:
                # still fallen - check if confirmed long enough to alert
                time_fallen = current_time - self.fallen_since
                if (time_fallen >= FALL_CONFIRM_S and
                        current_time - self.last_alert_time > FALL_COOLDOWN_S):
                    self.last_alert_time = current_time
                    return "fall", hip_drop_pct, torso_angle

        return None, None, None

    def is_fallen(self):
        return self.state == "fallen"


# posture module
# extracts head position, shoulder symmetry, neck angle per frame
# all measurements normalized to body scale so camera distance doesnt matter

def extract_posture_metrics(kps, body_scale_px):
    """returns dict with head_forward_norm, shoulder_sym, body_lean_deg,
    neck_angle_deg, posture_score"""
    p = {}

    ls   = kps[KP_LEFT_SHOULDER];  rs = kps[KP_RIGHT_SHOULDER]
    lh   = kps[KP_LEFT_HIP];       rh = kps[KP_RIGHT_HIP]
    le   = kps[KP_LEFT_EAR];       re = kps[KP_RIGHT_EAR]
    nose = kps[KP_NOSE]

    # body lean - angle of torso from vertical
    if ls[2] > KP_CONF and rs[2] > KP_CONF and lh[2] > KP_CONF and rh[2] > KP_CONF:
        p["body_lean_deg"] = angle_from_vertical(
            midpoint(lh[:2], rh[:2]),
            midpoint(ls[:2], rs[:2])
        )
    else:
        p["body_lean_deg"] = None

    # shoulder symmetry - how level are the shoulders
    # asymmetry can indicate scoliosis, muscle imbalance, or stroke
    if ls[2] > KP_CONF and rs[2] > KP_CONF and body_scale_px > 0:
        shoulder_y_diff  = abs(ls[1] - rs[1])
        p["shoulder_sym"] = max(0.0, 1.0 - (shoulder_y_diff / body_scale_px))
    else:
        p["shoulder_sym"] = None

    # head forward position - how far is the nose ahead of the shoulder midpoint
    # normalized to body scale. positive = head forward
    # parkinson's early marker and text neck indicator
    if nose[2] > KP_CONF and ls[2] > KP_CONF and rs[2] > KP_CONF and body_scale_px > 0:
        shoulder_mid_x        = (ls[0] + rs[0]) / 2
        p["head_forward_norm"] = (nose[0] - shoulder_mid_x) / body_scale_px
    else:
        p["head_forward_norm"] = None

    # neck angle - angle of ear to shoulder vector from vertical
    # >25 degrees = forward head posture (text neck)
    # use whichever ear has higher confidence
    neck_angle = None
    if ls[2] > KP_CONF and le[2] > KP_CONF:
        neck_angle = abs(angle_from_vertical(ls[:2], le[:2]))
    elif rs[2] > KP_CONF and re[2] > KP_CONF:
        neck_angle = abs(angle_from_vertical(rs[:2], re[:2]))
    p["neck_angle_deg"] = neck_angle

    # composite posture score 0-100, higher = better posture
    # subtract penalties for each metric out of range
    score    = 100.0
    penalties = 0

    if p["body_lean_deg"] is not None:
        score    -= min(abs(p["body_lean_deg"]) * 2.0, 30.0)
        penalties += 1
    if p["shoulder_sym"] is not None:
        score    -= min((1.0 - p["shoulder_sym"]) * 150.0, 25.0)
        penalties += 1
    if p["head_forward_norm"] is not None:
        score    -= min(abs(p["head_forward_norm"]) * 100.0, 25.0)
        penalties += 1
    if p["neck_angle_deg"] is not None:
        score    -= min(max(0, p["neck_angle_deg"] - 10) * 1.0, 20.0)
        penalties += 1

    p["posture_score"] = max(0.0, score) if penalties > 0 else None

    return p


def aggregate_posture(posture_frames):
    """median of all per-frame posture readings over the session"""
    def med(key):
        vals = [f[key] for f in posture_frames if f.get(key) is not None]
        return float(np.median(vals)) if vals else None

    return {
        "head_forward_norm": med("head_forward_norm"),
        "shoulder_sym":      med("shoulder_sym"),
        "body_lean_deg":     med("body_lean_deg"),
        "neck_angle_deg":    med("neck_angle_deg"),
        "posture_score":     med("posture_score"),
    }


# gait metrics

def detect_camera_mode(hip_positions):
    """detect if camera is side facing, front facing, or angled
    based on how the hip centroid moves over the session"""
    if len(hip_positions) < 10:
        return "unknown"
    xs      = [p[0] for p in hip_positions]
    ys      = [p[1] for p in hip_positions]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    if x_range > y_range * 2.5:
        return "side"
    elif y_range > x_range * 1.5:
        return "front"
    return "angled"


def compute_px_per_meter(kps):
    """compute pixels per meter using hip to ankle distance as reference
    use whichever side has higher confidence keypoints"""
    lh = kps[KP_LEFT_HIP];  la = kps[KP_LEFT_ANKLE]
    rh = kps[KP_RIGHT_HIP]; ra = kps[KP_RIGHT_ANKLE]
    left_conf  = min(lh[2], la[2])
    right_conf = min(rh[2], ra[2])
    if left_conf >= right_conf and left_conf > KP_CONF:
        dist_px = distance(lh[:2], la[:2])
    elif right_conf > KP_CONF:
        dist_px = distance(rh[:2], ra[:2])
    else:
        return 0.0
    return (dist_px / REAL_HIP_ANKLE_M) if dist_px >= 10 else 0.0


def extract_frame_metrics(kps, prev_hip_x, fps, px_per_m):
    """extract gait metrics from a single frame's keypoints"""
    m  = {}
    lh = kps[KP_LEFT_HIP]; rh = kps[KP_RIGHT_HIP]
    m["px_per_m"]   = px_per_m
    m["body_scale"] = px_per_m * REAL_HIP_ANKLE_M

    # hip centroid position and walking speed
    if lh[2] > KP_CONF and rh[2] > KP_CONF:
        hip_x      = (lh[0] + rh[0]) / 2
        hip_y      = (lh[1] + rh[1]) / 2
        m["hip_x"] = hip_x
        m["hip_y"] = hip_y
        if prev_hip_x is not None and px_per_m > 0:
            m["walking_speed_ms"] = (abs(hip_x - prev_hip_x) / px_per_m) * fps / INFER_EVERY_N
    else:
        m["hip_x"] = None
        m["hip_y"] = None

    # body lean
    ls = kps[KP_LEFT_SHOULDER]; rs = kps[KP_RIGHT_SHOULDER]
    if ls[2] > KP_CONF and rs[2] > KP_CONF and lh[2] > KP_CONF and rh[2] > KP_CONF:
        m["body_lean_deg"] = angle_from_vertical(
            midpoint(lh[:2], rh[:2]),
            midpoint(ls[:2], rs[:2])
        )
    else:
        m["body_lean_deg"] = None

    # step symmetry - ratio of left vs right hip-to-ankle distance
    la = kps[KP_LEFT_ANKLE]; ra = kps[KP_RIGHT_ANKLE]
    if lh[2] > KP_CONF and rh[2] > KP_CONF and la[2] > KP_CONF and ra[2] > KP_CONF:
        ls_ = distance(lh[:2], la[:2])
        rs_ = distance(rh[:2], ra[:2])
        m["step_symmetry"] = (min(ls_, rs_) / max(ls_, rs_)) if max(ls_, rs_) > 0 else None
    else:
        m["step_symmetry"] = None

    return m


def aggregate_gait(frame_metrics, camera_mode):
    """aggregate frame level metrics into a single session snapshot
    uses median for robustness against outliers"""
    def med(key):
        vals = [m[key] for m in frame_metrics if m.get(key) is not None]
        return float(np.median(vals)) if vals else None

    walking_speed_ms = med("walking_speed_ms")
    body_lean_deg    = med("body_lean_deg")
    step_symmetry    = med("step_symmetry")
    body_scale_px    = med("body_scale")
    px_per_m         = med("px_per_m")

    # lateral sway = std dev of hip X position normalized to body height
    hip_xs    = [m["hip_x"] for m in frame_metrics if m.get("hip_x") is not None]
    sway_px   = float(np.std(hip_xs)) if len(hip_xs) > 2 else None
    sway_norm = (sway_px / body_scale_px) if (sway_px and body_scale_px) else None

    # asymmetry percentage (0 = perfect, 100 = completely asymmetric)
    asymmetry_pct = ((1 - step_symmetry) * 100) if step_symmetry is not None else None

    # posture score from gait data (lean + sway)
    posture_score = None
    if body_lean_deg is not None and sway_norm is not None:
        posture_score = max(0, 100 - min(abs(body_lean_deg) * 2, 50) - min(sway_norm * 200, 50))

    return {
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
        "frame_count":       len(frame_metrics),
        "cadence_spm":       None,
    }


# print helpers

def print_session(label, session_count, gait, posture):
    """print gait and posture blocks separately as requested"""
    ts = datetime.now().strftime("%H:%M:%S")
    print("")
    print("=" * 52)
    print("  " + label + " " + str(session_count) + " — " + ts)
    print("=" * 52)

    # gait block
    print("  === GAIT ===")
    print("  Camera mode:    " + str(gait["camera_mode"]))
    print("  Walking speed:  " + fv(gait["walking_speed_ms"], ".3f", " m/s"))
    print("  Body lean:      " + fv(gait["body_lean_deg"],    ".1f", " deg"))
    print("  Lateral sway:   " + fv(gait["lateral_sway_norm"],".3f", " norm"))
    print("  Step symmetry:  " + fv(gait["step_symmetry"],    ".3f"))
    print("  Asymmetry:      " + fv(gait["asymmetry_pct"],    ".1f", "%"))
    print("  Body scale:     " + fv(gait["body_scale_px"],    ".1f", " px"))
    print("  Frames used:    " + str(gait["frame_count"]))

    # posture block
    print("")
    print("  === POSTURE ===")
    print("  Head forward:   " + fv(posture.get("head_forward_norm"), ".3f", " norm"))
    print("  Shoulder sym:   " + fv(posture.get("shoulder_sym"),      ".3f"))
    print("  Neck angle:     " + fv(posture.get("neck_angle_deg"),    ".1f", " deg"))
    print("  Body lean:      " + fv(posture.get("body_lean_deg"),     ".1f", " deg"))
    print("  Posture score:  " + fv(posture.get("posture_score"),     ".1f", "/100"))

    # warnings if any thresholds exceeded
    if (posture.get("head_forward_norm") or 0) > POSTURE_HEAD_FORWARD_WARN:
        print("  [!] Head forward posture detected")
    if (posture.get("shoulder_sym") or 1) < POSTURE_SHOULDER_SYM_WARN:
        print("  [!] Shoulder asymmetry detected")
    if abs(posture.get("body_lean_deg") or 0) > POSTURE_LEAN_WARN:
        print("  [!] Excessive body lean")
    if (posture.get("neck_angle_deg") or 0) > 25:
        print("  [!] Forward head posture (neck angle)")

    print("=" * 52)
    print("")


# debug overlay

def draw_overlay(frame, kps, fm, pm, session_info, fall_state):
    """draw skeleton, fall banner, and live metric readout on frame"""
    connections = [
        (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
        (KP_LEFT_SHOULDER, KP_LEFT_HIP),
        (KP_RIGHT_SHOULDER, KP_RIGHT_HIP),
        (KP_LEFT_HIP, KP_RIGHT_HIP),
        (KP_LEFT_HIP, KP_LEFT_KNEE),
        (KP_RIGHT_HIP, KP_RIGHT_KNEE),
        (KP_LEFT_KNEE, KP_LEFT_ANKLE),
        (KP_RIGHT_KNEE, KP_RIGHT_ANKLE),
        (KP_LEFT_SHOULDER, KP_LEFT_ELBOW),
        (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW),
        (KP_LEFT_SHOULDER, KP_NOSE),
        (KP_RIGHT_SHOULDER, KP_NOSE),
    ]
    skel_color = (0, 0, 255) if fall_state else (0, 180, 100)
    for a, b in connections:
        if kps[a][2] > KP_CONF and kps[b][2] > KP_CONF:
            cv2.line(frame,
                     (int(kps[a][0]), int(kps[a][1])),
                     (int(kps[b][0]), int(kps[b][1])),
                     skel_color, 2)
    for kp in kps:
        if kp[2] > KP_CONF:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 220, 130), -1)

    # red banner when fall detected
    if fall_state:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 180), -1)
        cv2.putText(frame, "FALL DETECTED", (12, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    # live metrics overlay
    lines = [
        "VIGIL — Gait + Posture + Fall",
        "Mode:      " + str(session_info.get("camera_mode", "...")),
        "Speed:     " + fv(fm.get("walking_speed_ms"), ".2f", " m/s"),
        "Lean:      " + fv(fm.get("body_lean_deg"),    ".1f", " deg"),
        "Symmetry:  " + fv(fm.get("step_symmetry"),    ".2f"),
        "---",
        "Head fwd:  " + fv(pm.get("head_forward_norm"), ".3f"),
        "Shldr sym: " + fv(pm.get("shoulder_sym"),      ".3f"),
        "Neck ang:  " + fv(pm.get("neck_angle_deg"),    ".1f", " deg"),
        "Posture:   " + fv(pm.get("posture_score"),     ".1f", "/100"),
    ]
    y_start = 80 if fall_state else 24
    for i, line in enumerate(lines):
        color = (0, 220, 130) if i == 0 else (100, 100, 100) if line == "---" else (200, 200, 200)
        cv2.putText(frame, line, (12, y_start + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
    return frame


# main pipeline

def run(source, debug=False, write_db=True):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip3 install ultralytics")
        sys.exit(1)

    init_tables()

    print("[INFO] Loading YOLOv8-Pose from " + MODEL_PATH)
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded")

    # open video source - supports video file, rtsp stream, or csi/usb camera
    if str(source).isdigit():
        cap = cv2.VideoCapture(int(source))
    elif str(source).startswith("rtsp://"):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("[ERROR] Could not open source: " + str(source))
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps if total_frames > 0 else 0

    print("[INFO] Source opened: fps=" + str(round(fps, 1)) +
          "  frames=" + str(total_frames) +
          "  duration=" + str(round(duration_s, 1)) + "s")
    print("[INFO] Session window: " + str(SESSION_DURATION) + "s  debug=" + str(debug))

    fall_detector  = FallDetector(fps)
    frame_num      = 0
    session_start  = time.time()
    frame_metrics  = []
    posture_frames = []
    hip_positions  = []
    prev_hip_x     = None
    session_count  = 0
    fall_count     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream")
            break

        frame_num += 1

        # only run inference every N frames to save compute
        if frame_num % INFER_EVERY_N != 0:
            continue

        # pose estimation
        results = model(frame, conf=MIN_CONF, verbose=False, device="cuda")

        # get the largest most confident person in frame
        # will need to revisit this once we have multi-person scenarios
        best_kps      = None
        best_conf     = 0.0
        best_box_area = 0

        for result in results:
            if result.keypoints is None:
                continue
            boxes   = result.boxes
            kps_all = result.keypoints.data.cpu().numpy()
            for i, kp in enumerate(kps_all):
                if boxes is not None and i < len(boxes):
                    conf = float(boxes.conf[i])
                    box  = boxes.xyxy[i].cpu().numpy()
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if conf > best_conf and area > best_box_area:
                        best_conf     = conf
                        best_kps      = kp
                        best_box_area = area

        if best_kps is None:
            continue

        px_per_m     = compute_px_per_meter(best_kps)
        body_scale   = px_per_m * REAL_HIP_ANKLE_M
        current_time = time.time()

        # fall detection - runs every frame for real time response
        event, hip_drop_pct, torso_angle = fall_detector.update(
            best_kps, body_scale, current_time
        )
        if event == "fall":
            fall_count += 1
            print("[FALL] CONFIRMED FALL #" + str(fall_count) +
                  "  drop=" + str(round(hip_drop_pct * 100, 1)) +
                  "%  angle=" + str(round(torso_angle, 1)) + "deg")
            if write_db:
                log_fall(hip_drop_pct, torso_angle, body_scale)
            send_fall_alert(hip_drop_pct, torso_angle)

        # gait metrics - accumulated per frame, aggregated at session end
        fm = extract_frame_metrics(best_kps, prev_hip_x, fps, px_per_m)
        frame_metrics.append(fm)
        if fm.get("hip_x") is not None:
            hip_positions.append((fm["hip_x"], fm["hip_y"]))
            prev_hip_x = fm["hip_x"]

        # posture metrics - same deal, accumulated per frame
        pm = extract_posture_metrics(best_kps, body_scale)
        posture_frames.append(pm)

        # debug overlay if enabled
        if debug:
            session_info = {
                "camera_mode": detect_camera_mode(hip_positions),
                "frame_count": len(frame_metrics),
            }
            frame = draw_overlay(frame, best_kps, fm, pm, session_info,
                                 fall_detector.is_fallen())
            dh, dw = frame.shape[:2]
            if dw > 1280:
                scale = 1280 / dw
                frame = cv2.resize(frame, (1280, int(dh * scale)))
            cv2.imshow("VIGIL", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # session window complete - aggregate and write
        if time.time() - session_start >= SESSION_DURATION:
            camera_mode   = detect_camera_mode(hip_positions)
            gait_snap     = aggregate_gait(frame_metrics, camera_mode)
            posture_snap  = aggregate_posture(posture_frames)
            session_count += 1
            print_session("SESSION", session_count, gait_snap, posture_snap)
            if write_db:
                write_gait_snapshot(gait_snap)
                write_posture_snapshot(posture_snap)
            # reset for next window
            frame_metrics  = []
            posture_frames = []
            hip_positions  = []
            prev_hip_x     = None
            session_start  = time.time()

    # end of stream - process any remaining frames
    if len(frame_metrics) >= 10:
        camera_mode   = detect_camera_mode(hip_positions)
        gait_snap     = aggregate_gait(frame_metrics, camera_mode)
        posture_snap  = aggregate_posture(posture_frames)
        session_count += 1
        print_session("FINAL SESSION", session_count, gait_snap, posture_snap)
        if write_db:
            write_gait_snapshot(gait_snap)
            write_posture_snapshot(posture_snap)

    cap.release()
    if debug:
        cv2.destroyAllWindows()

    print("[INFO] Done — " + str(session_count) + " sessions, " +
          str(fall_count) + " falls detected")


# entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIGIL Gait Snapshot + Posture + Fall Detection")
    parser.add_argument("--source",  required=True, help="Video file, RTSP URL, or camera index (0)")
    parser.add_argument("--debug",   action="store_true", help="Show debug window with skeleton overlay")
    parser.add_argument("--no-db",   action="store_true", help="Dry run - dont write to database")
    args = parser.parse_args()
    run(source=args.source, debug=args.debug, write_db=not args.no_db)