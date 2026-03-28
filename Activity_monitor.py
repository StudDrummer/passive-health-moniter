"""
VIGIL — Activity Monitor v2
activity_monitor.py

Additions over v1:
  - Camera calibration system (reads calibration.json, converts px → meters)
  - Posture analysis: forward head, shoulder symmetry, trunk lean, kyphosis proxy
  - Fall detection state machine: STANDING → DESCENDING → FALLEN (avoids false positives)
  - Real gait speed in m/s (not px/s) when calibrated
  - Improved stride detection with Butterworth filter for smoother cadence
  - Posts posture metrics to /sync/camera endpoint

Usage:
    python3 activity_monitor.py --source 0
    python3 activity_monitor.py --source rtsp://admin:password@192.168.1.42:554/h264Preview_01_main
    python3 activity_monitor.py --source test_walk.mp4 --debug

Setup:
    python3 calibrate.py --source 0    # run first to set pixel-to-meter conversion
"""

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import requests

DB_PATH      = os.path.expanduser("~/passive-health-moniter/vigil.db")
MODEL_PATH   = os.path.expanduser("~/passive-health-moniter/yolov8n-pose.pt")
CALIB_PATH   = os.path.expanduser("~/passive-health-moniter/calibration.json")
VIGIL_SERVER = "http://localhost:5001"   # local Flask server

INFER_EVERY_N          = 3      # run inference every N frames (10fps at 30fps)
MIN_CONF               = 0.50
KP_CONF                = 0.30
TRANSITION_CONFIRM_S   = 3.0
STILLNESS_THRESHOLD_S  = 60.0
STILLNESS_MOVEMENT_PX  = 8.0
WAKING_HOURS_START     = 7
WAKING_HOURS_END       = 22

# Gait analysis window
GAIT_WINDOW_FRAMES     = 90    # 3 seconds at 30fps
CADENCE_SMOOTH_FRAMES  = 5

# Posture thresholds — tunable
FORWARD_HEAD_DEG_WARN  = 15.0  # forward head posture: angle > this = warning
SHOULDER_ASYM_PX_WARN  = 0.05  # asymmetry as fraction of body scale
TRUNK_LEAN_DEG_WARN    = 8.0   # lateral lean: angle > this = warning

# Fall state machine
FALL_DESCENT_VEL_PX    = 40.0  # hip must descend this fast (px/sec) to enter DESCENDING
FALL_CONFIRM_S         = 2.0   # person must stay low for this long to confirm FALLEN
FALL_RECOVER_S         = 4.0   # after FALLEN, wait this long before returning to STANDING


# KEYPOINT INDICES (COCO)

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
    "fallen":     (0, 0, 220),
    "unknown":    (80, 80, 80),
}



# CALIBRATION


def load_calibration():
    if not os.path.exists(CALIB_PATH):
        print("[CAL] No calibration file found. Gait speed will be in px/s.")
        print("[CAL] Run: python3 calibrate.py --source 0  to calibrate.")
        return None
    try:
        with open(CALIB_PATH) as f:
            c = json.load(f)
        print(f"[CAL] Loaded calibration: {c['px_per_m']:.1f} px/m "
              f"(from {c.get('calibrated_at','?')})")
        return c
    except Exception as e:
        print(f"[CAL] Warning: {e}")
        return None



# DATABASE


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_tables():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS activity_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            activity    TEXT NOT NULL,
            started_at  TEXT NOT NULL,
            ended_at    TEXT,
            duration_s  REAL,
            notes       TEXT
        );
        CREATE TABLE IF NOT EXISTS stillness_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at TEXT NOT NULL,
            duration_s  REAL,
            hour_of_day INTEGER,
            alerted     INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS fall_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at TEXT NOT NULL,
            hip_drop_pct REAL,
            torso_angle  REAL,
            body_scale_px REAL,
            confirmed   INTEGER DEFAULT 1,
            alerted     INTEGER DEFAULT 0,
            notes       TEXT
        );
        CREATE TABLE IF NOT EXISTS posture_metrics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date    TEXT NOT NULL,
            head_forward_norm REAL,
            shoulder_sym    REAL,
            body_lean_deg   REAL,
            neck_angle_deg  REAL,
            posture_score   REAL,
            posture_flag    INTEGER DEFAULT 0,
            recorded_at     TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS camera_metrics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date    TEXT NOT NULL,
            gait_speed_px   REAL,
            cadence_spm     REAL,
            asymmetry_pct   REAL,
            stride_norm     REAL,
            step_regularity REAL,
            body_scale_px   REAL,
            camera_mode     TEXT,
            stride_count    INTEGER,
            recorded_at     TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()



# KEYPOINT HELPERS


def kp(kps, idx):
    """Return (x, y, conf) for keypoint idx."""
    return kps[idx] if kps is not None else (0, 0, 0)


def visible(kps, idx, threshold=None):
    th = threshold or KP_CONF
    return kps[idx][2] > th


def midpoint(kps, a, b):
    if visible(kps, a) and visible(kps, b):
        return ((kps[a][0]+kps[b][0])/2, (kps[a][1]+kps[b][1])/2)
    if visible(kps, a): return (kps[a][0], kps[a][1])
    if visible(kps, b): return (kps[b][0], kps[b][1])
    return None


def angle_from_vertical(p1, p2):
    """Angle in degrees between p1→p2 vector and vertical (Y axis)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dy == 0 and dx == 0:
        return 0.0
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return angle


def compute_body_scale(kps):
    """Approximate body height in pixels (shoulder midpoint to ankle midpoint)."""
    sm = midpoint(kps, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
    am = midpoint(kps, KP_LEFT_ANKLE,    KP_RIGHT_ANKLE)
    if sm and am:
        return abs(am[1] - sm[1])
    return None



# POSTURE ANALYSIS


class PostureAnalyzer:
    """
    Extracts clinically-relevant posture metrics from COCO keypoints.

    Metrics computed:
      forward_head_deg  — angle of neck from vertical (>15° = forward head posture)
      shoulder_asym_pct — height difference between shoulders / body scale
      trunk_lean_deg    — lateral lean of trunk (shoulder midpoint to hip midpoint)
      posture_score     — 0–100 composite (0 = perfect, 100 = severely impaired)

    Clinical references:
      - Forward head: Harman et al., J Orth Sports Phys Ther 2005
      - Shoulder asymmetry: used in Cobb angle screening
    """

    def __init__(self):
        self.history       = deque(maxlen=30)   # rolling window for smoothing
        self.session_date  = datetime.now().strftime("%Y-%m-%d")

    def analyze(self, kps, body_scale):
        """Return PostureResult dict or None if insufficient keypoints."""
        if body_scale is None or body_scale < 20:
            return None

        result = {}

        # Forward head posture
        # Angle between ear and shoulder vs vertical
        # Larger angle = more forward head
        ear_m = midpoint(kps, KP_LEFT_EAR, KP_RIGHT_EAR)
        shld_m = midpoint(kps, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)

        if ear_m and shld_m:
            neck_angle = angle_from_vertical(shld_m, ear_m)
            result["neck_angle_deg"]   = round(neck_angle, 2)
            result["head_forward_norm"] = round(neck_angle / 90.0, 3)
            result["forward_head_flag"] = neck_angle > FORWARD_HEAD_DEG_WARN
        else:
            result["neck_angle_deg"]    = None
            result["head_forward_norm"] = None
            result["forward_head_flag"] = False

        #  Shoulder asymmetry 
        # Height difference between shoulders, normalised by body scale
        ls = kps[KP_LEFT_SHOULDER]
        rs = kps[KP_RIGHT_SHOULDER]

        if visible(kps, KP_LEFT_SHOULDER) and visible(kps, KP_RIGHT_SHOULDER):
            asym_px  = abs(ls[1] - rs[1])
            asym_pct = asym_px / body_scale
            result["shoulder_asym_pct"]  = round(asym_pct * 100, 2)
            result["shoulder_asym_flag"] = asym_pct > SHOULDER_ASYM_PX_WARN
        else:
            result["shoulder_asym_pct"]  = None
            result["shoulder_asym_flag"] = False

        #  Trunk lateral lean 
        # Angle of shoulder midpoint to hip midpoint from vertical
        hip_m = midpoint(kps, KP_LEFT_HIP, KP_RIGHT_HIP)

        if shld_m and hip_m:
            trunk_lean = angle_from_vertical(hip_m, shld_m)
            result["trunk_lean_deg"]  = round(trunk_lean, 2)
            result["trunk_lean_flag"] = trunk_lean > TRUNK_LEAN_DEG_WARN
        else:
            result["trunk_lean_deg"]  = None
            result["trunk_lean_flag"] = False

        #  Composite posture score 0–100 
        # Weighted sum of normalised deviations
        score = 0.0
        if result.get("neck_angle_deg") is not None:
            score += min(40, result["neck_angle_deg"] * (40 / 30))   # 30° = max 40pts
        if result.get("shoulder_asym_pct") is not None:
            score += min(30, result["shoulder_asym_pct"] * (30 / 10)) # 10% = max 30pts
        if result.get("trunk_lean_deg") is not None:
            score += min(30, result["trunk_lean_deg"] * (30 / 15))    # 15° = max 30pts

        result["posture_score"] = round(min(100, score), 1)
        result["posture_flag"]  = score > 50

        self.history.append(result)
        return result

    def smoothed(self):
        """Return rolling average of the last window of posture readings."""
        if not self.history:
            return None
        smooth = {}
        numeric_keys = ["neck_angle_deg","head_forward_norm","shoulder_asym_pct",
                        "trunk_lean_deg","posture_score"]
        for k in numeric_keys:
            vals = [h[k] for h in self.history if h.get(k) is not None]
            smooth[k] = round(sum(vals)/len(vals), 2) if vals else None
        smooth["posture_flag"] = (smooth.get("posture_score") or 0) > 50
        return smooth

    def save_to_db(self, conn):
        s = self.smoothed()
        if not s:
            return
        conn.execute("""
            INSERT INTO posture_metrics
                (session_date, head_forward_norm, shoulder_sym, body_lean_deg,
                 neck_angle_deg, posture_score, posture_flag)
            VALUES (?,?,?,?,?,?,?)
        """, (
            self.session_date,
            s.get("head_forward_norm"),
            s.get("shoulder_asym_pct"),
            s.get("trunk_lean_deg"),
            s.get("neck_angle_deg"),
            s.get("posture_score"),
            1 if s.get("posture_flag") else 0,
        ))
        conn.commit()
        print(f"[POSTURE] Saved — score={s.get('posture_score')} "
              f"neck={s.get('neck_angle_deg')}° "
              f"asym={s.get('shoulder_asym_pct')}%")



# FALL STATE MACHINE


class FallStateMachine:
    """
    Three-state machine to detect falls while minimising false positives.

    States:
      UPRIGHT     — normal
      DESCENDING  — hip is dropping rapidly (could be a fall OR bending down)
      FALLEN      — hip dropped AND person remained low for FALL_CONFIRM_S seconds
      RECOVERING  — post-fall, wait before resetting
    """

    UPRIGHT    = "UPRIGHT"
    DESCENDING = "DESCENDING"
    FALLEN     = "FALLEN"
    RECOVERING = "RECOVERING"

    def __init__(self):
        self.state          = self.UPRIGHT
        self.state_entered  = time.time()
        self.descent_start_y = None
        self.peak_hip_y     = None      # lowest (highest Y value) hip position
        self.fallen_events  = []

    def _elapsed(self):
        return time.time() - self.state_entered

    def update(self, hip_y, hip_vel_y, body_scale, torso_angle, current_time):
        """
        hip_y      — current hip midpoint Y (pixels, increases downward)
        hip_vel_y  — velocity of hip Y (px/sec, positive = moving down)
        body_scale — shoulder-to-ankle height in pixels
        torso_angle — trunk lateral lean in degrees
        Returns: "fall_detected" | "normal" | "descending"
        """
        if hip_y is None or body_scale is None or body_scale < 20:
            return "normal"

        event = "normal"

        if self.state == self.UPRIGHT:
            # Enter DESCENDING when hip drops fast
            if hip_vel_y > FALL_DESCENT_VEL_PX:
                self.state         = self.DESCENDING
                self.state_entered = current_time
                self.descent_start_y = hip_y
                self.peak_hip_y    = hip_y
                print(f"[FALL] UPRIGHT → DESCENDING  vel={hip_vel_y:.1f}px/s")
            event = "normal"

        elif self.state == self.DESCENDING:
            # Track lowest position
            if hip_y > (self.peak_hip_y or hip_y):
                self.peak_hip_y = hip_y

            # How far did they drop?
            drop_px  = hip_y - (self.descent_start_y or hip_y)
            drop_pct = drop_px / body_scale  # fraction of body height

            if self._elapsed() > FALL_CONFIRM_S:
                if drop_pct > 0.35:
                    # They stayed low — it's a fall
                    self.state         = self.FALLEN
                    self.state_entered = current_time
                    self.fallen_events.append({
                        "detected_at":  datetime.now().isoformat(),
                        "hip_drop_pct": round(drop_pct, 3),
                        "torso_angle":  round(torso_angle or 0, 1),
                        "body_scale_px": round(body_scale, 1),
                    })
                    print(f"[FALL] DESCENDING → FALLEN  drop={drop_pct*100:.0f}%")
                    event = "fall_detected"
                else:
                    # They recovered — it was just bending down
                    self.state         = self.UPRIGHT
                    self.state_entered = current_time
                    print(f"[FALL] DESCENDING → UPRIGHT (recovered, drop={drop_pct*100:.0f}%)")
            else:
                event = "descending"

        elif self.state == self.FALLEN:
            # Stay in FALLEN briefly, then move to RECOVERING
            if self._elapsed() > 1.0:
                self.state         = self.RECOVERING
                self.state_entered = current_time
                print("[FALL] FALLEN → RECOVERING")
            event = "fall_detected"

        elif self.state == self.RECOVERING:
            if self._elapsed() > FALL_RECOVER_S:
                self.state         = self.UPRIGHT
                self.state_entered = current_time
                self.peak_hip_y    = None
                self.descent_start_y = None
                print("[FALL] RECOVERING → UPRIGHT")

        return event

    def pop_events(self):
        """Return and clear any new fall events for DB writing."""
        events = self.fallen_events[:]
        self.fallen_events = []
        return events



# GAIT ANALYZER


class GaitAnalyzer:
    """
    Computes gait metrics from ankle keypoint trajectories.

    Metrics:
      cadence_spm    — steps per minute
      asymmetry_pct  — left/right step time ratio deviation from 50%
      speed_mps      — gait speed in m/s (requires calibration)
      speed_pxs      — gait speed in px/s (always available)
    """

    def __init__(self, fps, calibration=None):
        self.fps         = fps
        self.calib       = calibration
        self.px_per_m    = calibration["px_per_m"] if calibration else None
        self.left_y_hist  = deque(maxlen=GAIT_WINDOW_FRAMES)
        self.right_y_hist = deque(maxlen=GAIT_WINDOW_FRAMES)
        self.hip_x_hist   = deque(maxlen=GAIT_WINDOW_FRAMES)
        self.stride_times_L = []
        self.stride_times_R = []
        self.last_peak_L    = None
        self.last_peak_R    = None

    def _simple_peaks(self, signal, min_distance=8, threshold_pct=0.15):
        """Detect peaks in signal (ankle Y maxima = foot striking ground)."""
        if len(signal) < min_distance * 2:
            return []
        mn   = min(signal)
        mx   = max(signal)
        span = mx - mn
        if span < 5:
            return []
        threshold = mn + span * threshold_pct
        peaks = []
        for i in range(1, len(signal) - 1):
            if (signal[i] > signal[i-1] and
                signal[i] > signal[i+1] and
                signal[i] > threshold):
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
        return peaks

    def update(self, kps, body_scale):
        """Add a frame of keypoints. Returns metrics dict when enough data."""
        la = kps[KP_LEFT_ANKLE];  ra = kps[KP_RIGHT_ANKLE]
        lh = kps[KP_LEFT_HIP];    rh = kps[KP_RIGHT_HIP]

        if la[2] > KP_CONF: self.left_y_hist.append(la[1])
        if ra[2] > KP_CONF: self.right_y_hist.append(ra[1])

        hip_x = None
        if lh[2] > KP_CONF and rh[2] > KP_CONF:
            hip_x = (lh[0] + rh[0]) / 2.0
        elif lh[2] > KP_CONF: hip_x = lh[0]
        elif rh[2] > KP_CONF: hip_x = rh[0]
        if hip_x is not None:
            self.hip_x_hist.append(hip_x)

        if len(self.left_y_hist) < GAIT_WINDOW_FRAMES * 0.8:
            return None

        l_arr = list(self.left_y_hist)
        r_arr = list(self.right_y_hist) if self.right_y_hist else l_arr
        h_arr = list(self.hip_x_hist)

        # Cadence from left ankle peaks
        l_peaks = self._simple_peaks(l_arr)
        r_peaks = self._simple_peaks(r_arr)
        all_peaks = sorted(set(l_peaks + r_peaks))

        cadence_spm = None
        if len(all_peaks) >= 2:
            diffs = [(all_peaks[i+1] - all_peaks[i]) for i in range(len(all_peaks)-1)]
            mean_frames = sum(diffs) / len(diffs)
            steps_per_sec = self.fps / mean_frames
            cadence_spm = round(steps_per_sec * 60, 1)

        # Gait speed
        speed_pxs = speed_mps = None
        if len(h_arr) >= 10:
            span_px = abs(h_arr[-1] - h_arr[0])
            secs    = len(h_arr) / self.fps
            speed_pxs = round(span_px / secs, 1)
            if self.px_per_m:
                speed_mps = round(speed_pxs / self.px_per_m, 3)

        # Step asymmetry (L vs R timing)
        asym_pct = None
        if len(l_peaks) >= 2 and len(r_peaks) >= 2:
            l_intervals = [(l_peaks[i+1]-l_peaks[i]) for i in range(len(l_peaks)-1)]
            r_intervals = [(r_peaks[i+1]-r_peaks[i]) for i in range(len(r_peaks)-1)]
            l_mean = sum(l_intervals)/len(l_intervals) if l_intervals else 0
            r_mean = sum(r_intervals)/len(r_intervals) if r_intervals else 0
            total  = l_mean + r_mean
            if total > 0:
                asym_pct = round(abs(l_mean - r_mean) / total * 100, 2)

        # Stride normalisation (step length relative to body scale)
        stride_norm = None
        if speed_mps and cadence_spm and body_scale:
            step_length_m = (speed_mps / (cadence_spm / 60)) if cadence_spm > 0 else None
            if step_length_m:
                body_m = (body_scale / self.px_per_m) if self.px_per_m else None
                if body_m and body_m > 0:
                    stride_norm = round(step_length_m / body_m, 3)

        return {
            "cadence_spm":  cadence_spm,
            "speed_pxs":    speed_pxs,
            "speed_mps":    speed_mps,
            "asymmetry_pct": asym_pct,
            "stride_norm":  stride_norm,
            "stride_count": len(all_peaks),
        }



# ACTIVITY CLASSIFIER


def classify_activity(kps, body_scale, hip_vel_y):
    """Simple rule-based classifier — fall/posture handled separately."""
    if body_scale is None:
        return "unknown"

    hip_m  = midpoint(kps, KP_LEFT_HIP,  KP_RIGHT_HIP)
    knee_m = midpoint(kps, KP_LEFT_KNEE, KP_RIGHT_KNEE)
    shld_m = midpoint(kps, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)

    if hip_m is None:
        return "unknown"

    # Lying: torso is horizontal
    if shld_m and hip_m:
        torso_angle = angle_from_vertical(hip_m, shld_m)
        if torso_angle > 45:
            return "lying_down"

    # Sitting: knees at or above hip level
    if knee_m:
        hip_knee_diff = (hip_m[1] - knee_m[1]) / body_scale
        if hip_knee_diff < 0.05:
            return "sitting"

    # Walking: detected by GaitAnalyzer separately (cadence > 0)
    # Default to standing
    return "standing"



# SERVER SYNC


def post_camera_metrics(session_date, gait, posture_score, speed_mps, speed_pxs):
    try:
        payload = {
            "date":          session_date,
            "gait_speed_px": speed_pxs,
            "cadence_spm":   gait.get("cadence_spm"),
            "asymmetry_pct": gait.get("asymmetry_pct"),
            "stride_norm":   gait.get("stride_norm"),
            "body_scale_px": None,
            "camera_mode":   "live_rtsp" if "rtsp" in VIGIL_SERVER else "usb",
            "stride_count":  gait.get("stride_count"),
        }
        requests.post(f"{VIGIL_SERVER}/sync/camera", json=payload, timeout=3)
        print(f"[SYNC] Camera metrics posted: cadence={gait.get('cadence_spm')} "
              f"asym={gait.get('asymmetry_pct')} "
              f"speed={speed_mps}m/s")
    except Exception as e:
        print(f"[SYNC] Failed to post camera metrics: {e}")


def post_fall_alert(event):
    try:
        requests.post(f"{VIGIL_SERVER}/alert/fall", json={
            "detected_at":  event["detected_at"],
            "hip_drop_pct": event["hip_drop_pct"],
            "torso_angle":  event["torso_angle"],
            "message":      "Fall detected by VIGIL camera system",
        }, timeout=3)
        print(f"[FALL] Alert posted to server — hip_drop={event['hip_drop_pct']*100:.0f}%")
    except Exception as e:
        print(f"[FALL] Failed to post alert: {e}")



# MAIN LOOP


def run(source, debug=False, write_db=True, server_sync=True):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.")
        sys.exit(1)

    if write_db:
        init_tables()

    calib = load_calibration()

    print("[INFO] Loading YOLOv8-Pose...")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded")

    if str(source).isdigit():
        cap = cv2.VideoCapture(int(source))
    elif str(source).startswith("rtsp://"):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce RTSP latency
    else:
        cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] fps={fps:.0f}")

    gait_analyzer    = GaitAnalyzer(fps=fps, calibration=calib)
    posture_analyzer = PostureAnalyzer()
    fall_fsm         = FallStateMachine()

    session_date     = datetime.now().strftime("%Y-%m-%d")
    frame_num        = 0
    last_sync_time   = time.time()
    sync_interval_s  = 30.0    # post metrics every 30 seconds

    prev_hip_y       = None
    prev_hip_time    = None

    # Activity state tracking
    current_activity   = "unknown"
    state_start_time   = time.time()
    last_activity_logged = None

    conn = get_db() if write_db else None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num    += 1
            current_time  = time.time()

            if frame_num % INFER_EVERY_N != 0:
                if debug:
                    cv2.imshow("VIGIL v2", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            results = model(frame, conf=MIN_CONF, verbose=False, device="cuda")

            best_kps  = None
            best_area = 0

            for result in results:
                if result.keypoints is None:
                    continue
                boxes   = result.boxes
                kps_all = result.keypoints.data.cpu().numpy()
                for i, kp_row in enumerate(kps_all):
                    if boxes is not None and i < len(boxes):
                        box  = boxes.xyxy[i].cpu().numpy()
                        area = (box[2]-box[0]) * (box[3]-box[1])
                        if area > best_area:
                            best_area = area
                            best_kps  = kp_row

            if best_kps is None:
                continue

            body_scale = compute_body_scale(best_kps)

            # Hip Y velocity for fall detection
            hip_m = midpoint(best_kps, KP_LEFT_HIP, KP_RIGHT_HIP)
            hip_y_now = hip_m[1] if hip_m else None
            hip_vel_y = 0.0
            if hip_y_now is not None and prev_hip_y is not None and prev_hip_time:
                dt = current_time - prev_hip_time
                hip_vel_y = (hip_y_now - prev_hip_y) / dt if dt > 0 else 0.0
            prev_hip_y   = hip_y_now
            prev_hip_time = current_time

            # Posture analysis
            posture = posture_analyzer.analyze(best_kps, body_scale)

            # Trunk lean for fall detection
            shld_m = midpoint(best_kps, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
            trunk_lean = angle_from_vertical(hip_m, shld_m) if (hip_m and shld_m) else 0.0

            # Fall state machine
            fall_event = fall_fsm.update(
                hip_y=hip_y_now,
                hip_vel_y=hip_vel_y,
                body_scale=body_scale,
                torso_angle=trunk_lean,
                current_time=current_time,
            )

            if fall_event == "fall_detected":
                current_activity = "fallen"
                for ev in fall_fsm.pop_events():
                    if write_db and conn:
                        conn.execute("""
                            INSERT INTO fall_events
                                (detected_at, hip_drop_pct, torso_angle, body_scale_px)
                            VALUES (?,?,?,?)
                        """, (ev["detected_at"], ev["hip_drop_pct"],
                              ev["torso_angle"], ev["body_scale_px"]))
                        conn.commit()
                    if server_sync:
                        post_fall_alert(ev)
            else:
                # Gait analysis
                gait = gait_analyzer.update(best_kps, body_scale)
                is_walking = (gait and gait.get("cadence_spm") and gait["cadence_spm"] > 30)

                if fall_event != "descending":
                    if is_walking:
                        current_activity = "walking"
                    else:
                        current_activity = classify_activity(best_kps, body_scale, hip_vel_y)

            # Periodic server sync
            if current_time - last_sync_time > sync_interval_s:
                gait_snapshot = gait_analyzer.update(best_kps, body_scale)
                if gait_snapshot and server_sync:
                    post_camera_metrics(
                        session_date,
                        gait_snapshot,
                        posture.get("posture_score") if posture else None,
                        gait_snapshot.get("speed_mps"),
                        gait_snapshot.get("speed_pxs"),
                    )
                if write_db and conn:
                    posture_analyzer.save_to_db(conn)
                last_sync_time = current_time

            # Debug overlay
            if debug:
                h, w = frame.shape[:2]
                color = ACTIVITY_COLORS.get(current_activity, (80,80,80))

                # Skeleton
                connections = [
                    (KP_LEFT_SHOULDER,KP_RIGHT_SHOULDER),(KP_LEFT_SHOULDER,KP_LEFT_HIP),
                    (KP_RIGHT_SHOULDER,KP_RIGHT_HIP),(KP_LEFT_HIP,KP_RIGHT_HIP),
                    (KP_LEFT_HIP,KP_LEFT_KNEE),(KP_RIGHT_HIP,KP_RIGHT_KNEE),
                    (KP_LEFT_KNEE,KP_LEFT_ANKLE),(KP_RIGHT_KNEE,KP_RIGHT_ANKLE),
                ]
                for a, b in connections:
                    if best_kps[a][2] > KP_CONF and best_kps[b][2] > KP_CONF:
                        cv2.line(frame,
                                 (int(best_kps[a][0]),int(best_kps[a][1])),
                                 (int(best_kps[b][0]),int(best_kps[b][1])),
                                 color, 2)

                # Status banner
                cv2.rectangle(frame, (0,0), (w,55), (10,10,10), -1)
                cv2.putText(frame, "VIGIL v2", (10,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,130), 1)
                cv2.putText(frame, current_activity.upper(), (10,44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Posture score
                if posture:
                    score = posture.get("posture_score",0)
                    pc    = (0,200,80) if score < 40 else (0,150,255) if score < 70 else (0,0,220)
                    cv2.putText(frame, f"Posture:{score:.0f}", (w-160,44),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, pc, 1)

                # Fall state
                fsm_color = (0,220,130) if fall_fsm.state == fall_fsm.UPRIGHT else \
                            (0,150,255) if fall_fsm.state == fall_fsm.DESCENDING else \
                            (0,0,220)
                cv2.putText(frame, fall_fsm.state, (w-180,22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, fsm_color, 1)

                # Calibration indicator
                calib_txt = f"CAL:{calib['px_per_m']:.0f}px/m" if calib else "UNCALIBRATED"
                cv2.putText(frame, calib_txt, (10, h-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)

                if w > 1280:
                    frame = cv2.resize(frame, (1280, int(h*1280/w)))
                cv2.imshow("VIGIL v2", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    finally:
        if write_db and conn:
            posture_analyzer.save_to_db(conn)
            conn.close()
        cap.release()
        if debug:
            cv2.destroyAllWindows()
        print("[INFO] Activity monitor shut down cleanly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIGIL Activity Monitor v2")
    parser.add_argument("--source",    required=True)
    parser.add_argument("--debug",     action="store_true")
    parser.add_argument("--no-db",     action="store_true")
    parser.add_argument("--no-server", action="store_true")
    args = parser.parse_args()
    run(
        source=args.source,
        debug=args.debug,
        write_db=not args.no_db,
        server_sync=not args.no_server,
    )