"""
VIGIL — Passive Health Monitor
sync/server.py — v6

Changes from v5:
- PATCH 1: Removed SendGrid crash-on-import; replaced with safe lazy stub
- PATCH 2: Added /ml/scores/latest endpoint
- PATCH 3: Migration extended with new camera metric columns
- PATCH 4: _rebuild_daily_summary wired for new camera columns
- Removed duplicate CREATE TABLE for posture_metrics
- Cleaned up imports (removed bare `import os` duplicate)
- weekly_summary fixed: removed reference to non-existent anomaly_detail column
"""

import json
import math
import os
import shutil
import tempfile
from datetime import datetime

from flask import Flask, jsonify, request, send_file
import sqlite3

# ── At top of file, after existing imports ──────────────────────────
import sys, os
VIGIL_ML_DIR = os.path.expanduser("~/passive-health-moniter/vigil_ml")
sys.path.insert(0, VIGIL_ML_DIR)
from server_ml_patch import register_ml_routes

# ── After app = Flask(__name__) ──────────────────────────────────────
register_ml_routes(app, DB_PATH, VIGIL_ML_DIR)

# ── In your /sync/healthkit handler, after _rebuild_daily_summary() ─
# Add this non-blocking background scoring trigger:
import threading

def _bg_score():
    try:
        import vigil_inference as inf
        conn = sqlite3.connect(DB_PATH)
        n = conn.execute("SELECT COUNT(*) FROM daily_summary").fetchone()[0]
        conn.close()
        if n >= 7:
            result = inf.score_from_db(DB_PATH)
            inf.save_scores_to_db(DB_PATH, result)
            print(f"[ML] Scored: health_index={result['health_index']}")
    except Exception as e:
        print(f"[ML] Scoring error: {e}")

threading.Thread(target=_bg_score, daemon=True).start()

app     = Flask(__name__)
DB_PATH = os.path.expanduser("~/passive-health-moniter/vigil.db")

CUMULATIVE_METRICS = {
    "StepCount", "ActiveEnergyBurned", "FlightsClimbed",
    "DistanceWalkingRunning", "SleepHours",
}

# ── Database ──────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS healthkit_metrics (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_type TEXT    NOT NULL,
            value       REAL    NOT NULL,
            unit        TEXT,
            start_date  TEXT    NOT NULL,
            end_date    TEXT,
            source      TEXT,
            received_at TEXT    DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS camera_metrics (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date        TEXT    NOT NULL,
            gait_speed_px       REAL,
            cadence_spm         REAL,
            asymmetry_pct       REAL,
            stride_norm         REAL,
            step_regularity     REAL,
            body_scale_px       REAL,
            camera_mode         TEXT,
            stride_count        INTEGER,
            stride_variability  REAL,
            arm_swing_asymmetry REAL,
            step_width          REAL,
            cadence_variability REAL,
            recorded_at         TEXT    DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS posture_metrics (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date      TEXT    NOT NULL,
            head_forward_norm REAL,
            shoulder_sym      REAL,
            body_lean_deg     REAL,
            neck_angle_deg    REAL,
            posture_score     REAL,
            posture_flag      INTEGER DEFAULT 0,
            recorded_at       TEXT    DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS activity_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            activity   TEXT    NOT NULL,
            started_at TEXT    NOT NULL,
            ended_at   TEXT,
            duration_s REAL,
            notes      TEXT
        );
        CREATE TABLE IF NOT EXISTS stillness_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at TEXT    NOT NULL,
            duration_s  REAL,
            hour_of_day INTEGER,
            alerted     INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS fall_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at   TEXT    NOT NULL,
            hip_drop_pct  REAL,
            torso_angle   REAL,
            body_scale_px REAL,
            confirmed     INTEGER DEFAULT 1,
            alerted       INTEGER DEFAULT 0,
            notes         TEXT
        );
        CREATE TABLE IF NOT EXISTS alerts (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT    NOT NULL,
            detected_at TEXT   NOT NULL,
            payload    TEXT,
            seen       INTEGER DEFAULT 0,
            created_at TEXT    DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS daily_summary (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            date                    TEXT    UNIQUE NOT NULL,
            step_count              REAL,
            walking_speed_ms        REAL,
            walking_asymmetry_pct   REAL,
            walking_step_length_m   REAL,
            double_support_pct      REAL,
            hrv_sdnn                REAL,
            resting_hr              REAL,
            sleep_hours             REAL,
            active_calories         REAL,
            spo2_avg                REAL,
            respiratory_rate        REAL,
            vo2_max                 REAL,
            wrist_temp              REAL,
            camera_cadence_spm      REAL,
            camera_asymmetry_pct    REAL,
            camera_gait_speed       REAL,
            stride_variability      REAL,
            arm_swing_asymmetry     REAL,
            step_width              REAL,
            cadence_variability     REAL,
            posture_score           REAL,
            anomaly_score           REAL,
            anomaly_flag            INTEGER DEFAULT 0,
            updated_at              TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_hk_type_date
            ON healthkit_metrics(metric_type, start_date);
        CREATE TABLE IF NOT EXISTS ml_scores (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            scored_at         TEXT    NOT NULL,
            condition_id      TEXT,
            condition_label   TEXT,
            probability       REAL,
            score_0_100       REAL,
            ref_prob          REAL,
            iso_prob          REAL,
            data_quality      TEXT,
            days_of_data      INTEGER,
            published_auc     REAL,
            insufficient_data INTEGER DEFAULT 0
        );
    """)

    # Non-destructive migrations for any pre-existing DB
    existing_ds = {row[1] for row in conn.execute("PRAGMA table_info(daily_summary)")}
    for col, typ in [
        ("double_support_pct",   "REAL"),
        ("sleep_hours",          "REAL"),
        ("respiratory_rate",     "REAL"),
        ("vo2_max",              "REAL"),
        ("wrist_temp",           "REAL"),
        ("stride_variability",   "REAL"),
        ("arm_swing_asymmetry",  "REAL"),
        ("step_width",           "REAL"),
        ("cadence_variability",  "REAL"),
        ("posture_score",        "REAL"),
    ]:
        if col not in existing_ds:
            conn.execute(f"ALTER TABLE daily_summary ADD COLUMN {col} {typ}")
            print(f"Migration: added daily_summary.{col}")

    existing_cam = {row[1] for row in conn.execute("PRAGMA table_info(camera_metrics)")}
    for col, typ in [
        ("stride_variability",   "REAL"),
        ("arm_swing_asymmetry",  "REAL"),
        ("step_width",           "REAL"),
        ("cadence_variability",  "REAL"),
    ]:
        if col not in existing_cam:
            conn.execute(f"ALTER TABLE camera_metrics ADD COLUMN {col} {typ}")
            print(f"Migration: added camera_metrics.{col}")

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

# ── Helpers ───────────────────────────────────────────────────────────────

def fix_units(metric_type, value):
    """Normalise known unit quirks from Apple Health exports."""
    if metric_type == "HeartRateVariabilitySDNN" and value < 1.0:
        return value * 1000.0          # seconds → ms
    if metric_type == "OxygenSaturation" and value <= 1.0:
        return value * 100.0           # 0-1 ratio → %
    return value


def date_only(iso_str):
    return iso_str[:10] if iso_str else datetime.now().strftime("%Y-%m-%d")


def _rebuild_daily_summary(conn, date_str):
    """Aggregate all sources into a single daily_summary row for date_str."""

    def avg(mt):
        r = conn.execute(
            "SELECT AVG(value) FROM healthkit_metrics "
            "WHERE metric_type=? AND substr(start_date,1,10)=?",
            (mt, date_str)).fetchone()
        return r[0] if r and r[0] is not None else None

    def latest(mt):
        """Today avg; falls back to 3-day rolling avg for once-a-day metrics."""
        r = conn.execute(
            "SELECT AVG(value) FROM healthkit_metrics "
            "WHERE metric_type=? AND substr(start_date,1,10)=?",
            (mt, date_str)).fetchone()
        if r and r[0] is not None:
            return r[0]
        r = conn.execute(
            "SELECT AVG(value) FROM healthkit_metrics "
            "WHERE metric_type=? AND start_date >= date(?,'-3 days')",
            (mt, date_str)).fetchone()
        return r[0] if r and r[0] is not None else None

    def day_max(mt):
        r = conn.execute(
            "SELECT MAX(value) FROM healthkit_metrics "
            "WHERE metric_type=? AND substr(start_date,1,10)=?",
            (mt, date_str)).fetchone()
        return r[0] if r and r[0] is not None else None

    cam = conn.execute("""
        SELECT AVG(cadence_spm), AVG(asymmetry_pct), AVG(gait_speed_px),
               AVG(stride_variability), AVG(arm_swing_asymmetry),
               AVG(step_width), AVG(cadence_variability)
        FROM camera_metrics WHERE session_date=?
    """, (date_str,)).fetchone()

    posture = conn.execute(
        "SELECT AVG(posture_score) FROM posture_metrics "
        "WHERE substr(recorded_at,1,10)=?",
        (date_str,)).fetchone()
    posture_score = posture[0] if posture and posture[0] is not None else None

    conn.execute("""
        INSERT INTO daily_summary (
            date,
            step_count, hrv_sdnn, resting_hr, sleep_hours, active_calories,
            spo2_avg, respiratory_rate, vo2_max, wrist_temp,
            walking_speed_ms, walking_asymmetry_pct, walking_step_length_m,
            double_support_pct,
            camera_cadence_spm, camera_asymmetry_pct, camera_gait_speed,
            stride_variability, arm_swing_asymmetry, step_width, cadence_variability,
            posture_score
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(date) DO UPDATE SET
            step_count=excluded.step_count,
            hrv_sdnn=excluded.hrv_sdnn,
            resting_hr=excluded.resting_hr,
            sleep_hours=excluded.sleep_hours,
            active_calories=excluded.active_calories,
            spo2_avg=excluded.spo2_avg,
            respiratory_rate=excluded.respiratory_rate,
            vo2_max=excluded.vo2_max,
            wrist_temp=excluded.wrist_temp,
            walking_speed_ms=excluded.walking_speed_ms,
            walking_asymmetry_pct=excluded.walking_asymmetry_pct,
            walking_step_length_m=excluded.walking_step_length_m,
            double_support_pct=excluded.double_support_pct,
            camera_cadence_spm=excluded.camera_cadence_spm,
            camera_asymmetry_pct=excluded.camera_asymmetry_pct,
            camera_gait_speed=excluded.camera_gait_speed,
            stride_variability=excluded.stride_variability,
            arm_swing_asymmetry=excluded.arm_swing_asymmetry,
            step_width=excluded.step_width,
            cadence_variability=excluded.cadence_variability,
            posture_score=excluded.posture_score,
            updated_at=datetime('now')
    """, (
        date_str,
        day_max("StepCount"),
        latest("HeartRateVariabilitySDNN"),
        latest("RestingHeartRate"),
        day_max("SleepHours"),
        day_max("ActiveEnergyBurned"),
        avg("OxygenSaturation"),
        avg("RespiratoryRate"),
        latest("Vo2Max"),
        avg("BodyTemperature"),
        avg("WalkingSpeed"),
        avg("WalkingAsymmetryPercentage"),
        avg("WalkingStepLength"),
        avg("WalkingDoubleSupportPercentage"),
        cam[0] if cam else None,
        cam[1] if cam else None,
        cam[2] if cam else None,
        cam[3] if cam else None,
        cam[4] if cam else None,
        cam[5] if cam else None,
        cam[6] if cam else None,
        posture_score,
    ))


def _check_metric_anomalies(conn, date_str):
    """
    Compare today's metrics to the 14-day rolling mean ± std dev.
    Writes a health_signal alert when a metric is >1.5 std devs from baseline.
    Fires at most once per metric per day.
    """
    # (column, display_label, unit, higher_is_worse)
    METRIC_COLS = {
        "hrv_sdnn":              ("HRV",            "ms",  False),
        "resting_hr":            ("Resting HR",      "bpm", True),
        "spo2_avg":              ("SpO₂",           "%",   False),
        "walking_asymmetry_pct": ("Gait Asymmetry",  "%",   True),
        "walking_speed_ms":      ("Walking Speed",   "m/s", False),
        "sleep_hours":           ("Sleep",           "hrs", False),
    }
    THRESHOLD = 1.5

    today_row = conn.execute(
        "SELECT * FROM daily_summary WHERE date=?", (date_str,)).fetchone()
    if not today_row:
        return

    hist = conn.execute("""
        SELECT * FROM daily_summary
        WHERE date < ? ORDER BY date DESC LIMIT 14
    """, (date_str,)).fetchall()

    if len(hist) < 3:
        return  # insufficient baseline

    for col, (label, unit, higher_is_worse) in METRIC_COLS.items():
        today_val = today_row[col]
        if today_val is None:
            continue

        hist_vals = [row[col] for row in hist if row[col] is not None]
        if len(hist_vals) < 3:
            continue

        mean = sum(hist_vals) / len(hist_vals)
        std  = math.sqrt(sum((v - mean) ** 2 for v in hist_vals) / len(hist_vals))
        if std < 0.001:
            continue

        z = (today_val - mean) / std
        is_anomaly = (higher_is_worse and z > THRESHOLD) or (not higher_is_worse and z < -THRESHOLD)
        if not is_anomaly:
            continue

        # Deduplicate: one alert per metric per day
        existing = conn.execute("""
            SELECT id FROM alerts
            WHERE alert_type='health_signal'
              AND date(detected_at)=?
              AND json_extract(payload,'$.metric')=?
        """, (date_str, label)).fetchone()
        if existing:
            continue

        direction = "elevated" if z > 0 else "low"
        conn.execute(
            "INSERT INTO alerts (alert_type, detected_at, payload) VALUES (?,?,?)",
            (
                "health_signal",
                datetime.now().isoformat(),
                json.dumps({
                    "metric":    label,
                    "value":     round(today_val, 2),
                    "unit":      unit,
                    "baseline":  round(mean, 2),
                    "z_score":   round(z, 2),
                    "direction": direction,
                    "message":   (
                        f"{label} is {direction} vs your 14-day baseline "
                        f"({today_val:.1f} vs avg {mean:.1f} {unit})"
                    ),
                }),
            )
        )
        print(f"Health signal alert: {label} {direction} (z={z:.2f})")


# ── SendGrid (Phase 2 — lazy, never runs on import) ───────────────────────

SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY")


def send_weekly_report_email(to_email: str, subject: str, html_body: str) -> bool:
    """Send weekly health report via SendGrid. Returns True if sent."""
    if not SENDGRID_API_KEY:
        print("[INFO] SENDGRID_API_KEY not set — skipping email")
        return False
    try:
        import sendgrid as sg_lib
        from sendgrid.helpers.mail import Mail, Email, To, Content
        sg   = sg_lib.SendGridAPIClient(api_key=SENDGRID_API_KEY)
        mail = Mail(
            from_email=Email("noreply@vigil.health"),
            to_emails=To(to_email),
            subject=subject,
            html_content=Content("text/html", html_body),
        )
        response = sg.client.mail.send.post(request_body=mail.get())
        print(f"Email sent to {to_email}: status {response.status_code}")
        return response.status_code == 202
    except Exception as e:
        print(f"SendGrid error: {e}")
        return False


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/health")
def health_check():
    return jsonify({"status": "ok", "db": DB_PATH})


@app.route("/status")
def full_status():
    """Machine-readable health check with table row counts."""
    conn   = get_db()
    counts = {}
    for table in ["healthkit_metrics", "camera_metrics", "alerts",
                  "daily_summary", "posture_metrics", "fall_events"]:
        try:
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception:
            counts[table] = 0
    latest = conn.execute("SELECT MAX(start_date) FROM healthkit_metrics").fetchone()[0]
    conn.close()
    return jsonify({
        "status":        "ok",
        "db":            DB_PATH,
        "table_counts":  counts,
        "latest_sample": latest,
        "server_time":   datetime.now().isoformat(),
    })


# ── Sync ──────────────────────────────────────────────────────────────────

@app.route("/sync/healthkit", methods=["POST"])
def sync_healthkit():
    data = request.get_json()
    if not data or "samples" not in data:
        return jsonify({"error": "no samples"}), 400

    conn     = get_db()
    inserted = skipped = 0

    for s in data["samples"]:
        try:
            mt  = s.get("type")
            rv  = s.get("value")
            sd  = s.get("startDate", "")
            if mt is None or rv is None:
                skipped += 1
                continue
            val = fix_units(mt, float(rv))

            if mt in CUMULATIVE_METRICS:
                day = date_only(sd)
                ex  = conn.execute(
                    "SELECT id, value FROM healthkit_metrics "
                    "WHERE metric_type=? AND substr(start_date,1,10)=?",
                    (mt, day)).fetchone()
                if ex is None:
                    conn.execute(
                        "INSERT INTO healthkit_metrics "
                        "(metric_type,value,unit,start_date,end_date,source) "
                        "VALUES (?,?,?,?,?,?)",
                        (mt, val, s.get("unit"), sd,
                         s.get("endDate"), s.get("source", "AppleWatch")))
                    inserted += 1
                elif val > ex["value"]:
                    conn.execute(
                        "UPDATE healthkit_metrics SET value=?, start_date=? WHERE id=?",
                        (val, sd, ex["id"]))
                    inserted += 1
                else:
                    skipped += 1
            else:
                ex = conn.execute(
                    "SELECT id FROM healthkit_metrics "
                    "WHERE metric_type=? AND start_date=?",
                    (mt, sd)).fetchone()
                if ex is None:
                    conn.execute(
                        "INSERT INTO healthkit_metrics "
                        "(metric_type,value,unit,start_date,end_date,source) "
                        "VALUES (?,?,?,?,?,?)",
                        (mt, val, s.get("unit"), sd,
                         s.get("endDate"), s.get("source", "AppleWatch")))
                    inserted += 1
                else:
                    skipped += 1
        except Exception as e:
            print(f"Insert error: {e}")
            skipped += 1

    conn.commit()

    affected  = {date_only(s.get("startDate", "")) for s in data["samples"]}
    today_str = datetime.now().strftime("%Y-%m-%d")

    for d in affected:
        try:
            _rebuild_daily_summary(conn, d)
        except Exception as e:
            print(f"Summary error {d}: {e}")

    try:
        _check_metric_anomalies(conn, today_str)
    except Exception as e:
        print(f"Anomaly check error: {e}")

    conn.commit()
    conn.close()
    print(f"Synced: inserted={inserted} skipped={skipped}")
    return jsonify({"inserted": inserted, "skipped": skipped})


@app.route("/sync/camera", methods=["POST"])
def sync_camera():
    data = request.get_json()
    if not data:
        return jsonify({"error": "no data"}), 400
    conn = get_db()
    conn.execute("""
        INSERT INTO camera_metrics (
            session_date, gait_speed_px, cadence_spm, asymmetry_pct,
            stride_norm, step_regularity, body_scale_px, camera_mode, stride_count,
            stride_variability, arm_swing_asymmetry, step_width, cadence_variability
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data.get("date", datetime.now().strftime("%Y-%m-%d")),
        data.get("gait_speed_px"),
        data.get("cadence_spm"),
        data.get("asymmetry_pct"),
        data.get("stride_norm"),
        data.get("step_regularity"),
        data.get("body_scale_px"),
        data.get("camera_mode"),
        data.get("stride_count"),
        data.get("stride_variability"),
        data.get("arm_swing_asymmetry"),
        data.get("step_width"),
        data.get("cadence_variability"),
    ))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})


# ── Summaries ─────────────────────────────────────────────────────────────

@app.route("/summary/today")
def today_summary():
    """Returns only today's summary row (midnight-bounded)."""
    today = datetime.now().strftime("%Y-%m-%d")
    conn  = get_db()
    row   = conn.execute(
        "SELECT * FROM daily_summary WHERE date=?", (today,)).fetchone()
    conn.close()
    return jsonify(dict(row)) if row else jsonify({"date": today, "status": "no data yet"})


@app.route("/summary/recent")
def recent_summary():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM daily_summary ORDER BY date DESC LIMIT 30").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/summary/weekly")
def weekly_summary():
    """Returns 7-day summary with anomaly breakdown for the Trends screen."""
    conn = get_db()
    rows = conn.execute("""
        SELECT date, step_count, hrv_sdnn, resting_hr, sleep_hours,
               active_calories, spo2_avg, respiratory_rate,
               walking_speed_ms, walking_asymmetry_pct, double_support_pct,
               anomaly_flag, anomaly_score
        FROM daily_summary
        ORDER BY date DESC LIMIT 7
    """).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


# ── Alerts ────────────────────────────────────────────────────────────────

@app.route("/alert/health_signal", methods=["POST"])
def alert_health_signal():
    data = request.get_json()
    if not data:
        return jsonify({"error": "no data"}), 400
    conn = get_db()
    conn.execute(
        "INSERT INTO alerts (alert_type, detected_at, payload) VALUES (?,?,?)",
        (
            "health_signal",
            data.get("detected_at", datetime.now().isoformat()),
            json.dumps({
                "metric":  data.get("metric"),
                "message": data.get("message"),
                "value":   data.get("value"),
                "mean":    data.get("mean"),
                "z_score": data.get("z_score"),
            }),
        ))
    conn.commit()
    conn.close()
    print(f"HEALTH SIGNAL: {data.get('message')}")
    return jsonify({"status": "ok"})


@app.route("/alert/fall", methods=["POST"])
def alert_fall():
    data = request.get_json()
    if not data:
        return jsonify({"error": "no data"}), 400
    conn = get_db()
    conn.execute(
        "INSERT INTO alerts (alert_type, detected_at, payload) VALUES (?,?,?)",
        (
            "fall",
            data.get("detected_at", datetime.now().isoformat()),
            json.dumps({
                "hip_drop_pct": data.get("hip_drop_pct"),
                "torso_angle":  data.get("torso_angle"),
                "message":      data.get("message", "Fall detected"),
            }),
        ))
    conn.commit()
    conn.close()
    print("FALL ALERT received")
    return jsonify({"status": "ok"})


@app.route("/alerts/ack", methods=["POST"])
def alerts_ack():
    data = request.get_json()
    if not data or "id" not in data:
        return jsonify({"error": "no id"}), 400
    conn = get_db()
    conn.execute("UPDATE alerts SET seen=1 WHERE id=?", (data["id"],))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})


@app.route("/alerts/recent")
def alerts_recent():
    mark_seen = request.args.get("mark_seen", "0") == "1"
    conn      = get_db()
    rows      = conn.execute(
        "SELECT * FROM alerts WHERE seen=0 ORDER BY created_at DESC LIMIT 20"
    ).fetchall()
    alerts = []
    for row in rows:
        a = dict(row)
        try:
            a["payload"] = json.loads(a["payload"])
        except Exception:
            pass
        alerts.append(a)
    if mark_seen and alerts:
        ids = [a["id"] for a in alerts]
        conn.execute(
            "UPDATE alerts SET seen=1 WHERE id IN ({})".format(",".join("?" * len(ids))),
            ids)
        conn.commit()
    conn.close()
    return jsonify(alerts)


@app.route("/alerts/all")
def alerts_all():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM alerts ORDER BY created_at DESC LIMIT 100").fetchall()
    conn.close()
    alerts = []
    for row in rows:
        a = dict(row)
        try:
            a["payload"] = json.loads(a["payload"])
        except Exception:
            pass
        alerts.append(a)
    return jsonify(alerts)


# ── Posture ───────────────────────────────────────────────────────────────

@app.route("/posture/recent")
def posture_recent():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM posture_metrics ORDER BY recorded_at DESC LIMIT 100"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


# ── ML Scores ─────────────────────────────────────────────────────────────

@app.route("/ml/scores/latest", methods=["GET"])
def ml_scores_latest():
    """Return most recent nightly ML classifier scores."""
    try:
        conn = get_db()
        tbl  = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ml_scores'"
        ).fetchone()
        if not tbl:
            conn.close()
            return jsonify({"scores": [], "scored_at": None, "status": "no ml data yet"})

        row = conn.execute(
            "SELECT scored_at FROM ml_scores ORDER BY scored_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            conn.close()
            return jsonify({"scores": [], "scored_at": None})

        scores = conn.execute("""
            SELECT condition_id, condition_label, probability, score_0_100,
                   ref_prob, iso_prob, data_quality, days_of_data,
                   published_auc, insufficient_data
            FROM ml_scores WHERE scored_at=?
            ORDER BY score_0_100 DESC
        """, (row["scored_at"],)).fetchall()
        conn.close()
        return jsonify({
            "scored_at": row["scored_at"],
            "scores":    [dict(s) for s in scores],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Debug / Admin ─────────────────────────────────────────────────────────

@app.route("/debug/dump")
def debug_dump():
    conn    = get_db()
    metrics = conn.execute("""
        SELECT metric_type,
               COUNT(*)          AS count,
               ROUND(AVG(value),2) AS avg,
               ROUND(MIN(value),2) AS min,
               ROUND(MAX(value),2) AS max,
               MAX(start_date)   AS latest
        FROM healthkit_metrics
        GROUP BY metric_type
        ORDER BY COUNT(*) DESC
    """).fetchall()
    sc = conn.execute("SELECT COUNT(*) FROM daily_summary").fetchone()[0]
    ts = conn.execute("SELECT COUNT(*) FROM healthkit_metrics").fetchone()[0]
    conn.close()
    return jsonify({
        "total_samples":   ts,
        "daily_summaries": sc,
        "by_metric":       [dict(r) for r in metrics],
    })


@app.route("/debug/clean", methods=["POST"])
def debug_clean():
    conn   = get_db()
    before = conn.execute("SELECT COUNT(*) FROM healthkit_metrics").fetchone()[0]

    # Keep only the highest cumulative value per metric per day
    for metric in CUMULATIVE_METRICS:
        conn.execute("""
            DELETE FROM healthkit_metrics WHERE metric_type=?
            AND id NOT IN (
                SELECT id FROM (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               PARTITION BY metric_type, substr(start_date,1,10)
                               ORDER BY value DESC
                           ) AS rn
                    FROM healthkit_metrics WHERE metric_type=?
                ) WHERE rn=1
            )
        """, (metric, metric))

    cl = "('" + "','".join(CUMULATIVE_METRICS) + "')"
    conn.execute(f"""
        DELETE FROM healthkit_metrics WHERE metric_type NOT IN {cl}
        AND id NOT IN (
            SELECT MIN(id) FROM healthkit_metrics WHERE metric_type NOT IN {cl}
            GROUP BY metric_type, start_date
        )
    """)
    conn.commit()

    after = conn.execute("SELECT COUNT(*) FROM healthkit_metrics").fetchone()[0]
    for (d,) in conn.execute(
            "SELECT DISTINCT substr(start_date,1,10) FROM healthkit_metrics").fetchall():
        _rebuild_daily_summary(conn, d)
    conn.commit()
    conn.close()
    return jsonify({"before": before, "after": after, "removed": before - after})


@app.route("/export/sqlite")
def export_sqlite():
    """Download a copy of the raw SQLite database."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    shutil.copy2(DB_PATH, tmp.name)
    return send_file(
        tmp.name,
        as_attachment=True,
        download_name=f"vigil_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db",
        mimetype="application/octet-stream",
    )


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print("VIGIL sync server starting on port 5001…")
    app.run(host="0.0.0.0", port=5001, debug=False)