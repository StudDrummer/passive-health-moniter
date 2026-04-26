"""
VIGIL ML Inference Engine
==========================
Loads trained models and runs inference on live patient data from SQLite.
Called by:
  1. ml_scorer.py (nightly cron on Jetson)
  2. server.py /ml/scores/latest endpoint
  3. Real-time path (called on every sync completion)

Architecture:
  SQLite daily_summary rows → vigil_features_v2.py → model.predict_proba → score

Output per condition:
  {
    "condition_id": "afib",
    "condition_label": "Atrial Fibrillation Risk",
    "category": "cardiovascular",
    "probability": 0.23,
    "score_0_100": 23.0,
    "level": "low",          # low / moderate / elevated / high
    "alert_threshold": 0.35,
    "urgent": false,
    "insufficient_data": false,
    "completeness": 0.86,
    "data_quality": "good",   # good / fair / limited
    "published_auc": 0.97,
    "top_signals": ["hrv_cv_14d: 0.82", ...],
    "scored_at": "2025-04-15T08:00:00"
  }
"""

import os, json, math, sqlite3
from datetime import datetime
from typing import Optional
import numpy as np
import joblib

# ── Feature extraction: prefer v2 (14 conditions), fall back to v1 ────────────
try:
    import vigil_features_v2 as vf
    _VF_VERSION = 2
except ImportError:
    import vigil_features as vf
    _VF_VERSION = 1


def _extract(condition_id: str, rows: list):
    """
    Compatibility wrapper around vf.extract().

    v1 returns: (feat_array, feat_names, completeness)
    v2 returns: just a numpy array (no names or completeness)

    Always returns: (feat_array, feat_names, completeness)
    """
    result = vf.extract(condition_id, rows)

    # v1: tuple of 3
    if isinstance(result, tuple) and len(result) == 3:
        feat, feat_names, completeness = result
        return np.array(feat).flatten(), feat_names, float(completeness)

    # v2: bare numpy array
    feat = np.array(result).flatten()
    # Estimate completeness from how many features are non-zero/non-nan
    valid = np.sum(~np.isnan(feat) & (feat != 0))
    completeness = float(valid) / len(feat) if len(feat) > 0 else 0.0
    return feat, [], completeness


# ─── Hardcoded metadata for all 14 conditions ─────────────────────────────────
# Used as fallback when no _meta.json file exists on disk.

_CONDITION_META = {
    'afib': {
        'condition_label':  'Atrial Fibrillation Risk',
        'category':         'cardiovascular',
        'alert_threshold':  0.40,
        'urgent_threshold': 0.65,
        'published_auc':    0.97,
    },
    'parkinsons': {
        'condition_label':  "Parkinson's / Movement Risk",
        'category':         'neurological',
        'alert_threshold':  0.40,
        'urgent_threshold': 0.65,
        'published_auc':    0.96,
    },
    'sleep_apnea': {
        'condition_label':  'Sleep Apnea Risk',
        'category':         'respiratory',
        'alert_threshold':  0.40,
        'urgent_threshold': 0.65,
        'published_auc':    0.94,
    },
    'heart_failure': {
        'condition_label':  'Cardiac Decompensation Risk',
        'category':         'cardiovascular',
        'alert_threshold':  0.40,
        'urgent_threshold': 0.65,
        'published_auc':    0.90,
    },
    'infection': {
        'condition_label':  'Acute Infection Signal',
        'category':         'immune',
        'alert_threshold':  0.40,
        'urgent_threshold': 0.65,
        'published_auc':    0.82,
    },
    'frailty': {
        'condition_label':  'Frailty / Low Fitness',
        'category':         'functional',
        'alert_threshold':  0.40,
        'urgent_threshold': 0.65,
        'published_auc':    0.88,
    },
    'stress': {
        'condition_label':  'Chronic Stress / Autonomic Overload',
        'category':         'mental_health',
        'alert_threshold':  0.40,
        'urgent_threshold': 0.65,
        'published_auc':    0.93,
    },
    'depression': {
        'condition_label':  'Depression / MDD Pattern',
        'category':         'mental_health',
        'alert_threshold':  0.40,
        'urgent_threshold': 0.65,
        'published_auc':    0.78,
    },
    'metabolic': {
        'condition_label':  'Metabolic / Insulin Resistance Risk',
        'category':         'metabolic',
        'alert_threshold':  0.40,
        'urgent_threshold': 0.65,
        'published_auc':    0.81,
    },
    'fall_risk': {
        'condition_label':  'Fall Risk',
        'category':         'functional',
        'alert_threshold':  0.40,
        'urgent_threshold': 0.65,
        'published_auc':    0.96,
    },
    'hypertension': {
        'condition_label':  'Hypertension Risk',
        'category':         'cardiovascular',
        'alert_threshold':  0.40,
        'urgent_threshold': 0.65,
        'published_auc':    0.88,
    },
    'copd': {
        'condition_label':  'COPD / Respiratory Disease',
        'category':         'respiratory',
        'alert_threshold':  0.40,
        'urgent_threshold': 0.65,
        'published_auc':    0.87,
    },
    'thyroid': {
        'condition_label':  'Thyroid Dysfunction',
        'category':         'endocrine',
        'alert_threshold':  0.40,
        'urgent_threshold': 0.65,
        'published_auc':    0.82,
    },
    'anemia': {
        'condition_label':  'Anemia / Low Perfusion',
        'category':         'hematologic',
        'alert_threshold':  0.40,
        'urgent_threshold': 0.65,
        'published_auc':    0.84,
    },
}


# ─── Model registry ───────────────────────────────────────────────────────────

_MODELS_CACHE: dict = {}   # condition_id → (model, meta)

def _models_dir():
    """Locate models dir relative to this script or via env var."""
    env = os.environ.get('VIGIL_MODELS_DIR')
    if env and os.path.isdir(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, 'models')


def _load_model(condition_id: str):
    """Load and cache a model + meta from disk. Falls back to hardcoded meta."""
    if condition_id in _MODELS_CACHE:
        return _MODELS_CACHE[condition_id]
    mdir = _models_dir()
    model_path = os.path.join(mdir, f"{condition_id}_model.joblib")
    meta_path  = os.path.join(mdir, f"{condition_id}_meta.json")
    if not os.path.exists(model_path):
        return None, _CONDITION_META.get(condition_id, {})
    model = joblib.load(model_path)
    # Use disk meta if present, otherwise fall back to hardcoded
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
    else:
        meta = _CONDITION_META.get(condition_id, {})
    _MODELS_CACHE[condition_id] = (model, meta)
    return model, meta


def _level(prob: float, alert_t: float, urgent_t: float) -> str:
    if prob >= urgent_t:      return 'high'
    if prob >= alert_t:       return 'elevated'
    if prob >= alert_t * 0.6: return 'moderate'
    return 'low'


def _quality(completeness: float, n_days: int) -> str:
    if completeness >= 0.7 and n_days >= 14: return 'good'
    if completeness >= 0.4 and n_days >= 7:  return 'fair'
    return 'limited'


# ─── Single-condition scorer ──────────────────────────────────────────────────

def score_condition(condition_id: str, rows: list) -> dict:
    """
    Score one condition from daily summary rows.
    rows: list of dicts, most recent first (from SQL ORDER BY date DESC).
    """
    model, meta = _load_model(condition_id)
    scored_at = datetime.now().isoformat()

    base = {
        "condition_id":    condition_id,
        "condition_label": meta.get('condition_label', condition_id),
        "category":        meta.get('category', 'unknown'),
        "scored_at":       scored_at,
        "published_auc":   meta.get('published_auc', 0.0),
        "alert_threshold": meta.get('alert_threshold', 0.40),
        "urgent_threshold":meta.get('urgent_threshold', 0.65),
    }

    # ── Feature extraction ───────────────────────────────────────────────
    try:
        feat, feat_names, completeness = _extract(condition_id, rows)
    except Exception as e:
        return {**base,
                "probability": 0.0, "score_0_100": 0.0, "level": "low",
                "urgent": False, "insufficient_data": True,
                "completeness": 0.0, "data_quality": "limited",
                "error": str(e), "top_signals": []}

    n_days = len(rows)
    quality = _quality(completeness, n_days)

    if completeness < 0.25 or model is None:
        return {**base,
                "probability": 0.0, "score_0_100": 0.0, "level": "low",
                "urgent": False, "insufficient_data": True,
                "completeness": float(completeness),
                "data_quality": quality, "top_signals": []}

    # ── Inference ────────────────────────────────────────────────────────
    try:
        prob = float(model.predict_proba(feat.reshape(1, -1))[0, 1])
    except Exception as e:
        return {**base,
                "probability": 0.0, "score_0_100": 0.0, "level": "low",
                "urgent": False, "insufficient_data": True,
                "completeness": float(completeness),
                "data_quality": quality,
                "error": f"Inference error: {e}", "top_signals": []}

    alert_t  = base["alert_threshold"]
    urgent_t = base["urgent_threshold"]
    level    = _level(prob, alert_t, urgent_t)

    # ── Top contributing signals (SHAP-lite: feature × magnitude) ────────
    top_signals = []
    if feat_names and len(feat) == len(feat_names):
        deviations = [(name, float(abs(feat[i])))
                      for i, name in enumerate(feat_names)
                      if name != 'completeness_flag']
        deviations.sort(key=lambda x: -x[1])
        top_signals = [f"{name}: {val:.2f}" for name, val in deviations[:4]]

    return {
        **base,
        "probability":       round(prob, 4),
        "score_0_100":       round(prob * 100, 1),
        "level":             level,
        "urgent":            prob >= urgent_t,
        "insufficient_data": False,
        "completeness":      round(float(completeness), 2),
        "data_quality":      quality,
        "top_signals":       top_signals,
    }


# ─── Full patient scorer ──────────────────────────────────────────────────────

CONDITIONS = [
    'afib', 'parkinsons', 'sleep_apnea', 'heart_failure',
    'infection', 'frailty', 'stress', 'depression',
    'metabolic', 'fall_risk', 'hypertension', 'copd',
    'thyroid', 'anemia',
]

def score_all(rows: list) -> dict:
    """
    Score all 14 conditions from daily summary rows.
    Returns dict ready to store in ml_scores table or return via API.
    """
    if not rows:
        return {"scored_at": datetime.now().isoformat(), "scores": [], "health_index": 100}

    scores = [score_condition(cid, rows) for cid in CONDITIONS]

    # Overall health index: 100 − weighted mean of valid scores
    valid = [s for s in scores if not s['insufficient_data']]
    if valid:
        weighted = sum(s['probability'] * s['completeness'] for s in valid)
        weights  = sum(s['completeness'] for s in valid)
        avg_prob = weighted / weights if weights > 0 else 0
        health_index = max(0, round(100 - avg_prob * 100))
    else:
        health_index = 100

    return {
        "scored_at":    datetime.now().isoformat(),
        "health_index": health_index,
        "scores":       scores,
    }


# ─── SQLite integration ───────────────────────────────────────────────────────

def score_from_db(db_path: str, n_days: int = 30) -> dict:
    """
    Load daily_summary rows from SQLite and run full scoring.
    This is what ml_scorer.py calls every night.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM daily_summary ORDER BY date DESC LIMIT ?",
        (n_days,)
    ).fetchall()
    conn.close()
    return score_all([dict(r) for r in rows])


def save_scores_to_db(db_path: str, result: dict):
    """
    Persist scored results to ml_scores table.
    Creates table if not present.
    """
    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ml_scores (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            scored_at         TEXT NOT NULL,
            condition_id      TEXT,
            condition_label   TEXT,
            category          TEXT,
            probability       REAL,
            score_0_100       REAL,
            level             TEXT,
            urgent            INTEGER,
            completeness      REAL,
            data_quality      TEXT,
            insufficient_data INTEGER,
            published_auc     REAL,
            health_index      INTEGER,
            top_signals       TEXT
        )
    """)

    scored_at    = result.get('scored_at', datetime.now().isoformat())
    health_index = result.get('health_index', 100)

    for s in result.get('scores', []):
        conn.execute("""
            INSERT INTO ml_scores (
                scored_at, condition_id, condition_label, category,
                probability, score_0_100, level, urgent,
                completeness, data_quality, insufficient_data,
                published_auc, health_index, top_signals
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            scored_at,
            s['condition_id'],
            s['condition_label'],
            s['category'],
            s['probability'],
            s['score_0_100'],
            s['level'],
            int(s.get('urgent', False)),
            s['completeness'],
            s['data_quality'],
            int(s.get('insufficient_data', False)),
            s['published_auc'],
            health_index,
            json.dumps(s.get('top_signals', [])),
        ))

    conn.commit()
    conn.close()


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser(description='VIGIL ML Inference')
    parser.add_argument('--db', required=True, help='Path to vigil.db')
    parser.add_argument('--save', action='store_true', help='Save results to ml_scores table')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    args = parser.parse_args()

    result = score_from_db(args.db)
    if args.save:
        save_scores_to_db(args.db, result)
        print(f"Saved {len(result['scores'])} scores to {args.db}")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nHealth Index: {result['health_index']}/100")
        print(f"Scored at: {result['scored_at']}\n")
        for s in result['scores']:
            flag = '🚨' if s.get('urgent') else ('⚠️ ' if s['level'] == 'elevated' else '  ')
            status = '—' if s['insufficient_data'] else f"{s['score_0_100']:5.1f}"
            print(f"  {flag} {s['condition_label']:<40} {status}  [{s['level']:<9}] {s['data_quality']}")