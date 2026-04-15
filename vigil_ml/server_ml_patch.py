"""
VIGIL server.py — ML endpoints patch
======================================
Add these routes to your existing server.py on the Jetson.

INSTRUCTIONS:
1. Copy this file to ~/passive-health-moniter/server_ml_patch.py
2. In your server.py, add at the top:
       from server_ml_patch import register_ml_routes
3. After you create your Flask app (app = Flask(__name__)), add:
       register_ml_routes(app, DB_PATH, VIGIL_ML_DIR)
   Where VIGIL_ML_DIR is the path to this vigil_ml folder.

Or just paste the route functions directly into server.py.

New endpoints added:
  GET  /ml/scores/latest       → latest scored results for all conditions
  GET  /ml/scores/history      → last 30 scoring runs
  POST /ml/score/now           → trigger immediate scoring (called after sync)
  GET  /ml/status              → model load status + version info
"""

import os, sys, json, sqlite3
from datetime import datetime
from flask import jsonify, request


def register_ml_routes(app, db_path: str, vigil_ml_dir: str):
    """Call this from server.py after creating the Flask app."""

    # Add vigil_ml to path
    if vigil_ml_dir not in sys.path:
        sys.path.insert(0, vigil_ml_dir)

    # Lazy-import (only loads models on first request)
    _inference = None
    def _get_inference():
        nonlocal _inference
        if _inference is None:
            import vigil_ml.vigil_inference as vigil_inference
            _inference = vigil_inference
        return _inference

    # ── GET /ml/scores/latest ─────────────────────────────────────────────
    @app.route('/ml/scores/latest', methods=['GET'])
    def ml_scores_latest():
        """
        Returns the most recent scoring run.
        If no scores in DB, triggers scoring on-demand.
        """
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            # Get the most recent scored_at timestamp
            latest_row = conn.execute(
                "SELECT scored_at FROM ml_scores ORDER BY id DESC LIMIT 1"
            ).fetchone()

            if not latest_row:
                conn.close()
                # No scores yet — run now
                inf = _get_inference()
                result = inf.score_from_db(db_path)
                inf.save_scores_to_db(db_path, result)
                return jsonify(result)

            scored_at = latest_row['scored_at']
            rows = conn.execute(
                "SELECT * FROM ml_scores WHERE scored_at = ? ORDER BY id",
                (scored_at,)
            ).fetchall()
            conn.close()

            scores = []
            health_index = 100
            for r in rows:
                health_index = r['health_index']
                scores.append({
                    "condition_id":     r['condition_id'],
                    "condition_label":  r['condition_label'],
                    "category":         r['category'],
                    "probability":      r['probability'],
                    "score_0_100":      r['score_0_100'],
                    "level":            r['level'],
                    "urgent":           bool(r['urgent']),
                    "completeness":     r['completeness'],
                    "data_quality":     r['data_quality'],
                    "insufficient_data":bool(r['insufficient_data']),
                    "published_auc":    r['published_auc'],
                    "top_signals":      json.loads(r['top_signals'] or '[]'),
                })
            return jsonify({
                "scored_at":    scored_at,
                "health_index": health_index,
                "scores":       scores,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500


    # ── GET /ml/scores/history ────────────────────────────────────────────
    @app.route('/ml/scores/history', methods=['GET'])
    def ml_scores_history():
        """Returns last 30 scoring runs, grouped by scored_at."""
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT DISTINCT scored_at, health_index
                FROM ml_scores
                ORDER BY scored_at DESC
                LIMIT 30
            """).fetchall()
            conn.close()
            return jsonify([{"scored_at": r['scored_at'],
                             "health_index": r['health_index']} for r in rows])
        except Exception as e:
            return jsonify({"error": str(e)}), 500


    # ── POST /ml/score/now ────────────────────────────────────────────────
    @app.route('/ml/score/now', methods=['POST'])
    def ml_score_now():
        """
        Trigger immediate ML scoring.
        Called automatically by server.py after a successful HealthKit sync
        (when enough data exists — call only if daily_summary has ≥7 rows).
        """
        try:
            inf = _get_inference()
            result = inf.score_from_db(db_path, n_days=30)
            inf.save_scores_to_db(db_path, result)
            return jsonify({"status": "ok", "health_index": result['health_index'],
                            "conditions_scored": len(result['scores'])})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


    # ── GET /ml/status ────────────────────────────────────────────────────
    @app.route('/ml/status', methods=['GET'])
    def ml_status():
        """Returns model load status and version info."""
        try:
            models_dir = os.path.join(vigil_ml_dir, 'models')
            conditions = ['afib', 'parkinsons', 'sleep_apnea',
                          'heart_failure', 'infection', 'frailty']
            status = []
            for cid in conditions:
                meta_path  = os.path.join(models_dir, f"{cid}_meta.json")
                model_path = os.path.join(models_dir, f"{cid}_model.joblib")
                if os.path.exists(meta_path) and os.path.exists(model_path):
                    meta = json.load(open(meta_path))
                    status.append({
                        "condition_id":  cid,
                        "version":       meta.get('version'),
                        "cv_roc_auc":    meta.get('cv_roc_auc_mean'),
                        "published_auc": meta.get('published_auc'),
                        "loaded":        True,
                    })
                else:
                    status.append({"condition_id": cid, "loaded": False})
            return jsonify({"models": status, "models_dir": models_dir})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    print("[VIGIL ML] Routes registered: /ml/scores/latest, /ml/scores/history, "
          "/ml/score/now, /ml/status")


# ─── server.py integration snippet ────────────────────────────────────────────
"""
Add these lines to your existing server.py:

# At top of file:
import os, sys
VIGIL_ML_DIR = os.path.expanduser("~/passive-health-moniter/vigil_ml")
sys.path.insert(0, VIGIL_ML_DIR)
from server_ml_patch import register_ml_routes

# After app = Flask(__name__):
register_ml_routes(app, DB_PATH, VIGIL_ML_DIR)

# In your existing sync endpoint, after _rebuild_daily_summary(),
# add a background scoring trigger (non-blocking):
import threading
def _maybe_score():
    try:
        import vigil_inference as inf
        # Only score if we have ≥7 days
        conn = sqlite3.connect(DB_PATH)
        n = conn.execute("SELECT COUNT(*) FROM daily_summary").fetchone()[0]
        conn.close()
        if n >= 7:
            result = inf.score_from_db(DB_PATH)
            inf.save_scores_to_db(DB_PATH, result)
    except Exception as e:
        print(f"[ML] Background scoring error: {e}")

# In sync handler, after successful daily_summary rebuild:
threading.Thread(target=_maybe_score, daemon=True).start()
"""
