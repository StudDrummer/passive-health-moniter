"""
VIGIL ML Nightly Scorer
========================
Runs as a cron job on the Jetson, reads daily_summary, writes ml_scores.

Install:
  crontab -e
  0 3 * * * cd ~/passive-health-moniter && python3 ml_scorer.py >> scorer.log 2>&1

Or run manually:
  python3 ml_scorer.py [--db ~/passive-health-moniter/vigil.db]
"""

import os, sys, argparse, json
from datetime import datetime

# Add vigil_ml dir to path (sibling or same dir)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import vigil_inference as inf

DB_DEFAULT = os.path.expanduser("~/passive-health-moniter/vigil.db")
MODELS_DIR = os.path.join(_HERE, "models")


def run(db_path: str):
    if not os.path.exists(db_path):
        print(f"[ml_scorer] DB not found: {db_path}")
        return

    # Ensure models exist
    if not os.path.exists(os.path.join(MODELS_DIR, "afib_model.joblib")):
        print("[ml_scorer] Models not found — running training first...")
        import vigil_train
        vigil_train.train_all(MODELS_DIR)
        print("[ml_scorer] Training complete.")

    print(f"[ml_scorer] {datetime.now():%Y-%m-%d %H:%M} — scoring from {db_path}")
    result = inf.score_from_db(db_path, n_days=30)
    inf.save_scores_to_db(db_path, result)

    print(f"[ml_scorer] Health Index: {result['health_index']}/100")
    for s in result['scores']:
        flag  = '[URGENT] ' if s.get('urgent') else ''
        score = '—' if s['insufficient_data'] else f"{s['score_0_100']:.1f}"
        print(f"  {flag}{s['condition_label']:<38} {score:>6}  [{s['level']}]")
    print(f"[ml_scorer] Done. {len(result['scores'])} conditions scored.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=DB_DEFAULT)
    args = parser.parse_args()
    run(args.db)
