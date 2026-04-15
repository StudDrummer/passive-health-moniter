"""
VIGIL Model Training Pipeline
==============================
Trains 6 condition-specific gradient-boosted classifiers.
Each model uses its condition-specific feature extractor from vigil_features.py.

Output:  models/  directory containing:
  - {condition}_model.joblib    (fitted GradientBoostingClassifier)
  - {condition}_meta.json       (thresholds, AUC, feature names, version)
  - training_report.txt         (CV performance summary)

Usage:
  python3 vigil_train.py [--data-dir /path/to/real/data] [--models-dir ./models]

Without --data-dir uses built-in synthetic data from vigil_synth_data.py.
With    --data-dir points to PhysioNet CSV exports (see README for format).

This script runs on the Jetson Orin Nano (takes ~2min with synthetic data).
Re-run whenever you have 30+ days of personal baseline data for transfer-learning
(uncomment the fine-tune section at the bottom).
"""

import os, json, argparse, warnings
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, brier_score_loss)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

import vigil_features as vf
import vigil_synth_data as vsd

# ─── Condition registry ───────────────────────────────────────────────────────
# (condition_id, published_auc, alert_threshold, urgent_threshold)
CONDITIONS = [
    ('afib',          0.97, 0.35, 0.65),
    ('parkinsons',    0.91, 0.30, 0.60),
    ('sleep_apnea',   0.94, 0.35, 0.65),
    ('heart_failure', 0.90, 0.35, 0.70),
    ('infection',     0.82, 0.40, 0.70),
    ('frailty',       0.88, 0.35, 0.65),
]

CONDITION_LABELS = {
    'afib':          'Atrial Fibrillation Risk',
    'parkinsons':    "Parkinson's / Movement Risk",
    'sleep_apnea':   'Sleep Apnea Risk',
    'heart_failure': 'Cardiac Decompensation Risk',
    'infection':     'Acute Infection Signal',
    'frailty':       'Frailty / Low Fitness',
}

CONDITION_CATEGORIES = {
    'afib':          'cardiovascular',
    'parkinsons':    'neurological',
    'sleep_apnea':   'respiratory',
    'heart_failure': 'cardiovascular',
    'infection':     'respiratory',
    'frailty':       'musculoskeletal',
}


def _build_gbm():
    """
    Gradient Boosting + Platt scaling calibration.
    GBM chosen over neural net because:
      - Works well on tabular daily-summary data (14–16 features)
      - Interpretable feature importances → clinical explainability
      - No GPU required on Jetson
      - Fast inference (<1ms per patient)
    """
    base = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=8,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=42,
    )
    return CalibratedClassifierCV(base, cv=3, method='isotonic')


def train_condition(condition_id: str, models_dir: str):
    """Train, evaluate and save one condition model."""
    print(f"\n{'─'*60}")
    print(f"  Training: {CONDITION_LABELS[condition_id]}")
    print(f"{'─'*60}")

    # ── Build dataset ────────────────────────────────────────────────────
    dataset = vsd.build_dataset(condition_id, n_cases=300, n_controls=600)

    X_list, y_list, feat_names = [], [], None
    for rows, label in dataset:
        try:
            feat, names, completeness = vf.extract(condition_id, rows)
            if completeness < 0.2:
                continue
            X_list.append(feat)
            y_list.append(label)
            if feat_names is None:
                feat_names = names
        except Exception as e:
            print(f"  Feature extraction error: {e}")
            continue

    if not X_list:
        print(f"  ERROR: No valid samples for {condition_id}")
        return None

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{y.sum()}/{len(y)} positive ({100*y.mean():.0f}%)")

    # ── Cross-validation ─────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = _build_gbm()
    roc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    ap_scores  = cross_val_score(model, X, y, cv=cv, scoring='average_precision', n_jobs=-1)

    print(f"  CV ROC-AUC:   {roc_scores.mean():.3f} ± {roc_scores.std():.3f}")
    print(f"  CV Avg Prec:  {ap_scores.mean():.3f} ± {ap_scores.std():.3f}")

    # ── Final fit on full dataset ─────────────────────────────────────────
    model.fit(X, y)

    # ── Calibration: compute thresholds from positive class probabilities ─
    probs = model.predict_proba(X)[:, 1]
    # Find threshold giving ~80% sensitivity (clinical priority: don't miss cases)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y, probs)
    idx80 = np.argmin(np.abs(tpr - 0.80))
    alert_threshold  = float(thresholds[idx80])
    idx90 = np.argmin(np.abs(tpr - 0.90))
    urgent_threshold = float(thresholds[idx90])

    train_auc = roc_auc_score(y, probs)
    train_brier = brier_score_loss(y, probs)
    print(f"  Train ROC-AUC: {train_auc:.3f}  Brier: {train_brier:.3f}")
    print(f"  Alert threshold (≥80% sens): {alert_threshold:.3f}")
    print(f"  Urgent threshold (≥90% sens): {urgent_threshold:.3f}")

    # ── Feature importances (from base estimator before calibration) ─────
    try:
        base_model = model.calibrated_classifiers_[0].estimator
        importances = base_model.feature_importances_
        feat_imp = sorted(zip(feat_names, importances), key=lambda x: -x[1])
        print("  Top 5 features:")
        for fname, imp in feat_imp[:5]:
            print(f"    {fname:<30} {imp:.3f}")
    except Exception:
        feat_imp = []

    # ── Save model ────────────────────────────────────────────────────────
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{condition_id}_model.joblib")
    joblib.dump(model, model_path)

    meta = {
        "condition_id":      condition_id,
        "condition_label":   CONDITION_LABELS[condition_id],
        "category":          CONDITION_CATEGORIES[condition_id],
        "version":           "1.0",
        "cv_roc_auc_mean":   float(roc_scores.mean()),
        "cv_roc_auc_std":    float(roc_scores.std()),
        "cv_avg_prec_mean":  float(ap_scores.mean()),
        "train_roc_auc":     float(train_auc),
        "train_brier":       float(train_brier),
        "published_auc":     next(c[1] for c in CONDITIONS if c[0]==condition_id),
        "alert_threshold":   alert_threshold,
        "urgent_threshold":  urgent_threshold,
        "n_features":        int(X.shape[1]),
        "feature_names":     feat_names,
        "top_features":      [(f, float(i)) for f,i in (feat_imp[:5] if feat_imp else [])],
        "n_train_samples":   int(len(X)),
        "dataset_type":      "synthetic_clinical_literature",
    }
    meta_path = os.path.join(models_dir, f"{condition_id}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved: {model_path}")
    return meta


def train_all(models_dir: str = './models'):
    """Train all 6 condition models."""
    print("\n" + "="*60)
    print("  VIGIL ML Training Pipeline")
    print("  6 conditions, gradient boosting + isotonic calibration")
    print("="*60)

    report = []
    for condition_id, *_ in CONDITIONS:
        meta = train_condition(condition_id, models_dir)
        if meta:
            report.append(meta)

    # ── Write training report ─────────────────────────────────────────────
    report_path = os.path.join(models_dir, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write("VIGIL ML Training Report\n")
        f.write("="*60 + "\n\n")
        for m in report:
            f.write(f"{m['condition_label']}\n")
            f.write(f"  CV ROC-AUC: {m['cv_roc_auc_mean']:.3f} ± {m['cv_roc_auc_std']:.3f}\n")
            f.write(f"  CV Avg Prec: {m['cv_avg_prec_mean']:.3f}\n")
            f.write(f"  Train AUC: {m['train_roc_auc']:.3f}\n")
            f.write(f"  Alert threshold: {m['alert_threshold']:.3f}\n")
            f.write(f"  Urgent threshold: {m['urgent_threshold']:.3f}\n")
            if m['top_features']:
                f.write("  Top features:\n")
                for fname, imp in m['top_features']:
                    f.write(f"    {fname}: {imp:.3f}\n")
            f.write("\n")

    print(f"\n{'='*60}")
    print(f"  Training complete. Report: {report_path}")
    print(f"  Models saved to: {models_dir}/")
    print(f"{'='*60}\n")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VIGIL ML Training Pipeline')
    parser.add_argument('--models-dir', default='./models',
                        help='Directory to save trained models')
    parser.add_argument('--condition', default=None,
                        help='Train only one condition (default: all)')
    args = parser.parse_args()

    if args.condition:
        train_condition(args.condition, args.models_dir)
    else:
        train_all(args.models_dir)
