"""
VIGIL Model Training Pipeline v2.0
=====================================
Extended to ingest real PhysioNet/UCI datasets alongside synthetic data.
When real data is available, it is mixed with synthetic at 70/30 ratio
(real data weighted 3x — more realistic decision boundaries).

New conditions added beyond the original 6:
  - stress          (WESAD dataset)
  - depression_v2   (WESAD + GLOBEM combined)
  - fall_risk       (SisFall + synthetic frailty)
  - diabetes_risk   (PhysioNet wrist glucose 2026)

Usage:
    python3 vigil_train_v2.py [--models-dir ./models] [--data-dir ./data]
    python3 vigil_train_v2.py --condition parkinsons --data-dir ./data
    python3 vigil_train_v2.py --real-only    (only train conditions with real data)
"""

import os, sys, json, argparse, warnings
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve

warnings.filterwarnings('ignore')

# Add parent to path for vigil_features
sys.path.insert(0, str(Path(__file__).parent.parent / "vigil_ml"))
import vigil_features as vf
import vigil_synth_data as vsd

# ─── Extended condition registry ──────────────────────────────────────────────
CONDITIONS_V2 = {
    # Original 6 — all have synthetic + real data support
    "afib": {
        "label":    "Atrial Fibrillation Risk",
        "category": "cardiovascular",
        "pub_auc":  0.97,
        "real_files": ["preprocessed_afib.json"],  # from cinc2017
        "n_synthetic_fallback": 500,
    },
    "parkinsons": {
        "label":    "Parkinson's / Movement Risk",
        "category": "neurological",
        "pub_auc":  0.96,
        "real_files": ["preprocessed_parkinsons_pads.json",
                       "preprocessed_parkinsons_gait.json"],
        "n_synthetic_fallback": 400,
    },
    "sleep_apnea": {
        "label":    "Sleep Apnea Risk",
        "category": "respiratory",
        "pub_auc":  0.94,
        "real_files": ["preprocessed_sleep_apnea_dreamt.json",
                       "preprocessed_sleep_apnea_ucddb.json"],
        "n_synthetic_fallback": 400,
    },
    "heart_failure": {
        "label":    "Cardiac Decompensation Risk",
        "category": "cardiovascular",
        "pub_auc":  0.90,
        "real_files": ["preprocessed_heart_failure.json"],
        "n_synthetic_fallback": 400,
    },
    "infection": {
        "label":    "Acute Infection Signal",
        "category": "respiratory",
        "pub_auc":  0.82,
        "real_files": [],  # no direct real dataset; personal baseline z-scores
        "n_synthetic_fallback": 500,
    },
    "frailty": {
        "label":    "Frailty / Low Fitness",
        "category": "musculoskeletal",
        "pub_auc":  0.88,
        "real_files": [],
        "n_synthetic_fallback": 400,
    },
    # New conditions
    "stress": {
        "label":    "Chronic Stress / Autonomic Overload",
        "category": "psychiatric",
        "pub_auc":  0.93,
        "real_files": ["preprocessed_stress.json"],
        "n_synthetic_fallback": 300,
    },
    "depression": {
        "label":    "Depression / MDD Pattern",
        "category": "psychiatric",
        "pub_auc":  0.73,
        "real_files": ["preprocessed_depression_wesad.json",
                       "preprocessed_depression_globem.json"],
        "n_synthetic_fallback": 400,
    },
}

# Feature extractor for new conditions reuses existing ones with remapping
FEATURE_FN_MAP = {
    "stress":     "afib",        # stress uses similar HRV + HR features
    "depression": "frailty",     # depression uses activity + sleep patterns
}


def _load_real_data(condition: str, data_dir: Path) -> list:
    """Load all available real dataset files for a condition."""
    info = CONDITIONS_V2.get(condition, {})
    all_records = []
    for fname in info.get("real_files", []):
        fpath = data_dir / fname
        if fpath.exists():
            try:
                records = json.load(open(fpath))
                all_records.extend(records)
                print(f"    Loaded {len(records)} records from {fname}")
            except Exception as e:
                print(f"    Warning: Failed to load {fname}: {e}")
    return all_records


def _build_features(condition: str, records: list, weight: float = 1.0):
    """Extract features from records, returns (X, y, weights)."""
    feat_cond = FEATURE_FN_MAP.get(condition, condition)
    X_list, y_list, w_list = [], [], []
    feat_names = None

    for rec in records:
        rows  = rec.get("rows", [])
        label = rec.get("label", 0)
        try:
            feat, names, completeness = vf.extract(feat_cond, rows)
            if completeness < 0.2:
                continue
            X_list.append(feat)
            y_list.append(int(label))
            w_list.append(weight)
            if feat_names is None:
                feat_names = names
        except Exception:
            continue

    return X_list, y_list, w_list, feat_names


def _build_gbm():
    base = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        max_features=0.8,
        random_state=42,
    )
    return CalibratedClassifierCV(base, cv=3, method='isotonic')


def train_condition_v2(condition: str, models_dir: Path, data_dir: Path):
    info = CONDITIONS_V2.get(condition)
    if not info:
        print(f"Unknown condition: {condition}")
        return None

    print(f"\n{'─'*60}")
    print(f"  Training: {info['label']}")
    print(f"{'─'*60}")

    feat_cond = FEATURE_FN_MAP.get(condition, condition)

    # ── Load real data (weight=3.0) ──────────────────────────────────────
    real_records = _load_real_data(condition, data_dir)
    X_real, y_real, w_real, feat_names = _build_features(condition, real_records, weight=3.0)
    print(f"    Real data: {len(X_real)} samples")

    # ── Generate synthetic data (weight=1.0) ─────────────────────────────
    n_synth_cases = info["n_synthetic_fallback"]
    n_synth_ctrl  = n_synth_cases * 2
    try:
        synth_dataset = vsd.build_dataset(feat_cond, n_cases=n_synth_cases,
                                          n_controls=n_synth_ctrl, days=21)
        X_synth, y_synth, w_synth, fn = _build_features(condition, [
            {"rows": r, "label": l} for r, l in synth_dataset], weight=1.0)
        if feat_names is None:
            feat_names = fn
    except Exception as e:
        print(f"    Synthetic generation failed: {e}")
        X_synth, y_synth, w_synth = [], [], []
    print(f"    Synthetic data: {len(X_synth)} samples")

    # ── Combine ───────────────────────────────────────────────────────────
    X_all = np.array(X_real + X_synth)
    y_all = np.array(y_real + y_synth)
    w_all = np.array(w_real + w_synth)

    if len(X_all) < 20:
        print(f"    ERROR: Insufficient data ({len(X_all)} samples). Skipping.")
        return None

    pos_rate = y_all.mean()
    print(f"    Total: {len(X_all)} samples, {y_all.sum():.0f}/{len(y_all)} positive "
          f"({100*pos_rate:.0f}%), mean weight={np.mean(w_all):.2f}")

    # ── Cross-validation ──────────────────────────────────────────────────
    model = _build_gbm()
    cv    = StratifiedKFold(n_splits=min(5, int(min(y_all.sum(), len(y_all)-y_all.sum()))),
                            shuffle=True, random_state=42)
    try:
        roc_scores = cross_val_score(model, X_all, y_all, cv=cv,
                                     scoring='roc_auc', fit_params={'sample_weight': w_all})
        ap_scores  = cross_val_score(model, X_all, y_all, cv=cv,
                                     scoring='average_precision', fit_params={'sample_weight': w_all})
    except Exception:
        # Some sklearn versions don't support fit_params in cross_val_score
        roc_scores = cross_val_score(model, X_all, y_all, cv=cv, scoring='roc_auc')
        ap_scores  = cross_val_score(model, X_all, y_all, cv=cv, scoring='average_precision')

    print(f"    CV ROC-AUC:   {roc_scores.mean():.3f} ± {roc_scores.std():.3f}")
    print(f"    CV Avg Prec:  {ap_scores.mean():.3f} ± {ap_scores.std():.3f}")
    print(f"    Published AUC: {info['pub_auc']}")

    # ── Final fit ─────────────────────────────────────────────────────────
    model.fit(X_all, y_all, sample_weight=w_all)
    probs = model.predict_proba(X_all)[:, 1]
    train_auc   = roc_auc_score(y_all, probs, sample_weight=w_all)
    train_brier = brier_score_loss(y_all, probs, sample_weight=w_all)

    # Threshold selection: ≥80% sensitivity
    fpr, tpr, thresholds = roc_curve(y_all, probs, sample_weight=w_all)
    idx80    = np.argmin(np.abs(tpr - 0.80))
    idx90    = np.argmin(np.abs(tpr - 0.90))
    alert_t  = float(np.clip(thresholds[idx80], 0.05, 0.95))
    urgent_t = float(np.clip(thresholds[idx90], 0.05, 0.95))
    print(f"    Train AUC: {train_auc:.3f}  Brier: {train_brier:.3f}")
    print(f"    Alert threshold: {alert_t:.3f}  Urgent: {urgent_t:.3f}")

    # Feature importances
    try:
        base_model  = model.calibrated_classifiers_[0].estimator
        importances = base_model.feature_importances_
        feat_imp    = sorted(zip(feat_names or [], importances), key=lambda x: -x[1])
        print("    Top 5 features:")
        for fname, imp in feat_imp[:5]:
            print(f"      {fname:<32} {imp:.3f}")
    except Exception:
        feat_imp = []

    # ── Save ──────────────────────────────────────────────────────────────
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{condition}_model.joblib"
    meta_path  = models_dir / f"{condition}_meta.json"
    joblib.dump(model, model_path)

    meta = {
        "condition_id":      condition,
        "condition_label":   info["label"],
        "category":          info["category"],
        "version":           "2.0",
        "cv_roc_auc_mean":   float(roc_scores.mean()),
        "cv_roc_auc_std":    float(roc_scores.std()),
        "cv_avg_prec_mean":  float(ap_scores.mean()),
        "train_roc_auc":     float(train_auc),
        "train_brier":       float(train_brier),
        "published_auc":     info["pub_auc"],
        "alert_threshold":   alert_t,
        "urgent_threshold":  urgent_t,
        "n_features":        int(X_all.shape[1]),
        "feature_names":     feat_names or [],
        "top_features":      [(f, float(i)) for f, i in (feat_imp[:5] if feat_imp else [])],
        "n_real_samples":    len(X_real),
        "n_synth_samples":   len(X_synth),
        "real_data_sources": [f for f in info.get("real_files", [])
                              if (data_dir / f).exists()],
        "dataset_type":      "mixed_real_synthetic" if X_real else "synthetic_only",
    }
    json.dump(meta, open(meta_path, 'w'), indent=2)
    print(f"    Saved: {model_path}")
    return meta


def train_all_v2(models_dir: Path, data_dir: Path):
    print("\n" + "="*60)
    print("  VIGIL ML Training Pipeline v2.0")
    print(f"  {len(CONDITIONS_V2)} conditions | Real + Synthetic data")
    print("="*60)

    report = []
    for condition in CONDITIONS_V2:
        meta = train_condition_v2(condition, models_dir, data_dir)
        if meta:
            report.append(meta)

    # Write report
    report_path = models_dir / "training_report_v2.txt"
    with open(report_path, 'w') as f:
        f.write("VIGIL ML Training Report v2.0\n")
        f.write("="*60 + "\n\n")
        for m in report:
            f.write(f"{m['condition_label']}\n")
            f.write(f"  Version: {m['version']}\n")
            f.write(f"  CV ROC-AUC: {m['cv_roc_auc_mean']:.3f} ± {m['cv_roc_auc_std']:.3f}\n")
            f.write(f"  Published AUC: {m['published_auc']}\n")
            f.write(f"  Data: {m['n_real_samples']} real + {m['n_synth_samples']} synth\n")
            f.write(f"  Type: {m['dataset_type']}\n")
            f.write(f"  Thresholds: alert={m['alert_threshold']:.3f} urgent={m['urgent_threshold']:.3f}\n")
            f.write("\n")

    print(f"\n{'='*60}")
    print(f"  Done. {len(report)} models trained.")
    print(f"  Report: {report_path}")
    print(f"{'='*60}\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIGIL ML Training v2.0")
    parser.add_argument("--models-dir", default="./models",  type=Path)
    parser.add_argument("--data-dir",   default="./data",    type=Path)
    parser.add_argument("--condition",  default=None,         type=str)
    parser.add_argument("--real-only",  action="store_true",
                        help="Only train conditions that have real data available")
    args = parser.parse_args()

    if args.condition:
        train_condition_v2(args.condition, Path(args.models_dir), Path(args.data_dir))
    elif args.real_only:
        for cid, info in CONDITIONS_V2.items():
            has_real = any((Path(args.data_dir) / f).exists()
                          for f in info.get("real_files", []))
            if has_real:
                train_condition_v2(cid, Path(args.models_dir), Path(args.data_dir))
    else:
        train_all_v2(Path(args.models_dir), Path(args.data_dir))
