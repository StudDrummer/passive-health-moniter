"""
VIGIL Feature Engineering Library
==================================
Shared feature extractors for all 6 ML models.
Input: list of daily_summary dicts (snake_case, from SQLite).
Output: numpy arrays ready for sklearn/XGBoost inference.

All functions return (feature_vector, feature_names, completeness_0_to_1).
completeness < 0.4 → do not surface score to user.
"""

import math
import numpy as np
from typing import Optional

# ─── Low-level signal utilities ──────────────────────────────────────────────

def _safe(v, default=0.0):
    """Return float or default if None/NaN."""
    if v is None: return default
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default

def _vals(rows, key, n=None):
    """Extract list of valid floats for a key from rows."""
    src = rows[:n] if n else rows
    return [float(r[key]) for r in src if r.get(key) is not None
            and not math.isnan(float(r[key])) and float(r[key]) > 0]

def _mean(lst): return sum(lst)/len(lst) if lst else 0.0
def _std(lst):
    if len(lst) < 2: return 0.0
    m = _mean(lst)
    return math.sqrt(sum((x-m)**2 for x in lst)/(len(lst)-1))
def _cv(lst): return (_std(lst)/_mean(lst)*100) if _mean(lst)>0 else 0.0
def _zscore(val, lst):
    if not lst or val is None: return 0.0
    m=_mean(lst); s=_std(lst)
    return (float(val)-m)/s if s>0.001 else 0.0
def _slope(lst):
    """Linear regression slope (units per day). Positive = increasing."""
    n=len(lst)
    if n<3: return 0.0
    xs=list(range(n)); mx=_mean(xs); my=_mean(lst)
    num=sum((x-mx)*(y-my) for x,y in zip(xs,lst))
    den=sum((x-mx)**2 for x in xs)
    return num/den if den>0 else 0.0
def _pct_change(lst, recent_n=3, base_n=10):
    """% change of recent_n mean vs preceding base_n mean."""
    if len(lst) < recent_n+3: return 0.0
    recent = _mean(lst[:recent_n])
    base   = _mean(lst[recent_n:recent_n+base_n])
    return ((recent-base)/base*100) if base>0 else 0.0

# ─── Per-condition feature extractors ────────────────────────────────────────

def features_afib(rows):
    """
    AFib proxy from resting HR + HRV patterns.
    Published AUC: PPG-based AF detection ~0.97 (Hannun et al. Nature Med 2019).
    Signals: HRV day-to-day CV, mean HRV, HR elevation z-score, HR/HRV coupling.
    Needs: ≥7 days resting_hr + hrv_sdnn.
    """
    names = [
        "hrv_mean_14d",       "hrv_cv_14d",         "hrv_slope_14d",
        "rhr_mean_14d",       "rhr_cv_14d",          "rhr_slope_14d",
        "hr_hrv_ratio",       "hrv_suppression_z",   "rhr_elevation_z",
        "days_hrv_lt20",      "days_rhr_gt90",        "spo2_mean",
        "spo2_dip_count",     "completeness_flag",
    ]
    n_needed = 7
    hrv  = _vals(rows, 'hrv_sdnn', 14)
    rhr  = _vals(rows, 'resting_hr', 14)
    spo2 = _vals(rows, 'spo2_avg', 14)
    completeness = min(len(hrv), len(rhr)) / n_needed

    feat = [
        _mean(hrv),                              # hrv_mean_14d
        _cv(hrv),                                # hrv_cv_14d  ← key AFib signal
        _slope(hrv),                             # hrv_slope_14d
        _mean(rhr),                              # rhr_mean_14d
        _cv(rhr),                                # rhr_cv_14d  ← erratic = AFib
        _slope(rhr),                             # rhr_slope_14d
        _mean(rhr)/_mean(hrv) if _mean(hrv)>0 else 0,  # hr_hrv_ratio
        _zscore(_mean(hrv[:3]), hrv[3:]) * -1,  # hrv_suppression_z (inverted)
        _zscore(_mean(rhr[:3]), rhr[3:]),        # rhr_elevation_z
        sum(1 for r in rows[:14] if _safe(r.get('hrv_sdnn'))<20),   # days_hrv_lt20
        sum(1 for r in rows[:14] if _safe(r.get('resting_hr'))>90), # days_rhr_gt90
        _mean(spo2) if spo2 else 97.0,           # spo2_mean
        sum(1 for v in spo2 if v<95),            # spo2_dip_count
        completeness,
    ]
    return np.array(feat, dtype=np.float32), names, min(1.0, completeness)


def features_parkinsons(rows):
    """
    Parkinson's gait signatures.
    Published balanced accuracy: ~91% (Pham et al. J Neurological Sci 2017).
    Signals: gait asymmetry, stride variability, walking speed decline,
             arm swing asymmetry, cadence variability, double support.
    Needs: ≥7 days walking metrics.
    """
    names = [
        "asym_mean_14d",      "asym_cv_14d",          "asym_slope_14d",
        "speed_mean_14d",     "speed_slope_14d",       "speed_cv_14d",
        "dbl_support_mean",   "stride_var_mean",       "arm_swing_asym_mean",
        "cadence_var_mean",   "step_len_mean",         "step_len_slope",
        "cam_asym_mean",      "gait_composite",        "days_speed_lt08",
        "completeness_flag",
    ]
    asym     = _vals(rows, 'walking_asymmetry_pct', 14)
    speed    = _vals(rows, 'walking_speed_ms', 14)
    dbl      = _vals(rows, 'double_support_pct', 14)
    sv       = _vals(rows, 'stride_variability', 14)
    arm      = _vals(rows, 'arm_swing_asymmetry', 14)
    cadvar   = _vals(rows, 'cadence_variability', 14)
    steplen  = _vals(rows, 'walking_step_length_m', 14)
    cam_asym = _vals(rows, 'camera_asymmetry_pct', 14)

    n_core   = len([x for x in [asym, speed] if len(x)>=3])
    completeness = n_core / 2

    asym_mean  = _mean(asym)
    speed_mean = _mean(speed)
    dbl_mean   = _mean(dbl)
    # Composite gait risk (0–1): combines asymmetry, speed, double-support
    gait_composite = (
        min(1, asym_mean/15.0) * 0.35 +
        max(0, (1.2 - speed_mean) / 0.8) * 0.35 +
        min(1, max(0, (dbl_mean - 20)/15)) * 0.30
    ) if speed_mean > 0 else 0.0

    feat = [
        asym_mean,
        _cv(asym),
        _slope(asym),
        speed_mean,
        _slope(speed),
        _cv(speed),
        dbl_mean,
        _mean(sv),
        _mean(arm),
        _mean(cadvar),
        _mean(steplen),
        _slope(steplen),
        _mean(cam_asym),
        gait_composite,
        sum(1 for v in speed if v < 0.8),
        completeness,
    ]
    return np.array(feat, dtype=np.float32), names, min(1.0, completeness)


def features_sleep_apnea(rows):
    """
    Sleep apnea proxy.
    Published: Apple Watch SA detection sensitivity 89%, specificity 98.5%.
    Signals: SpO2 mean/dip, resp rate, sleep fragmentation, sleep duration CV.
    Needs: ≥5 days spo2 + sleep_hours.
    """
    names = [
        "spo2_mean_7d",       "spo2_min_7d",          "spo2_cv_7d",
        "spo2_lt94_days",     "spo2_lt96_days",        "resp_mean_7d",
        "resp_cv_7d",         "sleep_mean_7d",         "sleep_cv_7d",
        "hypersomnia_days",   "short_sleep_days",      "spo2_resp_product",
        "resp_slope",         "spo2_slope",            "completeness_flag",
    ]
    spo2  = _vals(rows, 'spo2_avg', 7)
    resp  = _vals(rows, 'respiratory_rate', 7)
    sleep = _vals(rows, 'sleep_hours', 7)
    n_core = len([x for x in [spo2, sleep] if len(x)>=3])
    completeness = n_core / 2

    spo2_mean  = _mean(spo2)
    resp_mean  = _mean(resp)
    feat = [
        spo2_mean,
        min(spo2) if spo2 else 97.0,
        _cv(spo2),
        sum(1 for v in spo2 if v < 94),
        sum(1 for v in spo2 if v < 96),
        resp_mean,
        _cv(resp),
        _mean(sleep),
        _cv(sleep),
        sum(1 for v in sleep if v > 9.5),    # hypersomnia (compensatory)
        sum(1 for v in sleep if v < 5.5),    # insomnia/fragmented
        spo2_mean * resp_mean if spo2_mean > 0 and resp_mean > 0 else 0,
        _slope(resp),
        _slope(spo2),
        completeness,
    ]
    return np.array(feat, dtype=np.float32), names, min(1.0, completeness)


def features_heart_failure(rows):
    """
    Cardiac decompensation / heart failure risk.
    Bhatt et al. JACC Heart Failure 2017: wearable pVO2 decline predicts HF hospitalization.
    Signals: VO2max decline, HRV collapse, resting HR rise, activity collapse,
             SpO2 drop, step count decline.
    Needs: ≥14 days.
    """
    names = [
        "vo2_mean_30d",       "vo2_slope_30d",         "vo2_pct_change",
        "rhr_mean_14d",       "rhr_slope_14d",          "rhr_elevation",
        "hrv_mean_14d",       "hrv_slope_14d",           "hrv_collapse_z",
        "steps_mean_14d",     "steps_pct_change",        "steps_slope",
        "spo2_mean_7d",       "spo2_dip_days",          "active_cal_slope",
        "completeness_flag",
    ]
    vo2   = _vals(rows, 'vo2_max', 30)
    rhr   = _vals(rows, 'resting_hr', 14)
    hrv   = _vals(rows, 'hrv_sdnn', 14)
    steps = _vals(rows, 'step_count', 14)
    spo2  = _vals(rows, 'spo2_avg', 7)
    cals  = _vals(rows, 'active_calories', 14)

    n_core = len([x for x in [rhr, steps] if len(x)>=5])
    completeness = n_core / 2

    feat = [
        _mean(vo2),
        _slope(vo2),
        _pct_change(vo2, 3, 7),
        _mean(rhr),
        _slope(rhr),
        max(0, _mean(rhr) - 75),              # elevation above 75bpm
        _mean(hrv),
        _slope(hrv),
        _zscore(_mean(hrv[:3]), hrv[3:]) * -1 if len(hrv)>3 else 0,  # collapse
        _mean(steps),
        _pct_change(steps, 3, 7),
        _slope(steps),
        _mean(spo2) if spo2 else 97.0,
        sum(1 for v in spo2 if v < 93),
        _slope(cals),
        completeness,
    ]
    return np.array(feat, dtype=np.float32), names, min(1.0, completeness)


def features_infection(rows):
    """
    Acute infection (COVID/flu) early warning.
    Systematic review AUC 0.52–0.92 (Shapiro et al. NPJ Digit Med 2021).
    Uses personal baseline z-scores, not population thresholds.
    Signals: HR elevation vs personal baseline, HRV suppression,
             SpO2 dip, sleep disruption, temp elevation.
    Needs: ≥14 days for baseline (days 3-16) + today.
    """
    names = [
        "hr_z_today",         "hrv_z_today",           "spo2_z_today",
        "sleep_z_today",      "steps_z_today",         "temp_today",
        "temp_z_today",       "hr_hrv_divergence",      "infection_composite",
        "days_since_hr_spike","days_since_hrv_drop",   "resp_z_today",
        "hr_3d_trend",        "hrv_3d_trend",           "completeness_flag",
    ]
    # Baseline = days 3–16 (skip recent 2 in case infection already started)
    baseline_rhr = _vals(rows[3:17], 'resting_hr')
    baseline_hrv = _vals(rows[3:17], 'hrv_sdnn')
    baseline_spo = _vals(rows[3:17], 'spo2_avg')
    baseline_slp = _vals(rows[3:17], 'sleep_hours')
    baseline_stp = _vals(rows[3:17], 'step_count')
    baseline_tmp = _vals(rows[3:17], 'wrist_temp') if any(r.get('wrist_temp') for r in rows[3:17]) else []
    baseline_rsp = _vals(rows[3:17], 'respiratory_rate')

    today = rows[0] if rows else {}
    n_core = len([x for x in [baseline_rhr, baseline_hrv] if len(x)>=3])
    completeness = n_core / 2

    hr_z   = _zscore(_safe(today.get('resting_hr')),   baseline_rhr)
    hrv_z  = _zscore(_safe(today.get('hrv_sdnn')),     baseline_hrv) * -1  # invert: low HRV = bad
    spo_z  = _zscore(_safe(today.get('spo2_avg')),     baseline_spo) * -1  # invert
    slp_z  = _zscore(_safe(today.get('sleep_hours')),  baseline_slp) * -1
    stp_z  = _zscore(_safe(today.get('step_count')),   baseline_stp) * -1
    tmp_z  = _zscore(_safe(today.get('wrist_temp')),   baseline_tmp)
    rsp_z  = _zscore(_safe(today.get('respiratory_rate')), baseline_rsp)
    # Infection: HR and HRV diverge (HR up, HRV down simultaneously)
    div    = (hr_z + hrv_z) / 2.0
    composite = max(0, (hr_z * 0.3 + hrv_z * 0.3 + spo_z * 0.2 + tmp_z * 0.2))

    recent_rhr = _vals(rows[:3], 'resting_hr')
    recent_hrv = _vals(rows[:3], 'hrv_sdnn')

    feat = [
        hr_z, hrv_z, spo_z, slp_z, stp_z,
        _safe(today.get('wrist_temp')),
        tmp_z,
        div,
        min(1.0, max(0.0, composite)),
        # days since last HR spike above baseline+1.5σ
        next((i for i,r in enumerate(rows[:14])
              if _zscore(_safe(r.get('resting_hr')), baseline_rhr) > 1.5), 14),
        # days since last HRV drop below baseline-1.5σ
        next((i for i,r in enumerate(rows[:14])
              if _zscore(_safe(r.get('hrv_sdnn')), baseline_hrv) < -1.5), 14),
        rsp_z,
        _slope(recent_rhr) if len(recent_rhr)>=2 else 0,
        _slope(recent_hrv) if len(recent_hrv)>=2 else 0,
        completeness,
    ]
    return np.array(feat, dtype=np.float32), names, min(1.0, completeness)


def features_frailty(rows):
    """
    Frailty / functional decline.
    Apple Watch frailty prediction sensitivity 83-90% (Guo et al. JAMDA 2021).
    Signals: walking speed, step count, VO2max, double support time,
             step length, activity calories, flights climbed.
    Needs: ≥7 days.
    """
    names = [
        "speed_mean_14d",     "speed_slope_14d",       "speed_lt10_days",
        "steps_mean_14d",     "steps_slope_14d",        "steps_lt3k_days",
        "vo2_mean",           "dbl_support_mean",       "step_len_mean",
        "active_cal_mean",    "active_cal_slope",        "flights_mean",
        "speed_steps_product","frailty_composite",       "exercise_min_mean",
        "completeness_flag",
    ]
    speed   = _vals(rows, 'walking_speed_ms', 14)
    steps   = _vals(rows, 'step_count', 14)
    vo2     = _vals(rows, 'vo2_max', 30)
    dbl     = _vals(rows, 'double_support_pct', 14)
    steplen = _vals(rows, 'walking_step_length_m', 14)
    cals    = _vals(rows, 'active_calories', 14)
    flights = _vals(rows, 'flights_climbed', 14)
    exmin   = _vals(rows, 'exercise_minutes', 14)

    n_core  = len([x for x in [speed, steps] if len(x)>=5])
    completeness = n_core / 2

    speed_mean = _mean(speed)
    steps_mean = _mean(steps)
    vo2_mean   = _mean(vo2)
    # Fried frailty phenotype proxy (0–1)
    frailty_composite = (
        max(0, (5000 - steps_mean) / 5000) * 0.30 +
        max(0, (1.0  - speed_mean) / 0.6)  * 0.35 +
        max(0, (25   - vo2_mean)   / 15)   * 0.25 +
        min(1, max(0, (_mean(dbl)-20)/15)) * 0.10
    ) if speed_mean > 0 else 0.5

    feat = [
        speed_mean,
        _slope(speed),
        sum(1 for v in speed if v < 1.0),
        steps_mean,
        _slope(steps),
        sum(1 for v in steps if v < 3000),
        vo2_mean,
        _mean(dbl),
        _mean(steplen),
        _mean(cals),
        _slope(cals),
        _mean(flights),
        speed_mean * steps_mean / 10000 if speed_mean > 0 else 0,
        frailty_composite,
        _mean(exmin),
        completeness,
    ]
    return np.array(feat, dtype=np.float32), names, min(1.0, completeness)


# ─── Feature dispatcher ───────────────────────────────────────────────────────

FEATURE_FNS = {
    'afib':        features_afib,
    'parkinsons':  features_parkinsons,
    'sleep_apnea': features_sleep_apnea,
    'heart_failure': features_heart_failure,
    'infection':   features_infection,
    'frailty':     features_frailty,
}

def extract(condition_id: str, rows: list):
    """
    Main entry point.
    Returns (np.array, feature_names, completeness_0_to_1).
    """
    fn = FEATURE_FNS.get(condition_id)
    if fn is None:
        raise ValueError(f"Unknown condition: {condition_id}")
    return fn(rows)
