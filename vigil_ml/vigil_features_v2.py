"""
vigil_features_v2.py  —  VIGIL Feature Extraction (14-condition edition)
=========================================================================
Drop-in replacement / extension of vigil_features.py.
Called by vigil_inference.py to compute per-condition feature vectors
from a subject's last N days of daily_summary rows.

Import this instead of vigil_features.py, or copy the new entries into
the existing file.

Public API (unchanged):
    FEATURE_FNS   — dict mapping condition_id → feature function
    extract(condition, rows) → np.ndarray
"""

import numpy as np
from typing import List, Dict, Any

Row = Dict[str, Any]


def _safe(v, default=0.0) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _get(rows: List[Row], key: str) -> List[float]:
    return [_safe(r.get(key)) for r in rows if r.get(key) is not None]


def _mean(vals) -> float:
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else 0.0


def _std(vals) -> float:
    v = [x for x in vals if x is not None]
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def _cv(vals) -> float:
    m = _mean(vals)
    return float(_std(vals) / m * 100) if m != 0 else 0.0


def _trend(vals) -> float:
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return 0.0
    x = np.arange(len(v), dtype=float)
    return float(np.polyfit(x, v, 1)[0])


def _min(vals) -> float:
    v = [x for x in vals if x is not None]
    return float(np.min(v)) if v else 0.0


def _max(vals) -> float:
    v = [x for x in vals if x is not None]
    return float(np.max(v)) if v else 0.0


# ─── Individual feature functions ────────────────────────────────────────────

def features_afib(rows: List[Row]) -> np.ndarray:
    hrv   = _get(rows, 'hrv_sdnn')
    rmssd = _get(rows, 'hrv_rmssd')
    pnn50 = _get(rows, 'hrv_pnn50')
    hr    = _get(rows, 'resting_hr')
    spo2  = _get(rows, 'spo2_avg')
    return np.array([
        _mean(hrv),   _std(hrv),   _cv(hrv),  _trend(hrv),
        _mean(rmssd), _mean(pnn50),
        _mean(hr),    _cv(hr),     _trend(hr),
        _mean(spo2),  _min(spo2),
    ], dtype=float)


def features_parkinsons(rows: List[Row]) -> np.ndarray:
    asym  = _get(rows, 'walking_asymmetry_pct')
    speed = _get(rows, 'walking_speed_ms')
    sv    = _get(rows, 'stride_variability')
    arm   = _get(rows, 'arm_swing_asymmetry')
    cv    = _get(rows, 'cadence_variability')
    ds    = _get(rows, 'double_support_pct')
    sl    = _get(rows, 'walking_step_length_m')
    tr    = _get(rows, 'tremor_amplitude')
    return np.array([
        _mean(asym),  _trend(asym),
        _mean(speed), _trend(speed),
        _mean(sv),    _cv(sv),
        _mean(arm),   _mean(cv),
        _mean(ds),    _mean(sl),
        _mean(tr),    _trend(tr),
    ], dtype=float)


def features_sleep_apnea(rows: List[Row]) -> np.ndarray:
    spo2  = _get(rows, 'spo2_avg')
    smin  = _get(rows, 'spo2_min')
    dips  = _get(rows, 'spo2_dips_below94')
    rr    = _get(rows, 'respiratory_rate')
    hr    = _get(rows, 'resting_hr')
    sleep = _get(rows, 'sleep_hours')
    hrv   = _get(rows, 'hrv_sdnn')
    return np.array([
        _mean(spo2),  _min(smin),  _mean(dips), _max(dips),
        _trend(spo2), _mean(rr),   _mean(hr),   _cv(hr),
        _mean(sleep), _cv(sleep),  _mean(hrv),
    ], dtype=float)


def features_heart_failure(rows: List[Row]) -> np.ndarray:
    hr   = _get(rows, 'resting_hr')
    spo2 = _get(rows, 'spo2_avg')
    rr   = _get(rows, 'respiratory_rate')
    hrv  = _get(rows, 'hrv_sdnn')
    std14 = _get(rows, 'resting_hr_std_14d')
    return np.array([
        _mean(hr),   _trend(hr),  _cv(hr),
        _mean(spo2), _min(spo2),
        _mean(rr),   _trend(rr),
        _mean(hrv),  _trend(hrv),
        _mean(std14),
    ], dtype=float)


def features_infection(rows: List[Row]) -> np.ndarray:
    hr   = _get(rows, 'resting_hr')
    temp = _get(rows, 'wrist_temp')
    rr   = _get(rows, 'respiratory_rate')
    spo2 = _get(rows, 'spo2_avg')
    step = _get(rows, 'step_count')
    hrv  = _get(rows, 'hrv_sdnn')
    return np.array([
        _mean(hr),   _trend(hr),  _max(hr),
        _mean(temp), _trend(temp),
        _mean(rr),   _mean(spo2),
        _trend(step), _mean(hrv),  _trend(hrv),
    ], dtype=float)


def features_frailty(rows: List[Row]) -> np.ndarray:
    step  = _get(rows, 'step_count')
    cal   = _get(rows, 'active_calories')
    sleep = _get(rows, 'sleep_hours')
    hr    = _get(rows, 'resting_hr')
    asym  = _get(rows, 'walking_asymmetry_pct')
    speed = _get(rows, 'walking_speed_ms')
    sv    = _get(rows, 'stride_variability')
    accel = _get(rows, 'accel_mag_std')
    return np.array([
        _mean(step),  _trend(step),  _cv(step),
        _mean(cal),   _trend(cal),
        _mean(sleep), _cv(sleep),
        _mean(hr),    _trend(hr),
        _mean(asym),  _mean(speed),  _mean(sv),
        _mean(accel),
    ], dtype=float)


def features_stress(rows: List[Row]) -> np.ndarray:
    hr    = _get(rows, 'resting_hr')
    hrv   = _get(rows, 'hrv_sdnn')
    rmssd = _get(rows, 'hrv_rmssd')
    temp  = _get(rows, 'wrist_temp')
    eda   = _get(rows, 'eda_mean')
    rr    = _get(rows, 'respiratory_rate')
    step  = _get(rows, 'step_count')
    sleep = _get(rows, 'sleep_hours')
    cal   = _get(rows, 'active_calories')
    return np.array([
        _mean(hr),    _trend(hr),
        _mean(hrv),   _trend(hrv),
        _mean(rmssd),
        _mean(temp),  _trend(temp),
        _mean(eda),
        _mean(rr),
        _mean(step),  _trend(step),
        _mean(sleep), _mean(cal),
    ], dtype=float)


def features_depression(rows: List[Row]) -> np.ndarray:
    step   = _get(rows, 'step_count')
    sleep  = _get(rows, 'sleep_hours')
    hr     = _get(rows, 'resting_hr')
    cal    = _get(rows, 'active_calories')
    hrv    = _get(rows, 'hrv_sdnn')
    social = _get(rows, 'social_duration')
    return np.array([
        _mean(step),   _trend(step),  _cv(step),
        _mean(sleep),  _trend(sleep), _cv(sleep),
        _mean(hr),     _trend(hr),
        _mean(cal),    _trend(cal),
        _mean(hrv),    _trend(hrv),
        _mean(social), _trend(social),
    ], dtype=float)


def features_metabolic(rows: List[Row]) -> np.ndarray:
    gluc  = _get(rows, 'glucose_mean_mgdl')
    gstd  = _get(rows, 'glucose_std_mgdl')
    gpeak = _get(rows, 'glucose_peak_mgdl')
    hr    = _get(rows, 'resting_hr')
    step  = _get(rows, 'step_count')
    cal   = _get(rows, 'active_calories')
    return np.array([
        _mean(gluc),  _trend(gluc),
        _mean(gstd),
        _mean(gpeak),
        _mean(hr),    _trend(hr),
        _mean(step),  _trend(step),
        _mean(cal),
    ], dtype=float)


def features_fall_risk(rows: List[Row]) -> np.ndarray:
    accel_m  = _get(rows, 'accel_mag_mean')
    accel_s  = _get(rows, 'accel_mag_std')
    peak_rms = _get(rows, 'accel_peak_rms')
    asym     = _get(rows, 'walking_asymmetry_pct')
    cadence  = _get(rows, 'cadence')
    sv       = _get(rows, 'stride_variability')
    return np.array([
        _mean(accel_m),   _mean(accel_s),
        _max(peak_rms),   _mean(peak_rms),
        _mean(asym),
        _mean(cadence),   _mean(sv),
        _cv(cadence),
    ], dtype=float)


def features_hypertension(rows: List[Row]) -> np.ndarray:
    sbp  = _get(rows, 'sbp_estimated')
    dbp  = _get(rows, 'dbp_estimated')
    pp   = _get(rows, 'pulse_pressure')
    hr   = _get(rows, 'resting_hr')
    hrv  = _get(rows, 'hrv_sdnn')
    return np.array([
        _mean(sbp),  _trend(sbp),  _cv(sbp),
        _mean(dbp),  _trend(dbp),
        _mean(pp),
        _mean(hr),   _cv(hr),
        _mean(hrv),  _trend(hrv),
    ], dtype=float)


def features_copd(rows: List[Row]) -> np.ndarray:
    spo2  = _get(rows, 'spo2_avg')
    smin  = _get(rows, 'spo2_min')
    sstd  = _get(rows, 'spo2_std')
    dips  = _get(rows, 'spo2_dips_below94')
    rr    = _get(rows, 'respiratory_rate')
    hr    = _get(rows, 'resting_hr')
    return np.array([
        _mean(spo2),  _min(smin),
        _mean(sstd),  _mean(dips),
        _trend(spo2),
        _mean(rr),    _trend(rr),   _cv(rr),
        _mean(hr),
    ], dtype=float)


def features_thyroid(rows: List[Row]) -> np.ndarray:
    hr    = _get(rows, 'resting_hr')
    hrv   = _get(rows, 'hrv_sdnn')
    temp  = _get(rows, 'wrist_temp')
    step  = _get(rows, 'step_count')
    sleep = _get(rows, 'sleep_hours')
    trend = _get(rows, 'resting_hr_trend')
    return np.array([
        _mean(hr),    _trend(hr),   _cv(hr),
        _mean(hrv),   _trend(hrv),
        _mean(temp),  _trend(temp),
        _mean(step),  _trend(step),
        _mean(sleep),
        _mean(trend),
    ], dtype=float)


def features_anemia(rows: List[Row]) -> np.ndarray:
    spo2 = _get(rows, 'spo2_avg')
    sstd = _get(rows, 'spo2_std')
    hr   = _get(rows, 'resting_hr')
    hrv  = _get(rows, 'hrv_sdnn')
    step = _get(rows, 'step_count')
    return np.array([
        _mean(spo2), _min(spo2),
        _mean(sstd), _trend(spo2),
        _mean(hr),   _trend(hr),   _cv(hr),
        _mean(hrv),
        _mean(step), _trend(step),
    ], dtype=float)


# ─── Public API ───────────────────────────────────────────────────────────────

FEATURE_FNS = {
    'afib':          features_afib,
    'parkinsons':    features_parkinsons,
    'sleep_apnea':   features_sleep_apnea,
    'heart_failure': features_heart_failure,
    'infection':     features_infection,
    'frailty':       features_frailty,
    'stress':        features_stress,
    'depression':    features_depression,
    'metabolic':     features_metabolic,
    'fall_risk':     features_fall_risk,
    'hypertension':  features_hypertension,
    'copd':          features_copd,
    'thyroid':       features_thyroid,
    'anemia':        features_anemia,
}

CONDITIONS = list(FEATURE_FNS.keys())


def extract(condition: str, rows: List[Row]) -> np.ndarray:
    """
    Public entry point.
    Returns feature vector for the given condition from daily_summary rows.
    Raises KeyError for unknown conditions.
    """
    fn = FEATURE_FNS[condition]
    return fn(rows).reshape(1, -1)
