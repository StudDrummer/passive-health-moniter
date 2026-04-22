#!/usr/bin/env python3
"""
VIGIL Training Pipeline v2 — 14-Condition Extended Edition
===========================================================
Trains one Random Forest classifier per condition using
preprocessed JSON datasets.  Merges real data with synthetic
augmentation, applies class weighting, and produces calibrated
.joblib model files ready for vigil_inference.py.

Conditions (14):
  afib          heart_failure   infection        fall_risk
  parkinsons    sleep_apnea     frailty          metabolic
  stress        depression      hypertension     copd
  thyroid       anemia

Usage:
    python3 vigil_train_v2.py --data-dir ./data --models-dir ../vigil_ml/models
    python3 vigil_train_v2.py --condition parkinsons
    python3 vigil_train_v2.py --condition afib --data-dir ./data
"""

import os, sys, json, argparse, warnings
import numpy as np
from pathlib import Path
from typing import Optional

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
VIGIL_ML   = HERE.parent / "vigil_ml"

# ─── Feature extraction — mirrors vigil_features.py ──────────────────────────

def _safe(v, default=0.0):
    """Return float v or default if None/NaN."""
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _trend(vals):
    """Linear trend slope of a list of values."""
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return 0.0
    x = np.arange(len(v), dtype=float)
    return float(np.polyfit(x, v, 1)[0])


def _cv(vals):
    """Coefficient of variation (%)."""
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return 0.0
    m = np.mean(v)
    return float(np.std(v, ddof=1) / m * 100) if m != 0 else 0.0


def _nn(vals, n=7):
    return [x for x in vals if x is not None][:n]


# ── Per-condition feature extractors ──────────────────────────────────────────

def features_afib(rows):
    hrv  = [r.get('hrv_sdnn')    for r in rows]
    rmssd = [r.get('hrv_rmssd')  for r in rows]
    pnn  = [r.get('hrv_pnn50')   for r in rows]
    hr   = [r.get('resting_hr')  for r in rows]
    spo2 = [r.get('spo2_avg')    for r in rows]
    hrv_v = _nn(hrv)
    hr_v  = _nn(hr)
    return [
        _safe(np.mean(_nn(hrv)))  if _nn(hrv)  else 0,
        _safe(np.std(_nn(hrv)))   if _nn(hrv)  else 0,
        _cv(hrv),
        _trend(hrv),
        _safe(np.mean(_nn(rmssd))) if _nn(rmssd) else 0,
        _safe(np.mean(_nn(pnn)))   if _nn(pnn)   else 0,
        _safe(np.mean(hr_v))       if hr_v        else 0,
        _cv(hr),
        _trend(hr),
        _safe(np.mean(_nn(spo2)))  if _nn(spo2)  else 97.0,
        _safe(np.min(_nn(spo2)))   if _nn(spo2)  else 95.0,
    ]


def features_parkinsons(rows):
    asym  = [r.get('walking_asymmetry_pct')  for r in rows]
    speed = [r.get('walking_speed_ms')        for r in rows]
    sv    = [r.get('stride_variability')      for r in rows]
    arm   = [r.get('arm_swing_asymmetry')     for r in rows]
    cv    = [r.get('cadence_variability')     for r in rows]
    ds    = [r.get('double_support_pct')      for r in rows]
    sl    = [r.get('walking_step_length_m')   for r in rows]
    tr    = [r.get('tremor_amplitude')        for r in rows]
    return [
        _safe(np.mean(_nn(asym)))   if _nn(asym)  else 0,
        _trend(asym),
        _safe(np.mean(_nn(speed)))  if _nn(speed) else 0,
        _trend(speed),
        _safe(np.mean(_nn(sv)))     if _nn(sv)    else 0,
        _cv(sv),
        _safe(np.mean(_nn(arm)))    if _nn(arm)   else 0,
        _safe(np.mean(_nn(cv)))     if _nn(cv)    else 0,
        _safe(np.mean(_nn(ds)))     if _nn(ds)    else 0,
        _safe(np.mean(_nn(sl)))     if _nn(sl)    else 0,
        _safe(np.mean(_nn(tr)))     if _nn(tr)    else 0,
        _trend(tr),
    ]


def features_sleep_apnea(rows):
    spo2  = [r.get('spo2_avg')           for r in rows]
    smin  = [r.get('spo2_min')           for r in rows]
    dips  = [r.get('spo2_dips_below94')  for r in rows]
    rr    = [r.get('respiratory_rate')   for r in rows]
    hr    = [r.get('resting_hr')         for r in rows]
    sleep = [r.get('sleep_hours')        for r in rows]
    hrv   = [r.get('hrv_sdnn')           for r in rows]
    return [
        _safe(np.mean(_nn(spo2)))  if _nn(spo2) else 97.0,
        _safe(np.min(_nn(smin)))   if _nn(smin) else 95.0,
        _safe(np.mean(_nn(dips)))  if _nn(dips) else 0,
        _safe(np.max(_nn(dips)))   if _nn(dips) else 0,
        _trend(spo2),
        _safe(np.mean(_nn(rr)))    if _nn(rr)   else 14.0,
        _safe(np.mean(_nn(hr)))    if _nn(hr)   else 65.0,
        _cv(hr),
        _safe(np.mean(_nn(sleep))) if _nn(sleep) else 7.0,
        _cv(sleep),
        _safe(np.mean(_nn(hrv)))   if _nn(hrv)  else 40.0,
    ]


def features_heart_failure(rows):
    hr   = [r.get('resting_hr')         for r in rows]
    spo2 = [r.get('spo2_avg')           for r in rows]
    rr   = [r.get('respiratory_rate')   for r in rows]
    hrv  = [r.get('hrv_sdnn')           for r in rows]
    std14 = [r.get('resting_hr_std_14d') for r in rows]
    return [
        _safe(np.mean(_nn(hr)))    if _nn(hr)   else 75.0,
        _trend(hr),
        _cv(hr),
        _safe(np.mean(_nn(spo2)))  if _nn(spo2) else 97.0,
        _safe(np.min(_nn(spo2)))   if _nn(spo2) else 94.0,
        _safe(np.mean(_nn(rr)))    if _nn(rr)   else 14.0,
        _trend(rr),
        _safe(np.mean(_nn(hrv)))   if _nn(hrv)  else 40.0,
        _trend(hrv),
        _safe(np.mean(_nn(std14))) if _nn(std14) else 8.0,
    ]


def features_infection(rows):
    hr   = [r.get('resting_hr')       for r in rows]
    temp = [r.get('wrist_temp')        for r in rows]
    rr   = [r.get('respiratory_rate') for r in rows]
    spo2 = [r.get('spo2_avg')          for r in rows]
    step = [r.get('step_count')        for r in rows]
    hrv  = [r.get('hrv_sdnn')          for r in rows]
    return [
        _safe(np.mean(_nn(hr)))   if _nn(hr)   else 72.0,
        _trend(hr),
        _safe(np.max(_nn(hr)))    if _nn(hr)   else 80.0,
        _safe(np.mean(_nn(temp))) if _nn(temp) else 33.0,
        _trend(temp),
        _safe(np.mean(_nn(rr)))   if _nn(rr)   else 14.0,
        _safe(np.mean(_nn(spo2))) if _nn(spo2) else 97.0,
        _trend(step),
        _safe(np.mean(_nn(hrv)))  if _nn(hrv)  else 40.0,
        _trend(hrv),
    ]


def features_frailty(rows):
    step  = [r.get('step_count')            for r in rows]
    cal   = [r.get('active_calories')       for r in rows]
    sleep = [r.get('sleep_hours')           for r in rows]
    hr    = [r.get('resting_hr')            for r in rows]
    asym  = [r.get('walking_asymmetry_pct') for r in rows]
    speed = [r.get('walking_speed_ms')      for r in rows]
    sv    = [r.get('stride_variability')    for r in rows]
    accel = [r.get('accel_mag_std')         for r in rows]
    return [
        _safe(np.mean(_nn(step)))  if _nn(step)  else 5000.0,
        _trend(step),
        _cv(step),
        _safe(np.mean(_nn(cal)))   if _nn(cal)   else 250.0,
        _trend(cal),
        _safe(np.mean(_nn(sleep))) if _nn(sleep) else 7.0,
        _cv(sleep),
        _safe(np.mean(_nn(hr)))    if _nn(hr)    else 70.0,
        _trend(hr),
        _safe(np.mean(_nn(asym)))  if _nn(asym)  else 5.0,
        _safe(np.mean(_nn(speed))) if _nn(speed) else 1.2,
        _safe(np.mean(_nn(sv)))    if _nn(sv)    else 2.0,
        _safe(np.mean(_nn(accel))) if _nn(accel) else 0.5,
    ]


def features_stress(rows):
    hr    = [r.get('resting_hr')       for r in rows]
    hrv   = [r.get('hrv_sdnn')         for r in rows]
    rmssd = [r.get('hrv_rmssd')        for r in rows]
    temp  = [r.get('wrist_temp')       for r in rows]
    eda   = [r.get('eda_mean')         for r in rows]
    rr    = [r.get('respiratory_rate') for r in rows]
    step  = [r.get('step_count')       for r in rows]
    sleep = [r.get('sleep_hours')      for r in rows]
    cal   = [r.get('active_calories')  for r in rows]
    return [
        _safe(np.mean(_nn(hr)))    if _nn(hr)    else 72.0,
        _trend(hr),
        _safe(np.mean(_nn(hrv)))   if _nn(hrv)   else 40.0,
        _trend(hrv),
        _safe(np.mean(_nn(rmssd))) if _nn(rmssd) else 35.0,
        _safe(np.mean(_nn(temp)))  if _nn(temp)  else 33.0,
        _trend(temp),
        _safe(np.mean(_nn(eda)))   if _nn(eda)   else 2.0,
        _safe(np.mean(_nn(rr)))    if _nn(rr)    else 14.0,
        _safe(np.mean(_nn(step)))  if _nn(step)  else 6000.0,
        _trend(step),
        _safe(np.mean(_nn(sleep))) if _nn(sleep) else 7.0,
        _safe(np.mean(_nn(cal)))   if _nn(cal)   else 300.0,
    ]


def features_depression(rows):
    step   = [r.get('step_count')       for r in rows]
    sleep  = [r.get('sleep_hours')      for r in rows]
    hr     = [r.get('resting_hr')       for r in rows]
    cal    = [r.get('active_calories')  for r in rows]
    hrv    = [r.get('hrv_sdnn')         for r in rows]
    social = [r.get('social_duration')  for r in rows]
    return [
        _safe(np.mean(_nn(step)))   if _nn(step)   else 5000.0,
        _trend(step),
        _cv(step),
        _safe(np.mean(_nn(sleep)))  if _nn(sleep)  else 7.0,
        _trend(sleep),
        _cv(sleep),
        _safe(np.mean(_nn(hr)))     if _nn(hr)     else 70.0,
        _trend(hr),
        _safe(np.mean(_nn(cal)))    if _nn(cal)    else 300.0,
        _trend(cal),
        _safe(np.mean(_nn(hrv)))    if _nn(hrv)    else 40.0,
        _trend(hrv),
        _safe(np.mean(_nn(social))) if _nn(social) else 2.5,
        _trend(social),
    ]


def features_metabolic(rows):
    gluc  = [r.get('glucose_mean_mgdl')  for r in rows]
    gstd  = [r.get('glucose_std_mgdl')   for r in rows]
    gpeak = [r.get('glucose_peak_mgdl')  for r in rows]
    hr    = [r.get('resting_hr')          for r in rows]
    step  = [r.get('step_count')          for r in rows]
    cal   = [r.get('active_calories')     for r in rows]
    return [
        _safe(np.mean(_nn(gluc)))  if _nn(gluc)  else 90.0,
        _trend(gluc),
        _safe(np.mean(_nn(gstd)))  if _nn(gstd)  else 10.0,
        _safe(np.mean(_nn(gpeak))) if _nn(gpeak) else 120.0,
        _safe(np.mean(_nn(hr)))    if _nn(hr)    else 72.0,
        _trend(hr),
        _safe(np.mean(_nn(step)))  if _nn(step)  else 6000.0,
        _trend(step),
        _safe(np.mean(_nn(cal)))   if _nn(cal)   else 300.0,
    ]


def features_fall_risk(rows):
    accel_m  = [r.get('accel_mag_mean')      for r in rows]
    accel_s  = [r.get('accel_mag_std')       for r in rows]
    peak_rms = [r.get('accel_peak_rms')      for r in rows]
    asym     = [r.get('walking_asymmetry_pct') for r in rows]
    cadence  = [r.get('cadence')              for r in rows]
    sv       = [r.get('stride_variability')   for r in rows]
    return [
        _safe(np.mean(_nn(accel_m)))  if _nn(accel_m)  else 1.0,
        _safe(np.mean(_nn(accel_s)))  if _nn(accel_s)  else 0.3,
        _safe(np.max(_nn(peak_rms)))  if _nn(peak_rms) else 2.0,
        _safe(np.mean(_nn(peak_rms))) if _nn(peak_rms) else 1.5,
        _safe(np.mean(_nn(asym)))     if _nn(asym)     else 5.0,
        _safe(np.mean(_nn(cadence)))  if _nn(cadence)  else 100.0,
        _safe(np.mean(_nn(sv)))       if _nn(sv)       else 2.0,
        _cv(cadence),
    ]


def features_hypertension(rows):
    sbp  = [r.get('sbp_estimated')     for r in rows]
    dbp  = [r.get('dbp_estimated')     for r in rows]
    pp   = [r.get('pulse_pressure')    for r in rows]
    hr   = [r.get('resting_hr')        for r in rows]
    hrv  = [r.get('hrv_sdnn')          for r in rows]
    return [
        _safe(np.mean(_nn(sbp)))  if _nn(sbp) else 120.0,
        _trend(sbp),
        _cv(sbp),
        _safe(np.mean(_nn(dbp)))  if _nn(dbp) else 80.0,
        _trend(dbp),
        _safe(np.mean(_nn(pp)))   if _nn(pp)  else 40.0,
        _safe(np.mean(_nn(hr)))   if _nn(hr)  else 72.0,
        _cv(hr),
        _safe(np.mean(_nn(hrv)))  if _nn(hrv) else 40.0,
        _trend(hrv),
    ]


def features_copd(rows):
    spo2  = [r.get('spo2_avg')          for r in rows]
    smin  = [r.get('spo2_min')          for r in rows]
    sstd  = [r.get('spo2_std')          for r in rows]
    dips  = [r.get('spo2_dips_below94') for r in rows]
    rr    = [r.get('respiratory_rate')  for r in rows]
    hr    = [r.get('resting_hr')        for r in rows]
    return [
        _safe(np.mean(_nn(spo2)))  if _nn(spo2) else 97.0,
        _safe(np.min(_nn(smin)))   if _nn(smin) else 94.0,
        _safe(np.mean(_nn(sstd)))  if _nn(sstd) else 1.5,
        _safe(np.mean(_nn(dips)))  if _nn(dips) else 0.0,
        _trend(spo2),
        _safe(np.mean(_nn(rr)))    if _nn(rr)   else 14.0,
        _trend(rr),
        _cv(rr),
        _safe(np.mean(_nn(hr)))    if _nn(hr)   else 72.0,
    ]


def features_thyroid(rows):
    hr    = [r.get('resting_hr')        for r in rows]
    hrv   = [r.get('hrv_sdnn')          for r in rows]
    temp  = [r.get('wrist_temp')        for r in rows]
    step  = [r.get('step_count')        for r in rows]
    sleep = [r.get('sleep_hours')       for r in rows]
    trend = [r.get('resting_hr_trend')  for r in rows]
    return [
        _safe(np.mean(_nn(hr)))    if _nn(hr)    else 72.0,
        _trend(hr),
        _cv(hr),
        _safe(np.mean(_nn(hrv)))   if _nn(hrv)   else 40.0,
        _trend(hrv),
        _safe(np.mean(_nn(temp)))  if _nn(temp)  else 33.0,
        _trend(temp),
        _safe(np.mean(_nn(step)))  if _nn(step)  else 6000.0,
        _trend(step),
        _safe(np.mean(_nn(sleep))) if _nn(sleep) else 7.0,
        _safe(np.mean(_nn(trend))) if _nn(trend) else 0.0,
    ]


def features_anemia(rows):
    spo2 = [r.get('spo2_avg')    for r in rows]
    sstd = [r.get('spo2_std')    for r in rows]
    hr   = [r.get('resting_hr')  for r in rows]
    hrv  = [r.get('hrv_sdnn')    for r in rows]
    step = [r.get('step_count')  for r in rows]
    return [
        _safe(np.mean(_nn(spo2))) if _nn(spo2) else 97.0,
        _safe(np.min(_nn(spo2)))  if _nn(spo2) else 95.0,
        _safe(np.mean(_nn(sstd))) if _nn(sstd) else 1.5,
        _trend(spo2),
        _safe(np.mean(_nn(hr)))   if _nn(hr)   else 75.0,
        _trend(hr),
        _cv(hr),
        _safe(np.mean(_nn(hrv)))  if _nn(hrv)  else 40.0,
        _safe(np.mean(_nn(step))) if _nn(step) else 5000.0,
        _trend(step),
    ]


# ─── Condition registry ───────────────────────────────────────────────────────

CONDITION_CONFIG = {
    "afib": {
        "label":         "Atrial Fibrillation Risk",
        "feature_fn":    features_afib,
        "data_files":    ["preprocessed_afib.json"],
        "synth_positive": {
            "hrv_sdnn": 12, "hrv_rmssd": 8, "hrv_pnn50": 2,
            "resting_hr": 95, "spo2_avg": 95.5, "spo2_min": 93.0,
        },
        "synth_negative": {
            "hrv_sdnn": 45, "hrv_rmssd": 38, "hrv_pnn50": 20,
            "resting_hr": 68, "spo2_avg": 98.0, "spo2_min": 96.5,
        },
        "published_auc": 0.97,
    },
    "parkinsons": {
        "label":         "Parkinson's / Movement Risk",
        "feature_fn":    features_parkinsons,
        "data_files":    ["preprocessed_parkinsons_pads.json",
                          "preprocessed_parkinsons_gait.json"],
        "synth_positive": {
            "walking_asymmetry_pct": 13, "walking_speed_ms": 0.9,
            "stride_variability": 7, "arm_swing_asymmetry": 22,
            "cadence_variability": 8, "double_support_pct": 25,
            "walking_step_length_m": 0.57, "tremor_amplitude": 0.12,
        },
        "synth_negative": {
            "walking_asymmetry_pct": 4, "walking_speed_ms": 1.3,
            "stride_variability": 2, "arm_swing_asymmetry": 5,
            "cadence_variability": 2.8, "double_support_pct": 18,
            "walking_step_length_m": 0.72, "tremor_amplitude": 0.02,
        },
        "published_auc": 0.96,
    },
    "sleep_apnea": {
        "label":         "Sleep Apnea Risk",
        "feature_fn":    features_sleep_apnea,
        "data_files":    ["preprocessed_sleep_apnea.json",
                          "preprocessed_sleep_apnea_dreamt.json",
                          "preprocessed_sleep_apnea_ucddb.json"],
        "synth_positive": {
            "spo2_avg": 91, "spo2_min": 84, "spo2_dips_below94": 35,
            "respiratory_rate": 19, "resting_hr": 72, "sleep_hours": 8.5,
            "hrv_sdnn": 28,
        },
        "synth_negative": {
            "spo2_avg": 97.5, "spo2_min": 95, "spo2_dips_below94": 1,
            "respiratory_rate": 13, "resting_hr": 62, "sleep_hours": 7.0,
            "hrv_sdnn": 50,
        },
        "published_auc": 0.94,
    },
    "heart_failure": {
        "label":         "Cardiac Decompensation Risk",
        "feature_fn":    features_heart_failure,
        "data_files":    ["preprocessed_heart_failure.json"],
        "synth_positive": {
            "resting_hr": 88, "spo2_avg": 93, "spo2_min": 90,
            "respiratory_rate": 21, "hrv_sdnn": 18, "resting_hr_std_14d": 14,
        },
        "synth_negative": {
            "resting_hr": 65, "spo2_avg": 98, "spo2_min": 96,
            "respiratory_rate": 13, "hrv_sdnn": 52, "resting_hr_std_14d": 5,
        },
        "published_auc": 0.90,
    },
    "infection": {
        "label":         "Acute Infection Signal",
        "feature_fn":    features_infection,
        "data_files":    ["preprocessed_infection_capno.json"],
        "synth_positive": {
            "resting_hr": 98, "wrist_temp": 34.8, "respiratory_rate": 21,
            "spo2_avg": 96, "step_count": 1800, "hrv_sdnn": 22,
        },
        "synth_negative": {
            "resting_hr": 65, "wrist_temp": 33.2, "respiratory_rate": 13,
            "spo2_avg": 98.5, "step_count": 7500, "hrv_sdnn": 50,
        },
        "published_auc": 0.82,
    },
    "frailty": {
        "label":         "Frailty / Low Fitness",
        "feature_fn":    features_frailty,
        "data_files":    ["preprocessed_frailty_sisfall.json",
                          "preprocessed_parkinsons_gait.json"],
        "synth_positive": {
            "step_count": 2200, "active_calories": 110, "sleep_hours": 9.0,
            "resting_hr": 78, "walking_asymmetry_pct": 12, "walking_speed_ms": 0.85,
            "stride_variability": 5, "accel_mag_std": 0.18,
        },
        "synth_negative": {
            "step_count": 9500, "active_calories": 420, "sleep_hours": 7.2,
            "resting_hr": 62, "walking_asymmetry_pct": 4, "walking_speed_ms": 1.35,
            "stride_variability": 1.8, "accel_mag_std": 0.45,
        },
        "published_auc": 0.88,
    },
    "stress": {
        "label":         "Chronic Stress / Autonomic Overload",
        "feature_fn":    features_stress,
        "data_files":    ["preprocessed_stress.json",
                          "preprocessed_stress_studentlife.json"],
        "synth_positive": {
            "resting_hr": 82, "hrv_sdnn": 22, "hrv_rmssd": 18,
            "wrist_temp": 34.2, "eda_mean": 3.5, "respiratory_rate": 19,
            "step_count": 3000, "sleep_hours": 5.5, "active_calories": 150,
        },
        "synth_negative": {
            "resting_hr": 62, "hrv_sdnn": 52, "hrv_rmssd": 45,
            "wrist_temp": 33.0, "eda_mean": 1.8, "respiratory_rate": 13,
            "step_count": 8000, "sleep_hours": 7.5, "active_calories": 380,
        },
        "published_auc": 0.93,
    },
    "depression": {
        "label":         "Depression / MDD Pattern",
        "feature_fn":    features_depression,
        "data_files":    ["preprocessed_depression.json",
                          "preprocessed_depression_wesad.json",
                          "preprocessed_depression_globem.json",
                          "preprocessed_depression_studentlife.json"],
        "synth_positive": {
            "step_count": 2500, "sleep_hours": 9.5, "resting_hr": 76,
            "active_calories": 120, "hrv_sdnn": 30, "social_duration": 0.8,
        },
        "synth_negative": {
            "step_count": 9000, "sleep_hours": 7.0, "resting_hr": 63,
            "active_calories": 380, "hrv_sdnn": 55, "social_duration": 3.5,
        },
        "published_auc": 0.78,
    },
    "metabolic": {
        "label":         "Metabolic / Insulin Resistance Risk",
        "feature_fn":    features_metabolic,
        "data_files":    ["preprocessed_metabolic.json"],
        "synth_positive": {
            "glucose_mean_mgdl": 155, "glucose_std_mgdl": 38,
            "glucose_peak_mgdl": 210, "resting_hr": 78,
            "step_count": 3800, "active_calories": 180,
        },
        "synth_negative": {
            "glucose_mean_mgdl": 88, "glucose_std_mgdl": 12,
            "glucose_peak_mgdl": 115, "resting_hr": 64,
            "step_count": 9000, "active_calories": 400,
        },
        "published_auc": 0.81,
    },
    "fall_risk": {
        "label":         "Fall Risk",
        "feature_fn":    features_fall_risk,
        "data_files":    ["preprocessed_fall_risk.json"],
        "synth_positive": {
            "accel_mag_mean": 1.1, "accel_mag_std": 0.55, "accel_peak_rms": 8.5,
            "walking_asymmetry_pct": 14, "cadence": 72, "stride_variability": 6,
        },
        "synth_negative": {
            "accel_mag_mean": 0.95, "accel_mag_std": 0.25, "accel_peak_rms": 2.1,
            "walking_asymmetry_pct": 4, "cadence": 105, "stride_variability": 2,
        },
        "published_auc": 0.96,
    },
    "hypertension": {
        "label":         "Hypertension Risk",
        "feature_fn":    features_hypertension,
        "data_files":    ["preprocessed_hypertension.json"],
        "synth_positive": {
            "sbp_estimated": 148, "dbp_estimated": 94, "pulse_pressure": 54,
            "resting_hr": 80, "hrv_sdnn": 24,
        },
        "synth_negative": {
            "sbp_estimated": 115, "dbp_estimated": 72, "pulse_pressure": 43,
            "resting_hr": 64, "hrv_sdnn": 50,
        },
        "published_auc": 0.88,
    },
    "copd": {
        "label":         "COPD / Respiratory Disease",
        "feature_fn":    features_copd,
        "data_files":    ["preprocessed_copd.json",
                          "preprocessed_copd_bidmc.json",
                          "preprocessed_copd_capno.json"],
        "synth_positive": {
            "spo2_avg": 91, "spo2_min": 85, "spo2_std": 4.0,
            "spo2_dips_below94": 28, "respiratory_rate": 22, "resting_hr": 85,
        },
        "synth_negative": {
            "spo2_avg": 97.5, "spo2_min": 95, "spo2_std": 1.2,
            "spo2_dips_below94": 1, "respiratory_rate": 13, "resting_hr": 65,
        },
        "published_auc": 0.87,
    },
    "thyroid": {
        "label":         "Thyroid Dysfunction",
        "feature_fn":    features_thyroid,
        "data_files":    ["preprocessed_thyroid.json"],
        "synth_positive": {
            "resting_hr": 98, "hrv_sdnn": 20, "wrist_temp": 34.5,
            "step_count": 3500, "sleep_hours": 9.0, "resting_hr_trend": 0.2,
        },
        "synth_negative": {
            "resting_hr": 68, "hrv_sdnn": 48, "wrist_temp": 33.1,
            "step_count": 8000, "sleep_hours": 7.0, "resting_hr_trend": 0.0,
        },
        "published_auc": 0.82,
    },
    "anemia": {
        "label":         "Anemia / Low Perfusion",
        "feature_fn":    features_anemia,
        "data_files":    ["preprocessed_anemia.json",
                          "preprocessed_anemia_bidmc.json",
                          "preprocessed_anemia_mimic.json"],
        "synth_positive": {
            "spo2_avg": 93.0, "spo2_min": 90.0, "spo2_std": 1.8,
            "resting_hr": 85, "hrv_sdnn": 25, "step_count": 2800,
        },
        "synth_negative": {
            "spo2_avg": 98.0, "spo2_min": 96.0, "spo2_std": 0.9,
            "resting_hr": 65, "hrv_sdnn": 52, "step_count": 8000,
        },
        "published_auc": 0.84,
    },
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_real_data(data_dir: Path, data_files: list, feature_fn):
    """Load preprocessed JSON files → (X, y)."""
    X, y = [], []
    loaded_files = 0
    for fname in data_files:
        fp = data_dir / fname
        if not fp.exists():
            continue
        try:
            records = json.load(open(fp))
            for rec in records:
                rows  = rec.get('rows', [])
                label = rec.get('label', -1)
                if label not in (0, 1) or len(rows) < 3:
                    continue
                feats = feature_fn(rows)
                if any(not np.isfinite(f) for f in feats):
                    continue
                X.append(feats)
                y.append(int(label))
            loaded_files += 1
        except Exception as e:
            print(f"    WARN loading {fname}: {e}")
    return np.array(X, dtype=float), np.array(y, dtype=int), loaded_files


def generate_synthetic(pos_template: dict, neg_template: dict,
                        feature_fn, n_each: int = 600, seed: int = 42):
    """Generate synthetic records from per-condition templates."""
    rng = np.random.default_rng(seed)
    X, y = [], []

    def _make_rows(template, n_days=14, noise=0.12):
        rows = []
        for d in range(n_days):
            row = {"date": f"day_{d}"}
            for k, v in template.items():
                row[k] = max(0.0, float(v) * (1 + rng.normal(0, noise)))
            rows.append(row)
        return rows

    for label, tmpl in [(1, pos_template), (0, neg_template)]:
        for _ in range(n_each):
            rows  = _make_rows(tmpl, n_days=14, noise=0.15)
            feats = feature_fn(rows)
            if any(not np.isfinite(f) for f in feats):
                continue
            X.append(feats)
            y.append(label)
    return np.array(X, dtype=float), np.array(y, dtype=int)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_condition(condition: str, data_dir: Path, models_dir: Path):
    """Train one Random Forest model for a condition. Returns metrics dict."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib

    cfg = CONDITION_CONFIG.get(condition)
    if not cfg:
        print(f"  Unknown condition: {condition}")
        return None

    print(f"\n{'─'*60}")
    print(f"  Training: {cfg['label']}")
    print(f"{'─'*60}")

    # ── Load real data ────────────────────────────────────────────────────────
    X_real, y_real, n_files = load_real_data(
        data_dir, cfg['data_files'], cfg['feature_fn'])
    n_real = len(X_real)

    # ── Synthetic augmentation ────────────────────────────────────────────────
    # More synthetic if real data is scarce
    n_synth = max(400, 1200 - n_real)
    X_syn, y_syn = generate_synthetic(
        cfg['synth_positive'], cfg['synth_negative'],
        cfg['feature_fn'], n_each=n_synth // 2)

    # ── Combine ───────────────────────────────────────────────────────────────
    if n_real > 0:
        X = np.vstack([X_real, X_syn])
        y = np.concatenate([y_real, y_syn])
    else:
        print(f"  No real data found — using synthetic only")
        X, y = X_syn, y_syn

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    print(f"  Real data:      {n_real} samples from {n_files} file(s)")
    print(f"  Synthetic data: {len(X_syn)} samples")
    print(f"  Total:          {len(X)} | {n_pos} pos ({100*n_pos/len(X):.0f}%)")

    # ── Class weights ─────────────────────────────────────────────────────────
    ratio = n_neg / max(n_pos, 1)
    sample_weights = np.where(y == 1, ratio, 1.0)
    mean_w = float(np.mean(sample_weights))
    print(f"  Class weight ratio: {ratio:.2f}  mean_weight: {mean_w:.2f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=4,
        min_samples_split=6,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    rf),
    ])
    calibrated = CalibratedClassifierCV(pipeline, cv=5, method='isotonic')

    # ── Cross-validation ──────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(calibrated, X, y,
                              scoring='roc_auc', cv=cv,
                              fit_params={'sample_weight': sample_weights})
    cv_ap  = cross_val_score(calibrated, X, y,
                              scoring='average_precision', cv=cv,
                              fit_params={'sample_weight': sample_weights})

    print(f"  CV ROC-AUC:   {np.mean(cv_auc):.3f} ± {np.std(cv_auc):.3f}")
    print(f"  CV Avg Prec:  {np.mean(cv_ap):.3f}  ± {np.std(cv_ap):.3f}")
    print(f"  Published AUC:{cfg['published_auc']:.2f}")

    # ── Final fit ─────────────────────────────────────────────────────────────
    calibrated.fit(X, y, sample_weight=sample_weights)

    # ── Feature importances ───────────────────────────────────────────────────
    try:
        importances = (calibrated.estimator.named_steps['clf']
                       .feature_importances_)
        n_feat = len(importances)
        # Unnamed features
        feat_names = [f"feat_{i}" for i in range(n_feat)]
        top5 = np.argsort(importances)[::-1][:5]
        print(f"  Top 5 features:")
        for i in top5:
            print(f"    {feat_names[i]:35s} {importances[i]:.3f}")
    except Exception:
        pass

    # ── Save model ────────────────────────────────────────────────────────────
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / f"{condition}_model.joblib"
    joblib.dump(calibrated, out_path)
    print(f"  Saved: {out_path}")

    return {
        "condition":     condition,
        "label":         cfg['label'],
        "n_samples":     len(X),
        "n_real":        n_real,
        "cv_auc_mean":   float(np.mean(cv_auc)),
        "cv_auc_std":    float(np.std(cv_auc)),
        "cv_ap_mean":    float(np.mean(cv_ap)),
        "published_auc": cfg['published_auc'],
        "model_path":    str(out_path),
    }


# ─── Report ───────────────────────────────────────────────────────────────────

def write_report(results: list, out_path: Path):
    lines = [
        "VIGIL Training Report v2",
        "=" * 70,
        "",
    ]
    for r in results:
        if r is None:
            continue
        lines += [
            f"Condition:      {r['condition']}",
            f"  Label:        {r['label']}",
            f"  Samples:      {r['n_samples']} ({r['n_real']} real)",
            f"  CV AUC:       {r['cv_auc_mean']:.3f} ± {r['cv_auc_std']:.3f}",
            f"  CV AvgPrec:   {r['cv_ap_mean']:.3f}",
            f"  Published AUC:{r['published_auc']:.2f}",
            f"  Model:        {r['model_path']}",
            "",
        ]
    lines.append(f"\nTotal conditions trained: {sum(1 for r in results if r)}")
    report_text = "\n".join(lines)
    out_path.write_text(report_text)
    print(f"\n  Report → {out_path}")
    print(report_text)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIGIL Training Pipeline v2")
    parser.add_argument("--condition", type=str, default=None,
                        help="Train single condition (default: all)")
    parser.add_argument("--data-dir",  type=Path,
                        default=HERE / "data",
                        help="Path to preprocessed JSON files")
    parser.add_argument("--models-dir", type=Path,
                        default=HERE.parent / "vigil_ml" / "models",
                        help="Output directory for .joblib models")
    parser.add_argument("--report-dir", type=Path, default=None,
                        help="Directory for training report (default: models-dir)")
    args = parser.parse_args()

    try:
        import sklearn, joblib
    except ImportError:
        print("ERROR: scikit-learn and joblib required.")
        print("  pip3 install scikit-learn joblib --break-system-packages")
        sys.exit(1)

    report_dir = args.report_dir or args.models_dir
    conditions = ([args.condition] if args.condition
                  else list(CONDITION_CONFIG.keys()))

    print(f"\nVIGIL Training Pipeline v2")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Models dir: {args.models_dir}")
    print(f"  Conditions: {conditions}\n")

    results = []
    for cond in conditions:
        r = train_condition(cond, args.data_dir, args.models_dir)
        results.append(r)

    write_report(results, report_dir / "training_report_v2.txt")
    print("\nDone.")
