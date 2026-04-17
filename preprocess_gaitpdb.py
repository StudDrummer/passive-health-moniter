# preprocess_gaitpdb.py — run once, produces preprocessed_parkinsons.json
import wfdb, numpy as np, json, os, glob

DATA_DIR = os.path.expanduser("~/passive-health-moniter/datasets/gait_pd/")
OUT_FILE = os.path.expanduser("~/passive-health-moniter/vigil_ml/data/parkinsons_real.json")

records = glob.glob(os.path.join(DATA_DIR, "**/*.hea"), recursive=True)
dataset = []

for rec_path in records:
    rec_name = rec_path.replace('.hea','')
    try:
        record = wfdb.rdrecord(rec_name)
        ann    = wfdb.rdann(rec_name, 'atr') if os.path.exists(rec_name+'.atr') else None
        
        # GaitPDB has: left force, right force, left ankle, right ankle accel
        # We extract stride timing from the force plate data
        fs = record.fs
        sig = record.p_signal  # shape (n_samples, n_channels)
        
        # Stride detection from left foot force (channel 0)
        # A stride = time between consecutive heel strikes (local maxima)
        from scipy.signal import find_peaks
        left_force  = sig[:, 0]
        right_force = sig[:, 1] if sig.shape[1] > 1 else left_force
        
        left_peaks,  _ = find_peaks(left_force,  height=0.3*left_force.max(),  distance=int(0.3*fs))
        right_peaks, _ = find_peaks(right_force, height=0.3*right_force.max(), distance=int(0.3*fs))
        
        if len(left_peaks) < 4 or len(right_peaks) < 4:
            continue
            
        left_intervals  = np.diff(left_peaks)  / fs  # seconds between strides
        right_intervals = np.diff(right_peaks) / fs
        
        # Asymmetry: % difference in mean stride time L vs R
        asym = abs(np.mean(left_intervals) - np.mean(right_intervals)) / \
               np.mean([np.mean(left_intervals), np.mean(right_intervals)]) * 100
        
        # Stride variability: coefficient of variation
        all_intervals = np.concatenate([left_intervals, right_intervals])
        stride_var = (np.std(all_intervals) / np.mean(all_intervals)) * 100
        
        # Walking speed: use cadence approximation (need calibration for real speed)
        cadence = 60 / np.mean(all_intervals)  # steps per minute
        # Typical relationship: speed ≈ cadence × step_length / 60
        # Approximate step length from stride interval for now
        speed_approx = cadence * 0.70 / 60  # rough m/s estimate
        
        # PD label: filename prefix 'Pt' = patient, 'Co' = control, 'Pd' = PD
        label_char = os.path.basename(rec_name)[:2]
        is_pd = 1 if label_char in ['Pd', 'pt', 'Si'] else 0
        
        # Build a pseudo-daily-summary row (one "day" = one walking bout)
        row = {
            "date": rec_name,
            "walking_asymmetry_pct": float(asym),
            "walking_speed_ms":      float(speed_approx),
            "stride_variability":    float(stride_var),
            "cadence_variability":   float((np.std(np.diff(left_peaks)/fs) / np.mean(np.diff(left_peaks)/fs)) * 100),
            "double_support_pct":    20.0,  # not directly in this dataset
            "walking_step_length_m": speed_approx * 60 / cadence if cadence > 0 else 0.7,
        }
        dataset.append({"rows": [row] * 14, "label": is_pd})
        
    except Exception as e:
        print(f"Skipping {rec_name}: {e}")
        continue

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
json.dump(dataset, open(OUT_FILE, 'w'))
print(f"Preprocessed {len(dataset)} records → {OUT_FILE}")