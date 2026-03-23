# ==============================================================================
# EEG Seizure Prediction Pipeline - Adaptive Baseline (Zero Data Loss)
# Developed and tested in Google Colab
#
# Datasets: Siena Scalp EEG + CHB-MIT Scalp EEG
# Method:   Patient-specific Random Forest with multi-band voting
#           and adaptive preictal/baseline split
# ==============================================================================

import numpy as np
import pandas as pd
import mne
import os
import gc
import scipy.signal
from scipy.signal import hilbert
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
# from google.colab import drive   # Uncomment when running in Google Colab

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# --- Google Colab paths ---
# Uncomment the line below when running in Google Colab:
# drive.mount('/content/drive')

BASE_PATH         = '/content/drive/MyDrive/SeizurePrediction'
SIENA_DATA_PATH   = os.path.join(BASE_PATH, 'siena-scalp')
MIT_DATA_PATH     = os.path.join(BASE_PATH, 'chb-mit')
SIENA_ANNOTATIONS = os.path.join(BASE_PATH, 'annotations_siena.csv')
MIT_ANNOTATIONS   = os.path.join(BASE_PATH, 'annotations_mit.csv')

# --- Windowing ---
WINDOW_SEC  = 4     # Window length in seconds
OVERLAP_SEC = 2     # Step size (50% overlap)

# --- Alarm logic ---
VOTE_THRESHOLD   = 0.60   # Minimum average probability to trigger an alarm
PERSISTENCE_SEC  = 8      # Alarm must persist this many seconds to be confirmed
REFRACTORY_SEC   = 60     # Minimum gap between two separate false-positive events

# --- Frequency bands ---
BANDS = {
    'Alpha': [8,  13],
    'Beta':  [13, 30],
    'Broad': [1,  45],
}

mne.set_log_level('ERROR')


# ==============================================================================
# 2. DATASET ADAPTER  (Siena + CHB-MIT)
# ==============================================================================

def normalize_test_number(val):
    """
    Convert float-formatted test IDs back to strings.
    Example: '1.0' -> '1', but '4.5.6' is left unchanged.
    """
    s = str(val).strip()
    if s.endswith('.0'):
        return s[:-2]
    return s


def hms_to_seconds(time_str):
    """Parse a HH:MM:SS timestamp string into total seconds."""
    if pd.isna(time_str) or str(time_str).strip() == '-':
        return None
    try:
        t = str(time_str).strip().replace(':', '.')
        parts = t.split('.')
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return h * 3600 + m * 60 + s
    except Exception:
        return None


def get_relative_onset(reg_start, event_time):
    """
    Return the seizure onset in seconds relative to the start of the recording.
    Handles overnight recordings where the seizure timestamp wraps past midnight.
    """
    s = hms_to_seconds(reg_start)
    e = hms_to_seconds(event_time)
    if s is None or e is None:
        return None
    diff = e - s
    if diff < 0:
        diff += 24 * 3600   # overnight wrap
    return diff


class DatasetAdapter:
    """Loads and normalises annotation CSVs for both datasets into a common format."""

    @staticmethod
    def load_annotations(csv_path, dataset_type):
        print(f"[{dataset_type}] Loading annotations from: {csv_path}")
        try:
            df = pd.read_csv(csv_path, dtype=str)
            records = []

            for _, row in df.iterrows():
                try:
                    patient  = str(row['patient']).strip()
                    test_nb  = normalize_test_number(row['test_number'])
                    sz_id    = str(row['seizure_id']).strip()
                    onset    = get_relative_onset(
                        row['reg_start_time'],
                        row['seizure_start_time']
                    )
                    if onset is None:
                        continue

                    # Build full path according to dataset convention
                    if dataset_type == 'SIENA':
                        filename  = f"{patient}-{test_nb}.edf"
                        full_path = os.path.join(SIENA_DATA_PATH, patient, filename)
                    else:   # MIT
                        nb_str    = test_nb.zfill(2) if test_nb.isdigit() else test_nb
                        filename  = f"{patient}_{nb_str}.edf"
                        full_path = os.path.join(MIT_DATA_PATH, patient, filename)

                    records.append({
                        'patient':       patient,
                        'seizure_id':    sz_id,
                        'onset_rel_sec': onset,
                        'full_path':     full_path,
                        'exists':        os.path.exists(full_path),
                    })
                except Exception:
                    continue

            return pd.DataFrame(records)

        except Exception as exc:
            print(f"Error loading annotations: {exc}")
            return pd.DataFrame()


# ==============================================================================
# 3. SIGNAL PROCESSING
# ==============================================================================

def preprocess(raw):
    """
    Apply the full preprocessing chain to a raw MNE object:
      1. Notch filter at 50 Hz (power-line interference)
      2. Band-pass filter 1-45 Hz (FIR, Hamming window)
      3. Common Average Reference (CAR)
      4. Linear detrending per channel
      5. First-order temporal differentiation
    Returns a new RawArray with the processed data.
    """
    raw.notch_filter(freqs=50, fir_design='firwin', verbose=False)
    raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin', verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)

    data = raw.get_data()
    data = scipy.signal.detrend(data, axis=1, type='linear')
    # First-order difference: amplifies epileptic spikes, suppresses slow drift
    data = np.diff(data, axis=1, prepend=data[:, :1])

    return mne.io.RawArray(data, raw.info, verbose=False)


def extract_features(window):
    """
    Compute the four-element feature vector for one EEG window.

    Features:
      - log Line Length  : measures signal complexity and agitation
      - log Energy       : mean power across channels
      - Focality Index   : max(channel energy) / mean(channel energy)
                           High values indicate spatially focal activity
                           typical of seizure onset; near-1 values indicate
                           diffuse noise.
      - Imaginary Phase Sync : mean absolute imaginary part of the phase
                               synchrony vector. Immune to volume conduction
                               (instantaneous noise has Im ~ 0).
    """
    ll       = np.mean(np.abs(np.diff(window, axis=-1)))
    energy   = np.sum(window ** 2, axis=1)
    mean_en  = np.mean(energy)
    focality = np.max(energy) / (mean_en + 1e-9)

    analytic = hilbert(window, axis=1)
    z        = np.mean(np.exp(1j * np.angle(analytic)), axis=0)
    sync     = np.mean(np.abs(np.imag(z)))

    return [np.log1p(ll), np.log1p(mean_en), focality, sync]


def sliding_window_features(raw, fmin, fmax):
    """
    Filter the signal to [fmin, fmax] Hz and extract features from
    overlapping windows (size = WINDOW_SEC, step = OVERLAP_SEC).

    Returns:
      features : ndarray of shape (n_windows, 4)
      times    : ndarray of window centre times in seconds
    """
    filtered = raw.copy().filter(fmin, fmax, verbose=False)
    data     = filtered.get_data()
    sf       = filtered.info['sfreq']
    win_samp = int(WINDOW_SEC  * sf)
    stp_samp = int(OVERLAP_SEC * sf)

    features, times = [], []
    for start in range(0, data.shape[1] - win_samp, stp_samp):
        w = data[:, start : start + win_samp]
        features.append(extract_features(w))
        times.append((start + win_samp / 2) / sf)

    return np.array(features), np.array(times)


# ==============================================================================
# 4. ADAPTIVE TRAINING / TESTING PIPELINE
# ==============================================================================

def run_pipeline(dataset_name):
    """
    Main pipeline. For each patient, performs leave-one-seizure-out
    cross-validation:

      Training phase
      --------------
      For every seizure that is NOT the test seizure, extract baseline and
      preictal windows using an adaptive split strategy:
        - If recording >= 20 min before seizure: use last 15 min as preictal,
          10 min before that as baseline.
        - Otherwise: split the available time in half (first half = baseline,
          second half = preictal).
      Three independent Random Forest models are trained, one per frequency band
      (Alpha, Beta, Broad).

      Testing phase
      -------------
      The held-out seizure is evaluated. The three models vote (arithmetic mean
      of probabilities). A temporally smoothed vote above VOTE_THRESHOLD that
      persists for at least PERSISTENCE_SEC seconds triggers a detection.

      Metrics
      -------
        Sensitivity  : proportion of seizures detected in the preictal window
        Specificity  : proportion of baseline minutes without a false alarm
        FAR          : false alarms per hour of baseline
    """
    csv_path = SIENA_ANNOTATIONS if dataset_name == 'SIENA' else MIT_ANNOTATIONS
    df       = DatasetAdapter.load_annotations(csv_path, dataset_name)

    if df.empty:
        print("No data found. Check annotation CSV paths.")
        return

    patients     = df['patient'].unique()
    global_stats = []

    print(f"\nStarting {dataset_name} | {len(patients)} patients | Strategy: Adaptive Split")
    print("-" * 72)

    for patient in patients:
        p_df           = df[df['patient'] == patient]
        valid_seizures = p_df[p_df['exists'] == True]

        # Need at least 2 valid recordings: one to train on, one to test
        if len(valid_seizures) < 2:
            continue

        print(f"Patient {patient}  ({len(valid_seizures)} seizures)")
        pt_TP = pt_FP = pt_TN = pt_FN = 0
        pt_base_hours = 0.0

        for i in range(len(valid_seizures)):
            test_row  = valid_seizures.iloc[i]
            test_sz   = test_row['seizure_id']
            train_rows = valid_seizures.drop(valid_seizures.index[i])

            # ------------------------------------------------------------------
            # Training
            # ------------------------------------------------------------------
            X_train = {b: [] for b in BANDS}
            y_train = []
            trained = False

            for _, tr_row in train_rows.iterrows():
                try:
                    onset          = tr_row['onset_rel_sec']
                    available_time = onset

                    if available_time < 30:
                        # Too little data even for a minimal split
                        continue

                    if available_time > 1200:
                        # Comfortable: take 10 min baseline, 15 min preictal
                        split_point = onset - 900
                        load_start  = split_point - 600
                    else:
                        # Short recording: split available time 50/50
                        load_start  = 0
                        split_point = available_time / 2.0

                    raw = mne.io.read_raw_edf(
                        tr_row['full_path'], preload=False, verbose='ERROR'
                    )
                    raw.crop(load_start, onset).load_data().pick_types(eeg=True)
                    raw = preprocess(raw)

                    # Use Alpha times as reference to determine window labels
                    _, times    = sliding_window_features(raw, *BANDS['Alpha'])
                    abs_times   = times + load_start
                    idx_base    = abs_times < split_point
                    idx_pre     = abs_times >= split_point

                    if np.sum(idx_base) > 1 and np.sum(idx_pre) > 1:
                        trained = True
                        y_chunk = np.concatenate([
                            np.zeros(np.sum(idx_base)),
                            np.ones(np.sum(idx_pre))
                        ])
                        y_train.append(y_chunk)

                        for band in BANDS:
                            feats, _ = sliding_window_features(raw, *BANDS[band])
                            X_train[band].append(
                                np.concatenate([feats[idx_base], feats[idx_pre]])
                            )

                    del raw
                    gc.collect()

                except Exception:
                    continue

            if not trained:
                print(f"  Seizure {test_sz}: SKIP (could not train on remaining files)")
                continue

            # Fit one scaler + Random Forest per band
            models  = {}
            scalers = {}
            y_all   = np.concatenate(y_train)

            for band in BANDS:
                X_all = np.concatenate(X_train[band])
                sc    = RobustScaler()
                X_sc  = sc.fit_transform(X_all)
                clf   = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=6,
                    class_weight='balanced',
                    n_jobs=-1
                )
                clf.fit(X_sc, y_all)
                models[band]  = clf
                scalers[band] = sc

            # ------------------------------------------------------------------
            # Testing
            # ------------------------------------------------------------------
            try:
                onset_test    = test_row['onset_rel_sec']
                t_test_start  = max(0, onset_test - 2700)   # up to 45 min before

                raw_t = mne.io.read_raw_edf(
                    test_row['full_path'], preload=False, verbose='ERROR'
                )
                real_end = min(onset_test + 30, raw_t.times[-1])
                raw_t.crop(t_test_start, real_end).load_data().pick_types(eeg=True)
                raw_t = preprocess(raw_t)

                probs = {}
                tt    = None
                for band in BANDS:
                    feats, t = sliding_window_features(raw_t, *BANDS[band])
                    if tt is None:
                        tt = t
                    probs[band] = models[band].predict_proba(
                        scalers[band].transform(feats)
                    )[:, 1]

                del raw_t
                gc.collect()

                # Majority vote across bands + temporal smoothing
                vote        = (probs['Alpha'] + probs['Beta'] + probs['Broad']) / 3.0
                vote_smooth = (
                    pd.Series(vote)
                    .rolling(5, center=True)
                    .mean()
                    .fillna(0)
                    .to_numpy()
                )

                # Times relative to seizure onset (0 = onset)
                t_rel = tt - (onset_test - t_test_start)

                # Define preictal evaluation zone dynamically
                if (onset_test - t_test_start) > 900:
                    pre_thresh = -900       # last 15 min
                else:
                    pre_thresh = -(onset_test - t_test_start) / 2.0

                # --- TP / FN (preictal zone) ---
                pre_mask = (t_rel >= pre_thresh) & (t_rel <= 0)
                v_pre    = vote_smooth[pre_mask]
                detected = False
                cnt      = 0
                req      = PERSISTENCE_SEC / OVERLAP_SEC

                for v in v_pre:
                    if v > VOTE_THRESHOLD:
                        cnt += 1
                    else:
                        cnt = 0
                    if cnt >= req:
                        detected = True
                        break

                TP = 1 if detected else 0
                FN = 0 if detected else 1

                # --- FP / TN (baseline zone) ---
                base_mask = t_rel < pre_thresh
                v_base    = vote_smooth[base_mask]
                t_base    = t_rel[base_mask]
                dur_base  = int(np.sum(base_mask)) * OVERLAP_SEC

                FP = TN = 0
                if dur_base > 0:
                    tot_mins  = max(1, int(dur_base / 60.0))
                    bad_mins  = 0
                    cnt       = 0
                    last_t    = -99999

                    for k, v in enumerate(v_base):
                        cur_t = t_base[k]
                        if cur_t - last_t < REFRACTORY_SEC:
                            cnt = 0
                            continue
                        if v > VOTE_THRESHOLD:
                            cnt += 1
                        else:
                            cnt = 0
                        if cnt >= req:
                            FP       += 1
                            bad_mins += 1
                            last_t    = cur_t
                            cnt       = 0

                    TN = max(0, tot_mins - bad_mins)

                pt_TP         += TP
                pt_FN         += FN
                pt_FP         += FP
                pt_TN         += TN
                pt_base_hours += dur_base / 3600.0

                status = "DETECTED" if TP else "MISSED"
                print(f"  Seizure {test_sz}: {status}  (FP={FP}, avail={onset_test/60:.1f} min)")

                global_stats.append({
                    'Dataset':  dataset_name,
                    'Patient':  patient,
                    'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN,
                    'Hours': dur_base / 3600.0,
                })

            except Exception as exc:
                print(f"  Seizure {test_sz}: ERROR - {str(exc)[:60]}")
                continue

        # Per-patient summary
        if pt_TP + pt_FN > 0:
            sens = pt_TP / (pt_TP + pt_FN) * 100
            denom_spec = pt_TN + pt_FP
            spec = pt_TN / denom_spec * 100 if denom_spec > 0 else 0.0
            far  = pt_FP / pt_base_hours    if pt_base_hours > 0 else 0.0
            print(
                f"  Summary {patient}: "
                f"Sensitivity={sens:.1f}%  Specificity={spec:.1f}%  FAR={far:.2f}/h"
            )

    # --------------------------------------------------------------------------
    # Global report
    # --------------------------------------------------------------------------
    if not global_stats:
        print("No results to report.")
        return

    res  = pd.DataFrame(global_stats)
    G_TP = res['TP'].sum()
    G_FN = res['FN'].sum()
    G_TN = res['TN'].sum()
    G_FP = res['FP'].sum()
    G_Hr = res['Hours'].sum()

    total_pos = G_TP + G_FN
    total_neg = G_TN + G_FP

    Sens = G_TP / total_pos * 100 if total_pos > 0 else 0.0
    Spec = G_TN / total_neg * 100 if total_neg > 0 else 0.0
    FAR  = G_FP / G_Hr            if G_Hr      > 0 else 0.0

    print("\n" + "=" * 72)
    print(f"FINAL RESULTS  {dataset_name}")
    print(f"  Sensitivity : {Sens:.2f}%  ({G_TP}/{total_pos})")
    print(f"  Specificity : {Spec:.2f}%")
    print(f"  FAR         : {FAR:.2f} /h")
    print("=" * 72)

    out_csv = os.path.join(
        BASE_PATH, f"results_adaptive_{dataset_name}.csv"
    )
    res.to_csv(out_csv, index=False)
    print(f"Results saved to: {out_csv}")


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == '__main__':
    run_pipeline('SIENA')
    # run_pipeline('MIT')   # Uncomment to also run on CHB-MIT
