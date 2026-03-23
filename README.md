# EEG Seizure Prediction — Adaptive Baseline Pipeline

A patient-specific machine learning pipeline for predicting epileptic seizures
from scalp EEG recordings. Developed as a semester project for the Knowledge-Based
Systems course at the Faculty of Automation and Computer Science,
Technical University of Cluj-Napoca (2026).

The system was implemented and tested in Google Colab using Google Drive as
the data storage backend.

---

## Problem statement

Epilepsy affects approximately 50 million people worldwide. Around 30% of patients
do not respond to drug treatment. A system that can identify the preictal state
(the period immediately before a seizure) would allow patients to take preventive
action, alert caregivers, or activate neurostimulation devices, reducing the risk
of injury.

The goal of this project is not seizure detection (identifying a seizure that is
already happening), but seizure prediction — identifying the transition from the
interictal baseline state to the preictal state before the clinical onset.

---

## Architecture overview

```
EDF file
   |
   v
Preprocessing (notch, bandpass, CAR, detrend, diff)
   |
   +-- Alpha band (8-13 Hz) --> features --> Random Forest (alpha)  --+
   |                                                                   |
   +-- Beta  band (13-30 Hz) --> features --> Random Forest (beta)  --+--> Vote --> Alarm
   |                                                                   |
   +-- Broad band (1-45 Hz)  --> features --> Random Forest (broad) --+
```

Three independent Random Forest classifiers are trained per patient, one for
each frequency band. Their output probabilities are averaged into a single vote
signal. A detection is confirmed only if the smoothed vote exceeds a threshold
and persists for a minimum duration, which substantially reduces false alarms.

---

## Signal processing pipeline

The following steps are applied to every raw EEG segment before feature extraction:

1. Notch filter at 50 Hz — removes power-line interference
2. Band-pass FIR filter 1-45 Hz (Hamming window) — removes slow drift and
   high-frequency muscle artifacts
3. Common Average Reference (CAR) — subtracts the mean of all channels from each
   channel, suppressing noise common to all electrodes
4. Linear detrending — removes residual slow baseline wander within each window
5. First-order temporal differentiation — amplifies sharp epileptic spikes while
   further suppressing slow background activity

---

## Feature vector

Each 4-second window is represented by four scalar features:

| Feature | Description |
|---|---|
| log Line Length | Mean absolute amplitude difference between consecutive samples. Rises steeply at seizure onset due to simultaneous increase in amplitude and frequency. |
| log Energy | Mean signal power across channels. |
| Focality Index | max(channel energy) / mean(channel energy). Near 1 for diffuse noise; high for spatially localised epileptic discharges. |
| Imaginary Phase Sync | Mean absolute imaginary part of the phase synchrony vector (Hilbert transform). Immune to volume conduction: instantaneous noise has Im ≈ 0; genuine neural interaction has Im > 0. |

The logarithmic transforms on Line Length and Energy linearise their
distributions and improve classifier performance.

---

## Adaptive split strategy

A key design challenge is that available recording time before a seizure varies
widely across files. The pipeline handles this adaptively:

- If more than 20 minutes are available before onset: the last 15 minutes are
  labelled preictal, and the 10 minutes before that are labelled baseline.
- If less than 20 minutes are available: the time is split in half, with the
  first half as baseline and the second half as preictal.

This avoids discarding short recordings entirely, which would reduce the
training set for patients with few seizures.

---

## Classification and voting

Each band model is a Random Forest with the following configuration:

| Parameter | Value | Rationale |
|---|---|---|
| n_estimators | 50 | Balance between speed and stability |
| max_depth | 6 | Prevents memorising noise in the baseline |
| class_weight | balanced | Compensates for the large imbalance between baseline and preictal windows (preictal windows represent ~5-6% of total data) |

Features are scaled with RobustScaler before training and testing, which
reduces the influence of outliers from movement artifacts.

The final vote is the arithmetic mean of the three band probabilities:

    vote = (p_alpha + p_beta + p_broad) / 3

A smoothed vote (5-window rolling mean) above 0.60 that persists for at
least 8 seconds (2 consecutive windows at 50% overlap) confirms a detection.

This design exploits the complementary properties of the bands:
the Alpha model is sensitive but produces false alarms during relaxation;
the Beta model is more specific but may miss subtle preictal changes;
the Broad model adds stability. Requiring agreement across bands substantially
reduces false positives.

---

## Evaluation methodology

Leave-one-seizure-out cross-validation is performed per patient. For each
seizure, the model is trained on all other available seizures from the same
patient, then tested on the held-out one.

Metrics:

- Sensitivity: proportion of seizures detected within the preictal window
- Specificity: proportion of baseline minutes without a false alarm
- FAR (False Alarm Rate): number of false alarms per hour of baseline

---

## Datasets

### Siena Scalp EEG
- 14 adult patients, 9 male and 5 female, ages 20-71
- Sampling rate: 512 Hz (downsampled to 256 Hz for compatibility)
- Equipment: EB Neuro and Natus Quantum LTM amplifiers
- Annotations follow ILAE classification (IAS, FBTC)
- Source: University of Siena, Neurology Unit

### CHB-MIT Scalp EEG
- 23 paediatric patients, 18 female and 5 male, ages 1.5-22
- Sampling rate: 256 Hz
- Collected at Children's Hospital Boston
- Standard 10-20 bipolar montage

Both datasets use EDF (European Data Format) files and require annotation CSV
files mapping each seizure to its recording file and onset time.

---

## Results

| Metric | Value |
|---|---|
| Sensitivity | 83.77% |
| Specificity | 56.20% |
| FAR | < 26 / hour |

Results were better on Siena than on CHB-MIT, consistent with the higher
sampling rate providing better spectral resolution and cleaner signal quality.

---

## Setup and usage

### Prerequisites

This project was developed and tested in Google Colab. To run it locally,
install the required packages:

```bash
pip install mne numpy pandas scipy scikit-learn
```

### Data preparation

1. Download the datasets from PhysioNet:
   - CHB-MIT: https://physionet.org/content/chbmit/1.0.0/
   - Siena:   https://physionet.org/content/siena-scalp-eeg/1.0.0/

2. Prepare two annotation CSV files with the following columns:

```
patient, test_number, seizure_id, reg_start_time, seizure_start_time
```

where `reg_start_time` and `seizure_start_time` are in `HH:MM:SS` format.

### Google Colab usage

```python
from google.colab import drive
drive.mount('/content/drive')

# Update BASE_PATH in seizure_prediction.py to match your Drive structure,
# then run:
run_pipeline('SIENA')
run_pipeline('MIT')
```

### Local usage

Update the path constants at the top of the file and run:

```bash
python seizure_prediction.py
```

---

## Known limitations

- Heavy muscle artifacts (chewing, facial movement) can produce high Line
  Length values that resemble seizure activity if not adequately filtered.
- Model performance depends on the quality of the baseline segment used for
  training. A baseline contaminated with micro-discharges or artifacts may
  reduce specificity.
- The preictal window definition is approximate. A recording where a patient
  was already agitated before seizure onset may reduce sensitivity.
- Specificity on CHB-MIT is lower than on Siena, likely due to the lower
  sampling rate and higher baseline variability in paediatric recordings.

---

## Possible future directions

- Replace Rolling Mean smoothing with an Exponential Moving Average (EMA)
  with variable adaptation speed (faster during active wakefulness, slower
  during sleep).
- Add Weighted Phase Lag Index (wPLI) as an additional feature for stronger
  noise immunity.
- Hybridise with a CNN-LSTM network to capture long-range temporal patterns
  across 30-minute preictal windows.
- Port the pipeline to an embedded system (ESP32 or Raspberry Pi) for
  real-time edge inference on a wearable EEG device.

---

## Dependencies

| Package | Purpose |
|---|---|
| mne | EDF file loading, filtering, montage handling |
| numpy | Array operations, windowing |
| pandas | Annotation loading, result aggregation |
| scipy | Detrending, Hilbert transform |
| scikit-learn | Random Forest, RobustScaler |

---

## References

1. Commission on Classification and Terminology of the ILAE, Epilepsia, 1989.
2. WHO Epilepsy Fact Sheet, 2024.
3. Eadie M.J., Expert Review of Neurotherapeutics, 2012.
4. Detti et al., Processes, 2020.
5. Detti et al., IEEE Trans. Biomed. Eng., 2019.
6. Paszkiel S., Analysis and Classification of EEG Signals for BCI, Springer, 2020.
7. Wolpaw et al., Clinical Neurophysiology, 2002.

---

## License

MIT
