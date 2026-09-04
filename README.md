# ESP Well Failure Prediction — 30-Day Early Warning System

Random Forest classifier that predicts Electrical Submersible Pump (ESP) failures up to 30 days in advance from sensor and production data, developed for a graduate data analytics course project.

## Problem

ESP failures cause costly unplanned shutdowns in oil wells. Reactive maintenance means expensive downtime and lost production, and harsh downhole environments (sand, heat) can reduce ESP life to roughly two years. Rule-based SCADA alarms rely on static thresholds that trigger too late or generate false positives. The goal was to predict failure probability per well, up to 30 days ahead, and translate it into an actionable risk tier.

## Data

- **Source:** SPE Ecuador Section E-Challenge Machine Learning Contest dataset — historical ESP sensor and production data from deepwater wells.
- **Scale:** 94 labeled training wells, 15 held-out test wells; high-frequency intra-day sensor readings plus daily production reports.
- **Inputs:** Vibration, intake pressure, motor temperature, current, frequency, BOPD, BFPD.
- **Target:** Binary — will this well fail within the next 30 days?
- **Challenge:** Extreme class imbalance (failures are ~0.2% of rows).

## Approach

1. **Temporal alignment** — a custom structurer resamples production data to daily cadence, forward-fills gaps, and matches each production day to the nearest sensor snapshot (±1 day).
2. **Downsampling** — multiple intra-day sensor readings averaged to one row per well per day.
3. **Feature engineering** — outlier clipping at the 1st–99th percentile per well, median imputation, and engineered ratio features (BOPD/FREQ×60, BFPD/FREQ×60, PRESS_INT/PRESS_DESC).
4. **Label engineering** — a 7-day pre-failure window flagged as "failure approaching," rolled up into a 30-day lookahead target.
5. **Modeling** — Random Forest (100 trees, depth tuned via k-fold cross-validation, class_weight="balanced"), with SMOTE oversampling applied to the training set only to address class imbalance (test set left untouched, no data leakage).
6. **Deployment output** — a 30-day cumulative failure probability per well, rolled into a 5-day alert window and mapped to CRITICAL / HIGH / MEDIUM / LOW risk tiers.

## Results

| Metric | Value |
|---|---|
| Precision (failures) | 64% |
| Recall (failures) | 91% |
| F1 | 0.75 |
| AUC | 0.992 |

The model correctly caught 91% of real failures with only 78 false alarms out of 4,014 normal well-days across the 15 held-out test wells. Vibration was the strongest predictor, followed by intake pressure and motor current, which together accounted for 31% of model decisions; the top three features alone explained 52% of model behavior.

## Repository Contents

- `Data_Analytics_Project_ESP.ipynb` — full data pipeline, feature engineering, model training, and evaluation.
- `ESP_Failure_Prediction_Presentation.pptx` — final project presentation and results summary.
- `Project_Proposal.docx` — original project proposal and methodology.

Raw high-frequency sensor archives (multiple GB) are not included in this repository; see the SPE E-Challenge dataset and competition materials for source data.

## Team

Regnold Chinowaita & Diego De la Cruz Torres

## Future Work

Incorporate real-time streaming data and benchmark against XGBoost and LSTM architectures.
