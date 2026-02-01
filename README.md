# PMSM Virtual Sensor: Rotor Temperature Forecasting

An end-to-end **time series forecasting system** for predicting **Permanent Magnet Surface Temperature (`pm`)**
**60 seconds ahead** using high-frequency sensor data from a **Permanent Magnet Synchronous Motor (PMSM)**.

This project demonstrates the design of an **industrial-grade virtual sensor**
for thermal monitoring, early warning, and model-driven diagnostics.

---

## Business Motivation

Direct temperature sensing on motor components is often impractical due to
cost, installation constraints, and reliability issues.

A data-driven **virtual sensor** enables:
- Early detection of overheating
- Preventive maintenance strategies
- Reduced hardware and maintenance costs
- Safer and more reliable motor operation

Operational thresholds used in this project:
- **Warning:** `pm ≥ 100°C`
- **Critical:** `pm ≥ 120°C`

---

## Dataset

**Source:**  
This dataset was collected and published by **Paderborn University (University of Paderborn, Germany)**  
and is publicly distributed via **Kaggle** under the title **Electric Motor Temperature**.

Required raw file:
```
measures_v2.csv   (~300 MB)
```

Expected location:
```
data/raw/measures_v2.csv
```

Dataset characteristics:
- Sampling frequency: **2 Hz**
- Multiple operating sessions identified via `profile_id`
- Real-world PMSM test bench measurements
- Raw data directory is intentionally **excluded from version control**

---

## Problem Definition

- Task type: **Supervised time series forecasting**
- Prediction target: `pm(t + 60s)`
- Input context window: **120 seconds** (240 timesteps)
- Forecast horizon: **60 seconds**
- Data splitting: **Session-based** (`profile_id`) to eliminate leakage

---

## Feature Scenarios

### Scenario S1 — Production Feasible
Signals typically available in embedded or production systems:
```
u_d, u_q, i_d, i_q, motor_speed, coolant, ambient
```

### Scenario S2 — Upper Bound
Extended thermal and load information:
```
S1 + torque, stator_winding, stator_tooth, stator_yoke
```

---

## Modeling Approaches

### Gradient Boosted Decision Trees (GBDT)
- Lagged and rolling statistical features
- Strong and interpretable baseline for tabular time series
- Efficient training and inference

Execution:
```
python scripts/run_train.py
```

### Deep Learning — LSTM
- Sequence-to-one forecasting architecture
- Learns temporal dynamics directly from raw sequences
- Used as an upper-bound performance reference

Execution:
```
python scripts/run_train_dl.py
```

---

## Repository Structure

```
electric-motor-temperature/
├── app/
│   └── streamlit_app.py
├── artifacts/
│   ├── models/
│   └── reports/
├── data/
│   ├── raw/          # measures_v2.csv (gitignored)
│   └── processed/
├── scripts/
│   ├── run_train.py
│   ├── run_train_dl.py
│   ├── run_backtest.py
│   ├── check_alarms.py
│   └── persistence_baseline.py
└── src/
    ├── config.py
    ├── data.py
    ├── split.py
    ├── features.py
    ├── train.py
    ├── evaluate.py
    ├── utils.py
    └── dl/
        ├── dataset.py
        ├── model.py
        └── train.py
```

---

## Reproducibility

### Environment setup
```
pip install -r requirements.txt
pip install torch
```

### Dataset download (recommended)
```
python scripts/download_dataset.py --outdir data/raw
```

Prerequisites:
- Kaggle account
- `kaggle.json` API token located at:
  - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
  - Linux / macOS: `~/.kaggle/kaggle.json`

### Model training
```
python scripts/run_train.py
```

### Interactive demo
```
streamlit run app/streamlit_app.py
```

---

## Outputs

Training produces the following artifacts:
- Trained models under `artifacts/models/`
- Evaluation reports under `artifacts/reports/`
- Interactive Streamlit-based visualization

---

## Engineering Highlights

- Explicit **forecast horizon** and windowing strategy
- Strict **data leakage prevention**
- Clear separation between **deployable** and **upper-bound** features
- Modular, extensible pipeline design

---

## Future Work

- Online / streaming inference
- Probabilistic forecasting with uncertainty bounds
- Remaining Useful Life (RUL) estimation
- Edge deployment optimization

