
# PMSM Virtual Sensor: Rotor Temperature Forecasting
=======
<img src="assets/demo.gif" width="900" alt="Streamlit demo preview" />
>>>>>>> 5f2013f (Update README and add demo asset)

# PMSM Virtual Sensor — Rotor Temperature Forecasting (pm, 60s Ahead)

End-to-end time series forecasting pipeline to estimate **Permanent Magnet Surface Temperature (`pm`) 60 seconds ahead** from high-frequency PMSM sensor signals, with an interactive Streamlit demo.

**Research context (Paderborn University / LEA):**  
https://ei.uni-paderborn.de/lea/research/forschungsprojekte/electrical-drives-and-mechatronic-systems/temperature-estimation-of-vital-components-in-electric-motors-using-machine-learning

**Public data source:** Electric Motor Temperature dataset (Kaggle)  
https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature

## What this delivers
- Leakage-safe split by session (`profile_id`) to avoid optimistic evaluation
- Trainable models (tree-based baseline + LSTM reference)
- Saved artifacts (models + evaluation reports)
- Streamlit app for interactive inspection/visualization

## Quickstart (run the demo locally)
```bash
pip install -r requirements.txt
pip install torch
streamlit run app/streamlit_app.py
<<<<<<< HEAD
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
=======
Results (fill with real numbers)
Model	Metric (MAE or RMSE)
Persistence baseline	TODO
GBDT / Tree model	TODO
LSTM (reference)	TODO
The dataset file is not included in this repository. Please access it via Kaggle and comply with its license/terms.

