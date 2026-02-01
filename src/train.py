from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import os
import json
import joblib
import numpy as np
import pandas as pd

from .config import Config
from .split import split_profiles
from .features import build_xy
from .utils import get_model
from .evaluate import regression_metrics, alarm_metrics


def _persistence_baseline_metrics(
    df_raw: pd.DataFrame,
    anchor_idx: pd.Series,
    y_true: pd.Series,
    cfg: Config,
):
    """
    Persistence baseline: pm(t+H) ≈ pm(t)
    Using anchor index t (window end) to fetch pm(t).
    """
    # df_raw must be reset_index(drop=True) consistent with build_xy
    pm = df_raw[cfg.target_col].reset_index(drop=True)

    anchors = anchor_idx.astype(int).values
    pm_t = pm.iloc[anchors].astype(float).values  # pm(t)

    y_true_arr = y_true.astype(float).values      # pm(t+H)

    # same interface as regression_metrics output keys
    mae = float(np.mean(np.abs(y_true_arr - pm_t)))
    rmse = float(np.sqrt(np.mean((y_true_arr - pm_t) ** 2)))
    return {"persistence_mae": mae, "persistence_rmse": rmse}


def train_scenario(df: pd.DataFrame, cfg: Config, scenario_name: str, base_features):
    os.makedirs(cfg.out_models_dir, exist_ok=True)
    os.makedirs(cfg.out_reports_dir, exist_ok=True)

    split = split_profiles(df, cfg)
    train_profiles = set(split["train"].tolist())
    val_profiles = set(split["val"].tolist())
    test_profiles = set(split["test"].tolist())

    # IMPORTANT: build_xy resets index internally; we also need a consistent df0 for persistence lookup
    df0 = df.reset_index(drop=True)

    # New build_xy signature
    X_all, y_all, anchor_all, groups_all = build_xy(df0, cfg, base_features, return_anchor=True)

    grp = groups_all.values
    idx_train = np.array([g in train_profiles for g in grp], dtype=bool)
    idx_val = np.array([g in val_profiles for g in grp], dtype=bool)
    idx_test = np.array([g in test_profiles for g in grp], dtype=bool)

    feature_cols = list(X_all.columns)

    X_train, y_train = X_all.loc[idx_train, feature_cols], y_all.loc[idx_train]
    X_val, y_val = X_all.loc[idx_val, feature_cols], y_all.loc[idx_val]
    X_test, y_test = X_all.loc[idx_test, feature_cols], y_all.loc[idx_test]

    anchor_val = anchor_all.loc[idx_val].reset_index(drop=True)
    anchor_test = anchor_all.loc[idx_test].reset_index(drop=True)

    model, backend = get_model(cfg.random_state)

    fit_kwargs = {}
    if backend == "lightgbm":
        import lightgbm as lgb
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "eval_metric": "l1",
            "callbacks": [lgb.early_stopping(100, verbose=False)],
        }
    elif backend == "xgboost":
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "verbose": False,
            "early_stopping_rounds": 100,
        }

    model.fit(X_train, y_train, **fit_kwargs)

    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    # Baseline metrics (computed on same val/test rows)
    pers_val = _persistence_baseline_metrics(df0, anchor_val, y_val, cfg)
    pers_test = _persistence_baseline_metrics(df0, anchor_test, y_test, cfg)

    report = {
        "scenario": scenario_name,
        "backend": backend,
        "base_features": base_features,
        "n_features": int(X_train.shape[1]),
        "counts": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
        "val": {
            **regression_metrics(y_val, pred_val),
            **alarm_metrics(y_val.values, pred_val, cfg.warn_th, cfg.crit_th),
            **pers_val,
        },
        "test": {
            **regression_metrics(y_test, pred_test),
            **alarm_metrics(y_test.values, pred_test, cfg.warn_th, cfg.crit_th),
            **pers_test,
        },
        "profile_split": {
            "train": sorted(list(train_profiles)),
            "val": sorted(list(val_profiles)),
            "test": sorted(list(test_profiles)),
        },
        "feature_schema": {
            "input_window_seconds": cfg.input_steps / 2.0,
            "horizon_seconds": cfg.horizon_steps / 2.0,
            "sampling_hz": 2.0,
        },
    }

    tag = f"{scenario_name}_{backend}"

    # Save model bundle
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "cfg": cfg,  # optional; consider storing cfg.__dict__ later
        },
        os.path.join(cfg.out_models_dir, f"{tag}.joblib"),
    )

    with open(os.path.join(cfg.out_reports_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report
