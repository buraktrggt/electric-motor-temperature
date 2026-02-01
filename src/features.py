from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import Config


def make_target(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby(cfg.group_col, sort=False)
    df["y_pm_tplus"] = g[cfg.target_col].shift(-cfg.horizon_steps)
    return df


def add_group_features(
    df: pd.DataFrame,
    group_col: str,
    feature_cols: List[str],
    lags: Tuple[int, ...],
    roll_windows: Tuple[int, ...],
) -> pd.DataFrame:
    """
    Leakage-safe feature engineering within each profile_id.
    Uses concat to avoid DataFrame fragmentation.
    """
    df = df.copy()
    g = df.groupby(group_col, sort=False)

    new_cols = {}

    for col in feature_cols:
        # lag + delta-lag
        for L in lags:
            lag_name = f"{col}_lag{L}"
            lag_s = g[col].shift(L)
            new_cols[lag_name] = lag_s
            new_cols[f"{col}_dlag{L}"] = df[col] - lag_s

        # rolling stats (exclude current sample)
        shifted = g[col].shift(1)
        for w in roll_windows:
            r = shifted.groupby(df[group_col]).rolling(w, min_periods=max(5, w // 10))
            new_cols[f"{col}_rmean{w}"] = r.mean().reset_index(level=0, drop=True)
            new_cols[f"{col}_rstd{w}"] = r.std().reset_index(level=0, drop=True)

    feat_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, feat_df], axis=1)
    return df


def build_xy(
    df_raw: pd.DataFrame,
    cfg: Config,
    base_features: List[str],
    return_anchor: bool = True,
):
    """
    Returns:
      X: engineered features (no group_col inside by default)
      y: target pm(t+horizon)
      anchor_idx: raw index (within df_raw after reset) representing window end time t
      groups: group_col aligned with X (optional but useful)

    Design:
      - anchor_idx corresponds to time t (window end).
      - y corresponds to pm(t+horizon).
    """
    # Important: normalize index to 0..n-1 so anchor_idx is meaningful in Streamlit
    df0 = df_raw.reset_index(drop=True).copy()

    df = make_target(df0, cfg)
    df = add_group_features(df, cfg.group_col, base_features, cfg.lags, cfg.roll_windows)

    y = df["y_pm_tplus"]

    engineered_cols = [
        c for c in df.columns
        if any(c == b or c.startswith(b + "_") for b in base_features)
    ]
    X = df[engineered_cols]

    # anchor time t is the row index in df0 (window end); we must also ensure enough history for lags/rolls
    # Your features already create NaNs for early rows; masking handles it.

    mask = (~y.isna()) & (~X.isna().any(axis=1))
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    groups = df.loc[mask, cfg.group_col].reset_index(drop=True)

    if return_anchor:
        anchor_idx = df.loc[mask].index.to_series().reset_index(drop=True)  # raw time index t
        return X, y, anchor_idx, groups

    return X, y
