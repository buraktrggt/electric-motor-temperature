from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import Config


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    @staticmethod
    def fit(x: np.ndarray) -> "Scaler":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return Scaler(mean=mean, std=std)


def _make_windows_for_profile(
    df_p: pd.DataFrame,
    cfg: Config,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sequences for a single profile.

    X_seq: [N, input_steps, n_features] using raw features
    y:     [N] pm(t+horizon)
    anchor:[N] raw time index t (window end)
    """
    df_p = df_p.reset_index(drop=True)
    x_raw = df_p[feature_cols].astype(float).values
    pm = df_p[cfg.target_col].astype(float).values

    T = len(df_p)
    in_steps = int(cfg.input_steps)
    H = int(cfg.horizon_steps)

    # anchor t must allow full history window and future target
    # window uses [t-in_steps+1 .. t], target is pm[t+H]
    start_t = in_steps - 1
    end_t = T - 1 - H
    if end_t < start_t:
        return np.empty((0, in_steps, len(feature_cols))), np.empty((0,)), np.empty((0,))

    X_list = []
    y_list = []
    a_list = []

    for t in range(start_t, end_t + 1):
        x_win = x_raw[t - in_steps + 1 : t + 1]  # inclusive
        y_t = pm[t + H]
        if np.isnan(x_win).any() or np.isnan(y_t):
            continue
        X_list.append(x_win)
        y_list.append(y_t)
        a_list.append(t)

    if not X_list:
        return np.empty((0, in_steps, len(feature_cols))), np.empty((0,)), np.empty((0,))

    X_seq = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=float)
    anchor = np.asarray(a_list, dtype=int)
    return X_seq, y, anchor


class PMSMSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        profiles: List[int],
        feature_cols: List[str],
        scaler: Optional[Scaler] = None,
        fit_scaler: bool = False,
    ):
        self.cfg = cfg
        self.feature_cols = feature_cols

        # Build sequences by profile and concatenate
        X_all = []
        y_all = []
        anchor_all = []
        pid_all = []

        for pid in profiles:
            df_p = df[df[cfg.group_col] == pid]
            X_seq, y, anchor = _make_windows_for_profile(df_p, cfg, feature_cols)
            if len(X_seq) == 0:
                continue
            X_all.append(X_seq)
            y_all.append(y)
            anchor_all.append(anchor)
            pid_all.append(np.full((len(y),), pid, dtype=int))

        if not X_all:
            self.X = np.empty((0, cfg.input_steps, len(feature_cols)), dtype=float)
            self.y = np.empty((0,), dtype=float)
            self.anchor = np.empty((0,), dtype=int)
            self.profile_id = np.empty((0,), dtype=int)
            self.scaler = scaler
            return

        X = np.concatenate(X_all, axis=0)
        y = np.concatenate(y_all, axis=0)
        anchor = np.concatenate(anchor_all, axis=0)
        pids = np.concatenate(pid_all, axis=0)

        # Fit scaler on flattened windows (train only) if requested
        if fit_scaler:
            flat = X.reshape(-1, X.shape[-1])
            self.scaler = Scaler.fit(flat)
        else:
            self.scaler = scaler

        if self.scaler is not None:
            flat = X.reshape(-1, X.shape[-1])
            flat = self.scaler.transform(flat)
            X = flat.reshape(X.shape)

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.anchor = anchor.astype(np.int64)
        self.profile_id = pids.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return {
            "x": torch.from_numpy(self.X[idx]),
            "y": torch.tensor(self.y[idx]),
            "anchor": torch.tensor(self.anchor[idx]),
            "profile_id": torch.tensor(self.profile_id[idx]),
        }
