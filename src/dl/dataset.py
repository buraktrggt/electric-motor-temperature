from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import Config


@dataclass
class Scaler:
    mean: np.ndarray  # (F,)
    std: np.ndarray   # (F,)

    def transform_window(self, x: np.ndarray) -> np.ndarray:
        # x: (T, F)
        return (x - self.mean) / self.std

    @staticmethod
    def fit_from_df(df: pd.DataFrame, feature_cols: List[str]) -> "Scaler":
        x = df[feature_cols].astype(np.float32).values
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return Scaler(mean=mean.astype(np.float32), std=std.astype(np.float32))


class PMSMSequenceDataset(Dataset):
    """
    Lazy dataset:
    - Stores only (profile_id, anchor_t) index pairs
    - Builds window slices on-the-fly in __getitem__
    - Leakage-safe if you pass profiles belonging to that split only
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        profiles: List[int],
        feature_cols: List[str],
        scaler: Optional[Scaler] = None,
    ):
        self.cfg = cfg
        self.feature_cols = feature_cols
        self.scaler = scaler

        self.input_steps = int(getattr(cfg, "input_steps", 240))
        self.horizon_steps = int(cfg.horizon_steps)

        # Build per-profile arrays once (raw)
        self._prof_x: Dict[int, np.ndarray] = {}
        self._prof_pm: Dict[int, np.ndarray] = {}
        self._prof_len: Dict[int, int] = {}

        # Index list: tuples (pid, anchor_t)
        self._index: List[Tuple[int, int]] = []

        # Build indices by scanning each profile (cheap, no big tensors)
        for pid in profiles:
            df_p = df[df[cfg.group_col] == pid].reset_index(drop=True)
            if len(df_p) == 0:
                continue

            x = df_p[feature_cols].astype(np.float32).values  # (T, F)
            pm = df_p[cfg.target_col].astype(np.float32).values  # (T,)

            T = len(df_p)
            self._prof_x[pid] = x
            self._prof_pm[pid] = pm
            self._prof_len[pid] = T

            start_t = self.input_steps - 1
            end_t = T - 1 - self.horizon_steps
            if end_t < start_t:
                continue

            # We need windows without NaNs and targets without NaNs
            # Checking NaNs per window naively is O(T*input_steps); but input_steps=240 so OK.
            for t in range(start_t, end_t + 1):
                x_win = x[t - self.input_steps + 1 : t + 1]
                y = pm[t + self.horizon_steps]
                if np.isnan(y) or np.isnan(x_win).any():
                    continue
                self._index.append((pid, t))

        if len(self._index) == 0:
            raise RuntimeError("No valid sequences were constructed (check window/horizon or NaNs).")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        pid, t = self._index[idx]
        x = self._prof_x[pid]
        pm = self._prof_pm[pid]

        x_win = x[t - self.input_steps + 1 : t + 1]  # (input_steps, F)
        y = pm[t + self.horizon_steps]               # scalar target pm(t+H)

        if self.scaler is not None:
            x_win = self.scaler.transform_window(x_win)

        return {
            "x": torch.from_numpy(x_win.astype(np.float32)),   # (T,F)
            "y": torch.tensor(float(y), dtype=torch.float32),
            "anchor": torch.tensor(int(t), dtype=torch.int64),
            "profile_id": torch.tensor(int(pid), dtype=torch.int64),
        }
