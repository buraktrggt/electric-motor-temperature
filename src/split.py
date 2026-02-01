from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import Config

def split_profiles(df: pd.DataFrame, cfg: Config):
    profiles = df[cfg.group_col].dropna().unique()
    profiles = np.array(profiles)

    trainval, test = train_test_split(
        profiles,
        test_size=cfg.test_size_profiles,
        random_state=cfg.random_state,
        shuffle=True,
    )

    val_rel = cfg.val_size_profiles / (1.0 - cfg.test_size_profiles)
    train, val = train_test_split(
        trainval,
        test_size=val_rel,
        random_state=cfg.random_state,
        shuffle=True,
    )

    return {"train": train, "val": val, "test": test}
