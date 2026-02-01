from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import pandas as pd
from .config import Config

def load_data(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)

    # Basic hygiene
    df = df.dropna(subset=[cfg.group_col, cfg.target_col]).reset_index(drop=True)

    # If you have a timestamp column, sort here (recommended):
    # df = df.sort_values([cfg.group_col, "timestamp"]).reset_index(drop=True)

    return df
