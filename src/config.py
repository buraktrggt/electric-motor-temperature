from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from dataclasses import dataclass
from typing import Tuple, List

S1_FEATURES: List[str] = ["u_d", "u_q", "i_d", "i_q", "motor_speed", "coolant", "ambient"]
S2_EXTRA: List[str] = ["torque", "stator_winding", "stator_tooth", "stator_yoke"]
S2_FEATURES: List[str] = S1_FEATURES + S2_EXTRA

@dataclass(frozen=True)
class Config:
    csv_path: str = "C:/Users/Burakk/Desktop/electric-motor-temperature/data/raw/measures_v2.csv"
    group_col: str = "profile_id"
    target_col: str = "pm"
    sampling_hz: float = 2.0

    input_window_s: int = 120
    horizon_s: int = 60
    input_window_steps: int = 240     # 120s * 2Hz
    horizon_steps: int = 120          # 60s * 2Hz
    hz: float = 2.0

# Sequence/DL window definition
    input_seconds: int = 120
    input_steps: int = 240  # = input_seconds * hz

    warn_th: float = 100.0
    crit_th: float = 120.0

    test_size_profiles: float = 0.20
    val_size_profiles: float = 0.20
    random_state: int = 42

    lags: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 120, 240)
    roll_windows: Tuple[int, ...] = (20, 60, 120, 240)

    csv_path: str = str(project_root / "data" / "raw" / "measures_v2.csv")
    out_models_dir: str = str(project_root / "artifacts" / "models")
    out_reports_dir: str = str(project_root / "artifacts" / "reports")

