from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import Config, S1_FEATURES, S2_FEATURES
from src.data import load_data
from src.dl.train import train_dl_scenario

def main():
    cfg = Config()
    df = load_data(cfg)

    print("Training DL S1 (PyTorch LSTM) ...")
    r1 = train_dl_scenario(df, cfg, "S1", S1_FEATURES)
    print(r1)

    print("Training DL S2 (PyTorch LSTM) ...")
    r2 = train_dl_scenario(df, cfg, "S2", S2_FEATURES)
    print(r2)


if __name__ == "__main__":
    main()

