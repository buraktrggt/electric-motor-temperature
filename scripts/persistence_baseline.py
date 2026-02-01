from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import math
import numpy as np

from src.config import Config, S1_FEATURES  # split için S1 yeterli
from src.data import load_data
from src.split import split_profiles
from src.features import build_xy
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def main():
    cfg = Config()
    df = load_data(cfg)

    # same leakage-safe profile split
    spl = split_profiles(df, cfg)

    # Build leakage-safe dataset index (this creates y = pm(t+60s) and keeps group ids aligned)
    X, y = build_xy(df, cfg, S1_FEATURES)

    # Persistence prediction: y_hat = pm(t)
    # IMPORTANT: X rows are aligned to the original df rows after dropping NaNs.
    # So we can recover pm(t) by re-reading it from df using the same mask logic.
    # To do that robustly, we rebuild the same mask by recomputing target and feature NaNs indirectly:
    # Easiest: compute pm(t) directly from X using the raw (non-lagged) column if present.
    # In our pipeline, raw base features are included, but pm is NOT in X. So we compute pm(t) here:

    # Create y_true with profile ids
    groups = X[cfg.group_col].values

    # Recompute pm(t) aligned to y by using the fact that build_xy drops rows with any NaN in X
    # We'll reproduce a minimal alignment by reconstructing a frame with group+pm and applying the same operations.
    # Simpler and safe: build pm(t) via group-wise shifting of y:
    # y = pm(t+H). So pm(t) for the same row equals pm(t+H) shifted by +H within each profile.
    # That is: pm(t) = y shifted by +H inside each profile.
    import pandas as pd
    tmp = pd.DataFrame({cfg.group_col: groups, "y": y.values})
    tmp["pm_t"] = tmp.groupby(cfg.group_col, sort=False)["y"].shift(cfg.horizon_steps)

    # First horizon_steps rows per profile will be NaN after shifting; drop them for persistence eval
    valid = ~tmp["pm_t"].isna()
    y_true = tmp.loc[valid, "y"].values
    y_pred = tmp.loc[valid, "pm_t"].values
    groups_valid = tmp.loc[valid, cfg.group_col].values

    def eval_split(name, profile_ids):
        mask = np.isin(groups_valid, profile_ids)
        yt = y_true[mask]
        yp = y_pred[mask]
        out = {
            "n": int(len(yt)),
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(rmse(yt, yp)),
            "warn_precision": None,
            "warn_recall": None,
            "warn_f1": None,
        }
        # Warning alarm metrics (critical is not evaluable in your dataset)
        yt_w = (yt >= cfg.warn_th).astype(int)
        yp_w = (yp >= cfg.warn_th).astype(int)

        tp = ((yt_w == 1) & (yp_w == 1)).sum()
        fp = ((yt_w == 0) & (yp_w == 1)).sum()
        fn = ((yt_w == 1) & (yp_w == 0)).sum()

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

        out["warn_precision"] = float(prec)
        out["warn_recall"] = float(rec)
        out["warn_f1"] = float(f1)
        return out

    print("PERSISTENCE BASELINE (pm(t+60) ≈ pm(t))")
    print("TRAIN", eval_split("train", spl["train"]))
    print("VAL  ", eval_split("val", spl["val"]))
    print("TEST ", eval_split("test", spl["test"]))

    # Note about VAL having no warn events in your current split
    print("\nNOTE: If VAL has warn>=100 count = 0, alarm metrics on VAL are not meaningful.")


if __name__ == "__main__":
    main()
