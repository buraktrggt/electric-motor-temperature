from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))

def regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
    }

def alarm_metrics(y_true: np.ndarray, y_pred: np.ndarray, warn_th=100.0, crit_th=120.0):
    def prf(yt, yp):
        tp = ((yt == 1) & (yp == 1)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        return prec, rec, f1

    out = {}
    yt_w = (y_true >= warn_th).astype(int)
    yp_w = (y_pred >= warn_th).astype(int)
    yt_c = (y_true >= crit_th).astype(int)
    yp_c = (y_pred >= crit_th).astype(int)

    pw, rw, f1w = prf(yt_w, yp_w)
    pc, rc, f1c = prf(yt_c, yp_c)

    out.update({"warn_precision": pw, "warn_recall": rw, "warn_f1": f1w})
    out.update({"crit_precision": pc, "crit_recall": rc, "crit_f1": f1c})
    return out
