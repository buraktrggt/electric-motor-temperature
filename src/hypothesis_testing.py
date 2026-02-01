# hypothesis_tests.py
# ------------------------------------------------------------
# Statistical tests for forecast comparison:
# - Paired one-sample t-test on d = loss_base - loss_model
# - Wilcoxon signed-rank test on d
# - Diebold–Mariano (DM) test with HAC variance (Newey-West style)
#
# Usage (CLI):
#   python hypothesis_tests.py --csv path/to/predictions.csv --model_col y_pred_gbdt
#
# Expected CSV columns:
#   y_true, y_pred_persist, <model prediction column>, optionally profile_id
#
# Notes:
# - Time-series errors are autocorrelated; DM test is the primary test.
# - Paired t-test / Wilcoxon are reported as supporting evidence.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# -----------------------------
# Loss functions
# -----------------------------
def loss_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(y_true - y_pred)


def loss_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    e = (y_true - y_pred)
    return e * e


# -----------------------------
# Helpers
# -----------------------------
def _to_1d_float(arr) -> np.ndarray:
    a = np.asarray(arr).astype(np.float64).reshape(-1)
    return a


def _clean_triplet(y_true, y_model, y_base) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    yt = _to_1d_float(y_true)
    ym = _to_1d_float(y_model)
    yb = _to_1d_float(y_base)
    if not (len(yt) == len(ym) == len(yb)):
        raise ValueError(f"Length mismatch: y_true={len(yt)}, y_model={len(ym)}, y_base={len(yb)}")

    mask = np.isfinite(yt) & np.isfinite(ym) & np.isfinite(yb)
    return yt[mask], ym[mask], yb[mask]


def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta effect size: P(x>y) - P(x<y). Here we can use x=loss_base, y=loss_model
    Interpretable: + means model tends to have smaller loss if you compute on d=loss_base-loss_model.
    """
    x = _to_1d_float(x)
    y = _to_1d_float(y)
    # O(n^2) worst-case; use ranks approximation for large arrays
    # For large n, a rank-based implementation is better; this is acceptable up to ~50k.
    nx, ny = len(x), len(y)
    if nx * ny > 2_000_000:
        # rank-based approximation
        xy = np.concatenate([x, y])
        ranks = stats.rankdata(xy, method="average")
        rx = ranks[:nx].sum()
        # Mann–Whitney U
        U = rx - nx * (nx + 1) / 2.0
        delta = (2 * U) / (nx * ny) - 1
        return float(delta)

    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float((gt - lt) / (nx * ny))


def _newey_west_var(x: np.ndarray, lag: int) -> float:
    """
    HAC variance estimate for mean of x_t using Bartlett kernel up to lag.
    Returns gamma0 + 2*sum_{k=1..lag} w_k * gamma_k, where gamma_k is autocovariance.
    """
    x = _to_1d_float(x)
    n = len(x)
    if n < 3:
        return np.nan

    x = x - x.mean()
    gamma0 = np.dot(x, x) / n

    var = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)  # Bartlett
        gamma_k = np.dot(x[k:], x[:-k]) / n
        var += 2.0 * w * gamma_k
    return float(var)


def diebold_mariano(
    loss_diff: np.ndarray,
    h: int = 1,
    alternative: str = "greater",
    lag: Optional[int] = None,
) -> Dict[str, float]:
    """
    Diebold–Mariano test for equal predictive accuracy.
    Inputs:
      loss_diff_t = L_base_t - L_model_t
        => positive mean implies model is better.
      h: forecast horizon in steps (here 120). In DM, a common choice is lag = h-1 (or h).
      alternative: "greater" tests E[loss_diff] > 0 (model better), "two-sided", "less"
      lag: HAC truncation lag. If None, uses max(h-1, 1) but caps to reasonable size relative to n.

    Returns dict with DM statistic and p-value (normal approximation).
    """
    d = _to_1d_float(loss_diff)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"dm_stat": np.nan, "p_value": np.nan, "n": float(n), "lag": np.nan}

    d_bar = d.mean()

    if lag is None:
        lag = max(h - 1, 1)
    # Avoid absurd lags for finite n
    lag = int(min(lag, max(1, n // 5)))

    hac = _newey_west_var(d, lag=lag)  # variance of d_t (not of mean yet)
    if not np.isfinite(hac) or hac <= 0:
        return {"dm_stat": np.nan, "p_value": np.nan, "n": float(n), "lag": float(lag)}

    # variance of sample mean
    var_mean = hac / n
    dm_stat = d_bar / np.sqrt(var_mean)

    # Normal approximation (common in practice)
    if alternative == "greater":
        p = 1.0 - stats.norm.cdf(dm_stat)
    elif alternative == "less":
        p = stats.norm.cdf(dm_stat)
    elif alternative == "two-sided":
        p = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")

    return {"dm_stat": float(dm_stat), "p_value": float(p), "n": float(n), "lag": float(lag)}


# -----------------------------
# Main evaluation
# -----------------------------
@dataclass
class TestResult:
    metric: str
    model_mae: float
    base_mae: float
    model_rmse: float
    base_rmse: float
    mean_improvement_mae: float
    median_improvement_mae: float
    t_p_mae: float
    wilcoxon_p_mae: float
    dm_stat_mae: float
    dm_p_mae: float
    dm_stat_mse: float
    dm_p_mse: float
    cliffs_delta_mae: float
    n: int


def run_tests(
    y_true,
    y_pred_model,
    y_pred_persist,
    horizon_steps: int = 120,
    dm_lag: Optional[int] = None,
    alternative: str = "greater",
) -> TestResult:
    yt, ym, yb = _clean_triplet(y_true, y_pred_model, y_pred_persist)

    # point metrics
    e_model = yt - ym
    e_base = yt - yb

    mae_model = float(np.mean(np.abs(e_model)))
    mae_base = float(np.mean(np.abs(e_base)))
    rmse_model = float(np.sqrt(np.mean(e_model**2)))
    rmse_base = float(np.sqrt(np.mean(e_base**2)))

    # Improvement distribution on MAE loss
    l_model_mae = np.abs(e_model)
    l_base_mae = np.abs(e_base)
    d_mae = l_base_mae - l_model_mae  # positive => model better

    mean_impr = float(np.mean(d_mae))
    med_impr = float(np.median(d_mae))

    # Supporting paired tests (note: autocorrelation exists)
    # One-sample t-test on d > 0
    t_res = stats.ttest_1samp(d_mae, popmean=0.0, alternative=alternative)
    t_p = float(t_res.pvalue)

    # Wilcoxon signed-rank (requires non-zero diffs)
    d_nonzero = d_mae[d_mae != 0]
    if len(d_nonzero) < 10:
        w_p = np.nan
    else:
        w_res = stats.wilcoxon(d_nonzero, alternative=alternative, zero_method="wilcox")
        w_p = float(w_res.pvalue)

    # DM tests on MAE and MSE losses
    dm_mae = diebold_mariano(d_mae, h=horizon_steps, alternative=alternative, lag=dm_lag)

    l_model_mse = e_model**2
    l_base_mse = e_base**2
    d_mse = l_base_mse - l_model_mse
    dm_mse = diebold_mariano(d_mse, h=horizon_steps, alternative=alternative, lag=dm_lag)

    # Effect size (distributional): Cliff's delta using losses
    cd = cliff_delta(l_base_mae, l_model_mae)  # + => base loss tends to be larger

    return TestResult(
        metric="pm_60s_ahead",
        model_mae=mae_model,
        base_mae=mae_base,
        model_rmse=rmse_model,
        base_rmse=rmse_base,
        mean_improvement_mae=mean_impr,
        median_improvement_mae=med_impr,
        t_p_mae=t_p,
        wilcoxon_p_mae=w_p,
        dm_stat_mae=float(dm_mae["dm_stat"]),
        dm_p_mae=float(dm_mae["p_value"]),
        dm_stat_mse=float(dm_mse["dm_stat"]),
        dm_p_mse=float(dm_mse["p_value"]),
        cliffs_delta_mae=float(cd),
        n=int(len(yt)),
    )


def run_tests_by_profile(
    df: pd.DataFrame,
    y_true_col: str,
    y_model_col: str,
    y_base_col: str,
    profile_col: str = "profile_id",
    horizon_steps: int = 120,
    dm_lag: Optional[int] = None,
    alternative: str = "greater",
) -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby(profile_col):
        res = run_tests(
            g[y_true_col].values,
            g[y_model_col].values,
            g[y_base_col].values,
            horizon_steps=horizon_steps,
            dm_lag=dm_lag,
            alternative=alternative,
        )
        rows.append(
            {
                "profile_id": pid,
                "n": res.n,
                "model_mae": res.model_mae,
                "base_mae": res.base_mae,
                "model_rmse": res.model_rmse,
                "base_rmse": res.base_rmse,
                "mean_impr_mae": res.mean_improvement_mae,
                "median_impr_mae": res.median_improvement_mae,
                "t_p_mae": res.t_p_mae,
                "wilcoxon_p_mae": res.wilcoxon_p_mae,
                "dm_stat_mae": res.dm_stat_mae,
                "dm_p_mae": res.dm_p_mae,
                "dm_stat_mse": res.dm_stat_mse,
                "dm_p_mse": res.dm_p_mse,
                "cliffs_delta_mae": res.cliffs_delta_mae,
            }
        )
    out = pd.DataFrame(rows).sort_values(["dm_p_mae", "profile_id"])
    return out


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="CSV containing y_true and predictions")
    ap.add_argument("--model_col", type=str, required=True, help="Column name for model prediction (e.g., y_pred_gbdt)")
    ap.add_argument("--true_col", type=str, default="y_true")
    ap.add_argument("--base_col", type=str, default="y_pred_persist")
    ap.add_argument("--profile_col", type=str, default="profile_id")
    ap.add_argument("--horizon_steps", type=int, default=120)
    ap.add_argument("--dm_lag", type=int, default=None, help="HAC lag for DM test; default uses min(h-1, n//5)")
    ap.add_argument("--alternative", type=str, default="greater", choices=["greater", "less", "two-sided"])
    ap.add_argument("--by_profile", action="store_true", help="Also compute per-profile results")
    return ap.parse_args()


def main():
    args = _parse_args()
    df = pd.read_csv(args.csv)

    res = run_tests(
        df[args.true_col].values,
        df[args.model_col].values,
        df[args.base_col].values,
        horizon_steps=args.horizon_steps,
        dm_lag=args.dm_lag,
        alternative=args.alternative,
    )

    summary = pd.DataFrame(
        [
            {
                "metric": res.metric,
                "n": res.n,
                "model_mae": res.model_mae,
                "base_mae": res.base_mae,
                "model_rmse": res.model_rmse,
                "base_rmse": res.base_rmse,
                "mean_impr_mae": res.mean_improvement_mae,
                "median_impr_mae": res.median_improvement_mae,
                "t_p_mae": res.t_p_mae,
                "wilcoxon_p_mae": res.wilcoxon_p_mae,
                "dm_stat_mae": res.dm_stat_mae,
                "dm_p_mae": res.dm_p_mae,
                "dm_stat_mse": res.dm_stat_mse,
                "dm_p_mse": res.dm_p_mse,
                "cliffs_delta_mae": res.cliffs_delta_mae,
            }
        ]
    )

    print("\n=== OVERALL RESULTS ===")
    print(summary.to_string(index=False))

    if args.by_profile and args.profile_col in df.columns:
        prof = run_tests_by_profile(
            df,
            y_true_col=args.true_col,
            y_model_col=args.model_col,
            y_base_col=args.base_col,
            profile_col=args.profile_col,
            horizon_steps=args.horizon_steps,
            dm_lag=args.dm_lag,
            alternative=args.alternative,
        )
        print("\n=== PER-PROFILE RESULTS (sorted by dm_p_mae) ===")
        print(prof.to_string(index=False))


if __name__ == "__main__":
    main()
