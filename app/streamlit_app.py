from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import math
import time
from typing import Optional, List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.config import Config, S1_FEATURES, S2_FEATURES
from src.data import load_data
from src.features import build_xy

# PyTorch model class
from src.dl.model import LSTMRegressor


# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="PMSM Virtual Sensor Forecasting", layout="wide")
cfg = Config()
HZ = float(getattr(cfg, "hz", 2.0))  # dataset sampling rate


# -----------------------------
# Column dictionary (DATA DICT)
# -----------------------------
COLUMN_INFO: Dict[str, str] = {
    "u_q": "Voltage q-component measurement in dq-coordinates (V).",
    "coolant": "Coolant temperature (°C).",
    "stator_winding": "Stator winding temperature (°C) measured with thermocouples.",
    "u_d": "Voltage d-component measurement in dq-coordinates (V).",
    "stator_tooth": "Stator tooth temperature (°C) measured with thermocouples.",
    "motor_speed": "Motor speed (rpm).",
    "i_d": "Current d-component measurement in dq-coordinates (A).",
    "i_q": "Current q-component measurement in dq-coordinates (A).",
    "pm": "Permanent magnet (rotor) temperature (°C) measured with thermocouples, transmitted wirelessly.",
    "stator_yoke": "Stator yoke temperature (°C) measured with thermocouples.",
    "ambient": "Ambient temperature (°C).",
    "torque": "Motor torque (Nm).",
    "profile_id": "Measurement session id. Each distinct measurement session can be identified through this integer id.",
}


# -----------------------------
# Caching: data + model loading
# -----------------------------
@st.cache_resource
def load_df_cached() -> pd.DataFrame:
    return load_data(cfg)


@st.cache_resource
def load_joblib_bundle(path: str) -> Dict[str, Any]:
    return joblib.load(path)


@st.cache_resource
def load_torch_checkpoint(path: str) -> Dict[str, Any]:
    # Always CPU load for Streamlit stability (unless you explicitly move to CUDA)
    return torch.load(path, map_location="cpu")


# -----------------------------
# Helpers
# -----------------------------
def list_files(dirpath: str, patterns: List[str]) -> List[str]:
    p = Path(dirpath)
    if not p.exists():
        return []
    files: List[str] = []
    for pat in patterns:
        files.extend([x.name for x in p.glob(pat)])
    return sorted(list(dict.fromkeys(files)))


def load_report(report_path: str) -> Dict[str, Any]:
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return math.sqrt(float(np.mean((y_true - y_pred) ** 2)))


def alarm_label(temp_c: float, warn_th: float, crit_th: float) -> str:
    if temp_c is None or (isinstance(temp_c, float) and math.isnan(temp_c)):
        return "N/A"
    if temp_c >= crit_th:
        return "CRITICAL"
    if temp_c >= warn_th:
        return "WARNING"
    return "NORMAL"


def seconds_from_index(i: int, hz: float) -> float:
    return float(i / hz)


def safe_float(x) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def avg_rise_rate(series: pd.Series, idx: int, lookback_seconds: int, hz: float) -> float:
    lookback_steps = int(lookback_seconds * hz)
    if idx - lookback_steps < 0:
        return float("nan")
    a = safe_float(series.iloc[idx])
    b = safe_float(series.iloc[idx - lookback_steps])
    if math.isnan(a) or math.isnan(b):
        return float("nan")
    return (a - b) / float(lookback_seconds)


def torch_forward_batch(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """
    x: (B, T, F) float32
    returns np.ndarray shape (B,)
    """
    with torch.no_grad():
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.detach().cpu().numpy().reshape(-1)
        return out


# -----------------------------
# Header
# -----------------------------
st.title("PMSM Virtual Sensor: Rotor Temperature Forecast (pm) — 60s Ahead")
st.caption("Leakage-safe split by profile_id • GBDT (engineered) / LSTM (sequence) • 2 Hz sampling")

df = load_df_cached()

# -----------------------------
# Sidebar: Scenario / Model selection
# -----------------------------
st.sidebar.header("Model")

scenario = st.sidebar.selectbox("Scenario", ["S1 (production)", "S2 (upper-bound)"], key="sb_scenario")
base_features = S1_FEATURES if scenario.startswith("S1") else S2_FEATURES

model_files = list_files(cfg.out_models_dir, ["*.joblib", "*.pt"])
if not model_files:
    st.sidebar.error(
        f"No models found under: {cfg.out_models_dir}\n\n"
        "Train first (python scripts/run_train.py) from repo root."
    )
    st.stop()

model_file = st.sidebar.selectbox("Model file", model_files, key="sb_model_file")
model_path = str(Path(cfg.out_models_dir) / model_file)

report_path = str(Path(cfg.out_reports_dir) / model_file.replace(".joblib", ".json").replace(".pt", ".json"))

backend_kind: Optional[str] = None

# GBDT bundle
feature_cols_gbdt: Optional[List[str]] = None

# DL bundle
feature_cols_dl: Optional[List[str]] = None
scaler_mean: Optional[np.ndarray] = None
scaler_std: Optional[np.ndarray] = None

# Load model (CACHED)
try:
    if model_file.endswith(".joblib"):
        bundle = load_joblib_bundle(model_path)
        model = bundle["model"]
        feature_cols_gbdt = bundle["feature_cols"]
        backend_kind = "gbdt"

    elif model_file.endswith(".pt"):
        ckpt = load_torch_checkpoint(model_path)
        feature_cols_dl = ckpt["feature_cols"]

        model = LSTMRegressor(n_features=len(feature_cols_dl))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        scaler_mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
        scaler_std = np.array(ckpt["scaler_std"], dtype=np.float32)
        scaler_std = np.where(scaler_std < 1e-6, 1.0, scaler_std)

        backend_kind = "pytorch"
    else:
        raise ValueError("Unsupported model file type")
except Exception as e:
    st.error(f"Model not found/readable: {model_path} ({e})")
    st.stop()

# Load report (optional)
report: Optional[Dict[str, Any]] = None
if Path(report_path).exists():
    try:
        report = load_report(report_path)
    except Exception:
        report = None


# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_explore, tab_profile, tab_infer = st.tabs(
    ["Overview", "Data Explorer", "Profile Forecast", "Inference"]
)

# =============================
# TAB 1: Overview
# =============================
with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Profiles", f"{df[cfg.group_col].nunique():,}")
    c3.metric("Sampling", f"{HZ:.1f} Hz")
    c4.metric("Horizon", f"{cfg.horizon_steps} steps")
    c5.metric("Horizon (s)", f"{cfg.horizon_steps / HZ:.0f} s")

    st.markdown(
        f"""
**Target:** `pm(t + {cfg.horizon_steps / HZ:.0f}s)` (rotor temperature ahead).  
**Reason:** sampling is {HZ:.1f} Hz ⇒ horizon_steps / Hz = seconds.
"""
    )

    st.write(f"**Backend:** {backend_kind} | **Scenario:** {scenario} | **Base features:** {len(base_features)}")

    if report is not None:
        st.subheader("Training report")
        st.json(report, expanded=False)
    else:
        st.info("No report JSON found for this model (optional).")


# =============================
# TAB 2: Data Explorer
# =============================
with tab_explore:
    st.subheader("Dataset Dictionary & Visualization")
    left, right = st.columns([1, 2])

    with left:
        col_select = st.selectbox("Select a column", list(COLUMN_INFO.keys()), index=0, key="de_col")
        st.write(f"**{col_select}**")
        st.caption(COLUMN_INFO.get(col_select, "No description available."))
        bins = st.slider("Histogram bins", 10, 120, 40, key="de_bins")
        show_by_profile = st.checkbox("Show per-profile mean (top 25)", value=False, key="de_profmean")

    with right:
        s = df[col_select]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Valid", f"{s.notna().sum():,}")
        r2.metric("Missing", f"{s.isna().sum():,}")
        if pd.api.types.is_numeric_dtype(s):
            r3.metric("Mean", f"{float(s.mean()):.3g}")
            r4.metric("Std", f"{float(s.std()):.3g}")
        else:
            r3.metric("Unique", f"{s.nunique():,}")
            r4.metric("—", "—")

        if pd.api.types.is_numeric_dtype(s):
            hist = s.dropna().value_counts(bins=bins).sort_index()
            hist_df = pd.DataFrame({"bin": hist.index.astype(str), "count": hist.values}).set_index("bin")
            st.bar_chart(hist_df)

            if show_by_profile:
                prof_means = df.groupby(cfg.group_col)[col_select].mean().sort_values(ascending=False).head(25)
                st.bar_chart(prof_means)
        else:
            st.dataframe(s.value_counts().head(50), width="stretch")

    with st.expander("Show full dataset dictionary"):
        dd = pd.DataFrame({"column": list(COLUMN_INFO.keys()), "description": list(COLUMN_INFO.values())})
        st.dataframe(dd, width="stretch")


# =============================
# TAB 3: Profile Forecast
# =============================
with tab_profile:
    st.subheader("Profile-level Forecast (True vs Predicted)")

    pid = st.selectbox("Select profile_id", sorted(df[cfg.group_col].unique().tolist()), key="pf_pid")
    df_p = df[df[cfg.group_col] == pid].reset_index(drop=True)
    pm_series = df_p[cfg.target_col].reset_index(drop=True).astype(float)

    st.caption(f"Profile rows: {len(df_p):,} | Scenario: {scenario} | Model: {model_file}")

    # Controls
    cA, cB, cC, cD, cE = st.columns([1, 1, 1, 2, 2])
    with cA:
        animate = st.checkbox("Animate", value=False, key="pf_animate")
    with cB:
        speed_ms = st.slider("Speed (ms)", 30, 500, 80, key="pf_speed")
    with cC:
        step = st.slider("Step", 5, 200, 40, key="pf_step")
    with cD:
        max_points = st.slider(
            "Max points shown",
            200,
            min(5000, max(200, len(df_p))),
            min(1200, min(5000, max(200, len(df_p)))),
            key="pf_maxp",
        )
    with cE:
        if backend_kind == "pytorch":
            stride = st.selectbox("PyTorch stride", [1, 2, 4, 8], index=2, key="pf_stride")
        else:
            stride = 1

    chart_slot = st.empty()
    err_slot = st.empty()

    out: Optional[pd.DataFrame] = None

    # --------
    # GBDT path
    # --------
    if backend_kind == "gbdt":
        X_all, y_all, anchor_all, _ = build_xy(df_p, cfg, base_features, return_anchor=True)
        if len(X_all) == 0:
            st.error("Not enough data after lag/rolling/target shift for this profile.")
            st.stop()

        feature_cols = feature_cols_gbdt  # type: ignore
        y_pred = model.predict(X_all[feature_cols].values)

        anchors = anchor_all.astype(int).values
        persistence_pred = pm_series.iloc[anchors].values  # pm(t)

        out = pd.DataFrame(
            {
                "t_seconds": [seconds_from_index(i, HZ) for i in anchors],
                "current_pm_t": persistence_pred.astype(float),
                "y_true_pm_tplus": y_all.values.astype(float),
                "y_pred_pm_tplus": y_pred.astype(float),
            }
        )
        out["persistence_pm_tplus"] = out["current_pm_t"]
        out["abs_error_model"] = (out["y_true_pm_tplus"] - out["y_pred_pm_tplus"]).abs()
        out["abs_error_persistence"] = (out["y_true_pm_tplus"] - out["persistence_pm_tplus"]).abs()

    # ----------
    # PyTorch path (button + batching + session cache)
    # ----------
    else:
        input_steps = int(getattr(cfg, "input_steps", 240))
        H = int(cfg.horizon_steps)
        feature_cols = feature_cols_dl  # type: ignore

        T = len(df_p)
        start_t = input_steps - 1
        end_t = T - 1 - H
        if end_t < start_t:
            st.error("Not enough samples in this profile for the selected input window and horizon.")
            st.stop()

        anchors = np.arange(start_t, end_t + 1, stride, dtype=int)

        # Controls specific to PT
        b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
        with b1:
            batch_size = st.selectbox("Batch size", [8, 16, 32, 64], index=2, key="pf_pt_bs")
        with b2:
            max_anchors = st.selectbox("Max anchors", [500, 1000, 2000, 5000], index=1, key="pf_pt_maxa")
        with b3:
            auto_compute = st.checkbox("Compute on load", value=False, key="pf_pt_autocomp")
        with b4:
            compute_btn = st.button("Compute Forecast (PyTorch)", key="pf_pt_compute")

        anchors = anchors[: min(len(anchors), int(max_anchors))]

        cache_key = (
            f"pt_out::{model_file}::{scenario}::pid={pid}::in={input_steps}::H={H}::"
            f"stride={stride}::maxa={len(anchors)}::bs={batch_size}"
        )

        if (compute_btn or auto_compute) and cache_key not in st.session_state:
            t0 = time.time()

            preds: List[float] = []
            trues: List[float] = []
            currents: List[float] = []
            t_seconds: List[float] = []

            # Pre-extract matrices once
            feat_mat = df_p[feature_cols].astype(np.float32).values
            pm_arr = pm_series.astype(np.float32).values

            model.eval()
            for i0 in range(0, len(anchors), int(batch_size)):
                batch_anchors = anchors[i0 : i0 + int(batch_size)]

                x_list = []
                valid_mask = []
                for t in batch_anchors:
                    x_win = feat_mat[t - input_steps + 1 : t + 1, :]
                    if np.isnan(x_win).any():
                        valid_mask.append(False)
                        x_list.append(np.zeros((input_steps, feat_mat.shape[1]), dtype=np.float32))
                    else:
                        valid_mask.append(True)
                        x_list.append(x_win)

                x_batch = np.stack(x_list, axis=0)  # (B, T, F)
                x_batch = (x_batch - scaler_mean) / scaler_std  # type: ignore
                x_tensor = torch.tensor(x_batch, dtype=torch.float32)

                y_hat = torch_forward_batch(model, x_tensor)  # (B,)

                for t, ok, phat in zip(batch_anchors, valid_mask, y_hat):
                    t_seconds.append(seconds_from_index(int(t), HZ))
                    currents.append(float(pm_arr[t]))
                    trues.append(float(pm_arr[t + H]))
                    preds.append(float(phat) if ok else float("nan"))

            out_df = pd.DataFrame(
                {
                    "t_seconds": np.array(t_seconds, dtype=float),
                    "current_pm_t": np.array(currents, dtype=float),
                    "y_true_pm_tplus": np.array(trues, dtype=float),
                    "y_pred_pm_tplus": np.array(preds, dtype=float),
                }
            )
            out_df["persistence_pm_tplus"] = out_df["current_pm_t"]
            out_df["abs_error_model"] = (out_df["y_true_pm_tplus"] - out_df["y_pred_pm_tplus"]).abs()
            out_df["abs_error_persistence"] = (out_df["y_true_pm_tplus"] - out_df["persistence_pm_tplus"]).abs()
            out_df = out_df.dropna(subset=["y_true_pm_tplus", "y_pred_pm_tplus"]).reset_index(drop=True)

            st.session_state[cache_key] = {
                "out": out_df,
                "elapsed_s": float(time.time() - t0),
                "n_anchors": int(len(anchors)),
            }

        if cache_key in st.session_state:
            out = st.session_state[cache_key]["out"]
            st.caption(
                f"PyTorch computed & cached. anchors={st.session_state[cache_key]['n_anchors']} | "
                f"elapsed={st.session_state[cache_key]['elapsed_s']:.2f}s | stride={stride}"
            )
        else:
            st.info("PyTorch forecast is not computed yet. Click **Compute Forecast (PyTorch)** to run once and cache.")
            st.stop()

    # Metrics (common)
    if out is None or len(out) == 0:
        st.error("No forecast output produced.")
        st.stop()

    prof_mae = float(out["abs_error_model"].mean())
    prof_rmse = rmse(out["y_true_pm_tplus"], out["y_pred_pm_tplus"])
    pers_mae = float(out["abs_error_persistence"].mean())
    pers_rmse = rmse(out["y_true_pm_tplus"], out["persistence_pm_tplus"])

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Model MAE (°C)", f"{prof_mae:.3f}")
    m2.metric("Model RMSE (°C)", f"{prof_rmse:.3f}")
    m3.metric("Persistence MAE (°C)", f"{pers_mae:.3f}")
    m4.metric("Persistence RMSE (°C)", f"{pers_rmse:.3f}")
    m5.metric("Warn threshold", f"{cfg.warn_th:.0f} °C")
    m6.metric("Crit threshold", f"{cfg.crit_th:.0f} °C")

    st.write("**True vs Predicted (pm at t+horizon), with Persistence baseline**")

    # Live metric placeholders
    live_cols = st.columns(6)
    live_idx_ph = live_cols[0].empty()
    live_cur_ph = live_cols[1].empty()
    live_pred_ph = live_cols[2].empty()
    live_true_ph = live_cols[3].empty()
    live_err_ph = live_cols[4].empty()
    live_alarm_ph = live_cols[5].empty()

    def update_live_metrics(last_row: pd.Series, k: int):
        live_idx_ph.metric("Frame index", f"{k:,}")
        live_cur_ph.metric("Current pm(t)", f"{float(last_row['current_pm_t']):.2f} °C")
        live_pred_ph.metric("Pred pm(t+h)", f"{float(last_row['y_pred_pm_tplus']):.2f} °C")
        live_true_ph.metric("True pm(t+h)", f"{float(last_row['y_true_pm_tplus']):.2f} °C")
        live_err_ph.metric("Abs err (model)", f"{float(last_row['abs_error_model']):.2f} °C")
        live_alarm_ph.metric("Pred alarm", alarm_label(float(last_row["y_pred_pm_tplus"]), cfg.warn_th, cfg.crit_th))

    def render_until(k: int):
        k = max(1, min(k, len(out)))
        view = out.iloc[:k].copy()
        if len(view) > max_points:
            view = view.iloc[-max_points:]

        plot_df = view.set_index("t_seconds")[["y_true_pm_tplus", "y_pred_pm_tplus", "persistence_pm_tplus"]]
        chart_slot.line_chart(plot_df)

        err_df = view.set_index("t_seconds")[["abs_error_model", "abs_error_persistence"]]
        err_slot.line_chart(err_df)

        update_live_metrics(view.iloc[-1], k)

    if animate:
        k = min(200, len(out))
        while k <= len(out):
            render_until(k)
            time.sleep(speed_ms / 1000.0)
            k += step
        render_until(len(out))
    else:
        render_until(len(out))

    st.divider()

    idx = st.slider("Inspect an index", 0, len(out) - 1, min(200, len(out) - 1), key="pf_inspect")
    row = out.iloc[idx]
    i1, i2, i3, i4, i5 = st.columns(5)
    i1.metric("t (s)", f"{row['t_seconds']:.1f}")
    i2.metric("Current pm(t)", f"{row['current_pm_t']:.2f} °C")
    i3.metric("Pred pm(t+h)", f"{row['y_pred_pm_tplus']:.2f} °C")
    i4.metric("True pm(t+h)", f"{row['y_true_pm_tplus']:.2f} °C")
    i5.metric("Persistence", f"{row['persistence_pm_tplus']:.2f} °C")

    with st.expander("Show prediction table (first 300 rows)"):
        st.dataframe(out.head(300), width="stretch")


# =============================
# TAB 4: Inference
# =============================
with tab_infer:
    st.subheader("Inference (history-based)")

    pid2 = st.selectbox(
        "Profile for inference",
        sorted(df[cfg.group_col].unique().tolist()),
        key="inf_pid2",
    )
    df_p2 = df[df[cfg.group_col] == pid2].reset_index(drop=True)
    pm_series2 = df_p2[cfg.target_col].reset_index(drop=True).astype(float)

    if backend_kind == "pytorch":
        input_steps = int(getattr(cfg, "input_steps", 240))
        H = int(cfg.horizon_steps)
        feature_cols = feature_cols_dl  # type: ignore

        t = st.slider(
            "Anchor index (raw time)",
            input_steps - 1,
            len(df_p2) - 1 - H,
            input_steps,
            key="inf_anchor_pt",
        )

        x_win = df_p2.loc[t - input_steps + 1 : t, feature_cols].astype(float).values
        if np.isnan(x_win).any():
            st.warning("Selected window contains NaNs. Choose a different index.")
            st.stop()

        x_win = (x_win - scaler_mean) / scaler_std  # type: ignore
        x_tensor = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0)
        pred = float(torch_forward_batch(model, x_tensor)[0])

        current_pm = float(pm_series2.iloc[t])
        y_true = float(pm_series2.iloc[t + H])
        persistence_pred = current_pm
        delta_h = pred - current_pm

        lookback_seconds = st.selectbox(
            "Avg rise rate window (seconds)", [30, 60, 120, 180], index=1, key="inf_lb_pt"
        )
        rate_avg = avg_rise_rate(pm_series2, t, lookback_seconds, HZ)

        a1, a2, a3, a4, a5, a6 = st.columns(6)
        a1.metric("Current pm(t)", f"{current_pm:.2f} °C")
        a2.metric("Predicted pm(t+h)", f"{pred:.2f} °C")
        a3.metric("True pm(t+h)", f"{y_true:.2f} °C")
        a4.metric("Δ over horizon (model)", f"{delta_h:+.2f} °C")
        a5.metric(
            f"Avg rise rate (last {lookback_seconds}s)",
            "N/A" if math.isnan(rate_avg) else f"{rate_avg:+.4f} °C/s",
        )
        a6.metric("Persistence pm(t+h)", f"{persistence_pred:.2f} °C")

        b1, b2, b3 = st.columns(3)
        b1.metric("Current alarm", alarm_label(current_pm, cfg.warn_th, cfg.crit_th))
        b2.metric("Predicted alarm (t+h)", alarm_label(pred, cfg.warn_th, cfg.crit_th))
        b3.metric("Persistence alarm (t+h)", alarm_label(persistence_pred, cfg.warn_th, cfg.crit_th))

        st.divider()
        st.write("**Local context (relative time in seconds)**")

        window = st.slider("Context window (seconds)", 30, 600, 180, key="inf_ctx_pt")
        w_steps = int(window * HZ)

        start = max(0, t - w_steps)
        end = min(len(pm_series2) - 1, t + w_steps)

        t_rel = np.arange(end - start + 1) / HZ
        ctx = pd.DataFrame({"t_sec_relative": t_rel, "pm(t)": pm_series2.iloc[start : end + 1].values}).set_index(
            "t_sec_relative"
        )
        st.line_chart(ctx)

    else:
        # GBDT inference: engineered + anchor mapped
        X2, y2, anchor2, _ = build_xy(df_p2, cfg, base_features, return_anchor=True)
        if len(X2) == 0:
            st.error("Not enough data after lag/rolling/target shift for this profile.")
            st.stop()

        feature_cols = feature_cols_gbdt  # type: ignore

        idx2 = st.slider("Index", 0, len(X2) - 1, min(200, len(X2) - 1), key="inf_idx_gbdt")

        x_row = X2.loc[idx2, feature_cols].values.reshape(1, -1)
        pred = float(model.predict(x_row)[0])

        anchor = int(anchor2.iloc[idx2])

        current_pm = float(pm_series2.iloc[anchor])
        y_true = float(y2.loc[idx2])
        persistence_pred = current_pm
        delta_h = pred - current_pm

        lookback_seconds = st.selectbox(
            "Avg rise rate window (seconds)", [30, 60, 120, 180], index=1, key="inf_lb_gbdt"
        )
        rate_avg = avg_rise_rate(pm_series2, anchor, lookback_seconds, HZ)

        a1, a2, a3, a4, a5, a6 = st.columns(6)
        a1.metric("Current pm(t)", f"{current_pm:.2f} °C")
        a2.metric("Predicted pm(t+h)", f"{pred:.2f} °C")
        a3.metric("True pm(t+h)", f"{y_true:.2f} °C")
        a4.metric("Δ over horizon (model)", f"{delta_h:+.2f} °C")
        a5.metric(
            f"Avg rise rate (last {lookback_seconds}s)",
            "N/A" if math.isnan(rate_avg) else f"{rate_avg:+.4f} °C/s",
        )
        a6.metric("Persistence pm(t+h)", f"{persistence_pred:.2f} °C")

        b1, b2, b3 = st.columns(3)
        b1.metric("Current alarm", alarm_label(current_pm, cfg.warn_th, cfg.crit_th))
        b2.metric("Predicted alarm (t+h)", alarm_label(pred, cfg.warn_th, cfg.crit_th))
        b3.metric("Persistence alarm (t+h)", alarm_label(persistence_pred, cfg.warn_th, cfg.crit_th))

        st.divider()
        st.write("**Local context (relative time in seconds)**")

        window = st.slider("Context window (seconds)", 30, 600, 180, key="inf_ctx_gbdt")
        w_steps = int(window * HZ)

        start = max(0, anchor - w_steps)
        end = min(len(pm_series2) - 1, anchor + w_steps)

        t_rel = np.arange(end - start + 1) / HZ
        ctx = pd.DataFrame({"t_sec_relative": t_rel, "pm(t)": pm_series2.iloc[start : end + 1].values}).set_index(
            "t_sec_relative"
        )
        st.line_chart(ctx)
