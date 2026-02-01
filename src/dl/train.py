from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import json
import os
from dataclasses import asdict
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..split import split_profiles
from ..evaluate import regression_metrics, alarm_metrics
from .dataset import PMSMSequenceDataset, Scaler
from .model import LSTMRegressor


def _seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    ys = []
    ps = []
    anchors = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x)
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())
        anchors.append(batch["anchor"].cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.array([])
    y_pred = np.concatenate(ps) if ps else np.array([])
    anchor = np.concatenate(anchors) if anchors else np.array([])
    return y_true, y_pred, anchor


def train_dl_scenario(
    df,
    cfg: Config,
    scenario_name: str,
    base_features: List[str],
    hidden: int = 128,
    layers: int = 2,
    dropout: float = 0.1,
    batch_size: int = 256,
    lr: float = 1e-3,
    max_epochs: int = 20,
    patience: int = 3,
):
    os.makedirs(cfg.out_models_dir, exist_ok=True)
    os.makedirs(cfg.out_reports_dir, exist_ok=True)

    _seed_everything(cfg.random_state)

    split = split_profiles(df, cfg)
    train_profiles = split["train"].tolist()
    val_profiles = split["val"].tolist()
    test_profiles = split["test"].tolist()

    # Datasets + scaler fitted on TRAIN only
    from .dataset import PMSMSequenceDataset, Scaler

    df_train = df[df[cfg.group_col].isin(train_profiles)]
    scaler = Scaler.fit_from_df(df_train, base_features)

    train_ds = PMSMSequenceDataset(df, cfg, train_profiles, base_features, scaler=scaler)
    val_ds = PMSMSequenceDataset(df, cfg, val_profiles, base_features, scaler=scaler)
    test_ds = PMSMSequenceDataset(df, cfg, test_profiles, base_features, scaler=scaler)


    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("One of the splits has 0 sequences; check profiles and window/horizon settings.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = LSTMRegressor(n_features=len(base_features), hidden=hidden, layers=layers, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.L1Loss()  # MAE training objective

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        # validate
        yv, pv, _ = _eval(model, val_loader, device)
        val_mae = float(np.mean(np.abs(yv - pv)))

        if val_mae < best_val - 1e-4:
            best_val = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # predictions
    yv, pv, _ = _eval(model, val_loader, device)
    yt, pt, _ = _eval(model, test_loader, device)

    report = {
        "scenario": scenario_name,
        "backend": "pytorch_lstm",
        "base_features": base_features,
        "sequence": {
            "input_steps": int(cfg.input_steps),
            "horizon_steps": int(cfg.horizon_steps),
            "sampling_hz": 2.0,
        },
        "model_hparams": {"hidden": hidden, "layers": layers, "dropout": dropout, "batch_size": batch_size, "lr": lr},
        "counts": {"train": int(len(train_ds)), "val": int(len(val_ds)), "test": int(len(test_ds))},
        "val": {
            **regression_metrics(yv, pv),
            **alarm_metrics(yv, pv, cfg.warn_th, cfg.crit_th),
        },
        "test": {
            **regression_metrics(yt, pt),
            **alarm_metrics(yt, pt, cfg.warn_th, cfg.crit_th),
        },
        "profile_split": {
            "train": sorted(train_profiles),
            "val": sorted(val_profiles),
            "test": sorted(test_profiles),
        },
    }

    tag = f"{scenario_name}_pytorch_lstm"

    # save model + scaler
    save_obj = {
        "state_dict": model.state_dict(),
        "feature_cols": base_features,
        "scaler_mean": scaler.mean.tolist() if scaler else None,
        "scaler_std": scaler.std.tolist() if scaler else None,
        "cfg": {
            "input_steps": int(cfg.input_steps),
            "horizon_steps": int(cfg.horizon_steps),
            "warn_th": float(cfg.warn_th),
            "crit_th": float(cfg.crit_th),
            "target_col": cfg.target_col,
            "group_col": cfg.group_col,
        },
    }
    torch.save(save_obj, os.path.join(cfg.out_models_dir, f"{tag}.pt"))

    with open(os.path.join(cfg.out_reports_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report

