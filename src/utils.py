from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from typing import Tuple, Any

def get_model(random_state: int = 42) -> Tuple[Any, str]:
    """Return (model, backend_name) with a preference order."""
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
        )
        return model, "lightgbm"
    except Exception:
        pass

    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=2500,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
        return model, "xgboost"
    except Exception:
        pass

    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=10,
        max_iter=700,
        random_state=random_state,
    )
    return model, "sklearn_hgbr"
