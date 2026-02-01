from pathlib import Path
import sys
# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.config import Config, S1_FEATURES
from src.data import load_data
from src.split import split_profiles
from src.features import build_xy

cfg = Config()
df = load_data(cfg)

spl = split_profiles(df, cfg)
X, y = build_xy(df, cfg, S1_FEATURES)

def count_for(profiles):
    mask = X[cfg.group_col].isin(profiles).values
    yy = y[mask].values
    return {
        "n": int(len(yy)),
        "warn>=100": int((yy >= cfg.warn_th).sum()),
        "crit>=120": int((yy >= cfg.crit_th).sum())
    }

print("TRAIN", count_for(spl["train"]))
print("VAL  ", count_for(spl["val"]))
print("TEST ", count_for(spl["test"]))
