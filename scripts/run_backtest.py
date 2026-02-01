from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
"""Optional: rolling evaluation/backtest.
Left as a placeholder for now. You can extend this to evaluate multiple cut points per profile."""

def main():
    print("Backtest placeholder. Implement rolling cut evaluation per profile_id.")

if __name__ == "__main__":
    main()
