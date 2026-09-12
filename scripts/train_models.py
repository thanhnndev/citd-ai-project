#!/usr/bin/env python3
"""Train CatBoost for the four leak-aware splits (Random / Grouped / WF / Purged WF).

Writes OOF probability tables and fold metrics to outputs/catboost_training/.

Run:  uv run python scripts/train_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citd_ml.training import train_catboost

if __name__ == "__main__":
    train_catboost.main()
