#!/usr/bin/env python3
"""Build the processed CatBoost dataset (features + triple-barrier labels).

The strategy is replayed on M15 bars, every opened position is labelled with
the triple-barrier rule, 23 features are computed from data available before
entry, and the frozen training dataset is written to data/processed/.

Examples
--------
# Frozen pre-holdout training set (25,008 rows)
uv run python scripts/build_dataset.py

# Full-history regeneration used by holdout stage 1 (~30,040 rows)
uv run python scripts/build_dataset.py --holdout 2100-01-01 \
    --output outputs/holdout/stage1/dataset_catboost_full_regenerated.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citd_ml.features import build_features

if __name__ == "__main__":
    build_features.main()
