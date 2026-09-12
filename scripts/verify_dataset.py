#!/usr/bin/env python3
"""Validate data/processed/dataset_catboost.csv after it is built.

Run:  uv run python scripts/verify_dataset.py
Every assertion must pass. A non-zero exit code means STOP.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citd_ml.features import verify_dataset

if __name__ == "__main__":
    sys.exit(verify_dataset.main())
