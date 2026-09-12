#!/usr/bin/env python3
"""Re-run training and backtest twice and verify byte-identical results.

Writes outputs/verification/reproducibility.json.

Run:  uv run python scripts/verify_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citd_ml.verification import verify_pipeline

if __name__ == "__main__":
    verify_pipeline.main()
