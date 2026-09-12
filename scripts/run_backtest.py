#!/usr/bin/env python3
"""Run the fixed-universe backtest and the OOF score-gated sweep (20-80%).

Writes reports to outputs/backtest/.

Run:  uv run python scripts/run_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citd_ml.backtest import backtest_pyramid_local

if __name__ == "__main__":
    backtest_pyramid_local.main()
