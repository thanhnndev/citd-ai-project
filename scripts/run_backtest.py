#!/usr/bin/env python3
"""Run the fixed-universe backtest and the OOF score-gated sweep (20-80%).

Writes reports to outputs/backtest/ and reads OOF scores from outputs/catboost_training/.

Run:  uv run python scripts/run_backtest.py
      uv run python scripts/run_backtest.py \
          --output-dir outputs/step4_thread1/backtest \
          --oof-dir outputs/step4_thread1/catboost_training
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citd_ml.backtest import backtest_pyramid_local


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Thư mục ghi báo cáo backtest (mặc định: outputs/backtest).",
    )
    parser.add_argument(
        "--oof-dir",
        type=Path,
        default=None,
        help="Thư mục chứa oof_*.csv (mặc định: outputs/catboost_training).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    backtest_pyramid_local.main(output_dir=args.output_dir, oof_dir=args.oof_dir)
