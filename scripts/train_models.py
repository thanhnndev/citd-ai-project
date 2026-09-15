#!/usr/bin/env python3
"""Train CatBoost for the four leak-aware splits (Random / Grouped / WF / Purged WF).

Writes OOF probability tables and fold metrics to outputs/catboost_training/.

Run:  uv run python scripts/train_models.py
      uv run python scripts/train_models.py --output-dir outputs/step4_thread1/catboost_training
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citd_ml.training import train_catboost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Thư mục ghi kết quả (mặc định: outputs/catboost_training).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_catboost.main(output_dir=args.output_dir)
