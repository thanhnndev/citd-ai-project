"""Validate the regenerated dataset against the frozen training dataset and split holdout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citd_ml import paths
from citd_ml.features.build_features import FEATURES


OUT_DIR = paths.HOLDOUT_STAGE1_DIR
OLD_DATASET = paths.DATASET_CSV
REGENERATED = OUT_DIR / "dataset_catboost_full_regenerated.csv"
HOLDOUT = OUT_DIR / "dataset_catboost_holdout.csv"
REPORT = OUT_DIR / "stage1_validation_report.json"
HOLDOUT_START = paths.HOLDOUT_START
KEYS = ["origin_bar", "entry_bar", "leg"]


def main() -> int:
    features = list(FEATURES)
    old = pd.read_csv(OLD_DATASET, dtype=str, keep_default_na=False)
    regenerated = pd.read_csv(REGENERATED, dtype=str, keep_default_na=False)
    before = regenerated.loc[regenerated["entry_time"] < HOLDOUT_START].copy()
    holdout = regenerated.loc[regenerated["entry_time"] >= HOLDOUT_START].copy()

    old_keyed = old.set_index(KEYS, verify_integrity=True)
    before_keyed = before.set_index(KEYS, verify_integrity=True)
    missing = old_keyed.index.difference(before_keyed.index)
    common = old_keyed.index.intersection(before_keyed.index)

    mismatches: list[dict[str, str]] = []
    mismatch_count = 0
    for key in common:
        for column in features:
            old_value = old_keyed.at[key, column]
            new_value = before_keyed.at[key, column]
            if old_value != new_value:
                mismatch_count += 1
                if len(mismatches) < 20:
                    mismatches.append({
                        "origin_bar": key[0], "entry_bar": key[1], "leg": key[2],
                        "feature": column, "old": old_value, "regenerated": new_value,
                    })

    checks = {
        "pre_holdout_rows_expected_25012": len(before) == 25012,
        "all_25008_frozen_keys_present": len(old) == 25008 and len(common) == len(old) and len(missing) == 0,
        "feature_values_match_exactly": mismatch_count == 0,
    }
    report = {
        "holdout_start": HOLDOUT_START,
        "paths": {
            "frozen_dataset": str(OLD_DATASET),
            "regenerated_dataset": str(REGENERATED),
            "holdout_dataset": str(HOLDOUT),
        },
        "counts": {
            "regenerated_total_rows": len(regenerated),
            "pre_holdout_rows": len(before),
            "frozen_rows": len(old),
            "matching_keys": len(common),
            "missing_frozen_keys": len(missing),
            "feature_cells_compared": len(common) * len(features),
            "feature_cells_mismatched": mismatch_count,
            "holdout_rows": len(holdout),
        },
        "checks": checks,
        "missing_keys": [list(key) for key in missing[:20]],
        "feature_mismatch_examples": mismatches,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if not all(checks.values()):
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("STAGE 1 FAILED: holdout file was not written.")
        return 1

    holdout.to_csv(HOLDOUT, index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"STAGE 1 PASSED: wrote {HOLDOUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
