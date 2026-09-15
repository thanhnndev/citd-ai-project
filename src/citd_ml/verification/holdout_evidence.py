"""Load and validate evidence required before publishing the holdout report."""

from __future__ import annotations

import json
from pathlib import Path

from citd_ml import paths


def load_evidence(stage3_dir: Path | str | None = None) -> dict[str, dict]:
    """Đọc bằng chứng; stage3_dir=None giữ nguyên hành vi cũ (outputs/holdout/stage3)."""
    # stage3_dir cho phép dựng báo cáo từ nhánh chạy lại (ví dụ run2) mà không
    # ghi đè output Stage 3/4 canonical.
    target_stage3 = paths.HOLDOUT_STAGE3_DIR if stage3_dir is None else Path(stage3_dir)
    files = {
        "stage1": paths.HOLDOUT_STAGE1_DIR / "stage1_validation_report.json",
        "train": paths.HOLDOUT_STAGE2_DIR / "train_run1_report.json",
        "repeat": paths.HOLDOUT_STAGE2_DIR / "stage2_reproducibility_report.json",
        "stage3": target_stage3 / "stage3_report.json",
    }
    return {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in files.items()
    }


def validate_evidence(evidence: dict[str, dict]) -> None:
    """Fail closed when any prerequisite for a PASS report is false."""
    stage1 = evidence["stage1"]
    train = evidence["train"]
    repeat = evidence["repeat"]
    stage3 = evidence["stage3"]
    failures = []
    counts = stage1["counts"]

    for name, passed in stage1["checks"].items():
        if passed is not True:
            failures.append(f"stage1.{name}")
    expected_counts = {
        "regenerated_total_rows": 30040,
        "pre_holdout_rows": 25012,
        "frozen_rows": 25008,
        "matching_keys": 25008,
        "missing_frozen_keys": 0,
        "feature_cells_compared": 575184,
        "feature_cells_mismatched": 0,
        "holdout_rows": 5028,
    }
    for name, expected in expected_counts.items():
        if counts[name] != expected:
            failures.append(f"stage1.{name}")
    for name in ("purge_rows", "embargo_rows", "rows_removed_by_either_condition"):
        if train[name] != 0:
            failures.append(f"train.{name}")
    if train["holdout_file_unchanged"] is not True:
        failures.append("train.holdout_file_unchanged")
    if train["train_rows_before_filtering"] != 25008:
        failures.append("train.train_rows_before_filtering")
    if train["train_rows_after_filtering"] != 25008:
        failures.append("train.train_rows_after_filtering")
    if train["holdout_rows_unchanged"] != 5028:
        failures.append("train.holdout_rows_unchanged")
    for name in (
        "prediction_hashes_equal",
        "predictions_exactly_equal",
        "holdout_file_unchanged",
    ):
        if repeat[name] is not True:
            failures.append(f"repeat.{name}")
    if repeat["max_prediction_abs_difference"] != 0.0:
        failures.append("repeat.max_prediction_abs_difference")
    if stage3["holdout_file_unchanged"] is not True:
        failures.append("stage3.holdout_file_unchanged")
    if stage3["baseline_comparison"]["passed"] is not True:
        failures.append("stage3.baseline_comparison.passed")
    if stage3["baseline_comparison"]["mismatched_fields"]:
        failures.append("stage3.baseline_comparison.mismatched_fields")
    if stage3["baseline_comparison"]["actual_holdout_trades"] != 5028:
        failures.append("stage3.baseline_comparison.actual_holdout_trades")
    if stage3["baseline_comparison"]["reference_holdout_trades"] != 5028:
        failures.append("stage3.baseline_comparison.reference_holdout_trades")
    if stage3["holdout_rows_scored"] != 5028:
        failures.append("stage3.holdout_rows_scored")

    if failures:
        raise ValueError(
            "Refusing to generate a PASS report; failed evidence: "
            + ", ".join(failures)
        )
