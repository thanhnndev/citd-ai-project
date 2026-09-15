"""Load and validate evidence required before publishing the holdout report."""

from __future__ import annotations

import json
from pathlib import Path

from citd_ml import paths


CANONICAL_DIR = paths.PROJECT_ROOT / "outputs" / "step4_thread1"
RUN_HISTORY_JSON = paths.VERIFICATION_DIR / "holdout_run_history.json"
SENSITIVITY_JSON = (
    paths.PROJECT_ROOT / "outputs" / "thread_count_sensitivity" / "thread_count_sensitivity.json"
)
STAGE5_JSON = paths.HOLDOUT_DIR / "repro" / "stage5_repro_report.json"

# Ba lần mở holdout niêm phong; số AUC/F1 này là bất biến của sổ lịch sử, không
# được phép lệch khi dựng báo cáo.
EXPECTED_HOLDOUT_RUNS = (
    {"commit_short": "dc25cd3", "roc_auc": 0.60502452277619, "f1_at_0_5": 0.40170679670832066},
    {"commit_short": "5f46e41", "roc_auc": 0.6023152558719406, "f1_at_0_5": 0.4064693317058285},
    {"commit_short": "7748828", "roc_auc": 0.6045544538928682, "f1_at_0_5": 0.40220723482526055},
)

SENSITIVITY_TRUE_FACTS = (
    "two_same_config_tc1_runs_exactly_equal_train",
    "two_same_config_tc1_runs_exactly_equal_holdout",
    "fixed_thread_count_training_is_repeatable",
    "repeated_inference_on_same_model_exactly_equal",
)

SENSITIVITY_FALSE_FACTS = (
    "prediction_thread_count_changed_predictions",
    "all_configs_produced_exactly_equal_train_predictions",
    "all_configs_produced_exactly_equal_holdout_predictions",
)

SENSITIVITY_POSITIVE_DELTAS = (
    "tc2_vs_tc1_a_train_max_abs_diff",
    "tc2_vs_tc1_a_holdout_max_abs_diff",
    "tc_default_vs_tc1_a_train_max_abs_diff",
    "tc_default_vs_tc1_a_holdout_max_abs_diff",
)

TOLERANCE = 1e-12


def load_evidence(
    stage3_dir: Path | str | None = None,
    *,
    canonical_dir: Path | str | None = None,
    run_history_json: Path | str | None = None,
    sensitivity_json: Path | str | None = None,
    stage5_json: Path | str | None = None,
    canonical_verification_json: Path | str | None = None,
) -> dict[str, dict]:
    """Đọc bằng chứng; stage3_dir=None giữ nguyên hành vi cũ (outputs/holdout/stage3).

    Các tham số keyword cho phép dựng báo cáo từ nhánh chạy lại (ví dụ run2) mà
    không ghi đè output canonical; mặc định trỏ về cây canonical
    outputs/step4_thread1 và các bằng chứng đã commit.
    """
    # stage3_dir cho phép dựng báo cáo từ nhánh chạy lại (ví dụ run2) mà không
    # ghi đè output Stage 3/4 canonical.
    target_stage3 = paths.HOLDOUT_STAGE3_DIR if stage3_dir is None else Path(stage3_dir)
    target_canonical = CANONICAL_DIR if canonical_dir is None else Path(canonical_dir)
    target_verification = (
        target_canonical / "verification" / "reproducibility.json"
        if canonical_verification_json is None
        else Path(canonical_verification_json)
    )
    files = {
        "stage1": paths.HOLDOUT_STAGE1_DIR / "stage1_validation_report.json",
        "train": paths.HOLDOUT_STAGE2_DIR / "train_run1_report.json",
        "repeat": paths.HOLDOUT_STAGE2_DIR / "stage2_reproducibility_report.json",
        "stage3": target_stage3 / "stage3_report.json",
        "run_history": RUN_HISTORY_JSON if run_history_json is None else Path(run_history_json),
        "sensitivity": SENSITIVITY_JSON if sensitivity_json is None else Path(sensitivity_json),
        "stage5": STAGE5_JSON if stage5_json is None else Path(stage5_json),
        "canonical_verification": target_verification,
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
    for name in ("prediction_hashes_equal", "predictions_exactly_equal", "holdout_file_unchanged"):
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

    versions = train.get("versions", {})
    for name in ("platform", "plotly"):
        if not versions.get(name):
            failures.append(f"train.versions.{name}")

    run_history = evidence.get("run_history", {})
    runs = run_history.get("runs", [])
    if len(runs) != 3:
        failures.append("run_history.runs")
    by_commit = {run.get("commit_short"): run for run in runs}
    for expected in EXPECTED_HOLDOUT_RUNS:
        run = by_commit.get(expected["commit_short"])
        if run is None:
            failures.append(f"run_history.{expected['commit_short']}")
            continue
        classification = run.get("classification", {})
        for key, field in (("roc_auc", "roc_auc"), ("f1_at_0_5", "f1_at_0_5")):
            actual = classification.get(key)
            if actual is None or abs(float(actual) - expected[field]) > TOLERANCE:
                failures.append(f"run_history.{expected['commit_short']}.{key}")
    for check in run_history.get("assertions", {}).get("checks", []):
        if check.get("passed") is not True or check.get("abs_diff") != 0.0:
            failures.append(f"run_history.assertions.{check.get('metric')}")
    history_integrity = run_history.get("assertions", {}).get("history_integrity", {})
    if history_integrity.get("commits_are_ancestors_of_head") is not True:
        failures.append("run_history.assertions.commits_are_ancestors_of_head")

    sensitivity = evidence.get("sensitivity", {})
    facts = sensitivity.get("conclusion_facts", {})
    for name in SENSITIVITY_TRUE_FACTS:
        if facts.get(name) is not True:
            failures.append(f"sensitivity.conclusion_facts.{name}")
    for name in SENSITIVITY_FALSE_FACTS:
        if facts.get(name) is not False:
            failures.append(f"sensitivity.conclusion_facts.{name}")
    for name in SENSITIVITY_POSITIVE_DELTAS:
        value = facts.get(name)
        if value is None or float(value) <= 0.0:
            failures.append(f"sensitivity.conclusion_facts.{name}")

    stage5 = evidence.get("stage5", {})
    if stage5.get("status") != "PASS":
        failures.append("stage5.status")
    stage5_items = stage5.get("items", {})
    if not stage5_items:
        failures.append("stage5.items")
    for name, item in stage5_items.items():
        if item.get("pass") is not True:
            failures.append(f"stage5.items.{name}")

    canonical_verification = evidence.get("canonical_verification", {})
    if canonical_verification.get("status") != "passed":
        failures.append("canonical_verification.status")
    if "verify_pipeline.py" not in canonical_verification.get("command", ""):
        failures.append("canonical_verification.command")
    output_dirs = canonical_verification.get("output_dirs", {})
    for name in ("training", "backtest"):
        if not output_dirs.get(name):
            failures.append(f"canonical_verification.output_dirs.{name}")

    if failures:
        raise ValueError(
            "Refusing to generate a PASS report; failed evidence: "
            + ", ".join(failures)
        )
