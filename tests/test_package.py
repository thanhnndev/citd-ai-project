"""Lightweight repository/layout tests.

These tests do not require the heavy ML dependencies; they validate the package
layout, the frozen research constants, and the consistency of the committed
report/evidence artifacts (CSV/JSON/Markdown only).
"""

from __future__ import annotations

import csv
from copy import deepcopy
import json
import os

import pytest

from citd_ml import __version__, paths
from citd_ml.verification.holdout_evidence import load_evidence, validate_evidence


CANONICAL_DIR = paths.PROJECT_ROOT / "outputs" / "step4_thread1"
RUN_HISTORY_JSON = paths.VERIFICATION_DIR / "holdout_run_history.json"
SENSITIVITY_JSON = (
    paths.PROJECT_ROOT / "outputs" / "thread_count_sensitivity" / "thread_count_sensitivity.json"
)
STAGE5_JSON = paths.HOLDOUT_DIR / "repro" / "stage5_repro_report.json"
PNG_CHARTS = (
    "equity-curve-chunk2-5-top50.png",
    "equity-curve-holdout-top50.png",
)
EXPECTED_HOLDOUT_RUNS = {
    "dc25cd3": (0.60502452277619, 0.40170679670832066),
    "5f46e41": (0.6023152558719406, 0.4064693317058285),
    "7748828": (0.6045544538928682, 0.40220723482526055),
}
EXPECTED_SENSITIVITY_FACTS = (
    "two_same_config_tc1_runs_exactly_equal_train",
    "two_same_config_tc1_runs_exactly_equal_holdout",
    "fixed_thread_count_training_is_repeatable",
    "prediction_thread_count_changed_predictions",
    "tc2_vs_tc1_a_train_max_abs_diff",
    "tc2_vs_tc1_a_holdout_max_abs_diff",
    "tc_default_vs_tc1_a_train_max_abs_diff",
    "tc_default_vs_tc1_a_holdout_max_abs_diff",
    "holdout_top50_net_profit_R_range_across_configs",
)


def read_csv_rows(path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def to_number(value: str) -> float:
    return float(value.replace(",", "").replace("+", ""))


def test_version_is_exposed() -> None:
    assert __version__ == "1.0.0"


def test_project_root_layout() -> None:
    assert (paths.PROJECT_ROOT / "src" / "citd_ml").is_dir()
    assert (paths.PROJECT_ROOT / "scripts").is_dir()
    assert (paths.PROJECT_ROOT / "data").is_dir()
    assert (paths.PROJECT_ROOT / "outputs").is_dir()


def test_frozen_constants() -> None:
    assert paths.HOLDOUT_START == "2025-02-08 15:30:00"
    assert paths.EMBARGO_START_BAR == 199_918
    assert paths.VERTICAL_BARS == 50


def test_expected_paths_are_inside_project() -> None:
    for path in (
        paths.RAW_M1_CSV,
        paths.DATASET_CSV,
        paths.TRADELIST_CSV,
        paths.LABELS_CSV,
        paths.CATBOOST_TRAINING_DIR,
        paths.BACKTEST_DIR,
        paths.HOLDOUT_STAGE1_DIR,
        paths.HOLDOUT_STAGE4_DIR,
    ):
        assert paths.PROJECT_ROOT in (path, *path.parents)


def test_holdout_evidence_and_summary_are_consistent() -> None:
    repeat = json.loads(
        (paths.HOLDOUT_STAGE2_DIR / "stage2_reproducibility_report.json").read_text()
    )
    assert repeat["verification_scope"] == (
        "two independent training runs on the current machine"
    )
    assert repeat["predictions_exactly_equal"] is True
    assert repeat["max_prediction_abs_difference"] == 0.0
    assert repeat["cross_machine_claim"] == "not established by this command"

    classification = read_csv_rows(paths.HOLDOUT_STAGE4_DIR / "table1_classification_metrics.csv")
    holdout = next(
        row
        for row in classification
        if row["block"] == "canonical_thread_count_1" and row["method"] == "holdout"
    )
    assert holdout["roc_auc"] == "0.6046"
    assert holdout["f1_at_0_5"] == "0.4022"

    for filename in (
        "equity-curve-chunk2-5-top50.html",
        "equity-curve-holdout-top50.html",
    ):
        html = (paths.HOLDOUT_STAGE4_DIR / filename).read_text(encoding="utf-8")
        assert f'id="{filename.removesuffix(".html")}"' in html


def test_report_embeds_png_charts() -> None:
    report = (paths.DOCS_DIR / "BAO_CAO_KET_QUA_HOLDOUT.md").read_text(encoding="utf-8")
    for filename in PNG_CHARTS:
        assert (paths.HOLDOUT_STAGE4_DIR / filename).is_file()
        # holdout_stage4_report.py writes Markdown links with "/" on every OS,
        # so normalise the separator before comparing (os.sep is "\" on Windows).
        relative = os.path.relpath(
            paths.HOLDOUT_STAGE4_DIR / filename, paths.DOCS_DIR
        ).replace(os.sep, "/")
        assert "![Biểu đồ" in report
        assert f"]({relative})" in report


def test_stage4_table1_canonical_matches_metrics_chunk2_5() -> None:
    table1 = read_csv_rows(paths.HOLDOUT_STAGE4_DIR / "table1_classification_metrics.csv")
    canonical = {
        row["method"]: row
        for row in table1
        if row["block"] == "canonical_thread_count_1"
    }
    expected = read_csv_rows(CANONICAL_DIR / "catboost_training" / "metrics_chunk2_5.csv")
    assert len(expected) == 4
    for row in expected:
        assert canonical[row["method"]]["roc_auc"] == row["roc_auc"]
        assert canonical[row["method"]]["f1_at_0_5"] == row["f1"]
    assert canonical["holdout"]["roc_auc"] == "0.6046"
    assert canonical["holdout"]["f1_at_0_5"] == "0.4022"


def test_stage4_table2_holdout_matches_stage3_summary() -> None:
    table2 = read_csv_rows(paths.HOLDOUT_STAGE4_DIR / "table2_financial_metrics_top50.csv")
    holdout_rows = {
        row["method"]: row for row in table2 if row["block"] == "holdout"
    }
    assert set(holdout_rows) == {"baseline_holdout", "top_50"}
    stage3 = {
        int(row["keep_pct"]): row
        for row in read_csv_rows(paths.HOLDOUT_STAGE3_DIR / "stage3_backtest_summary.csv")
    }
    for keep_pct, method in ((100, "baseline_holdout"), (50, "top_50")):
        source = stage3[keep_pct]
        target = holdout_rows[method]
        assert target["trades"] == f"{int(source['trades']):,}"
        assert to_number(target["net_profit_R"]) == pytest.approx(
            float(source["net_profit_R"]), abs=5e-3
        )
        assert to_number(target["max_dd_R"]) == pytest.approx(
            float(source["max_dd_R"]), abs=5e-3
        )
        assert to_number(target["profit_factor"]) == pytest.approx(
            float(source["profit_factor"]), abs=5e-5
        )
        assert to_number(target["win_rate_pct"]) == pytest.approx(
            float(source["win_rate_pct"]), abs=5e-3
        )


def test_stage4_table3_has_four_methods_times_seven_levels() -> None:
    rows = read_csv_rows(paths.HOLDOUT_STAGE4_DIR / "table3_branch_sweep_20_80.csv")
    methods = {"random_kfold", "grouped_kfold", "walk_forward", "purged_walk_forward"}
    levels = {20, 30, 40, 50, 60, 70, 80}
    assert len(rows) == 28
    assert {row["method"] for row in rows} == methods
    assert {int(row["keep_pct"]) for row in rows} == levels
    assert {(row["method"], int(row["keep_pct"])) for row in rows} == {
        (method, level) for method in methods for level in levels
    }


def test_run_history_has_three_expected_holdout_runs() -> None:
    history = json.loads(RUN_HISTORY_JSON.read_text(encoding="utf-8"))
    assert len(history["runs"]) == 3
    for run in history["runs"]:
        expected_auc, expected_f1 = EXPECTED_HOLDOUT_RUNS[run["commit_short"]]
        assert run["classification"]["roc_auc"] == pytest.approx(expected_auc, abs=1e-12)
        assert run["classification"]["f1_at_0_5"] == pytest.approx(expected_f1, abs=1e-12)
    assert all(check["passed"] for check in history["assertions"]["checks"])


def test_sensitivity_conclusion_facts_present() -> None:
    sensitivity = json.loads(SENSITIVITY_JSON.read_text(encoding="utf-8"))
    facts = sensitivity["conclusion_facts"]
    for key in EXPECTED_SENSITIVITY_FACTS:
        assert key in facts
    assert facts["fixed_thread_count_training_is_repeatable"] is True
    assert facts["prediction_thread_count_changed_predictions"] is False
    assert facts["tc2_vs_tc1_a_holdout_max_abs_diff"] > 0.0
    assert facts["tc_default_vs_tc1_a_holdout_max_abs_diff"] > 0.0


def test_stage5_repro_status_is_pass() -> None:
    stage5 = json.loads(STAGE5_JSON.read_text(encoding="utf-8"))
    assert stage5["status"] == "PASS"
    assert all(item["pass"] is True for item in stage5["items"].values())
    charts = stage5["items"]["stage4_charts"]["charts"]
    for filename in ("equity-curve-chunk2-5-top50.html", "equity-curve-holdout-top50.html", *PNG_CHARTS):
        assert charts[filename]["bytes_identical"] is True


def test_report_generation_fails_closed_on_invalid_evidence() -> None:
    evidence = load_evidence()
    validate_evidence(evidence)

    invalid = deepcopy(evidence)
    invalid["stage1"]["checks"]["feature_values_match_exactly"] = False
    with pytest.raises(ValueError, match="stage1.feature_values_match_exactly"):
        validate_evidence(invalid)

    invalid = deepcopy(evidence)
    invalid["repeat"]["predictions_exactly_equal"] = False
    with pytest.raises(ValueError, match="repeat.predictions_exactly_equal"):
        validate_evidence(invalid)

    invalid = deepcopy(evidence)
    invalid["stage3"]["baseline_comparison"]["actual_holdout_trades"] = 5027
    with pytest.raises(ValueError, match="actual_holdout_trades"):
        validate_evidence(invalid)

    invalid = deepcopy(evidence)
    invalid["train"]["versions"].pop("platform")
    with pytest.raises(ValueError, match="train.versions.platform"):
        validate_evidence(invalid)

    invalid = deepcopy(evidence)
    invalid["run_history"]["runs"].pop()
    with pytest.raises(ValueError, match="run_history.runs"):
        validate_evidence(invalid)

    invalid = deepcopy(evidence)
    invalid["sensitivity"]["conclusion_facts"]["fixed_thread_count_training_is_repeatable"] = False
    with pytest.raises(ValueError, match="sensitivity.conclusion_facts"):
        validate_evidence(invalid)

    invalid = deepcopy(evidence)
    invalid["stage5"]["status"] = "FAIL"
    with pytest.raises(ValueError, match="stage5.status"):
        validate_evidence(invalid)

    invalid = deepcopy(evidence)
    invalid["canonical_verification"]["status"] = "failed"
    with pytest.raises(ValueError, match="canonical_verification.status"):
        validate_evidence(invalid)


# ---------------------------------------------------------------- notebook
NOTEBOOK = paths.PROJECT_ROOT / "Do_An_Meta_Labeling_BTCUSD.ipynb"
NOTEBOOK_KERNEL_NAME = "citd-ml"
NOTEBOOK_KERNEL_SCRIPT = paths.SCRIPTS_DIR / "register_notebook_kernel.py"


def test_notebook_is_bound_to_the_project_kernel() -> None:
    """Notebook phải khai báo kernel `citd-ml`, không phải kernel `python3` mặc định.

    Kernel mặc định trỏ về interpreter khác sẽ thiếu catboost và hỏng ngay ô đầu.
    """
    if not NOTEBOOK.is_file():
        pytest.skip("Notebook không có trong repo")

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    kernelspec = notebook["metadata"]["kernelspec"]
    assert kernelspec["name"] == NOTEBOOK_KERNEL_NAME
    assert kernelspec["language"] == "python"
    assert NOTEBOOK_KERNEL_SCRIPT.is_file(), (
        f"Thiếu {NOTEBOOK_KERNEL_SCRIPT.name} để đăng ký lại kernel {NOTEBOOK_KERNEL_NAME}"
    )


def test_notebook_reads_no_raw_m1_file_and_is_committed_clean() -> None:
    """Notebook chỉ đọc data/processed + outputs, và được commit không kèm output."""
    if not NOTEBOOK.is_file():
        pytest.skip("Notebook không có trong repo")

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "BTCUSD_m1_2018_to_now.csv" not in source, (
        "Notebook không được phụ thuộc file thô ~209 MB"
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
