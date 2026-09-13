"""Lightweight repository/layout tests.

These tests do not require the heavy ML dependencies; they validate the package
layout and the frozen research constants.
"""

from __future__ import annotations

import csv
import json

from citd_ml import __version__, paths


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

    with (paths.HOLDOUT_STAGE4_DIR / "table1_classification_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        classification = list(csv.DictReader(stream))
    assert classification[-1] == {
        "method": "Holdout",
        "roc_auc": "0.6046",
        "f1_at_0_5": "0.4022",
    }

    for filename in (
        "equity-curve-chunk2-5-top50.html",
        "equity-curve-holdout-top50.html",
    ):
        html = (paths.HOLDOUT_STAGE4_DIR / filename).read_text(encoding="utf-8")
        assert f'id="{filename.removesuffix(".html")}"' in html
