"""Lightweight repository/layout tests.

These tests do not require the heavy ML dependencies; they validate the package
layout and the frozen research constants.
"""

from __future__ import annotations

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
