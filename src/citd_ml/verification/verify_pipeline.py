from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from citd_ml import paths
from citd_ml.backtest import backtest_pyramid_local as backtest
from citd_ml.training import train_catboost as training

MANIFEST = paths.VERIFICATION_DIR / "reproducibility.json"
SOURCE_FILES = (
    paths.DOCS_DIR / "BAN_GIAO_task_train_catboost.md",
    paths.RAW_M1_CSV,
    paths.DATASET_CSV,
    paths.TRADELIST_CSV,
    paths.PACKAGE_DIR / "features" / "build_features.py",
    paths.PACKAGE_DIR / "strategy" / "pyramid_strategy.py",
    paths.PACKAGE_DIR / "training" / "pre_train.py",
    paths.PACKAGE_DIR / "training" / "split_data.py",
    paths.PACKAGE_DIR / "training" / "train_catboost.py",
    paths.PACKAGE_DIR / "backtest" / "backtest_pyramid_local.py",
    paths.PACKAGE_DIR / "verification" / "verify_pipeline.py",
)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture_training() -> dict[str, pd.DataFrame]:
    captured = {}

    def capture(metrics, summary, scores):
        captured["metrics_by_fold.csv"] = metrics
        captured["metrics_summary.csv"] = summary
        for method, frame in scores.items():
            captured[f"oof_{method}.csv"] = frame

    with patch.object(training, "save_outputs", capture):
        training.main()
    if len(captured) != 6:
        raise AssertionError("Training không tạo đủ 6 bảng kết quả")
    return captured


def capture_backtest() -> dict[str, pd.DataFrame]:
    captured = {}

    def capture(trades, universe, summary, sweep, top50):
        captured.update(
            {
                "baseline_tradelist.csv": trades,
                "backtest_scored_universe.csv": universe,
                "backtest_summary_top50.csv": summary,
                "backtest_retention_sweep.csv": sweep,
            }
        )
        for method, frame in top50.items():
            captured[f"trades_top50_{method}.csv"] = frame

    with patch.object(backtest, "save_outputs", capture):
        backtest.main()
    if len(captured) != 8:
        raise AssertionError("Backtest không tạo đủ 8 bảng kết quả")
    return captured


def check_repeated_runs(name, run, output_dir: Path) -> dict:
    print(f"\nVERIFY {name}: run 1/2 (CSV không bị ghi đè)", flush=True)
    first = run()
    print(f"\nVERIFY {name}: run 2/2 (CSV không bị ghi đè)", flush=True)
    second = run()
    if first.keys() != second.keys():
        raise AssertionError(f"{name}: hai lượt chạy sinh tên output khác nhau")
    hashes = {}
    for filename, frame in first.items():
        # Check unrounded values as well as the serialized CSV output.
        pd.testing.assert_frame_equal(frame, second[filename], check_exact=True)
        raw = frame.to_csv(
            index=False, float_format="%.12g", date_format="%Y-%m-%d %H:%M:%S"
        ).encode("utf-8")
        path = output_dir / filename
        if raw != path.read_bytes():
            raise AssertionError(
                f"{name}: kết quả chạy lại khác CSV đang báo cáo: {path}"
            )
        hashes[str(path.relative_to(paths.PROJECT_ROOT))] = hashlib.sha256(raw).hexdigest()
        print(f"  PASS exact repeat + saved CSV: {filename}", flush=True)
    return {
        "runs": 2,
        "unrounded_frames_identical": True,
        "matches_saved_csv_bytes": True,
        "output_sha256": hashes,
    }


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    source_hashes = {
        str(path.relative_to(paths.PROJECT_ROOT)): file_hash(path) for path in SOURCE_FILES
    }
    csv_paths = sorted(paths.CATBOOST_TRAINING_DIR.glob("*.csv"))
    csv_paths += sorted(paths.BACKTEST_DIR.glob("*.csv"))
    saved_hashes = {
        str(path.relative_to(paths.PROJECT_ROOT)): file_hash(path) for path in csv_paths
    }
    train_result = check_repeated_runs(
        "training", capture_training, paths.CATBOOST_TRAINING_DIR
    )
    backtest_result = check_repeated_runs(
        "backtest", capture_backtest, paths.BACKTEST_DIR
    )
    for relative, digest in {**source_hashes, **saved_hashes}.items():
        if file_hash(paths.PROJECT_ROOT / relative) != digest:
            raise AssertionError(f"File thay đổi trong khi kiểm chứng: {relative}")
    evidence = {
        "status": "passed",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": "uv run python scripts/verify_pipeline.py",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("catboost", "scikit-learn", "pandas", "numpy")
        },
        "model_params": training.MODEL_PARAMS,
        "source_sha256": source_hashes,
        "training": train_result,
        "backtest": backtest_result,
        "backtest_scope": "fixed baseline candidates; shadow signals and gated execution",
        "scope_note": "Repeatability verifies implementation, not acceptance of the shadow design assumption.",
        "original_and_saved_csv_unchanged": True,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nPASS: bằng chứng kiểm chứng được lưu tại {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
