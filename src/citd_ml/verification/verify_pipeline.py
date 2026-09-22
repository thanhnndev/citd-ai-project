from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from functools import partial
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
    # Scripts holdout/thí nghiệm sinh artifact niêm phong; ghi hash để biết
    # manifest được tạo bởi đúng phiên bản mã nguồn.
    paths.SCRIPTS_DIR / "holdout_stage2_train.py",
    paths.SCRIPTS_DIR / "holdout_stage3_backtest.py",
    paths.SCRIPTS_DIR / "holdout_stage4_report.py",
    paths.SCRIPTS_DIR / "holdout_stage5_repro_check.py",
    paths.SCRIPTS_DIR / "thread_count_sensitivity.py",
)


def file_hash(path: Path) -> str:
    if not path.is_file():
        # RAW_M1_CSV (~209 MB) không được commit, nên đây là nguyên nhân thiếu
        # file phổ biến nhất khi chạy trên bản clone mới.
        hint = (
            " Xem data/raw/README.md: file M1 thô không được commit, phải đặt lại"
            " đúng đường dẫn trước khi chạy verify_pipeline."
            if path == paths.RAW_M1_CSV
            else ""
        )
        raise FileNotFoundError(f"Thiếu file cần hash cho manifest: {path}.{hint}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture_training(output_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    captured = {}

    def capture(metrics, summary, scores, output_dir=None):
        captured["metrics_by_fold.csv"] = metrics
        captured["metrics_summary.csv"] = summary
        for method, frame in scores.items():
            captured[f"oof_{method}.csv"] = frame

    with patch.object(training, "save_outputs", capture):
        training.main(output_dir=output_dir)
    if len(captured) != 6:
        raise AssertionError("Training không tạo đủ 6 bảng kết quả")
    return captured


def capture_backtest(
    output_dir: Path | str | None = None,
    oof_dir: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    captured = {}

    def capture(trades, universe, summary, sweep, top50, output_dir=None):
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
        backtest.main(output_dir=output_dir, oof_dir=oof_dir)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm chứng lặp lại pipeline train/backtest.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=paths.CATBOOST_TRAINING_DIR,
        help="Thư mục CSV train để đối chiếu (mặc định: outputs/catboost_training).",
    )
    parser.add_argument(
        "--backtest-dir",
        type=Path,
        default=paths.BACKTEST_DIR,
        help="Thư mục CSV backtest để đối chiếu (mặc định: outputs/backtest).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST,
        help="File JSON ghi bằng chứng (mặc định: outputs/verification/reproducibility.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = Path(args.train_dir).resolve()
    backtest_dir = Path(args.backtest_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    started = datetime.now(timezone.utc).isoformat()
    source_hashes = {
        str(path.relative_to(paths.PROJECT_ROOT)): file_hash(path) for path in SOURCE_FILES
    }
    csv_paths = sorted(train_dir.glob("*.csv"))
    csv_paths += sorted(backtest_dir.glob("*.csv"))
    saved_hashes = {
        str(path.relative_to(paths.PROJECT_ROOT)): file_hash(path) for path in csv_paths
    }
    train_result = check_repeated_runs(
        "training", partial(capture_training, train_dir), train_dir
    )
    backtest_result = check_repeated_runs(
        "backtest", partial(capture_backtest, backtest_dir, train_dir), backtest_dir
    )
    for relative, digest in {**source_hashes, **saved_hashes}.items():
        if file_hash(paths.PROJECT_ROOT / relative) != digest:
            raise AssertionError(f"File thay đổi trong khi kiểm chứng: {relative}")
    evidence = {
        "status": "passed",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([sys.executable, *sys.argv]),
        "output_dirs": {
            "training": str(train_dir.relative_to(paths.PROJECT_ROOT)),
            "backtest": str(backtest_dir.relative_to(paths.PROJECT_ROOT)),
        },
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
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nPASS: bằng chứng kiểm chứng được lưu tại {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
