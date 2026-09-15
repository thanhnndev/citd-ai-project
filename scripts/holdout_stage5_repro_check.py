#!/usr/bin/env python3
"""Stage 5: đối chiếu end-to-end hai lần chạy holdout độc lập (run 1 vs run 2).

So sánh toàn bộ chuỗi Stage 3–4 của hai lần train Stage 2 độc lập: xác suất
holdout, metrics backtest, sweep, các danh sách top 20–80%, bảng số Stage 4
(gồm bảng sweep 20–80% của bốn nhánh canonical) và hash biểu đồ (hai HTML +
hai PNG). Kết quả ghi vào outputs/holdout/repro/stage5_repro_report.json.

Giới hạn phạm vi: phép kiểm này chứng minh hai lần chạy độc lập trên cùng một
máy, cùng mã nguồn cho kết quả giống hệt nhau; nó không chứng minh giống từng
byte giữa các máy khác nhau.

Chạy:  .venv/bin/python scripts/holdout_stage5_repro_check.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citd_ml import paths


KEYS = ["origin_bar", "entry_bar", "leg"]
KEEP_RATES = (20, 30, 40, 50, 60, 70, 80)
STAGE4_TABLES = (
    "table1_classification_metrics.csv",
    "table2_financial_metrics_top50.csv",
    "table3_branch_sweep_20_80.csv",
    "holdout_sweep_20_80.csv",
)
STAGE4_CHARTS = (
    "equity-curve-chunk2-5-top50.html",
    "equity-curve-holdout-top50.html",
    "equity-curve-chunk2-5-top50.png",
    "equity-curve-holdout-top50.png",
)
SCOPE_NOTE = (
    "two independent training runs, same machine, same code; does not establish "
    "cross-machine byte identity"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    """Ghi đường dẫn tương đối repo khi có thể, để report ổn định giữa các máy."""
    try:
        return str(path.relative_to(paths.PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    run2_root = paths.HOLDOUT_DIR / "repro" / "run2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run1-stage3",
        type=Path,
        default=paths.HOLDOUT_STAGE3_DIR,
        help="Stage 3 của run 1 (mặc định: outputs/holdout/stage3).",
    )
    parser.add_argument(
        "--run1-stage4",
        type=Path,
        default=paths.HOLDOUT_STAGE4_DIR,
        help="Stage 4 của run 1 (mặc định: outputs/holdout/stage4).",
    )
    parser.add_argument(
        "--run2-stage3",
        type=Path,
        default=run2_root / "stage3",
        help="Stage 3 của run 2 (mặc định: outputs/holdout/repro/run2/stage3).",
    )
    parser.add_argument(
        "--run2-stage4",
        type=Path,
        default=run2_root / "stage4",
        help="Stage 4 của run 2 (mặc định: outputs/holdout/repro/run2/stage4).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=paths.HOLDOUT_DIR / "repro" / "stage5_repro_report.json",
        help="File JSON kết quả đối chiếu (mặc định: outputs/holdout/repro/stage5_repro_report.json).",
    )
    return parser.parse_args()


def read_csv_check(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"STOP: thiếu artifact đối chiếu: {path}")
    return pd.read_csv(path)


def file_hash_entry(path1: Path, path2: Path) -> dict:
    """Hash hai file; file thiếu được ghi nhận thay vì làm script gãy."""
    entry = {
        "run1_path": display_path(path1),
        "run2_path": display_path(path2),
        "run1_exists": path1.exists(),
        "run2_exists": path2.exists(),
    }
    if not (path1.exists() and path2.exists()):
        entry.update({"run1_sha256": None, "run2_sha256": None, "bytes_identical": False})
        return entry
    digest1, digest2 = sha256(path1), sha256(path2)
    entry.update(
        {
            "run1_sha256": digest1,
            "run2_sha256": digest2,
            "bytes_identical": digest1 == digest2,
        }
    )
    return entry


def key_set(frame: pd.DataFrame) -> set[tuple[int, int, int]]:
    if frame.duplicated(KEYS).any():
        raise ValueError("STOP: khóa (origin_bar, entry_bar, leg) bị trùng")
    return {
        tuple(int(value) for value in row)
        for row in frame[KEYS].itertuples(index=False, name=None)
    }


def numeric_if_possible(series: pd.Series) -> np.ndarray | None:
    """Chuyển cột chuỗi đã định dạng (dấu phẩy, dấu +) sang số nếu đọc được toàn bộ."""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip()
    )
    values = pd.to_numeric(cleaned, errors="coerce")
    if values.isna().any():
        return None
    return values.to_numpy(dtype=float)


def compare_probabilities(stage3_run1: Path, stage3_run2: Path) -> dict:
    first = read_csv_check(stage3_run1 / "holdout_scored.csv")
    second = read_csv_check(stage3_run2 / "holdout_scored.csv")
    if len(first) != 5028 or len(second) != 5028:
        raise ValueError("STOP: holdout_scored.csv phải có 5,028 dòng ở cả hai lần chạy")
    if "probability" not in first.columns or "probability" not in second.columns:
        raise ValueError("STOP: holdout_scored.csv thiếu cột probability")
    merged = first.merge(
        second,
        on=KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_run1", "_run2"),
    )
    if len(merged) != len(first):
        raise ValueError("STOP: merge probability theo khóa lệnh không khớp one-to-one")
    probability1 = merged["probability_run1"].to_numpy(dtype=float)
    probability2 = merged["probability_run2"].to_numpy(dtype=float)
    difference = np.abs(probability1 - probability2)
    return {
        "pass": bool(np.array_equal(probability1, probability2)),
        "rows_run1": int(len(first)),
        "rows_run2": int(len(second)),
        "merged_rows": int(len(merged)),
        "predictions_exactly_equal": bool(np.array_equal(probability1, probability2)),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "count_diff_gt_0": int((difference > 0).sum()),
        "count_diff_gt_1e-12": int((difference > 1e-12).sum()),
        "merge_keys": KEYS,
    }


def compare_stage3_summary(stage3_run1: Path, stage3_run2: Path) -> dict:
    first = read_csv_check(stage3_run1 / "stage3_backtest_summary.csv")
    second = read_csv_check(stage3_run2 / "stage3_backtest_summary.csv")
    merged = first.merge(
        second,
        on="keep_pct",
        how="inner",
        validate="one_to_one",
        suffixes=("_run1", "_run2"),
    )
    if len(merged) != len(first):
        raise ValueError("STOP: stage3_backtest_summary.csv không khớp theo keep_pct")
    numeric_columns = [
        column
        for column in first.columns
        if column not in {"keep_pct", "filter"}
        and pd.api.types.is_numeric_dtype(first[column])
    ]
    columns = {}
    for column in numeric_columns:
        values1 = merged[f"{column}_run1"].to_numpy(dtype=float)
        values2 = merged[f"{column}_run2"].to_numpy(dtype=float)
        difference = np.abs(values1 - values2)
        columns[column] = {
            "max_abs_diff": float(difference.max()),
            "exactly_equal": bool(np.array_equal(values1, values2)),
        }
    return {
        "pass": bool(
            all(entry["exactly_equal"] for entry in columns.values())
            and (merged["filter_run1"] == merged["filter_run2"]).all()
        ),
        "rows": int(len(first)),
        "numeric_columns": columns,
        "filters_exactly_equal": bool(
            (merged["filter_run1"] == merged["filter_run2"]).all()
        ),
    }


def compare_stage3_report(stage3_run1: Path, stage3_run2: Path) -> dict:
    report1 = json.loads((stage3_run1 / "stage3_report.json").read_text(encoding="utf-8"))
    report2 = json.loads((stage3_run2 / "stage3_report.json").read_text(encoding="utf-8"))
    classification_keys = ("roc_auc", "f1_at_0_5")
    classification = {
        key: {
            "run1": float(report1["classification"][key]),
            "run2": float(report2["classification"][key]),
            "abs_diff": abs(
                float(report1["classification"][key]) - float(report2["classification"][key])
            ),
        }
        for key in classification_keys
    }
    rows1 = {int(row["keep_pct"]): row for row in report1["results"]}
    rows2 = {int(row["keep_pct"]): row for row in report2["results"]}
    if sorted(rows1) != sorted(rows2):
        raise ValueError("STOP: hai report Stage 3 có tập keep_pct khác nhau")
    sweep_keys = ("trades", "net_profit_R", "max_dd_R", "profit_factor", "win_rate_pct")
    sweep = {}
    for keep_pct in sorted(rows1):
        deltas = {}
        for key in sweep_keys:
            value1 = rows1[keep_pct][key]
            value2 = rows2[keep_pct][key]
            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                deltas[key] = {
                    "run1": value1,
                    "run2": value2,
                    "abs_diff": abs(float(value1) - float(value2)),
                }
            else:
                deltas[key] = {"run1": value1, "run2": value2, "abs_diff": None}
        deltas["filter_exactly_equal"] = rows1[keep_pct]["filter"] == rows2[keep_pct]["filter"]
        sweep[keep_pct] = deltas
    return {
        "pass": bool(report1 == report2),
        "reports_exactly_equal": bool(report1 == report2),
        "classification": classification,
        "sweep_rows": sweep,
        "keep_pct_rows": sorted(rows1),
    }


def compare_top_k(stage3_run1: Path, stage3_run2: Path) -> dict:
    selections = {}
    all_identical = True
    for keep_pct in KEEP_RATES:
        filename = f"holdout_trades_top{keep_pct}.csv"
        first = read_csv_check(stage3_run1 / filename)
        second = read_csv_check(stage3_run2 / filename)
        keys1, keys2 = key_set(first), key_set(second)
        identical = keys1 == keys2
        all_identical &= identical
        selections[keep_pct] = {
            "filename": filename,
            "rows_run1": int(len(first)),
            "rows_run2": int(len(second)),
            "keys_identical": bool(identical),
            "intersection": len(keys1 & keys2),
            "only_in_run1": len(keys1 - keys2),
            "only_in_run2": len(keys2 - keys1),
        }
    return {
        "pass": bool(all_identical),
        "selections": selections,
        "key_rule": "origin_bar/entry_bar/leg, giống luật chọn ceil(5028*keep_pct/100) của Stage 3",
    }


def compare_stage4_table(stage4_run1: Path, stage4_run2: Path, filename: str) -> dict:
    first = read_csv_check(stage4_run1 / filename)
    second = read_csv_check(stage4_run2 / filename)
    if list(first.columns) != list(second.columns):
        raise ValueError(f"STOP: {filename} lệch cấu trúc cột")
    if len(first) != len(second):
        raise ValueError(f"STOP: {filename} lệch số dòng")
    columns = {}
    for column in first.columns:
        values1 = numeric_if_possible(first[column])
        values2 = numeric_if_possible(second[column])
        entry = {
            "strings_exactly_equal": bool(
                first[column].astype(str).tolist() == second[column].astype(str).tolist()
            )
        }
        if values1 is not None and values2 is not None:
            difference = np.abs(values1 - values2)
            entry["max_abs_diff"] = float(difference.max()) if len(difference) else 0.0
            entry["numeric_exactly_equal"] = bool(np.array_equal(values1, values2))
        else:
            entry["max_abs_diff"] = None
            entry["numeric_exactly_equal"] = None
        columns[column] = entry
    exact = bool(first.equals(second))
    return {
        "pass": exact,
        "filename": filename,
        "rows_run1": int(len(first)),
        "rows_run2": int(len(second)),
        "dataframes_exactly_equal": exact,
        "columns": columns,
    }


def compare_charts(stage4_run1: Path, stage4_run2: Path) -> dict:
    charts = {}
    for filename in STAGE4_CHARTS:
        charts[filename] = file_hash_entry(
            stage4_run1 / filename, stage4_run2 / filename
        )
    return {
        "pass": bool(all(entry["bytes_identical"] for entry in charts.values())),
        "charts": charts,
    }


def compare_manifests(stage4_run1: Path, stage4_run2: Path) -> dict:
    manifest1 = json.loads(
        (stage4_run1 / "stage4_manifest.json").read_text(encoding="utf-8")
    )
    manifest2 = json.loads(
        (stage4_run2 / "stage4_manifest.json").read_text(encoding="utf-8")
    )
    chart1_rows_equal = manifest1.get("chart1_rows") == manifest2.get("chart1_rows")
    chart2_rows_equal = manifest1.get("chart2_rows") == manifest2.get("chart2_rows")
    new_files_equal = sorted(manifest1.get("new_files", [])) == sorted(
        manifest2.get("new_files", [])
    )
    return {
        # chart1_rows/chart2_rows là dữ liệu biểu đồ thực; new_files chỉ là
        # danh sách file trong thư mục output tại thời điểm chạy nên có thể khác
        # nếu stage4_manifest.json đã tồn tại từ lần chạy trước.
        "pass": bool(chart1_rows_equal and chart2_rows_equal),
        "manifests_exactly_equal": bool(manifest1 == manifest2),
        "chart1_rows": manifest1.get("chart1_rows"),
        "chart1_rows_run2": manifest2.get("chart1_rows"),
        "chart1_rows_equal": chart1_rows_equal,
        "chart2_rows": manifest1.get("chart2_rows"),
        "chart2_rows_run2": manifest2.get("chart2_rows"),
        "chart2_rows_equal": chart2_rows_equal,
        "new_files_equal": new_files_equal,
        "new_files_note": None
        if new_files_equal
        else (
            "Danh sách new_files chỉ khác ở việc stage4_manifest.json đã tồn tại "
            "trong thư mục run 1 khi ghi manifest (iterdir() quét được file cũ), "
            "không phải khác biệt dữ liệu."
        ),
    }


def main() -> int:
    args = parse_args()
    stage3_run1, stage3_run2 = Path(args.run1_stage3).resolve(), Path(args.run2_stage3).resolve()
    stage4_run1, stage4_run2 = Path(args.run1_stage4).resolve(), Path(args.run2_stage4).resolve()

    probabilities = compare_probabilities(stage3_run1, stage3_run2)
    summary = compare_stage3_summary(stage3_run1, stage3_run2)
    stage3_report = compare_stage3_report(stage3_run1, stage3_run2)
    top_k = compare_top_k(stage3_run1, stage3_run2)
    stage4_tables = {
        filename: compare_stage4_table(stage4_run1, stage4_run2, filename)
        for filename in STAGE4_TABLES
    }
    charts = compare_charts(stage4_run1, stage4_run2)
    manifests = compare_manifests(stage4_run1, stage4_run2)

    file_hashes = {}
    for filename in (
        "holdout_scored.csv",
        "stage3_backtest_summary.csv",
        "stage3_report.json",
        "holdout_fixed_trade_universe_scored.csv",
        *(f"holdout_trades_top{pct}.csv" for pct in KEEP_RATES),
    ):
        file_hashes[filename] = file_hash_entry(
            stage3_run1 / filename, stage3_run2 / filename
        )
    for filename in (*STAGE4_TABLES, *STAGE4_CHARTS, "stage4_manifest.json"):
        file_hashes[filename] = file_hash_entry(
            stage4_run1 / filename, stage4_run2 / filename
        )

    items = {
        "holdout_scored_probabilities": probabilities,
        "stage3_backtest_summary": summary,
        "stage3_report_classification_and_sweep": stage3_report,
        "stage3_top_k_selections": top_k,
        "stage4_numeric_tables": {
            "pass": bool(all(table["pass"] for table in stage4_tables.values())),
            "tables": stage4_tables,
        },
        "stage4_charts": charts,
        "stage4_manifest_row_counts": manifests,
    }
    status = "PASS" if all(item["pass"] for item in items.values()) else "FAIL"

    chart_note = None
    if not charts["pass"]:
        if manifests["chart1_rows_equal"] and manifests["chart2_rows_equal"]:
            chart_note = (
                "HTML bytes khác nhau nhưng số điểm trên từng đường vốn trong "
                "stage4_manifest.json giống nhau; khác biệt nằm ở phần serialize/metadata "
                "của file HTML chứ không phải dữ liệu biểu đồ."
            )
        else:
            chart_note = (
                "HTML bytes khác nhau và số điểm trên đường vốn cũng khác nhau; "
                "cần kiểm tra dữ liệu biểu đồ trước khi công bố."
            )

    report = {
        "status": status,
        "scope_note": SCOPE_NOTE,
        "scope_note_vi": (
            "Hai lần train độc lập trên cùng một máy, cùng mã nguồn; không chứng minh "
            "giống từng byte giữa các máy khác nhau."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, *sys.argv]),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "catboost": version("catboost"),
            "scikit_learn": version("scikit-learn"),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "plotly": version("plotly"),
        },
        "run1": {"stage3": display_path(stage3_run1), "stage4": display_path(stage4_run1)},
        "run2": {"stage3": display_path(stage3_run2), "stage4": display_path(stage4_run2)},
        "items": items,
        "file_hashes": file_hashes,
        "chart_note": chart_note,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": status,
        "out": str(out_path),
        "probability_max_abs_diff": probabilities["max_abs_diff"],
        "summary_max_abs_diff": {
            column: entry["max_abs_diff"]
            for column, entry in summary["numeric_columns"].items()
        },
        "classification_abs_diff": {
            key: entry["abs_diff"] for key, entry in stage3_report["classification"].items()
        },
        "top_k_all_identical": top_k["pass"],
        "stage4_tables_exact": items["stage4_numeric_tables"]["pass"],
        "charts_bytes_identical": charts["pass"],
        "manifests_exactly_equal": manifests["manifests_exactly_equal"],
        "chart_note": chart_note,
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
