"""Stage 4: assemble computed metrics, run history, sweep tables and charts for the holdout report.

Mọi số liệu của bốn cách chia được tính lại từ artifact (không chép tay):
- Khối chính lấy từ cây canonical ``--canonical-dir`` (mặc định
  ``outputs/step4_thread1``), nơi pipeline bốn split chạy với ``thread_count=1``.
- Khối tham chiếu lấy từ hai thư mục bàn giao đã đóng băng
  ``outputs/catboost_training`` và ``outputs/backtest``; chỉ đọc, không ghi.

Ngoài hai biểu đồ HTML (Plotly), script xuất thêm hai ảnh PNG (matplotlib) từ
đúng dữ liệu đường vốn để nhúng trực tiếp vào báo cáo Markdown.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citd_ml import paths
from citd_ml.verification.holdout_evidence import load_evidence, validate_evidence


DEFAULT_CANONICAL_DIR = paths.PROJECT_ROOT / "outputs" / "step4_thread1"
DEFAULT_SENSITIVITY_JSON = (
    paths.PROJECT_ROOT / "outputs" / "thread_count_sensitivity" / "thread_count_sensitivity.json"
)
DEFAULT_RUN_HISTORY_JSON = paths.VERIFICATION_DIR / "holdout_run_history.json"
DEFAULT_STAGE5_JSON = paths.HOLDOUT_DIR / "repro" / "stage5_repro_report.json"

METHODS = ("random_kfold", "grouped_kfold", "walk_forward", "purged_walk_forward")
METHOD_LABELS = {
    "random_kfold": "Cách 1 — Random K-Fold",
    "grouped_kfold": "Cách 1b — Grouped K-Fold",
    "walk_forward": "Cách 2 — Walk-forward",
    "purged_walk_forward": "Cách 3 — WF + Purge/Embargo",
}
FINANCIAL_METHOD_LABELS = {
    "baseline": "Baseline (khúc 2–5)",
    **METHOD_LABELS,
    "baseline_holdout": "Baseline holdout",
    "top_50": "Holdout, top 50%",
}
KEEP_LEVELS = (20, 30, 40, 50, 60, 70, 80)
CHUNK_ROW_EDGES = [5001, 10003, 15004, 20006, 25008]
CANONICAL_METRICS_CSV = "metrics_chunk2_5.csv"
TABLE1_BLOCK_CANONICAL = "canonical_thread_count_1"
TABLE1_BLOCK_HANDOVER = "handover_no_thread_pin"
TABLE1_BLOCK_DELTA = "delta_canonical_minus_handover"
TABLE2_BLOCK_CANONICAL = "chunk2_5_thread_count_1"
TABLE2_BLOCK_HANDOVER = "chunk2_5_handover_no_thread_pin"
TABLE2_BLOCK_HOLDOUT = "holdout"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage3-dir",
        type=Path,
        default=paths.HOLDOUT_STAGE3_DIR,
        help="Thư mục chứa output Stage 3 (mặc định: outputs/holdout/stage3).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=paths.HOLDOUT_STAGE4_DIR,
        help="Thư mục ghi bảng và biểu đồ Stage 4 (mặc định: outputs/holdout/stage4).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=paths.DOCS_DIR / "BAO_CAO_KET_QUA_HOLDOUT.md",
        help="Đường dẫn báo cáo Markdown (mặc định: docs/BAO_CAO_KET_QUA_HOLDOUT.md).",
    )
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        default=DEFAULT_CANONICAL_DIR,
        help="Cây canonical bốn split thread_count=1 (mặc định: outputs/step4_thread1).",
    )
    parser.add_argument(
        "--sensitivity-json",
        type=Path,
        default=DEFAULT_SENSITIVITY_JSON,
        help="JSON thí nghiệm thread_count (mặc định: outputs/thread_count_sensitivity/thread_count_sensitivity.json).",
    )
    parser.add_argument(
        "--run-history-json",
        type=Path,
        default=DEFAULT_RUN_HISTORY_JSON,
        help="Sổ ba phiên bản kết quả holdout lịch sử (mặc định: outputs/verification/holdout_run_history.json).",
    )
    parser.add_argument(
        "--stage5-json",
        type=Path,
        default=DEFAULT_STAGE5_JSON,
        help="JSON kiểm lặp run1 vs run2 (mặc định: outputs/holdout/repro/stage5_repro_report.json).",
    )
    parser.add_argument(
        "--canonical-verification-json",
        type=Path,
        default=None,
        help="Manifest verify_pipeline của cây canonical (mặc định: <canonical-dir>/verification/reproducibility.json).",
    )
    return parser.parse_args()


ARTIFACT_DESCRIPTIONS = {
    "dataset_catboost_full_regenerated.csv": "Dataset tái sinh trên toàn bộ lịch sử để kiểm tra và tách holdout",
    "dataset_catboost_holdout.csv": "Dataset holdout từ mốc 2025-02-08 15:30:00",
    "stage1_validation_report.json": "Số liệu kiểm khóa cũ và đối chiếu 23 feature",
    "catboost_final_holdout_run1.cbm": "Model CatBoost cuối dùng để chấm holdout",
    "catboost_final_holdout_run2.cbm": "Model train độc lập lần hai để kiểm lặp",
    "train_predictions_run1.npy": "Prediction trên tập train của lượt 1",
    "train_predictions_run2.npy": "Prediction trên tập train của lượt 2",
    "train_run1_report.json": "Cấu hình, phiên bản, hash và kiểm biên của lượt train 1",
    "train_run2_report.json": "Cấu hình, phiên bản, hash và kiểm biên của lượt train 2",
    "stage2_reproducibility_report.json": "So sánh hai lượt train trên cùng máy",
    "holdout_scored.csv": "5,028 dòng holdout kèm xác suất CatBoost",
    "holdout_fixed_trade_universe_scored.csv": "Vũ trụ lệnh baseline holdout cố định kèm điểm",
    "stage3_backtest_summary.csv": "Metrics baseline và sweep top 20–80%",
    "stage3_report.json": "Kết quả phân loại, replay baseline và backtest holdout",
    "table1_classification_metrics.csv": "Bảng chỉ số phân loại: khối canonical, khối bàn giao và chênh lệch",
    "table2_financial_metrics_top50.csv": "Bảng tài chính top 50%: khối bốn cách chia, khối bàn giao và khối holdout",
    "table3_branch_sweep_20_80.csv": "Sweep 20–80% của bốn nhánh canonical thread_count=1",
    "holdout_sweep_20_80.csv": "Bảng tài chính holdout theo bảy mức giữ lệnh",
    "stage4_tables.md": "Bốn bảng kết quả ở định dạng Markdown",
    "equity-curve-chunk2-5-top50.html": "Biểu đồ bậc thang baseline và 4 nhánh trên khúc 2–5",
    "equity-curve-holdout-top50.html": "Biểu đồ bậc thang baseline và top 50% trên holdout",
    "equity-curve-chunk2-5-top50.png": "Ảnh PNG biểu đồ bậc thang baseline và 4 nhánh trên khúc 2–5",
    "equity-curve-holdout-top50.png": "Ảnh PNG biểu đồ bậc thang baseline và top 50% trên holdout",
    "implementation_decisions.md": "Các quyết định triển khai Stage 4",
    "stage4_manifest.json": "Danh sách output, số điểm trên từng đường vốn và thiết lập xuất PNG",
}


# ------------------------------------------------------------------ số liệu
def chunk1_removed_metrics(oof_dir: Path) -> dict[str, dict[str, float]]:
    """Trung bình ROC-AUC/F1@0.5 theo fold sau khi bỏ khúc 1 (fold >= 1, chunk != 1).

    Chunk chia theo ``row_id`` với biên [0, 5001, 10003, 15004, 20006, 25008]
    (searchsorted side="right"), giống quy tắc của backtest_pyramid_local.py.
    """
    metrics: dict[str, dict[str, float]] = {}
    for method in METHODS:
        path = oof_dir / f"oof_{method}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Thiếu OOF để tính chỉ số: {path}")
        frame = pd.read_csv(path)
        chunk = 1 + np.searchsorted(CHUNK_ROW_EDGES, frame["row_id"].to_numpy(), side="right")
        kept = frame.loc[(frame["fold"] >= 1) & (chunk != 1)]
        if kept.empty:
            raise ValueError(f"Không còn fold nào sau khi bỏ khúc 1: {path}")
        aucs, f1s = [], []
        for _, fold_frame in kept.groupby("fold", sort=True):
            labels = fold_frame["label"].to_numpy()
            probability = fold_frame["probability"].to_numpy()
            aucs.append(roc_auc_score(labels, probability))
            f1s.append(f1_score(labels, probability >= 0.5))
        metrics[method] = {
            "roc_auc": float(np.mean(aucs)),
            "f1_at_0_5": float(np.mean(f1s)),
            "folds": len(aucs),
            "rows": int(len(kept)),
        }
    return metrics


def assert_same_4dp(label: str, actual: float, expected: float, context: str) -> None:
    if f"{actual:.4f}" != f"{expected:.4f}":
        raise ValueError(
            f"{context}: {label} tính được {actual:.10f} nhưng tham chiếu {expected:.10f} "
            "(khác nhau ở 4 chữ số thập phân)"
        )


def canonical_metrics_or_fail(canonical_dir: Path) -> dict[str, dict[str, float]]:
    """Tính lại khối canonical và đối chiếu metrics_chunk2_5.csv của chính cây đó."""
    metrics = chunk1_removed_metrics(canonical_dir / "catboost_training")
    reported_path = canonical_dir / "catboost_training" / CANONICAL_METRICS_CSV
    if not reported_path.exists():
        raise FileNotFoundError(f"Thiếu bảng đối chiếu canonical: {reported_path}")
    reported = pd.read_csv(reported_path).set_index("method")
    for method in METHODS:
        label = f"{method}.roc_auc"
        assert_same_4dp(label, metrics[method]["roc_auc"], float(reported.loc[method, "roc_auc"]), str(reported_path))
        label = f"{method}.f1_at_0_5"
        assert_same_4dp(label, metrics[method]["f1_at_0_5"], float(reported.loc[method, "f1"]), str(reported_path))
    return metrics


def handover_metrics_or_fail(run_history: dict) -> dict[str, dict[str, float]]:
    """Tính lại khối bàn giao từ OOF đóng băng và đối chiếu số đã công bố trong sổ lịch sử."""
    branches = {
        branch["method"]: branch
        for branch in run_history["handover_reference"]["table1_classification_chunk1_removed"]["branches"]
    }
    metrics = chunk1_removed_metrics(paths.CATBOOST_TRAINING_DIR)
    for method in METHODS:
        branch = branches.get(method)
        if branch is None:
            raise ValueError(f"Sổ lịch sử thiếu nhánh bàn giao: {method}")
        assert_same_4dp(
            f"{method}.roc_auc",
            metrics[method]["roc_auc"],
            float(branch["handover_reported_roc_auc"]),
            "holdout_run_history.handover_reference",
        )
        assert_same_4dp(
            f"{method}.f1_at_0_5",
            metrics[method]["f1_at_0_5"],
            float(branch["handover_reported_f1_at_0_5"]),
            "holdout_run_history.handover_reference",
        )
    return metrics


# ---------------------------------------------------------------- định dạng
def fmt_count(value: float) -> str:
    return f"{int(value):,}"


def fmt_r(value: float) -> str:
    return f"{value:+,.2f}"


def fmt_num(value: float) -> str:
    return f"{value:,.2f}"


def fmt_ratio(value: float) -> str:
    return f"{value:.4f}"


def fmt_win(value: float) -> str:
    return f"{value:.2f}"


def fmt_delta(value: float) -> str:
    return f"{value:+.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n"


RUN_DIR_PATTERN = re.compile(r"run[\s_-]*(\d+)$", re.IGNORECASE)


def run_label_from_stage3(stage3_dir: Path) -> str:
    """Nhãn lượt chạy suy từ đường dẫn Stage 3 (``…/repro/run2/stage3`` → ``run 2``)."""
    for part in reversed(stage3_dir.parts):
        match = RUN_DIR_PATTERN.search(part)
        if match:
            return f"run {match.group(1)}"
    return "run 1"


def relative_link(path: Path, report_dir: Path, label: Path | str | None = None) -> str:
    try:
        target = str(path.relative_to(paths.PROJECT_ROOT))
    except ValueError:
        target = str(path)
    link = os.path.relpath(path, report_dir).replace(os.sep, "/")
    return f"[`{target if label is None else label}`]({link})"


# ---------------------------------------------------------------- biểu đồ
def equity_by_close_time(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all same-close-time trade R before creating the line point."""
    ordered = trades.sort_values(["close_time", "ticket"], kind="stable")
    at_time = ordered.groupby("close_time", sort=True, as_index=False)["R"].sum()
    at_time["equity_R"] = at_time["R"].cumsum()
    start = pd.DataFrame({"close_time": [at_time["close_time"].iat[0] - pd.Timedelta(nanoseconds=1)], "R": [0.0], "equity_R": [0.0]})
    return pd.concat([start, at_time], ignore_index=True)


def add_trace(fig: go.Figure, curve: pd.DataFrame, name: str) -> None:
    fig.add_trace(go.Scatter(x=curve["close_time"], y=curve["equity_R"], name=name, mode="lines", line=({"width": 1.6, "shape": "hv"}), hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Equity: %{y:.3f} R<extra>" + name + "</extra>"))


def write_chart(path: Path, title: str, curves: list[tuple[str, pd.DataFrame]]) -> None:
    fig = go.Figure()
    for name, curve in curves:
        add_trace(fig, curve, name)
    fig.update_layout(title=title, xaxis_title="close_time", yaxis_title="R cộng dồn", hovermode="x unified", template="plotly_white", legend_title_text="Chuỗi", margin={"l": 72, "r": 28, "t": 72, "b": 72})
    html = pio.to_html(
        fig,
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False},
        div_id=path.stem,
    )
    path.write_text(html, encoding="utf-8")


def write_png(path: Path, title: str, curves: list[tuple[str, pd.DataFrame]]) -> None:
    """Xuất PNG từ đúng dữ liệu đường vốn; tham số cố định, không timestamp -> byte ổn định."""
    figure, axes = plt.subplots(figsize=(12.0, 5.4), dpi=110)
    for name, curve in curves:
        axes.step(curve["close_time"], curve["equity_R"], where="post", label=name, linewidth=1.4)
    axes.set_title(title)
    axes.set_xlabel("close_time")
    axes.set_ylabel("R cộng dồn")
    axes.grid(True, linewidth=0.4, alpha=0.4)
    axes.legend(loc="upper left", fontsize=8, framealpha=0.9)
    figure.autofmt_xdate()
    figure.tight_layout()
    # metadata={"Software": None} bỏ tag phiên bản matplotlib khỏi PNG, để byte
    # ảnh không phụ thuộc phiên bản thư viện.
    figure.savefig(path, format="png", dpi=110, metadata={"Software": None})
    plt.close(figure)


# ------------------------------------------------------------------ bảng
def build_table1(canonical: dict[str, dict[str, float]], handover: dict[str, dict[str, float]], holdout_classification: dict) -> dict[str, list[dict[str, str]]]:
    main_rows = [
        {"method": method, "roc_auc": fmt_ratio(canonical[method]["roc_auc"]), "f1_at_0_5": fmt_ratio(canonical[method]["f1_at_0_5"])}
        for method in METHODS
    ]
    main_rows.append(
        {
            "method": "holdout",
            "roc_auc": fmt_ratio(float(holdout_classification["roc_auc"])),
            "f1_at_0_5": fmt_ratio(float(holdout_classification["f1_at_0_5"])),
        }
    )
    handover_rows = [
        {"method": method, "roc_auc": fmt_ratio(handover[method]["roc_auc"]), "f1_at_0_5": fmt_ratio(handover[method]["f1_at_0_5"])}
        for method in METHODS
    ]
    delta_rows = [
        {
            "method": method,
            "roc_auc": fmt_delta(canonical[method]["roc_auc"] - handover[method]["roc_auc"]),
            "f1_at_0_5": fmt_delta(canonical[method]["f1_at_0_5"] - handover[method]["f1_at_0_5"]),
        }
        for method in METHODS
    ]
    return {TABLE1_BLOCK_CANONICAL: main_rows, TABLE1_BLOCK_HANDOVER: handover_rows, TABLE1_BLOCK_DELTA: delta_rows}


def build_table2(canonical_dir: Path, stage3_summary: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    canonical_top50 = pd.read_csv(canonical_dir / "backtest" / "backtest_summary_top50.csv").set_index("method")
    handover_top50 = pd.read_csv(paths.BACKTEST_DIR / "backtest_summary_top50.csv").set_index("method")
    if canonical_top50.index.duplicated().any() or handover_top50.index.duplicated().any():
        raise ValueError("Bảng top 50% canonical/bàn giao bị trùng method")

    def rows_from(summary: pd.DataFrame) -> list[dict[str, str]]:
        rows = []
        for method in ("baseline", *METHODS):
            row = summary.loc[method]
            rows.append(
                {
                    "method": method,
                    "trades": fmt_count(row["trades"]),
                    "net_profit_R": fmt_r(row["net_profit_R"]),
                    "max_dd_R": fmt_num(row["max_dd_R"]),
                    "profit_factor": fmt_ratio(row["profit_factor"]),
                    "win_rate_pct": fmt_win(row["win_rate_pct"]),
                }
            )
        return rows

    baseline_holdout = stage3_summary.loc[stage3_summary.keep_pct == 100].iloc[0]
    top50_holdout = stage3_summary.loc[stage3_summary.keep_pct == 50].iloc[0]
    holdout_rows = [
        {
            "method": "baseline_holdout",
            "trades": fmt_count(baseline_holdout.trades),
            "net_profit_R": fmt_r(baseline_holdout.net_profit_R),
            "max_dd_R": fmt_num(baseline_holdout.max_dd_R),
            "profit_factor": fmt_ratio(baseline_holdout.profit_factor),
            "win_rate_pct": fmt_win(baseline_holdout.win_rate_pct),
        },
        {
            "method": "top_50",
            "trades": fmt_count(top50_holdout.trades),
            "net_profit_R": fmt_r(top50_holdout.net_profit_R),
            "max_dd_R": fmt_num(top50_holdout.max_dd_R),
            "profit_factor": fmt_ratio(top50_holdout.profit_factor),
            "win_rate_pct": fmt_win(top50_holdout.win_rate_pct),
        },
    ]
    return {
        TABLE2_BLOCK_CANONICAL: rows_from(canonical_top50),
        TABLE2_BLOCK_HANDOVER: rows_from(handover_top50),
        TABLE2_BLOCK_HOLDOUT: holdout_rows,
    }


def build_table3(canonical_dir: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    sweep = pd.read_csv(canonical_dir / "backtest" / "backtest_retention_sweep.csv")
    baseline = sweep.loc[sweep.method == "baseline"]
    metric_columns = ["trades", "net_profit_R", "max_dd_R", "profit_factor", "win_rate_pct"]
    if len(baseline) == 0 or baseline[metric_columns].drop_duplicates().shape[0] != 1:
        raise ValueError("Baseline trong backtest_retention_sweep.csv phải giống nhau ở mọi mức giữ")
    baseline_row = baseline.iloc[0]
    branches = sweep.loc[sweep.method.isin(METHODS) & sweep.comparison_keep_pct.isin(KEEP_LEVELS)]
    if len(branches) != len(METHODS) * len(KEEP_LEVELS):
        raise ValueError("Sweep canonical phải có đủ 4 nhánh × 7 mức giữ")
    rows: list[dict[str, str]] = []
    for method in METHODS:
        for keep_pct in KEEP_LEVELS:
            row = branches.loc[(branches.method == method) & (branches.comparison_keep_pct == keep_pct)]
            if len(row) != 1:
                raise ValueError(f"Sweep canonical lệch tại {method} × {keep_pct}%")
            row = row.iloc[0]
            rows.append(
                {
                    "method": method,
                    "keep_pct": str(keep_pct),
                    "trades": fmt_count(row.trades),
                    "net_profit_R": fmt_r(row.net_profit_R),
                    "max_dd_R": fmt_num(row.max_dd_R),
                    "profit_factor": fmt_ratio(row.profit_factor),
                    "win_rate_pct": fmt_win(row.win_rate_pct),
                }
            )
    baseline_text = (
        f"Baseline (không lọc, khúc 2–5): {fmt_count(baseline_row.trades)} lệnh, "
        f"{fmt_r(baseline_row.net_profit_R)} R, MaxDD {fmt_num(baseline_row.max_dd_R)} R, "
        f"PF {fmt_ratio(baseline_row.profit_factor)}, win rate {fmt_win(baseline_row.win_rate_pct)}%."
    )
    return rows, baseline_text


def build_table4(stage3_summary: pd.DataFrame) -> list[dict[str, str]]:
    sweep = stage3_summary.loc[stage3_summary.keep_pct.isin(KEEP_LEVELS)].copy()
    if len(sweep) != len(KEEP_LEVELS):
        raise ValueError("Sweep holdout phải có đủ bảy mức giữ")
    return [
        {
            "filter": str(row.filter),
            "trades": fmt_count(row.trades),
            "net_profit_R": fmt_r(row.net_profit_R),
            "max_dd_R": fmt_num(row.max_dd_R),
            "profit_factor": fmt_ratio(row.profit_factor),
            "win_rate_pct": fmt_win(row.win_rate_pct),
        }
        for row in sweep.sort_values("keep_pct").itertuples()
    ]


def table_markdown(headers: list[str], rows: list[list[str]]) -> str:
    return markdown_table(headers, rows)


def table1_markdown_blocks(table1: dict[str, list[dict[str, str]]]) -> str:
    main_rows = [[METHOD_LABELS.get(row["method"], "Holdout"), row["roc_auc"], row["f1_at_0_5"]] for row in table1[TABLE1_BLOCK_CANONICAL]]
    handover_rows = [[METHOD_LABELS[row["method"]], row["roc_auc"], row["f1_at_0_5"]] for row in table1[TABLE1_BLOCK_HANDOVER]]
    delta_rows = [[METHOD_LABELS[row["method"]], row["roc_auc"], row["f1_at_0_5"]] for row in table1[TABLE1_BLOCK_DELTA]]
    headers = ["Cách chia", "ROC-AUC", "F1 @0.5"]
    return (
        "**Khối chính — 4 cách chia chạy lại với `thread_count=1` (cây canonical):**\n\n"
        + table_markdown(headers, main_rows)
        + "\n**Khối tham chiếu — bàn giao (không ghim `thread_count`):**\n\n"
        + table_markdown(headers, handover_rows)
        + "\n**Chênh lệch canonical − bàn giao (tính trên giá trị chưa làm tròn):**\n\n"
        + table_markdown(["Cách chia", "ΔROC-AUC", "ΔF1 @0.5"], delta_rows)
    )


def table2_markdown_blocks(table2: dict[str, list[dict[str, str]]]) -> str:
    headers = ["", "Số lệnh", "Net profit (R)", "MaxDD (R)", "Profit factor", "Win rate %"]

    def block(block_key: str, caption: str) -> str:
        rows = [[FINANCIAL_METHOD_LABELS[row["method"]], row["trades"], row["net_profit_R"], row["max_dd_R"], row["profit_factor"], row["win_rate_pct"]] for row in table2[block_key]]
        return f"**{caption}:**\n\n" + table_markdown(headers, rows)

    return (
        block(TABLE2_BLOCK_CANONICAL, "Khối chính — 4 cách chia trên khúc 2–5, giữ top 50%, `thread_count=1`")
        + "\n"
        + block(TABLE2_BLOCK_HANDOVER, "Khối tham chiếu — bàn giao (không ghim `thread_count`)")
        + "\n"
        + block(TABLE2_BLOCK_HOLDOUT, "Khối holdout — giữ top 50%")
    )


def table3_markdown_rows(table3: list[dict[str, str]]) -> list[list[str]]:
    return [[METHOD_LABELS[row["method"]], f"Top {row['keep_pct']}%", row["trades"], row["net_profit_R"], row["max_dd_R"], row["profit_factor"], row["win_rate_pct"]] for row in table3]


def table4_markdown_rows(table4: list[dict[str, str]]) -> list[list[str]]:
    return [[row["filter"], row["trades"], row["net_profit_R"], row["max_dd_R"], row["profit_factor"], row["win_rate_pct"]] for row in table4]


# ---------------------------------------------------------------- xuất file
def write_tables(out_dir: Path, tables: dict) -> None:
    table1 = tables["table1"]
    table1_rows = []
    for block, rows in (
        (TABLE1_BLOCK_CANONICAL, table1[TABLE1_BLOCK_CANONICAL]),
        (TABLE1_BLOCK_HANDOVER, table1[TABLE1_BLOCK_HANDOVER]),
        (TABLE1_BLOCK_DELTA, table1[TABLE1_BLOCK_DELTA]),
    ):
        table1_rows.extend({"block": block, **row} for row in rows)
    pd.DataFrame(table1_rows, columns=["block", "method", "roc_auc", "f1_at_0_5"]).to_csv(out_dir / "table1_classification_metrics.csv", index=False)

    table2 = tables["table2"]
    table2_rows = []
    for block, rows in (
        (TABLE2_BLOCK_CANONICAL, table2[TABLE2_BLOCK_CANONICAL]),
        (TABLE2_BLOCK_HANDOVER, table2[TABLE2_BLOCK_HANDOVER]),
        (TABLE2_BLOCK_HOLDOUT, table2[TABLE2_BLOCK_HOLDOUT]),
    ):
        table2_rows.extend({"block": block, **row} for row in rows)
    pd.DataFrame(
        table2_rows,
        columns=["block", "method", "trades", "net_profit_R", "max_dd_R", "profit_factor", "win_rate_pct"],
    ).to_csv(out_dir / "table2_financial_metrics_top50.csv", index=False)

    pd.DataFrame(
        tables["table3"],
        columns=["method", "keep_pct", "trades", "net_profit_R", "max_dd_R", "profit_factor", "win_rate_pct"],
    ).to_csv(out_dir / "table3_branch_sweep_20_80.csv", index=False)

    pd.DataFrame(
        tables["table4"],
        columns=["filter", "trades", "net_profit_R", "max_dd_R", "profit_factor", "win_rate_pct"],
    ).to_csv(out_dir / "holdout_sweep_20_80.csv", index=False)

    tables_md = [
        "# Bảng 1 — Chỉ số phân loại\n",
        table1_markdown_blocks(table1),
        "\n# Bảng 2 — Chỉ số tài chính, giữ top 50%\n",
        table2_markdown_blocks(table2),
        "\n# Bảng 3 — Sweep 20–80% của 4 cách chia (thread_count=1)\n",
        tables["baseline_text"] + "\n",
        table_markdown(
            ["Cách chia", "Mức giữ", "Số lệnh", "Net profit (R)", "MaxDD (R)", "Profit factor", "Win rate %"],
            table3_markdown_rows(tables["table3"]),
        ),
        "\n# Bảng 4 — Sweep holdout 20–80%\n",
        table_markdown(["Lọc", "Số lệnh", "Net profit (R)", "MaxDD (R)", "Profit factor", "Win rate %"], table4_markdown_rows(tables["table4"])),
    ]
    (out_dir / "stage4_tables.md").write_text("\n".join(tables_md).rstrip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------- báo cáo
def output_inventory(report_dir: Path) -> list[list[str]]:
    rows = []
    for path in sorted(paths.HOLDOUT_DIR.glob("stage*/*")):
        stage = path.parent.name.removeprefix("stage")
        if path.suffix == ".csv":
            detail = f"{sum(1 for _ in path.open(encoding='utf-8')) - 1:,} dòng"
        elif path.suffix == ".npy":
            detail = f"{len(np.load(path, allow_pickle=False)):,} giá trị"
        else:
            detail = "artifact"
        description = ARTIFACT_DESCRIPTIONS.get(path.name)
        if description is None and path.name.startswith("holdout_trades_top"):
            keep_pct = path.stem.removeprefix("holdout_trades_top")
            description = f"Danh sách lệnh holdout được giữ ở mức top {keep_pct}%"
        if description is None:
            raise ValueError(f"Missing artifact description: {path.name}")
        link = os.path.relpath(path, report_dir).replace(os.sep, "/")
        rows.append([f"[`{path.name}`]({link})", stage, description, detail, f"{path.stat().st_size:,} B"])
    return rows


def sensitivity_section(sensitivity: dict, sensitivity_path: Path) -> str:
    configs = {config["name"]: config for config in sensitivity["configs"]}
    deltas = sensitivity["deltas_vs_tc1_a"]
    facts = sensitivity["conclusion_facts"]
    rows = []
    labels = {"tc1_a": "tc1_a", "tc1_b": "tc1_b (lặp cùng cấu hình)", "tc2": "tc=2", "tc_default": "tc=-1 (mặc định)"}
    for name, label in labels.items():
        config = configs[name]
        delta = deltas[name]
        rows.append(
            [
                label,
                str(config["thread_count"]),
                f"{delta['train']['max_abs_diff']:.6f}",
                f"{delta['holdout']['max_abs_diff']:.6f}",
                f"{delta['holdout']['pearson_corr']:.4f}",
                fmt_ratio(config["classification"]["roc_auc"]),
                fmt_ratio(config["classification"]["f1_at_0_5"]),
                fmt_r(config["top50"]["net_profit_R"]),
                f"{config['top50']['overlap_count_vs_tc1_a']}/{config['top50']['trades']}",
            ]
        )
    table = markdown_table(
        ["Cấu hình", "`thread_count`", "Train max\\|Δp\\|", "Holdout max\\|Δp\\|", "Holdout Pearson r", "Holdout ROC-AUC", "Holdout F1", "Top 50% net R", "Top 50% trùng"],
        rows,
    )
    return (
        "Thí nghiệm có kiểm soát: cùng dữ liệu, feature, hyperparameter và `random_seed=42`, "
        f"chỉ đổi `thread_count` (nguồn: `{sensitivity_path.relative_to(paths.PROJECT_ROOT)}`).\n\n"
        + table
        + "\n"
        + f"Trên máy này, `thread_count` **có** làm thay đổi kết quả: đổi cấu hình làm "
        + f"toàn bộ prediction thay đổi (train {fmt_count(facts['tc2_vs_tc1_a_train_count_diff_gt_1e-12'])}/"
        + f"{fmt_count(sensitivity['data']['train_rows'])} và holdout "
        + f"{fmt_count(facts['tc2_vs_tc1_a_holdout_count_diff_gt_1e-12'])}/"
        + f"{fmt_count(sensitivity['data']['holdout_rows'])} dòng lệch quá `1e-12` ở cả "
        + f"`tc=2` và `tc=-1` so với `tc=1`), kéo theo chọn top 50% và net R đổi. "
        + f"Hai lần chạy cùng cấu hình `tc=1` cho prediction giống hệt nhau "
        + f"(train max|Δp| = {deltas['tc1_b']['train']['max_abs_diff']:.1f}, holdout max|Δp| = "
        + f"{deltas['tc1_b']['holdout']['max_abs_diff']:.1f}), nên khác biệt đến từ việc đổi cấu hình "
        + "chứ không phải bất định giữa hai lần chạy. "
        + f"Ảnh hưởng nằm ở bước train: cùng model đã fit, đổi prediction `thread_count` 1/2/-1 cho "
        + f"prediction giống hệt nhau (max|Δp| = 0.0). CatBoost mô tả `thread_count` là tham số tốc độ "
        + "không ảnh hưởng kết quả; phép đo trên máy này xác nhận điều đó ở bước prediction nhưng "
        + f"không xác nhận ở bước train (holdout ROC-AUC từ "
        + f"{fmt_ratio(facts['holdout_roc_auc_range_across_configs'][0])} "
        + f"đến {fmt_ratio(facts['holdout_roc_auc_range_across_configs'][1])}, "
        + f"top-50 net R từ {fmt_r(facts['holdout_top50_net_profit_R_range_across_configs'][0])} "
        + f"đến {fmt_r(facts['holdout_top50_net_profit_R_range_across_configs'][1])} R). "
        + "Phạm vi: một máy, một build CatBoost, một dataset, một seed; thí nghiệm không quy hết "
        + "sai lệch liên máy cho `thread_count` (xem giới hạn trong JSON).\n"
    )


def data_notes_section(stage3_dir: Path, report_dir: Path) -> str:
    holdout = pd.read_csv(stage3_dir / "holdout_fixed_trade_universe_scored.csv", parse_dates=["entry_time", "label_end_time"])
    frozen = pd.read_csv(paths.DATASET_CSV)
    regenerated = pd.read_csv(paths.HOLDOUT_STAGE1_DIR / "dataset_catboost_full_regenerated.csv", parse_dates=["entry_time", "label_end_time"])
    holdout_start = pd.Timestamp(paths.HOLDOUT_START)
    pre_holdout = regenerated.loc[regenerated.entry_time < holdout_start]
    frozen_keys = set(map(tuple, frozen[["origin_bar", "entry_bar", "leg"]].to_numpy()))
    boundary = pre_holdout.loc[[key not in frozen_keys for key in map(tuple, pre_holdout[["origin_bar", "entry_bar", "leg"]].to_numpy())]]
    if len(boundary) == 0:
        raise ValueError("Không tìm thấy lệnh biên nào giữa dataset tái sinh và dataset đóng băng")
    boundary_rows = [
        [
            str(row.entry_time),
            str(row.label_end_time),
            fmt_count(row.origin_bar),
            fmt_count(row.entry_bar),
            str(row.leg),
        ]
        for row in boundary.sort_values(["entry_time", "leg"]).itertuples()
    ]
    total_check = len(frozen) + len(boundary) + len(holdout)
    regenerated_rows = len(regenerated)
    if total_check != regenerated_rows:
        raise ValueError(
            f"25,008 + lệnh biên + holdout = {total_check:,} nhưng dataset tái sinh có {regenerated_rows:,} dòng"
        )
    period_text = (
        f"Holdout: entry_time từ {holdout.entry_time.min()} đến {holdout.entry_time.max()}; "
        f"label_end_time muộn nhất {holdout.label_end_time.max()}."
    )
    return (
        f"- {period_text}\n"
        f"- {len(boundary)} lệnh biên (entry_time trước mốc holdout nhưng không có trong dataset đóng băng "
        f"`{paths.DATASET_CSV.relative_to(paths.PROJECT_ROOT)}`) được xác định bằng so khớp khóa "
        f"`(origin_bar, entry_bar, leg)`, không chép tay:\n\n"
        + markdown_table(["entry_time", "label_end_time", "origin_bar", "entry_bar", "leg"], boundary_rows)
        + f"\n- {len(boundary)} lệnh này không thuộc train và cũng không thuộc holdout: "
        f"{fmt_count(len(frozen))} (train) + {len(boundary)} (biên) + {fmt_count(len(holdout))} (holdout) "
        f"= {fmt_count(regenerated_rows)} dòng của dataset tái sinh toàn lịch sử.\n"
        f"- Quy ước chữ số: ROC-AUC/F1/PF 4 chữ số thập phân; net R và MaxDD 2 chữ số thập phân, "
        f"dùng dấu phẩy phân cách hàng nghìn và R có dấu; win rate 2 chữ số thập phân; "
        f"số lệnh dùng dấu phẩy phân cách hàng nghìn.\n"
    )


def verification_section(evidence: dict, canonical_verification_path: Path) -> str:
    stage1 = evidence["stage1"]
    train = evidence["train"]
    repeat = evidence["repeat"]
    stage5 = evidence["stage5"]
    stage5_items = stage5["items"]
    versions = train["versions"]
    counts = stage1["counts"]
    comparison = evidence["stage3"]["baseline_comparison"]
    mismatch_fields = ", ".join(comparison["mismatched_fields"]) or "không có"
    charts = stage5_items["stage4_charts"]["charts"]
    classification_deltas = stage5_items["stage3_report_classification_and_sweep"]["classification"]
    backtest_deltas = stage5_items["stage3_backtest_summary"]["numeric_columns"]
    max_backtest_delta = max(entry["max_abs_diff"] for entry in backtest_deltas.values())
    stage5_text = (
        f"**{stage5['status']}** — max \\|Δprobability\\| = "
        f"{stage5_items['holdout_scored_probabilities']['max_abs_diff']:.1f}; "
        f"ΔROC-AUC = {classification_deltas['roc_auc']['abs_diff']:.1f}; "
        f"ΔF1 = {classification_deltas['f1_at_0_5']['abs_diff']:.1f}; "
        f"max Δmetric backtest = {max_backtest_delta:.1f}; "
        f"bảng số giống hệt nhau; biểu đồ byte-identical: "
        + ", ".join(f"`{name}`" for name in sorted(charts))
    )
    canonical_verification = evidence["canonical_verification"]
    output_dirs = canonical_verification["output_dirs"]
    verification_status = "PASS" if canonical_verification["status"] == "passed" else str(canonical_verification["status"]).upper()
    verification_text = (
        f"**{verification_status}** — manifest `{os.path.relpath(canonical_verification_path, paths.PROJECT_ROOT)}`; "
        f"phạm vi: `{output_dirs['training']}` + `{output_dirs['backtest']}`; "
        f"hai thư mục bàn giao đóng băng không bị ghi đè và không tái lập byte trên Linux"
    )
    cross_machine = (
        "**GIỚI HẠN** — teammate Windows báo prediction trùng trong `1e-15` (không byte-identical, "
        "không ghi metric hay artifact đối chứng); không thể tái tạo từ git."
    )
    rows = [
        ["Dataset tái sinh trước holdout", f"**PASS** — {counts['matching_keys']:,}/{counts['frozen_rows']:,} khóa cũ; thiếu {counts['missing_frozen_keys']}"],
        ["Giá trị feature", f"**PASS** — {counts['feature_cells_compared']:,} ô đã so; lệch {counts['feature_cells_mismatched']}"],
        ["Purge / embargo tại biên holdout", f"**PASS** — cắt {train['purge_rows']} / {train['embargo_rows']} dòng; train còn {train['train_rows_after_filtering']:,}"],
        ["Holdout không bị sửa khi train/chấm điểm", "**PASS** — SHA-256 trước/sau giữ nguyên"],
        ["Replay baseline so với tradelist bàn giao", f"**PASS** — {comparison['actual_holdout_trades']:,}/{comparison['reference_holdout_trades']:,} lệnh; trường lệch: {mismatch_fields}"],
        [
            "Lặp train trên cùng máy (Stage 2)",
            f"**PASS** — {repeat['prediction_count']:,} predictions giống hệt; max abs diff = "
            f"{repeat['max_prediction_abs_difference']}; `train_run1_report.json`/`train_run2_report.json` đã ghi "
            f"`platform` ({versions['platform']}) và `plotly` ({versions['plotly']})",
        ],
        ["Tái lập run1 vs run2 (Stage 3–4: chấm điểm holdout, backtest, bảng, biểu đồ)", stage5_text],
        ["`verify_pipeline` trên cây canonical `thread_count=1`", verification_text],
        [
            "File model `.cbm`",
            f"**GHI NHẬN** — SHA-256 hai file khác nhau (`{repeat['model_run1_sha256'][:12]}…` vs "
            f"`{repeat['model_run2_sha256'][:12]}…`); binary model không phải tiêu chí PASS",
        ],
        ["Kiểm chứng liên máy", cross_machine],
    ]
    footer = (
        f"\nMôi trường sinh artifact: OS {versions['platform']}; Python {versions['python']}; "
        f"CatBoost {versions['catboost']}; scikit-learn {versions['scikit_learn']}; "
        f"pandas {versions['pandas']}; NumPy {versions['numpy']}; Plotly {versions['plotly']}; "
        f"Matplotlib {versions['matplotlib']}.\n"
    )
    return markdown_table(["Phép kiểm", "Kết quả"], rows) + footer


def ledger_section(run_history: dict, stage3_dir: Path, canonical_dir: Path) -> str:
    runs = run_history["runs"]
    if len(runs) != 3:
        raise ValueError(f"Sổ lịch sử phải có đúng ba phiên bản kết quả holdout, thấy {len(runs)}")
    rows = []
    for run in runs:
        environment = run["environment"]
        env_text = environment["value"] + ("" if environment.get("recorded") else " (suy từ đường dẫn)")
        thread_count = run.get("thread_count")
        classification = run["classification"]
        top50 = run["top50_holdout"]
        rows.append(
            [
                run["label"],
                f"`{run['commit_short']}`",
                str(run["commit_date"])[:10],
                env_text,
                "không ghim" if thread_count is None else f"`{thread_count}`",
                fmt_ratio(classification["roc_auc"]),
                fmt_ratio(classification["f1_at_0_5"]),
                fmt_r(top50["net_profit_R"]),
                fmt_ratio(top50["profit_factor"]),
            ]
        )
    baseline_net_r = {round(run["baseline_holdout"]["net_profit_R"], 6) for run in runs}
    if len(baseline_net_r) != 1:
        raise ValueError("Baseline holdout phải giống nhau ở cả ba lần mở")
    external = run_history["external_verification"]
    claims = "\n".join(f"- {claim}" for claim in external["claims"])
    canonical = runs[-1]
    return (
        markdown_table(
            ["Phiên bản", "Commit", "Ngày commit", "Môi trường suy ra", "`thread_count`", "ROC-AUC", "F1 @0.5", "Top 50% net R", "Top 50% PF"],
            rows,
        )
        + f"\nBaseline holdout giống nhau ở cả ba lần: {fmt_count(runs[0]['baseline_holdout']['trades'])} lệnh, "
        + f"{fmt_r(runs[0]['baseline_holdout']['net_profit_R'])} R (baseline là tradelist tĩnh, không phụ thuộc model).\n"
        + f"\nXác minh độc lập của teammate (`external_verification` trong sổ lịch sử): "
        + f"{external['reported_by']}, ngày {external['reported_on']}, môi trường {external['reported_environment']}, "
        + f"nhánh `{external['branch']}`, lệnh `{external['command']}`.\n\n"
        + claims
        + "\n\nTheo tin nhắn teammate: prediction trên tập train giữa hai máy trùng trong `1e-15` "
        + "(không byte-identical); lần xác minh này không ghi lại metric nào và không có artifact "
        + "đối chứng được commit, nên chỉ là thông tin tham khảo.\n"
        + f"\nArtifact canonical của báo cáo: lần 3 — commit `{canonical['commit_short']}`, "
        + f"`thread_count={canonical['thread_count']}`. Các bảng holdout lấy từ "
        + f"`{stage3_dir.relative_to(paths.PROJECT_ROOT)}` ({run_label_from_stage3(stage3_dir)}) và bảng bốn cách chia canonical lấy từ cây "
        + f"`{canonical_dir.relative_to(paths.PROJECT_ROOT)}`; "
        + f"hai thư mục bàn giao đóng băng `{paths.CATBOOST_TRAINING_DIR.relative_to(paths.PROJECT_ROOT)}` và "
        + f"`{paths.BACKTEST_DIR.relative_to(paths.PROJECT_ROOT)}` chỉ còn là khối tham chiếu.\n"
        + "\nLý do chạy lại: cấu hình CatBoost ban đầu chưa ghim `thread_count`; lần 3 chỉ thêm "
        + "`thread_count=1` như thiết lập kỹ thuật, không đổi feature, hyperparameter mô hình, tập train, "
        + "tập holdout hay luật chọn top-k. Các số lịch sử truy xuất được giữ trong sổ.\n"
    )


def evidence_section(report_dir: Path, stage5_path: Path, canonical_dir: Path, run_history_path: Path) -> str:
    canonical_label = str(canonical_dir.relative_to(paths.PROJECT_ROOT)) + "/"
    rows = [
        [
            relative_link(canonical_dir, report_dir, canonical_label),
            "Cây canonical bốn split `thread_count=1`: `comparison_report.json`, `verification/reproducibility.json`, OOF, hai bảng backtest và `metrics_chunk2_5.csv` dùng cho Bảng 1–3.",
        ],
        [
            relative_link(paths.PROJECT_ROOT / "outputs" / "thread_count_sensitivity" / "thread_count_sensitivity.json", report_dir),
            "Thí nghiệm `thread_count`: bốn cấu hình, deltas prediction, top-50 và `conclusion_facts` lưu tại Phụ lục A.",
        ],
        [
            relative_link(run_history_path, report_dir),
            "Ba phiên bản kết quả holdout lịch sử truy xuất từ Git: AUC/F1, top-50, sweep, pairwise deltas, xác minh teammate và khối tham chiếu bàn giao.",
        ],
        [
            relative_link(stage5_path, report_dir),
            "Kết quả đối chiếu run1 vs run2 toàn chuỗi Stage 3–4: status, deltas số học và hash bảng/biểu đồ dùng cho Phần 4.",
        ],
        [
            relative_link(canonical_dir / "verification" / "reproducibility.json", report_dir),
            "Manifest `verify_pipeline` của cây canonical: hai lượt train/backtest độc lập, hash output và phiên bản môi trường.",
        ],
        [
            relative_link(paths.PROJECT_ROOT / "outputs" / "holdout" / "repro" / "run2", report_dir, "outputs/holdout/repro/run2/"),
            "Báo cáo và artifact run 2 độc lập (`stage3`, `stage4`, `BAO_CAO_holdout_run2.md`) để đối chiếu run1.",
        ],
    ]
    return markdown_table(["Đường dẫn", "Chứa gì"], rows)


def build_report(
    tables: dict,
    evidence: dict,
    report_path: Path,
    stage5_path: Path,
    out_dir: Path,
    canonical_dir: Path,
    canonical_verification_path: Path,
    run_history_path: Path,
    stage3_dir: Path,
    sensitivity_path: Path,
) -> None:
    table1 = tables["table1"]
    train = evidence["train"]
    versions = train["versions"]
    counts = evidence["stage1"]["counts"]
    run_history = evidence["run_history"]
    param_text = " · ".join(f"`{key}={value}`" for key, value in train["params"].items())
    environment_table = markdown_table(
        ["Thành phần", "Giá trị"],
        [
            ["OS (`platform`)", f"`{versions['platform']}`"],
            ["Python", f"`{versions['python']}`"],
            ["CatBoost", f"`{versions['catboost']}`"],
            ["scikit-learn", f"`{versions['scikit_learn']}`"],
            ["pandas", f"`{versions['pandas']}`"],
            ["NumPy", f"`{versions['numpy']}`"],
            ["Plotly", f"`{versions['plotly']}`"],
            ["Matplotlib", f"`{versions['matplotlib']}`"],
        ],
    )
    report_dir = report_path.parent
    png1_rel = os.path.relpath(out_dir / "equity-curve-chunk2-5-top50.png", report_dir).replace(os.sep, "/")
    png2_rel = os.path.relpath(out_dir / "equity-curve-holdout-top50.png", report_dir).replace(os.sep, "/")
    html1_rel = os.path.relpath(out_dir / "equity-curve-chunk2-5-top50.html", report_dir).replace(os.sep, "/")
    html2_rel = os.path.relpath(out_dir / "equity-curve-holdout-top50.html", report_dir).replace(os.sep, "/")

    report = f"""# BÁO CÁO KẾT QUẢ TRAIN VÀ HOLDOUT

> Phạm vi: tổng hợp artifact đã chốt của 4 cách chia trên tập 80% đầu và kết
> quả holdout niêm phong. Báo cáo chỉ ghi số liệu, quyết định triển khai và bằng
> chứng kiểm chứng; không biện luận hay kết luận thay báo cáo chính.

## Lịch sử mở holdout và lý do chạy lại

Bảng A ghi ba phiên bản kết quả holdout lịch sử truy xuất được từ Git, lưu trong
`{run_history_path.relative_to(paths.PROJECT_ROOT)}`. Đây không phải nhật ký đầy đủ
của mọi lần thực thi hoặc chấm điểm: sổ dùng ba commit đã xác định, không ghi nhận
các lần không được lưu vào Git. Ngày trong bảng là ngày commit, không phải log thời điểm chạy.
Các lần lặp để kiểm chứng và thí nghiệm bổ sung ngày 15/09 được trình bày riêng ở Phụ lục A và Phần 4.

{ledger_section(run_history, stage3_dir, canonical_dir)}
## Phần 1 — Số liệu

### 1.1. Cấu hình train cuối

| Thuộc tính | Giá trị |
|---|---|
| Tập train trước/sau purge + embargo | {train['train_rows_before_filtering']:,} / {train['train_rows_after_filtering']:,} dòng |
| Tập holdout | {counts['holdout_rows']:,} dòng; không bỏ dòng |
| Feature đầu vào | {len(train['features'])} `FEATURES`; không đưa `META` vào `X` |
| Hyperparameter và thiết lập kỹ thuật | {param_text} |

Môi trường sinh artifact (từ `train_run1_report.json`):

{environment_table}
### 1.2. Bảng 1 — Chỉ số phân loại

{table1_markdown_blocks(table1)}
Ghi chú: bốn dòng Cách 1–3 của khối canonical là trung bình chỉ số theo fold sau
khi bỏ khúc 1 để các nhánh được đo trên cùng phạm vi khúc 2–5, tính lại từ bốn
file OOF trong cây canonical và khớp `metrics_chunk2_5.csv` ở 4 chữ số thập
phân. Khối bàn giao được tính lại từ bốn OOF đóng băng trong
`outputs/catboost_training` và khớp số đã công bố trong sổ lịch sử. Dòng Holdout
được tính một lần trên toàn bộ {counts['holdout_rows']:,} mẫu holdout, không phải
trung bình theo fold.

### 1.3. Bảng 2 — Chỉ số tài chính, giữ top 50%

{table2_markdown_blocks(tables['table2'])}
### 1.4. Bảng 3 — Sweep 20–80% của 4 cách chia (`thread_count=1`)

{tables['baseline_text']}

{table_markdown(
        ["Cách chia", "Mức giữ", "Số lệnh", "Net profit (R)", "MaxDD (R)", "Profit factor", "Win rate %"],
        table3_markdown_rows(tables['table3']),
    )}
### 1.5. Bảng 4 — Sweep holdout 20–80%

{table_markdown(["Lọc", "Số lệnh", "Net profit (R)", "MaxDD (R)", "Profit factor", "Win rate %"], table4_markdown_rows(tables['table4']))}
### 1.6. Cấu hình số luồng

Cấu hình chính thức cố định `thread_count=1`. Sau đó, ngày 15/09/2026 đã chạy thí nghiệm đổi `thread_count`, có chấm holdout ở `thread_count=2` và `-1`. Số liệu chính thức không đổi. Chi tiết thí nghiệm đã thực hiện được lưu tại Phụ lục A.

### 1.7. Ghi chú dữ liệu

{data_notes_section(Path(paths.HOLDOUT_STAGE3_DIR), report_dir)}
## Phần 2 — Biểu đồ

![Biểu đồ 1 — Baseline và 4 nhánh top 50%, khúc 2–5]({png1_rel})

![Biểu đồ 2 — Baseline holdout và Holdout top 50%]({png2_rel})

- [Biểu đồ 1 — bản tương tác HTML]({html1_rel})
- [Biểu đồ 2 — bản tương tác HTML]({html2_rel})

Biểu đồ HTML dựng bằng Python/Plotly; ảnh PNG xuất lại từ đúng dữ liệu đường
vốn bằng matplotlib (`dpi=110`, figsize 12.0 × 5.4 inch, không có timestamp
trong phần chữ). Cả hai dạng đều dùng đường bậc thang ngang-rồi-dọc
(`hv`/`steps-post`): equity chỉ thay đổi tại `close_time`, không nội suy tuyến
tính giữa hai lần đóng lệnh. Trục dọc là R cộng dồn từ 0. File HTML chứa Plotly
nội tuyến nên mở độc lập được.

## Phần 3 — Quyết định triển khai

1. Dùng `ceil(n × keep_pct / 100)` cho số lệnh giữ lại: top 50% của {fmt_count(tables['chunk25_baseline_trades'])} lệnh là {fmt_count(tables['chunk25_top50_trades'])}; top 50% của {fmt_count(counts['holdout_rows'])} lệnh là {fmt_count(tables['holdout_top50_trades'])}.
2. Xếp `probability` giảm dần, dùng `row_id` tăng dần để phá hòa; vũ trụ lệnh baseline giữ cố định nên lệnh bị loại không làm đổi tín hiệu sau đó.
3. Equity ghi nhận tại `close_time`; các lệnh đóng cùng lúc được cộng thành một điểm rồi mới cập nhật đường vốn.
4. Khối chính của Bảng 1–3 là cây canonical `thread_count=1` (`outputs/step4_thread1`); số bàn giao (không ghim `thread_count`) chỉ còn là khối tham chiếu được dán nhãn, kèm bảng chênh lệch canonical − bàn giao ở Bảng 1.
5. Cố định `thread_count=1` cho cấu hình chính thức.
6. Dùng đường bậc thang ngang-rồi-dọc (`hv` cho HTML, `steps-post` cho PNG) để equity giữ nguyên giữa hai mốc đóng lệnh và chỉ nhảy tại thời điểm R được ghi nhận.
7. Purge và embargo đều trả về 0 dòng, nên train giữ nguyên {train['train_rows_after_filtering']:,} dòng; điều kiện lọc được chạy trước khi quyết định không loại dòng nào.
8. So khớp giá baseline dùng sai số tuyệt đối `5e-4`, theo validator bàn giao; giá vào, giá ra và R được kiểm dưới cùng ngưỡng này.
9. Chấm điểm holdout lấy danh sách 23 `FEATURES` trực tiếp từ `build_features.py`; không tự liệt kê cột.
10. Biểu đồ 1 dùng `backtest_scored_universe.csv` của cây canonical và chỉ nhận các dòng `chunk` 2–5; biểu đồ 2 bắt đầu lại từ R = 0 tại holdout, dùng trực tiếp vũ trụ baseline và danh sách Top 50% đã lưu từ Giai đoạn 3.
11. Bốn lệnh biên không thuộc tập train và không được đưa vào chỉ số đánh giá holdout. Replay toàn lịch sử vẫn xử lý giai đoạn này để duy trì trạng thái chiến lược; các lệnh được liệt kê ở mục 1.7.
12. Hai thư mục bàn giao `outputs/catboost_training` và `outputs/backtest` chỉ được đọc, không bị ghi đè; mọi bảng canonical, sweep và biểu đồ đều lấy từ cây `outputs/step4_thread1`.

## Phần 4 — Kiểm chứng

{verification_section(evidence, canonical_verification_path)}
## Phần 5 — Bảng file sinh ra

{markdown_table(["File", "Giai đoạn", "Chứa gì", "Quy mô", "Kích thước"], output_inventory(report_dir))}
### Bằng chứng bổ sung

{evidence_section(report_dir, stage5_path, canonical_dir, run_history_path)}

## Phụ lục A — Thí nghiệm `thread_count` đã thực hiện

{sensitivity_section(evidence['sensitivity'], sensitivity_path)}"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    stage3_dir = Path(args.stage3_dir).resolve()
    canonical_dir = Path(args.canonical_dir).resolve()
    report_path = Path(args.report).resolve()
    run_history_path = Path(args.run_history_json).resolve()
    sensitivity_path = Path(args.sensitivity_json).resolve()
    stage5_path = Path(args.stage5_json).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_verification_path = (
        Path(args.canonical_verification_json).resolve()
        if args.canonical_verification_json is not None
        else canonical_dir / "verification" / "reproducibility.json"
    )
    evidence = load_evidence(
        stage3_dir,
        canonical_dir=canonical_dir,
        run_history_json=run_history_path,
        sensitivity_json=sensitivity_path,
        stage5_json=stage5_path,
        canonical_verification_json=canonical_verification_path,
    )
    validate_evidence(evidence)

    output_dirs = evidence["canonical_verification"]["output_dirs"]
    expected_dirs = {
        "training": str((canonical_dir / "catboost_training").relative_to(paths.PROJECT_ROOT)),
        "backtest": str((canonical_dir / "backtest").relative_to(paths.PROJECT_ROOT)),
    }
    for key, expected in expected_dirs.items():
        if Path(output_dirs[key]) != Path(expected):
            raise ValueError(
                f"Manifest verify_pipeline không thuộc cây canonical: {key} = {output_dirs[key]!r}, cần {expected!r}"
            )

    canonical_metrics = canonical_metrics_or_fail(canonical_dir)
    handover_metrics = handover_metrics_or_fail(evidence["run_history"])
    stage3_summary = pd.read_csv(stage3_dir / "stage3_backtest_summary.csv")

    table1 = build_table1(canonical_metrics, handover_metrics, evidence["stage3"]["classification"])
    table2 = build_table2(canonical_dir, stage3_summary)
    table3, table3_baseline_text = build_table3(canonical_dir)
    table4 = build_table4(stage3_summary)
    tables = {
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "table4": table4,
        "baseline_text": table3_baseline_text,
        "chunk25_baseline_trades": int(table2[TABLE2_BLOCK_CANONICAL][0]["trades"].replace(",", "")),
        "chunk25_top50_trades": int(table2[TABLE2_BLOCK_CANONICAL][1]["trades"].replace(",", "")),
        "holdout_top50_trades": int(table2[TABLE2_BLOCK_HOLDOUT][1]["trades"].replace(",", "")),
    }
    write_tables(out_dir, tables)

    # Chart 1: canonical tree, chunk 2–5 only, fixed top-50 rule.
    universe = pd.read_csv(canonical_dir / "backtest" / "backtest_scored_universe.csv", parse_dates=["close_time"])
    evaluation = universe.loc[universe.chunk.isin([2, 3, 4, 5])].copy()
    expected_trades = tables["chunk25_baseline_trades"]
    if len(evaluation) != expected_trades:
        raise ValueError(f"Số lệnh chunk 2–5 canonical là {len(evaluation):,}, lệch baseline {expected_trades:,}")
    curves1 = [("Baseline", equity_by_close_time(evaluation))]
    names = {"random_kfold": "Cách 1 — Random K-Fold", "grouped_kfold": "Cách 1b — Grouped K-Fold", "walk_forward": "Cách 2 — Walk-forward", "purged_walk_forward": "Cách 3 — WF + Purge/Embargo"}
    expected = int(np.ceil(len(evaluation) * .5))
    for key, label in names.items():
        selected = evaluation.sort_values([f"probability_{key}", "row_id"], ascending=[False, True], kind="stable").iloc[:expected]
        if len(selected) != expected:
            raise ValueError("Top-50 selection size is invalid")
        curves1.append((label, equity_by_close_time(selected)))
    write_chart(out_dir / "equity-curve-chunk2-5-top50.html", "Đường vốn: baseline và 4 nhánh, khúc 2–5, top 50%", curves1)
    write_png(out_dir / "equity-curve-chunk2-5-top50.png", "Đường vốn: baseline và 4 nhánh, khúc 2–5, top 50%", curves1)

    # Chart 2: only consumes the Stage 3 fixed baseline universe and its saved top-50 list.
    baseline = pd.read_csv(stage3_dir / "holdout_fixed_trade_universe_scored.csv", parse_dates=["close_time"])
    top50 = pd.read_csv(stage3_dir / "holdout_trades_top50.csv", parse_dates=["close_time"])
    baseline_holdout_rows = int(stage3_summary.loc[stage3_summary.keep_pct == 100, "trades"].iloc[0])
    top50_holdout_rows = int(stage3_summary.loc[stage3_summary.keep_pct == 50, "trades"].iloc[0])
    if len(baseline) != baseline_holdout_rows or len(top50) != top50_holdout_rows:
        raise ValueError(
            f"Artifact Stage 3 sai số dòng: baseline {len(baseline):,}/{baseline_holdout_rows:,}, "
            f"top 50% {len(top50):,}/{top50_holdout_rows:,}"
        )
    curves2 = [("Baseline holdout", equity_by_close_time(baseline)), ("Holdout, top 50%", equity_by_close_time(top50))]
    write_chart(out_dir / "equity-curve-holdout-top50.html", "Đường vốn: baseline và holdout top 50%", curves2)
    write_png(out_dir / "equity-curve-holdout-top50.png", "Đường vốn: baseline và holdout top 50%", curves2)

    decisions = """# Quyết định triển khai

1. Trục thời gian của cả hai biểu đồ dùng `close_time`, vì R của một lệnh chỉ hoàn tất tại thời điểm đóng lệnh.
2. Các lệnh có cùng `close_time` được sắp xếp ổn định theo `ticket`, cộng R của chúng thành một điểm thời gian duy nhất; điểm đó là equity sau toàn bộ lệnh đóng cùng lúc.
3. Mỗi đường được thêm điểm R = 0 ngay trước `close_time` đầu tiên, để đường vốn bắt đầu từ 0 mà không thay đổi mốc dữ liệu giao dịch.
4. Biểu đồ 1 chỉ nhận các dòng `chunk` 2–5 của cây canonical `{canonical}`; top 50% của từng nhánh dùng `ceil(n × 50%)`, xếp xác suất giảm dần rồi `row_id` tăng dần khi hòa điểm — đúng luật đã chốt ở bước trước.
5. Biểu đồ 2 bắt đầu lại từ R = 0 tại holdout, dùng trực tiếp vũ trụ baseline và danh sách Top 50% đã lưu từ Giai đoạn 3; không nối với equity của khúc 2–5.
6. Đường vốn dùng dạng bậc thang ngang-rồi-dọc (`hv`) cho HTML; PNG xuất bằng matplotlib từ đúng dữ liệu đó với `drawstyle="steps-post"`, `dpi=110`, figsize cố định và không có timestamp trong phần chữ nên byte ổn định giữa các lần chạy.
7. Bảng 1–3 dùng khối canonical `thread_count=1`; số bàn giao không ghim `thread_count` được giữ nguyên trong khối tham chiếu dán nhãn và không trộn vào khối chính.
8. Bảng 3 lấy trực tiếp từ `{canonical}/backtest/backtest_retention_sweep.csv`; baseline chỉ nêu một lần phía trên bảng.
9. Bốn lệnh biên được liệt kê trong mục 1.7: không thuộc tập train và không được đưa vào chỉ số đánh giá holdout. Replay toàn lịch sử vẫn xử lý giai đoạn này để duy trì trạng thái chiến lược.
10. Purge và embargo đều trả về {purge_rows} dòng, nên train giữ nguyên {train_rows:,} dòng; điều kiện lọc được chạy trước khi quyết định không loại dòng nào.
11. So khớp giá baseline dùng sai số tuyệt đối `5e-4`, theo validator bàn giao; giá vào, giá ra và R được kiểm dưới cùng ngưỡng này.
12. Chấm điểm holdout lấy danh sách 23 `FEATURES` trực tiếp từ `build_features.py`; không tự liệt kê cột.
13. Hai thư mục bàn giao `outputs/catboost_training` và `outputs/backtest` chỉ được đọc, không bị ghi đè; mọi bảng canonical, sweep và biểu đồ đều lấy từ cây `{canonical}`.
"""
    (out_dir / "implementation_decisions.md").write_text(
        decisions.format(
            canonical=canonical_dir.relative_to(paths.PROJECT_ROOT),
            train_rows=evidence["train"]["train_rows_after_filtering"],
            purge_rows=evidence["train"]["purge_rows"],
            embargo_rows=evidence["train"]["embargo_rows"],
        ),
        encoding="utf-8",
    )
    manifest = {
        "new_files": sorted(p.name for p in out_dir.iterdir()),
        "chart1_rows": {name: len(curve) for name, curve in curves1},
        "chart2_rows": {name: len(curve) for name, curve in curves2},
        "png_export": {"dpi": 110, "figsize_inches": [12.0, 5.4], "drawstyle": "steps-post", "no_timestamps": True},
    }
    (out_dir / "stage4_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    build_report(
        tables,
        evidence,
        report_path,
        stage5_path,
        out_dir,
        canonical_dir,
        canonical_verification_path,
        run_history_path,
        stage3_dir,
        sensitivity_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
