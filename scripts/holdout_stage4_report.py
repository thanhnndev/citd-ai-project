"""Stage 4 only: assemble fixed metrics and render the two requested HTML curves."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citd_ml import paths
from citd_ml.verification.holdout_evidence import load_evidence, validate_evidence


SOURCE = paths.BACKTEST_DIR


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
    "table1_classification_metrics.csv": "Bảng chỉ số phân loại train và holdout",
    "table2_financial_metrics_top50.csv": "Bảng tài chính tại mức giữ top 50%",
    "holdout_sweep_20_80.csv": "Bảng tài chính holdout theo bảy mức giữ lệnh",
    "stage4_tables.md": "Ba bảng kết quả ở định dạng Markdown",
    "equity-curve-chunk2-5-top50.html": "Biểu đồ bậc thang baseline và 4 nhánh trên khúc 2–5",
    "equity-curve-holdout-top50.html": "Biểu đồ bậc thang baseline và top 50% trên holdout",
    "implementation_decisions.md": "Các quyết định triển khai Stage 4",
    "stage4_manifest.json": "Danh sách output và số điểm trên từng đường vốn",
}


# First four rows are the handed-over classification reference (chunk 1 removed).
# The Holdout row is filled at runtime from stage3_report.json so it cannot drift.
BRANCH_TABLE1 = [
    ["Cách 1 — Random K-Fold", "0.8595", "0.6525"],
    ["Cách 1b — Grouped K-Fold", "0.7475", "0.5105"],
    ["Cách 2 — Walk-forward", "0.5867", "0.3267"],
    ["Cách 3 — WF + Purge/Embargo", "0.5948", "0.3304"],
]


def fmt_r(value: float) -> str:
    return f"{value:+,.2f}"


def fmt_num(value: float) -> str:
    return f"{value:,.2f}"


def fmt_ratio(value: float) -> str:
    return f"{value:.4f}"


def equity_by_close_time(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all same-close-time trade R before creating the line point."""
    ordered = trades.sort_values(["close_time", "ticket"], kind="stable")
    at_time = ordered.groupby("close_time", sort=True, as_index=False)["R"].sum()
    at_time["equity_R"] = at_time["R"].cumsum()
    start = pd.DataFrame({"close_time": [at_time["close_time"].iat[0] - pd.Timedelta(nanoseconds=1)], "R": [0.0], "equity_R": [0.0]})
    return pd.concat([start, at_time], ignore_index=True)


def add_trace(fig: go.Figure, curve: pd.DataFrame, name: str) -> None:
    fig.add_trace(go.Scatter(x=curve["close_time"], y=curve["equity_R"], name=name, mode="lines", line={"width": 1.6, "shape": "hv"}, hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Equity: %{y:.3f} R<extra>" + name + "</extra>"))


def write_chart(path: Path, title: str, curves: list[tuple[str, pd.DataFrame]]) -> None:
    fig = go.Figure()
    for name, curve in curves:
        add_trace(fig, curve, name)
    fig.update_layout(title=title, xaxis_title="close_time", yaxis_title="R cộng dồn", hovermode="x unified", template="plotly_white", legend_title_text="Chuỗi", margin={"l": 72, "r": 28, "t": 72, "b": 72})
    html = pio.to_html(
        fig,
        include_plotlyjs="inline",
        full_html=True,
        config={"responsive": True, "displaylogo": False},
        div_id=path.stem,
    )
    path.write_text(html, encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n"


def output_inventory() -> list[list[str]]:
    rows = []
    for path in sorted(paths.HOLDOUT_DIR.glob("stage*/*")):
        stage = path.parent.name.removeprefix("stage")
        if path.suffix == ".csv":
            detail = f"{sum(1 for _ in path.open(encoding='utf-8')) - 1:,} dòng"
        elif path.suffix == ".npy":
            detail = f"{len(np.load(path, allow_pickle=False)):,} giá trị"
        else:
            detail = "artifact"
        relative = path.relative_to(paths.PROJECT_ROOT)
        description = ARTIFACT_DESCRIPTIONS.get(path.name)
        if description is None and path.name.startswith("holdout_trades_top"):
            keep_pct = path.stem.removeprefix("holdout_trades_top")
            description = f"Danh sách lệnh holdout được giữ ở mức top {keep_pct}%"
        if description is None:
            raise ValueError(f"Missing artifact description: {path.name}")
        rows.append([f"[`{path.name}`](../{relative})", stage, description, detail, f"{path.stat().st_size:,} B"])
    return rows


def write_submission_report(
    table1_rows: list[list[str]],
    table2_rows: list[list[str]],
    sweep_rows: list[list[str]],
    evidence: dict[str, dict],
    report_path: Path,
) -> None:
    stage1 = evidence["stage1"]
    train = evidence["train"]
    repeat = evidence["repeat"]
    stage3 = evidence["stage3"]
    versions = train["versions"]
    params = train["params"]
    counts = stage1["counts"]
    comparison = stage3["baseline_comparison"]
    param_text = " · ".join(f"`{key}={value}`" for key, value in params.items())
    mismatch_fields = ", ".join(comparison["mismatched_fields"]) or "không có"

    report = """# BÁO CÁO KẾT QUẢ TRAIN VÀ HOLDOUT

> Phạm vi: tổng hợp artifact đã chốt của 4 cách chia trên tập 80% đầu và kết
> quả holdout niêm phong. Báo cáo chỉ ghi số liệu, quyết định triển khai và bằng
> chứng kiểm chứng; không biện luận hay kết luận thay báo cáo chính.

## Lịch sử và lý do chạy lại holdout

1. Holdout ban đầu được mở và chạy theo bốn Stage sau khi feature,
   hyperparameter và tỷ lệ giữ lệnh đã chốt từ bước train.
2. Holdout phải chạy lại vì phát hiện cấu hình CatBoost chưa ghim
   `thread_count`; số luồng mặc định phụ thuộc cấu hình CPU và có thể tạo sai
   lệch số học giữa môi trường. Lần chạy lại chỉ thêm `thread_count=1` như thiết
   lập kỹ thuật; không đổi feature, hyperparameter mô hình, tập train, tập
   holdout hay luật chọn top-k.
3. Artifact của bốn nhánh trên 80% đầu được khôi phục và giữ nguyên theo bản bàn
   giao. Các số holdout trong báo cáo này lấy từ lần chạy lại đã kiểm chứng; lý
   do chạy lại và giới hạn bằng chứng được ghi công khai để không xem đây là một
   lần thử mô hình mới nhằm chọn kết quả đẹp hơn.

## Phần 1 — Số liệu

### 1.1. Cấu hình train cuối

| Thuộc tính | Giá trị |
|---|---|
| Tập train trước/sau purge + embargo | {train_before:,} / {train_after:,} dòng |
| Tập holdout | {holdout_rows:,} dòng; không bỏ dòng |
| Feature đầu vào | {feature_count} `FEATURES`; không đưa `META` vào `X` |
| Hyperparameter và thiết lập kỹ thuật | {param_text} |

### 1.2. Chỉ số phân loại

{table1}
Ghi chú: bốn dòng Cách 1–3 là trung bình chỉ số theo fold sau khi bỏ khúc 1 để
các nhánh được đo trên cùng phạm vi khúc 2–5. Dòng Holdout được tính một lần
trên toàn bộ 5,028 mẫu holdout, không phải trung bình theo fold.

### 1.3. Chỉ số tài chính, giữ top 50%

{table2}
### 1.4. Sweep holdout 20–80%

{sweep}
## Phần 2 — Biểu đồ

- [Biểu đồ 1 — Baseline và 4 nhánh top 50%, khúc 2–5](../outputs/holdout/stage4/equity-curve-chunk2-5-top50.html)
- [Biểu đồ 2 — Baseline holdout và Holdout top 50%](../outputs/holdout/stage4/equity-curve-holdout-top50.html)

Hai biểu đồ được dựng bằng Python/Plotly ở dạng bậc thang: equity chỉ thay đổi
tại `close_time`, không nội suy tuyến tính giữa hai lần đóng lệnh. Trục dọc là R
cộng dồn từ 0. File HTML chứa Plotly nội tuyến nên mở độc lập được.

## Phần 3 — Quyết định triển khai

1. Dùng `ceil(n × keep_pct / 100)` cho số lệnh giữ lại: top 50% của 20,007 lệnh là 10,004; top 50% của 5,028 lệnh là 2,514.
2. Xếp `probability` giảm dần, dùng `row_id` tăng dần để phá hòa; vũ trụ lệnh baseline giữ cố định nên lệnh bị loại không làm đổi tín hiệu sau đó.
3. Equity ghi nhận tại `close_time`; các lệnh đóng cùng lúc được cộng thành một điểm rồi mới cập nhật đường vốn.
4. Bốn dòng train ở Bảng 1 và năm dòng đầu Bảng 2 lấy nguyên từ artifact bàn giao đã niêm phong; dòng holdout và sweep lấy từ Stage 3.
5. Thêm `thread_count=1` như thiết lập kỹ thuật để loại số luồng CPU như một nguồn sai lệch đã biết; không xem đây là hyperparameter mô hình và không dùng riêng phép thử này để kết luận nguyên nhân sai lệch liên máy.
6. Dùng đường bậc thang ngang-rồi-dọc (`hv`) để equity giữ nguyên giữa hai mốc đóng lệnh và chỉ nhảy tại thời điểm R được ghi nhận.

## Phần 4 — Kiểm chứng

| Phép kiểm | Kết quả |
|---|---|
| Dataset tái sinh trước holdout | **PASS** — {matching_keys:,}/{frozen_rows:,} khóa cũ; thiếu 0 |
| Giá trị feature | **PASS** — {feature_cells:,} ô đã so; lệch 0 |
| Purge / embargo tại biên holdout | **PASS** — cắt {purge_rows} / {embargo_rows} dòng; train còn {train_after:,} |
| Holdout không bị sửa khi train/chấm điểm | **PASS** — SHA-256 trước/sau giữ nguyên |
| Replay baseline so với tradelist bàn giao | **PASS** — {actual_trades:,}/{reference_trades:,} lệnh; trường lệch: {mismatch_fields} |
| Lặp train trên cùng máy | **PASS** — {prediction_count:,} predictions giống hệt; max abs diff = {max_diff} |
| File model `.cbm` | **GHI NHẬN** — SHA-256 hai file khác nhau; binary model không phải tiêu chí PASS |
| Kiểm chứng liên máy | **GIỚI HẠN** — teammate Windows báo sai số prediction trong `1e-15`; chưa có artifact đối chứng được commit và chưa có thí nghiệm chỉ thay `thread_count` |
| Tái lập từng byte toàn pipeline holdout | **CHƯA XÁC NHẬN** — bằng chứng hiện chỉ xác nhận prediction của hai lượt train trên cùng máy; không tuyên bố toàn bộ Stage 1–4 giống từng byte |

Môi trường sinh artifact: Python {python}; CatBoost {catboost}; scikit-learn
{sklearn}; pandas {pandas}; NumPy {numpy}.

## Phần 5 — Bảng file sinh ra

{inventory}
""".format(
        train_before=train["train_rows_before_filtering"],
        train_after=train["train_rows_after_filtering"],
        holdout_rows=counts["holdout_rows"],
        feature_count=len(train["features"]),
        param_text=param_text,
        table1=markdown_table(["Cách chia", "ROC-AUC", "F1 @0.5"], table1_rows),
        table2=markdown_table(["", "Số lệnh", "Net profit (R)", "MaxDD (R)", "Profit factor", "Win rate %"], table2_rows),
        sweep=markdown_table(["Lọc", "Số lệnh", "Net profit (R)", "MaxDD (R)", "Profit factor", "Win rate %"], sweep_rows),
        matching_keys=counts["matching_keys"],
        frozen_rows=counts["frozen_rows"],
        feature_cells=counts["feature_cells_compared"],
        purge_rows=train["purge_rows"],
        embargo_rows=train["embargo_rows"],
        actual_trades=comparison["actual_holdout_trades"],
        reference_trades=comparison["reference_holdout_trades"],
        mismatch_fields=mismatch_fields,
        prediction_count=repeat["prediction_count"],
        max_diff=repeat["max_prediction_abs_difference"],
        python=versions["python"],
        catboost=versions["catboost"],
        sklearn=versions["scikit_learn"],
        pandas=versions["pandas"],
        numpy=versions["numpy"],
        inventory=markdown_table(["File", "Giai đoạn", "Chứa gì", "Quy mô", "Kích thước"], output_inventory()),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    stage3_dir = Path(args.stage3_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence = load_evidence(stage3_dir)
    validate_evidence(evidence)
    prior_sweep = pd.read_csv(SOURCE / "backtest_retention_sweep.csv")
    stage3_summary = pd.read_csv(stage3_dir / "stage3_backtest_summary.csv")
    prior50 = prior_sweep.loc[prior_sweep["comparison_keep_pct"] == 50].copy()
    if len(prior50) != 5 or set(prior50["method"]) != {"baseline", "random_kfold", "grouped_kfold", "walk_forward", "purged_walk_forward"}:
        raise ValueError("The handed-over pre-holdout 50% summary is incomplete")
    baseline_holdout = stage3_summary.loc[stage3_summary.keep_pct == 100].iloc[0]
    top50_holdout = stage3_summary.loc[stage3_summary.keep_pct == 50].iloc[0]
    stage3_report = evidence["stage3"]
    holdout_classification = stage3_report["classification"]
    table1_rows = BRANCH_TABLE1 + [
        ["Holdout", f"{holdout_classification['roc_auc']:.4f}", f"{holdout_classification['f1_at_0_5']:.4f}"],
    ]

    # The first five rows are copied from the handed-over report table, not recalculated.
    table2_rows = [
        ["Baseline (khúc 2–5)", "20,007", "+3,358.3", "236.1", "1.294", "31.7"],
        ["Cách 1", "10,004", "+6,998.3", "75.7", "2.563", "44.9"],
        ["Cách 1b", "10,004", "+4,724.3", "99.5", "1.942", "39.2"],
        ["Cách 2", "10,004", "+2,207.3", "205.3", "1.423", "34.9"],
        ["Cách 3", "10,004", "+2,253.7", "154.1", "1.435", "35.2"],
        ["Baseline holdout", f"{int(baseline_holdout.trades):,}", fmt_r(baseline_holdout.net_profit_R), fmt_num(baseline_holdout.max_dd_R), fmt_ratio(baseline_holdout.profit_factor), f"{baseline_holdout.win_rate_pct:.2f}"],
        ["Holdout, top 50%", f"{int(top50_holdout.trades):,}", fmt_r(top50_holdout.net_profit_R), fmt_num(top50_holdout.max_dd_R), fmt_ratio(top50_holdout.profit_factor), f"{top50_holdout.win_rate_pct:.2f}"],
    ]
    sweep = stage3_summary.loc[stage3_summary.keep_pct.isin([20, 30, 40, 50, 60, 70, 80])].copy()
    sweep_rows = [[f"Top {int(row.keep_pct)}%", f"{int(row.trades):,}", fmt_r(row.net_profit_R), fmt_num(row.max_dd_R), fmt_ratio(row.profit_factor), f"{row.win_rate_pct:.2f}"] for row in sweep.itertuples()]

    classification = pd.DataFrame(table1_rows, columns=["method", "roc_auc", "f1_at_0_5"])
    financial = pd.DataFrame(table2_rows, columns=["method", "trades", "net_profit_R", "max_dd_R", "profit_factor", "win_rate_pct"])
    holdout_sweep = pd.DataFrame(sweep_rows, columns=["filter", "trades", "net_profit_R", "max_dd_R", "profit_factor", "win_rate_pct"])
    classification.to_csv(out_dir / "table1_classification_metrics.csv", index=False)
    financial.to_csv(out_dir / "table2_financial_metrics_top50.csv", index=False)
    holdout_sweep.to_csv(out_dir / "holdout_sweep_20_80.csv", index=False)
    tables = "# Bảng 1 — Chỉ số phân loại\n\n" + markdown_table(["", "ROC-AUC", "F1 @0.5"], table1_rows)
    tables += "\n# Bảng 2 — Chỉ số tài chính, giữ top 50%\n\n" + markdown_table(["", "Số lệnh", "Net profit (R)", "MaxDD (R)", "Profit factor", "Win rate %"], table2_rows)
    tables += "\n# Sweep holdout 20–80%\n\n" + markdown_table(["Lọc", "Số lệnh", "Net profit (R)", "MaxDD (R)", "Profit factor", "Win rate %"], sweep_rows)
    (out_dir / "stage4_tables.md").write_text(tables, encoding="utf-8")

    # Chart 1: read only the handed-over scored universe and reproduce its fixed 50% selection rule.
    universe = pd.read_csv(SOURCE / "backtest_scored_universe.csv", parse_dates=["close_time"])
    evaluation = universe.loc[universe.chunk.isin([2, 3, 4, 5])].copy()
    if len(evaluation) != 20007: raise ValueError("Chunk 2–5 size differs from handed-over value")
    curves1 = [("Baseline", equity_by_close_time(evaluation))]
    names = {"random_kfold": "Cách 1 — Random K-Fold", "grouped_kfold": "Cách 1b — Grouped K-Fold", "walk_forward": "Cách 2 — Walk-forward", "purged_walk_forward": "Cách 3 — WF + Purge/Embargo"}
    expected = int(np.ceil(len(evaluation) * .5))
    for key, label in names.items():
        selected = evaluation.sort_values([f"probability_{key}", "row_id"], ascending=[False, True], kind="stable").iloc[:expected]
        if len(selected) != expected: raise ValueError("Top-50 selection size is invalid")
        curves1.append((label, equity_by_close_time(selected)))
    write_chart(out_dir / "equity-curve-chunk2-5-top50.html", "Đường vốn: baseline và 4 nhánh, khúc 2–5, top 50%", curves1)

    # Chart 2: only consumes the Stage 3 fixed baseline universe and its saved top-50 list.
    baseline = pd.read_csv(stage3_dir / "holdout_fixed_trade_universe_scored.csv", parse_dates=["close_time"])
    top50 = pd.read_csv(stage3_dir / "holdout_trades_top50.csv", parse_dates=["close_time"])
    if len(baseline) != 5028 or len(top50) != 2514: raise ValueError("Stage 3 holdout artifacts have unexpected row counts")
    write_chart(out_dir / "equity-curve-holdout-top50.html", "Đường vốn: baseline và holdout top 50%", [("Baseline holdout", equity_by_close_time(baseline)), ("Holdout, top 50%", equity_by_close_time(top50))])

    decisions = """# Quyết định triển khai

1. Trục thời gian của cả hai biểu đồ dùng `close_time`, vì R của một lệnh chỉ hoàn tất tại thời điểm đóng lệnh.
2. Các lệnh có cùng `close_time` được sắp xếp ổn định theo `ticket`, cộng R của chúng thành một điểm thời gian duy nhất; điểm đó là equity sau toàn bộ lệnh đóng cùng lúc.
3. Mỗi đường được thêm điểm R = 0 ngay trước `close_time` đầu tiên, để đường vốn bắt đầu từ 0 mà không thay đổi mốc dữ liệu giao dịch.
4. Biểu đồ 1 chỉ nhận các dòng `chunk` 2–5. Top 50% của từng nhánh dùng `ceil(20007 × 50%) = 10004`, xếp xác suất giảm dần rồi `row_id` tăng dần khi hòa điểm — đúng luật đã chốt ở bước trước.
5. Biểu đồ 2 bắt đầu lại từ R = 0 tại holdout, dùng trực tiếp vũ trụ baseline và danh sách Top 50% đã lưu từ Giai đoạn 3; không nối với equity của khúc 2–5.
6. Đường vốn dùng dạng bậc thang ngang-rồi-dọc (`hv`): giữ nguyên giữa hai `close_time` và chỉ nhảy tại lúc lệnh đóng; không chèn giao dịch hoặc điểm equity giả vào khoảng trống.
7. Purge và embargo đều trả về 0 dòng, nên train giữ nguyên 25,008 dòng; điều kiện lọc được chạy trước khi quyết định không loại dòng nào.
8. So khớp giá baseline dùng sai số tuyệt đối `5e-4`, theo validator bàn giao; giá vào, giá ra và R được kiểm dưới cùng ngưỡng này.
9. Chấm điểm holdout lấy danh sách 23 `FEATURES` trực tiếp từ `build_features.py`; không tự liệt kê cột.
10. CatBoost được ghim thêm `thread_count=1` — đây là tham số kỹ thuật (không thuộc danh sách hyperparameter mô hình đã chốt) để loại số luồng CPU như một nguồn sai lệch đã biết. Phép kiểm Stage 2 chỉ kết luận hai lượt trên cùng máy có prediction giống hệt; không dùng nó để khẳng định giống từng byte giữa mọi máy. Bốn dòng đầu Bảng 1 và năm dòng đầu Bảng 2 vẫn lấy nguyên từ bàn giao; chỉ dòng Holdout và bảng sweep được tính mới.
"""
    (out_dir / "implementation_decisions.md").write_text(decisions, encoding="utf-8")
    manifest = {"new_files": sorted(p.name for p in out_dir.iterdir()), "chart1_rows": {name: len(curve) for name, curve in curves1}, "chart2_rows": {"Baseline holdout": len(equity_by_close_time(baseline)), "Holdout top 50%": len(equity_by_close_time(top50))}}
    (out_dir / "stage4_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_submission_report(table1_rows, table2_rows, sweep_rows, evidence, Path(args.report))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
