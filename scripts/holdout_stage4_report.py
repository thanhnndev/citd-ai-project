"""Stage 4 only: assemble fixed metrics and render the two requested HTML curves."""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citd_ml import paths


SOURCE = paths.BACKTEST_DIR
STAGE3 = paths.HOLDOUT_STAGE3_DIR
OUT = paths.HOLDOUT_STAGE4_DIR


# First four rows are the handed-over classification reference (chunk 1 removed).
# The Holdout row is filled at runtime from stage3_report.json so it cannot drift.
BRANCH_TABLE1 = [
    ["Cách 1 — Random K-Fold", "0.8595", "0.6525"],
    ["Cách 1b — Grouped K-Fold", "0.7475", "0.5105"],
    ["Cách 2 — Walk-forward", "0.5867", "0.3267"],
    ["Cách 3 — WF + Purge/Embargo", "0.5948", "0.3304"],
]


def fmt_r(value: float) -> str:
    return f"{value:+,.10f}"


def fmt_num(value: float) -> str:
    return f"{value:,.10f}"


def equity_by_close_time(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all same-close-time trade R before creating the line point."""
    ordered = trades.sort_values(["close_time", "ticket"], kind="stable")
    at_time = ordered.groupby("close_time", sort=True, as_index=False)["R"].sum()
    at_time["equity_R"] = at_time["R"].cumsum()
    start = pd.DataFrame({"close_time": [at_time["close_time"].iat[0] - pd.Timedelta(nanoseconds=1)], "R": [0.0], "equity_R": [0.0]})
    return pd.concat([start, at_time], ignore_index=True)


def add_trace(fig: go.Figure, curve: pd.DataFrame, name: str) -> None:
    fig.add_trace(go.Scatter(x=curve["close_time"], y=curve["equity_R"], name=name, mode="lines", line={"width": 1.6}, hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Equity: %{y:.3f} R<extra>" + name + "</extra>"))


def write_chart(path: Path, title: str, curves: list[tuple[str, pd.DataFrame]]) -> None:
    fig = go.Figure()
    for name, curve in curves:
        add_trace(fig, curve, name)
    fig.update_layout(title=title, xaxis_title="close_time", yaxis_title="R cộng dồn", hovermode="x unified", template="plotly_white", legend_title_text="Chuỗi", margin={"l": 72, "r": 28, "t": 72, "b": 72})
    fig.write_html(path, include_plotlyjs="inline", full_html=True, config={"responsive": True, "displaylogo": False})


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior_sweep = pd.read_csv(SOURCE / "backtest_retention_sweep.csv")
    stage3_summary = pd.read_csv(STAGE3 / "stage3_backtest_summary.csv")
    prior50 = prior_sweep.loc[prior_sweep["comparison_keep_pct"] == 50].copy()
    if len(prior50) != 5 or set(prior50["method"]) != {"baseline", "random_kfold", "grouped_kfold", "walk_forward", "purged_walk_forward"}:
        raise ValueError("The handed-over pre-holdout 50% summary is incomplete")
    baseline_holdout = stage3_summary.loc[stage3_summary.keep_pct == 100].iloc[0]
    top50_holdout = stage3_summary.loc[stage3_summary.keep_pct == 50].iloc[0]
    stage3_report = json.loads((STAGE3 / "stage3_report.json").read_text(encoding="utf-8"))
    holdout_classification = stage3_report["classification"]
    table1_rows = BRANCH_TABLE1 + [
        ["Holdout", f"{holdout_classification['roc_auc']}", f"{holdout_classification['f1_at_0_5']}"],
    ]

    # The first five rows are copied from the handed-over report table, not recalculated.
    table2_rows = [
        ["Baseline (khúc 2–5)", "20,007", "+3,358.3", "236.1", "1.294", "31.7"],
        ["Cách 1", "10,004", "+6,998.3", "75.7", "2.563", "44.9"],
        ["Cách 1b", "10,004", "+4,724.3", "99.5", "1.942", "39.2"],
        ["Cách 2", "10,004", "+2,207.3", "205.3", "1.423", "34.9"],
        ["Cách 3", "10,004", "+2,253.7", "154.1", "1.435", "35.2"],
        ["Baseline holdout", f"{int(baseline_holdout.trades):,}", fmt_r(baseline_holdout.net_profit_R), fmt_num(baseline_holdout.max_dd_R), f"{baseline_holdout.profit_factor:.10f}", f"{baseline_holdout.win_rate_pct:.10f}"],
        ["Holdout, top 50%", f"{int(top50_holdout.trades):,}", fmt_r(top50_holdout.net_profit_R), fmt_num(top50_holdout.max_dd_R), f"{top50_holdout.profit_factor:.10f}", f"{top50_holdout.win_rate_pct:.10f}"],
    ]
    sweep = stage3_summary.loc[stage3_summary.keep_pct.isin([20, 30, 40, 50, 60, 70, 80])].copy()
    sweep_rows = [[f"Top {int(row.keep_pct)}%", f"{int(row.trades):,}", fmt_r(row.net_profit_R), fmt_num(row.max_dd_R), f"{row.profit_factor:.10f}", f"{row.win_rate_pct:.10f}"] for row in sweep.itertuples()]

    classification = pd.DataFrame(table1_rows, columns=["method", "roc_auc", "f1_at_0_5"])
    financial = pd.DataFrame(table2_rows, columns=["method", "trades", "net_profit_R", "max_dd_R", "profit_factor", "win_rate_pct"])
    holdout_sweep = pd.DataFrame(sweep_rows, columns=["filter", "trades", "net_profit_R", "max_dd_R", "profit_factor", "win_rate_pct"])
    classification.to_csv(OUT / "table1_classification_metrics.csv", index=False)
    financial.to_csv(OUT / "table2_financial_metrics_top50.csv", index=False)
    holdout_sweep.to_csv(OUT / "holdout_sweep_20_80.csv", index=False)
    tables = "# Bảng 1 — Chỉ số phân loại\n\n" + markdown_table(["", "ROC-AUC", "F1 @0.5"], table1_rows)
    tables += "\n# Bảng 2 — Chỉ số tài chính, giữ top 50%\n\n" + markdown_table(["", "Số lệnh", "Net profit (R)", "MaxDD (R)", "Profit factor", "Win rate %"], table2_rows)
    tables += "\n# Sweep holdout 20–80%\n\n" + markdown_table(["Lọc", "Số lệnh", "Net profit (R)", "MaxDD (R)", "Profit factor", "Win rate %"], sweep_rows)
    (OUT / "stage4_tables.md").write_text(tables, encoding="utf-8")

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
    write_chart(OUT / "equity-curve-chunk2-5-top50.html", "Đường vốn: baseline và 4 nhánh, khúc 2–5, top 50%", curves1)

    # Chart 2: only consumes the Stage 3 fixed baseline universe and its saved top-50 list.
    baseline = pd.read_csv(STAGE3 / "holdout_fixed_trade_universe_scored.csv", parse_dates=["close_time"])
    top50 = pd.read_csv(STAGE3 / "holdout_trades_top50.csv", parse_dates=["close_time"])
    if len(baseline) != 5028 or len(top50) != 2514: raise ValueError("Stage 3 holdout artifacts have unexpected row counts")
    write_chart(OUT / "equity-curve-holdout-top50.html", "Đường vốn: baseline và holdout top 50%", [("Baseline holdout", equity_by_close_time(baseline)), ("Holdout, top 50%", equity_by_close_time(top50))])

    decisions = """# Quyết định triển khai

1. Trục thời gian của cả hai biểu đồ dùng `close_time`, vì R của một lệnh chỉ hoàn tất tại thời điểm đóng lệnh.
2. Các lệnh có cùng `close_time` được sắp xếp ổn định theo `ticket`, cộng R của chúng thành một điểm thời gian duy nhất; điểm đó là equity sau toàn bộ lệnh đóng cùng lúc.
3. Mỗi đường được thêm điểm R = 0 ngay trước `close_time` đầu tiên, để đường vốn bắt đầu từ 0 mà không thay đổi mốc dữ liệu giao dịch.
4. Biểu đồ 1 chỉ nhận các dòng `chunk` 2–5. Top 50% của từng nhánh dùng `ceil(20007 × 50%) = 10004`, xếp xác suất giảm dần rồi `row_id` tăng dần khi hòa điểm — đúng luật đã chốt ở bước trước.
5. Biểu đồ 2 bắt đầu lại từ R = 0 tại holdout, dùng trực tiếp vũ trụ baseline và danh sách Top 50% đã lưu từ Giai đoạn 3; không nối với equity của khúc 2–5.
6. Các khoảng thời gian không có lệnh được nối bằng đường liên tục giữa các điểm đóng lệnh; không chèn giao dịch hoặc điểm equity giả vào các khoảng trống.
7. Purge và embargo đều trả về 0 dòng, nên train giữ nguyên 25,008 dòng; điều kiện lọc được chạy trước khi quyết định không loại dòng nào.
8. So khớp giá baseline dùng sai số tuyệt đối `5e-4`, theo validator bàn giao; giá vào, giá ra và R được kiểm dưới cùng ngưỡng này.
9. Chấm điểm holdout lấy danh sách 23 `FEATURES` trực tiếp từ `build_features.py`; không tự liệt kê cột.
10. CatBoost được ghim thêm `thread_count=1` — đây là tham số kỹ thuật (không thuộc danh sách hyperparameter mô hình đã chốt) nhằm để hai lần train tái lập giống hệt. Bốn dòng đầu Bảng 1 và năm dòng đầu Bảng 2 vẫn lấy nguyên từ bàn giao; chỉ dòng Holdout và bảng sweep được tính mới.
"""
    (OUT / "implementation_decisions.md").write_text(decisions, encoding="utf-8")
    manifest = {"new_files": sorted(p.name for p in OUT.iterdir()), "chart1_rows": {name: len(curve) for name, curve in curves1}, "chart2_rows": {"Baseline holdout": len(equity_by_close_time(baseline)), "Holdout top 50%": len(equity_by_close_time(top50))}}
    (OUT / "stage4_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
