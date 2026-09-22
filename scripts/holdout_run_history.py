#!/usr/bin/env python3
"""Sổ đối chiếu lịch sử các lần mở holdout niêm phong, dựng lại từ git history.

Script đọc artifact của ba commit từng mở holdout bằng `git show <commit>:<path>`
(không checkout, không ghi đè file nào) rồi ghi ra:

    outputs/verification/holdout_run_history.json
    outputs/verification/holdout_run_history.md

Ngoài ra script tính lại tham chiếu bàn giao bước 4 (Bảng 1 bỏ khúc 1, Bảng 2
top 50% khúc 2–5) từ artifact đang có ở HEAD để phần `handover_reference` trong
sổ có thể đối chiếu độc lập. Nếu git history lệch so với các con số đã biết,
script dừng ngay (STOP).

Chạy:  .venv/bin/python scripts/holdout_run_history.py
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citd_ml import paths


SCORED_PATH = "outputs/holdout/stage3/holdout_scored.csv"
STAGE3_REPORT_PATH = "outputs/holdout/stage3/stage3_report.json"
STAGE2_REPORT_PATH = "outputs/holdout/stage2/train_run1_report.json"

KEYS = ["origin_bar", "entry_bar", "leg"]
SCORED_ROWS = 5028
KEEP_RATES = (20, 30, 40, 50, 60, 70, 80)
TOP_KEEP = int(np.ceil(SCORED_ROWS * 50 / 100))
TOLERANCE = 1e-12
ROUNDING_TOLERANCE = 5e-5  # giá trị bàn giao chỉ ghi 4 chữ số thập phân
CHUNK_EDGES = [0, 5001, 10003, 15004, 20006, 25008]

# Số đã biết của ba lần chạy. Lệch quá TOLERANCE nghĩa là lịch sử/artifact đã đổi.
RUN_SPECS = [
    {
        "commit": "dc25cd3",
        "label": "Lần 1 — Windows gốc (không ghim thread_count)",
        "expected_auc": 0.60502452277619,
        "expected_f1": 0.40170679670832066,
        "expected_top50_net_R": -23.833050158080326,
    },
    {
        "commit": "5f46e41",
        "label": "Lần 2 — Linux đa luồng (không ghim thread_count)",
        "expected_auc": 0.6023152558719406,
        "expected_f1": 0.4064693317058285,
        "expected_top50_net_R": 30.78592225711712,
    },
    {
        "commit": "7748828",
        "label": "Lần 3 — Linux thread_count=1 (artifact hiện tại)",
        "expected_auc": 0.6045544538928682,
        "expected_f1": 0.40220723482526055,
        "expected_top50_net_R": -36.552754995889984,
    },
]
EXPECTED_BASELINE_NET_R = 245.9296826592543

# Tham chiếu bàn giao bước 4: mean theo fold sau khi bỏ khúc 1 (đã làm tròn 4 chữ số).
BRANCH_SPECS = [
    ("random_kfold", "Cách 1 — Random K-Fold", 0.8595, 0.6525),
    ("grouped_kfold", "Cách 1b — Grouped K-Fold", 0.7475, 0.5105),
    ("walk_forward", "Cách 2 — Walk-forward", 0.5867, 0.3267),
    ("purged_walk_forward", "Cách 3 — WF + Purge/Embargo", 0.5948, 0.3304),
]


# --------------------------------------------------------------------- git I/O
def git_bytes(*args: str) -> bytes:
    """Chạy git và trả stdout dạng bytes; lỗi thì dừng ngay."""
    result = subprocess.run(["git", *args], cwd=paths.PROJECT_ROOT, capture_output=True)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", "replace").strip()
        raise SystemExit(f"STOP: git {' '.join(args)} thất bại: {message}")
    return result.stdout


def git_text(*args: str) -> str:
    return git_bytes(*args).decode("utf-8")


def resolve_commit(short_commit: str) -> str:
    full = git_text("rev-parse", "--verify", f"{short_commit}^{{commit}}").strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", full, "HEAD"],
        cwd=paths.PROJECT_ROOT,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise SystemExit(f"STOP: {short_commit} không phải tổ tiên của HEAD; git history đã phân kỳ")
    return full


def provenance_entry(short_commit: str, full_commit: str, path: str) -> dict:
    content = git_bytes("show", f"{short_commit}:{path}")
    return {
        "commit": full_commit,
        "commit_short": short_commit,
        "path": path,
        "git_blob_sha1": git_text("rev-parse", f"{short_commit}:{path}").strip(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "read_via": f"git show {short_commit}:{path}",
    }


# ------------------------------------------------------------ holdout run data
def infer_environment(model_path: str) -> str:
    if "\\" in model_path or (len(model_path) > 1 and model_path[1] == ":"):
        return "Windows"
    return "Linux"


def check_metric(name: str, recovered: float, expected: float) -> dict:
    difference = abs(float(recovered) - float(expected))
    if difference > TOLERANCE:
        raise SystemExit(f"STOP: git history lệch kỳ vọng tại {name}: nhận {recovered!r}, kỳ vọng {expected!r}, |Δ|={difference:.6g}")
    return {
        "metric": name,
        "recovered": float(recovered),
        "expected": float(expected),
        "abs_diff": difference,
        "passed": True,
    }


def sweep_entry(row: dict) -> dict:
    return {
        "keep_pct": int(row["keep_pct"]),
        "filter": row["filter"],
        "trades": int(row["trades"]),
        "net_profit_R": float(row["net_profit_R"]),
        "max_dd_R": float(row["max_dd_R"]),
        "profit_factor": float(row["profit_factor"]),
        "win_rate_pct": float(row["win_rate_pct"]),
    }


def top50_selection(scored: pd.DataFrame) -> pd.DataFrame:
    """Luật Stage 3: giữ ceil(5028×50%)=2514 dòng, probability giảm dần rồi row_id tăng dần."""
    ordered = scored.sort_values(["probability", "row_id"], ascending=[False, True], kind="stable")
    selected = ordered.head(TOP_KEEP)
    if len(selected) != TOP_KEEP:
        raise SystemExit(f"STOP: top 50% phải giữ {TOP_KEEP} dòng, nhận {len(selected)}")
    return selected


def build_run(spec: dict, run_index: int) -> dict:
    short_commit = spec["commit"]
    full_commit = resolve_commit(short_commit)

    scored_bytes = git_bytes("show", f"{short_commit}:{SCORED_PATH}")
    stage3_bytes = git_bytes("show", f"{short_commit}:{STAGE3_REPORT_PATH}")
    stage2_bytes = git_bytes("show", f"{short_commit}:{STAGE2_REPORT_PATH}")

    scored = pd.read_csv(io.BytesIO(scored_bytes))
    stage3 = json.loads(stage3_bytes)
    stage2 = json.loads(stage2_bytes)

    if len(scored) != SCORED_ROWS:
        raise SystemExit(f"STOP: {short_commit} có {len(scored)} dòng scored, kỳ vọng {SCORED_ROWS}")
    if scored.duplicated(KEYS).any():
        raise SystemExit(f"STOP: {short_commit} có khóa (origin_bar, entry_bar, leg) trùng lặp")
    scored.insert(0, "row_id", np.arange(len(scored), dtype=np.int64))

    results = stage3["results"]
    baseline = next(row for row in results if int(row["keep_pct"]) == 100)
    sweep = sorted((row for row in results if int(row["keep_pct"]) != 100), key=lambda row: int(row["keep_pct"]))
    if [int(row["keep_pct"]) for row in sweep] != list(KEEP_RATES):
        raise SystemExit(f"STOP: {short_commit} thiếu sweep 20–80%")
    top50 = next(row for row in sweep if int(row["keep_pct"]) == 50)

    classification = stage3["classification"]
    model_path = stage2["model_path"]
    params = stage2["params"]
    checks = [
        check_metric(f"{short_commit}.roc_auc", classification["roc_auc"], spec["expected_auc"]),
        check_metric(f"{short_commit}.f1_at_0_5", classification["f1_at_0_5"], spec["expected_f1"]),
        check_metric(f"{short_commit}.top50_net_profit_R", top50["net_profit_R"], spec["expected_top50_net_R"]),
        check_metric(f"{short_commit}.baseline_net_profit_R", baseline["net_profit_R"], EXPECTED_BASELINE_NET_R),
    ]

    thread_count = params.get("thread_count")
    return {
        "run_index": run_index,
        "label": spec["label"],
        "commit": full_commit,
        "commit_short": short_commit,
        "commit_date": git_text("show", "-s", "--format=%cI", full_commit).strip(),
        "commit_subject": git_text("show", "-s", "--format=%s", full_commit).strip(),
        "environment": {
            "value": infer_environment(model_path),
            "recorded": False,
            "basis": f"model_path={model_path}",
            "note": "train_run1_report.json không có trường OS/platform; giá trị này chỉ suy ra từ dạng đường dẫn, không phải bằng chứng môi trường đầy đủ.",
        },
        "os_platform_recorded": False,
        "thread_count": thread_count,
        "thread_count_recorded": thread_count is not None,
        "classification": {
            "roc_auc": float(classification["roc_auc"]),
            "f1_at_0_5": float(classification["f1_at_0_5"]),
        },
        "baseline_holdout": sweep_entry(baseline),
        "top50_holdout": sweep_entry(top50),
        "sweep": [sweep_entry(row) for row in sweep],
        "model_params": params,
        "library_versions": stage2["versions"],
        "stage2_artifacts": {
            "run_id": int(stage2["run_id"]),
            "model_sha256": stage2["model_sha256"],
            "prediction_sha256": stage2["prediction_sha256"],
            "holdout_sha256_before": stage2["holdout_sha256_before"],
            "holdout_sha256_after": stage2["holdout_sha256_after"],
            "holdout_file_unchanged": bool(stage2["holdout_file_unchanged"]),
            "train_rows_after_filtering": int(stage2["train_rows_after_filtering"]),
        },
        "scored_file": {
            "rows": int(len(scored)),
            "unique_trade_keys": True,
            "sha256": hashlib.sha256(scored_bytes).hexdigest(),
        },
        "checks": checks,
        "_scored": scored,
    }


# ---------------------------------------------------------------- pairwise
def pairwise_delta(run_a: dict, run_b: dict) -> dict:
    merged = run_a["_scored"].merge(
        run_b["_scored"],
        on=KEYS,
        suffixes=("_a", "_b"),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != SCORED_ROWS:
        raise SystemExit(f"STOP: merge {run_a['commit_short']} × {run_b['commit_short']} chỉ khớp {len(merged)} dòng")
    probability_a = merged["probability_a"].to_numpy(dtype=float)
    probability_b = merged["probability_b"].to_numpy(dtype=float)
    difference = np.abs(probability_a - probability_b)

    keys_a = {tuple(row) for row in top50_selection(run_a["_scored"])[KEYS].itertuples(index=False, name=None)}
    keys_b = {tuple(row) for row in top50_selection(run_b["_scored"])[KEYS].itertuples(index=False, name=None)}
    intersection = keys_a & keys_b
    union = keys_a | keys_b
    return {
        "run_a": run_a["commit_short"],
        "run_b": run_b["commit_short"],
        "commit_a": run_a["commit"],
        "commit_b": run_b["commit"],
        "merged_rows": int(len(merged)),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "pearson_r": float(np.corrcoef(probability_a, probability_b)[0, 1]),
        "count_diff_gt_1e-12": int((difference > 1e-12).sum()),
        "top50": {
            "selection_size": TOP_KEEP,
            "intersection": len(intersection),
            "only_in_a": len(keys_a - keys_b),
            "only_in_b": len(keys_b - keys_a),
            "jaccard": len(intersection) / len(union),
        },
    }


# -------------------------------------------------------- handover reference
def branch_table() -> dict:
    branches = []
    for method, display_name, reported_auc, reported_f1 in BRANCH_SPECS:
        frame = pd.read_csv(paths.CATBOOST_TRAINING_DIR / f"oof_{method}.csv")
        if len(frame) != 25008:
            raise SystemExit(f"STOP: oof_{method}.csv có {len(frame)} dòng, kỳ vọng 25,008")
        frame["chunk"] = (
            np.searchsorted(np.asarray(CHUNK_EDGES[1:]), frame["row_id"].to_numpy(), side="right") + 1
        )
        evaluation = frame.loc[(frame["fold"] >= 1) & (frame["chunk"] != 1)]
        per_fold = []
        for fold, group in evaluation.groupby("fold", sort=True):
            per_fold.append({
                "fold": int(fold),
                "rows": int(len(group)),
                "roc_auc": float(roc_auc_score(group["label"], group["probability"])),
                "f1_at_0_5": float(f1_score(group["label"], group["probability"] >= 0.5)),
            })
        mean_auc = float(np.mean([row["roc_auc"] for row in per_fold]))
        mean_f1 = float(np.mean([row["f1_at_0_5"] for row in per_fold]))
        if abs(mean_auc - reported_auc) > ROUNDING_TOLERANCE or abs(mean_f1 - reported_f1) > ROUNDING_TOLERANCE:
            raise SystemExit(
                f"STOP: {method} tính lại {mean_auc:.6f}/{mean_f1:.6f} lệch bàn giao {reported_auc}/{reported_f1}"
            )
        branches.append({
            "method": method,
            "display_name": display_name,
            "mean_roc_auc": mean_auc,
            "mean_f1_at_0_5": mean_f1,
            "handover_reported_roc_auc": reported_auc,
            "handover_reported_f1_at_0_5": reported_f1,
            "abs_diff_roc_auc": abs(mean_auc - reported_auc),
            "abs_diff_f1_at_0_5": abs(mean_f1 - reported_f1),
            "per_fold": per_fold,
        })
    return {
        "rule": "giữ fold>=1 và chunk!=1; tính AUC/F1 từng fold trên phần còn lại rồi lấy trung bình; chunk chia theo row_id với biên [0,5001,10003,15004,20006,25008] (searchsorted side=right, giống backtest_pyramid_local.py)",
        "expected_precision": "artifact bàn giao ghi 4 chữ số thập phân; dung sai đối chiếu 5e-5",
        "branches": branches,
    }


def backtest_reference() -> dict:
    summary = pd.read_csv(paths.BACKTEST_DIR / "backtest_summary_top50.csv")
    retention = pd.read_csv(paths.BACKTEST_DIR / "backtest_retention_sweep.csv")
    expected_methods = {"baseline", "random_kfold", "grouped_kfold", "walk_forward", "purged_walk_forward"}
    if len(summary) != 5 or set(summary["method"]) != expected_methods:
        raise SystemExit("STOP: backtest_summary_top50.csv không đủ 5 phương pháp bàn giao")
    if len(retention) != 35 or set(retention["method"]) != expected_methods:
        raise SystemExit("STOP: backtest_retention_sweep.csv không đúng cấu trúc bàn giao")
    return {
        "top50_summary": {
            "source": "outputs/backtest/backtest_summary_top50.csv",
            "rows": json.loads(summary.to_json(orient="records")),
        },
        "retention_sweep": {
            "source": "outputs/backtest/backtest_retention_sweep.csv",
            "rows": json.loads(retention.to_json(orient="records")),
        },
    }


# ------------------------------------------------------------------ output
def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "\n".join("| " + " | ".join(row) + " |" for row in rows) + "\n"


def fmt_metric(value: float) -> str:
    return f"{value:.12g}"


def render_markdown(ledger: dict) -> str:
    lines: list[str] = []
    runs = ledger["runs"]

    lines.append("# SỔ ĐỐI CHIẾU LỊCH SỬ CHẠY HOLDOUT")
    lines.append("")
    lines.append(
        "> Sinh tự động bởi `scripts/holdout_run_history.py` từ git history (đọc bằng "
        f"`git show`, không checkout). HEAD khi sinh: `{ledger['head']['commit_short']}` "
        f"({ledger['head']['commit_date']})."
    )
    lines.append(">")
    lines.append(
        "> Mục đích: ghi số của ba phiên bản kết quả holdout lịch sử truy xuất được từ Git để người review "
        "tự kiểm tra. Sổ không xác nhận tổng số lần thực thi hoặc các lần không được lưu trong Git."
    )
    lines.append("")
    lines.append("## 1. Ba phiên bản kết quả holdout lịch sử")
    lines.append("")
    rows = []
    for run in runs:
        thread = str(run["thread_count"]) if run["thread_count_recorded"] else "không ghi (mặc định máy)"
        environment = f"{run['environment']['value']} (suy ra)"
        rows.append([
            f"{run['run_index']}",
            f"`{run['commit_short']}`",
            run["commit_date"][:10],
            environment,
            thread,
            fmt_metric(run["classification"]["roc_auc"]),
            fmt_metric(run["classification"]["f1_at_0_5"]),
            f"{run['top50_holdout']['net_profit_R']:+,.4f}",
            f"{run['top50_holdout']['profit_factor']:.4f}",
        ])
    lines.append(markdown_table(
        ["Lần", "Commit", "Ngày", "Môi trường", "thread_count", "ROC-AUC", "F1 @0.5", "Top 50% net R", "Top 50% PF"],
        rows,
    ))
    lines.append("")
    lines.append(
        "Ghi chú môi trường: `train_run1_report.json` của cả ba lần chạy **không có trường "
        "OS/platform**; cột Môi trường chỉ suy ra từ dạng đường dẫn trong `model_path` "
        "(`C:\\...` → Windows, `/home/...` → Linux), không phải bằng chứng môi trường đầy đủ. "
        "Chỉ lần 3 ghi `thread_count=1`; hai lần đầu không ghim tham số này."
    )
    lines.append("")
    lines.append(
        "Lần 3 (`7748828`) là artifact holdout hiện hành: dòng Holdout trong "
        "`docs/BAO_CAO_KET_QUA_HOLDOUT.md` (0.6046 / 0.4022, top 50% = −36.55 R) lấy từ lần này. "
        "Hai lần trước là số cũ đã từng công bố, được thu hồi ở đây để không thất lạc."
    )
    lines.append("")
    lines.append("### 1b. Baseline holdout và sweep 20–80% theo từng lần chạy")
    lines.append("")
    rows = []
    for keep_pct, filter_name in [(100, "Baseline (100%)")] + [(pct, f"Top {pct}%") for pct in KEEP_RATES]:
        values = []
        for run in runs:
            entry = run["baseline_holdout"] if keep_pct == 100 else next(row for row in run["sweep"] if row["keep_pct"] == keep_pct)
            values.append(f"{entry['net_profit_R']:+,.2f} ({entry['profit_factor']:.4f})")
        rows.append([filter_name] + values)
    lines.append(markdown_table(["Lọc (net R · PF)"] + [f"`{run['commit_short']}`" for run in runs], rows))
    lines.append("")
    lines.append("### 1c. Hash holdout dataset theo từng lần chạy")
    lines.append("")
    rows = []
    for run in runs:
        artifacts = run["stage2_artifacts"]
        rows.append([
            f"`{run['commit_short']}`",
            f"`{artifacts['holdout_sha256_before'][:12]}`",
            f"`{artifacts['holdout_sha256_after'][:12]}`",
            "không đổi" if artifacts["holdout_file_unchanged"] else "ĐỔI",
            f"{artifacts['train_rows_after_filtering']:,}",
        ])
    lines.append(markdown_table(["Lần", "SHA-256 trước train", "SHA-256 sau train", "Holdout bị sửa?", "Số dòng train"], rows))
    lines.append("")
    linux_hashes = sorted({run["stage2_artifacts"]["holdout_sha256_before"][:8] for run in runs if run["environment"]["value"] == "Linux"})
    lines.append(
        f"Hai lần chạy trên Linux dùng đúng cùng một file holdout (`{linux_hashes[0]}...`); "
        "lần Windows đầu dùng bản dataset gốc trước khi tái sinh trên Linux. "
        "Mỗi lần chạy đều ghi `holdout_file_unchanged = true` trong report Stage 2."
    )
    lines.append("")
    lines.append("## 2. Đối chiếu pairwise prediction")
    lines.append("")
    rows = []
    for delta in ledger["pairwise_deltas"]:
        rows.append([
            f"`{delta['run_a']}` × `{delta['run_b']}`",
            f"{delta['max_abs_diff']:.6g}",
            f"{delta['mean_abs_diff']:.6g}",
            f"{delta['pearson_r']:.6f}",
            f"{delta['count_diff_gt_1e-12']:,} / {delta['merged_rows']:,}",
            f"{delta['top50']['intersection']:,} / {delta['top50']['selection_size']:,}",
            f"{delta['top50']['jaccard']:.4f}",
        ])
    lines.append(markdown_table(
        ["Cặp lần chạy", "max abs(Δp)", "mean abs(Δp)", "Pearson r", "Số dòng lệch > 1e-12", "Top 50 trùng", "Jaccard"],
        rows,
    ))
    lines.append("")
    lines.append(
        f"Luật top 50% (theo Stage 3): giữ `ceil(5,028 × 50%) = {TOP_KEEP:,}` dòng, xếp "
        "`probability` giảm dần rồi `row_id` tăng dần khi hòa; `row_id` là thứ tự dòng 0-based "
        "của chính file `holdout_scored.csv` mỗi lần chạy. Merge theo ba khóa "
        "`(origin_bar, entry_bar, leg)`."
    )
    lines.append("")
    lines.append("## 3. Kiểm chứng ngoài — do teammate báo cáo")
    lines.append("")
    verification = ledger["external_verification"]
    lines.append(f"- **Người báo:** {verification['reported_by']}; **ngày:** {verification['reported_on']}; **môi trường:** {verification['reported_environment']}; **nhánh:** `{verification['branch']}`.")
    lines.append(f"- **Lệnh đã dùng:** `{verification['command']}`.")
    for claim in verification["claims"]:
        lines.append(f"- {claim}")
    lines.append("")
    lines.append(
        "Kết luận báo cáo của teammate: hai lượt train độc lập trên cùng một máy cho "
        "prediction giống hệt nhau; prediction của teammate so với prediction bản hiện có "
        "**trùng trong 1e-15, không byte-identical** giữa hai máy; không ghi lại số metric nào."
    )
    lines.append("")
    lines.append(f"**Trạng thái bằng chứng:** {verification['limitations']}")
    lines.append("")
    lines.append("## 4. Tham chiếu bàn giao bước 4 (tính lại tại HEAD)")
    lines.append("")
    lines.append("### 4a. Bảng 1 — phân loại, bỏ khúc 1, mean theo fold")
    lines.append("")
    rows = []
    branch = ledger["handover_reference"]["table1_classification_chunk1_removed"]
    for entry in branch["branches"]:
        rows.append([
            entry["display_name"],
            f"{entry['mean_roc_auc']:.6f}",
            f"{entry['handover_reported_roc_auc']:.4f}",
            f"{entry['mean_f1_at_0_5']:.6f}",
            f"{entry['handover_reported_f1_at_0_5']:.4f}",
        ])
    lines.append(markdown_table(["Cách chia", "AUC tính lại", "AUC bàn giao", "F1 tính lại", "F1 bàn giao"], rows))
    lines.append("")
    lines.append(f"Luật tính: {branch['rule']}. {branch['expected_precision']}.")
    lines.append("")
    lines.append("### 4b. Bảng 2 — tài chính top 50%, khúc 2–5 (nguyên từ artifact bàn giao)")
    lines.append("")
    rows = []
    for entry in ledger["handover_reference"]["table2_financial_top50"]["rows"]:
        rows.append([
            entry["display_name"],
            f"{int(entry['trades']):,}",
            f"{entry['net_profit_R']:+,.2f}",
            f"{entry['max_dd_R']:.2f}",
            f"{entry['profit_factor']:.4f}",
            f"{entry['win_rate_pct']:.2f}",
        ])
    lines.append(markdown_table(["Phương pháp", "Số lệnh", "Net profit (R)", "MaxDD (R)", "Profit factor", "Win rate %"], rows))
    lines.append("")
    lines.append("Các dòng khác của `backtest_retention_sweep.csv` (20–80%) nằm nguyên trong file JSON.")
    lines.append("")
    lines.append("## 5. Nguồn dữ liệu (provenance)")
    lines.append("")
    rows = []
    for entry in ledger["provenance"]["entries"]:
        rows.append([
            f"`{entry['commit_short']}`",
            f"`{entry['path']}`",
            f"`{entry['git_blob_sha1'][:12]}`",
            f"`{entry['sha256'][:12]}`",
            f"{entry['bytes']:,}",
        ])
    lines.append(markdown_table(["Commit", "Đường dẫn", "Git blob", "SHA-256 (12 ký tự đầu)", "Byte"], rows))
    lines.append("")
    lines.append("HEAD dùng cho tham chiếu bàn giao:")
    lines.append("")
    rows = []
    for entry in ledger["provenance"]["head_files"]:
        rows.append([
            f"`{entry['path']}`",
            f"{entry['rows']:,}" if entry.get("rows") is not None else "—",
            f"`{entry['sha256'][:12]}`",
        ])
    lines.append(markdown_table(["Đường dẫn", "Số dòng", "SHA-256 (12 ký tự đầu)"], rows))
    lines.append("")
    lines.append("## 6. Kết quả kiểm tra tự động")
    lines.append("")
    rows = []
    for check in ledger["assertions"]["checks"]:
        rows.append([
            check["metric"],
            fmt_metric(check["recovered"]),
            fmt_metric(check["expected"]),
            f"{check['abs_diff']:.3g}",
            "PASS" if check["passed"] else "FAIL",
        ])
    lines.append(markdown_table(["Chỉ số", "Thu hồi", "Kỳ vọng", "|Δ|", "Kết quả"], rows))
    lines.append("")
    lines.append(
        f"Dung sai cho phép: `{TOLERANCE:.0e}`. Cả ba commit đều là tổ tiên của HEAD; "
        f"mỗi file scored đủ {SCORED_ROWS:,} dòng và khóa không trùng. "
        "Nếu bất kỳ kiểm tra nào lệch, script dừng với mã khác 0 thay vì ghi sổ."
    )
    lines.append("")
    lines.append("## 7. Cách tái tạo")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/python scripts/holdout_run_history.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------- main
def main() -> None:
    runs = [build_run(spec, index + 1) for index, spec in enumerate(RUN_SPECS)]
    deltas = [pairwise_delta(run_a, run_b) for run_a, run_b in itertools.combinations(runs, 2)]
    checks = [check for run in runs for check in run["checks"]]
    backtest = backtest_reference()

    if runs[1]["stage2_artifacts"]["holdout_sha256_before"] != runs[2]["stage2_artifacts"]["holdout_sha256_before"]:
        raise SystemExit("STOP: hai lần chạy Linux không dùng cùng một file holdout")

    head_commit = git_text("rev-parse", "HEAD").strip()
    ledger = {
        "scope_note": (
            "Sổ này truy xuất ba phiên bản kết quả holdout lịch sử từ ba commit Git định sẵn; không xác nhận tổng số lần thực thi. "
            "Nó chứng minh các con số cũ đã được ghi lại và không bị thay thế, nhưng không tự "
            "chứng minh không có cherry-picking: người review cần đối chiếu chính các số đã thu hồi."
        ),
        "head": {
            "commit": head_commit,
            "commit_short": head_commit[:7],
            "commit_date": git_text("show", "-s", "--format=%cI", head_commit).strip(),
            "commit_subject": git_text("show", "-s", "--format=%s", head_commit).strip(),
        },
        "provenance": {
            "method": "git show <commit>:<path> (không checkout)",
            "entries": [
                provenance_entry(spec["commit"], run["commit"], path)
                for spec, run in zip(RUN_SPECS, runs)
                for path in (SCORED_PATH, STAGE3_REPORT_PATH, STAGE2_REPORT_PATH)
            ],
            "head_files": [
                {
                    "path": f"outputs/catboost_training/oof_{method}.csv",
                    "rows": int(len(pd.read_csv(paths.CATBOOST_TRAINING_DIR / f"oof_{method}.csv"))),
                    "sha256": hashlib.sha256((paths.CATBOOST_TRAINING_DIR / f"oof_{method}.csv").read_bytes()).hexdigest(),
                }
                for method, _, _, _ in BRANCH_SPECS
            ] + [
                {
                    "path": "outputs/backtest/backtest_summary_top50.csv",
                    "rows": int(len(pd.read_csv(paths.BACKTEST_DIR / "backtest_summary_top50.csv"))),
                    "sha256": hashlib.sha256((paths.BACKTEST_DIR / "backtest_summary_top50.csv").read_bytes()).hexdigest(),
                },
                {
                    "path": "outputs/backtest/backtest_retention_sweep.csv",
                    "rows": int(len(pd.read_csv(paths.BACKTEST_DIR / "backtest_retention_sweep.csv"))),
                    "sha256": hashlib.sha256((paths.BACKTEST_DIR / "backtest_retention_sweep.csv").read_bytes()).hexdigest(),
                },
            ],
        },
        "runs": [{key: value for key, value in run.items() if key != "_scored"} for run in runs],
        "pairwise_deltas": deltas,
        "external_verification": {
            "reported_by": "Bùi Quốc Thịnh",
            "reported_on": "2026-09-12",
            "reported_environment": "Windows",
            "branch": "main",
            "command": "scripts/holdout_stage2_train.py --run-id 1|2 --verify",
            "claims": [
                "Hai lượt train độc lập trên cùng một máy cho prediction giống hệt nhau (run1 so với run2).",
                "Prediction của teammate so với prediction bản hiện có tương đương trong sai số 1e-15.",
                "Không byte-identical giữa hai máy khác nhau.",
                "Không ghi lại số metric nào kèm theo.",
            ],
            "evidence_class": "reported_by_teammate_not_recomputable_from_committed_artifacts",
            "limitations": (
                "Đây là báo cáo miệng của teammate, không có artifact đối chứng được commit "
                "(không có prediction/hash/log từ máy Windows trong repo) và không thể tái tạo "
                "từ git. Chỉ ghi nhận như thông tin tham khảo, không dùng thay bằng chứng PASS."
            ),
        },
        "handover_reference": {
            "table1_classification_chunk1_removed": branch_table(),
            "table2_financial_top50": {
                **backtest["top50_summary"],
                "note": "Các dòng này lấy nguyên từ artifact bàn giao đã niêm phong; chỉ đọc, không sửa.",
            },
            "backtest_retention_sweep": backtest["retention_sweep"],
        },
        "assertions": {
            "tolerance": TOLERANCE,
            "checks": checks,
            "history_integrity": {
                "commits_are_ancestors_of_head": True,
                "scored_rows_per_run": SCORED_ROWS,
                "unique_trade_keys_per_run": True,
                "sweep_rows_per_run": len(KEEP_RATES),
            },
        },
    }

    paths.VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    json_path = paths.VERIFICATION_DIR / "holdout_run_history.json"
    md_path = paths.VERIFICATION_DIR / "holdout_run_history.md"
    json_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    md_path.write_text(render_markdown(ledger), encoding="utf-8")

    print(f"Đã ghi {json_path.relative_to(paths.PROJECT_ROOT)} và {md_path.relative_to(paths.PROJECT_ROOT)}")
    for delta in deltas:
        top50 = delta["top50"]
        print(
            f"  {delta['run_a']} × {delta['run_b']}: max|Δ|={delta['max_abs_diff']:.6g}, "
            f"r={delta['pearson_r']:.6f}, top50 trùng {top50['intersection']}/{top50['selection_size']}"
        )


if __name__ == "__main__":
    main()
