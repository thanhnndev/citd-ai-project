"""Score the sealed holdout and evaluate fixed-universe retention rates."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "work" / "handover_holdout" / "Ban giao - Holdout" / "chay"
sys.path.insert(0, str(SOURCE))
import backtest_pyramid_local as base  # noqa: E402


HOLDOUT_START = pd.Timestamp("2025-02-08 15:30:00")
KEEP_RATES = (20, 30, 40, 50, 60, 70, 80)
M1_SOURCE = SOURCE / "BTCUSD_m1_2018_to_now.csv"
REFERENCE = SOURCE / "tradelist_pyramid_local.csv"
HOLDOUT_DATASET = ROOT / "holdout_stage1" / "dataset_catboost_holdout.csv"
MODEL = ROOT / "holdout_stage2" / "catboost_final_holdout_run1.cbm"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_features() -> list[str]:
    spec = importlib.util.spec_from_file_location("build_features", SOURCE / "build_features.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load build_features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.FEATURES)


def load_full_history():
    """Copy the handed-over M1-to-M15 loader, without its pre-holdout cut."""
    m1 = pd.read_csv(M1_SOURCE)
    base._require_columns(m1, ["Date", "Time", "Open", "High", "Low", "Close", "Volume"], M1_SOURCE.name)
    m1["dt"] = pd.to_datetime(m1["Date"] + " " + m1["Time"], format="%Y.%m.%d %H:%M:%S")
    m1 = m1.set_index("dt").sort_index(kind="stable")
    if not m1.index.is_monotonic_increasing:
        raise ValueError("Mốc thời gian M1 không tăng dần")
    m15 = m1.resample("15min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    slot = m1.index.floor("15min")
    pos = pd.Series(np.arange(len(m15)), index=m15.index)
    owner = pos.reindex(slot).values
    keep = ~np.isnan(owner)
    owner = owner[keep].astype(np.int64)
    m1_high = m1["High"].to_numpy()[keep]
    m1_low = m1["Low"].to_numpy()[keep]
    lo = np.searchsorted(owner, np.arange(len(m15)), "left")
    hi = np.searchsorted(owner, np.arange(len(m15)), "right")
    return m1, m15, m1_high, m1_low, lo, hi


def validate_full_reference(trades: pd.DataFrame) -> dict:
    """Use the handed-over comparison fields and numeric tolerance across all history."""
    reference = pd.read_csv(REFERENCE, parse_dates=["signal_time", "open_time", "close_time"])
    fields = ["signal_time", "open_time", "close_time", "leg", "entry_price", "exit_price", "R"]
    base._require_columns(reference, fields, REFERENCE.name)
    if len(trades) != len(reference):
        raise ValueError(f"Số lệnh baseline ({len(trades)}) khác tradelist bàn giao ({len(reference)})")
    check = trades.merge(reference[fields], on=["open_time", "leg"], how="outer", validate="one_to_one", indicator=True, suffixes=("", "_reference"))
    if not (check["_merge"] == "both").all():
        raise ValueError(f"Có {int((check['_merge'] != 'both').sum())} lệnh baseline không khớp trade list")
    for column in ["signal_time", "close_time"]:
        if not (check[column] == check[f"{column}_reference"]).all():
            raise ValueError(f"{column} khác tradelist bàn giao")
    for column in ["entry_price", "exit_price", "R"]:
        if not np.allclose(check[column].to_numpy(), check[f"{column}_reference"].to_numpy(), rtol=0.0, atol=5e-4):
            raise ValueError(f"{column} khác tradelist bàn giao")
    holdout_ref = reference.loc[reference["open_time"] >= HOLDOUT_START].copy()
    holdout_replay = trades.loc[trades["open_time"] >= HOLDOUT_START].copy()
    if len(holdout_ref) != len(holdout_replay):
        raise ValueError("Số lệnh baseline holdout khác reference")
    return {
        "reference_match": True,
        "full_history_trades": len(trades),
        "holdout_baseline_trades": len(holdout_replay),
        "numeric_tolerance": 5e-4,
    }


def select_top_percent(universe: pd.DataFrame, keep_pct: int) -> pd.DataFrame:
    keep_count = int(np.ceil(len(universe) * keep_pct / 100.0))
    ranked = universe.sort_values(["probability", "row_id"], ascending=[False, True], kind="stable")
    return ranked.iloc[:keep_count].copy()


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    features = load_features()
    holdout_hash_before = sha256(HOLDOUT_DATASET)
    model_hash_before = sha256(MODEL)
    holdout = pd.read_csv(HOLDOUT_DATASET, parse_dates=["entry_time", "label_end_time"])
    if len(holdout) != 5028 or holdout["entry_time"].lt(HOLDOUT_START).any():
        raise ValueError("Dataset holdout không đúng phạm vi niêm phong")
    if list(holdout[features].columns) != features or len(features) != 23:
        raise ValueError("Features không đúng hằng FEATURES")
    model = CatBoostClassifier()
    model.load_model(MODEL)
    probability = model.predict_proba(holdout[features])[:, 1]
    if probability.shape != (len(holdout),) or not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("Xác suất holdout không hợp lệ")
    scored = holdout.copy()
    scored.insert(0, "row_id", np.arange(len(scored), dtype=np.int64))
    scored["probability"] = probability
    auc = float(roc_auc_score(scored["label"], probability))
    f1 = float(f1_score(scored["label"], (probability >= 0.5).astype(np.int8)))

    m1, m15, m1_high, m1_low, lo, hi = load_full_history()
    baseline = base.backtest(m15, m1_high, m1_low, lo, hi, show_progress=False)
    reference_check = validate_full_reference(baseline)

    key = ["origin_bar", "entry_bar", "leg"]
    baseline_holdout = baseline.loc[baseline["open_time"] >= HOLDOUT_START].copy()
    universe = scored.merge(baseline_holdout, on=key, how="left", validate="one_to_one", indicator=True, suffixes=("_score", ""))
    if not (universe["_merge"] == "both").all():
        raise ValueError(f"Có {int((universe['_merge'] != 'both').sum())} score holdout không ghép được baseline")
    universe = universe.drop(columns="_merge")
    if len(universe) != len(baseline_holdout) or not (universe["entry_time"] == universe["open_time"]).all():
        raise ValueError("Universe score/trade holdout không khớp một-một")
    universe = universe.sort_values("row_id", kind="stable").reset_index(drop=True)

    metric_rows = [{"retention_pct": 100, "display_name": "Baseline", **base.calculate_metrics(baseline_holdout)}]
    selection_columns = {}
    selected_outputs: dict[int, pd.DataFrame] = {}
    for rate in KEEP_RATES:
        selected = select_top_percent(universe, rate)
        expected = int(np.ceil(len(universe) * rate / 100.0))
        if len(selected) != expected:
            raise ValueError(f"Top-{rate}% sai số lệnh")
        selected_outputs[rate] = selected
        selection_columns[f"selected_top_{rate}_pct"] = universe["row_id"].isin(selected["row_id"])
        metric_rows.append({"retention_pct": rate, "display_name": f"Holdout top {rate}%", **base.calculate_metrics(selected)})
    for name, values in selection_columns.items():
        universe[name] = values.to_numpy()
    metrics = pd.DataFrame(metric_rows)
    classification = pd.DataFrame([{"roc_auc": auc, "f1_at_0_5": f1, "rows": len(scored)}])

    csv_options = {"index": False, "float_format": "%.12g", "date_format": "%Y-%m-%d %H:%M:%S"}
    baseline.to_csv(HERE / "baseline_full_tradelist.csv", **csv_options)
    baseline_holdout.to_csv(HERE / "baseline_holdout_tradelist.csv", **csv_options)
    scored.to_csv(HERE / "holdout_scores.csv", **csv_options)
    universe.to_csv(HERE / "holdout_scored_universe.csv", **csv_options)
    metrics.to_csv(HERE / "holdout_metrics.csv", **csv_options)
    classification.to_csv(HERE / "holdout_classification_metrics.csv", **csv_options)
    for rate, selected in selected_outputs.items():
        selected.to_csv(HERE / f"trades_holdout_top_{rate}_pct.csv", **csv_options)
    report = {
        "holdout_start": str(HOLDOUT_START),
        "backtest_m1_range": [str(m1.index.min()), str(m1.index.max())],
        "backtest_m15_range": [str(m15.index.min()), str(m15.index.max())],
        "reference_validation": reference_check,
        "holdout_dataset_rows": len(holdout),
        "classification": {"roc_auc": auc, "f1_at_0_5": f1},
        "selection_rule": "keep_count = ceil(N * retention_pct / 100); probability descending, row_id ascending for ties",
        "randomness": "None: scoring, baseline replay, and top-k selection are deterministic.",
        "input_sha256_before_after": {
            "holdout_dataset_before": holdout_hash_before,
            "holdout_dataset_after": sha256(HOLDOUT_DATASET),
            "model_before": model_hash_before,
            "model_after": sha256(MODEL),
        },
    }
    report["input_files_unchanged"] = (
        report["input_sha256_before_after"]["holdout_dataset_before"] == report["input_sha256_before_after"]["holdout_dataset_after"]
        and report["input_sha256_before_after"]["model_before"] == report["input_sha256_before_after"]["model_after"]
    )
    (HERE / "stage3_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(classification.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
