"""Stage 3: score the untouched holdout and evaluate fixed baseline trades.

This is deliberately separate from every handed-over source file.  It replays
the strategy across all available M1 history, validates only the holdout slice
against the reference tradelist, and applies CatBoost ranking after the fixed
trade universe has been generated.  Thus a rejected trade cannot alter later
signals, pyramid state, entries, exits, or another trade's R outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citd_ml import paths
from citd_ml.features.build_features import FEATURES
from citd_ml.strategy.pyramid_strategy import PyramidStrategy


RAW = paths.RAW_M1_CSV
REFERENCE = paths.TRADELIST_CSV
HOLDOUT = paths.HOLDOUT_STAGE1_DIR / "dataset_catboost_holdout.csv"
HOLDOUT_START = pd.Timestamp(paths.HOLDOUT_START)
KEEP_RATES = (20, 30, 40, 50, 60, 70, 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        type=int,
        choices=[1, 2],
        default=1,
        help="Chọn model Stage 2 (catboost_final_holdout_run{run_id}.cbm). Mặc định: 1.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=paths.HOLDOUT_STAGE3_DIR,
        help="Thư mục ghi toàn bộ output Stage 3 (mặc định: outputs/holdout/stage3).",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_market() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp]:
    m1 = pd.read_csv(RAW)
    required = {"Date", "Time", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(m1.columns):
        raise ValueError(f"Raw M1 missing columns: {sorted(required - set(m1.columns))}")
    m1["dt"] = pd.to_datetime(m1["Date"] + " " + m1["Time"], format="%Y.%m.%d %H:%M:%S")
    m1 = m1.set_index("dt").sort_index(kind="stable")
    raw_start, raw_end = m1.index.min(), m1.index.max()
    m15 = m1.resample("15min").agg({"Open":"first", "High":"max", "Low":"min", "Close":"last", "Volume":"sum"}).dropna()
    owner = pd.Series(np.arange(len(m15)), index=m15.index).reindex(m1.index.floor("15min")).to_numpy()
    keep = ~np.isnan(owner)
    owner = owner[keep].astype(np.int64)
    lo = np.searchsorted(owner, np.arange(len(m15)), side="left")
    hi = np.searchsorted(owner, np.arange(len(m15)), side="right")
    return m15, m1["High"].to_numpy()[keep], m1["Low"].to_numpy()[keep], lo, hi, raw_start, raw_end


def replay_baseline(m15: pd.DataFrame, m1_high: np.ndarray, m1_low: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> pd.DataFrame:
    strat = PyramidStrategy(k=3)
    ind = strat.prepare(*(m15[c].to_numpy() for c in ["Open", "High", "Low", "Close", "Volume"]))
    ts, close = m15.index, m15["Close"].to_numpy()
    weekday, seconds = ts.dayofweek.to_numpy(), (ts.hour * 3600 + ts.minute * 60).to_numpy()
    rows = []
    for i in range(len(m15)):
        strat.open_bar(i, ind)
        closed = strat.scan_bar(m1_high[lo[i]:hi[i]], m1_low[lo[i]:hi[i]])
        closed += strat.friday_close(weekday[i], seconds[i], close[i])
        rows.extend((p.origin_bar, p.leg, p.entry_bar, i, p.entry_price, exit_price, reason) for p, reason, exit_price in closed)
    trades = pd.DataFrame(rows, columns=["origin_bar","leg","entry_bar","exit_bar","entry_price","exit_price","exit_reason"])
    if trades.empty or trades.duplicated(["origin_bar", "entry_bar", "leg"]).any():
        raise ValueError("Baseline replay is empty or has duplicate trade keys")
    trades = trades.sort_values(["entry_bar", "leg"], kind="stable").reset_index(drop=True)
    trades.insert(0, "ticket", np.arange(1, len(trades) + 1))
    trades.insert(1, "open_time", ts[trades.entry_bar].values)
    trades.insert(2, "close_time", ts[trades.exit_bar].values)
    trades.insert(3, "signal_time", ts[trades.origin_bar].values)
    trades["pl_pct"] = (trades.exit_price / trades.entry_price - 1) * 100
    trades["R"] = trades.pl_pct / (strat.sl_pct * 100)
    trades["bars_held"] = trades.exit_bar - trades.entry_bar
    return trades


def validate_holdout_baseline(trades: pd.DataFrame) -> dict:
    reference = pd.read_csv(REFERENCE, parse_dates=["signal_time", "open_time", "close_time"])
    actual = trades.loc[trades.open_time >= HOLDOUT_START].copy()
    expected = reference.loc[reference.open_time >= HOLDOUT_START].copy()
    keys = ["open_time", "leg"]
    check = actual.merge(expected[keys + ["signal_time", "close_time", "entry_price", "exit_price", "R"]], on=keys, how="outer", indicator=True, suffixes=("", "_reference"))
    mismatch = []
    if len(actual) != len(expected): mismatch.append("trade_count")
    if not (check._merge == "both").all(): mismatch.append("trade_keys")
    if not mismatch:
        for col in ["signal_time", "close_time"]:
            if not (check[col] == check[f"{col}_reference"]).all(): mismatch.append(col)
        for col in ["entry_price", "exit_price", "R"]:
            if not np.allclose(check[col], check[f"{col}_reference"], rtol=0, atol=5e-4): mismatch.append(col)
    report = {"passed": not mismatch, "actual_holdout_trades": len(actual), "reference_holdout_trades": len(expected), "mismatched_fields": mismatch}
    if mismatch: raise ValueError(f"STOP: baseline holdout comparison failed: {report}")
    return report


def metrics(trades: pd.DataFrame) -> dict:
    ordered = trades.sort_values(["close_time", "ticket"], kind="stable")
    r = ordered.R.to_numpy(float)
    equity = np.r_[0., np.cumsum(r)]
    gross_profit, gross_loss = r[r > 0].sum(), -r[r < 0].sum()
    return {"trades": len(trades), "net_profit_R": float(r.sum()), "max_dd_R": float(np.max(np.maximum.accumulate(equity)-equity)), "profit_factor": float(gross_profit/gross_loss) if gross_loss else float("inf"), "win_rate_pct": float((r > 0).mean()*100)}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    model_path = paths.HOLDOUT_STAGE2_DIR / f"catboost_final_holdout_run{args.run_id}.cbm"
    out_dir.mkdir(parents=True, exist_ok=True)
    features = list(FEATURES)
    if len(features) != 23: raise ValueError(f"Expected exactly 23 FEATURES, got {len(features)}")
    before = sha256(HOLDOUT)
    holdout = pd.read_csv(HOLDOUT, parse_dates=["entry_time", "label_end_time"])
    if len(holdout) != 5028 or set(features + ["label", "origin_bar", "entry_bar", "leg", "entry_time", "entry_price"]) - set(holdout.columns): raise ValueError("Holdout schema invalid")
    model = CatBoostClassifier(); model.load_model(model_path)
    holdout["probability"] = model.predict_proba(holdout[features])[:, 1]
    if sha256(HOLDOUT) != before: raise RuntimeError("STOP: holdout dataset changed during scoring")
    auc, f1 = roc_auc_score(holdout.label, holdout.probability), f1_score(holdout.label, holdout.probability >= .5)
    holdout.to_csv(out_dir / "holdout_scored.csv", index=False, float_format="%.12g", date_format="%Y-%m-%d %H:%M:%S")

    m15, m1_high, m1_low, lo, hi, raw_start, raw_end = load_market()
    trades = replay_baseline(m15, m1_high, m1_low, lo, hi)
    comparison = validate_holdout_baseline(trades)
    baseline = trades.loc[trades.open_time >= HOLDOUT_START].copy()
    universe = holdout.merge(baseline, left_on=["origin_bar","entry_bar","leg"], right_on=["origin_bar","entry_bar","leg"], how="left", validate="one_to_one", indicator=True, suffixes=("_dataset", ""))
    if not (universe._merge == "both").all() or len(universe) != len(holdout): raise ValueError("STOP: scored holdout and fixed baseline universe do not match one-to-one")
    if not (universe.entry_time == universe.open_time).all() or not np.allclose(universe.entry_price_dataset, universe.entry_price, rtol=0, atol=1e-8): raise ValueError("STOP: holdout metadata does not match baseline replay")
    universe = universe.drop(columns=["_merge", "entry_price_dataset"])
    universe.insert(0, "row_id", np.arange(len(universe)))
    rows = [{"keep_pct": 100, "filter": "Baseline", **metrics(universe)}]
    selections = {}
    for pct in KEEP_RATES:
        count = int(np.ceil(len(universe) * pct / 100))
        selected = universe.sort_values(["probability", "row_id"], ascending=[False, True], kind="stable").iloc[:count].copy()
        if len(selected) != count: raise ValueError("Top-k size failure")
        selections[pct] = selected
        selected.to_csv(out_dir / f"holdout_trades_top{pct}.csv", index=False, float_format="%.12g", date_format="%Y-%m-%d %H:%M:%S")
        rows.append({"keep_pct": pct, "filter": f"Top {pct}%", **metrics(selected)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "stage3_backtest_summary.csv", index=False, float_format="%.12g")
    universe.to_csv(out_dir / "holdout_fixed_trade_universe_scored.csv", index=False, float_format="%.12g", date_format="%Y-%m-%d %H:%M:%S")
    report = {"holdout_rows_scored": len(holdout), "features_from_build_features": features, "holdout_file_unchanged": True, "classification": {"roc_auc": float(auc), "f1_at_0_5": float(f1)}, "data_range": {"raw_m1_start": str(raw_start), "raw_m1_end": str(raw_end), "m15_resample_start": str(m15.index.min()), "m15_resample_end": str(m15.index.max()), "metric_holdout_start": str(HOLDOUT_START), "metric_holdout_end": str(baseline.open_time.max())}, "baseline_comparison": comparison, "baseline_full_history_trades": len(trades), "results": summary.to_dict(orient="records"), "top_k_rule": "keep_count=ceil(5028*keep_pct/100); sort probability descending then row_id ascending", "randomness": "No random component in scoring or backtest."}
    (out_dir / "stage3_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
