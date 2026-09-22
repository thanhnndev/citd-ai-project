"""Build the CatBoost dataset: fresh features + Tuan's Triple Barrier labels.

Design rules enforced here:
  * Every feature is computed from raw M15 OHLCV at the entry bar of THAT
    position, using only bars that closed before it (index entry_bar-1) plus
    the entry bar's own open. No frozen strategy indicator is reused.
  * Labels come from citd_ml.labeling.triple_barrier unchanged, for all 4 legs.
  * Nothing that encodes the absolute price level or the calendar position of
    the sample is emitted as a feature (see EXCLUDED below).
  * label_end_time is emitted so Cach 3 can purge/embargo correctly.

Output: dataset_catboost.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from citd_ml import paths
from citd_ml.labeling import triple_barrier as lab
from citd_ml.strategy import pyramid_strategy as strat_mod

HOLDOUT_START = pd.Timestamp(paths.HOLDOUT_START)
VERTICAL_BARS = paths.VERTICAL_BARS
SL_PCT = 0.006
ATR_MULT = 3.8

FEATURES = [
    # --- barrier geometry: how far the upper barrier is, in stop-loss units ---
    "breakeven_R",
    # --- volatility regime ---
    "atr14_pct", "atr_ratio_14_90", "vol20", "vol200", "vol_ratio_20_200", "range_pct",
    # --- trend / location in range (all ATR-normalised) ---
    "dist_ema20_atr", "dist_ema50_atr", "dist_ema200_atr",
    "ret20_atr", "ret50_atr", "pos_in_range50", "dist_hh20_atr",
    # --- the strategy's own signal, recomputed fresh at this bar ---
    "mom14", "mom_diff", "vwap_dist_atr", "vwap_slope_atr", "rsi14",
    # --- volume ---
    "vol_ratio_volume",
    # --- entry microstructure ---
    "gap_open_atr",
    # --- session ---
    "hour", "dow",
]

# Emitted for splitting / purging / reporting only. NEVER pass to the model.
#
# leg and entry_vs_base_R live here on purpose. They describe this project's
# scaffolding, not the market: a real strategy has no notion of "I am the 3rd
# pyramid leg of a family" or "I entered 0.4 stop-widths above the base order".
# The k=3 pyramid is a deliberately extreme stand-in for the ordinary label
# overlap any strategy produces, so handing the model a piece of that
# scaffolding would both leak family structure directly and make the measured
# bias impossible to generalise. Siblings are still told apart by genuine
# market differences (gap_open_atr 0.001, mom_diff 0.018, vwap_slope_atr 0.157).
META = ["entry_time", "label_end_time", "origin_bar", "entry_bar", "entry_price",
        "leg", "entry_vs_base_R"]

# Deliberately EXCLUDED, see docs/bao_cao_feature.md §3 for the full reasoning:
#   atr90_pct        - Spearman 1.000 with breakeven_R (identical feature)
#   entry_price      - near-perfect family fingerprint (sibling corr 0.9999)
#   upper_barrier / lower_barrier / atr_frozen (raw) - same problem
#   vol50, ret5_atr, pos_in_range20 - redundant with siblings already in the set


def resolve_positions(m15, m1_times, m1_high, m1_low, lo, hi, lab, PyramidStrategy):
    """Replay the unchanged strategy and label every position it opens."""
    strategy = PyramidStrategy(k=3)
    ind = strategy.prepare(
        m15["Open"].to_numpy(), m15["High"].to_numpy(), m15["Low"].to_numpy(),
        m15["Close"].to_numpy(), m15["Volume"].to_numpy(),
    )
    ts = m15.index
    weekdays = ts.dayofweek.to_numpy()
    seconds = (ts.hour * 3600 + ts.minute * 60).to_numpy()
    close = m15["Close"].to_numpy()

    rows = []
    for i in range(len(m15)):
        before = {id(p) for p in strategy.positions}
        strategy.open_bar(i, ind)
        opened = [p for p in strategy.positions if id(p) not in before and p.entry_bar == i]
        for p in opened:
            upper, lower = lab.calculate_barriers(
                p.entry_price, p.atr_frozen, strategy.atr_mult, strategy.sl_pct
            )
            label, bars, touch, incomplete = lab.label_one_entry(
                i, upper, lower, m1_times, m1_high, m1_low, lo, hi, vertical_bars=VERTICAL_BARS
            )
            rows.append({
                "leg": p.leg, "origin_bar": p.origin_bar, "entry_bar": i,
                "entry_time": ts[i], "entry_price": p.entry_price, "atr_frozen": p.atr_frozen,
                "label": label, "bars_to_label": bars, "exit_touch_time": touch,
                "_incomplete": incomplete,
            })
        strategy.scan_bar(m1_high[lo[i]:hi[i]], m1_low[lo[i]:hi[i]])
        strategy.friday_close(weekdays[i], seconds[i], close[i])

    return pd.DataFrame(rows)


def build_features(m15, pos, strat_mod):
    o = m15["Open"].to_numpy(); h = m15["High"].to_numpy()
    l = m15["Low"].to_numpy(); c = m15["Close"].to_numpy(); v = m15["Volume"].to_numpy()
    cs, vs = pd.Series(c), pd.Series(v)
    ret = cs.pct_change()

    atr14 = strat_mod.atr(h, l, c, 14)
    atr90 = strat_mod.atr(h, l, c, 90)
    vol20 = ret.rolling(20).std().to_numpy()
    vol200 = ret.rolling(200).std().to_numpy()
    ema20 = cs.ewm(span=20, adjust=False).mean().to_numpy()
    ema50 = cs.ewm(span=50, adjust=False).mean().to_numpy()
    ema200 = cs.ewm(span=200, adjust=False).mean().to_numpy()
    dif = cs.diff()
    rsi14 = (100 - 100 / (1 + dif.clip(lower=0).rolling(14).mean()
                          / (-dif.clip(upper=0)).rolling(14).mean())).to_numpy()
    mom14 = strat_mod.momentum(c, 14)
    vwap142 = strat_mod.vwap(h, l, c, v, 142)
    volma50 = vs.rolling(50).mean().to_numpy()
    hh20 = pd.Series(h).rolling(20).max().to_numpy()
    hh50 = pd.Series(h).rolling(50).max().to_numpy()
    ll50 = pd.Series(l).rolling(50).min().to_numpy()

    def back(a, k):
        return pd.Series(a).shift(k).to_numpy()

    ei = pos["entry_bar"].to_numpy()
    if ei.size and ei.min() < 1:
        # p = ei - 1 would wrap to the last bar of the series and leak the
        # future into the row. PyramidStrategy.warmup keeps entry_bar >= 200,
        # so this only fires if that invariant is ever broken.
        raise ValueError("entry_bar must be >= 1 so the previous bar exists")
    p = ei - 1                      # last fully-closed bar before entry
    entry_price = pos["entry_price"].to_numpy()
    atr_frozen = pos["atr_frozen"].to_numpy()

    base_price = pos.loc[pos["leg"] == 0].set_index("origin_bar")["entry_price"]
    base_of = pos["origin_bar"].map(base_price).to_numpy()

    f = pd.DataFrame(index=pos.index)
    f["breakeven_R"] = ATR_MULT * atr_frozen / (entry_price * SL_PCT)
    f["atr14_pct"] = atr14[p] / c[p]
    f["atr_ratio_14_90"] = atr14[p] / atr90[p]
    f["vol20"] = vol20[p]
    f["vol200"] = vol200[p]
    f["vol_ratio_20_200"] = vol20[p] / vol200[p]
    f["range_pct"] = (h[p] - l[p]) / c[p]
    f["dist_ema20_atr"] = (c[p] - ema20[p]) / atr14[p]
    f["dist_ema50_atr"] = (c[p] - ema50[p]) / atr14[p]
    f["dist_ema200_atr"] = (c[p] - ema200[p]) / atr14[p]
    f["ret20_atr"] = (c[p] - back(c, 20)[p]) / atr14[p]
    f["ret50_atr"] = (c[p] - back(c, 50)[p]) / atr14[p]
    f["pos_in_range50"] = (c[p] - ll50[p]) / np.maximum(hh50[p] - ll50[p], 1e-9)
    f["dist_hh20_atr"] = (c[p] - hh20[p]) / atr14[p]
    f["mom14"] = mom14[p]
    f["mom_diff"] = mom14[p] - back(mom14, 1)[p]
    f["vwap_dist_atr"] = (c[p] - vwap142[p]) / atr14[p]
    f["vwap_slope_atr"] = (vwap142[p] - back(vwap142, 1)[p]) / atr14[p]
    f["rsi14"] = rsi14[p]
    f["vol_ratio_volume"] = v[p] / volma50[p]
    f["gap_open_atr"] = (o[ei] - c[p]) / atr14[p]
    f["leg"] = pos["leg"].to_numpy()
    f["entry_vs_base_R"] = (entry_price / base_of - 1) / SL_PCT
    f["hour"] = m15.index.hour.to_numpy()[ei]
    f["dow"] = m15.index.dayofweek.to_numpy()[ei]
    return f


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CatBoost dataset for the pyramid strategy.")
    ap.add_argument("--m1-csv", type=Path, default=paths.RAW_M1_CSV)
    ap.add_argument("--output", type=Path, default=paths.DATASET_CSV)
    ap.add_argument("--holdout", default=str(HOLDOUT_START))
    args = ap.parse_args()

    m1, m15, m1_times, m1_high, m1_low, lo, hi = lab.prepare_data(
        args.m1_csv, pd.Timestamp(args.holdout)
    )
    print(f"M1 bars: {len(m1):,}   M15 bars: {len(m15):,}")

    pos = resolve_positions(m15, m1_times, m1_high, m1_low, lo, hi, lab, strat_mod.PyramidStrategy)
    print(f"Positions opened (all legs): {len(pos):,}")

    feats = build_features(m15, pos, strat_mod)

    # label_end_time = when the label became knowable. Cach 3 purges on this.
    ts = m15.index
    end_bar = np.minimum(pos["entry_bar"].to_numpy() + VERTICAL_BARS - 1, len(m15) - 1)
    expiry_time = ts[end_bar] + pd.Timedelta(minutes=15)
    touch = pd.to_datetime(pos["exit_touch_time"])
    label_end = touch.where(touch.notna(), pd.Series(expiry_time, index=pos.index))

    out = feats.copy()
    out["label"] = pos["label"].to_numpy()
    out["entry_time"] = pos["entry_time"].to_numpy()
    out["label_end_time"] = label_end.to_numpy()
    out["origin_bar"] = pos["origin_bar"].to_numpy()
    out["entry_bar"] = pos["entry_bar"].to_numpy()
    out["entry_price"] = pos["entry_price"].to_numpy()
    # leg / entry_vs_base_R come along from build_features() and are routed to
    # META below, so they reach the CSV for analysis but never reach the model.

    out = out.loc[~pos["_incomplete"].to_numpy()].copy()
    out = out.sort_values(["entry_time", "leg"], kind="stable").reset_index(drop=True)
    out = out[FEATURES + ["label"] + META]

    out.to_csv(args.output, index=False, float_format="%.8f",
               date_format="%Y-%m-%d %H:%M:%S")

    print(f"\nSaved: {args.output}")
    print(f"Rows: {len(out):,}   Features: {len(FEATURES)}")
    print(f"Label 1: {out['label'].mean():.2%}")
    print(f"Families (origin_bar): {out['origin_bar'].nunique():,}")
    print(f"Entry range: {out['entry_time'].min()} -> {out['entry_time'].max()}")
    nulls = out[FEATURES].isna().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print("NaN check: OK (no NaN in any feature)")
    else:
        print("NaN present (CatBoost handles natively):")
        for k, n in nulls.items():
            print(f"  {k}: {n:,}")


if __name__ == "__main__":
    main()
