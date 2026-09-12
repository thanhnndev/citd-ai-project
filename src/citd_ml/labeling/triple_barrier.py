"""Create Triple Barrier labels while preserving PyramidStrategy entry selection.

The original strategy is used unchanged to determine when every order opens
and closes. Triple Barrier labels are calculated separately for every position
and never alter the strategy state.

Output columns: leg (pyramid leg), origin_bar (source signal bar), entry_time
(entry timestamp), entry_price (M15 open), atr_frozen (prior ATR),
momentum_prev/momentum_prev2 and vwap_prev/vwap_prev2 (origin-signal values),
upper_barrier/lower_barrier (fixed barriers), label (0/1), bars_to_label (M15
bars processed), and exit_touch_time (M1 label touch time).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from citd_ml import paths
from citd_ml.strategy.pyramid_strategy import PyramidStrategy


HOLDOUT_START = pd.Timestamp(paths.HOLDOUT_START)
REQUIRED_COLUMNS = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
DEFAULT_DATA = paths.RAW_M1_CSV
DEFAULT_OUTPUT = paths.LABELS_CSV
OUTPUT_COLUMNS = [
    "leg", "origin_bar", "entry_time", "entry_price", "atr_frozen", "momentum_prev", "momentum_prev2",
    "vwap_prev", "vwap_prev2", "upper_barrier", "lower_barrier", "label",
    "bars_to_label", "exit_touch_time",
]
FLOAT_COLUMNS = [
    "entry_price", "atr_frozen", "momentum_prev", "momentum_prev2",
    "vwap_prev", "vwap_prev2", "upper_barrier", "lower_barrier",
]
INTEGER_COLUMNS = ["leg", "origin_bar", "label", "bars_to_label"]


def prepare_data(m1_csv: Path, holdout_start: pd.Timestamp):
    """Load M1, strictly exclude holdout, then reproduce backtest M15 mapping."""
    m1 = pd.read_csv(m1_csv)
    missing = set(REQUIRED_COLUMNS).difference(m1.columns)
    if missing:
        raise ValueError(f"Missing required M1 columns: {sorted(missing)}")

    m1["dt"] = pd.to_datetime(
        m1["Date"].astype(str) + " " + m1["Time"].astype(str),
        format="%Y.%m.%d %H:%M:%S",
        errors="raise",
    )
    m1 = m1.set_index("dt").sort_index()
    m1 = m1.loc[m1.index < holdout_start].copy()
    if m1.empty:
        raise ValueError("No M1 bars remain before the holdout boundary.")

    ohlcv = ["Open", "High", "Low", "Close", "Volume"]
    m1[ohlcv] = m1[ohlcv].apply(pd.to_numeric, errors="raise")
    m15 = (
        m1.resample("15min")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )

    # This is the same M1-to-M15 ownership mapping used by the original backtest.
    slot = m1.index.floor("15min")
    bar_number = pd.Series(np.arange(len(m15)), index=m15.index)
    owner = bar_number.reindex(slot).to_numpy()
    keep = ~np.isnan(owner)
    owner = owner[keep].astype(np.int64)
    lo = np.searchsorted(owner, np.arange(len(m15)), side="left")
    hi = np.searchsorted(owner, np.arange(len(m15)), side="right")

    return (
        m1,
        m15,
        m1.index.to_numpy()[keep],
        m1["High"].to_numpy()[keep],
        m1["Low"].to_numpy()[keep],
        lo,
        hi,
    )


def calculate_barriers(entry_price: float, atr_frozen: float, atr_multiple: float, sl_pct: float):
    """Return the fixed upper and lower Triple Barrier prices."""
    return entry_price + atr_multiple * atr_frozen, entry_price * (1.0 - sl_pct)


def label_one_entry(
    entry_bar: int,
    upper: float,
    lower: float,
    m1_times: np.ndarray,
    m1_high: np.ndarray,
    m1_low: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    vertical_bars: int = 50,
):
    """Apply the six Triple Barrier rules, returning label, bars, touch time, and horizon status.

    ``bars_to_label`` is one-based: a decision in the entry M15 bar is 1 and
    vertical expiry after all permitted bars is 50.
    """
    last_bar_exclusive = min(entry_bar + vertical_bars, len(lo))
    for bar in range(entry_bar, last_bar_exclusive):
        upper_touch_time = None
        for idx in range(lo[bar], hi[bar]):
            high, low = m1_high[idx], m1_low[idx]
            if high >= upper and low <= lower:  # Rule 1: same M1 candle hits both.
                return 0, bar - entry_bar + 1, m1_times[idx], False
            if low <= lower:  # Rule 2: lower barrier always has priority.
                return 0, bar - entry_bar + 1, m1_times[idx], False
            if high >= upper and upper_touch_time is None:  # Rule 3: keep scanning this M15 bar.
                upper_touch_time = m1_times[idx]
        if upper_touch_time is not None:  # Rule 4: safe at the end of this M15 bar.
            return 1, bar - entry_bar + 1, upper_touch_time, False

    # Rule 6. The final Boolean separately records whether the source ended
    # before a complete 50-bar horizon could be observed.
    return 0, last_bar_exclusive - entry_bar, pd.NaT, last_bar_exclusive < entry_bar + vertical_bars


def run_strategy_and_label(m15, m1_times, m1_high, m1_low, lo, hi, PyramidStrategy):
    """Run original state management and calculate independent labels for base entries."""
    strategy = PyramidStrategy(k=3)
    indicators = strategy.prepare(
        m15["Open"].to_numpy(),
        m15["High"].to_numpy(),
        m15["Low"].to_numpy(),
        m15["Close"].to_numpy(),
        m15["Volume"].to_numpy(),
    )
    timestamps = m15.index
    weekdays = timestamps.dayofweek.to_numpy()
    seconds = (timestamps.hour * 3600 + timestamps.minute * 60).to_numpy()
    close = m15["Close"].to_numpy()
    rows = []
    closed_orders = 0

    for i in range(len(m15)):
        # Detect every position actually created by the unchanged open_bar method.
        before_ids = {id(position) for position in strategy.positions}
        strategy.open_bar(i, indicators)
        new_positions = [
            position
            for position in strategy.positions
            if id(position) not in before_ids and position.entry_bar == i
        ]

        for position in new_positions:
            upper, lower = calculate_barriers(
                position.entry_price, position.atr_frozen, strategy.atr_mult, strategy.sl_pct
            )
            label, bars_to_label, touch_time, incomplete_horizon = label_one_entry(
                i, upper, lower, m1_times, m1_high, m1_low, lo, hi
            )
            origin = position.origin_bar
            rows.append(
                {
                    "leg": position.leg,
                    "origin_bar": origin,
                    "entry_time": timestamps[i],
                    "entry_price": position.entry_price,
                    "atr_frozen": position.atr_frozen,
                    "momentum_prev": indicators["mom"][origin - 1],
                    "momentum_prev2": indicators["mom"][origin - 2],
                    "vwap_prev": indicators["vwap"][origin - 1],
                    "vwap_prev2": indicators["vwap"][origin - 2],
                    "upper_barrier": upper,
                    "lower_barrier": lower,
                    "label": label,
                    "bars_to_label": bars_to_label,
                    "exit_touch_time": touch_time,
                    "_incomplete_horizon": incomplete_horizon,
                }
            )

        # These calls are intentionally the original strategy exit logic. They
        # change only base_open / future eligibility, never Triple Barrier labels.
        closed = strategy.scan_bar(m1_high[lo[i]:hi[i]], m1_low[lo[i]:hi[i]])
        closed += strategy.friday_close(weekdays[i], seconds[i], close[i])
        closed_orders += len(closed)

    return pd.DataFrame(rows), closed_orders


def print_summary(labels: pd.DataFrame, closed_orders: int) -> None:
    total = len(labels)
    label_1 = int((labels["label"] == 1).sum())
    label_0 = int((labels["label"] == 0).sum())
    incomplete = int(labels["_incomplete_horizon"].sum())
    print(f"All positions found (all legs): {total:,}")
    for leg in range(4):
        print(f"  leg={leg}: {(labels['leg'] == leg).sum():,}")
    print(f"Label 1: {label_1:,} ({label_1 / total:.2%})")
    print(f"Label 0: {label_0:,} ({label_0 / total:.2%})")
    print(f"All rows that original backtest would write (closed positions): {closed_orders:,}")
    if total != closed_orders:
        print(
            "INFO: raw position count differs before filtering because incomplete horizons "
            "and/or positions still open at the end are not written as closed trades."
        )
    if incomplete:
        print(
            f"WARNING: {incomplete:,} labels reached the end of the pre-holdout source before "
            "50 M15 bars; they are label 0 with empty exit_touch_time."
        )


def prepare_handoff_output(labels: pd.DataFrame) -> pd.DataFrame:
    """Filter incomplete horizons and enforce the exact final CSV schema and types."""
    output = labels.loc[~labels["_incomplete_horizon"], OUTPUT_COLUMNS].copy()
    output["entry_time"] = pd.to_datetime(output["entry_time"], errors="raise")
    output["exit_touch_time"] = pd.to_datetime(output["exit_touch_time"], errors="raise")
    output[FLOAT_COLUMNS] = output[FLOAT_COLUMNS].apply(pd.to_numeric, errors="raise").astype(float)
    output[INTEGER_COLUMNS] = output[INTEGER_COLUMNS].apply(pd.to_numeric, errors="raise").astype(int)
    if not output["label"].isin([0, 1]).all():
        raise ValueError("Output label must contain only 0 or 1.")
    return output.sort_values("entry_time", kind="stable").reset_index(drop=True)


def print_handoff_summary(output: pd.DataFrame, output_path: Path) -> None:
    """Print final delivery details after the CSV has been successfully written."""
    total = len(output)
    label_1 = int((output["label"] == 1).sum())
    label_0 = int((output["label"] == 0).sum())
    null_counts = output.isna().sum()
    null_counts = null_counts[null_counts > 0]
    print("\nHANDOFF SUMMARY / TOM TAT BAN GIAO")
    print(f"Saved file: {output_path.resolve()}")
    print(f"Total rows: {total:,}")
    print(f"Entry-time range: {output['entry_time'].min():%Y-%m-%d %H:%M:%S} to {output['entry_time'].max():%Y-%m-%d %H:%M:%S}")
    print(f"Label 1 / Label 0: {label_1:,} ({label_1 / total:.2%}) / {label_0:,} ({label_0 / total:.2%})")
    if null_counts.empty:
        print("NULL CHECK: OK - no NaN/null values in any output column.")
    else:
        print("WARNING: NaN/null values found in output columns:")
        for column, count in null_counts.items():
            print(f"  {column}: {count:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Triple Barrier labels from PyramidStrategy base entries.")
    parser.add_argument("--m1-csv", type=Path, default=DEFAULT_DATA, help="M1 CSV source; any post-holdout rows are ignored.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument("--holdout", default=str(HOLDOUT_START), help="Holdout start timestamp (exclusive).")
    args = parser.parse_args()

    holdout_start = pd.Timestamp(args.holdout)
    m1, m15, m1_times, m1_high, m1_low, lo, hi = prepare_data(args.m1_csv, holdout_start)
    print(f"M1 bars before holdout: {len(m1):,}")
    print(f"M15 bars before holdout: {len(m15):,}")
    labels, closed_orders = run_strategy_and_label(m15, m1_times, m1_high, m1_low, lo, hi, PyramidStrategy)
    print_summary(labels, closed_orders)

    removed_count = int(labels["_incomplete_horizon"].sum())
    labels_to_save = prepare_handoff_output(labels)
    print(f"Incomplete-horizon rows removed: {removed_count:,}")
    print(f"Rows remaining after filter: {len(labels_to_save):,}")
    if len(labels_to_save) == closed_orders:
        print("CHECK PASSED: final output count matches all closed positions in the original backtest.")
    else:
        print("WARNING: final output count does not match all closed positions in the original backtest.")
    labels_to_save.to_csv(
        args.output, index=False, float_format="%.8f", date_format="%Y-%m-%d %H:%M:%S"
    )
    print_handoff_summary(labels_to_save, args.output)


if __name__ == "__main__":
    main()
