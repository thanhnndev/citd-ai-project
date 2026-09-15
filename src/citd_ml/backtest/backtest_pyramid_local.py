"""Backtest the pyramid strategy and evaluate the four CatBoost OOF branches.

Each filtered replay uses a shadow strategy to keep the candidate-trade universe
identical to the dataset used by CatBoost. OOF ranks gate execution at opening.
This assumes rejected orders still affect the shadow's future signals; it is
not a simulation of signal generation from accepted positions alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from citd_ml import paths
from citd_ml.strategy.pyramid_strategy import PyramidStrategy
from citd_ml.training.pre_train import HOLDOUT_START, prepare_dataset
from citd_ml.training.split_data import make_all_folds


# --------------------------------------------------------------------- paths
SRC = paths.RAW_M1_CSV
REFERENCE_TRADELIST = paths.TRADELIST_CSV
OOF_DIR = paths.CATBOOST_TRAINING_DIR
OUTPUT_DIR = paths.BACKTEST_DIR

BASELINE_OUT = OUTPUT_DIR / "baseline_tradelist.csv"
SCORED_UNIVERSE_OUT = OUTPUT_DIR / "backtest_scored_universe.csv"
TOP50_SUMMARY_OUT = OUTPUT_DIR / "backtest_summary_top50.csv"
RETENTION_SWEEP_OUT = OUTPUT_DIR / "backtest_retention_sweep.csv"


def _resolve_dir(value: Path | str | None, default: Path) -> Path:
    """Cho phép ghi đè thư mục nhưng mặc định giữ nguyên đường dẫn chuẩn."""
    return default if value is None else Path(value)


# --------------------------------------------------------------- task config
K = 3
EXPECTED_DATASET_ROWS = 25_008
KEEP_RATES = (20, 30, 40, 50, 60, 70, 80)

METHODS = {
    "random_kfold": "Cách 1 - Random K-Fold",
    "grouped_kfold": "Cách 1b - Grouped K-Fold",
    "walk_forward": "Cách 2 - Walk-forward",
    "purged_walk_forward": "Cách 3 - Purged Walk-forward",
}


def _require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    """Fail early when an input artifact does not have the required schema."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} thiếu cột bắt buộc: {missing}")


# ---------------------------------------------------------------------- load
def load(path: Path = SRC):
    """Load M1 data strictly before the mandatory holdout boundary."""
    print(f"Reading data from: {path.resolve()}")
    m1 = pd.read_csv(path)
    _require_columns(
        m1,
        ["Date", "Time", "Open", "High", "Low", "Close", "Volume"],
        path.name,
    )

    m1["dt"] = pd.to_datetime(
        m1["Date"] + " " + m1["Time"],
        format="%Y.%m.%d %H:%M:%S",
    )
    m1 = m1.loc[m1["dt"] < HOLDOUT_START].copy()
    if m1.empty:
        raise ValueError("Không còn dữ liệu M1 trước HOLDOUT_START")

    m1 = m1.set_index("dt").sort_index(kind="stable")
    if not m1.index.is_monotonic_increasing:
        raise ValueError("Mốc thời gian M1 không tăng dần")

    print(f"  Holdout  : dt < {HOLDOUT_START}")
    print(f"  M1 bars  : {len(m1):,}")

    m15 = (
        m1.resample("15min")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )
    print(f"  M15 bars : {len(m15):,}")

    slot = m1.index.floor("15min")
    pos = pd.Series(np.arange(len(m15)), index=m15.index)
    owner = pos.reindex(slot).values

    keep = ~np.isnan(owner)
    owner = owner[keep].astype(np.int64)
    m1_high = m1["High"].to_numpy()[keep]
    m1_low = m1["Low"].to_numpy()[keep]

    lo = np.searchsorted(owner, np.arange(len(m15)), "left")
    hi = np.searchsorted(owner, np.arange(len(m15)), "right")
    return m15, m1_high, m1_low, lo, hi


# ------------------------------------------------------------------ backtest
def backtest(
    m15,
    m1_high,
    m1_low,
    lo,
    hi,
    strategy=None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Replay the unchanged strategy and return every completed candidate trade."""
    strat = strategy or PyramidStrategy(k=K)
    ind = strat.prepare(
        m15["Open"].to_numpy(),
        m15["High"].to_numpy(),
        m15["Low"].to_numpy(),
        m15["Close"].to_numpy(),
        m15["Volume"].to_numpy(),
    )

    ts = m15.index
    close = m15["Close"].to_numpy()
    weekday = ts.dayofweek.to_numpy()
    seconds = (ts.hour * 3600 + ts.minute * 60).to_numpy()

    rows = []
    n = len(m15)
    milestone = max(1, n // 20)
    for i in range(n):
        if show_progress and i % milestone == 0:
            print(f"  Backtest: {i:,}/{n:,} ({100 * i / n:.0f}%)", flush=True)

        strat.open_bar(i, ind)

        closed = strat.scan_bar(m1_high[lo[i] : hi[i]], m1_low[lo[i] : hi[i]])
        closed += strat.friday_close(weekday[i], seconds[i], close[i])

        for position, reason, exit_price in closed:
            rows.append(
                (
                    position.origin_bar,
                    position.leg,
                    position.entry_bar,
                    i,
                    position.entry_price,
                    exit_price,
                    reason,
                )
            )

    trades = pd.DataFrame(
        rows,
        columns=[
            "origin_bar",
            "leg",
            "entry_bar",
            "exit_bar",
            "entry_price",
            "exit_price",
            "exit_reason",
        ],
    )
    if trades.empty:
        raise ValueError("Backtest không tạo được lệnh nào")

    trades = trades.sort_values(["entry_bar", "leg"], kind="stable").reset_index(drop=True)
    trades.insert(0, "ticket", np.arange(1, len(trades) + 1))
    trades.insert(1, "open_time", ts[trades["entry_bar"]].values)
    trades.insert(2, "close_time", ts[trades["exit_bar"]].values)
    trades.insert(3, "signal_time", ts[trades["origin_bar"]].values)
    trades["pl_pct"] = (trades["exit_price"] / trades["entry_price"] - 1) * 100
    trades["R"] = trades["pl_pct"] / (strat.sl_pct * 100)
    trades["bars_held"] = trades["exit_bar"] - trades["entry_bar"]

    key = ["origin_bar", "entry_bar", "leg"]
    if trades.duplicated(key).any():
        raise ValueError(f"Backtest sinh khóa lệnh trùng: {key}")
    return trades


def _position_key(position) -> tuple[int, int, int]:
    """Stable identifier shared by dataset rows and backtest positions."""
    return (
        int(position.origin_bar),
        int(position.entry_bar),
        int(position.leg),
    )


def backtest_filtered(
    m15,
    m1_high,
    m1_low,
    lo,
    hi,
    score_table: pd.DataFrame,
    accepted_row_ids: set[int],
    first_eval_row: int,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Replay a scored branch with the score checked at each intended opening.

    ``shadow`` keeps the original strategy state and therefore emits exactly
    the candidate universe represented in the CatBoost dataset. ``executor``
    receives only accepted positions and manages their stops and exits. This
    makes rejection a decision at opening time without creating new, unscored
    families when a base order is rejected.
    """
    shadow = PyramidStrategy(k=K)
    executor = PyramidStrategy(k=K)
    ind = shadow.prepare(
        m15["Open"].to_numpy(),
        m15["High"].to_numpy(),
        m15["Low"].to_numpy(),
        m15["Close"].to_numpy(),
        m15["Volume"].to_numpy(),
    )
    executor_ind = executor.prepare(
        m15["Open"].to_numpy(),
        m15["High"].to_numpy(),
        m15["Low"].to_numpy(),
        m15["Close"].to_numpy(),
        m15["Volume"].to_numpy(),
    )

    key_columns = ["origin_bar", "entry_bar", "leg"]
    score_lookup = score_table.set_index(key_columns)["row_id"].to_dict()
    max_scored_entry_bar = int(score_table["entry_bar"].max())
    accepted_row_ids = {int(row_id) for row_id in accepted_row_ids}
    if not accepted_row_ids.issubset(set(score_lookup.values())):
        raise ValueError("Tập lệnh được nhận chứa row_id không có trong OOF")

    ts = m15.index
    close = m15["Close"].to_numpy()
    weekday = ts.dayofweek.to_numpy()
    seconds = (ts.hour * 3600 + ts.minute * 60).to_numpy()

    rows = []
    seen_candidates: set[tuple[int, int, int]] = set()
    unscored_candidates: set[tuple[int, int, int]] = set()
    n = len(m15)
    milestone = max(1, n // 20)
    for i in range(n):
        if show_progress and i % milestone == 0:
            print(
                f"  Filtered backtest: {i:,}/{n:,} ({100 * i / n:.0f}%)",
                flush=True,
            )

        # The shadow emits the intended openings using the unchanged strategy.
        shadow.open_bar(i, ind)
        new_candidates = []
        for position in shadow.positions:
            key = _position_key(position)
            if key not in seen_candidates:
                seen_candidates.add(key)
                new_candidates.append(position)

        # Check the score before opening each candidate. Chunk 1 is warm-up
        # because Walk-forward and Purged Walk-forward have no score there.
        for position in new_candidates:
            key = _position_key(position)
            if key not in score_lookup:
                # A terminal candidate can be absent because it never closes.
                # Track it and verify that assumption when the shadow exits.
                if position.entry_bar <= max_scored_entry_bar:
                    raise ValueError(f"Ứng viên chiến lược không có trong OOF: {key}")
                unscored_candidates.add(key)
                continue
            row_id = int(score_lookup[key])
            accept = row_id < first_eval_row or row_id in accepted_row_ids
            if accept:
                executor._open(
                    i,
                    executor_ind,
                    position.origin_bar,
                    position.leg,
                )

        # Advance the shadow's positions as well. Their close events are not
        # reported, but they must update shadow.base_open for future signals.
        shadow_closed = shadow.scan_bar(m1_high[lo[i] : hi[i]], m1_low[lo[i] : hi[i]])
        shadow_closed += shadow.friday_close(weekday[i], seconds[i], close[i])
        for position, _, _ in shadow_closed:
            if _position_key(position) in unscored_candidates:
                raise ValueError(
                    f"Ứng viên thiếu OOF đã đóng trước holdout: {_position_key(position)}"
                )

        # Match PyramidStrategy.open_bar's trailing-stop timing for accepted
        # positions, while keeping signal generation in the shadow only.
        for position in executor.positions:
            if position.entry_bar < i:
                position.trail(executor.atr_mult)

        closed = executor.scan_bar(m1_high[lo[i] : hi[i]], m1_low[lo[i] : hi[i]])
        closed += executor.friday_close(weekday[i], seconds[i], close[i])

        for position, reason, exit_price in closed:
            rows.append(
                (
                    position.origin_bar,
                    position.leg,
                    position.entry_bar,
                    i,
                    position.entry_price,
                    exit_price,
                    reason,
                )
            )

    if not set(score_lookup).issubset(seen_candidates):
        raise ValueError("Có lệnh OOF không xuất hiện trong luồng tín hiệu shadow")

    trades = pd.DataFrame(
        rows,
        columns=[
            "origin_bar",
            "leg",
            "entry_bar",
            "exit_bar",
            "entry_price",
            "exit_price",
            "exit_reason",
        ],
    )
    if trades.empty:
        raise ValueError("Filtered backtest không tạo được lệnh nào")

    trades = trades.sort_values(["entry_bar", "leg"], kind="stable").reset_index(drop=True)
    trades.insert(0, "ticket", np.arange(1, len(trades) + 1))
    trades.insert(1, "open_time", ts[trades["entry_bar"]].values)
    trades.insert(2, "close_time", ts[trades["exit_bar"]].values)
    trades.insert(3, "signal_time", ts[trades["origin_bar"]].values)
    trades["pl_pct"] = (trades["exit_price"] / trades["entry_price"] - 1) * 100
    trades["R"] = trades["pl_pct"] / (executor.sl_pct * 100)
    trades["bars_held"] = trades["exit_bar"] - trades["entry_bar"]

    if trades.duplicated(key_columns).any():
        raise ValueError("Filtered backtest sinh khóa lệnh trùng")
    return trades


def validate_against_reference(trades: pd.DataFrame) -> None:
    """Verify the unchanged strategy against the handed-over baseline file."""
    if not REFERENCE_TRADELIST.exists():
        raise FileNotFoundError(f"Thiếu tradelist bàn giao: {REFERENCE_TRADELIST}")

    reference = pd.read_csv(
        REFERENCE_TRADELIST,
        parse_dates=["signal_time", "open_time", "close_time"],
    )
    _require_columns(
        reference,
        [
            "signal_time",
            "open_time",
            "close_time",
            "leg",
            "entry_price",
            "exit_price",
            "R",
        ],
        REFERENCE_TRADELIST.name,
    )
    # The handed-over reference also contains later, out-of-scope trades.
    # Compare completed trades strictly before the same holdout boundary.
    reference = reference.loc[
        (reference["open_time"] < HOLDOUT_START)
        & (reference["close_time"] < HOLDOUT_START)
    ].copy()
    if len(trades) != len(reference):
        raise ValueError(
            f"Số lệnh baseline ({len(trades)}) khác tradelist bàn giao ({len(reference)})"
        )

    check = trades.merge(
        reference[
            [
                "signal_time",
                "open_time",
                "close_time",
                "leg",
                "entry_price",
                "exit_price",
                "R",
            ]
        ],
        on=["open_time", "leg"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_reference"),
    )
    if not (check["_merge"] == "both").all():
        missing = int((check["_merge"] != "both").sum())
        raise ValueError(f"Có {missing} lệnh không khớp giữa baseline và tradelist bàn giao")

    if not (check["signal_time"] == check["signal_time_reference"]).all():
        raise ValueError("signal_time khác tradelist bàn giao")
    if not (check["close_time"] == check["close_time_reference"]).all():
        raise ValueError("close_time khác tradelist bàn giao")

    for column in ["entry_price", "exit_price", "R"]:
        if not np.allclose(
            check[column].to_numpy(),
            check[f"{column}_reference"].to_numpy(),
            rtol=0.0,
            atol=5e-4,
        ):
            raise ValueError(f"{column} khác tradelist bàn giao")

    print(f"  Reference check: OK ({len(trades):,} lệnh khớp)")


# --------------------------------------------------------------- OOF scores
def load_score_tables(oof_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Load and strictly validate the four saved OOF probability tables."""
    target_dir = _resolve_dir(oof_dir, OOF_DIR)
    score_tables: dict[str, pd.DataFrame] = {}
    df, X, y, meta, _, chunks = prepare_dataset()
    meta_columns = list(meta.columns) + ["label"]
    expected_meta = df[meta_columns]
    expected_folds = make_all_folds(X, y, meta, chunks)
    first_eval_row = EXPECTED_DATASET_ROWS // 5

    for method in METHODS:
        path = target_dir / f"oof_{method}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Thiếu bảng điểm OOF: {path}")

        scores = pd.read_csv(path, parse_dates=["entry_time", "label_end_time"])
        _require_columns(
            scores,
            meta_columns + ["method", "fold", "probability"],
            path.name,
        )
        if len(scores) != EXPECTED_DATASET_ROWS:
            raise ValueError(
                f"{path.name}: có {len(scores):,} dòng, cần {EXPECTED_DATASET_ROWS:,}"
            )
        if not np.array_equal(scores["row_id"].to_numpy(), np.arange(len(scores))):
            raise ValueError(f"{path.name}: row_id không đúng 0..{len(scores) - 1}")
        if scores["method"].nunique() != 1 or scores["method"].iat[0] != method:
            raise ValueError(f"{path.name}: cột method không khớp tên file")
        if scores.duplicated(["origin_bar", "entry_bar", "leg"]).any():
            raise ValueError(f"{path.name}: khóa lệnh bị trùng")

        probability = scores["probability"]
        if method in {"random_kfold", "grouped_kfold"}:
            if probability.isna().any():
                raise ValueError(f"{path.name}: OOF phải phủ toàn bộ dataset")
        else:
            if probability.iloc[:first_eval_row].notna().any():
                raise ValueError(f"{path.name}: chunk 1 không được có dự đoán")
            if probability.iloc[first_eval_row:].isna().any():
                raise ValueError(f"{path.name}: chunk 2-5 phải có đủ dự đoán")

        finite_probability = probability.dropna().to_numpy()
        if not np.isfinite(finite_probability).all():
            raise ValueError(f"{path.name}: xác suất chứa NaN/Inf ngoài chunk 1")
        if not ((finite_probability >= 0.0) & (finite_probability <= 1.0)).all():
            raise ValueError(f"{path.name}: xác suất nằm ngoài [0, 1]")

        try:
            pd.testing.assert_frame_equal(
                scores[meta_columns], expected_meta,
                check_dtype=False, check_exact=True,
            )
        except AssertionError as error:
            raise ValueError(f"{path.name}: metadata/label khác dataset hiện tại") from error
        expected_fold_id = np.full(len(df), -1, dtype=np.int16)
        for fold_number, (_, test_idx) in enumerate(expected_folds[method], start=1):
            expected_fold_id[test_idx] = fold_number
        if not np.array_equal(scores["fold"].to_numpy(), expected_fold_id):
            raise ValueError(f"{path.name}: fold không đúng phép chia trong BAN_GIAO")

        score_tables[method] = scores

    print("  OOF check: OK (4 bảng điểm, cùng 25,008 lệnh)")
    return score_tables


def build_scored_universe(
    trades: pd.DataFrame,
    score_tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[int]]:
    """Attach all OOF scores to the same fixed candidate-trade universe."""
    if len(trades) != EXPECTED_DATASET_ROWS:
        raise ValueError("Baseline phải có đúng 25,008 lệnh, không bỏ qua lệnh dư")
    first_method = next(iter(METHODS))
    base_scores = score_tables[first_method][
        [
            "row_id",
            "entry_time",
            "origin_bar",
            "entry_bar",
            "entry_price",
            "leg",
            "label",
        ]
    ].copy()

    universe = base_scores.merge(
        trades,
        on=["origin_bar", "entry_bar", "leg"],
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_score", ""),
    )
    if not (universe["_merge"] == "both").all():
        missing = int((universe["_merge"] != "both").sum())
        raise ValueError(f"Có {missing} dòng OOF không tìm được kết quả backtest")
    universe = universe.drop(columns="_merge")

    if not (universe["entry_time"] == universe["open_time"]).all():
        raise ValueError("entry_time trong OOF không khớp open_time của backtest")
    if not np.allclose(
        universe["entry_price_score"].to_numpy(),
        universe["entry_price"].to_numpy(),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError("entry_price trong OOF không khớp backtest")
    universe = universe.drop(columns="entry_price_score")

    for method, scores in score_tables.items():
        probability_by_row = scores.set_index("row_id")["probability"]
        fold_by_row = scores.set_index("row_id")["fold"]
        universe[f"probability_{method}"] = universe["row_id"].map(probability_by_row)
        universe[f"fold_{method}"] = universe["row_id"].map(fold_by_row).astype(np.int64)

    n_rows = len(universe)
    edges = [n_rows * k // 5 for k in range(6)]
    universe["chunk"] = (
        np.searchsorted(
            np.asarray(edges[1:]),
            universe["row_id"].to_numpy(),
            side="right",
        )
        + 1
    )
    universe = universe.sort_values("row_id", kind="stable").reset_index(drop=True)

    if edges != [0, 5001, 10003, 15004, 20006, 25008]:
        raise ValueError(f"Ranh giới chunk không đúng BAN_GIAO: {edges}")
    if len(universe) != EXPECTED_DATASET_ROWS:
        raise ValueError("Vũ trụ lệnh backtest không đủ 25,008 dòng")

    print(f"  Join check: OK ({len(universe):,} OOF ↔ trade, edges={edges})")
    return universe, edges


# ----------------------------------------------------------- trade selection
def select_top_percent(
    evaluation: pd.DataFrame,
    probability_column: str,
    keep_pct: int,
) -> pd.DataFrame:
    """Keep an exact, deterministic top percentage by probability."""
    if not 0 < keep_pct <= 100:
        raise ValueError("keep_pct phải nằm trong (0, 100]")
    if evaluation[probability_column].isna().any():
        raise ValueError(f"{probability_column} còn thiếu xác suất trong chunk 2-5")

    keep_count = int(np.ceil(len(evaluation) * keep_pct / 100.0))
    ranked = evaluation.sort_values(
        [probability_column, "row_id"],
        ascending=[False, True],
        kind="stable",
    )
    selected = ranked.iloc[:keep_count].copy()
    if len(selected) != keep_count:
        raise ValueError("Số lệnh giữ lại không đúng tỷ lệ yêu cầu")
    return selected


def annotate_runtime_trades(
    runtime_trades: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """Attach dataset row IDs, chunk IDs and probabilities to executed trades."""
    key_columns = ["origin_bar", "entry_bar", "leg"]
    score_columns = ["row_id", "chunk"] + [
        column
        for column in universe.columns
        if column.startswith("probability_") or column.startswith("fold_")
    ]
    annotated = runtime_trades.merge(
        universe[key_columns + score_columns],
        on=key_columns,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not (annotated["_merge"] == "both").all():
        missing = int((annotated["_merge"] != "both").sum())
        raise ValueError(f"Filtered backtest có {missing} lệnh không ghép được row_id")
    annotated = annotated.drop(columns="_merge")
    if annotated["row_id"].duplicated().any():
        raise ValueError("Filtered backtest có row_id bị lặp")
    return annotated


def calculate_metrics(trades: pd.DataFrame) -> dict[str, float | int]:
    """Calculate the four mandatory metrics on chronologically closed trades."""
    if trades.empty:
        raise ValueError("Không thể tính metric trên tập lệnh rỗng")
    if trades["R"].isna().any() or not np.isfinite(trades["R"]).all():
        raise ValueError("Cột R có NaN/Inf")

    ordered = trades.sort_values(["close_time", "ticket"], kind="stable")
    r_values = ordered["R"].to_numpy(dtype=float)

    equity = np.concatenate(([0.0], np.cumsum(r_values)))
    running_peak = np.maximum.accumulate(equity)
    max_drawdown = float(np.max(running_peak - equity))

    gross_profit = float(r_values[r_values > 0].sum())
    gross_loss = float(-r_values[r_values < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "trades": len(trades),
        "net_profit_R": float(r_values.sum()),
        "max_dd_R": max_drawdown,
        "profit_factor": profit_factor,
        "win_rate_pct": float((r_values > 0).mean() * 100.0),
    }


def _metric_row(
    method: str,
    display_name: str,
    requested_keep_pct: int,
    selected: pd.DataFrame,
    universe_size: int,
) -> dict[str, float | int | str]:
    metrics = calculate_metrics(selected)
    return {
        "method": method,
        "display_name": display_name,
        "requested_keep_pct": requested_keep_pct,
        "actual_keep_pct": len(selected) / universe_size * 100.0,
        **metrics,
    }


def build_reports(
    universe: pd.DataFrame,
    first_eval_row: int,
    filtered_results: dict[tuple[str, int], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build reports from runtime-filtered trades on chunk 2-5."""
    evaluation = universe.loc[universe["row_id"] >= first_eval_row].copy()
    expected_evaluation_rows = EXPECTED_DATASET_ROWS - first_eval_row
    if len(evaluation) != expected_evaluation_rows:
        raise ValueError(
            f"Chunk 2-5 có {len(evaluation):,} dòng, cần {expected_evaluation_rows:,}"
        )
    if set(evaluation["chunk"].unique()) != {2, 3, 4, 5}:
        raise ValueError("Phạm vi đánh giá không đúng chunk 2-5")

    universe_size = len(evaluation)
    baseline_row = _metric_row("baseline", "Baseline", 100, evaluation, universe_size)

    top50_rows = [baseline_row]
    top50_trades: dict[str, pd.DataFrame] = {}
    for method, display_name in METHODS.items():
        selected = filtered_results[(method, 50)]
        selected = selected.loc[selected["row_id"] >= first_eval_row].copy()
        expected_count = int(np.ceil(len(evaluation) * 50 / 100.0))
        if len(selected) != expected_count:
            raise ValueError(
                f"{method} top 50% có {len(selected)} lệnh, cần {expected_count}"
            )
        top50_trades[method] = selected.sort_values("row_id", kind="stable")
        top50_rows.append(
            _metric_row(method, display_name, 50, selected, universe_size)
        )

    top50_summary = pd.DataFrame(top50_rows)

    sweep_rows = []
    for keep_pct in KEEP_RATES:
        baseline_at_rate = dict(baseline_row)
        baseline_at_rate["comparison_keep_pct"] = keep_pct
        sweep_rows.append(baseline_at_rate)

        for method, display_name in METHODS.items():
            selected = filtered_results[(method, keep_pct)]
            selected = selected.loc[selected["row_id"] >= first_eval_row].copy()
            expected_count = int(np.ceil(len(evaluation) * keep_pct / 100.0))
            if len(selected) != expected_count:
                raise ValueError(
                    f"{method} {keep_pct}% có {len(selected)} lệnh, cần {expected_count}"
                )
            row = _metric_row(method, display_name, keep_pct, selected, universe_size)
            row["comparison_keep_pct"] = keep_pct
            sweep_rows.append(row)

    retention_sweep = pd.DataFrame(sweep_rows)
    retention_sweep = retention_sweep[
        ["comparison_keep_pct"]
        + [
            column
            for column in retention_sweep.columns
            if column != "comparison_keep_pct"
        ]
    ]

    if len(top50_summary) != 5 or len(retention_sweep) != len(KEEP_RATES) * 5:
        raise ValueError("Số dòng báo cáo backtest không đúng yêu cầu")
    return top50_summary, retention_sweep, top50_trades


def save_outputs(
    trades: pd.DataFrame,
    universe: pd.DataFrame,
    top50_summary: pd.DataFrame,
    retention_sweep: pd.DataFrame,
    top50_trades: dict[str, pd.DataFrame],
    output_dir: Path | str | None = None,
) -> None:
    """Write only generated artifacts under outputs/backtest (or output_dir)."""
    target_dir = _resolve_dir(output_dir, OUTPUT_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    csv_options = {
        "index": False,
        "float_format": "%.12g",
        "date_format": "%Y-%m-%d %H:%M:%S",
    }
    trades.to_csv(target_dir / BASELINE_OUT.name, **csv_options)
    universe.to_csv(target_dir / SCORED_UNIVERSE_OUT.name, **csv_options)
    top50_summary.to_csv(target_dir / TOP50_SUMMARY_OUT.name, **csv_options)
    retention_sweep.to_csv(target_dir / RETENTION_SWEEP_OUT.name, **csv_options)

    for method, selected in top50_trades.items():
        selected.to_csv(target_dir / f"trades_top50_{method}.csv", **csv_options)


def print_summary(top50_summary: pd.DataFrame) -> None:
    """Print the exact table requested in section 5 of BAN_GIAO."""
    columns = [
        "display_name",
        "trades",
        "net_profit_R",
        "max_dd_R",
        "profit_factor",
        "win_rate_pct",
    ]
    print("\nKết quả backtest trên chunk 2-5 (baseline và top 50%):")
    print(
        top50_summary[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


def main(
    output_dir: Path | str | None = None,
    oof_dir: Path | str | None = None,
) -> None:
    """Run baseline, score each intended opening, and produce all reports."""
    m15, m1_high, m1_low, lo, hi = load()

    print("\nStarting unchanged baseline backtest...")
    trades = backtest(m15, m1_high, m1_low, lo, hi)
    print(f"  Completed candidate trades: {len(trades):,}")
    validate_against_reference(trades)

    print("\nLoading CatBoost OOF scores...")
    score_tables = load_score_tables(oof_dir)
    universe, edges = build_scored_universe(trades, score_tables)

    evaluation = universe.loc[universe["row_id"] >= edges[1]].copy()
    selected_row_ids: dict[tuple[str, int], set[int]] = {}
    for method in METHODS:
        for keep_pct in KEEP_RATES:
            selected = select_top_percent(
                evaluation,
                f"probability_{method}",
                keep_pct=keep_pct,
            )
            selected_row_ids[(method, keep_pct)] = set(selected["row_id"].astype(int))

    print("\nRunning score-gated backtests at intended openings...")
    filtered_results: dict[tuple[str, int], pd.DataFrame] = {}
    for method in METHODS:
        for keep_pct in KEEP_RATES:
            print(f"  {method}: giữ top {keep_pct}%")
            runtime_trades = backtest_filtered(
                m15,
                m1_high,
                m1_low,
                lo,
                hi,
                score_tables[method],
                selected_row_ids[(method, keep_pct)],
                first_eval_row=edges[1],
            )
            annotated = annotate_runtime_trades(runtime_trades, universe)
            actual_eval_ids = set(
                annotated.loc[annotated["row_id"] >= edges[1], "row_id"].astype(int)
            )
            if actual_eval_ids != selected_row_ids[(method, keep_pct)]:
                raise ValueError(
                    f"{method} {keep_pct}%: tập lệnh thực thi không khớp tập top-k"
                )
            filtered_results[(method, keep_pct)] = annotated

    top50_summary, retention_sweep, top50_trades = build_reports(
        universe,
        first_eval_row=edges[1],
        filtered_results=filtered_results,
    )
    if output_dir is None:
        # Giữ đúng 5 tham số để không phá vỡ call site cũ (verify_pipeline patch).
        save_outputs(trades, universe, top50_summary, retention_sweep, top50_trades)
    else:
        save_outputs(
            trades,
            universe,
            top50_summary,
            retention_sweep,
            top50_trades,
            output_dir=output_dir,
        )
    print_summary(top50_summary)

    target_dir = _resolve_dir(output_dir, OUTPUT_DIR)
    print(f"\nĐã lưu output tại: {target_dir}")
    print(f"  - {TOP50_SUMMARY_OUT.name}")
    print(f"  - {RETENTION_SWEEP_OUT.name}")
    print(f"  - {SCORED_UNIVERSE_OUT.name}")
    print("  - 4 tradelist top 50%")


if __name__ == "__main__":
    main()
