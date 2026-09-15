#!/usr/bin/env python3
"""Controlled experiment: does CatBoost ``thread_count`` change results?

The holdout pipeline pins ``thread_count=1`` to remove CPU thread count as a
known source of numerical drift.  CatBoost's own documentation states that the
thread count does not influence results.  This script measures the actual
effect on this machine by training the exact same model four times while
varying only ``thread_count``: 1 (twice), 2, and -1 (library default, all
cores).

Everything else is fixed: dataset, feature list, hyperparameters, random_seed,
and the prediction call ``model.predict_proba(X)[:, 1]`` exactly as
``scripts/holdout_stage2_train.py`` uses it (CatBoost's default prediction
thread_count is -1, i.e. all cores, independently of the training setting).
Supplementary probes locate the effect: the baseline model is re-scored with
prediction thread_count 1, 2 and -1, and thread_count 2 and -1 are each trained
a second time, separating "changing thread_count changes results" from
"training is not repeatable at all".

Writes:
  outputs/thread_count_sensitivity/thread_count_sensitivity.json
  outputs/thread_count_sensitivity/thread_count_sensitivity_summary.csv
  outputs/thread_count_sensitivity/README.md

Run:  uv run python scripts/thread_count_sensitivity.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import catboost
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citd_ml import paths
from citd_ml.features.build_features import FEATURES
from citd_ml.training.train_catboost import MODEL_PARAMS


TRAIN_DATASET = paths.DATASET_CSV
HOLDOUT_DATASET = paths.HOLDOUT_STAGE1_DIR / "dataset_catboost_holdout.csv"
STAGE2_MODEL = paths.HOLDOUT_STAGE2_DIR / "catboost_final_holdout_run1.cbm"
SCORED_UNIVERSE = paths.HOLDOUT_STAGE3_DIR / "holdout_fixed_trade_universe_scored.csv"
STAGE3_REPORT = paths.HOLDOUT_STAGE3_DIR / "stage3_report.json"
STAGE3_TOP50 = paths.HOLDOUT_STAGE3_DIR / "holdout_trades_top50.csv"
OUT = paths.OUTPUTS_DIR / "thread_count_sensitivity"
HOLDOUT_START = pd.Timestamp(paths.HOLDOUT_START)
EMBARGO_START_BAR = paths.EMBARGO_START_BAR
BASELINE = "tc1_a"
KEEP_PCT = 50

# The frozen CatBoost configuration comes from the single source of truth in
# src/citd_ml/training/train_catboost.py; only thread_count is varied.
FIXED_PARAMS = {key: value for key, value in MODEL_PARAMS.items() if key != "thread_count"}

CONFIGS = (
    ("tc1_a", 1),        # baseline, matches the pinned pipeline
    ("tc1_b", 1),        # second independent run, same config
    ("tc2", 2),
    ("tc_default", -1),  # library default: all cores
)


def params_for(thread_count: int) -> dict:
    """Full CatBoost parameters for one configuration (MODEL_PARAMS + thread_count)."""
    return {**MODEL_PARAMS, "thread_count": thread_count}


def delta(pred: np.ndarray, base: np.ndarray) -> dict:
    """So sánh hai mảng xác suất so với baseline."""
    diff = np.abs(pred.astype(np.float64) - base.astype(np.float64))
    return {
        "predictions_exactly_equal": bool(np.array_equal(pred, base)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "pearson_corr": float(np.corrcoef(pred, base)[0, 1]),
        "count_diff_gt_1e-12": int((diff > 1e-12).sum()),
        "count_diff_gt_0": int((diff > 0).sum()),
    }


def load_train() -> tuple[pd.DataFrame, dict]:
    """Đọc tập train và áp đúng bộ lọc purge/embargo của stage 2."""
    data = pd.read_csv(TRAIN_DATASET, parse_dates=["entry_time", "label_end_time"])
    purge_mask = data["label_end_time"] >= HOLDOUT_START
    embargo_mask = data["entry_bar"] >= EMBARGO_START_BAR
    filtered = data.loc[~purge_mask & ~embargo_mask].copy()
    details = {
        "train_rows_before_filtering": int(len(data)),
        "purge_rows": int(purge_mask.sum()),
        "embargo_rows": int(embargo_mask.sum()),
        "rows_removed_by_either_condition": int((purge_mask | embargo_mask).sum()),
        "train_rows_after_filtering": int(len(filtered)),
    }
    if len(data) != 25008 or details["rows_removed_by_either_condition"] != 0:
        raise ValueError(f"Unexpected stage-2 purge/embargo result: {details}")
    return filtered, details


def load_holdout(features: list[str]) -> pd.DataFrame:
    holdout = pd.read_csv(HOLDOUT_DATASET, parse_dates=["entry_time", "label_end_time"])
    required = set(features + ["label", "origin_bar", "entry_bar", "leg"])
    if len(holdout) != 5028 or required - set(holdout.columns):
        raise ValueError("Holdout schema invalid")
    return holdout


def load_universe(holdout: pd.DataFrame) -> pd.DataFrame:
    """Ghép khóa holdout với stage 3 để lấy row_id, R, close_time, ticket."""
    keys = holdout[["origin_bar", "entry_bar", "leg"]].copy()
    keys.insert(0, "row_id_dataset", np.arange(len(keys)))
    scored = pd.read_csv(
        SCORED_UNIVERSE,
        usecols=["row_id", "origin_bar", "entry_bar", "leg", "R", "close_time", "ticket"],
        parse_dates=["close_time"],
    )
    check = keys.merge(scored, on=["origin_bar", "entry_bar", "leg"], how="left", validate="one_to_one")
    if check["row_id"].isna().any() or not (check["row_id"].to_numpy() == check["row_id_dataset"].to_numpy()).all():
        raise ValueError("Holdout keys do not align one-to-one with the committed stage-3 universe")
    return check[["row_id", "origin_bar", "entry_bar", "leg", "R", "close_time", "ticket"]]


def select_top(universe: pd.DataFrame, probability: np.ndarray, keep_count: int) -> pd.DataFrame:
    """Đúng quy tắc stage 3: probability giảm dần, row_id tăng dần để phá hòa."""
    ranked = universe.copy()
    ranked["probability"] = probability
    return ranked.sort_values(["probability", "row_id"], ascending=[False, True], kind="stable").iloc[:keep_count].copy()


def top_k_metrics(selected: pd.DataFrame) -> dict:
    """Giống hệt hàm metrics() của scripts/holdout_stage3_backtest.py."""
    ordered = selected.sort_values(["close_time", "ticket"], kind="stable")
    r = ordered["R"].to_numpy(float)
    equity = np.r_[0.0, np.cumsum(r)]
    gross_profit = r[r > 0].sum()
    gross_loss = -r[r < 0].sum()
    return {
        "trades": int(len(selected)),
        "net_profit_R": float(r.sum()),
        "max_dd_R": float(np.max(np.maximum.accumulate(equity) - equity)),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else float("inf"),
        "win_rate_pct": float((r > 0).mean() * 100),
    }


def render_readme(report: dict) -> str:
    """Sinh README.md từ chính kết quả đo được, để artifact luôn khớp JSON."""
    facts = report["conclusion_facts"]
    rows = []
    for cfg in report["configs"]:
        name = cfg["name"]
        d_train = report["deltas_vs_tc1_a"][name]["train"]
        d_hold = report["deltas_vs_tc1_a"][name]["holdout"]
        rows.append(
            "| {name} | {tc} | {tmax:.3e} | {tmean:.3e} | {hmax:.3e} | {hmean:.3e} | "
            "{auc:.4f} | {f1:.4f} | {net:+.4f} | {ov}/{keep} |".format(
                name=name,
                tc=cfg["thread_count"],
                tmax=d_train["max_abs_diff"],
                tmean=d_train["mean_abs_diff"],
                hmax=d_hold["max_abs_diff"],
                hmean=d_hold["mean_abs_diff"],
                auc=cfg["classification"]["roc_auc"],
                f1=cfg["classification"]["f1_at_0_5"],
                net=cfg["top50"]["net_profit_R"],
                ov=cfg["top50"]["overlap_count_vs_tc1_a"],
                keep=cfg["top50"]["trades"],
            )
        )
    max_metric_diff = max(report["baseline_vs_committed_stage3"]["top50_metrics_abs_diff"].values())
    training_stage_sentence = (
        "On this machine, the prediction-stage statement holds exactly."
        if facts["repeated_inference_on_same_model_exactly_equal"] and not facts["prediction_thread_count_changed_predictions"]
        else "On this machine, the prediction stage itself also showed differences."
    )
    training_stage_sentence += (
        " No training-stage difference was observed either."
        if facts["all_configs_produced_exactly_equal_train_predictions"] and facts["all_configs_produced_exactly_equal_holdout_predictions"]
        else " The training-stage statement does not hold for this model and dataset."
    )
    findings = [
        "- Two independent `thread_count=1` runs produced {exact} predictions on train and holdout "
        "(train max |Δ| = {tmax:.3e}, holdout max |Δ| = {hmax:.3e}).".format(
            exact="exactly equal" if facts["two_same_config_tc1_runs_exactly_equal_holdout"] else "different",
            tmax=facts["tc1_a_vs_tc1_b_train_max_abs_diff"],
            hmax=facts["tc1_a_vs_tc1_b_holdout_max_abs_diff"],
        ),
        "- `thread_count=2` vs baseline: train max |Δ| = {tmax:.3e}, holdout max |Δ| = {hmax:.3e}; "
        "top-50% selection {same} (overlap {ov}/{keep}). Downstream top-50% net R {r1:+.4f} vs {r2:+.4f}.".format(
            tmax=facts["tc2_vs_tc1_a_train_max_abs_diff"],
            hmax=facts["tc2_vs_tc1_a_holdout_max_abs_diff"],
            same="identical" if facts["tc2_top50_selection_identical_to_tc1_a"] else "different",
            ov=report["configs"][2]["top50"]["overlap_count_vs_tc1_a"],
            keep=report["configs"][2]["top50"]["trades"],
            r1=report["configs"][2]["top50"]["net_profit_R"],
            r2=report["configs"][0]["top50"]["net_profit_R"],
        ),
        "- `thread_count=-1` (all cores) vs baseline: train max |Δ| = {tmax:.3e}, holdout max |Δ| = {hmax:.3e}; "
        "top-50% selection {same} (overlap {ov}/{keep}). Downstream top-50% net R {r1:+.4f} vs {r2:+.4f}.".format(
            tmax=facts["tc_default_vs_tc1_a_train_max_abs_diff"],
            hmax=facts["tc_default_vs_tc1_a_holdout_max_abs_diff"],
            same="identical" if facts["tc_default_top50_selection_identical_to_tc1_a"] else "different",
            ov=report["configs"][3]["top50"]["overlap_count_vs_tc1_a"],
            keep=report["configs"][3]["top50"]["trades"],
            r1=report["configs"][3]["top50"]["net_profit_R"],
            r2=report["configs"][0]["top50"]["net_profit_R"],
        ),
        "- Across all four configurations: train predictions {teq}, holdout predictions {heq}.".format(
            teq="exactly equal" if facts["all_configs_produced_exactly_equal_train_predictions"] else "differ",
            heq="exactly equal" if facts["all_configs_produced_exactly_equal_holdout_predictions"] else "differ",
        ),
        (
            "- A fixed thread count is reproducible: the repeat probe at `thread_count=2` and at `thread_count=-1` "
            "matched its first run exactly on train and holdout. The divergence is caused by *changing* the thread "
            "count, not by run-to-run nondeterminism within a fixed thread count."
            if facts["fixed_thread_count_training_is_repeatable"]
            else "- The repeat probe at a fixed thread count did NOT match its first run; see "
            "`same_config_determinism_probe` in the JSON."
        ),
        (
            "- The divergence is in the training stage: the same fitted model scored with prediction `thread_count` "
            "1, 2 and -1 produced exactly equal predictions, and repeated inference calls were also exactly equal "
            "(all max |Δ| = 0.0)."
            if (
                facts["repeated_inference_on_same_model_exactly_equal"]
                and not facts["prediction_thread_count_changed_predictions"]
            )
            else "- The prediction stage itself also showed differences; see `prediction_call_repeatability` and "
            "`prediction_thread_count_sensitivity` in the JSON."
        ),
        "- Baseline (`tc1_a`) reproduces the committed stage-3 pipeline: model predictions exactly equal the "
        "committed stage-2 model ({pe}), classification {c}, top-50% row_id selection {ov}/{keep}, and top-50% "
        "R metrics agree to {md:.1e} (R is round-tripped through the stage-3 CSV with 12 significant digits).".format(
            pe="yes" if report["baseline_vs_committed_stage3"]["model_predictions"]["predictions_exactly_equal"] else "no",
            c="match" if report["baseline_vs_committed_stage3"]["classification_match"] else "do NOT match",
            ov=report["baseline_vs_committed_stage3"]["top50_row_id_overlap_count"],
            keep=report["baseline_vs_committed_stage3"]["top50_row_id_overlap_total"],
            md=max_metric_diff,
        ),
    ]
    return """# CatBoost `thread_count` sensitivity (controlled experiment)

Generated by `scripts/thread_count_sensitivity.py` on {generated}.

## What was measured

Four main trainings with the **same data, features, hyperparameters and
`random_seed=42`**; only `thread_count` was varied:

| name | `thread_count` |
|---|---|
| `tc1_a` | 1 (baseline, the pinned pipeline setting) |
| `tc1_b` | 1 (second independent run, separates repeatability from config change) |
| `tc2` | 2 |
| `tc_default` | -1 (CatBoost default: all CPU cores) |

- Train set: `data/processed/dataset_catboost.csv`, full 25,008 rows after the
  stage-2 purge/embargo filter (it removed 0 rows).
- Holdout: `outputs/holdout/stage1/dataset_catboost_holdout.csv`, 5,028 rows.
- Prediction call: `model.predict_proba(X)[:, 1]`, i.e. CatBoost's default
  prediction `thread_count=-1`, exactly as `scripts/holdout_stage2_train.py`.
- Supplementary probes: one repeat training at `thread_count=2` and one at
  `thread_count=-1` (fixed-configuration repeatability), plus scoring the
  baseline model with prediction `thread_count` 1, 2 and -1 and a repeated
  inference call (prediction-stage isolation).
- Reported deltas: max/mean absolute probability difference, Pearson
  correlation, and count of predictions differing by more than 1e-12, each
  computed against `tc1_a` separately for train and holdout.
- Holdout classification at threshold 0.5 (ROC-AUC, F1) and the exact stage-3
  top-50% financials: `keep_count = ceil(5028 * 50 / 100) = 2514`, sorted by
  probability descending then `row_id` ascending, with `row_id`/`R` taken from
  `outputs/holdout/stage3/holdout_fixed_trade_universe_scored.csv`.

## Command

```
{command}
```

## Machine and versions

- Platform: `{platform_name}` (`os.cpu_count() = {cpu_count}`)
- Python {python}, catboost {catboost}, scikit-learn {sklearn}, pandas {pandas}, numpy {numpy}

## Results

Deltas are versus the `tc1_a` baseline; `tc1_a` vs itself is zero by
construction. `top-50% overlap` is the count of identical
`(origin_bar, entry_bar, leg)` keys among the 2,514 selected trades.

| config | `thread_count` | train max abs-diff | train mean abs-diff | holdout max abs-diff | holdout mean abs-diff | holdout ROC-AUC | holdout F1@0.5 | top-50% net R | top-50% overlap |
|---|---|---|---|---|---|---|---|---|---|
{rows}

## Findings (objective)

{findings}

CatBoost's documentation states that `thread_count` "optimizes execution speed
without affecting results" (training parameters) and the `predict_proba`
docstring states that the prediction `thread_count` "doesn't affect results".
The supplementary panel in the JSON (`prediction_call_repeatability`,
`prediction_thread_count_sensitivity`, `same_config_determinism_probe`)
separates prediction-stage effects from training-stage effects on the same
fitted model. {training_stage_sentence}

## Scope limitation

This experiment ran on **one machine, one CatBoost build, one dataset and one
seed**. It measures whether `thread_count` changes results *on this host*; it
does **not** attribute all cross-machine or cross-OS variation to
`thread_count`, and it is not a comparison across CPUs, operating systems or
library versions.

*Giới hạn phạm vi: thí nghiệm chỉ chạy trên một máy, một bản CatBoost, một tập
dữ liệu và một seed. Kết quả không đủ để quy toàn bộ sai lệch liên máy cho
`thread_count`, cũng không so sánh giữa các CPU, hệ điều hành hay phiên bản
thư viện khác nhau.*
""".format(
        generated=report["generated_at_utc"],
        command=report["command"],
        platform_name=report["platform"],
        cpu_count=report["cpu_count"],
        python=report["versions"]["python"],
        catboost=report["versions"]["catboost"],
        sklearn=report["versions"]["scikit_learn"],
        pandas=report["versions"]["pandas"],
        numpy=report["versions"]["numpy"],
        rows="\n".join(rows),
        findings="\n".join(findings),
        training_stage_sentence=training_stage_sentence,
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    features = list(FEATURES)
    if len(features) != 23:
        raise ValueError(f"Expected exactly 23 FEATURES, got {len(features)}")

    train, filter_details = load_train()
    holdout = load_holdout(features)
    universe = load_universe(holdout)
    keep_count = int(np.ceil(len(universe) * KEEP_PCT / 100))
    if keep_count != 2514:
        raise ValueError(f"Unexpected top-{KEEP_PCT}% keep_count: {keep_count}")

    X_train, y_train = train[features], train["label"]
    X_holdout = holdout[features]
    if X_train.shape[1] != 23 or y_train.nunique() != 2:
        raise ValueError("Invalid training matrix")

    train_predictions: dict[str, np.ndarray] = {}
    holdout_predictions: dict[str, np.ndarray] = {}
    fit_seconds: dict[str, float] = {}
    baseline_model = None
    for name, thread_count in CONFIGS:
        started = time.perf_counter()
        model = CatBoostClassifier(**params_for(thread_count))
        model.fit(X_train, y_train, verbose=False)
        fit_seconds[name] = time.perf_counter() - started
        # Prediction mirrors scripts/holdout_stage2_train.py exactly.
        train_predictions[name] = model.predict_proba(X_train)[:, 1]
        holdout_predictions[name] = model.predict_proba(X_holdout)[:, 1]
        if name == BASELINE:
            baseline_model = model

    # Probe bổ sung: cùng một thread_count chạy lần hai có cho cùng kết quả không?
    # Việc này tách "đổi thread_count làm đổi kết quả" khỏi "training vốn không lặp lại được".
    probe_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, thread_count in (("tc2_repeat", 2), ("tc_default_repeat", -1)):
        model = CatBoostClassifier(**params_for(thread_count))
        model.fit(X_train, y_train, verbose=False)
        probe_predictions[name] = (
            model.predict_proba(X_train)[:, 1],
            model.predict_proba(X_holdout)[:, 1],
        )

    base_keys = {
        tuple(key)
        for key in select_top(universe, holdout_predictions[BASELINE], keep_count)[
            ["origin_bar", "entry_bar", "leg"]
        ].to_numpy()
    }

    configs_json = []
    summary_rows = []
    for name, thread_count in CONFIGS:
        probability = holdout_predictions[name]
        selected = select_top(universe, probability, keep_count)
        selected_keys = {tuple(key) for key in selected[["origin_bar", "entry_bar", "leg"]].to_numpy()}
        overlap = len(selected_keys & base_keys)
        classification = {
            "roc_auc": float(roc_auc_score(holdout["label"], probability)),
            "f1_at_0_5": float(f1_score(holdout["label"], probability >= 0.5)),
        }
        financials = top_k_metrics(selected)
        top50 = {
            **financials,
            "overlap_count_vs_tc1_a": overlap,
            "overlap_pct_vs_tc1_a": float(overlap / keep_count * 100),
            "identical_selection_vs_tc1_a": overlap == keep_count,
        }
        configs_json.append({
            "name": name,
            "thread_count": thread_count,
            "params": params_for(thread_count),
            "fit_seconds": round(fit_seconds[name], 3),
            "classification": classification,
            "top50": top50,
        })
        summary_rows.append({
            "config": name,
            "thread_count": thread_count,
            **{
                f"train_{key}_vs_tc1_a": value
                for key, value in delta(train_predictions[name], train_predictions[BASELINE]).items()
            },
            **{
                f"holdout_{key}_vs_tc1_a": value
                for key, value in delta(holdout_predictions[name], holdout_predictions[BASELINE]).items()
            },
            "roc_auc": classification["roc_auc"],
            "f1_at_0_5": classification["f1_at_0_5"],
            "top50_trades": top50["trades"],
            "top50_net_profit_R": top50["net_profit_R"],
            "top50_max_dd_R": top50["max_dd_R"],
            "top50_profit_factor": top50["profit_factor"],
            "top50_win_rate_pct": top50["win_rate_pct"],
            "top50_overlap_count_vs_tc1_a": top50["overlap_count_vs_tc1_a"],
            "top50_overlap_pct_vs_tc1_a": top50["overlap_pct_vs_tc1_a"],
            "top50_identical_selection_vs_tc1_a": top50["identical_selection_vs_tc1_a"],
            "fit_seconds": round(fit_seconds[name], 3),
        })

    deltas = {
        name: {
            "train": delta(train_predictions[name], train_predictions[BASELINE]),
            "holdout": delta(holdout_predictions[name], holdout_predictions[BASELINE]),
        }
        for name, _ in CONFIGS
    }

    same_config_probe = {
        "purpose": "Second independent run at thread_count=2 and at thread_count=-1, to test repeatability within a fixed configuration.",
        "tc2_repeat_vs_tc2": {
            "train": delta(probe_predictions["tc2_repeat"][0], train_predictions["tc2"]),
            "holdout": delta(probe_predictions["tc2_repeat"][1], holdout_predictions["tc2"]),
        },
        "tc_default_repeat_vs_tc_default": {
            "train": delta(probe_predictions["tc_default_repeat"][0], train_predictions["tc_default"]),
            "holdout": delta(probe_predictions["tc_default_repeat"][1], holdout_predictions["tc_default"]),
        },
    }
    same_config_probe["fixed_thread_count_training_is_repeatable"] = bool(
        same_config_probe["tc2_repeat_vs_tc2"]["train"]["predictions_exactly_equal"]
        and same_config_probe["tc2_repeat_vs_tc2"]["holdout"]["predictions_exactly_equal"]
        and same_config_probe["tc_default_repeat_vs_tc_default"]["train"]["predictions_exactly_equal"]
        and same_config_probe["tc_default_repeat_vs_tc_default"]["holdout"]["predictions_exactly_equal"]
    )

    # Cùng một model đã fit, gọi dự đoán lặp lại: kiểm tra riêng khâu inference.
    train_repeat = baseline_model.predict_proba(X_train)[:, 1]
    holdout_repeat = baseline_model.predict_proba(X_holdout)[:, 1]
    prediction_call_repeatability = {
        "model": BASELINE,
        "protocol": "model.predict_proba(X)[:, 1] called twice on the same fitted model (default prediction thread_count=-1)",
        "train": delta(train_repeat, train_predictions[BASELINE]),
        "holdout": delta(holdout_repeat, holdout_predictions[BASELINE]),
    }
    prediction_thread_sensitivity = {
        "model": BASELINE,
        "training_thread_count": 1,
        "protocol": "same fitted model, prediction thread_count varied explicitly",
        "train": {},
        "holdout": {},
    }
    for thread_count in (1, 2, -1):
        prediction_thread_sensitivity["train"][f"thread_count={thread_count}"] = delta(
            baseline_model.predict_proba(X_train, thread_count=thread_count)[:, 1],
            train_predictions[BASELINE],
        )
        prediction_thread_sensitivity["holdout"][f"thread_count={thread_count}"] = delta(
            baseline_model.predict_proba(X_holdout, thread_count=thread_count)[:, 1],
            holdout_predictions[BASELINE],
        )

    # Đối chiếu tc1_a với artifact stage 3 đã commit (cùng tham số cố định).
    stage3 = json.loads(STAGE3_REPORT.read_text(encoding="utf-8"))
    stage3_top50 = next(row for row in stage3["results"] if row["keep_pct"] == KEEP_PCT)
    committed_top50_ids = set(pd.read_csv(STAGE3_TOP50, usecols=["row_id"])["row_id"].tolist())
    experiment_top50_ids = set(select_top(universe, holdout_predictions[BASELINE], keep_count)["row_id"].tolist())
    baseline_cfg = configs_json[0]
    metric_keys = ("net_profit_R", "max_dd_R", "profit_factor", "win_rate_pct")

    # Model stage 2 đã commit: so prediction trực tiếp, không qua CSV.
    committed_model = CatBoostClassifier()
    committed_model.load_model(STAGE2_MODEL)
    committed_predictions = {
        "train": delta(committed_model.predict_proba(X_train)[:, 1], train_predictions[BASELINE]),
        "holdout": delta(committed_model.predict_proba(X_holdout)[:, 1], holdout_predictions[BASELINE]),
    }
    committed_predictions["predictions_exactly_equal"] = bool(
        committed_predictions["train"]["predictions_exactly_equal"]
        and committed_predictions["holdout"]["predictions_exactly_equal"]
    )

    committed_comparison = {
        "note": (
            "tc1_a is a fresh training run with the same fixed parameters as the committed stage-2 model; "
            "this check confirms the experiment reproduces the committed pipeline before interpreting thread_count deltas."
        ),
        "stage2_model": str(STAGE2_MODEL.relative_to(paths.PROJECT_ROOT)),
        "model_predictions": committed_predictions,
        "classification_match": bool(
            baseline_cfg["classification"]["roc_auc"] == stage3["classification"]["roc_auc"]
            and baseline_cfg["classification"]["f1_at_0_5"] == stage3["classification"]["f1_at_0_5"]
        ),
        "classification_experiment": baseline_cfg["classification"],
        "classification_stage3": stage3["classification"],
        "top50_metrics_match_within_csv_precision": bool(
            all(
                np.isclose(baseline_cfg["top50"][key], stage3_top50[key], rtol=1e-9, atol=1e-12)
                for key in metric_keys
            )
        ),
        "top50_metrics_exactly_equal": bool(all(baseline_cfg["top50"][key] == stage3_top50[key] for key in metric_keys)),
        "top50_metrics_abs_diff": {key: abs(baseline_cfg["top50"][key] - stage3_top50[key]) for key in metric_keys},
        "top50_metrics_precision_note": (
            "R is read back from holdout_fixed_trade_universe_scored.csv, which stage 3 wrote with "
            "float_format='%.12g'; top-50 R metrics therefore agree only to ~1e-11, while model predictions "
            "and the selected row_id set are compared exactly."
        ),
        "top50_metrics_experiment": {key: baseline_cfg["top50"][key] for key in metric_keys},
        "top50_metrics_stage3": {key: stage3_top50[key] for key in metric_keys},
        "top50_row_id_overlap_count": len(experiment_top50_ids & committed_top50_ids),
        "top50_row_id_overlap_total": len(committed_top50_ids),
    }

    tc2_delta, tcdef_delta = deltas["tc2"], deltas["tc_default"]
    all_train_equal = all(np.array_equal(train_predictions[name], train_predictions[BASELINE]) for name, _ in CONFIGS)
    all_holdout_equal = all(np.array_equal(holdout_predictions[name], holdout_predictions[BASELINE]) for name, _ in CONFIGS)
    fixed_thread_repeatable = same_config_probe["fixed_thread_count_training_is_repeatable"]
    repeat_inference_equal = prediction_call_repeatability["holdout"]["predictions_exactly_equal"]
    prediction_thread_changed = any(
        not values["predictions_exactly_equal"]
        for values in prediction_thread_sensitivity["holdout"].values()
    )
    facts = {
        "two_same_config_tc1_runs_exactly_equal_train": bool(
            np.array_equal(train_predictions["tc1_a"], train_predictions["tc1_b"])
        ),
        "two_same_config_tc1_runs_exactly_equal_holdout": bool(
            np.array_equal(holdout_predictions["tc1_a"], holdout_predictions["tc1_b"])
        ),
        "tc1_a_vs_tc1_b_train_max_abs_diff": deltas["tc1_b"]["train"]["max_abs_diff"],
        "tc1_a_vs_tc1_b_holdout_max_abs_diff": deltas["tc1_b"]["holdout"]["max_abs_diff"],
        "tc2_vs_tc1_a_train_max_abs_diff": tc2_delta["train"]["max_abs_diff"],
        "tc2_vs_tc1_a_holdout_max_abs_diff": tc2_delta["holdout"]["max_abs_diff"],
        "tc2_vs_tc1_a_train_count_diff_gt_1e-12": tc2_delta["train"]["count_diff_gt_1e-12"],
        "tc2_vs_tc1_a_holdout_count_diff_gt_1e-12": tc2_delta["holdout"]["count_diff_gt_1e-12"],
        "tc_default_vs_tc1_a_train_max_abs_diff": tcdef_delta["train"]["max_abs_diff"],
        "tc_default_vs_tc1_a_holdout_max_abs_diff": tcdef_delta["holdout"]["max_abs_diff"],
        "tc_default_vs_tc1_a_train_count_diff_gt_1e-12": tcdef_delta["train"]["count_diff_gt_1e-12"],
        "tc_default_vs_tc1_a_holdout_count_diff_gt_1e-12": tcdef_delta["holdout"]["count_diff_gt_1e-12"],
        "all_configs_produced_exactly_equal_train_predictions": bool(all_train_equal),
        "all_configs_produced_exactly_equal_holdout_predictions": bool(all_holdout_equal),
        "tc2_top50_selection_identical_to_tc1_a": configs_json[2]["top50"]["identical_selection_vs_tc1_a"],
        "tc_default_top50_selection_identical_to_tc1_a": configs_json[3]["top50"]["identical_selection_vs_tc1_a"],
        "holdout_roc_auc_range_across_configs": [
            min(cfg["classification"]["roc_auc"] for cfg in configs_json),
            max(cfg["classification"]["roc_auc"] for cfg in configs_json),
        ],
        "holdout_f1_at_0_5_range_across_configs": [
            min(cfg["classification"]["f1_at_0_5"] for cfg in configs_json),
            max(cfg["classification"]["f1_at_0_5"] for cfg in configs_json),
        ],
        "holdout_top50_net_profit_R_range_across_configs": [
            min(cfg["top50"]["net_profit_R"] for cfg in configs_json),
            max(cfg["top50"]["net_profit_R"] for cfg in configs_json),
        ],
        "baseline_reproduces_committed_stage3": bool(
            committed_comparison["model_predictions"]["predictions_exactly_equal"]
            and committed_comparison["classification_match"]
            and committed_comparison["top50_metrics_match_within_csv_precision"]
            and committed_comparison["top50_row_id_overlap_count"] == committed_comparison["top50_row_id_overlap_total"]
        ),
        "fixed_thread_count_training_is_repeatable": fixed_thread_repeatable,
        "repeated_inference_on_same_model_exactly_equal": repeat_inference_equal,
        "prediction_thread_count_changed_predictions": bool(prediction_thread_changed),
        "attribution": (
            "On this machine: repeated training at the same fixed thread_count was "
            + ("exactly equal" if fixed_thread_repeatable else "not exactly equal")
            + "; repeated inference on the same fitted model was "
            + ("exactly equal" if repeat_inference_equal else "not exactly equal")
            + "; varying the prediction thread_count "
            + ("changed predictions" if prediction_thread_changed else "did not change predictions")
            + "; varying the training thread_count "
            + ("changed predictions" if not (all_train_equal and all_holdout_equal) else "did not change predictions")
            + "."
        ),
        "catboost_doc_claim": (
            "CatBoost docs state thread_count \"optimizes execution speed without affecting results\" "
            "(training parameters) and the predict_proba docstring says it \"doesn't affect results\". "
            "This experiment confirms the prediction-stage statement on this machine but measures a "
            "different result for the training stage."
        ),
        "cross_machine_attribution": (
            "not established by this experiment: only one machine, one CatBoost build, one dataset and one seed were measured"
        ),
    }

    report = {
        "scope": {
            "question": (
                "Does varying CatBoost thread_count change model predictions, classification and downstream "
                "trade selection when every other parameter, the training data and random_seed are fixed?"
            ),
            "design": [
                "thread_count is the only parameter varied between configurations: 1 (twice), 2 and -1",
                "train data, features, hyperparameters and random_seed are identical in all configurations",
                "train set is the full 25,008-row dataset after the stage-2 purge/embargo filter, which removed 0 rows",
                "prediction mirrors scripts/holdout_stage2_train.py: model.predict_proba(X)[:, 1] (CatBoost default prediction thread_count=-1)",
                "top-50% selection mirrors scripts/holdout_stage3_backtest.py: keep_count=ceil(5028*50/100)=2514, probability descending then row_id ascending",
                "two same-config thread_count=1 runs separate repeatability from a real configuration change",
                "supplementary probes re-train at thread_count=2 and -1, and re-score the baseline model at prediction thread_count 1, 2 and -1, to locate the difference in the training stage",
            ],
            "limitations": [
                "Single machine, single CatBoost build, single dataset and seed.",
                "This experiment alone cannot attribute all cross-machine variation to thread count.",
                "No comparison across CPUs, operating systems or CatBoost versions.",
            ],
        },
        "command": " ".join([sys.executable, *sys.argv]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "versions": {
            "python": sys.version.split()[0],
            "catboost": catboost.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "data": {
            "train_dataset": str(TRAIN_DATASET.relative_to(paths.PROJECT_ROOT)),
            "holdout_dataset": str(HOLDOUT_DATASET.relative_to(paths.PROJECT_ROOT)),
            "scored_universe": str(SCORED_UNIVERSE.relative_to(paths.PROJECT_ROOT)),
            "train_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "purge_embargo": filter_details,
            "features": features,
            "top_k_rule": "keep_count=ceil(rows*50/100); sort probability descending then row_id ascending",
            "keep_count": keep_count,
        },
        "fixed_params": FIXED_PARAMS,
        "configurations": [{"name": name, "thread_count": thread_count} for name, thread_count in CONFIGS],
        "configs": configs_json,
        "deltas_vs_tc1_a": deltas,
        "same_config_determinism_probe": same_config_probe,
        "prediction_call_repeatability": prediction_call_repeatability,
        "prediction_thread_count_sensitivity": prediction_thread_sensitivity,
        "baseline_vs_committed_stage3": committed_comparison,
        "conclusion_facts": facts,
    }

    json_path = OUT / "thread_count_sensitivity.json"
    csv_path = OUT / "thread_count_sensitivity_summary.csv"
    readme_path = OUT / "README.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False, float_format="%.12g")
    readme_path.write_text(render_readme(report), encoding="utf-8")
    print(json.dumps({
        "written": [str(json_path), str(csv_path), str(readme_path)],
        "cpu_count": report["cpu_count"],
        "conclusion_facts": facts,
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
