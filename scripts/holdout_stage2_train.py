"""Train and reproducibly validate the final CatBoost model for the holdout stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import catboost
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citd_ml import paths
from citd_ml.features.build_features import FEATURES


HERE = paths.HOLDOUT_STAGE2_DIR
TRAIN_DATASET = paths.DATASET_CSV
HOLDOUT_DATASET = paths.HOLDOUT_STAGE1_DIR / "dataset_catboost_holdout.csv"
HOLDOUT_START = pd.Timestamp(paths.HOLDOUT_START)
EMBARGO_START_BAR = paths.EMBARGO_START_BAR
PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "auto_class_weights": "Balanced",
    "eval_metric": "AUC",
    "random_seed": 42,
    # Pinned to 1 to remove CPU thread-count as a known source of numerical
    # variation. Exact equality is verified per environment; cross-machine
    # equivalence needs its own comparison artifact.
    "thread_count": 1,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def precheck(features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = pd.read_csv(TRAIN_DATASET, parse_dates=["entry_time", "label_end_time"])
    holdout_hash_before = sha256(HOLDOUT_DATASET)
    holdout = pd.read_csv(HOLDOUT_DATASET, parse_dates=["entry_time", "label_end_time"])
    purge_mask = data["label_end_time"] >= HOLDOUT_START
    embargo_mask = data["entry_bar"] >= EMBARGO_START_BAR
    purge_rows = data.loc[purge_mask, ["entry_time", "label_end_time", "origin_bar", "entry_bar", "leg"]]
    embargo_rows = data.loc[embargo_mask, ["entry_time", "label_end_time", "origin_bar", "entry_bar", "leg"]]
    filtered = data.loc[~purge_mask & ~embargo_mask].copy()
    details = {
        "holdout_start": str(HOLDOUT_START),
        "embargo_start_bar": EMBARGO_START_BAR,
        "train_rows_before_filtering": len(data),
        "purge_rows": len(purge_rows),
        "embargo_rows": len(embargo_rows),
        "rows_removed_by_either_condition": int((purge_mask | embargo_mask).sum()),
        "train_rows_after_filtering": len(filtered),
        "max_train_entry_bar": int(data["entry_bar"].max()),
        "max_train_label_end_time": str(data["label_end_time"].max()),
        "purge_row_details": purge_rows.to_dict(orient="records"),
        "embargo_row_details": embargo_rows.to_dict(orient="records"),
        "holdout_rows_unchanged": len(holdout),
        "holdout_sha256_before": holdout_hash_before,
        "features": features,
    }
    return filtered, holdout, details


def run_training(run_id: int) -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    features = list(FEATURES)
    filtered, holdout, details = precheck(features)
    if details["purge_rows"] != 0 or details["embargo_rows"] != 0:
        (HERE / f"precheck_run{run_id}.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(details, ensure_ascii=False, indent=2))
        print("STOP: purge or embargo removed rows; no model trained.")
        return 1

    X = filtered[features]
    y = filtered["label"]
    if X.shape[1] != 23 or y.nunique() != 2:
        raise ValueError("Invalid training matrix")

    model = CatBoostClassifier(**PARAMS)
    model.fit(X, y, verbose=False)
    model_path = HERE / f"catboost_final_holdout_run{run_id}.cbm"
    prediction_path = HERE / f"train_predictions_run{run_id}.npy"
    model.save_model(model_path)
    np.save(prediction_path, model.predict_proba(X)[:, 1])

    details.update({
        "run_id": run_id,
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256(prediction_path),
        "holdout_sha256_after": sha256(HOLDOUT_DATASET),
        "holdout_file_unchanged": details["holdout_sha256_before"] == sha256(HOLDOUT_DATASET),
        "versions": {
            "python": sys.version.split()[0],
            "catboost": catboost.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            # Bổ sung theo góp ý review: ghi rõ hệ điều hành và phiên bản plotly
            # (plotly là thư viện dựng biểu đồ của Stage 4, trước đây chưa ghi).
            "platform": platform.platform(),
            "plotly": version("plotly"),
        },
        "params": PARAMS,
    })
    (HERE / f"train_run{run_id}_report.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(details, ensure_ascii=False, indent=2))
    return 0


def verify() -> int:
    model1, model2 = HERE / "catboost_final_holdout_run1.cbm", HERE / "catboost_final_holdout_run2.cbm"
    pred1, pred2 = HERE / "train_predictions_run1.npy", HERE / "train_predictions_run2.npy"
    if not all(path.exists() for path in (model1, model2, pred1, pred2)):
        raise FileNotFoundError("Both independent training runs must exist before verification")
    first = np.load(pred1, allow_pickle=False)
    second = np.load(pred2, allow_pickle=False)
    report = {
        "verification_scope": "two independent training runs on the current machine",
        "model_hashes_equal": sha256(model1) == sha256(model2),
        "model_run1_sha256": sha256(model1),
        "model_run2_sha256": sha256(model2),
        "prediction_hashes_equal": sha256(pred1) == sha256(pred2),
        "predictions_exactly_equal": bool(np.array_equal(first, second)),
        "prediction_count": len(first),
        "max_prediction_abs_difference": float(np.max(np.abs(first - second))),
        "holdout_file_unchanged": sha256(HOLDOUT_DATASET) == json.loads((HERE / "train_run1_report.json").read_text(encoding="utf-8"))["holdout_sha256_before"],
        "acceptance_rule": "prediction arrays must be byte-identical and holdout input must remain unchanged; CBM hashes are recorded but not required because serialized model metadata can differ",
        "cross_machine_claim": "not established by this command",
    }
    (HERE / "stage2_reproducibility_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all((report["predictions_exactly_equal"], report["prediction_hashes_equal"], report["holdout_file_unchanged"])) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, choices=[1, 2])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify == (args.run_id is not None):
        parser.error("Choose exactly one of --run-id or --verify")
    return verify() if args.verify else run_training(args.run_id)


if __name__ == "__main__":
    sys.exit(main())
