from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score

from citd_ml import paths
from citd_ml.training.pre_train import prepare_dataset
from citd_ml.training.split_data import make_all_folds, validate_all_folds

OUTPUT_DIR = paths.CATBOOST_TRAINING_DIR

MODEL_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "auto_class_weights": "Balanced",
    "eval_metric": "AUC",
    "random_seed": 42,
    # Pinned to 1 to remove CPU thread-count as a known source of numerical
    # variation. This supports repeatability but does not, by itself, prove
    # byte-identical output across operating systems or CPU architectures.
    "thread_count": 1,
}

METHOD_ORDER = [
    "random_kfold",
    "grouped_kfold",
    "walk_forward",
    "purged_walk_forward",
]


def make_model() -> CatBoostClassifier:
    return CatBoostClassifier(**MODEL_PARAMS)


def _validate_probability(probability: np.ndarray, test_size: int) -> None:
    if probability.shape != (test_size,):
        raise ValueError(
            f"Probability phải có shape ({test_size},), hiện có {probability.shape}"
        )
    if not np.isfinite(probability).all():
        raise ValueError("Probability có NaN hoặc giá trị vô cực")
    if not ((probability >= 0.0) & (probability <= 1.0)).all():
        raise ValueError("Probability phải nằm trong đoạn [0, 1]")


def _validate_oof_coverage(
    method: str,
    probability: np.ndarray,
    fold_id: np.ndarray,
    first_walk_test_row: int,
) -> None:
    expected = np.ones(len(probability), dtype=bool)
    if method in {"walk_forward", "purged_walk_forward"}:
        expected[:first_walk_test_row] = False

    if not np.isfinite(probability[expected]).all():
        raise ValueError(f"{method}: còn thiếu xác suất OOF trong vùng test")
    if np.isfinite(probability[~expected]).any():
        raise ValueError(f"{method}: có xác suất tại vùng không được dự đoán")
    if np.any(fold_id[expected] < 1):
        raise ValueError(f"{method}: còn thiếu fold_id trong vùng test")
    if np.any(fold_id[~expected] != -1):
        raise ValueError(f"{method}: fold_id xuất hiện ngoài vùng test")


def train_method(
    method: str,
    folds,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    first_walk_test_row: int,
) -> tuple[list[dict], pd.DataFrame]:
    oof_probability = np.full(len(X), np.nan, dtype=np.float64)
    fold_id = np.full(len(X), -1, dtype=np.int16)
    metrics = []

    print(f"\nTraining: {method}")
    for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
        if np.isfinite(oof_probability[test_idx]).any():
            raise ValueError(f"{method} fold {fold_number}: test bị dự đoán lặp")

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        if y_train.nunique() != 2 or y_test.nunique() != 2:
            raise ValueError(f"{method} fold {fold_number}: train/test thiếu một lớp")

        model = make_model()
        model.fit(X_train, y_train, verbose=False)
        probability = model.predict_proba(X_test)[:, 1]
        _validate_probability(probability, len(test_idx))

        predicted_label = (probability >= 0.5).astype(np.int8)
        auc = roc_auc_score(y_test, probability)
        f1 = f1_score(y_test, predicted_label)

        oof_probability[test_idx] = probability
        fold_id[test_idx] = fold_number
        metrics.append(
            {
                "method": method,
                "fold": fold_number,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_positive_rate": y_train.mean(),
                "test_positive_rate": y_test.mean(),
                "roc_auc": auc,
                "f1": f1,
            }
        )
        print(
            f"  Fold {fold_number}: train={len(train_idx):,}, "
            f"test={len(test_idx):,}, AUC={auc:.6f}, F1={f1:.6f}"
        )

    _validate_oof_coverage(method, oof_probability, fold_id, first_walk_test_row)

    scores = meta.copy()
    scores.insert(0, "method", method)
    scores["label"] = y.to_numpy()
    scores["fold"] = fold_id
    scores["probability"] = oof_probability
    return metrics, scores


def build_summary(metrics_by_fold: pd.DataFrame) -> pd.DataFrame:
    summary = metrics_by_fold.groupby("method", sort=False, as_index=False).agg(
        folds=("fold", "count"),
        mean_roc_auc=("roc_auc", "mean"),
        mean_f1=("f1", "mean"),
    )
    if summary.set_index("method")["folds"].to_dict() != {
        "random_kfold": 5,
        "grouped_kfold": 5,
        "walk_forward": 4,
        "purged_walk_forward": 4,
    }:
        raise ValueError("Số fold trong bảng tổng hợp không đúng")
    return summary


def save_outputs(
    metrics_by_fold: pd.DataFrame,
    metrics_summary: pd.DataFrame,
    scores_by_method: dict[str, pd.DataFrame],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_by_fold.to_csv(
        OUTPUT_DIR / "metrics_by_fold.csv", index=False, float_format="%.12g"
    )
    metrics_summary.to_csv(
        OUTPUT_DIR / "metrics_summary.csv", index=False, float_format="%.12g"
    )

    for method in METHOD_ORDER:
        scores_by_method[method].to_csv(
            OUTPUT_DIR / f"oof_{method}.csv",
            index=False,
            float_format="%.12g",
            date_format="%Y-%m-%d %H:%M:%S",
        )


def main() -> None:
    df, X, y, meta, edges, chunks = prepare_dataset()
    all_folds = make_all_folds(X, y, meta, chunks)
    validate_all_folds(all_folds, meta, chunks, len(df))

    all_metrics = []
    scores_by_method = {}
    first_walk_test_row = edges[1]

    for method in METHOD_ORDER:
        metrics, scores = train_method(
            method,
            all_folds[method],
            X,
            y,
            meta,
            first_walk_test_row,
        )
        all_metrics.extend(metrics)
        scores_by_method[method] = scores

    metrics_by_fold = pd.DataFrame(all_metrics)
    if len(metrics_by_fold) != 18:
        raise ValueError(f"Phải có 18 kết quả fold, hiện có {len(metrics_by_fold)}")
    valid_metrics = metrics_by_fold[["roc_auc", "f1"]].apply(
        lambda column: column.between(0.0, 1.0).all()
    )
    if not valid_metrics.all():
        raise ValueError("ROC-AUC hoặc F1 nằm ngoài đoạn [0, 1]")

    metrics_summary = build_summary(metrics_by_fold)
    save_outputs(metrics_by_fold, metrics_summary, scores_by_method)

    print("\nKết quả trung bình:")
    print(
        metrics_summary.to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        )
    )
    print(f"\nĐã lưu output tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
