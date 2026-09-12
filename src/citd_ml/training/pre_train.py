from pathlib import Path

import numpy as np
import pandas as pd

from citd_ml import paths
from citd_ml.features.build_features import FEATURES, META

DATA_PATH = paths.DATASET_CSV
EXPECTED_ROWS = 25_008
EXPECTED_EDGES = [0, 5_001, 10_003, 15_004, 20_006, 25_008]
HOLDOUT_START = pd.Timestamp(paths.HOLDOUT_START)


def _validate_schema(df: pd.DataFrame) -> None:
    """Check dataset"""
    expected_columns = FEATURES + ["label"] + META
    missing_columns = [column for column in expected_columns if column not in df]
    unexpected_columns = [column for column in df if column not in expected_columns]

    if missing_columns or unexpected_columns:
        raise ValueError(
            "Schema dataset không hợp lệ. "
            f"Cột bị thiếu: {missing_columns}; "
            f"cột ngoài dự kiến: {unexpected_columns}"
        )


def _validate_prepared_data(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    edges: list[int],
) -> None:
    if len(FEATURES) != 23:
        raise ValueError(f"FEATURES phải có 23 cột, hiện có {len(FEATURES)}")
    if len(META) != 7:
        raise ValueError(f"META phải có 7 cột, hiện có {len(META)}")
    if set(FEATURES) & set(META):
        raise ValueError("FEATURES và META không được chứa cột trùng nhau")
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Dataset phải có {EXPECTED_ROWS:,} dòng, hiện có {len(df):,}")
    if X.shape != (EXPECTED_ROWS, 23):
        raise ValueError(f"X phải có shape ({EXPECTED_ROWS}, 23), hiện có {X.shape}")
    if y.shape != (EXPECTED_ROWS,):
        raise ValueError(f"y phải có shape ({EXPECTED_ROWS},), hiện có {y.shape}")
    if meta.shape != (EXPECTED_ROWS, 8):
        raise ValueError(
            f"Metadata phải có shape ({EXPECTED_ROWS}, 8), hiện có {meta.shape}"
        )
    if set(y.unique()) != {0, 1}:
        raise ValueError(f"label chỉ được chứa 0 và 1, hiện có {sorted(y.unique())}")
    if X.isna().any().any():
        raise ValueError("X có giá trị NaN")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise TypeError("Toàn bộ 23 feature phải có kiểu numeric")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("X có giá trị vô cực")
    if df[["entry_time", "label_end_time"]].isna().any().any():
        raise ValueError(
            "entry_time hoặc label_end_time có giá trị thời gian không hợp lệ"
        )
    if not df["entry_time"].is_monotonic_increasing:
        raise ValueError("entry_time chưa được sắp xếp tăng dần")
    if (df["label_end_time"] < df["entry_time"]).any():
        raise ValueError("Có label_end_time nằm trước entry_time")
    if (df[["entry_time", "label_end_time"]] >= HOLDOUT_START).any().any():
        raise ValueError(
            f"Dataset chứa entry_time/label_end_time từ {HOLDOUT_START} trở đi; "
            "không được dùng holdout và không tự xóa dòng để đổi ranh giới khúc"
        )
    if not np.array_equal(df["row_id"].to_numpy(), np.arange(EXPECTED_ROWS)):
        raise ValueError("row_id phải liên tục từ 0 đến 25007")
    if edges != EXPECTED_EDGES:
        raise ValueError(f"Ranh giới khúc không đúng: {edges}")

    family_sizes = df.groupby("origin_bar", sort=False).size()
    if not family_sizes.eq(4).all():
        raise ValueError("Mỗi origin_bar phải có đúng 4 leg")

    family_legs = df.groupby("origin_bar", sort=False)["leg"].apply(
        lambda values: set(values) == {0, 1, 2, 3}
    )
    if not family_legs.all():
        raise ValueError("Mỗi origin_bar phải chứa đủ leg 0, 1, 2 và 3")


def prepare_dataset(
    path: str | Path = DATA_PATH,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    list[int],
    list[np.ndarray],
]:
    dataset_path = Path(path)
    if not dataset_path.is_absolute():
        dataset_path = paths.PROJECT_ROOT / dataset_path
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)
    _validate_schema(df)

    for column in ["entry_time", "label_end_time"]:
        df[column] = pd.to_datetime(df[column], errors="raise")

    if df["label"].isna().any() or set(df["label"].unique()) != {0, 1}:
        raise ValueError("label chỉ được chứa 0 và 1 trước khi chuyển kiểu dữ liệu")

    df = df.sort_values("entry_time", kind="stable").reset_index(drop=True)
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))

    X = df.loc[:, FEATURES].copy()
    y = df["label"].astype("int8").copy()
    meta = df.loc[:, ["row_id"] + META].copy()

    n = len(df)
    edges = [n * k // 5 for k in range(6)]
    chunks = [np.arange(edges[k], edges[k + 1]) for k in range(5)]

    _validate_prepared_data(df, X, y, meta, edges)
    return df, X, y, meta, edges, chunks


def print_report(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    edges: list[int],
    chunks: list[np.ndarray],
) -> None:
    """Print report before create folds"""
    print("Kích thước ban đầu:", (len(df), len(FEATURES) + 1 + len(META)))
    print("Số feature:", len(FEATURES))
    print("Số metadata:", len(META))
    print("Cột bị thiếu: []")
    print("Cột ngoài dự kiến: []")
    print(df[["row_id", "entry_time", "origin_bar", "leg"]].head())
    print("\nThời gian tăng dần:", df["entry_time"].is_monotonic_increasing)
    print("\nX:", X.shape)
    print("y:", y.shape)
    print("Metadata:", meta.shape)
    print("\nKiểm tra dataset: OK")
    print("Phân bố nhãn:")
    print(y.value_counts().sort_index())
    print("Tỷ lệ label=1:", y.mean())
    print("\nEdges:", edges)

    for number, indices in enumerate(chunks, start=1):
        print(
            f"Khúc {number}: index {indices[0]}–{indices[-1]}, " f"{len(indices)} dòng"
        )


def main() -> None:
    prepared = prepare_dataset()
    print_report(*prepared)


if __name__ == "__main__":
    main()
