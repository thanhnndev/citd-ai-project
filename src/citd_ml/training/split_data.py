"""Tạo và kiểm tra bốn cách chia train/test theo BAN_GIAO."""

import numpy as np
from sklearn.model_selection import GroupKFold, KFold

from citd_ml.training.pre_train import prepare_dataset


Fold = tuple[np.ndarray, np.ndarray]
FoldList = list[Fold]


def make_random_kfold(X) -> FoldList:
    """Cách 1: Random K-Fold trên toàn bộ từng dòng."""
    splitter = KFold(
        n_splits=5,
        shuffle=True,
        random_state=0,
    )
    return list(splitter.split(X))


def make_grouped_kfold(X, y, meta) -> FoldList:
    """Cách 1b: Grouped K-Fold, giữ nguyên từng gia đình origin_bar."""
    splitter = GroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=0,
    )

    return list(
        splitter.split(
            X,
            y,
            groups=meta["origin_bar"],
        )
    )


def make_walk_forward(chunks: list[np.ndarray]) -> FoldList:
    """Cách 2: expanding window, học các khúc trước và test khúc kế tiếp."""
    folds = []

    for test_chunk in range(1, 5):
        train_idx = np.concatenate(chunks[:test_chunk])
        test_idx = chunks[test_chunk].copy()

        folds.append((train_idx, test_idx))

    return folds


def make_purged_walk_forward(walk_folds: FoldList, meta) -> FoldList:
    """Cách 3: chỉ purge/embargo train và giữ nguyên toàn bộ test."""
    purged_folds = []

    for train_idx, test_idx in walk_folds:
        test_start_time = meta.iloc[test_idx[0]]["entry_time"]
        test_start_bar = meta.iloc[test_idx[0]]["entry_bar"]

        train_meta = meta.iloc[train_idx]

        purge_ok = train_meta["label_end_time"] < test_start_time

        embargo_ok = train_meta["entry_bar"] < test_start_bar - 50

        keep = purge_ok & embargo_ok
        clean_train_idx = train_idx[keep.to_numpy()]

        purged_folds.append((clean_train_idx, test_idx.copy()))

    return purged_folds


def make_all_folds(X, y, meta, chunks: list[np.ndarray]) -> dict[str, FoldList]:
    """Tạo bốn nhánh split để file training có thể dùng chung."""
    walk_folds = make_walk_forward(chunks)
    return {
        "random_kfold": make_random_kfold(X),
        "grouped_kfold": make_grouped_kfold(X, y, meta),
        "walk_forward": walk_folds,
        "purged_walk_forward": make_purged_walk_forward(walk_folds, meta),
    }


def _validate_basic_folds(
    name: str,
    folds: FoldList,
    n_rows: int,
    expected_fold_count: int,
) -> np.ndarray:
    """Kiểm tra index hợp lệ, duy nhất và không giao nhau trong từng fold."""
    if len(folds) != expected_fold_count:
        raise ValueError(
            f"{name} phải có {expected_fold_count} fold, hiện có {len(folds)}"
        )

    coverage = np.zeros(n_rows, dtype=np.int8)
    for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
        for role, indices in [("train", train_idx), ("test", test_idx)]:
            if indices.ndim != 1 or len(indices) == 0:
                raise ValueError(f"{name} fold {fold_number}: {role}_idx không hợp lệ")
            if len(np.unique(indices)) != len(indices):
                raise ValueError(f"{name} fold {fold_number}: {role}_idx bị trùng")
            if indices.min() < 0 or indices.max() >= n_rows:
                raise ValueError(f"{name} fold {fold_number}: index vượt phạm vi")

        if np.intersect1d(train_idx, test_idx).size:
            raise ValueError(f"{name} fold {fold_number}: train và test bị giao nhau")
        coverage[test_idx] += 1

    return coverage


def validate_random_folds(folds: FoldList, n_rows: int) -> None:
    """Random K-Fold phải test mỗi dòng đúng một lần."""
    coverage = _validate_basic_folds("Random K-Fold", folds, n_rows, 5)
    if not np.all(coverage == 1):
        raise ValueError("Random K-Fold không phủ mỗi dòng test đúng một lần")


def validate_grouped_folds(folds: FoldList, meta) -> None:
    """Grouped K-Fold phải phủ đủ dữ liệu và không tách origin_bar."""
    coverage = _validate_basic_folds("Grouped K-Fold", folds, len(meta), 5)
    if not np.all(coverage == 1):
        raise ValueError("Grouped K-Fold không phủ mỗi dòng test đúng một lần")

    for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
        train_groups = set(meta.iloc[train_idx]["origin_bar"])
        test_groups = set(meta.iloc[test_idx]["origin_bar"])
        if not train_groups.isdisjoint(test_groups):
            raise ValueError(
                f"Grouped K-Fold fold {fold_number}: origin_bar bị tách hai phía"
            )


def validate_walk_folds(
    folds: FoldList,
    chunks: list[np.ndarray],
    n_rows: int,
) -> None:
    """Walk-forward phải dùng đúng expanding window và chỉ test khúc 2-5."""
    coverage = _validate_basic_folds("Walk-forward", folds, n_rows, 4)

    for test_chunk, (train_idx, test_idx) in enumerate(folds, start=1):
        expected_train = np.concatenate(chunks[:test_chunk])
        expected_test = chunks[test_chunk]
        if not np.array_equal(train_idx, expected_train):
            raise ValueError(f"Walk-forward fold {test_chunk}: train sai khúc")
        if not np.array_equal(test_idx, expected_test):
            raise ValueError(f"Walk-forward fold {test_chunk}: test sai khúc")
        if train_idx.max() >= test_idx.min():
            raise ValueError(f"Walk-forward fold {test_chunk}: train không nằm trước test")

    first_test_row = len(chunks[0])
    if np.any(coverage[:first_test_row] != 0):
        raise ValueError("Walk-forward không được dự đoán khúc 1")
    if np.any(coverage[first_test_row:] != 1):
        raise ValueError("Walk-forward phải test mỗi dòng khúc 2-5 đúng một lần")


def validate_purged_folds(
    purged_folds: FoldList,
    walk_folds: FoldList,
    meta,
) -> None:
    """Purged walk-forward phải chỉ xóa train và thỏa hai điều kiện biên."""
    coverage = _validate_basic_folds(
        "Purged walk-forward", purged_folds, len(meta), 4
    )

    for fold_number, ((walk_train, walk_test), (train_idx, test_idx)) in enumerate(
        zip(walk_folds, purged_folds), start=1
    ):
        if not np.array_equal(test_idx, walk_test):
            raise ValueError(
                f"Purged walk-forward fold {fold_number}: tập test đã bị thay đổi"
            )
        if not np.isin(train_idx, walk_train).all():
            raise ValueError(
                f"Purged walk-forward fold {fold_number}: train có dòng ngoài train gốc"
            )

        test_start_time = meta.iloc[test_idx[0]]["entry_time"]
        test_start_bar = meta.iloc[test_idx[0]]["entry_bar"]
        original_train_meta = meta.iloc[walk_train]
        expected_keep = (
            (original_train_meta["label_end_time"] < test_start_time)
            & (original_train_meta["entry_bar"] < test_start_bar - 50)
        )
        if not np.array_equal(train_idx, walk_train[expected_keep.to_numpy()]):
            raise ValueError(
                f"Purged walk-forward fold {fold_number}: train bị xóa thêm hoặc sai thứ tự"
            )
        train_meta = meta.iloc[train_idx]
        if (train_meta["label_end_time"] >= test_start_time).any():
            raise ValueError(
                f"Purged walk-forward fold {fold_number}: còn label tràn sang test"
            )
        if (train_meta["entry_bar"] >= test_start_bar - 50).any():
            raise ValueError(
                f"Purged walk-forward fold {fold_number}: còn vi phạm embargo"
            )

    first_test_row = walk_folds[0][1][0]
    if np.any(coverage[:first_test_row] != 0):
        raise ValueError("Purged walk-forward không được dự đoán khúc 1")
    if np.any(coverage[first_test_row:] != 1):
        raise ValueError(
            "Purged walk-forward phải test mỗi dòng khúc 2-5 đúng một lần"
        )


def validate_all_folds(
    all_folds: dict[str, FoldList],
    meta,
    chunks: list[np.ndarray],
    n_rows: int,
) -> None:
    """Chạy toàn bộ điều kiện nghiệm thu cho tầng splitting."""
    validate_random_folds(all_folds["random_kfold"], n_rows)
    validate_grouped_folds(all_folds["grouped_kfold"], meta)
    validate_walk_folds(all_folds["walk_forward"], chunks, n_rows)
    validate_purged_folds(
        all_folds["purged_walk_forward"], all_folds["walk_forward"], meta
    )


def print_fold_report(name: str, folds: FoldList, y, reference=None) -> None:
    """In kích thước và tỷ lệ nhãn của từng fold sau khi đã validation."""
    print(f"\n{name}")
    print("Fold | Train | Test | Removed | Train label=1 | Test label=1")
    for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
        removed = 0 if reference is None else len(reference[fold_number - 1][0]) - len(train_idx)
        print(
            f"{fold_number:>4} | {len(train_idx):>5} | {len(test_idx):>4} | "
            f"{removed:>7} | {y.iloc[train_idx].mean():>13.4%} | "
            f"{y.iloc[test_idx].mean():>12.4%}"
        )


def main() -> None:
    df, X, y, meta, edges, chunks = prepare_dataset()
    all_folds = make_all_folds(X, y, meta, chunks)
    validate_all_folds(all_folds, meta, chunks, len(df))

    print_fold_report("Cách 1 - Random K-Fold", all_folds["random_kfold"], y)
    print_fold_report("Cách 1b - Grouped K-Fold", all_folds["grouped_kfold"], y)
    print_fold_report("Cách 2 - Walk-forward", all_folds["walk_forward"], y)
    print_fold_report(
        "Cách 3 - Purged walk-forward",
        all_folds["purged_walk_forward"],
        y,
        reference=all_folds["walk_forward"],
    )
    print("\nKiểm tra bốn cách chia: OK")


if __name__ == "__main__":
    main()
