# Bàn giao — bước holdout

Việc phải làm: đọc `BAN_GIAO_task_holdout.md`.

File này chỉ mô tả pipeline đã có và thư mục có gì.

---

## Pipeline đã chạy tới đâu

| Bước | Code | Ra file |
|---|---|---|
| 1 | `pyramid_strategy.py` chạy qua `backtest_pyramid_local.py` | `tradelist_pyramid_local.csv` — 30,040 lệnh, 2018-01 → 2026-08 |
| 2 | `label_triple_barrier_mới.py` | `triple_barrier_labels_mới.csv` — nhãn triple barrier cho từng lệnh |
| 3 | `build_features.py`, kiểm bằng `verify_dataset.py` | `dataset_catboost.csv` — 25,008 dòng × 31 cột, phần trước mốc holdout |
| 4 | `pre_train.py` → `split_data.py` → `train_catboost.py` → `backtest_pyramid_local.py` | `outputs/` |
| 5 | **chưa làm — xem `BAN_GIAO_task_holdout.md`** | |

Bước 1–3 đã chốt xong, **không sửa gì trong đó**. Bước 4 chạy lại được, dùng làm khuôn cho bước 5.

## Bước 4 chạy thế nào

```powershell
python train_catboost.py           # 18 fold, ra 6 CSV trong outputs/catboost_training
python backtest_pyramid_local.py   # baseline + 4 nhánh × 7 mức giữ lệnh, ra 8 CSV trong outputs/backtest
python verify_pipeline.py          # chạy lặp 2 lượt, so từng byte, ghi outputs/verification
```

`pre_train.py` đọc và kiểm dataset, chia 5 khúc. `split_data.py` tạo 4 cách chia và tự kiểm. `train_catboost.py` train từng fold, lưu xác suất OOF. `backtest_pyramid_local.py` chạy lại chiến lược, ghép điểm OOF, lọc top-k, tính chỉ số.

**Logic backtest nhất quán từ đầu tới giờ.** Phần chạy lại chiến lược trong `backtest_pyramid_local.py` giữ nguyên như bản gốc; những gì thêm vào chỉ nằm ở lớp ngoài — cắt dữ liệu tại mốc holdout, ghép điểm, lọc, xuất báo cáo. Mỗi lần chạy nó tự đối chiếu baseline với `tradelist_pyramid_local.csv`; lệch là dừng và báo lỗi. Bước holdout giữ nguyên cách này.

---

## Hằng số cố định

| | |
|---|---|
| Mốc holdout | `2025-02-08 15:30:00`, ứng với bar M15 số 199,968 |
| Dataset | 25,008 dòng, 6,252 tín hiệu × 4 leg, tỷ lệ nhãn 1 = 26.2% |
| Feature / metadata | 23 cột `FEATURES`, 7 cột `META` — lấy từ `build_features.py` |
| Chia 5 khúc theo dòng | `[0, 5001, 10003, 15004, 20006, 25008]` |
| 4 cách chia | Random K-Fold · Grouped K-Fold · Walk-forward · WF + Purge/Embargo |
| Hyperparameter CatBoost | iterations 1000, lr 0.05, depth 6, l2_leaf_reg 3.0, `auto_class_weights` Balanced, `eval_metric` AUC, seed 42 |
| Môi trường đã kiểm | Python 3.12, catboost 1.2.10, scikit-learn 1.9.0, pandas 3.0.5, numpy 2.5.2 |

Không đổi bất kỳ dòng nào trong bảng này.

---

## Thư mục

```
Ban giao - Holdout/
├── README.md                     ← file này
├── BAN_GIAO_task_holdout.md      ← VIỆC PHẢI LÀM
│
├── tai_lieu/
│   └── BAN_GIAO_task_train_catboost.md    ← task bước 4, tra lại cách chia và luật lọc
│
└── chay/                         ← thư mục làm việc
    ├── BTCUSD_m1_2018_to_now.csv          dữ liệu gốc duy nhất
    ├── pyramid_strategy.py
    ├── tradelist_pyramid_local.csv         mốc đối chiếu baseline
    ├── label_triple_barrier_mới.py
    ├── triple_barrier_labels_mới.csv
    ├── build_features.py
    ├── verify_dataset.py
    ├── dataset_catboost.csv
    ├── pre_train.py
    ├── split_data.py
    ├── train_catboost.py
    ├── backtest_pyramid_local.py
    ├── verify_pipeline.py
    ├── BAN_GIAO_task_train_catboost.md     bản sao, `verify_pipeline.py` băm file này
    └── outputs/
        ├── catboost_training/    4 bảng OOF, metric từng fold, metric trung bình
        ├── backtest/             baseline, scored universe, top 50%, sweep 20–80%
        └── verification/         bằng chứng chạy lặp
```

**`chay/` phải để phẳng như vậy.** Code import lẫn nhau bằng đường dẫn tương đối, xếp lại vào thư mục con là gãy.

---

## Nộp gì

**Báo cáo kết quả** — 5 phần, liệt kê ở mục 6 của `BAN_GIAO_task_holdout.md`: bảng số, biểu đồ, quyết định triển khai, kiểm chứng, bảng file sinh ra.

Gom số cho gọn để Nhi biện luận ở báo cáo chính. Không viết phần diễn giải, biện luận, kết luận.

Báo cáo technical (file nào làm gì, hàm nào làm gì) là task riêng ở cuối đồ án, chưa phải bây giờ.
