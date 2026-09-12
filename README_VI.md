# CITD ML — Chiến lược Pyramid BTCUSD + Bộ lọc tín hiệu CatBoost

Đồ án machine learning trả lời câu hỏi: **khi nào thì nên tin một tín hiệu giao
dịch**. Model CatBoost ở đây là **bộ lọc tín hiệu**, không phải bộ ra quyết định
mua/bán: chiến lược pyramid quyết định mở lệnh nào, model chấm điểm từng lệnh —
điểm thấp thì bỏ qua.

Vấn đề trung tâm là **rò rỉ nhãn (label leakage)**. Nhãn triple-barrier được xác
định bằng cách nhìn tới tương lai, nên các lệnh vào gần nhau có nhãn chồng lấn.
Chiến lược còn cố ý mở 1 lệnh gốc + 3 lệnh nhồi (4 lệnh cùng một gia đình
`origin_bar`), nên anh em gần như chung số phận. Chia ngẫu nhiên sẽ tách anh em
sang hai phía train/test và thổi phồng điểm. Đồ án so sánh 4 cách chia với mức
độ chặt tăng dần, rồi đánh giá trên **giai đoạn holdout niêm phong** mà chưa ai
đụng tới trong suốt quá trình phát triển.

> Bản tiếng Anh: [`README.md`](README.md). Đề bài bàn giao gốc nằm trong
> [`docs/`](docs/).

---

## Kết quả

Mốc holdout cố định: **`2025-02-08 15:30:00`** (bar M15 số 199,968).

### Bảng 1 — Chỉ số phân loại, đã bỏ khúc 1 ở cả 4 nhánh

| Cách chia | ROC-AUC | F1 @ 0.5 |
|---|---:|---:|
| Cách 1 — Random K-Fold | 0.8595 | 0.6525 |
| Cách 1b — Grouped K-Fold | 0.7475 | 0.5105 |
| Cách 2 — Walk-forward | 0.5867 | 0.3267 |
| Cách 3 — WF + Purge/Embargo | 0.5948 | 0.3304 |
| **Holdout** | **0.6050** | **0.4017** |

Khoảng cách giữa Random K-Fold (0.86) và các cách chia có ý thức rò rỉ (~0.59)
chính là hiệu ứng leakage mà đồ án muốn chứng minh.

### Bảng 2 — Chỉ số tài chính, giữ top 50%

| | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---:|---:|---:|---:|---:|
| Baseline (khúc 2–5) | 20,007 | +3,358.3 | 236.1 | 1.294 | 31.7 |
| Cách 1 — Random K-Fold | 10,004 | +6,998.3 | 75.7 | 2.563 | 44.9 |
| Cách 1b — Grouped K-Fold | 10,004 | +4,724.3 | 99.5 | 1.942 | 39.2 |
| Cách 2 — Walk-forward | 10,004 | +2,207.3 | 205.3 | 1.423 | 34.9 |
| Cách 3 — WF + Purge/Embargo | 10,004 | +2,253.7 | 154.1 | 1.435 | 35.2 |
| **Baseline holdout** | 5,028 | +245.93 | 305.32 | 1.0992 | 35.28 |
| **Holdout, top 50%** | 2,514 | −23.83 | 272.38 | 0.9815 | 34.88 |

Trên holdout niêm phong, việc lọc bằng model **không** cải thiện so với
baseline. Bảng sweep đầy đủ 20–80% nằm trong
[`outputs/holdout/stage4/`](outputs/holdout/stage4/), báo cáo đầy đủ ở
[`docs/BAO_CAO_KET_QUA_HOLDOUT.md`](docs/BAO_CAO_KET_QUA_HOLDOUT.md).

---

## Cấu trúc repository

```
citd-ml-project/
├── README.md                  ← bản tiếng Anh
├── README_VI.md               ← file này
├── LICENSE                    MIT
├── pyproject.toml             metadata + phiên bản thư viện (uv)
├── requirements.txt           phiên bản tương tự cho pip thuần
├── .gitignore .gitattributes
│
├── data/
│   ├── raw/                   CSV BTCUSD M1 thô (KHÔNG commit, xem bên dưới)
│   └── processed/             input pipeline đã commit
│       ├── dataset_catboost.csv          25,008 dòng × 31 cột
│       ├── tradelist_pyramid_local.csv   30,040 lệnh baseline
│       └── triple_barrier_labels.csv     nhãn triple-barrier
│
├── src/citd_ml/               package Python cài đặt được
│   ├── paths.py               nguồn duy nhất cho đường dẫn/hằng số
│   ├── strategy/              máy trạng thái vào/ra lệnh pyramid
│   ├── labeling/              gán nhãn triple-barrier
│   ├── features/              sinh dataset + kiểm tra
│   ├── training/              chuẩn bị dữ liệu, 4 cách chia, train CatBoost
│   ├── backtest/              backtest baseline + lọc theo điểm
│   └── verification/          kiểm chứng tái lập từng byte
│
├── scripts/                   điểm chạy dòng lệnh
│   ├── build_dataset.py       sinh dataset
│   ├── verify_dataset.py      kiểm tra dataset
│   ├── train_models.py        train 4 nhánh, ghi bảng OOF
│   ├── run_backtest.py        baseline + sweep 20–80%
│   ├── verify_pipeline.py     chạy 2 lần, đối chiếu từng byte
│   ├── holdout_stage1_split.py
│   ├── holdout_stage2_train.py
│   ├── holdout_stage3_backtest.py
│   └── holdout_stage4_report.py
│
├── outputs/                   kết quả/bằng chứng đã commit
│   ├── catboost_training/     4 bảng OOF + metric từng fold
│   ├── backtest/              scored universe, sweep, top-50
│   ├── verification/          manifest tái lập
│   └── holdout/
│       ├── stage1/            dataset tái sinh + holdout, kiểm tra
│       ├── stage2/            model cuối (.cbm), dự đoán, báo cáo
│       ├── stage3/            holdout chấm điểm, top-20…80%, tổng hợp
│       └── stage4/            bảng, quyết định, biểu đồ HTML
│
├── docs/                      đề bài bàn giao + báo cáo kết quả
└── tests/                     test layout/hằng số nhẹ
```

---

## Cài đặt môi trường

**Bắt buộc Python 3.12** (lần chạy đã kiểm chứng dùng Python 3.12.14). Chọn
một trong hai cách: `uv` (khuyến nghị) hoặc `.venv` thuần.

### Cách A — `uv` (khuyến nghị)

`uv` tự tạo và quản lý `.venv/`, đồng thời cài đúng phiên bản đã ghim trong
`pyproject.toml`.

```bash
# 1. cài uv một lần (bỏ qua nếu đã có)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. tạo .venv + cài thư viện và package ở chế độ editable
uv sync

# 3. chạy mọi thứ trong môi trường của dự án
uv run python scripts/train_models.py
uv run pytest
```

`uv sync` đọc `pyproject.toml`, tạo `.venv/` và sinh file lock `uv.lock`. Nên
commit `uv.lock` để mọi bản clone có đúng phiên bản.

Thêm thư viện mới:

```bash
uv add <package>          # cập nhật pyproject.toml + uv.lock
```

### Cách B — `venv` thuần + `pip`

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                 # cài package citd_ml
python scripts/train_models.py
```

> Thư mục `.venv/` đã được `.gitignore` bỏ qua. Không commit nó. Trên Windows
> lệnh kích hoạt là `.venv\Scripts\activate`.

### Kiểm tra môi trường

```bash
uv run python -c "import catboost, sklearn, pandas, numpy, plotly; print('ok')"
uv run pytest
```

---

## Chuẩn bị dữ liệu

File thô `BTCUSD_m1_2018_to_now.csv` nặng **~209 MB**, vượt giới hạn 100 MB/file
của GitHub nên **không được commit**. Xem
[`data/raw/README.md`](data/raw/README.md) để biết định dạng và cách tải lại.

Đặt file tại:

```
data/raw/BTCUSD_m1_2018_to_now.csv
```

Toàn bộ `data/processed/` và `outputs/` **đã được commit**, nên có thể xem mọi
kết quả và chạy lại phần train/backtest mà không cần file thô. File thô chỉ cần
khi muốn sinh lại dataset từ đầu.

---

## Chạy pipeline

Các lệnh dưới đây chạy từ thư mục gốc repo, dùng `uv run` (thay bằng `python`
nếu bạn tự kích hoạt `.venv`).

### 1. Sinh dataset

```bash
uv run python scripts/build_dataset.py
```

Chạy lại chiến lược trên dữ liệu M1 trước holdout, gán nhãn triple-barrier cho
từng lệnh, tính 23 feature, ghi ra `data/processed/dataset_catboost.csv`
(25,008 dòng).

### 2. Kiểm tra dataset

```bash
uv run python scripts/verify_dataset.py
```

Kiểm tra số dòng/cột, miền giá trị feature, phân bố nhãn, ranh giới khúc và
việc holdout còn niêm phong. Exit code khác 0 nghĩa là **dừng lại**.

### 3. Train 4 nhánh

```bash
uv run python scripts/train_models.py
```

Train Random K-Fold, Grouped K-Fold, Walk-forward và Purged Walk-forward, ghi
xác suất OOF + metric từng fold vào `outputs/catboost_training/`.

### 4. Backtest và sweep

```bash
uv run python scripts/run_backtest.py
```

Chạy baseline vũ trụ lệnh cố định, lọc bằng từng bảng điểm OOF ở mức 20–80%,
ghi báo cáo vào `outputs/backtest/`.

### 5. Kiểm chứng tái lập

```bash
uv run python scripts/verify_pipeline.py
```

Chạy training và backtest hai lượt, khẳng định kết quả giống hệt các CSV đã
commit từng byte. Ghi `outputs/verification/reproducibility.json`.

---

## Quy trình holdout niêm phong

Holdout chỉ mở một lần, qua 4 giai đoạn. Chạy tuần tự.

```bash
# Giai đoạn 1 — tái sinh toàn bộ lịch sử và tách holdout
uv run python scripts/build_dataset.py --holdout 2100-01-01 \
    --output outputs/holdout/stage1/dataset_catboost_full_regenerated.csv
uv run python scripts/holdout_stage1_split.py

# Giai đoạn 2 — train model cuối 2 lần và kiểm tái lập
uv run python scripts/holdout_stage2_train.py --run-id 1
uv run python scripts/holdout_stage2_train.py --run-id 2
uv run python scripts/holdout_stage2_train.py --verify

# Giai đoạn 3 — chấm điểm holdout và backtest vũ trụ lệnh cố định
uv run python scripts/holdout_stage3_backtest.py

# Giai đoạn 4 — dựng bảng tổng hợp và biểu đồ đường vốn HTML
uv run python scripts/holdout_stage4_report.py
```

Giai đoạn 1 còn chứng minh bản tái sinh trước holdout chứa đủ 25,008 khóa cũ
và khớp toàn bộ 575,184 ô feature. Giai đoạn 2 kiểm tra purge/embargo cắt 0
dòng và hai lần train độc lập cho dự đoán giống hệt nhau.

---

## Hằng số cố định

Các giá trị này do đề bài bàn giao chốt, **không được đổi**
(`src/citd_ml/paths.py`).

| Hằng số | Giá trị |
|---|---|
| Mốc holdout | `2025-02-08 15:30:00` (bar M15 số 199,968) |
| Dataset | 25,008 dòng, 6,252 gia đình × 4 leg, tỷ lệ nhãn 1 = 26.2% |
| Feature / metadata | 23 `FEATURES`, 7 `META` |
| Ranh giới 5 khúc | `[0, 5001, 10003, 15004, 20006, 25008]` |
| Hyperparameter CatBoost | iterations 1000, lr 0.05, depth 6, l2_leaf_reg 3.0, `auto_class_weights=Balanced`, `eval_metric=AUC`, seed 42 |
| Môi trường đã kiểm | Python 3.12.14 · catboost 1.2.10 · scikit-learn 1.9.0 · pandas 3.0.5 · numpy 2.5.2 |

---

## Lưu ý và quy ước

- **Không đưa `META` vào model.** `leg` và `entry_vs_base_R` mô tả giàn giáo
  pyramid chứ không phải thị trường, sẽ rò rỉ cấu trúc gia đình.
- **Vũ trụ lệnh ứng viên là cố định.** Bỏ một lệnh không được làm đổi tín hiệu
  về sau; backtest giữ một chiến lược `shadow` chính vì lý do này.
- **Không sửa dataset hay `outputs/` đã commit** khi thử nghiệm. Ghi kết quả
  mới ra thư mục riêng để giữ nguyên các artifact đã niêm phong.
- Seed cố định và không có yếu tố ngẫu nhiên trong chấm điểm/backtest, nên các
  lần chạy lặp phải giống hệt từng byte.

## Tài liệu

| File | Nội dung |
|---|---|
| [`docs/BAO_CAO_KET_QUA_HOLDOUT.md`](docs/BAO_CAO_KET_QUA_HOLDOUT.md) | Báo cáo kết quả holdout (số liệu, biểu đồ, kiểm chứng) |
| [`docs/BAN_GIAO_task_holdout.md`](docs/BAN_GIAO_task_holdout.md) | Đề bài holdout gốc |
| [`docs/BAN_GIAO_task_train_catboost.md`](docs/BAN_GIAO_task_train_catboost.md) | Đề bài train gốc |
| [`docs/quy_trinh_lam_viec.md`](docs/quy_trinh_lam_viec.md) | Quy trình làm việc |
| [`docs/handover_README.md`](docs/handover_README.md) | README bàn giao gốc |

## Giấy phép

Phát hành theo giấy phép MIT — xem [`LICENSE`](LICENSE).
