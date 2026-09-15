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
độ chặt tăng dần, rồi đánh giá trên **giai đoạn holdout niêm phong**. Ba phiên
bản kết quả lịch sử truy xuất được từ Git; các lần kiểm chứng kỹ thuật và thí
nghiệm số luồng sau đó được công khai riêng. Sổ không xác nhận tổng số lần
thực thi hoặc chứng minh không quyết định nào chịu ảnh hưởng từ việc xem
holdout. Xem
[`outputs/verification/holdout_run_history.md`](outputs/verification/holdout_run_history.md).

> Bản tiếng Anh: [`README.md`](README.md). Đề bài bàn giao gốc nằm trong
> [`docs/`](docs/).

---

## Kết quả

Mốc holdout cố định: **`2025-02-08 15:30:00`** (bar M15 số 199,968).

Các khối chính dưới đây là **bản chạy lại canonical `thread_count=1`** đã commit
trong [`outputs/step4_thread1/`](outputs/step4_thread1/); cả 8 file CSV được đối
chiếu giống hệt từng byte với artifact `thread_count=1` của commit `7748828`.
Số nhánh bàn giao (sinh trước khi ghim `thread_count`) được giữ làm khối tham
chiếu có dán nhãn; các dòng holdout không đổi.

### Bảng 1 — Chỉ số phân loại, đã bỏ khúc 1 ở cả 4 nhánh

| Cách chia | ROC-AUC | F1 @ 0.5 |
|---|---:|---:|
| Cách 1 — Random K-Fold | 0.8582 | 0.6494 |
| Cách 1b — Grouped K-Fold | 0.7454 | 0.5077 |
| Cách 2 — Walk-forward | 0.5875 | 0.3288 |
| Cách 3 — WF + Purge/Embargo | 0.5895 | 0.3247 |
| **Holdout** | **0.6046** | **0.4022** |

Khối tham chiếu — artifact bàn giao (không ghim `thread_count`), kèm chênh lệch
canonical trừ tham chiếu:

| Cách chia | ROC-AUC tham chiếu | F1 tham chiếu | ΔROC-AUC | ΔF1 |
|---|---:|---:|---:|---:|
| Cách 1 — Random K-Fold | 0.8595 | 0.6525 | −0.0013 | −0.0031 |
| Cách 1b — Grouped K-Fold | 0.7475 | 0.5105 | −0.0021 | −0.0028 |
| Cách 2 — Walk-forward | 0.5867 | 0.3267 | +0.0007 | +0.0020 |
| Cách 3 — WF + Purge/Embargo | 0.5948 | 0.3304 | −0.0052 | −0.0057 |

Khoảng cách giữa Random K-Fold (~0.86) và các cách chia có ý thức rò rỉ (~0.59)
chính là hiệu ứng leakage mà đồ án muốn chứng minh.

### Bảng 2 — Chỉ số tài chính, giữ top 50%

Khối chính canonical `thread_count=1`:

| | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---:|---:|---:|---:|---:|
| Baseline (khúc 2–5) | 20,007 | +3,358.25 | 236.13 | 1.2940 | 31.71 |
| Cách 1 — Random K-Fold | 10,004 | +6,937.53 | 73.76 | 2.5437 | 44.65 |
| Cách 1b — Grouped K-Fold | 10,004 | +4,752.04 | 99.71 | 1.9500 | 39.26 |
| Cách 2 — Walk-forward | 10,004 | +2,148.31 | 211.29 | 1.4117 | 34.76 |
| Cách 3 — WF + Purge/Embargo | 10,004 | +2,119.48 | 142.36 | 1.4074 | 34.99 |

Khối tham chiếu — artifact bàn giao:

| | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---:|---:|---:|---:|---:|
| Baseline (khúc 2–5) | 20,007 | +3,358.25 | 236.13 | 1.2940 | 31.71 |
| Cách 1 — Random K-Fold | 10,004 | +6,998.30 | 75.71 | 2.5627 | 44.88 |
| Cách 1b — Grouped K-Fold | 10,004 | +4,724.34 | 99.50 | 1.9418 | 39.24 |
| Cách 2 — Walk-forward | 10,004 | +2,207.25 | 205.31 | 1.4226 | 34.86 |
| Cách 3 — WF + Purge/Embargo | 10,004 | +2,253.72 | 154.11 | 1.4350 | 35.17 |

Net R top 50% canonical lệch khối bàn giao −60.77 / +27.70 / −58.94 / −134.24 R
cho bốn nhánh theo thứ tự (đối chiếu từng file trong
[`outputs/step4_thread1/comparison_report.json`](outputs/step4_thread1/comparison_report.json)).

Khối holdout (không đổi):

| | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---:|---:|---:|---:|---:|
| **Baseline holdout** | 5,028 | +245.93 | 305.32 | 1.0992 | 35.28 |
| **Holdout, top 50%** | 2,514 | −36.55 | 263.25 | 0.9718 | 34.81 |

Trên holdout niêm phong, bộ lọc vẫn **không** thắng baseline không lọc: giữ top
50% còn 2,514 lệnh với **−36.55 R** và profit factor 0.972, so với **+245.93 R**
và profit factor 1.099 của baseline đủ 5,028 lệnh. Bảng sweep đầy đủ 20–80% nằm trong
[`outputs/holdout/stage4/`](outputs/holdout/stage4/), báo cáo đầy đủ ở
[`docs/BAO_CAO_KET_QUA_HOLDOUT.md`](docs/BAO_CAO_KET_QUA_HOLDOUT.md).

### Độ nhạy `thread_count` và lịch sử mở holdout

Thí nghiệm có kiểm soát chỉ đổi `thread_count`
([`outputs/thread_count_sensitivity/`](outputs/thread_count_sensitivity/)) cho
thấy trên máy này hai lần chạy `thread_count=1` giống hệt từng bit
(`max|Δp| = 0`), còn `thread_count=2` / `-1` làm **toàn bộ** prediction thay đổi
(train `max|Δp|` 0.2832 / 0.2999; holdout 0.3469 / 0.3352; Pearson holdout
0.9331 / 0.9409). Net R top 50% holdout: **−36.55 R** (`tc=1`) so với
**−12.26 R** (`tc=2`) và **+30.79 R** (mặc định). Ảnh hưởng nằm ở bước train,
không phải inference: chấm lại cùng model đã fit với prediction `thread_count`
1 / 2 / −1 cho xác suất giống hệt nhau. CatBoost mô tả `thread_count` là tham số
tốc độ không ảnh hưởng kết quả; phép đo trên máy này xác nhận ở bước prediction
nhưng không xác nhận ở bước train. Phạm vi: một máy, một build CatBoost, một
dataset, một seed.

Ba phiên bản kết quả holdout lịch sử truy xuất được từ Git nằm trong
[`outputs/verification/holdout_run_history.json`](outputs/verification/holdout_run_history.json)
([Markdown](outputs/verification/holdout_run_history.md)): `dc25cd3` Windows
0.60502 / 0.40171, top-50 −23.83 R; `5f46e41` Linux đa luồng
0.60232 / 0.40647, +30.79 R; `7748828` Linux `thread_count=1`
0.60455 / 0.40221, −36.55 R. Phép kiểm toàn chuỗi run1-vs-run2
([`outputs/holdout/repro/stage5_repro_report.json`](outputs/holdout/repro/stage5_repro_report.json))
**PASS** với mọi delta 0.0 và cả bốn biểu đồ giống hệt từng byte.

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
│   ├── thread_count_sensitivity.py   thí nghiệm thread_count có kiểm soát
│   ├── holdout_run_history.py        dựng lại 3 lần mở holdout từ git
│   ├── holdout_stage1_split.py
│   ├── holdout_stage2_train.py
│   ├── holdout_stage3_backtest.py
│   ├── holdout_stage4_report.py
│   └── holdout_stage5_repro_check.py đối chiếu chuỗi run1 vs run2
│
├── outputs/                   kết quả/bằng chứng đã commit
│   ├── catboost_training/     4 bảng OOF + metric từng fold bàn giao (tham chiếu đóng băng)
│   ├── backtest/              scored universe, sweep, top-50 bàn giao (tham chiếu đóng băng)
│   ├── step4_thread1/         bản chạy lại canonical thread_count=1 + comparison_report.json
│   ├── thread_count_sensitivity/  JSON thí nghiệm + CSV tổng hợp
│   ├── verification/          manifest bàn giao + holdout_run_history.json/.md
│   └── holdout/
│       ├── stage1/            dataset tái sinh + holdout, kiểm tra
│       ├── stage2/            model cuối (.cbm), dự đoán, báo cáo
│       ├── stage3/            holdout chấm điểm, top-20…80%, tổng hợp
│       ├── stage4/            bảng, quyết định, biểu đồ HTML
│       └── repro/             báo cáo stage5 run1-vs-run2 + cây run2 độc lập
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
uv sync --extra dev

# 3. chạy mọi thứ trong môi trường của dự án
uv run python scripts/train_models.py
uv run pytest
```

`uv sync --extra dev` đọc `pyproject.toml`, tạo `.venv/`, cài thêm `pytest` từ
nhóm `dev` và dùng file lock `uv.lock` đã commit để mọi bản clone có đúng phiên
bản.

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

> **Artifact bước 4 đã niêm phong.** `outputs/catboost_training/`,
> `outputs/backtest/` và `outputs/verification/reproducibility.json` là kết quả
> bước 4 bàn giao, được giữ nguyên từng byte. Chúng chỉ còn là **tham chiếu**:
> cây đối chiếu canonical là
> [`outputs/step4_thread1/`](outputs/step4_thread1/) (`thread_count=1`), và
> CatBoost phụ thuộc phần cứng nên các file bàn giao cố ý không tái lập byte
> trên Linux. Vì vậy các bước 3–5 dưới đây ghi vào cây canonical qua
> `--output-dir` / `--oof-dir`; chạy thiếu các cờ đó sẽ **ghi đè** thư mục bàn
> giao. Quy trình holdout niêm phong ở mục sau là phần bàn giao bước 5 và không
> đụng tới các thư mục đó.

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
uv run python scripts/train_models.py \
    --output-dir outputs/step4_thread1/catboost_training
```

Train Random K-Fold, Grouped K-Fold, Walk-forward và Purged Walk-forward, ghi
xác suất OOF + metric từng fold vào thư mục `--output-dir`
(mặc định: `outputs/catboost_training/`).

### 4. Backtest và sweep

```bash
uv run python scripts/run_backtest.py \
    --output-dir outputs/step4_thread1/backtest \
    --oof-dir outputs/step4_thread1/catboost_training
```

Chạy baseline vũ trụ lệnh cố định, lọc bằng từng bảng điểm OOF ở mức 20–80%,
ghi báo cáo vào `--output-dir` (mặc định: `outputs/backtest/`); `--oof-dir`
chọn bảng OOF để lọc.

### 5. Kiểm chứng tái lập (cây canonical)

```bash
uv run python scripts/verify_pipeline.py \
    --train-dir outputs/step4_thread1/catboost_training \
    --backtest-dir outputs/step4_thread1/backtest \
    --manifest outputs/step4_thread1/verification/reproducibility.json
```

Chạy training và backtest hai lượt, khẳng định kết quả giống hệt các CSV
canonical đã lưu từng byte (~10 phút). `--train-dir` / `--backtest-dir` chọn cây
để kiểm tra, `--manifest` chọn nơi ghi JSON bằng chứng.
`outputs/verification/reproducibility.json` là manifest **bàn giao** và được cố
ý giữ nguyên; manifest canonical nằm trong cây
`outputs/step4_thread1/verification/`. Chạy `verify_pipeline.py` không tham số
sẽ nhắm vào thư mục bàn giao và ghi đè
`outputs/verification/reproducibility.json`; không dùng trong luồng canonical.

---

## Quy trình holdout niêm phong

Quy trình chạy holdout qua 4 giai đoạn, rồi đối chiếu hai lượt chạy ở phép
kiểm thứ 5. Ba phiên bản kết quả lịch sử được ghi trong
[`outputs/verification/holdout_run_history.md`](outputs/verification/holdout_run_history.md).
Sổ không phải nhật ký đầy đủ mọi lần thực thi. Các lần kiểm chứng và thí
nghiệm số luồng là những lần truy cập holdout bổ sung. Chạy các giai đoạn tuần tự.

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

# Lần chạy 2 độc lập, ghi vào cây repro (không đụng stage3/stage4)
uv run python scripts/holdout_stage3_backtest.py --run-id 2 \
    --out-dir outputs/holdout/repro/run2/stage3
uv run python scripts/holdout_stage4_report.py \
    --stage3-dir outputs/holdout/repro/run2/stage3 \
    --out-dir outputs/holdout/repro/run2/stage4 \
    --report outputs/holdout/repro/run2/BAO_CAO_holdout_run2.md

# Giai đoạn 5 — đối chiếu run 1 và run 2 toàn chuỗi (Stage 3–4)
uv run python scripts/holdout_stage5_repro_check.py
```

Giai đoạn 3 nhận `--run-id` và `--out-dir`; Giai đoạn 4 nhận `--stage3-dir`,
`--out-dir` và `--report` (cùng các cờ ghi đè `--canonical-dir`,
`--sensitivity-json`, `--run-history-json`, `--stage5-json`). Cây run 2 nằm tại
[`outputs/holdout/repro/run2/`](outputs/holdout/repro/run2/), và Giai đoạn 5
đối chiếu hai lần chạy toàn chuỗi: **PASS** với mọi delta 0.0 và cả bốn biểu đồ
giống hệt từng byte
([`stage5_repro_report.json`](outputs/holdout/repro/stage5_repro_report.json)).

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
| Hyperparameter CatBoost | iterations 1000, lr 0.05, depth 6, l2_leaf_reg 3.0, `auto_class_weights=Balanced`, `eval_metric=AUC`, seed 42, `thread_count=1` |
| Môi trường đã kiểm | Python 3.12.14 · catboost 1.2.10 · scikit-learn 1.9.0 · pandas 3.0.5 · numpy 2.5.2 · plotly 7.0.0 |

> `thread_count=1` là **tham số kỹ thuật thêm vào**, không thuộc danh sách
> hyperparameter mô hình gốc. Ngày 12/09 ghim một luồng theo nghi ngờ kỹ thuật
> ban đầu; ngày 15/09 mới bổ sung thí nghiệm có kiểm soát. Phép đo trên máy này: hai lần
> chạy cùng cấu hình `thread_count=1` giống hệt từng bit (`max|Δp| = 0`), còn
> đổi `thread_count` lúc **train** sang `2` hoặc `-1` làm toàn bộ prediction
> thay đổi (train `max|Δp|` 0.2832 / 0.2999; holdout 0.3469 / 0.3352) và đổi
> net R top 50% holdout (−36.55 / −12.26 / +30.79 R). Ảnh hưởng nằm ở bước
> train, không phải inference. Điều này xác nhận mô tả của CatBoost ở bước
> prediction nhưng mâu thuẫn ở bước train; phạm vi là một máy, một build
> CatBoost, một dataset và một seed, nên không quy toàn bộ sai lệch liên máy
> cho số luồng. Các artifact nhánh bàn giao trong `outputs/catboost_training/`
> và `outputs/backtest/` được sinh khi chưa có nó và được giữ nguyên (xem
> Changelog).

---

## Lưu ý và quy ước

- **Không đưa `META` vào model.** `leg` và `entry_vs_base_R` mô tả giàn giáo
  pyramid chứ không phải thị trường, sẽ rò rỉ cấu trúc gia đình.
- **Vũ trụ lệnh ứng viên là cố định.** Bỏ một lệnh không được làm đổi tín hiệu
  về sau; backtest giữ một chiến lược `shadow` chính vì lý do này.
- **Không sửa dataset hay `outputs/` đã commit** khi thử nghiệm. Ghi kết quả
  mới ra thư mục riêng để giữ nguyên các artifact đã niêm phong.
- **Tái lập:** đã ghim cả `random_seed=42` lẫn `thread_count=1`. Trên máy tạo
  artifact, hai lượt train độc lập cho mảng prediction giống hệt; hai file
  `.cbm` khác SHA-256 do phần metadata tuần tự hóa. Bản chạy commit `7748828`
  được tái tạo từng byte bởi bản chạy lại canonical (8 file CSV được đối
  chiếu); manifest `verify_pipeline` bàn giao được giữ nguyên, manifest
  canonical nằm trong `outputs/step4_thread1/verification/`. Thí nghiệm có kiểm
  soát đã commit cho thấy `thread_count` **có** làm thay đổi kết quả train trên
  máy này (train `max|Δp|` 0.2832 / 0.2999; holdout 0.3469 / 0.3352; net R top
  50% holdout −36.55 so với −12.26 và +30.79 R cho `tc=1` / `tc=2` / mặc
  định), trong khi chấm lại cùng model đã fit thì không đổi. Bản
  `thread_count=1` được giữ có net R top 50% thấp hơn bản Linux đa luồng
  lịch sử (−36.55 R so với +30.79 R). Đây là dữ kiện hỗ trợ minh bạch, không
  tự chứng minh không cherry-pick hoặc thay thế lịch sử quyết định. Giữ top
  50% đã chốt; không chọn lại tỷ lệ theo sweep holdout. Tương đương liên máy **chưa
  được chứng minh**: teammate Windows báo prediction trùng trong `1e-15`
  (không byte-identical) và không có artifact đối chứng. Chuỗi đầy đủ run1-vs-
  run2 trên cùng máy (Stage 3–4) PASS, mọi delta 0.0 và biểu đồ giống hệt từng
  byte. Các artifact bước 4 bàn giao (`outputs/catboost_training/`,
  `outputs/backtest/`) được giữ nguyên từng byte và sinh trên máy gốc khi chưa
  có `thread_count`; bản chạy lại trên Linux không tái tạo các file gốc từng
  byte. Quan sát này không bảo đảm rằng mọi máy khác đều sẽ không tái lập được.

## Tài liệu

| File | Nội dung |
|---|---|
| [`docs/BAO_CAO_KET_QUA_HOLDOUT.md`](docs/BAO_CAO_KET_QUA_HOLDOUT.md) | Báo cáo kết quả holdout (số liệu, biểu đồ, kiểm chứng) |
| [`docs/BAN_GIAO_task_holdout.md`](docs/BAN_GIAO_task_holdout.md) | Đề bài holdout gốc |
| [`docs/BAN_GIAO_task_train_catboost.md`](docs/BAN_GIAO_task_train_catboost.md) | Đề bài train gốc |
| [`docs/quy_trinh_lam_viec.md`](docs/quy_trinh_lam_viec.md) | Quy trình làm việc |
| [`docs/handover_README.md`](docs/handover_README.md) | README bàn giao gốc |

## Changelog

### 2026-09-15 — Hoàn thiện báo cáo cho Nhi và đối chiếu yêu cầu

- Sửa báo cáo kết quả và script sinh: ba phiên bản lịch sử truy xuất từ Git, nghi ngờ ngày 12/09 và thí nghiệm ngày 15/09, giữ top 50%, phạm vi replay bốn lệnh biên.
- Thêm bảng đối chiếu sáu feedback trong phần kiểm chứng. Bỏ khẳng định kết quả xấu hơn chứng minh không cherry-pick hoặc Git ghi đủ mọi lần truy cập holdout.
- Sinh lại hai bản report và sổ lịch sử; số liệu và cấu hình model giữ nguyên. Chưa chứng minh tái lập liên máy.
- Tài liệu chính gửi Nhi là [báo cáo kết quả](docs/BAO_CAO_KET_QUA_HOLDOUT.md); audit context chỉ là hồ sơ kiểm tra bổ sung.

### 2026-09-15 — Vòng feedback review: chạy lại canonical thread_count, thí nghiệm có kiểm soát, sổ lịch sử holdout, kiểm chứng toàn chuỗi

- **Chạy lại canonical `thread_count=1` cho 4 cách chia** (`54eed41`,
  `outputs/step4_thread1/`): `scripts/train_models.py` và
  `scripts/run_backtest.py` nhận `--output-dir` / `--oof-dir` để chạy lại 4
  nhánh mà không đụng thư mục bàn giao đóng băng. Chỉ số chunk 2–5 canonical:
  0.8582/0.6494, 0.7454/0.5077, 0.5875/0.3288, 0.5895/0.3247 (bàn giao:
  0.8595/0.6525, 0.7475/0.5105, 0.5867/0.3267, 0.5948/0.3304); net R top 50%
  canonical +6,937.53 / +4,752.04 / +2,148.31 / +2,119.48 (baseline 20,007
  lệnh, +3,358.25 R) so với bàn giao +6,998.30 / +4,724.34 / +2,207.25 /
  +2,253.72. Cả 8 file CSV được đối chiếu giống hệt từng byte với commit
  `7748828`; phần so với bàn giao nằm trong `comparison_report.json`.
- **Thí nghiệm `thread_count` có kiểm soát** (`b278617`,
  `outputs/thread_count_sensitivity/`): chỉ đổi `thread_count`, hai lần chạy
  `thread_count=1` giống hệt từng bit (`max|Δp| = 0`), còn `thread_count=2` /
  `-1` làm toàn bộ prediction thay đổi (train `max|Δp|` 0.2832 / 0.2999;
  holdout 0.3469 / 0.3352; Pearson 0.9331 / 0.9409). Net R top 50% holdout:
  −36.55 (`tc=1`) / −12.26 (`tc=2`) / +30.79 (mặc định). Ảnh hưởng nằm ở bước
  train, không phải inference. CatBoost mô tả `thread_count` là tham số tốc độ;
  phép đo trên máy này xác nhận ở bước prediction nhưng mâu thuẫn ở bước train.
  Phạm vi: một máy, một build CatBoost, một dataset, một seed.
- **Dựng lại sổ lịch sử mở holdout** (`a3ff712`,
  `outputs/verification/holdout_run_history.json` + `.md`): ba phiên bản kết quả lịch sử được
  dựng lại từ git kèm sweep 20–80% đầy đủ và delta pairwise — `dc25cd3`
  Windows 0.60502/0.40171, top-50 −23.83 R; `5f46e41` Linux đa luồng
  0.60232/0.40647, +30.79 R; `7748828` Linux `thread_count=1`
  0.60455/0.40221, −36.55 R. Sổ cũng ghi xác minh Windows của teammate Bùi Quốc
  Thịnh (prediction trùng trong `1e-15`, không byte-identical, không ghi metric)
  như bằng chứng ngoài không tái tạo được, và kiểm lại cả 12 giá trị đã biết
  với dung sai `1e-12`.
- **Kiểm chứng toàn chuỗi run1-vs-run2** (`8003c82`,
  `outputs/holdout/repro/stage5_repro_report.json`): Stage 2/3/4 nhận thêm
  `--run-id` / `--out-dir` / `--stage3-dir` / `--report`; Stage 3–4 của run 2
  được sinh lại dưới `outputs/holdout/repro/run2/` và Stage 5 đối chiếu toàn
  chuỗi: **PASS**, mọi delta 0.0, cả bốn biểu đồ giống hệt từng byte. Report
  Stage 2 nay ghi cả OS (`platform`) và phiên bản plotly.
- **Manifest `verify_pipeline` canonical** (`8003c82`): `verify_pipeline.py`
  nhận `--train-dir` / `--backtest-dir` / `--manifest`; lượt chạy canonical
  (train ×2, backtest ×2) giống hệt `outputs/step4_thread1/` và ghi
  `outputs/step4_thread1/verification/reproducibility.json`. File đóng băng
  `outputs/verification/reproducibility.json` được giữ nguyên.
- **Sinh lại báo cáo kết quả từ artifact** (`ee24520`,
  `docs/BAO_CAO_KET_QUA_HOLDOUT.md`): khối chính Bảng 1/2/3 dùng cây canonical,
  số bàn giao thành khối tham chiếu có dán nhãn kèm bảng chênh lệch; Bảng 4 giữ
  sweep holdout 20–80%; thêm mục lịch sử mở holdout và thí nghiệm
  `thread_count`; nhúng ảnh PNG cạnh link HTML; ghi 4 lệnh biên
  (25,008 + 4 + 5,028 = 30,040) và giai đoạn holdout 2025-02-09 → 2026-08-21;
  cập nhật bảng kiểm chứng.
- **Làm mới bằng chứng bị vô hiệu bởi các sửa đổi mã nguồn:** sổ lịch sử chạy
  lại tại HEAD `ee24520`; lệnh `verify_pipeline` canonical ở trên chạy lại
  **PASS** (hai lượt train + hai lượt backtest, manifest canonical được làm
  mới); `holdout_stage5_repro_check.py` chạy lại **PASS**; bộ test
  **13 passed**.
- **Sửa lỗi best-practice từ audit mã nguồn** (`docs/CODE_BEST_PRACTICE_AUDIT.md`):
  `allow_writing_files=False` vào thẳng `MODEL_PARAMS` dùng chung cho Stage 2 và
  thí nghiệm `thread_count` (không còn sinh `catboost_info/`);
  `matplotlib==3.11.2` được khai báo/pin và ghi phiên bản vào evidence Stage
  2/Stage 5; PNG Stage 4 dùng `metadata={"Software": None}` để byte không còn
  phụ thuộc phiên bản matplotlib, và Plotly dùng `include_plotlyjs=True` đúng
  tài liệu. Toàn bộ evidence Stage 2–5 cùng manifest `verify_pipeline` canonical
  đã chạy lại với số liệu không đổi và prediction giống hệt từng byte; audit đã
  có thêm mục Resolution.

### 2026-09-13 — Hoàn thiện bàn giao holdout và phạm vi bằng chứng

- **Áp dụng feedback cuối cho báo cáo:** bổ sung lịch sử và lý do kỹ thuật phải
  chạy lại holdout; chú thích Bảng 1 là trung bình theo fold sau khi bỏ khúc 1;
  ghi rõ toàn pipeline tái lập từng byte **lúc đó** chưa được chứng minh — đã
  được thay thế ngày 2026-09-15: chuỗi run1-vs-run2 trên cùng máy nay PASS,
  còn tương đương liên máy vẫn còn giới hạn. Đồng thời mô tả mục đích cụ thể
  của từng artifact.
- **Chặn sinh báo cáo theo cơ chế fail-closed:** Stage 4 kiểm trực tiếp các cờ
  evidence và số đếm kỳ vọng từ Stage 1–3 trước khi ghi output. Test âm xác nhận
  feature, prediction hoặc replay baseline sai sẽ khiến script dừng, không thể
  xuất báo cáo PASS.
- **Đổi equity curve sang đường bậc thang:** cả 7 trace Plotly giữ nguyên giữa
  các `close_time` và chỉ nhảy khi ghi nhận R của lệnh đóng; số đường, phạm vi dữ
  liệu, endpoint CSV và kết quả báo cáo không đổi.
- **Hoàn thiện báo cáo để gửi nhóm trưởng:**
  `docs/BAO_CAO_KET_QUA_HOLDOUT.md` nay được sinh từ các artifact JSON/CSV đã
  commit của Stage 1–3. Báo cáo gồm cấu hình train cuối, bảng phân loại và tài
  chính, sweep holdout 20–80%, link biểu đồ, trạng thái kiểm chứng và danh mục
  output tự cập nhật.
- **Giới hạn phát biểu tái lập đúng bằng chứng:** hai lượt độc lập trên máy sinh
  artifact có 25,008 prediction giống hệt từng byte (`max abs difference =
  0.0`), trong khi hash hai file `.cbm` khác nhau. Kết quả teammate trên Windows
  (prediction trong sai số `1e-15`) được ghi là bằng chứng hỗ trợ, không dùng để
  kết luận `thread_count` là nguyên nhân duy nhất của mọi sai lệch liên máy.
- **Làm hai biểu đồ Plotly tái sinh ổn định:** cố định `div_id` trong HTML để
  chạy lại Stage 4 không tạo diff giả chỉ vì UUID ngẫu nhiên.
- **Bổ sung test nhất quán** cho bằng chứng reproducibility, bảng holdout đã làm
  tròn và ID biểu đồ deterministic. Bộ test lúc đó pass 5/5 (nay là 13 test,
  tính đến 2026-09-15).
- **Sửa hướng dẫn cài mới** thành `uv sync --extra dev`, bảo đảm có `pytest`
  trước khi chạy lệnh test trong README.
- **Valid lại toàn bộ quy trình holdout:** 25,008/25,008 khóa cũ và
  575,184/575,184 ô feature khớp; purge và embargo cắt 0 dòng; hai lượt train có
  prediction giống hệt; replay baseline khớp 5,028/5,028 lệnh holdout; metrics
  Stage 3 và báo cáo Stage 4 sinh lại không đổi.

### 2026-09-12 — Khôi phục artifact nhánh bàn giao, sửa biểu đồ 1, ghi deviation

- **Khôi phục** `data/processed/dataset_catboost.csv`,
  `outputs/catboost_training/`, `outputs/backtest/` và `outputs/verification/`
  về đúng bản bàn giao (trước đó đã bị sinh lại trên Linux), để Bảng 1/2 và
  biểu đồ 1 dùng đúng số nhánh đã niêm phong (thay thế ngày 2026-09-15: khối
  chính Bảng 1/2 nay dùng cây canonical `thread_count=1`; số bàn giao đóng băng
  trở thành khối tham chiếu có dán nhãn).
- **Sửa** biểu đồ 1 (`equity-curve-chunk2-5-top50.html`): nay dựng từ file
  `backtest_scored_universe.csv` bàn giao đã khôi phục, nên 4 điểm cuối của
  nhánh khớp Bảng 2 (+6,998.3 / +4,724.3 / +2,207.3 / +2,253.7 R)
  (thay thế ngày 2026-09-15: biểu đồ 1 dựng lại từ vũ trụ đã chấm điểm
  canonical `thread_count=1`
  `outputs/step4_thread1/backtest/backtest_scored_universe.csv`, endpoint
  +6.937,53 / +4.752,04 / +2.148,31 / +2.119,48 R).
- **Sửa** bảng file của báo cáo Stage 4 (đúng tên script và kích thước) và đồng
  bộ `outputs/holdout/stage4/implementation_decisions.md` với báo cáo.
- **Ghi rõ** deviation `thread_count=1` so với danh sách hyperparameter đã chốt
  — đây là tham số kỹ thuật, không phải hyperparameter mô hình.

### 2026-09-12 — Ghim `thread_count=1` (giảm một nguồn sai lệch số học)

- **Quan sát ban đầu:** CatBoost đặt `random_seed=42` nhưng không đặt
  `thread_count`, nên số luồng thực thi phụ thuộc cấu hình máy. Đây là một cơ
  chế có thể làm thay đổi thứ tự phép toán dấu phẩy động. Dữ liệu hiện lưu cho
  thấy các lượt chạy trước từng lệch, nhưng lúc đó không có thí nghiệm đối
  chứng chỉ thay `thread_count`; do đó không tuyên bố đây là nguyên nhân duy
  nhất (thay thế ngày 2026-09-15: thí nghiệm có kiểm soát nay đã commit trong
  `outputs/thread_count_sensitivity/` và cho thấy đổi `thread_count` làm thay
  đổi kết quả train trên máy này).
- **Fix:** ghim `thread_count=1` trong
  `src/citd_ml/training/train_catboost.py` và
  `scripts/holdout_stage2_train.py`.
- **Chạy lại** toàn bộ pipeline và holdout. Holdout ROC-AUC/F1 =
  **0.6046 / 0.4022**; holdout top-50% = **−36.55 R** (profit factor 0.972)
  so với baseline **+245.93 R** (profit factor 1.099).

### 2026-09-12 — Môi trường uv + chạy lại toàn bộ pipeline

- **Môi trường:** cài và ghim dự án bằng `uv sync` (Python 3.12.14,
  catboost 1.2.10, scikit-learn 1.9.0, pandas 3.0.5, numpy 2.5.2,
  plotly 7.0.0) và commit `uv.lock`.
- **Chạy lại toàn bộ pipeline:** `build_dataset → verify_dataset →
  train_models → run_backtest → verify_pipeline` đều pass, cộng thêm 4 giai
  đoạn holdout. Các artifact bước 4 sinh lại sau đó đã được khôi phục về bản
  bàn giao (xem mục trên); chỉ `outputs/holdout/` giữ kết quả chạy mới.
- **Chạy lại holdout:** các giai đoạn holdout đã chạy lại; số cuối cùng nằm ở
  mục phía trên (số phân loại/tài chính được cập nhật lần nữa khi ghim
  `thread_count`).
- **Sửa lỗi:**
  - `scripts/holdout_stage4_report.py` nay điền dòng Holdout của Bảng 1 từ
    `outputs/holdout/stage3/stage3_report.json` thay vì giá trị hard-code dễ
    lệch.
  - `scripts/holdout_stage1_split.py` bỏ `set_index(..., verify_integrity=True)`
    đã bị pandas deprecate, thay bằng kiểm tra trùng khóa tường minh.
  - Sửa link biểu đồ hỏng trong `docs/BAO_CAO_KET_QUA_HOLDOUT.md`
    (`../work/stage4_results/...` → `../outputs/holdout/stage4/...`).
- **Đã xóa:** `docs/legacy/` (bản stage-3 lưu trữ trước refactor). Bản chuẩn
  nằm ở `scripts/holdout_stage3_backtest.py` từ sau refactor và đã sinh toàn bộ
  artifact stage-3 đang commit, nên bản legacy là dư thừa.

## Giấy phép

Phát hành theo giấy phép MIT — xem [`LICENSE`](LICENSE).
