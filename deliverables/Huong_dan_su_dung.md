# Hướng dẫn sử dụng

Đồ án: **Lọc tín hiệu giao dịch BTCUSD bằng CatBoost — đo thiên lệch do cách chia dữ liệu**
Nhóm 11 — Project AI-UIT

---

## 1. Bài nộp gồm những gì

```
[Project AI-UIT] - Nhóm 11/
├── Danh sách nhóm.xlsx
├── Báo cáo/
│   ├── Scientific report (Word)
│   ├── Technical report (Word)
│   └── Slides (PowerPoint)
└── Chuong trinh/
    ├── Code/                      ← toàn bộ mã nguồn + dữ liệu đã xử lý + kết quả
    ├── Demo/                      ← biểu đồ kết quả (PNG + HTML tương tác)
    └── Hướng dẫn sử dụng.md       ← file này
```

Xem nhanh kết quả mà **không cần cài đặt gì**: mở các file trong `Demo/`.
File `.png` xem được ngay; file `.html` mở bằng trình duyệt để rê chuột đọc số
từng điểm trên đường equity.

---

## 2. Yêu cầu môi trường

| Thành phần | Phiên bản đã kiểm chứng |
|---|---|
| Python | 3.12 (bản sinh artifact: 3.12.14) |
| catboost | 1.2.10 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |
| numpy | 2.5.2 |
| plotly | 7.0.0 |
| matplotlib | 3.11.2 |

Toàn bộ pin phiên bản nằm trong `Code/pyproject.toml`, `Code/requirements.txt`
và `Code/uv.lock`.

---

## 3. Cài đặt

Mở terminal tại thư mục `Chuong trinh/Code/`.

### Cách A — dùng `uv` (khuyến nghị)

```bash
uv sync --extra dev
uv run python -c "import catboost, sklearn, pandas, numpy, plotly; print('ok')"
uv run pytest
```

`uv sync` đọc `uv.lock` nên mọi máy cài ra đúng cùng một bộ phiên bản.

### Cách B — `venv` + `pip`

```bash
python3.12 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest
```

Kết quả mong đợi của `pytest`: **13 passed**.

---

## 4. Dữ liệu

Thư mục `Code/data/`:

| Đường dẫn | Có trong bài nộp? | Ghi chú |
|---|---|---|
| `data/processed/dataset_catboost.csv` | ✅ | 25,008 dòng × 31 cột — dataset train |
| `data/processed/tradelist_pyramid_local.csv` | ✅ | 30,040 lệnh baseline |
| `data/processed/triple_barrier_labels.csv` | ✅ | nhãn triple-barrier |
| `data/raw/BTCUSD_m1_2018_to_now.csv` | ❌ | **~209 MB, đã lược bỏ khỏi bài nộp** |

Vì đã có sẵn `data/processed/` và toàn bộ `outputs/`, bạn **xem được mọi kết quả
và chạy lại phần train** mà không cần file thô.

File thô chỉ cần khi muốn (a) sinh lại dataset từ đầu, hoặc (b) chạy backtest —
vì mọi backtest đều replay chiến lược trên lịch sử M1. Định dạng và cách khôi
phục file thô được mô tả trong `Code/data/raw/README.md`. Sau khi có file, đặt
đúng tại `Code/data/raw/BTCUSD_m1_2018_to_now.csv`.

---

## 5. Chạy thử nhanh (không cần file thô)

```bash
# Kiểm tra tính toàn vẹn dataset — in ra bảng PASS/FAIL, exit code 0 là đạt
uv run python scripts/verify_dataset.py

# Train lại 4 nhánh chia dữ liệu (~vài phút), ghi vào thư mục mới để không đè artifact
uv run python scripts/train_models.py --output-dir outputs/thu_nghiem/catboost_training
```

> ⚠️ **Không ghi đè artifact đã niêm phong.** `outputs/catboost_training/`,
> `outputs/backtest/` và `outputs/verification/reproducibility.json` là kết quả
> bàn giao giữ nguyên từng byte. Luôn truyền `--output-dir` / `--oof-dir` trỏ
> sang thư mục mới khi chạy thử.

---

## 6. Chạy toàn bộ pipeline (cần file thô)

```bash
# 1. Sinh dataset từ lịch sử M1
uv run python scripts/build_dataset.py

# 2. Kiểm tra dataset
uv run python scripts/verify_dataset.py

# 3. Train 4 nhánh
uv run python scripts/train_models.py --output-dir outputs/step4_thread1/catboost_training

# 4. Backtest + sweep 20–80%
uv run python scripts/run_backtest.py \
    --output-dir outputs/step4_thread1/backtest \
    --oof-dir   outputs/step4_thread1/catboost_training

# 5. Kiểm chứng tái lập (chạy 2 lần, so từng byte, ~10 phút)
uv run python scripts/verify_pipeline.py \
    --train-dir    outputs/step4_thread1/catboost_training \
    --backtest-dir outputs/step4_thread1/backtest \
    --manifest     outputs/step4_thread1/verification/reproducibility.json
```

### Quy trình holdout niêm phong

> Holdout đã **đóng** từ 2026-09-18. Các lệnh dưới chỉ để tài liệu hoá quy trình
> lịch sử, không chạy thêm lần nào nữa.

```bash
uv run python scripts/holdout_stage1_split.py        # tách holdout
uv run python scripts/holdout_stage2_train.py --run-id 1
uv run python scripts/holdout_stage2_train.py --run-id 2
uv run python scripts/holdout_stage2_train.py --verify
uv run python scripts/holdout_stage3_backtest.py     # chấm điểm + backtest
uv run python scripts/holdout_stage4_report.py       # bảng + biểu đồ
uv run python scripts/holdout_stage5_repro_check.py  # so run 1 vs run 2
```

---

## 7. Đọc kết quả ở đâu

| Cần gì | Mở file nào |
|---|---|
| Tổng quan dự án (tiếng Việt) | `Code/README_VI.md` |
| Tổng quan dự án (tiếng Anh) | `Code/README.md` |
| Báo cáo kết quả holdout — số liệu cuối | `Code/docs/BAO_CAO_KET_QUA_HOLDOUT.md` |
| Báo cáo bước 4 — so 4 cách chia | `Code/docs/BAO_CAO_KET_QUA_STEP4.md` |
| Cơ sở chọn tham số triple barrier | `Code/docs/bao_cao_hyperparameter_triple_barrier.md` |
| Giải thích 23 feature | `Code/docs/bao_cao_feature.md` |
| Ý tưởng gốc của đồ án | `Code/docs/tom_tat_idea_goc.md` |
| Bảng số dạng CSV | `Code/outputs/holdout/stage4/table*.csv` |
| Biểu đồ equity | `Demo/` hoặc `Code/outputs/holdout/stage4/*.html` |

---

## 8. Lưu ý quan trọng

- **Không đưa cột `META` vào model.** `leg` và `entry_vs_base_R` mô tả cấu trúc
  pyramid chứ không phải thị trường; đưa vào sẽ rò rỉ quan hệ cùng họ lệnh.
- **Tập lệnh ứng viên là cố định.** Lọc bỏ một lệnh không được làm đổi tín hiệu
  phía sau — backtest giữ một chiến lược "bóng" đúng vì lý do này.
- **Không sửa dataset hay `outputs/` đã commit** khi thử nghiệm. Ghi kết quả mới
  sang thư mục mới để artifact niêm phong còn giá trị đối chứng.
- **`thread_count=1` là thiết lập kỹ thuật, không phải hyperparameter mô hình.**
  Trên máy sinh artifact, đổi `thread_count` khi train làm đổi **mọi** prediction;
  khi chỉ suy luận trên model đã fit thì không đổi. Chi tiết trong
  `Code/outputs/thread_count_sensitivity/`.
