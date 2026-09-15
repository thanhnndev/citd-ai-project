# Audit tái lập độc lập — CITD ML (BTCUSD pyramid + CatBoost)

> Phiên kiểm toán độc lập, chạy lại từ artifact đã commit mà không tin log sinh trước đó.
> Mọi bước chạy lại ghi vào `/tmp/opencode/audit/`; không ghi đè `outputs/`, `data/`, `docs/`
> (ngoại lệ duy nhất: tài liệu này). Không chạy `git add/commit/push`.

## 0. Định danh, môi trường, phạm vi

| Thành phần | Giá trị |
|---|---|
| HEAD lúc kiểm toán | `93cdf503f5e2ddc1aeceec41389be569d1987d3b` (`93cdf50` — *chore(evidence): refresh run-history HEAD after documentation commit*) |
| Trạng thái git trước kiểm | `git status --porcelain` rỗng |
| Python | `.venv/bin/python` — `3.12.14 (main, Aug 25 2026, 14:00:49) [Clang 22.1.3]` |
| OS (`platform`) | `Linux-7.2.5-1-cachyos-x86_64-with-glibc2.44` |
| catboost / scikit-learn | `1.2.10` / `1.9.0` |
| pandas / numpy | `3.0.5` / `2.5.2` |
| plotly / matplotlib | `7.0.0` / `3.11.2` |

Ba báo cáo môi trường đã commit (`train_run1_report.json`, `train_run2_report.json`,
`stage5_repro_report.json`) ghi đúng cùng bộ phiên bản và `platform` như trên.

Các script kiểm toán phụ trợ nằm trong `/tmp/opencode/audit/`:
`compare_csv_json.py`, `compare_markdown.py`, `compare_manifest.py`,
`check_links.py`, `check_claims.py`, `independent_recompute.py`.
Script `independent_recompute.py` không import `src/citd_ml`; chỉ dùng
pandas/numpy/scikit-learn và tự cài lại luật chia khúc.

---

## 1. Bước 1 — Git sạch và artifact đóng băng

```bash
git status --porcelain          # rỗng (trước kiểm)
git rev-parse HEAD              # 93cdf503f5e2ddc1aeceec41389be569d1987d3b
git diff --stat HEAD -- data/processed outputs/catboost_training outputs/backtest \
    outputs/verification/reproducibility.json outputs/holdout/stage1 \
    outputs/holdout/stage2 outputs/holdout/stage3
```

- `git diff --stat` cho đúng các đường dẫn yêu cầu: **rỗng** (exit 0, không dòng nào).
- Mở rộng `outputs/step4_thread1`, `outputs/holdout/repro`, `outputs/verification`,
  `outputs/thread_count_sensitivity`: cũng **rỗng**.
- Băm SHA-256 **93 file** thuộc các cây đóng băng trước khi chạy
  (`frozen_files_before.txt`) và so lại sau toàn bộ các bước: **không file nào đổi**
  (kể cả sau khi `verify_pipeline` chạy xong).
- HEAD không đổi trong suốt quá trình: `93cdf50`.

**Kết luận bước 1: PASS.**

## 2. Bước 2 — Chạy lại Stage 3 cho model run-2

```bash
.venv/bin/python scripts/holdout_stage3_backtest.py --run-id 2 \
    --out-dir /tmp/opencode/audit/stage3_run2
```

Đối chiếu 11 file với `outputs/holdout/repro/run2/stage3/` (so elementwise từng cột +
SHA-256 từng file):

| File | SHA-256 trùng | max abs diff cột số | Cột chuỗi/thời gian |
|---|---|---|---|
| `holdout_scored.csv` (5.028 dòng) | ✔ | 0.0 | khớp tuyệt đối |
| `holdout_fixed_trade_universe_scored.csv` (5.028 dòng) | ✔ | 0.0 | khớp tuyệt đối |
| `holdout_trades_top20.csv` | ✔ | 0.0 | khớp tuyệt đối |
| `holdout_trades_top30.csv` | ✔ | 0.0 | khớp tuyệt đối |
| `holdout_trades_top40.csv` | ✔ | 0.0 | khớp tuyệt đối |
| `holdout_trades_top50.csv` (2.514 dòng) | ✔ | 0.0 | khớp tuyệt đối |
| `holdout_trades_top60.csv` | ✔ | 0.0 | khớp tuyệt đối |
| `holdout_trades_top70.csv` | ✔ | 0.0 | khớp tuyệt đối |
| `holdout_trades_top80.csv` | ✔ | 0.0 | khớp tuyệt đối |
| `stage3_backtest_summary.csv` (8 dòng) | ✔ | 0.0 | khớp tuyệt đối |
| `stage3_report.json` | ✔ | deep-equal từng khóa | — |

**11/11 file byte-identical**, `max_abs_diff = 0.0` trên mọi cột số, JSON trùng cấu trúc
và giá trị. Phân loại chấm lại từ model run-2:
`roc_auc = 0.6045544538928682`, `f1_at_0_5 = 0.40220723482526055`
— trùng chính xác `stage3_report.json` đã commit. Không có thành phần ngẫu nhiên;
thời gian chạy ~4,4 giây. `stage3_report.json` xác nhận
`baseline_comparison.passed = true`, `actual_holdout_trades = reference_holdout_trades = 5028`,
`mismatched_fields = []`.

**Kết luận bước 2: PASS.**

## 3. Bước 3 — Chạy lại Stage 4 cho run-1

```bash
.venv/bin/python scripts/holdout_stage4_report.py \
    --out-dir /tmp/opencode/audit/stage4_run1 \
    --report /tmp/opencode/audit/BAO_CAO_run1.md
```

Đối chiếu `/tmp/opencode/audit/stage4_run1/` với `outputs/holdout/stage4/`:

| File | Kết quả so byte | Ghi chú |
|---|---|---|
| `equity-curve-chunk2-5-top50.png` | byte-identical | 94.055 B |
| `equity-curve-holdout-top50.png` | byte-identical | 81.513 B |
| `equity-curve-chunk2-5-top50.html` | byte-identical | 5.406.759 B |
| `equity-curve-holdout-top50.html` | byte-identical | 4.434.195 B |
| `holdout_sweep_20_80.csv` | byte-identical | số khớp elementwise, diff = 0.0 |
| `table1_classification_metrics.csv` | byte-identical | 13 dòng |
| `table2_financial_metrics_top50.csv` | byte-identical | 12 dòng |
| `table3_branch_sweep_20_80.csv` | byte-identical | 28 dòng |
| `stage4_tables.md` | byte-identical | 5.838 B |
| `implementation_decisions.md` | byte-identical | 2.566 B |
| `stage4_manifest.json` | khác đúng 1 dòng | xem bên dưới |

**Khác biệt duy nhất — `stage4_manifest.json`:** trường `new_files` của bản commit liệt kê
11 tên, gồm cả `"stage4_manifest.json"`; bản chạy vào thư mục trống liệt kê 10 tên (không có
chính nó). Nguyên nhân: script quét `out_dir` **trước khi** ghi manifest, nên khi chạy vào thư
mục đã tồn tại (bản commit) nó thấy file cũ, còn thư mục mới thì không. Toàn bộ trường số
`chart1_rows` / `chart2_rows` / `png_export` giống hệt. Đây là khác biệt tự-tham-chiếu của
bước ghi file, **không phải sai lệch dữ liệu**.

**Báo cáo Markdown:** `docs/BAO_CAO_KET_QUA_HOLDOUT.md` (26.588 B) vs
`/tmp/opencode/audit/BAO_CAO_run1.md` (28.280 B). File thô khác 84 dòng — đúng bằng
42 liên kết × 2 dòng (`<`/`>`), tức mọi khác biệt byte nằm trong đích liên kết tương đối.
Sau khi chuẩn hóa (thay mỗi đích link bằng đường dẫn tuyệt đối đã resolve):
**0 dòng nội dung khác biệt**, 42/42 liên kết cùng thứ tự. Trong đó:
- 38 liên kết resolve về **cùng đích tuyệt đối** (nhờ cơ chế `relative_link` giữ nhãn theo
  `PROJECT_ROOT` và tính `relpath` theo vị trí báo cáo);
- 4 liên kết biểu đồ ở Phần 2 khác vị trí (`stage4_run1/…` so với
  `../outputs/holdout/stage4/…`) — nhưng chính 4 file đó đã chứng minh byte-identical ở bảng trên.

**Kết luận bước 3: PASS** (1 khác biệt tự-tham-chiếu đã giải thích; báo cáo trùng nội dung
tuyệt đối sau chuẩn hóa).

## 4. Bước 4 — `verify_pipeline` trên cây canonical `thread_count=1`

```bash
.venv/bin/python scripts/verify_pipeline.py \
    --train-dir outputs/step4_thread1/catboost_training \
    --backtest-dir outputs/step4_thread1/backtest \
    --manifest /tmp/opencode/audit/reproducibility.json
```

Kết quả command: `EXIT=0`, in `PASS: bằng chứng kiểm chứng được lưu tại
/tmp/opencode/audit/reproducibility.json`, thời gian ~4 phút 5 giây. Cả hai lượt train và
hai lượt backtest đều in `PASS exact repeat + saved CSV` cho từng file; frame chưa làm tròn
`assert_frame_equal(check_exact=True)` giống hệt, và bytes CSV sinh lại trùng bytes CSV đã commit.

So `/tmp/opencode/audit/reproducibility.json` với
`outputs/step4_thread1/verification/reproducibility.json` — **deep-diff toàn bộ JSON**:
chỉ có **đúng 3 trường khác** (theo thiết kế, không mang thông tin khoa học):

| Trường | Scratch | Committed |
|---|---|---|
| `command` | `--manifest /tmp/opencode/audit/reproducibility.json` | `--manifest outputs/step4_thread1/verification/reproducibility.json` |
| `started_utc` | `2026-09-15T10:27:33…` | `2026-09-15T10:18:37…` |
| `completed_utc` | `2026-09-15T10:31:37…` | `2026-09-15T10:22:36…` |

Mọi trường còn lại trùng khít:

| Hạng mục | Kết quả |
|---|---|
| `status` | `passed` == `passed` |
| `output_dirs` | `outputs/step4_thread1/catboost_training` + `outputs/step4_thread1/backtest` |
| `packages` | catboost 1.2.10 / scikit-learn 1.9.0 / pandas 3.0.5 / numpy 2.5.2 |
| `model_params` | `iterations=1000`, `learning_rate=0.05`, `depth=6`, `l2_leaf_reg=3.0`, `auto_class_weights=Balanced`, `eval_metric=AUC`, `random_seed=42`, `thread_count=1` |
| `training.output_sha256` (6 hash) | trùng đủ khóa và giá trị, `runs=2`, `matches_saved_csv_bytes=true` |
| `backtest.output_sha256` (8 hash) | trùng đủ khóa và giá trị, `runs=2`, `matches_saved_csv_bytes=true` |
| `source_sha256` (16 hash) | trùng đủ 16/16 |
| `python` | chuỗi giống hệt |

**Kết luận bước 4: PASS** — manifest tái lập được ngoài 3 trường thời gian/đường dẫn manifest;
16 hash nguồn và 14 hash output trùng tuyệt đối.

## 5. Bước 5 — Kiểm tra toàn bộ liên kết/ảnh Markdown của báo cáo

```bash
.venv/bin/python /tmp/opencode/audit/check_links.py docs/BAO_CAO_KET_QUA_HOLDOUT.md \
    /tmp/opencode/audit/step5_links.json
```

**42 liên kết nội bộ** (40 file + 2 thư mục), **0 liên kết hỏng**, mọi đích tồn tại và không rỗng,
không có link ngoài. Danh sách 38 đích duy nhất (link lặp giữa Phần 2 và Phần 5 hiển thị `×2`):

| # | Đích (resolve từ `docs/`) | Loại | Kích thước / số file | Trạng thái |
|---|---|---|---|---|
| 1 | `../outputs/holdout/stage4/equity-curve-chunk2-5-top50.png` ×2 | file | 94.055 B | OK |
| 2 | `../outputs/holdout/stage4/equity-curve-holdout-top50.png` ×2 | file | 81.513 B | OK |
| 3 | `../outputs/holdout/stage4/equity-curve-chunk2-5-top50.html` ×2 | file | 5.406.759 B | OK |
| 4 | `../outputs/holdout/stage4/equity-curve-holdout-top50.html` ×2 | file | 4.434.195 B | OK |
| 5 | `../outputs/holdout/stage1/dataset_catboost_full_regenerated.csv` | file | 9.800.880 B | OK |
| 6 | `../outputs/holdout/stage1/dataset_catboost_holdout.csv` | file | 1.648.262 B | OK |
| 7 | `../outputs/holdout/stage1/stage1_validation_report.json` | file | 888 B | OK |
| 8 | `../outputs/holdout/stage2/catboost_final_holdout_run1.cbm` | file | 1.129.392 B | OK |
| 9 | `../outputs/holdout/stage2/catboost_final_holdout_run2.cbm` | file | 1.129.392 B | OK |
| 10 | `../outputs/holdout/stage2/stage2_reproducibility_report.json` | file | 722 B | OK |
| 11 | `../outputs/holdout/stage2/train_predictions_run1.npy` | file | 200.192 B | OK |
| 12 | `../outputs/holdout/stage2/train_predictions_run2.npy` | file | 200.192 B | OK |
| 13 | `../outputs/holdout/stage2/train_run1_report.json` | file | 1.949 B | OK |
| 14 | `../outputs/holdout/stage2/train_run2_report.json` | file | 1.949 B | OK |
| 15 | `../outputs/holdout/stage3/holdout_fixed_trade_universe_scored.csv` | file | 2.263.531 B | OK |
| 16 | `../outputs/holdout/stage3/holdout_scored.csv` | file | 1.661.606 B | OK |
| 17 | `../outputs/holdout/stage3/holdout_trades_top20.csv` | file | 454.493 B | OK |
| 18 | `../outputs/holdout/stage3/holdout_trades_top30.csv` | file | 680.410 B | OK |
| 19 | `../outputs/holdout/stage3/holdout_trades_top40.csv` | file | 906.297 B | OK |
| 20 | `../outputs/holdout/stage3/holdout_trades_top50.csv` | file | 1.132.050 B | OK |
| 21 | `../outputs/holdout/stage3/holdout_trades_top60.csv` | file | 1.358.175 B | OK |
| 22 | `../outputs/holdout/stage3/holdout_trades_top70.csv` | file | 1.584.187 B | OK |
| 23 | `../outputs/holdout/stage3/holdout_trades_top80.csv` | file | 1.810.896 B | OK |
| 24 | `../outputs/holdout/stage3/stage3_backtest_summary.csv` | file | 654 B | OK |
| 25 | `../outputs/holdout/stage3/stage3_report.json` | file | 3.288 B | OK |
| 26 | `../outputs/holdout/stage4/holdout_sweep_20_80.csv` | file | 366 B | OK |
| 27 | `../outputs/holdout/stage4/implementation_decisions.md` | file | 2.566 B | OK |
| 28 | `../outputs/holdout/stage4/stage4_manifest.json` | file | 856 B | OK |
| 29 | `../outputs/holdout/stage4/stage4_tables.md` | file | 5.838 B | OK |
| 30 | `../outputs/holdout/stage4/table1_classification_metrics.csv` | file | 750 B | OK |
| 31 | `../outputs/holdout/stage4/table2_financial_metrics_top50.csv` | file | 1.004 B | OK |
| 32 | `../outputs/holdout/stage4/table3_branch_sweep_20_80.csv` | file | 1.703 B | OK |
| 33 | `../outputs/step4_thread1` | thư mục | 18 file | OK |
| 34 | `../outputs/thread_count_sensitivity/thread_count_sensitivity.json` | file | 15.560 B | OK |
| 35 | `../outputs/verification/holdout_run_history.json` | file | 49.504 B | OK |
| 36 | `../outputs/holdout/repro/stage5_repro_report.json` | file | 28.153 B | OK |
| 37 | `../outputs/step4_thread1/verification/reproducibility.json` | file | 5.286 B | OK |
| 38 | `../outputs/holdout/repro/run2` | thư mục | 23 file | OK |

**Kết luận bước 5: PASS.**

## 6. Bước 6 — Đối chiếu bảng "Phần 4 — Kiểm chứng" với artifact gốc

```bash
.venv/bin/python /tmp/opencode/audit/check_claims.py <repo> /tmp/opencode/audit/step6_claims.json
```

| Claim trong báo cáo | Bằng chứng gốc | Kết quả |
|---|---|---|
| Stage 5 `PASS` | `outputs/holdout/repro/stage5_repro_report.json`: `"status": "PASS"`, `predictions_exactly_equal: true`, `max_abs_diff: 0.0`, 4 biểu đồ `bytes_identical: true` | PASS |
| Holdout SHA-256 không đổi khi train/chấm điểm | `train_run1_report.json` và `train_run2_report.json`: `holdout_sha256_before == holdout_sha256_after == 7be58aad56927e2060a66a99b947cad63b75597c26fc4964f68bfeb433914037`; hash này khớp file `dataset_catboost_holdout.csv` hiện tại | PASS |
| `platform` + `plotly` có trong báo cáo Stage 2 | Cả hai `train_run*_report.json` có `versions.platform = Linux-7.2.5-1-cachyos-x86_64-with-glibc2.44`, `versions.plotly = 7.0.0` | PASS |
| Bảng môi trường trong báo cáo khớp Stage 2 | 7/7 giá trị (platform, python, catboost, sklearn, pandas, numpy, plotly) xuất hiện đúng trong `train_run1_report.json` | PASS |
| Bổ sung: dòng Stage 1 | `stage1_validation_report.json`: 30.040 dòng tái sinh, 25.008/25.008 khóa cũ, 0 thiếu, 575.184 ô feature so sánh, 0 lệch, 5.028 dòng holdout; `pre_holdout_rows = 25.012 = 25.008 + 4` lệnh biên | PASS |

**Kết luận bước 6: PASS.**

## 7. Bước 7 — Tính độc lập ba số then chốt

```bash
.venv/bin/python /tmp/opencode/audit/independent_recompute.py <repo> \
    /tmp/opencode/audit/step7_recompute.json
```

Không import `src/citd_ml`; tự cài lại luật chia khúc theo `row_id` với biên
`[0, 5001, 10003, 15004, 20006, 25008]` (`searchsorted(side="right")`).

| Số kiểm | Tính độc lập | Báo cáo | Δ | Khớp mức làm tròn |
|---|---|---|---|---|
| (a) Holdout ROC-AUC (5.028 dòng) | `0.6045544538928682` | `0.6046` | −4,55e−05 | ✔ (4 chữ số) |
| (a) Holdout F1@0.5 | `0.40220723482526055` | `0.4022` | +7,23e−06 | ✔ (4 chữ số) |
| (b) Top 50% net R — `ceil(5028×0,5) = 2.514` lệnh | `−36.55275499584241` | `−36.55` | −2,75e−03 (mức 2 chữ số) | ✔ (2 chữ số) |
| (b) Baseline holdout net R (5.028 lệnh) | `+245.92968265920814` | `+245.93` | 0,00 | ✔ (2 chữ số) |
| (c) Random K-Fold bỏ khúc 1 — AUC trung bình fold | `0.858181611233993` | `0.8582` | −1,84e−05 | ✔ (4 chữ số) |
| (c) F1@0.5 cùng phạm vi | `0.6493677054961667` | `0.6494` | −3,23e−05 | ✔ (4 chữ số) |

Chi tiết:

- **(a)** `label_positive = 1.410`; giá trị khớp **chính xác từng chữ số** giá trị in-memory
  `classification` trong `stage3_report.json` (0.6045544538928682 / 0.40220723482526055).
- **(b)** Bộ khóa `(origin_bar, entry_bar, leg)` của 2.514 lệnh độc lập **trùng khít**
  `holdout_trades_top50.csv` (2.514/2.514). Net R tổng từ file top50 cũng bằng
  `−36.55275499584241`. Sai khác ~4,8e−11 so với số in-memory
  `−36.552754995889984` trong `stage3_report.json` là do CSV lưu `float_format="%.12g"`
  — nhỏ hơn ngưỡng làm tròn công bố 10 bậc độ lớn.
- **(c)** Sau khi bỏ khúc 1: 5 fold, 20.007 dòng; AUC từng fold
  `0.8503047324906408 / 0.862908874199241 / 0.8626728582169105 / 0.8592308221147236 /
  0.8557907691484489`; giá trị này khớp `metrics_chunk2_5.csv` của cây canonical
  (tự tính lại, không đọc bảng).

**Kết luận bước 7: PASS — cả ba số khớp báo cáo tới đúng mức làm tròn công bố.**

## 8. Bước 8 — Test suite

```bash
.venv/bin/python -m pytest -q
# .............  [100%]
# 13 passed in 0.02s
```

**Kết luận bước 8: PASS — 13/13 test.**

## 9. Trạng thái git sau kiểm toán

- Trước kiểm: `git status --porcelain` **rỗng**.
- Trong suốt các bước 1–8: **93 file đóng băng giữ nguyên SHA-256**, HEAD giữ nguyên.
- Sau các bước kiểm (17:31–17:32), `git status` xuất hiện **một file untracked do phiên khác
  tạo song song**: `docs/CODE_BEST_PRACTICE_AUDIT.md` (mtime `2026-09-15 17:32:18`,
  header ghi rõ là audit read-only của một phiên khác). Không lệnh nào của phiên này tham
  chiếu tên file đó và không script nào trong repo sinh ra nó. File này **không bị sửa/xóa**
  (là việc của phiên khác); nó không ảnh hưởng tới bất kỳ số liệu nào ở trên.
- Tài liệu này (`docs/INDEPENDENT_REPRO_AUDIT.md`) là file duy nhất phiên kiểm toán tạo ra.
  Không chạy `git add`, `git commit`, `git push`.

## 10. Tổng kết và phán quyết

| Bước | Nội dung | Kết quả |
|---|---|---|
| 1 | Git sạch + artifact đóng băng byte-equal với commit | **PASS** |
| 2 | Chạy lại Stage 3 run-2 → 11/11 file byte-identical, max abs diff = 0.0 | **PASS** |
| 3 | Chạy lại Stage 4 run-1 → 4 biểu đồ + 4 CSV + 2 MD byte-identical; báo cáo MD trùng nội dung sau chuẩn hóa; manifest khác 1 dòng tự-tham-chiếu | **PASS** |
| 4 | `verify_pipeline` PASS; 6/6 hash train + 8/8 hash backtest + 16/16 hash nguồn trùng; chỉ khác 3 trường thời gian/đường dẫn manifest | **PASS** |
| 5 | 42/42 liên kết báo cáo hợp lệ, 0 hỏng | **PASS** |
| 6 | Claim bảng kiểm chứng khớp artifact gốc (Stage 5, SHA holdout, platform/plotly) | **PASS** |
| 7 | Ba số then chốt tính độc lập khớp báo cáo tới mức làm tròn | **PASS** |
| 8 | `pytest -q` → 13 passed | **PASS** |

**Phán quyết cuối cùng: TÁI LẬP ĐƯỢC (REPRODUCIBLE) — 8/8 bước PASS.**

Không phát hiện sai lệch số liệu nào. Hai điểm cần ghi nhận để minh bạch, cả hai đều đã
được giải thích và không ảnh hưởng kết luận:

1. `outputs/holdout/stage4/stage4_manifest.json` khác bản chạy mới đúng một mục tự liệt kê
   trong `new_files` (`stage4_manifest.json`), do script quét thư mục trước khi ghi file và
   thư mục đích cũ đã tồn tại. Các trường số và hash của mọi artifact khác đều trùng.
2. Chênh lệch `4,8e−11` giữa net R đọc từ CSV (`%.12g`) và số in-memory trong `stage3_report.json`
   là do độ chính xác lưu CSV, không phải sai lệch logic.

Ba số quan trọng nhất đã được tính độc lập và khớp báo cáo:

- **Holdout ROC-AUC = 0.6046** (chính xác `0.6045544538928682`), **F1@0.5 = 0.4022**
  (`0.40220723482526055`).
- **Top 50% net R (2.514 lệnh) = −36.55 R** (`−36.55275499584241`).
- **Random K-Fold bỏ khúc 1, AUC trung bình fold = 0.8582** (`0.858181611233993`).

---

## Addendum 2026-09-15 — vòng best-practice (sau audit)

> Mục này được thêm sau vòng sửa best-practice cùng ngày (mô tả tại
> `docs/CODE_BEST_PRACTICE_AUDIT.md` §7). Toàn bộ nội dung audit ở trên là
> **snapshot tại HEAD `93cdf50`** và được giữ nguyên, không viết lại; các bảng
> kích thước ở mục 5 vẫn đúng như bằng chứng lịch sử.

Vòng best-practice **cố ý** đổi byte của một số artifact không mang thông tin
số học:

- **`.cbm`:** ghim `allow_writing_files=False` khi train nên
  `catboost_final_holdout_run1.cbm` và `catboost_final_holdout_run2.cbm` đổi
  hash (`09ac7b61…`/`3b6db983…` → `90a6c179…`/`00dc6f07…`) và nay mỗi file là
  **1.129.416 B** (trước: 1.129.392 B). Model binary không phải tiêu chí PASS.
- **PNG:** `metadata={"Software": None}` nên ảnh không còn tag `Software` phụ
  thuộc phiên bản matplotlib; kích thước hiện tại
  `equity-curve-chunk2-5-top50.png` = **93.985 B**,
  `equity-curve-holdout-top50.png` = **81.443 B** (trước: 94.055 B / 81.513 B),
  hai bản run 2 trong `outputs/holdout/repro/run2/stage4/` cùng kích thước.
- **JSON có metadata mới:** `train_run1_report.json` / `train_run2_report.json`
  nay **2.011 B** (trước 1.949 B, thêm `allow_writing_files` và
  `versions.matplotlib`), `thread_count_sensitivity.json` nay **16.782 B**
  (trước 15.560 B).

**Không có số liệu nào thay đổi.** Bằng chứng:

- 8 CSV canonical vẫn byte-identical với commit `7748828`
  (`outputs/step4_thread1/comparison_report.json`:
  `vs_commit_7748828.all_files_bytes_identical = true`); 11 file Stage 3
  `git diff --stat` rỗng; 4 bảng CSV Stage 4 giữ nguyên hash sau khi sinh lại.
- Bảng số trong báo cáo Stage 4 chỉ đổi phần trình bày: escape `\|` ở header
  mục 1.6 và dòng kiểm chứng, nhãn `(run 2)` cho báo cáo run 2; không ô số nào
  đổi.
- `chart1_rows` / `chart2_rows` trong `stage4_manifest.json` không đổi, tức dữ
  liệu đường vốn giữ nguyên.

Hai file bằng chứng đã được **refresh** trong vòng này:

| File | Kích thước hiện tại | Ghi chú |
|---|---|---|
| `outputs/holdout/repro/stage5_repro_report.json` | 28.181 B | `status = PASS`, mọi delta `0.0`, 4/4 biểu đồ (HTML + PNG) byte-identical giữa run 1 và run 2, `manifests_exactly_equal = true` |
| `outputs/step4_thread1/verification/reproducibility.json` | 5.320 B | manifest canonical `verify_pipeline`, `status = passed`, train ×2 + backtest ×2 |

Các kết luận PASS 8/8 bước và ba số then chốt của audit gốc vẫn nguyên giá
trị.
