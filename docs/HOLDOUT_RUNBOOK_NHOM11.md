# RUNBOOK HOLDOUT — NHÓM 11

> Tài liệu tra cứu lịch sử chạy. Holdout đã đóng; **không chạy lại, không chọn lại top-k, không dùng holdout để sửa feature/model**. Các lệnh dưới đây được giữ để giải thích quy trình, không phải lời kêu gọi chạy mới.

## 1. Mục đích và nguyên tắc

Holdout là phép đối chiếu cuối cho một model cuối đã train trên dữ liệu trước mốc `2025-02-08 15:30:00`.

Nguyên tắc:

1. Train chỉ dùng `data/processed/dataset_catboost.csv` đã đóng băng: 25.008 dòng.
2. Không thay feature, hyperparameter hoặc tỷ lệ giữ 50% sau khi xem holdout.
3. Dùng một cấu hình model cuối đã chốt, chấm toàn bộ 5.028 lệnh holdout.
4. Chạy hai lượt độc lập để kiểm tra tái lập, không phải để chọn model “đẹp nhất”.
5. Tách Stage 1–5 để mỗi kiểm tra có input/output và điểm dừng rõ ràng.

## 2. Chuẩn bị

```bash
uv sync --extra dev
```

Cần file:

```text
data/raw/BTCUSD_m1_2018_to_now.csv
```

Lệnh này dùng để tái sinh toàn bộ lịch sử và replay chiến lược. Dataset/artifact đã commit vẫn cho phép kiểm tra kết quả, nhưng backtest cần M1 thô.

## 3. Stage 1 — Tái sinh và tách holdout

```bash
uv run python scripts/build_dataset.py \
    --holdout 2100-01-01 \
    --output outputs/holdout/stage1/dataset_catboost_full_regenerated.csv

uv run python scripts/holdout_stage1_split.py
```

### Vì sao truyền `2100-01-01`?

`build_features.main()` truyền `--holdout` xuống `prepare_data()`, vốn lọc M1 bằng:

```python
m1.index < holdout_start
```

Mốc `2100-01-01` là sentinel ở tương lai để **giữ toàn bộ M1**. Sau đó `holdout_stage1_split.py` mới tách theo `entry_time`:

- `entry_time < 2025-02-08 15:30:00`: phần trước holdout.
- `entry_time >= 2025-02-08 15:30:00`: holdout.

Nếu truyền mốc holdout thật ngay từ `build_dataset`, lệnh đó sẽ không thể tái sinh phần holdout.

### Vì sao có 25.012 dòng trước mốc nhưng train chỉ dùng 25.008?

Bốn lệnh family biên mở trước mốc nhưng chỉ hoàn tất nhãn khi đã có dữ liệu tương lai. Khi tái sinh toàn bộ lịch sử, chúng xuất hiện thành 4 dòng bổ sung. Stage 1 vì vậy kiểm tra:

- Đủ 25.008 khóa `(origin_bar, entry_bar, leg)` của dataset đóng băng.
- 575.184 ô feature trùng tuyệt đối.
- Có 5.028 dòng holdout.
- Chỉ ghi file holdout sau khi mọi kiểm tra đạt.

Bốn dòng biên không thuộc train hoặc holdout: `25.008 + 4 + 5.028 = 30.040`.

## 4. Stage 2 — Train model cuối và kiểm tra lặp

```bash
uv run python scripts/holdout_stage2_train.py --run-id 1
uv run python scripts/holdout_stage2_train.py --run-id 2
uv run python scripts/holdout_stage2_train.py --verify
```

### Vì sao train hai lần?

Hai lượt dùng cùng:

```text
25.008 dòng train
23 feature
MODEL_PARAMS
random_seed=42
thread_count=1
```

Mục tiêu là kiểm tra xác suất có lặp lại trên cùng máy hay không. Tiêu chí PASS là prediction array giống nhau và holdout file không đổi. Hai file `.cbm` có thể khác SHA-256 vì metadata serialize khác nhau.

### Vì sao purge/embargo cuối cùng bằng 0?

Code kiểm tra trước train:

- `label_end_time < HOLDOUT_START`.
- `entry_bar < 199918`.

Tập train đã kết thúc trước biên đủ xa nên không có dòng bị cắt. Nếu kết quả khác 0, Stage 2 dừng và không train.

## 5. Stage 3 — Chấm điểm và backtest

```bash
uv run python scripts/holdout_stage3_backtest.py
```

Stage 3:

1. Load `catboost_final_holdout_run1.cbm`.
2. Chấm 5.028 dòng bằng đúng 23 feature.
3. Kiểm tra hash holdout trước/sau không đổi.
4. Replay M1 từ toàn bộ lịch sử để warm up và tạo baseline holdout.
5. Đối chiếu baseline với `tradelist_pyramid_local.csv`.
6. Merge score vào baseline bằng khóa `(origin_bar, entry_bar, leg)`.
7. Tính baseline và top 20/30/40/50/60/70/80%.

### Vì sao replay từ 2018?

Chiến lược cần lịch sử trước để:

- Tính indicator.
- Warm up 200 bar.
- Tái tạo đúng trạng thái `base_open`, pending leg và stop/trailing.

Chỉ phần lệnh có `open_time >= HOLDOUT_START` mới được tính vào chỉ số holdout.

### Vì sao giữ vũ trụ lệnh cố định?

Bộ lọc chỉ quyết định lệnh nào được tính vào kết quả. Việc loại một lệnh không làm thay đổi các tín hiệu baseline về sau. Nhờ đó baseline, top 20–80% và các cách chia dùng cùng một danh sách ứng viên.

Đây là phép đánh giá offline; top 50% toàn kỳ chưa phải threshold biết được tại thời điểm live.

## 6. Stage 4 — Bảng và biểu đồ

```bash
uv run python scripts/holdout_stage4_report.py
```

Stage 4 tạo:

- Bảng phân loại.
- Bảng tài chính top 50%.
- Sweep 20–80%.
- Bốn biểu đồ equity.
- `implementation_decisions.md`.
- Báo cáo `docs/BAO_CAO_KET_QUA_HOLDOUT.md`.

Stage 4 **fail-closed**: phải có Stage 1–3 evidence hợp lệ, baseline replay đúng, model lặp đúng và bằng chứng reproducibility trước khi xuất báo cáo PASS.

## 7. Run 2 và Stage 5 — Kiểm tra end-to-end

```bash
uv run python scripts/holdout_stage3_backtest.py --run-id 2 \
    --out-dir outputs/holdout/repro/run2/stage3

uv run python scripts/holdout_stage4_report.py \
    --stage3-dir outputs/holdout/repro/run2/stage3 \
    --out-dir outputs/holdout/repro/run2/stage4 \
    --report outputs/holdout/repro/run2/BAO_CAO_holdout_run2.md

uv run python scripts/holdout_stage5_repro_check.py
```

Stage 5 so sánh:

- 5.028 xác suất holdout.
- ROC-AUC/F1.
- Baseline và sweep 20–80%.
- Danh sách khóa lệnh top-k.
- Bốn bảng Stage 4.
- Bốn biểu đồ HTML/PNG.

Kết quả canonical: **PASS**, mọi delta bằng `0.0`, bốn biểu đồ byte-identical trên cùng máy/cùng code. Không suy ra kết luận byte-identical giữa các máy.

## 8. Kết quả cần trích dẫn

| Phép đo | Kết quả |
|---|---:|
| Holdout ROC-AUC | 0,6046 |
| Holdout F1 @0.5 | 0,4022 |
| Baseline holdout | +245,93 R |
| Holdout top 50% | −36,55 R |
| PF baseline / top 50% | 1,0992 / 0,9718 |
| Số holdout train bị purge/embargo | 0 |

## 9. Khi thầy hỏi “vì sao chạy như vậy?”

- **Vì sao full-history trước?** Để tái sinh cả holdout và kiểm tra dataset đóng băng.
- **Vì sao dùng dataset train cũ 25.008?** Đó là hợp đồng dữ liệu của bước trước; 4 dòng biên không được đưa vào train.
- **Vì sao train hai lần?** Để kiểm tra lặp, không để chọn kết quả tốt nhất.
- **Vì sao top 20–80% dùng chung prediction?** Để chỉ thay đổi tỷ lệ giữ, không train lại model.
- **Vì sao chạy lại Stage 3–4?** Để kiểm tra chuỗi đầu–cuối, không chỉ model.
- **Vì sao cần giới hạn top 50%?** 50% được chốt trước; sweep sau holdout chỉ để phân tích độ nhạy.
- **Vì sao còn giới hạn?** Holdout đã bị mở lại nhiều lần trong kiểm tra kỹ thuật; không thể chứng minh single-use hoàn toàn.

## 10. Nguồn truy ngược

- Lệnh và mục đích Stage 1: [`scripts/build_dataset.py:8-15`](../scripts/build_dataset.py), [`src/citd_ml/features/build_features.py:174-213`](../src/citd_ml/features/build_features.py), [`scripts/holdout_stage1_split.py:26-91`](../scripts/holdout_stage1_split.py).
- Lọc M1 theo sentinel: [`src/citd_ml/labeling/triple_barrier.py:42-56`](../src/citd_ml/labeling/triple_barrier.py).
- Train cuối, purge/embargo, train lặp: [`scripts/holdout_stage2_train.py:39-148`](../scripts/holdout_stage2_train.py).
- Score, replay, top-k: [`scripts/holdout_stage3_backtest.py:1-8`](../scripts/holdout_stage3_backtest.py), [`scripts/holdout_stage3_backtest.py:127-165`](../scripts/holdout_stage3_backtest.py).
- Bảng/biểu đồ và fail-closed: [`scripts/holdout_stage4_report.py:67-117`](../scripts/holdout_stage4_report.py), [`src/citd_ml/verification/holdout_evidence.py:89-140`](../src/citd_ml/verification/holdout_evidence.py).
- Đối chiếu run 1/run 2: [`scripts/holdout_stage5_repro_check.py:2-11`](../scripts/holdout_stage5_repro_check.py).
- Số liệu và lịch sử: [`docs/BAO_CAO_KET_QUA_HOLDOUT.md`](BAO_CAO_KET_QUA_HOLDOUT.md), [`outputs/holdout/stage1/stage1_validation_report.json`](../outputs/holdout/stage1/stage1_validation_report.json), [`outputs/holdout/stage2/stage2_reproducibility_report.json`](../outputs/holdout/stage2/stage2_reproducibility_report.json), [`outputs/holdout/repro/stage5_repro_report.json`](../outputs/holdout/repro/stage5_repro_report.json), [`outputs/verification/holdout_run_history.md`](../outputs/verification/holdout_run_history.md).
