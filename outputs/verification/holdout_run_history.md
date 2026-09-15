# SỔ ĐỐI CHIẾU LỊCH SỬ CHẠY HOLDOUT

> Sinh tự động bởi `scripts/holdout_run_history.py` từ git history (đọc bằng `git show`, không checkout). HEAD khi sinh: `2239ba7` (2026-09-13T22:57:01+07:00).
>
> Mục đích: ghi lại đầy đủ số của cả ba lần mở holdout niêm phong để người review tự kiểm tra, thay vì chỉ có lời thừa nhận trong changelog.

## 1. Ba lần chạy holdout

| Lần | Commit | Ngày | Môi trường | thread_count | ROC-AUC | F1 @0.5 | Top 50% net R | Top 50% PF |
|---|---|---|---|---|---|---|---|---|
| 1 | `dc25cd3` | 2026-09-12 | Windows (suy ra) | không ghi (mặc định máy) | 0.605024522776 | 0.401706796708 | -23.8331 | 0.9815 |
| 2 | `5f46e41` | 2026-09-12 | Linux (suy ra) | không ghi (mặc định máy) | 0.602315255872 | 0.406469331706 | +30.7859 | 1.0239 |
| 3 | `7748828` | 2026-09-12 | Linux (suy ra) | 1 | 0.604554453893 | 0.402207234825 | -36.5528 | 0.9718 |


Ghi chú môi trường: `train_run1_report.json` của cả ba lần chạy **không có trường OS/platform**; cột Môi trường chỉ suy ra từ dạng đường dẫn trong `model_path` (`C:\...` → Windows, `/home/...` → Linux), không phải bằng chứng môi trường đầy đủ. Chỉ lần 3 ghi `thread_count=1`; hai lần đầu không ghim tham số này.

Lần 3 (`7748828`) là artifact holdout hiện hành: dòng Holdout trong `docs/BAO_CAO_KET_QUA_HOLDOUT.md` (0.6046 / 0.4022, top 50% = −36.55 R) lấy từ lần này. Hai lần trước là số cũ đã từng công bố, được thu hồi ở đây để không thất lạc.

### 1b. Baseline holdout và sweep 20–80% theo từng lần chạy

| Lọc (net R · PF) | `dc25cd3` | `5f46e41` | `7748828` |
|---|---|---|---|
| Baseline (100%) | +245.93 (1.0992) | +245.93 (1.0992) | +245.93 (1.0992) |
| Top 20% | -25.04 (0.9489) | -11.06 (0.9773) | -30.01 (0.9385) |
| Top 30% | -39.33 (0.9480) | -1.77 (0.9976) | -50.01 (0.9343) |
| Top 40% | -44.40 (0.9567) | +28.39 (1.0279) | -53.17 (0.9485) |
| Top 50% | -23.83 (0.9815) | +30.79 (1.0239) | -36.55 (0.9718) |
| Top 60% | +64.51 (1.0414) | +35.21 (1.0224) | +73.16 (1.0474) |
| Top 70% | +164.70 (1.0918) | +122.08 (1.0672) | +186.92 (1.1040) |
| Top 80% | +249.63 (1.1227) | +160.38 (1.0781) | +259.91 (1.1278) |


### 1c. Hash holdout dataset theo từng lần chạy

| Lần | SHA-256 trước train | SHA-256 sau train | Holdout bị sửa? | Số dòng train |
|---|---|---|---|---|
| `dc25cd3` | `acd6a4f8b162` | `acd6a4f8b162` | không đổi | 25,008 |
| `5f46e41` | `7be58aad5692` | `7be58aad5692` | không đổi | 25,008 |
| `7748828` | `7be58aad5692` | `7be58aad5692` | không đổi | 25,008 |


Hai lần chạy trên Linux dùng đúng cùng một file holdout (`7be58aad...`); lần Windows đầu dùng bản dataset gốc trước khi tái sinh trên Linux. Mỗi lần chạy đều ghi `holdout_file_unchanged = true` trong report Stage 2.

## 2. Đối chiếu pairwise prediction

| Cặp lần chạy | max |Δp| | mean |Δp| | Pearson r | Số dòng lệch > 1e-12 | Top 50 trùng | Jaccard |
|---|---|---|---|---|---|---|
| `dc25cd3` × `5f46e41` | 0.31904 | 0.0509254 | 0.939613 | 5,028 / 5,028 | 2,275 / 2,514 | 0.8264 |
| `dc25cd3` × `7748828` | 0.152374 | 0.0226626 | 0.988033 | 5,028 / 5,028 | 2,397 / 2,514 | 0.9111 |
| `5f46e41` × `7748828` | 0.335194 | 0.0508108 | 0.940948 | 5,028 / 5,028 | 2,254 / 2,514 | 0.8125 |


Luật top 50% (theo Stage 3): giữ `ceil(5,028 × 50%) = 2,514` dòng, xếp `probability` giảm dần rồi `row_id` tăng dần khi hòa; `row_id` là thứ tự dòng 0-based của chính file `holdout_scored.csv` mỗi lần chạy. Merge theo ba khóa `(origin_bar, entry_bar, leg)`.

## 3. Kiểm chứng ngoài — do teammate báo cáo

- **Người báo:** Bùi Quốc Thịnh; **ngày:** 2026-09-12; **môi trường:** Windows; **nhánh:** `main`.
- **Lệnh đã dùng:** `scripts/holdout_stage2_train.py --run-id 1|2 --verify`.
- Hai lượt train độc lập trên cùng một máy cho prediction giống hệt nhau (run1 so với run2).
- Prediction của teammate so với prediction bản hiện có tương đương trong sai số 1e-15.
- Không byte-identical giữa hai máy khác nhau.
- Không ghi lại số metric nào kèm theo.

Kết luận báo cáo của teammate: hai lượt train độc lập trên cùng một máy cho prediction giống hệt nhau; prediction của teammate so với prediction bản hiện có **trùng trong 1e-15, không byte-identical** giữa hai máy; không ghi lại số metric nào.

**Trạng thái bằng chứng:** Đây là báo cáo miệng của teammate, không có artifact đối chứng được commit (không có prediction/hash/log từ máy Windows trong repo) và không thể tái tạo từ git. Chỉ ghi nhận như thông tin tham khảo, không dùng thay bằng chứng PASS.

## 4. Tham chiếu bàn giao bước 4 (tính lại tại HEAD)

### 4a. Bảng 1 — phân loại, bỏ khúc 1, mean theo fold

| Cách chia | AUC tính lại | AUC bàn giao | F1 tính lại | F1 bàn giao |
|---|---|---|---|---|
| Cách 1 — Random K-Fold | 0.859496 | 0.8595 | 0.652477 | 0.6525 |
| Cách 1b — Grouped K-Fold | 0.747469 | 0.7475 | 0.510503 | 0.5105 |
| Cách 2 — Walk-forward | 0.586725 | 0.5867 | 0.326728 | 0.3267 |
| Cách 3 — WF + Purge/Embargo | 0.594776 | 0.5948 | 0.330422 | 0.3304 |


Luật tính: giữ fold>=1 và chunk!=1; tính AUC/F1 từng fold trên phần còn lại rồi lấy trung bình; chunk chia theo row_id với biên [0,5001,10003,15004,20006,25008] (searchsorted side=right, giống backtest_pyramid_local.py). artifact bàn giao ghi 4 chữ số thập phân; dung sai đối chiếu 5e-5.

### 4b. Bảng 2 — tài chính top 50%, khúc 2–5 (nguyên từ artifact bàn giao)

| Phương pháp | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---|---|---|---|---|
| Baseline | 20,007 | +3,358.25 | 236.13 | 1.2940 | 31.71 |
| Cách 1 - Random K-Fold | 10,004 | +6,998.30 | 75.71 | 2.5627 | 44.88 |
| Cách 1b - Grouped K-Fold | 10,004 | +4,724.34 | 99.50 | 1.9418 | 39.24 |
| Cách 2 - Walk-forward | 10,004 | +2,207.25 | 205.31 | 1.4226 | 34.86 |
| Cách 3 - Purged Walk-forward | 10,004 | +2,253.72 | 154.11 | 1.4350 | 35.17 |


Các dòng khác của `backtest_retention_sweep.csv` (20–80%) nằm nguyên trong file JSON.

## 5. Nguồn dữ liệu (provenance)

| Commit | Đường dẫn | Git blob | SHA-256 (12 ký tự đầu) | Byte |
|---|---|---|---|---|
| `dc25cd3` | `outputs/holdout/stage3/holdout_scored.csv` | `05bf64e69941` | `0c0b3e15ab2c` | 1,666,608 |
| `dc25cd3` | `outputs/holdout/stage3/stage3_report.json` | `cb71482d1303` | `1d5a8ed7cd54` | 3,285 |
| `dc25cd3` | `outputs/holdout/stage2/train_run1_report.json` | `50601c082ba3` | `384f30cf1074` | 1,878 |
| `5f46e41` | `outputs/holdout/stage3/holdout_scored.csv` | `ee56ec7c56a3` | `833d784096a6` | 1,661,560 |
| `5f46e41` | `outputs/holdout/stage3/stage3_report.json` | `681272b33063` | `fbadee98a571` | 3,288 |
| `5f46e41` | `outputs/holdout/stage2/train_run1_report.json` | `8b36af29f4a4` | `f6815f6115ed` | 1,840 |
| `7748828` | `outputs/holdout/stage3/holdout_scored.csv` | `a0bb98a8854c` | `0039f3566ba6` | 1,661,606 |
| `7748828` | `outputs/holdout/stage3/stage3_report.json` | `fba1676f4bd3` | `f923461dc739` | 3,288 |
| `7748828` | `outputs/holdout/stage2/train_run1_report.json` | `80a21c95f49e` | `b1b743782baa` | 1,863 |


HEAD dùng cho tham chiếu bàn giao:

| Đường dẫn | Số dòng | SHA-256 (12 ký tự đầu) |
|---|---|---|
| `outputs/catboost_training/oof_random_kfold.csv` | 25,008 | `7d8c4bc3ff9d` |
| `outputs/catboost_training/oof_grouped_kfold.csv` | 25,008 | `f1a78d42763d` |
| `outputs/catboost_training/oof_walk_forward.csv` | 25,008 | `67ecac4c0da7` |
| `outputs/catboost_training/oof_purged_walk_forward.csv` | 25,008 | `fa33abe906d8` |
| `outputs/backtest/backtest_summary_top50.csv` | 5 | `961be98d4103` |
| `outputs/backtest/backtest_retention_sweep.csv` | 35 | `af99b68be72a` |


## 6. Kết quả kiểm tra tự động

| Chỉ số | Thu hồi | Kỳ vọng | |Δ| | Kết quả |
|---|---|---|---|---|
| dc25cd3.roc_auc | 0.605024522776 | 0.605024522776 | 0 | PASS |
| dc25cd3.f1_at_0_5 | 0.401706796708 | 0.401706796708 | 0 | PASS |
| dc25cd3.top50_net_profit_R | -23.8330501581 | -23.8330501581 | 0 | PASS |
| dc25cd3.baseline_net_profit_R | 245.929682659 | 245.929682659 | 0 | PASS |
| 5f46e41.roc_auc | 0.602315255872 | 0.602315255872 | 0 | PASS |
| 5f46e41.f1_at_0_5 | 0.406469331706 | 0.406469331706 | 0 | PASS |
| 5f46e41.top50_net_profit_R | 30.7859222571 | 30.7859222571 | 0 | PASS |
| 5f46e41.baseline_net_profit_R | 245.929682659 | 245.929682659 | 0 | PASS |
| 7748828.roc_auc | 0.604554453893 | 0.604554453893 | 0 | PASS |
| 7748828.f1_at_0_5 | 0.402207234825 | 0.402207234825 | 0 | PASS |
| 7748828.top50_net_profit_R | -36.5527549959 | -36.5527549959 | 0 | PASS |
| 7748828.baseline_net_profit_R | 245.929682659 | 245.929682659 | 0 | PASS |


Dung sai cho phép: `1e-12`. Cả ba commit đều là tổ tiên của HEAD; mỗi file scored đủ 5,028 dòng và khóa không trùng. Nếu bất kỳ kiểm tra nào lệch, script dừng với mã khác 0 thay vì ghi sổ.

## 7. Cách tái tạo

```bash
.venv/bin/python scripts/holdout_run_history.py
```
