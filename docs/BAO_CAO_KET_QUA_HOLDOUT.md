# BÁO CÁO KẾT QUẢ TRAIN VÀ HOLDOUT

> Phạm vi: tổng hợp artifact đã chốt của 4 cách chia trên tập 80% đầu và kết
> quả holdout niêm phong. Báo cáo chỉ ghi số liệu, quyết định triển khai và bằng
> chứng kiểm chứng; không biện luận hay kết luận thay báo cáo chính.

## Lịch sử mở holdout và lý do chạy lại

Bảng A ghi ba phiên bản kết quả holdout lịch sử truy xuất được từ Git, lưu trong
`outputs/verification/holdout_run_history.json`. Đây không phải nhật ký đầy đủ
của mọi lần thực thi hoặc chấm điểm: sổ dùng ba commit đã xác định, không ghi nhận
các lần không được lưu vào Git. Ngày trong bảng là ngày commit, không phải log thời điểm chạy.
Các lần lặp để kiểm chứng và thí nghiệm bổ sung ngày 15/09 được trình bày riêng ở Phụ lục A và Phần 4.

| Phiên bản | Commit | Ngày commit | Môi trường suy ra | `thread_count` | ROC-AUC | F1 @0.5 | Top 50% net R | Top 50% PF |
|---|---|---|---|---|---|---|---|---|
| Lần 1 — Windows gốc (không ghim thread_count) | `dc25cd3` | 2026-09-12 | Windows (suy từ đường dẫn) | không ghim | 0.6050 | 0.4017 | -23.83 | 0.9815 |
| Lần 2 — Linux đa luồng (không ghim thread_count) | `5f46e41` | 2026-09-12 | Linux (suy từ đường dẫn) | không ghim | 0.6023 | 0.4065 | +30.79 | 1.0239 |
| Lần 3 — Linux thread_count=1 (artifact hiện tại) | `7748828` | 2026-09-12 | Linux (suy từ đường dẫn) | `1` | 0.6046 | 0.4022 | -36.55 | 0.9718 |

Baseline holdout giống nhau ở cả ba lần: 5,028 lệnh, +245.93 R (baseline là tradelist tĩnh, không phụ thuộc model).

Xác minh độc lập của teammate (`external_verification` trong sổ lịch sử): Bùi Quốc Thịnh, ngày 2026-09-12, môi trường Windows, nhánh `main`, lệnh `scripts/holdout_stage2_train.py --run-id 1|2 --verify`.

- Hai lượt train độc lập trên cùng một máy cho prediction giống hệt nhau (run1 so với run2).
- Prediction của teammate so với prediction bản hiện có tương đương trong sai số 1e-15.
- Không byte-identical giữa hai máy khác nhau.
- Không ghi lại số metric nào kèm theo.

Theo tin nhắn teammate: prediction trên tập train giữa hai máy trùng trong `1e-15` (không byte-identical); lần xác minh này không ghi lại metric nào và không có artifact đối chứng được commit, nên chỉ là thông tin tham khảo.

Artifact canonical của báo cáo: lần 3 — commit `7748828`, `thread_count=1`. Các bảng holdout lấy từ `outputs/holdout/stage3` (run 1) và bảng bốn cách chia canonical lấy từ cây `outputs/step4_thread1`; hai thư mục bàn giao đóng băng `outputs/catboost_training` và `outputs/backtest` chỉ còn là khối tham chiếu.

Lý do chạy lại: cấu hình CatBoost ban đầu chưa ghim `thread_count`; lần 3 chỉ thêm `thread_count=1` như thiết lập kỹ thuật, không đổi feature, hyperparameter mô hình, tập train, tập holdout hay luật chọn top-k. Các số lịch sử truy xuất được giữ trong sổ.

## Phần 1 — Số liệu

### 1.1. Cấu hình train cuối

| Thuộc tính | Giá trị |
|---|---|
| Tập train trước/sau purge + embargo | 25,008 / 25,008 dòng |
| Tập holdout | 5,028 dòng; không bỏ dòng |
| Feature đầu vào | 23 `FEATURES`; không đưa `META` vào `X` |
| Hyperparameter và thiết lập kỹ thuật | `iterations=1000` · `learning_rate=0.05` · `depth=6` · `l2_leaf_reg=3.0` · `auto_class_weights=Balanced` · `eval_metric=AUC` · `random_seed=42` · `thread_count=1` · `allow_writing_files=False` |

Môi trường sinh artifact (từ `train_run1_report.json`):

| Thành phần | Giá trị |
|---|---|
| OS (`platform`) | `Linux-7.2.5-1-cachyos-x86_64-with-glibc2.44` |
| Python | `3.12.14` |
| CatBoost | `1.2.10` |
| scikit-learn | `1.9.0` |
| pandas | `3.0.5` |
| NumPy | `2.5.2` |
| Plotly | `7.0.0` |
| Matplotlib | `3.11.2` |

### 1.2. Bảng 1 — Chỉ số phân loại

**Khối chính — 4 cách chia chạy lại với `thread_count=1` (cây canonical):**

| Cách chia | ROC-AUC | F1 @0.5 |
|---|---|---|
| Cách 1 — Random K-Fold | 0.8582 | 0.6494 |
| Cách 1b — Grouped K-Fold | 0.7454 | 0.5077 |
| Cách 2 — Walk-forward | 0.5875 | 0.3288 |
| Cách 3 — WF + Purge/Embargo | 0.5895 | 0.3247 |
| Holdout | 0.6046 | 0.4022 |

**Khối tham chiếu — bàn giao (không ghim `thread_count`):**

| Cách chia | ROC-AUC | F1 @0.5 |
|---|---|---|
| Cách 1 — Random K-Fold | 0.8595 | 0.6525 |
| Cách 1b — Grouped K-Fold | 0.7475 | 0.5105 |
| Cách 2 — Walk-forward | 0.5867 | 0.3267 |
| Cách 3 — WF + Purge/Embargo | 0.5948 | 0.3304 |

**Chênh lệch canonical − bàn giao (tính trên giá trị chưa làm tròn):**

| Cách chia | ΔROC-AUC | ΔF1 @0.5 |
|---|---|---|
| Cách 1 — Random K-Fold | -0.0013 | -0.0031 |
| Cách 1b — Grouped K-Fold | -0.0021 | -0.0028 |
| Cách 2 — Walk-forward | +0.0007 | +0.0020 |
| Cách 3 — WF + Purge/Embargo | -0.0052 | -0.0057 |

Ghi chú: bốn dòng Cách 1–3 của khối canonical là trung bình chỉ số theo fold sau
khi bỏ khúc 1 để các nhánh được đo trên cùng phạm vi khúc 2–5, tính lại từ bốn
file OOF trong cây canonical và khớp `metrics_chunk2_5.csv` ở 4 chữ số thập
phân. Khối bàn giao được tính lại từ bốn OOF đóng băng trong
`outputs/catboost_training` và khớp số đã công bố trong sổ lịch sử. Dòng Holdout
được tính một lần trên toàn bộ 5,028 mẫu holdout, không phải
trung bình theo fold.

### 1.3. Bảng 2 — Chỉ số tài chính, giữ top 50%

**Khối chính — 4 cách chia trên khúc 2–5, giữ top 50%, `thread_count=1`:**

|  | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---|---|---|---|---|
| Baseline (khúc 2–5) | 20,007 | +3,358.25 | 236.13 | 1.2940 | 31.71 |
| Cách 1 — Random K-Fold | 10,004 | +6,937.53 | 73.76 | 2.5437 | 44.65 |
| Cách 1b — Grouped K-Fold | 10,004 | +4,752.04 | 99.71 | 1.9500 | 39.26 |
| Cách 2 — Walk-forward | 10,004 | +2,148.31 | 211.29 | 1.4117 | 34.76 |
| Cách 3 — WF + Purge/Embargo | 10,004 | +2,119.48 | 142.36 | 1.4074 | 34.99 |

**Khối tham chiếu — bàn giao (không ghim `thread_count`):**

|  | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---|---|---|---|---|
| Baseline (khúc 2–5) | 20,007 | +3,358.25 | 236.13 | 1.2940 | 31.71 |
| Cách 1 — Random K-Fold | 10,004 | +6,998.30 | 75.71 | 2.5627 | 44.88 |
| Cách 1b — Grouped K-Fold | 10,004 | +4,724.34 | 99.50 | 1.9418 | 39.24 |
| Cách 2 — Walk-forward | 10,004 | +2,207.25 | 205.31 | 1.4226 | 34.86 |
| Cách 3 — WF + Purge/Embargo | 10,004 | +2,253.72 | 154.11 | 1.4350 | 35.17 |

**Khối holdout — giữ top 50%:**

|  | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---|---|---|---|---|
| Baseline holdout | 5,028 | +245.93 | 305.32 | 1.0992 | 35.28 |
| Holdout, top 50% | 2,514 | -36.55 | 263.25 | 0.9718 | 34.81 |

### 1.4. Bảng 3 — Sweep 20–80% của 4 cách chia (`thread_count=1`)

Baseline (không lọc, khúc 2–5): 20,007 lệnh, +3,358.25 R, MaxDD 236.13 R, PF 1.2940, win rate 31.71%.

| Cách chia | Mức giữ | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---|---|---|---|---|---|
| Cách 1 — Random K-Fold | Top 20% | 4,002 | +5,248.16 | 13.49 | 6.3257 | 64.82 |
| Cách 1 — Random K-Fold | Top 30% | 6,003 | +6,640.38 | 23.91 | 4.4036 | 57.02 |
| Cách 1 — Random K-Fold | Top 40% | 8,003 | +7,136.31 | 34.59 | 3.2781 | 50.58 |
| Cách 1 — Random K-Fold | Top 50% | 10,004 | +6,937.53 | 73.76 | 2.5437 | 44.65 |
| Cách 1 — Random K-Fold | Top 60% | 12,005 | +6,451.10 | 110.82 | 2.0872 | 40.32 |
| Cách 1 — Random K-Fold | Top 70% | 14,005 | +5,800.28 | 136.66 | 1.7880 | 37.14 |
| Cách 1 — Random K-Fold | Top 80% | 16,006 | +4,980.00 | 178.18 | 1.5665 | 34.61 |
| Cách 1b — Grouped K-Fold | Top 20% | 4,002 | +2,803.94 | 33.27 | 2.7979 | 48.23 |
| Cách 1b — Grouped K-Fold | Top 30% | 6,003 | +3,696.16 | 47.29 | 2.4148 | 44.76 |
| Cách 1b — Grouped K-Fold | Top 40% | 8,003 | +4,325.22 | 67.22 | 2.1492 | 41.83 |
| Cách 1b — Grouped K-Fold | Top 50% | 10,004 | +4,752.04 | 99.71 | 1.9500 | 39.26 |
| Cách 1b — Grouped K-Fold | Top 60% | 12,005 | +4,772.24 | 150.80 | 1.7581 | 37.08 |
| Cách 1b — Grouped K-Fold | Top 70% | 14,005 | +4,521.23 | 191.83 | 1.5925 | 35.08 |
| Cách 1b — Grouped K-Fold | Top 80% | 16,006 | +4,297.70 | 193.95 | 1.4817 | 33.63 |
| Cách 2 — Walk-forward | Top 20% | 4,002 | +1,056.60 | 103.86 | 1.5463 | 37.21 |
| Cách 2 — Walk-forward | Top 30% | 6,003 | +1,327.33 | 159.63 | 1.4409 | 36.27 |
| Cách 2 — Walk-forward | Top 40% | 8,003 | +1,695.46 | 157.93 | 1.4115 | 35.34 |
| Cách 2 — Walk-forward | Top 50% | 10,004 | +2,148.31 | 211.29 | 1.4117 | 34.76 |
| Cách 2 — Walk-forward | Top 60% | 12,005 | +2,575.06 | 240.18 | 1.4051 | 34.37 |
| Cách 2 — Walk-forward | Top 70% | 14,005 | +2,760.58 | 283.51 | 1.3664 | 33.75 |
| Cách 2 — Walk-forward | Top 80% | 16,006 | +3,120.69 | 273.52 | 1.3581 | 33.50 |
| Cách 3 — WF + Purge/Embargo | Top 20% | 4,002 | +1,266.26 | 86.49 | 1.6598 | 38.26 |
| Cách 3 — WF + Purge/Embargo | Top 30% | 6,003 | +1,381.02 | 149.36 | 1.4623 | 36.40 |
| Cách 3 — WF + Purge/Embargo | Top 40% | 8,003 | +1,674.66 | 144.96 | 1.4107 | 35.67 |
| Cách 3 — WF + Purge/Embargo | Top 50% | 10,004 | +2,119.48 | 142.36 | 1.4074 | 34.99 |
| Cách 3 — WF + Purge/Embargo | Top 60% | 12,005 | +2,323.85 | 208.79 | 1.3642 | 34.12 |
| Cách 3 — WF + Purge/Embargo | Top 70% | 14,005 | +2,688.96 | 221.19 | 1.3578 | 33.77 |
| Cách 3 — WF + Purge/Embargo | Top 80% | 16,006 | +3,061.29 | 240.51 | 1.3510 | 33.38 |

### 1.5. Bảng 4 — Sweep holdout 20–80%

| Lọc | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---|---|---|---|---|
| Top 20% | 1,006 | -30.01 | 125.38 | 0.9385 | 36.18 |
| Top 30% | 1,509 | -50.01 | 210.77 | 0.9343 | 35.52 |
| Top 40% | 2,012 | -53.17 | 251.56 | 0.9485 | 35.04 |
| Top 50% | 2,514 | -36.55 | 263.25 | 0.9718 | 34.81 |
| Top 60% | 3,017 | +73.16 | 233.95 | 1.0474 | 35.60 |
| Top 70% | 3,520 | +186.92 | 197.24 | 1.1040 | 35.77 |
| Top 80% | 4,023 | +259.91 | 219.02 | 1.1278 | 35.52 |

### 1.6. Cấu hình số luồng

Cấu hình chính thức cố định `thread_count=1`. Sau đó, ngày 15/09/2026 đã chạy thí nghiệm đổi `thread_count`, có chấm holdout ở `thread_count=2` và `-1`. Số liệu chính thức không đổi. Chi tiết thí nghiệm đã thực hiện được lưu tại Phụ lục A.

### 1.7. Ghi chú dữ liệu

- Holdout: entry_time từ 2025-02-09 16:00:00 đến 2026-08-21 04:30:00; label_end_time muộn nhất 2026-08-21 07:15:00.
- 4 lệnh biên (entry_time trước mốc holdout nhưng không có trong dataset đóng băng `data/processed/dataset_catboost.csv`) được xác định bằng so khớp khóa `(origin_bar, entry_bar, leg)`, không chép tay:

| entry_time | label_end_time | origin_bar | entry_bar | leg |
|---|---|---|---|---|
| 2025-02-08 06:00:00 | 2025-02-08 18:30:00 | 199,930 | 199,930 | 0 |
| 2025-02-08 06:15:00 | 2025-02-08 18:45:00 | 199,930 | 199,931 | 1 |
| 2025-02-08 06:30:00 | 2025-02-08 19:00:00 | 199,930 | 199,932 | 2 |
| 2025-02-08 06:45:00 | 2025-02-08 19:15:00 | 199,930 | 199,933 | 3 |

- 4 lệnh này không thuộc train và cũng không thuộc holdout: 25,008 (train) + 4 (biên) + 5,028 (holdout) = 30,040 dòng của dataset tái sinh toàn lịch sử.
- Quy ước chữ số: ROC-AUC/F1/PF 4 chữ số thập phân; net R và MaxDD 2 chữ số thập phân, dùng dấu phẩy phân cách hàng nghìn và R có dấu; win rate 2 chữ số thập phân; số lệnh dùng dấu phẩy phân cách hàng nghìn.

## Phần 2 — Biểu đồ

![Biểu đồ 1 — Baseline và 4 nhánh top 50%, khúc 2–5](../outputs/holdout/stage4/equity-curve-chunk2-5-top50.png)

![Biểu đồ 2 — Baseline holdout và Holdout top 50%](../outputs/holdout/stage4/equity-curve-holdout-top50.png)

- [Biểu đồ 1 — bản tương tác HTML](../outputs/holdout/stage4/equity-curve-chunk2-5-top50.html)
- [Biểu đồ 2 — bản tương tác HTML](../outputs/holdout/stage4/equity-curve-holdout-top50.html)

Biểu đồ HTML dựng bằng Python/Plotly; ảnh PNG xuất lại từ đúng dữ liệu đường
vốn bằng matplotlib (`dpi=110`, figsize 12.0 × 5.4 inch, không có timestamp
trong phần chữ). Cả hai dạng đều dùng đường bậc thang ngang-rồi-dọc
(`hv`/`steps-post`): equity chỉ thay đổi tại `close_time`, không nội suy tuyến
tính giữa hai lần đóng lệnh. Trục dọc là R cộng dồn từ 0. File HTML chứa Plotly
nội tuyến nên mở độc lập được.

## Phần 3 — Quyết định triển khai

1. Dùng `ceil(n × keep_pct / 100)` cho số lệnh giữ lại: top 50% của 20,007 lệnh là 10,004; top 50% của 5,028 lệnh là 2,514.
2. Xếp `probability` giảm dần, dùng `row_id` tăng dần để phá hòa; vũ trụ lệnh baseline giữ cố định nên lệnh bị loại không làm đổi tín hiệu sau đó.
3. Equity ghi nhận tại `close_time`; các lệnh đóng cùng lúc được cộng thành một điểm rồi mới cập nhật đường vốn.
4. Khối chính của Bảng 1–3 là cây canonical `thread_count=1` (`outputs/step4_thread1`); số bàn giao (không ghim `thread_count`) chỉ còn là khối tham chiếu được dán nhãn, kèm bảng chênh lệch canonical − bàn giao ở Bảng 1.
5. Cố định `thread_count=1` cho cấu hình chính thức.
6. Dùng đường bậc thang ngang-rồi-dọc (`hv` cho HTML, `steps-post` cho PNG) để equity giữ nguyên giữa hai mốc đóng lệnh và chỉ nhảy tại thời điểm R được ghi nhận.
7. Purge và embargo đều trả về 0 dòng, nên train giữ nguyên 25,008 dòng; điều kiện lọc được chạy trước khi quyết định không loại dòng nào.
8. So khớp giá baseline dùng sai số tuyệt đối `5e-4`, theo validator bàn giao; giá vào, giá ra và R được kiểm dưới cùng ngưỡng này.
9. Chấm điểm holdout lấy danh sách 23 `FEATURES` trực tiếp từ `build_features.py`; không tự liệt kê cột.
10. Biểu đồ 1 dùng `backtest_scored_universe.csv` của cây canonical và chỉ nhận các dòng `chunk` 2–5; biểu đồ 2 bắt đầu lại từ R = 0 tại holdout, dùng trực tiếp vũ trụ baseline và danh sách Top 50% đã lưu từ Giai đoạn 3.
11. Bốn lệnh biên không thuộc tập train và không được đưa vào chỉ số đánh giá holdout. Replay toàn lịch sử vẫn xử lý giai đoạn này để duy trì trạng thái chiến lược; các lệnh được liệt kê ở mục 1.7.
12. Hai thư mục bàn giao `outputs/catboost_training` và `outputs/backtest` chỉ được đọc, không bị ghi đè; mọi bảng canonical, sweep và biểu đồ đều lấy từ cây `outputs/step4_thread1`.

## Phần 4 — Kiểm chứng

| Phép kiểm | Kết quả |
|---|---|
| Dataset tái sinh trước holdout | **PASS** — 25,008/25,008 khóa cũ; thiếu 0 |
| Giá trị feature | **PASS** — 575,184 ô đã so; lệch 0 |
| Purge / embargo tại biên holdout | **PASS** — cắt 0 / 0 dòng; train còn 25,008 |
| Holdout không bị sửa khi train/chấm điểm | **PASS** — SHA-256 trước/sau giữ nguyên |
| Replay baseline so với tradelist bàn giao | **PASS** — 5,028/5,028 lệnh; trường lệch: không có |
| Lặp train trên cùng máy (Stage 2) | **PASS** — 25,008 predictions giống hệt; max abs diff = 0.0; `train_run1_report.json`/`train_run2_report.json` đã ghi `platform` (Linux-7.2.5-1-cachyos-x86_64-with-glibc2.44) và `plotly` (7.0.0) |
| Tái lập run1 vs run2 (Stage 3–4: chấm điểm holdout, backtest, bảng, biểu đồ) | **PASS** — max \|Δprobability\| = 0.0; ΔROC-AUC = 0.0; ΔF1 = 0.0; max Δmetric backtest = 0.0; bảng số giống hệt nhau; biểu đồ byte-identical: `equity-curve-chunk2-5-top50.html`, `equity-curve-chunk2-5-top50.png`, `equity-curve-holdout-top50.html`, `equity-curve-holdout-top50.png` |
| `verify_pipeline` trên cây canonical `thread_count=1` | **PASS** — manifest `outputs/step4_thread1/verification/reproducibility.json`; phạm vi: `outputs/step4_thread1/catboost_training` + `outputs/step4_thread1/backtest`; hai thư mục bàn giao đóng băng không bị ghi đè và không tái lập byte trên Linux |
| File model `.cbm` | **GHI NHẬN** — SHA-256 hai file khác nhau (`90a6c1797a27…` vs `00dc6f07ef47…`); binary model không phải tiêu chí PASS |
| Kiểm chứng liên máy | **GIỚI HẠN** — teammate Windows báo prediction trùng trong `1e-15` (không byte-identical, không ghi metric hay artifact đối chứng); không thể tái tạo từ git. |

Môi trường sinh artifact: OS Linux-7.2.5-1-cachyos-x86_64-with-glibc2.44; Python 3.12.14; CatBoost 1.2.10; scikit-learn 1.9.0; pandas 3.0.5; NumPy 2.5.2; Plotly 7.0.0; Matplotlib 3.11.2.

## Phần 5 — Bảng file sinh ra

| File | Giai đoạn | Chứa gì | Quy mô | Kích thước |
|---|---|---|---|---|
| [`dataset_catboost_full_regenerated.csv`](../outputs/holdout/stage1/dataset_catboost_full_regenerated.csv) | 1 | Dataset tái sinh trên toàn bộ lịch sử để kiểm tra và tách holdout | 30,040 dòng | 9,800,880 B |
| [`dataset_catboost_holdout.csv`](../outputs/holdout/stage1/dataset_catboost_holdout.csv) | 1 | Dataset holdout từ mốc 2025-02-08 15:30:00 | 5,028 dòng | 1,648,262 B |
| [`stage1_validation_report.json`](../outputs/holdout/stage1/stage1_validation_report.json) | 1 | Số liệu kiểm khóa cũ và đối chiếu 23 feature | artifact | 888 B |
| [`catboost_final_holdout_run1.cbm`](../outputs/holdout/stage2/catboost_final_holdout_run1.cbm) | 2 | Model CatBoost cuối dùng để chấm holdout | artifact | 1,129,416 B |
| [`catboost_final_holdout_run2.cbm`](../outputs/holdout/stage2/catboost_final_holdout_run2.cbm) | 2 | Model train độc lập lần hai để kiểm lặp | artifact | 1,129,416 B |
| [`stage2_reproducibility_report.json`](../outputs/holdout/stage2/stage2_reproducibility_report.json) | 2 | So sánh hai lượt train trên cùng máy | artifact | 722 B |
| [`train_predictions_run1.npy`](../outputs/holdout/stage2/train_predictions_run1.npy) | 2 | Prediction trên tập train của lượt 1 | 25,008 giá trị | 200,192 B |
| [`train_predictions_run2.npy`](../outputs/holdout/stage2/train_predictions_run2.npy) | 2 | Prediction trên tập train của lượt 2 | 25,008 giá trị | 200,192 B |
| [`train_run1_report.json`](../outputs/holdout/stage2/train_run1_report.json) | 2 | Cấu hình, phiên bản, hash và kiểm biên của lượt train 1 | artifact | 2,011 B |
| [`train_run2_report.json`](../outputs/holdout/stage2/train_run2_report.json) | 2 | Cấu hình, phiên bản, hash và kiểm biên của lượt train 2 | artifact | 2,011 B |
| [`holdout_fixed_trade_universe_scored.csv`](../outputs/holdout/stage3/holdout_fixed_trade_universe_scored.csv) | 3 | Vũ trụ lệnh baseline holdout cố định kèm điểm | 5,028 dòng | 2,263,531 B |
| [`holdout_scored.csv`](../outputs/holdout/stage3/holdout_scored.csv) | 3 | 5,028 dòng holdout kèm xác suất CatBoost | 5,028 dòng | 1,661,606 B |
| [`holdout_trades_top20.csv`](../outputs/holdout/stage3/holdout_trades_top20.csv) | 3 | Danh sách lệnh holdout được giữ ở mức top 20% | 1,006 dòng | 454,493 B |
| [`holdout_trades_top30.csv`](../outputs/holdout/stage3/holdout_trades_top30.csv) | 3 | Danh sách lệnh holdout được giữ ở mức top 30% | 1,509 dòng | 680,410 B |
| [`holdout_trades_top40.csv`](../outputs/holdout/stage3/holdout_trades_top40.csv) | 3 | Danh sách lệnh holdout được giữ ở mức top 40% | 2,012 dòng | 906,297 B |
| [`holdout_trades_top50.csv`](../outputs/holdout/stage3/holdout_trades_top50.csv) | 3 | Danh sách lệnh holdout được giữ ở mức top 50% | 2,514 dòng | 1,132,050 B |
| [`holdout_trades_top60.csv`](../outputs/holdout/stage3/holdout_trades_top60.csv) | 3 | Danh sách lệnh holdout được giữ ở mức top 60% | 3,017 dòng | 1,358,175 B |
| [`holdout_trades_top70.csv`](../outputs/holdout/stage3/holdout_trades_top70.csv) | 3 | Danh sách lệnh holdout được giữ ở mức top 70% | 3,520 dòng | 1,584,187 B |
| [`holdout_trades_top80.csv`](../outputs/holdout/stage3/holdout_trades_top80.csv) | 3 | Danh sách lệnh holdout được giữ ở mức top 80% | 4,023 dòng | 1,810,896 B |
| [`stage3_backtest_summary.csv`](../outputs/holdout/stage3/stage3_backtest_summary.csv) | 3 | Metrics baseline và sweep top 20–80% | 8 dòng | 654 B |
| [`stage3_report.json`](../outputs/holdout/stage3/stage3_report.json) | 3 | Kết quả phân loại, replay baseline và backtest holdout | artifact | 3,288 B |
| [`equity-curve-chunk2-5-top50.html`](../outputs/holdout/stage4/equity-curve-chunk2-5-top50.html) | 4 | Biểu đồ bậc thang baseline và 4 nhánh trên khúc 2–5 | artifact | 5,406,759 B |
| [`equity-curve-chunk2-5-top50.png`](../outputs/holdout/stage4/equity-curve-chunk2-5-top50.png) | 4 | Ảnh PNG biểu đồ bậc thang baseline và 4 nhánh trên khúc 2–5 | artifact | 93,985 B |
| [`equity-curve-holdout-top50.html`](../outputs/holdout/stage4/equity-curve-holdout-top50.html) | 4 | Biểu đồ bậc thang baseline và top 50% trên holdout | artifact | 4,434,195 B |
| [`equity-curve-holdout-top50.png`](../outputs/holdout/stage4/equity-curve-holdout-top50.png) | 4 | Ảnh PNG biểu đồ bậc thang baseline và top 50% trên holdout | artifact | 81,443 B |
| [`holdout_sweep_20_80.csv`](../outputs/holdout/stage4/holdout_sweep_20_80.csv) | 4 | Bảng tài chính holdout theo bảy mức giữ lệnh | 7 dòng | 366 B |
| [`implementation_decisions.md`](../outputs/holdout/stage4/implementation_decisions.md) | 4 | Các quyết định triển khai Stage 4 | artifact | 2,654 B |
| [`stage4_manifest.json`](../outputs/holdout/stage4/stage4_manifest.json) | 4 | Danh sách output, số điểm trên từng đường vốn và thiết lập xuất PNG | artifact | 856 B |
| [`stage4_tables.md`](../outputs/holdout/stage4/stage4_tables.md) | 4 | Bốn bảng kết quả ở định dạng Markdown | artifact | 5,838 B |
| [`table1_classification_metrics.csv`](../outputs/holdout/stage4/table1_classification_metrics.csv) | 4 | Bảng chỉ số phân loại: khối canonical, khối bàn giao và chênh lệch | 13 dòng | 750 B |
| [`table2_financial_metrics_top50.csv`](../outputs/holdout/stage4/table2_financial_metrics_top50.csv) | 4 | Bảng tài chính top 50%: khối bốn cách chia, khối bàn giao và khối holdout | 12 dòng | 1,004 B |
| [`table3_branch_sweep_20_80.csv`](../outputs/holdout/stage4/table3_branch_sweep_20_80.csv) | 4 | Sweep 20–80% của bốn nhánh canonical thread_count=1 | 28 dòng | 1,703 B |

### Bằng chứng bổ sung

| Đường dẫn | Chứa gì |
|---|---|
| [`outputs/step4_thread1/`](../outputs/step4_thread1) | Cây canonical bốn split `thread_count=1`: `comparison_report.json`, `verification/reproducibility.json`, OOF, hai bảng backtest và `metrics_chunk2_5.csv` dùng cho Bảng 1–3. |
| [`outputs/thread_count_sensitivity/thread_count_sensitivity.json`](../outputs/thread_count_sensitivity/thread_count_sensitivity.json) | Thí nghiệm `thread_count`: bốn cấu hình, deltas prediction, top-50 và `conclusion_facts` lưu tại Phụ lục A. |
| [`outputs/verification/holdout_run_history.json`](../outputs/verification/holdout_run_history.json) | Ba phiên bản kết quả holdout lịch sử truy xuất từ Git: AUC/F1, top-50, sweep, pairwise deltas, xác minh teammate và khối tham chiếu bàn giao. |
| [`outputs/holdout/repro/stage5_repro_report.json`](../outputs/holdout/repro/stage5_repro_report.json) | Kết quả đối chiếu run1 vs run2 toàn chuỗi Stage 3–4: status, deltas số học và hash bảng/biểu đồ dùng cho Phần 4. |
| [`outputs/step4_thread1/verification/reproducibility.json`](../outputs/step4_thread1/verification/reproducibility.json) | Manifest `verify_pipeline` của cây canonical: hai lượt train/backtest độc lập, hash output và phiên bản môi trường. |
| [`outputs/holdout/repro/run2/`](../outputs/holdout/repro/run2) | Báo cáo và artifact run 2 độc lập (`stage3`, `stage4`, `BAO_CAO_holdout_run2.md`) để đối chiếu run1. |

## Phụ lục A — Thí nghiệm `thread_count` đã thực hiện

Thí nghiệm có kiểm soát: cùng dữ liệu, feature, hyperparameter và `random_seed=42`, chỉ đổi `thread_count` (nguồn: `outputs/thread_count_sensitivity/thread_count_sensitivity.json`).

| Cấu hình | `thread_count` | Train max\|Δp\| | Holdout max\|Δp\| | Holdout Pearson r | Holdout ROC-AUC | Holdout F1 | Top 50% net R | Top 50% trùng |
|---|---|---|---|---|---|---|---|---|
| tc1_a | 1 | 0.000000 | 0.000000 | 1.0000 | 0.6046 | 0.4022 | -36.55 | 2514/2514 |
| tc1_b (lặp cùng cấu hình) | 1 | 0.000000 | 0.000000 | 1.0000 | 0.6046 | 0.4022 | -36.55 | 2514/2514 |
| tc=2 | 2 | 0.283158 | 0.346856 | 0.9331 | 0.6042 | 0.4069 | -12.26 | 2252/2514 |
| tc=-1 (mặc định) | -1 | 0.299853 | 0.335194 | 0.9409 | 0.6023 | 0.4065 | +30.79 | 2254/2514 |

Trên máy này, `thread_count` **có** làm thay đổi kết quả: đổi cấu hình làm toàn bộ prediction thay đổi (train 25,008/25,008 và holdout 5,028/5,028 dòng lệch quá `1e-12` ở cả `tc=2` và `tc=-1` so với `tc=1`), kéo theo chọn top 50% và net R đổi. Hai lần chạy cùng cấu hình `tc=1` cho prediction giống hệt nhau (train max|Δp| = 0.0, holdout max|Δp| = 0.0), nên khác biệt đến từ việc đổi cấu hình chứ không phải bất định giữa hai lần chạy. Ảnh hưởng nằm ở bước train: cùng model đã fit, đổi prediction `thread_count` 1/2/-1 cho prediction giống hệt nhau (max|Δp| = 0.0). CatBoost mô tả `thread_count` là tham số tốc độ không ảnh hưởng kết quả; phép đo trên máy này xác nhận điều đó ở bước prediction nhưng không xác nhận ở bước train (holdout ROC-AUC từ 0.6023 đến 0.6046, top-50 net R từ -36.55 đến +30.79 R). Phạm vi: một máy, một build CatBoost, một dataset, một seed; thí nghiệm không quy hết sai lệch liên máy cho `thread_count` (xem giới hạn trong JSON).
