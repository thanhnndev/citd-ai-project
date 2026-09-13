# BÁO CÁO KẾT QUẢ TRAIN VÀ HOLDOUT

> Phạm vi: tổng hợp artifact đã chốt của 4 cách chia trên tập 80% đầu và kết
> quả holdout niêm phong. Báo cáo chỉ ghi số liệu, quyết định triển khai và bằng
> chứng kiểm chứng; không biện luận hay kết luận thay báo cáo chính.

## Phần 1 — Số liệu

### 1.1. Cấu hình train cuối

| Thuộc tính | Giá trị |
|---|---|
| Tập train trước/sau purge + embargo | 25,008 / 25,008 dòng |
| Tập holdout | 5,028 dòng; không bỏ dòng |
| Feature đầu vào | 23 `FEATURES`; không đưa `META` vào `X` |
| Hyperparameter và thiết lập kỹ thuật | `iterations=1000` · `learning_rate=0.05` · `depth=6` · `l2_leaf_reg=3.0` · `auto_class_weights=Balanced` · `eval_metric=AUC` · `random_seed=42` · `thread_count=1` |

### 1.2. Chỉ số phân loại

| Cách chia | ROC-AUC | F1 @0.5 |
|---|---|---|
| Cách 1 — Random K-Fold | 0.8595 | 0.6525 |
| Cách 1b — Grouped K-Fold | 0.7475 | 0.5105 |
| Cách 2 — Walk-forward | 0.5867 | 0.3267 |
| Cách 3 — WF + Purge/Embargo | 0.5948 | 0.3304 |
| Holdout | 0.6046 | 0.4022 |

### 1.3. Chỉ số tài chính, giữ top 50%

|  | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---|---|---|---|---|
| Baseline (khúc 2–5) | 20,007 | +3,358.3 | 236.1 | 1.294 | 31.7 |
| Cách 1 | 10,004 | +6,998.3 | 75.7 | 2.563 | 44.9 |
| Cách 1b | 10,004 | +4,724.3 | 99.5 | 1.942 | 39.2 |
| Cách 2 | 10,004 | +2,207.3 | 205.3 | 1.423 | 34.9 |
| Cách 3 | 10,004 | +2,253.7 | 154.1 | 1.435 | 35.2 |
| Baseline holdout | 5,028 | +245.93 | 305.32 | 1.0992 | 35.28 |
| Holdout, top 50% | 2,514 | -36.55 | 263.25 | 0.9718 | 34.81 |

### 1.4. Sweep holdout 20–80%

| Lọc | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---|---|---|---|---|
| Top 20% | 1,006 | -30.01 | 125.38 | 0.9385 | 36.18 |
| Top 30% | 1,509 | -50.01 | 210.77 | 0.9343 | 35.52 |
| Top 40% | 2,012 | -53.17 | 251.56 | 0.9485 | 35.04 |
| Top 50% | 2,514 | -36.55 | 263.25 | 0.9718 | 34.81 |
| Top 60% | 3,017 | +73.16 | 233.95 | 1.0474 | 35.60 |
| Top 70% | 3,520 | +186.92 | 197.24 | 1.1040 | 35.77 |
| Top 80% | 4,023 | +259.91 | 219.02 | 1.1278 | 35.52 |

## Phần 2 — Biểu đồ

- [Biểu đồ 1 — Baseline và 4 nhánh top 50%, khúc 2–5](../outputs/holdout/stage4/equity-curve-chunk2-5-top50.html)
- [Biểu đồ 2 — Baseline holdout và Holdout top 50%](../outputs/holdout/stage4/equity-curve-holdout-top50.html)

Hai biểu đồ được dựng bằng Python/Plotly, trục ngang là `close_time`, trục dọc
là R cộng dồn từ 0. File HTML chứa Plotly nội tuyến nên mở độc lập được.

## Phần 3 — Quyết định triển khai

1. Dùng `ceil(n × keep_pct / 100)` cho số lệnh giữ lại: top 50% của 20,007 lệnh là 10,004; top 50% của 5,028 lệnh là 2,514.
2. Xếp `probability` giảm dần, dùng `row_id` tăng dần để phá hòa; vũ trụ lệnh baseline giữ cố định nên lệnh bị loại không làm đổi tín hiệu sau đó.
3. Equity ghi nhận tại `close_time`; các lệnh đóng cùng lúc được cộng thành một điểm rồi mới cập nhật đường vốn.
4. Bốn dòng train ở Bảng 1 và năm dòng đầu Bảng 2 lấy nguyên từ artifact bàn giao đã niêm phong; dòng holdout và sweep lấy từ Stage 3.
5. Thêm `thread_count=1` như thiết lập kỹ thuật để loại số luồng CPU như một nguồn sai lệch đã biết; không xem đây là hyperparameter mô hình và không dùng riêng phép thử này để kết luận nguyên nhân sai lệch liên máy.

## Phần 4 — Kiểm chứng

| Phép kiểm | Kết quả |
|---|---|
| Dataset tái sinh trước holdout | **PASS** — 25,008/25,008 khóa cũ; thiếu 0 |
| Giá trị feature | **PASS** — 575,184 ô đã so; lệch 0 |
| Purge / embargo tại biên holdout | **PASS** — cắt 0 / 0 dòng; train còn 25,008 |
| Holdout không bị sửa khi train/chấm điểm | **PASS** — SHA-256 trước/sau giữ nguyên |
| Replay baseline so với tradelist bàn giao | **PASS** — 5,028/5,028 lệnh; trường lệch: không có |
| Lặp train trên cùng máy | **PASS** — 25,008 predictions giống hệt; max abs diff = 0.0 |
| File model `.cbm` | **GHI NHẬN** — SHA-256 hai file khác nhau; binary model không phải tiêu chí PASS |
| Kiểm chứng liên máy | **GIỚI HẠN** — teammate Windows báo sai số prediction trong `1e-15`; chưa có artifact đối chứng được commit và chưa có thí nghiệm chỉ thay `thread_count` |

Môi trường sinh artifact: Python 3.12.14; CatBoost 1.2.10; scikit-learn
1.9.0; pandas 3.0.5; NumPy 2.5.2.

## Phần 5 — Bảng file sinh ra

| File | Giai đoạn | Quy mô | Kích thước |
|---|---|---|---|
| [`dataset_catboost_full_regenerated.csv`](../outputs/holdout/stage1/dataset_catboost_full_regenerated.csv) | 1 | 30,040 dòng | 9,800,880 B |
| [`dataset_catboost_holdout.csv`](../outputs/holdout/stage1/dataset_catboost_holdout.csv) | 1 | 5,028 dòng | 1,648,262 B |
| [`stage1_validation_report.json`](../outputs/holdout/stage1/stage1_validation_report.json) | 1 | artifact | 888 B |
| [`catboost_final_holdout_run1.cbm`](../outputs/holdout/stage2/catboost_final_holdout_run1.cbm) | 2 | artifact | 1,129,392 B |
| [`catboost_final_holdout_run2.cbm`](../outputs/holdout/stage2/catboost_final_holdout_run2.cbm) | 2 | artifact | 1,129,392 B |
| [`stage2_reproducibility_report.json`](../outputs/holdout/stage2/stage2_reproducibility_report.json) | 2 | artifact | 722 B |
| [`train_predictions_run1.npy`](../outputs/holdout/stage2/train_predictions_run1.npy) | 2 | 25,008 giá trị | 200,192 B |
| [`train_predictions_run2.npy`](../outputs/holdout/stage2/train_predictions_run2.npy) | 2 | 25,008 giá trị | 200,192 B |
| [`train_run1_report.json`](../outputs/holdout/stage2/train_run1_report.json) | 2 | artifact | 1,863 B |
| [`train_run2_report.json`](../outputs/holdout/stage2/train_run2_report.json) | 2 | artifact | 1,863 B |
| [`holdout_fixed_trade_universe_scored.csv`](../outputs/holdout/stage3/holdout_fixed_trade_universe_scored.csv) | 3 | 5,028 dòng | 2,263,531 B |
| [`holdout_scored.csv`](../outputs/holdout/stage3/holdout_scored.csv) | 3 | 5,028 dòng | 1,661,606 B |
| [`holdout_trades_top20.csv`](../outputs/holdout/stage3/holdout_trades_top20.csv) | 3 | 1,006 dòng | 454,493 B |
| [`holdout_trades_top30.csv`](../outputs/holdout/stage3/holdout_trades_top30.csv) | 3 | 1,509 dòng | 680,410 B |
| [`holdout_trades_top40.csv`](../outputs/holdout/stage3/holdout_trades_top40.csv) | 3 | 2,012 dòng | 906,297 B |
| [`holdout_trades_top50.csv`](../outputs/holdout/stage3/holdout_trades_top50.csv) | 3 | 2,514 dòng | 1,132,050 B |
| [`holdout_trades_top60.csv`](../outputs/holdout/stage3/holdout_trades_top60.csv) | 3 | 3,017 dòng | 1,358,175 B |
| [`holdout_trades_top70.csv`](../outputs/holdout/stage3/holdout_trades_top70.csv) | 3 | 3,520 dòng | 1,584,187 B |
| [`holdout_trades_top80.csv`](../outputs/holdout/stage3/holdout_trades_top80.csv) | 3 | 4,023 dòng | 1,810,896 B |
| [`stage3_backtest_summary.csv`](../outputs/holdout/stage3/stage3_backtest_summary.csv) | 3 | 8 dòng | 654 B |
| [`stage3_report.json`](../outputs/holdout/stage3/stage3_report.json) | 3 | artifact | 3,288 B |
| [`equity-curve-chunk2-5-top50.html`](../outputs/holdout/stage4/equity-curve-chunk2-5-top50.html) | 4 | artifact | 5,406,264 B |
| [`equity-curve-holdout-top50.html`](../outputs/holdout/stage4/equity-curve-holdout-top50.html) | 4 | artifact | 4,434,169 B |
| [`holdout_sweep_20_80.csv`](../outputs/holdout/stage4/holdout_sweep_20_80.csv) | 4 | 7 dòng | 366 B |
| [`implementation_decisions.md`](../outputs/holdout/stage4/implementation_decisions.md) | 4 | artifact | 2,265 B |
| [`stage4_manifest.json`](../outputs/holdout/stage4/stage4_manifest.json) | 4 | artifact | 592 B |
| [`stage4_tables.md`](../outputs/holdout/stage4/stage4_tables.md) | 4 | artifact | 1,405 B |
| [`table1_classification_metrics.csv`](../outputs/holdout/stage4/table1_classification_metrics.csv) | 4 | 5 dòng | 213 B |
| [`table2_financial_metrics_top50.csv`](../outputs/holdout/stage4/table2_financial_metrics_top50.csv) | 4 | 7 dòng | 409 B |
