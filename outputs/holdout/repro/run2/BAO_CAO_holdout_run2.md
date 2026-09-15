# BÁO CÁO KẾT QUẢ TRAIN VÀ HOLDOUT

> Phạm vi: tổng hợp artifact đã chốt của 4 cách chia trên tập 80% đầu và kết
> quả holdout niêm phong. Báo cáo chỉ ghi số liệu, quyết định triển khai và bằng
> chứng kiểm chứng; không biện luận hay kết luận thay báo cáo chính.

## Lịch sử và lý do chạy lại holdout

1. Holdout ban đầu được mở và chạy theo bốn Stage sau khi feature,
   hyperparameter và tỷ lệ giữ lệnh đã chốt từ bước train.
2. Holdout phải chạy lại vì phát hiện cấu hình CatBoost chưa ghim
   `thread_count`; số luồng mặc định phụ thuộc cấu hình CPU và có thể tạo sai
   lệch số học giữa môi trường. Lần chạy lại chỉ thêm `thread_count=1` như thiết
   lập kỹ thuật; không đổi feature, hyperparameter mô hình, tập train, tập
   holdout hay luật chọn top-k.
3. Artifact của bốn nhánh trên 80% đầu được khôi phục và giữ nguyên theo bản bàn
   giao. Các số holdout trong báo cáo này lấy từ lần chạy lại đã kiểm chứng; lý
   do chạy lại và giới hạn bằng chứng được ghi công khai để không xem đây là một
   lần thử mô hình mới nhằm chọn kết quả đẹp hơn.

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

Ghi chú: bốn dòng Cách 1–3 là trung bình chỉ số theo fold sau khi bỏ khúc 1 để
các nhánh được đo trên cùng phạm vi khúc 2–5. Dòng Holdout được tính một lần
trên toàn bộ 5,028 mẫu holdout, không phải trung bình theo fold.

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

Hai biểu đồ được dựng bằng Python/Plotly ở dạng bậc thang: equity chỉ thay đổi
tại `close_time`, không nội suy tuyến tính giữa hai lần đóng lệnh. Trục dọc là R
cộng dồn từ 0. File HTML chứa Plotly nội tuyến nên mở độc lập được.

## Phần 3 — Quyết định triển khai

1. Dùng `ceil(n × keep_pct / 100)` cho số lệnh giữ lại: top 50% của 20,007 lệnh là 10,004; top 50% của 5,028 lệnh là 2,514.
2. Xếp `probability` giảm dần, dùng `row_id` tăng dần để phá hòa; vũ trụ lệnh baseline giữ cố định nên lệnh bị loại không làm đổi tín hiệu sau đó.
3. Equity ghi nhận tại `close_time`; các lệnh đóng cùng lúc được cộng thành một điểm rồi mới cập nhật đường vốn.
4. Bốn dòng train ở Bảng 1 và năm dòng đầu Bảng 2 lấy nguyên từ artifact bàn giao đã niêm phong; dòng holdout và sweep lấy từ Stage 3.
5. Thêm `thread_count=1` như thiết lập kỹ thuật để loại số luồng CPU như một nguồn sai lệch đã biết; không xem đây là hyperparameter mô hình và không dùng riêng phép thử này để kết luận nguyên nhân sai lệch liên máy.
6. Dùng đường bậc thang ngang-rồi-dọc (`hv`) để equity giữ nguyên giữa hai mốc đóng lệnh và chỉ nhảy tại thời điểm R được ghi nhận.

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
| Tái lập từng byte toàn pipeline holdout | **CHƯA XÁC NHẬN** — bằng chứng hiện chỉ xác nhận prediction của hai lượt train trên cùng máy; không tuyên bố toàn bộ Stage 1–4 giống từng byte |

Môi trường sinh artifact: Python 3.12.14; CatBoost 1.2.10; scikit-learn
1.9.0; pandas 3.0.5; NumPy 2.5.2.

## Phần 5 — Bảng file sinh ra

| File | Giai đoạn | Chứa gì | Quy mô | Kích thước |
|---|---|---|---|---|
| [`dataset_catboost_full_regenerated.csv`](../outputs/holdout/stage1/dataset_catboost_full_regenerated.csv) | 1 | Dataset tái sinh trên toàn bộ lịch sử để kiểm tra và tách holdout | 30,040 dòng | 9,800,880 B |
| [`dataset_catboost_holdout.csv`](../outputs/holdout/stage1/dataset_catboost_holdout.csv) | 1 | Dataset holdout từ mốc 2025-02-08 15:30:00 | 5,028 dòng | 1,648,262 B |
| [`stage1_validation_report.json`](../outputs/holdout/stage1/stage1_validation_report.json) | 1 | Số liệu kiểm khóa cũ và đối chiếu 23 feature | artifact | 888 B |
| [`catboost_final_holdout_run1.cbm`](../outputs/holdout/stage2/catboost_final_holdout_run1.cbm) | 2 | Model CatBoost cuối dùng để chấm holdout | artifact | 1,129,392 B |
| [`catboost_final_holdout_run2.cbm`](../outputs/holdout/stage2/catboost_final_holdout_run2.cbm) | 2 | Model train độc lập lần hai để kiểm lặp | artifact | 1,129,392 B |
| [`stage2_reproducibility_report.json`](../outputs/holdout/stage2/stage2_reproducibility_report.json) | 2 | So sánh hai lượt train trên cùng máy | artifact | 722 B |
| [`train_predictions_run1.npy`](../outputs/holdout/stage2/train_predictions_run1.npy) | 2 | Prediction trên tập train của lượt 1 | 25,008 giá trị | 200,192 B |
| [`train_predictions_run2.npy`](../outputs/holdout/stage2/train_predictions_run2.npy) | 2 | Prediction trên tập train của lượt 2 | 25,008 giá trị | 200,192 B |
| [`train_run1_report.json`](../outputs/holdout/stage2/train_run1_report.json) | 2 | Cấu hình, phiên bản, hash và kiểm biên của lượt train 1 | artifact | 1,949 B |
| [`train_run2_report.json`](../outputs/holdout/stage2/train_run2_report.json) | 2 | Cấu hình, phiên bản, hash và kiểm biên của lượt train 2 | artifact | 1,949 B |
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
| [`equity-curve-chunk2-5-top50.html`](../outputs/holdout/stage4/equity-curve-chunk2-5-top50.html) | 4 | Biểu đồ bậc thang baseline và 4 nhánh trên khúc 2–5 | artifact | 5,406,329 B |
| [`equity-curve-holdout-top50.html`](../outputs/holdout/stage4/equity-curve-holdout-top50.html) | 4 | Biểu đồ bậc thang baseline và top 50% trên holdout | artifact | 4,434,195 B |
| [`holdout_sweep_20_80.csv`](../outputs/holdout/stage4/holdout_sweep_20_80.csv) | 4 | Bảng tài chính holdout theo bảy mức giữ lệnh | 7 dòng | 366 B |
| [`implementation_decisions.md`](../outputs/holdout/stage4/implementation_decisions.md) | 4 | Các quyết định triển khai Stage 4 | artifact | 2,285 B |
| [`stage4_manifest.json`](../outputs/holdout/stage4/stage4_manifest.json) | 4 | Danh sách output và số điểm trên từng đường vốn | artifact | 592 B |
| [`stage4_tables.md`](../outputs/holdout/stage4/stage4_tables.md) | 4 | Ba bảng kết quả ở định dạng Markdown | artifact | 1,405 B |
| [`table1_classification_metrics.csv`](../outputs/holdout/stage4/table1_classification_metrics.csv) | 4 | Bảng chỉ số phân loại train và holdout | 5 dòng | 213 B |
| [`table2_financial_metrics_top50.csv`](../outputs/holdout/stage4/table2_financial_metrics_top50.csv) | 4 | Bảng tài chính tại mức giữ top 50% | 7 dòng | 409 B |
