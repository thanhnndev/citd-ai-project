# BÁO CÁO KẾT QUẢ HOLDOUT

> **Ghi chú lần chạy lại:** Báo cáo này được sinh lại trên môi trường `uv`
> (Python 3.12.14) ngày 2026-09-12. Bốn dòng đầu Bảng 1 và năm dòng đầu Bảng 2
> là số bàn giao cố định; các dòng Holdout, sweep và hash model bên dưới phản
> ánh lần chạy mới nhất. Chi tiết thay đổi xem mục CHANGELOG trong `README.md`.

## Phần 1 — Số liệu

### Bảng 1 — Chỉ số phân loại

|  | ROC-AUC | F1 @0.5 |
|---|---|---|
| Cách 1 — Random K-Fold | 0.8595 | 0.6525 |
| Cách 1b — Grouped K-Fold | 0.7475 | 0.5105 |
| Cách 2 — Walk-forward | 0.5867 | 0.3267 |
| Cách 3 — WF + Purge/Embargo | 0.5948 | 0.3304 |
| Holdout | 0.6023152558719406 | 0.4064693317058285 |

### Bảng 2 — Chỉ số tài chính, giữ top 50%

|  | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---|---|---|---|---|
| Baseline (khúc 2–5) | 20,007 | +3,358.3 | 236.1 | 1.294 | 31.7 |
| Cách 1 | 10,004 | +6,998.3 | 75.7 | 2.563 | 44.9 |
| Cách 1b | 10,004 | +4,724.3 | 99.5 | 1.942 | 39.2 |
| Cách 2 | 10,004 | +2,207.3 | 205.3 | 1.423 | 34.9 |
| Cách 3 | 10,004 | +2,253.7 | 154.1 | 1.435 | 35.2 |
| Baseline holdout | 5,028 | +245.9296826590 | 305.3174286700 | 1.0991525948 | 35.2824184566 |
| Holdout, top 50% | 2,514 | +30.7859222571 | 265.5417179040 | 1.0238644238 | 35.2028639618 |

### Sweep holdout 20–80%

| Lọc | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---|---|---|---|---|
| Top 20% | 1,006 | -11.0624231964 | 114.3400755380 | 0.9772911901 | 36.0834990060 |
| Top 30% | 1,509 | -1.7690864744 | 169.0056825710 | 0.9976356343 | 36.1829025845 |
| Top 40% | 2,012 | +28.3871682167 | 210.6927181210 | 1.0279104530 | 35.8349900596 |
| Top 50% | 2,514 | +30.7859222571 | 265.5417179040 | 1.0238644238 | 35.2028639618 |
| Top 60% | 3,017 | +35.2065388225 | 279.9499440880 | 1.0224013273 | 34.6370566788 |
| Top 70% | 3,520 | +122.0766849210 | 276.0070566960 | 1.0671919382 | 34.8579545455 |
| Top 80% | 4,023 | +160.3814321470 | 262.5306925160 | 1.0780902018 | 34.8993288591 |

## Phần 2 — Biểu đồ

- [Biểu đồ 1 — Baseline và 4 nhánh top 50%, khúc 2–5](../outputs/holdout/stage4/equity-curve-chunk2-5-top50.html)
- [Biểu đồ 2 — Baseline holdout và Holdout top 50%](../outputs/holdout/stage4/equity-curve-holdout-top50.html)

## Phần 3 — Quyết định triển khai

1. Trục thời gian của cả hai biểu đồ dùng `close_time`, vì R của một lệnh chỉ hoàn tất tại thời điểm đóng lệnh.
2. Các lệnh có cùng `close_time` được sắp xếp ổn định theo `ticket`, cộng R của chúng thành một điểm thời gian duy nhất; điểm đó là equity sau toàn bộ lệnh đóng cùng lúc.
3. Mỗi đường được thêm điểm R = 0 ngay trước `close_time` đầu tiên, để đường vốn bắt đầu từ 0 mà không thay đổi mốc dữ liệu giao dịch.
4. Biểu đồ 1 chỉ nhận các dòng `chunk` 2–5. Top 50% của từng nhánh dùng `ceil(20007 × 50%) = 10004`, xếp xác suất giảm dần rồi `row_id` tăng dần khi hòa điểm — đúng luật đã chốt ở bước trước.
5. Biểu đồ 2 bắt đầu lại từ R = 0 tại holdout, dùng trực tiếp vũ trụ baseline và danh sách Top 50% đã lưu từ Giai đoạn 3; không nối với equity của khúc 2–5.
6. Các khoảng thời gian không có lệnh được nối bằng đường liên tục giữa các điểm đóng lệnh; không chèn giao dịch hoặc điểm equity giả vào các khoảng trống.
7. Purge và embargo đều trả về 0 dòng, nên train giữ nguyên 25,008 dòng; điều kiện lọc được chạy trước khi quyết định không loại dòng nào.
8. So khớp giá baseline dùng sai số tuyệt đối `5e-4`, theo validator bàn giao; giá vào, giá ra và R được kiểm dưới cùng ngưỡng này.
9. Chấm điểm holdout lấy danh sách 23 `FEATURES` trực tiếp từ `build_features.py`; không tự liệt kê cột.

## Phần 4 — Kiểm chứng

1. Bản sinh lại trước mốc chứa đủ **25,008/25,008** khóa của dataset cũ; số khóa thiếu: **0**. Bản sinh lại có 25,012 dòng trước mốc, dataset cũ có 25,008 dòng.
2. **23** cột `FEATURES` được so sánh trên **575,184** ô; số ô lệch: **0**.
3. Purge cắt **0** dòng; embargo cắt **0** dòng; tổng dòng bị loại bởi một trong hai điều kiện: **0**; số dòng train sau lọc: **25,008**.
4. Backtest holdout khớp `tradelist_pyramid_local.csv`: **5,028/5,028** lệnh; `passed: true`; các trường lệch: không có.
5. Hai lần train: model SHA-256 khác nhau (`995d66ef93e625b053e2dffaa16a3953edcc410abffbbaa2b8b077647f50bbf3` và `116a4aa6cae3a4f57bad6294c2071bd3e045206442c6ccc37fd97a37fe85fa06`) do metadata build-info nhúng trong file; **25,008** predictions giống hệt (`max abs difference = 0.0`), hash predictions giống nhau và cấu trúc `oblivious_trees` giống hệt.
6. Phiên bản: Python **3.12.14**; CatBoost **1.2.10**; scikit-learn **1.9.0**; pandas **3.0.5**; NumPy **2.5.2**.

## Phần 5 — Bảng file sinh ra

| File | Giai đoạn | Chứa gì | Dòng / kích thước |
|---|---:|---|---|
| `dataset_catboost_full_regenerated.csv` | 1 | Dataset tái sinh toàn bộ | 30,040 dòng; 9,830,921 B |
| `dataset_catboost_holdout.csv` | 1 | Dataset holdout | 5,028 dòng; 1,653,291 B |
| `stage1_validation_report.json` | 1 | Kết quả kiểm dataset | 1,011 B |
| `verify_and_split_holdout.py` | 1 | Script kiểm và tách holdout | 3,781 B |
| `catboost_final_holdout_run1.cbm` | 2 | Model chính thức | 1,129,264 B |
| `catboost_final_holdout_run2.cbm` | 2 | Model train độc lập lần 2 | 1,129,264 B |
| `train_predictions_run1.npy` | 2 | Xác suất train lần 1 | 25,008 giá trị; 200,192 B |
| `train_predictions_run2.npy` | 2 | Xác suất train lần 2 | 25,008 giá trị; 200,192 B |
| `train_run1_report.json` | 2 | Log train lần 1 | 1,940 B |
| `train_run2_report.json` | 2 | Log train lần 2 | 1,940 B |
| `stage2_reproducibility_report.json` | 2 | Kết quả tái lập model | 399 B |
| `train_final_model_holdout.py` | 2 | Script train và kiểm tái lập | 6,468 B |
| `holdout_scored.csv` | 3 | Holdout kèm xác suất CatBoost | 5,028 dòng; 1,666,608 B |
| `holdout_fixed_trade_universe_scored.csv` | 3 | Vũ trụ lệnh holdout cố định kèm điểm | 5,028 dòng; 2,268,533 B |
| `holdout_trades_top20.csv` | 3 | Lệnh holdout Top 20% | 1,006 dòng; 455,214 B |
| `holdout_trades_top30.csv` | 3 | Lệnh holdout Top 30% | 1,509 dòng; 682,034 B |
| `holdout_trades_top40.csv` | 3 | Lệnh holdout Top 40% | 2,012 dòng; 908,772 B |
| `holdout_trades_top50.csv` | 3 | Lệnh holdout Top 50% | 2,514 dòng; 1,134,776 B |
| `holdout_trades_top60.csv` | 3 | Lệnh holdout Top 60% | 3,017 dòng; 1,361,199 B |
| `holdout_trades_top70.csv` | 3 | Lệnh holdout Top 70% | 3,520 dòng; 1,587,848 B |
| `holdout_trades_top80.csv` | 3 | Lệnh holdout Top 80% | 4,023 dòng; 1,814,961 B |
| `stage3_backtest_summary.csv` | 3 | Bảng metrics baseline và sweep holdout | 8 dòng; 660 B |
| `stage3_report.json` | 3 | Báo cáo chấm điểm, đối chiếu và backtest | 3,408 B |
| `stage3_holdout_backtest.py` | 3 | Script chấm điểm và backtest holdout | 10,148 B |
| `table1_classification_metrics.csv` | 4 | Bảng 1 | 5 dòng; 242 B |
| `table2_financial_metrics_top50.csv` | 4 | Bảng 2 | 7 dòng; 477 B |
| `holdout_sweep_20_80.csv` | 4 | Bảng sweep holdout | 7 dòng; 584 B |
| `stage4_tables.md` | 4 | Ba bảng Stage 4 | 1,731 B |
| `equity-curve-chunk2-5-top50.html` | 4 | Biểu đồ 1 HTML | 5,409,902 B |
| `equity-curve-holdout-top50.html` | 4 | Biểu đồ 2 HTML | 4,438,066 B |
| `implementation_decisions.md` | 4 | Quyết định triển khai | 1,249 B |
| `stage4_manifest.json` | 4 | Manifest Stage 4 | 614 B |
| `stage4_build_tables_and_charts.py` | 4 | Script dựng bảng và biểu đồ | 9,008 B |
