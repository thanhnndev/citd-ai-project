# BÁO CÁO KỸ THUẬT — PYRAMID BTCUSD / CATBOOST

Tài liệu này mô tả bản code đã chạy: dữ liệu đi qua những file nào, điều kiện nào chặn một lần chạy sai và artifact nào dùng để kiểm lại. Phần lập luận nghiên cứu nằm ở [Scientific report](../deliverables/Scientific_report_Nhom11.pdf); số liệu chi tiết nằm trong [báo cáo bước 4](BAO_CAO_KET_QUA_STEP4.md) và [báo cáo holdout](BAO_CAO_KET_QUA_HOLDOUT.md). Các số chính dưới đây lấy từ cây `outputs/step4_thread1/` với `thread_count=1`. Artifact bàn giao cũ không ghim số luồng được giữ riêng, không cộng lẫn.

## 1. Hệ thống thực sự làm gì

`PyramidStrategy` tạo lệnh mua BTCUSD. Mỗi tín hiệu có một lệnh gốc (`leg=0`) và ba lệnh nhồi (`leg=1..3`) cùng `origin_bar`. CatBoost không phát tín hiệu mua bán; nó chấm xác suất cho những lệnh mà chiến lược đã tạo. Bốn phép chia dữ liệu dùng chung dataset, 23 feature và cấu hình model để đo ảnh hưởng của cách đánh giá.

Backtest dùng tập ứng viên cố định. Khi một lệnh bị lọc, luồng `shadow` vẫn chạy chiến lược gốc để sinh đúng các ứng viên về sau; luồng `executor` chỉ mở những lệnh được giữ. Cách này kiểm chất lượng chọn lệnh trên cùng một tập ứng viên. Nó không mô phỏng một chiến lược live mà việc bỏ lệnh gốc có thể làm thay đổi các tín hiệu kế tiếp. Logic nằm trong [`backtest_pyramid_local.py`](../src/citd_ml/backtest/backtest_pyramid_local.py).

Đường đi chính của dữ liệu:

1. BTCUSD M1 → resample M15; giữ ánh xạ M1 trong từng bar để xét thứ tự chạm stop/target.
2. Replay `PyramidStrategy` → vị thế ứng viên; gán nhãn triple barrier và tính feature tại entry → dataset.
3. Sort dataset, chia fold, fit CatBoost từng fold → xác suất out-of-fold (OOF).
4. Ghép OOF với tradelist baseline theo `(origin_bar, entry_bar, leg)` → chọn top-k và chạy backtest lọc.
5. Train model cuối trên dữ liệu trước mốc; chấm holdout, replay baseline, lập bảng/hình và đối chiếu hai lượt chạy.

## 2. Hợp đồng dữ liệu

Nguồn giá là `data/raw/BTCUSD_m1_2018_to_now.csv`. File M1 khoảng 209 MB không nằm trong repo; cách khôi phục và schema ở [`data/raw/README.md`](../data/raw/README.md). Không có file này vẫn đọc được kết quả đã lưu và train từ dataset đóng băng; sinh dataset hoặc replay backtest cần khôi phục đúng nguồn M1.

| Tập | Ranh giới và quy mô | Vai trò |
|---|---|---|
| Train đóng băng | 25.008 lệnh, 31 cột; 6.252 họ, mỗi họ 4 leg | [`data/processed/dataset_catboost.csv`](../data/processed/dataset_catboost.csv) là đầu vào mặc định của code train |
| Lệnh biên | 4 entry trước mốc nhưng `label_end_time` qua mốc | Không vào train hoặc holdout |
| Holdout | 5.028 lệnh từ `2025-02-08 15:30:00`, bar M15 `199968` | Chấm bằng model cuối; không dùng để fit |
| Toàn lịch sử tái sinh | 30.040 lệnh = 25.008 + 4 + 5.028 | Đầu vào đối chiếu ở Stage 1 |

Mốc holdout được chốt theo thời gian. Theo *số lệnh*, 5.028/30.040 là khoảng 16,74%; vì vậy không gọi tập này là đúng 20% số lệnh như ước lượng trong [đề cương ban đầu](tom_tat_idea_goc.md). Bốn lệnh biên và kiểm tra khóa được ghi trong [báo cáo holdout, mục 1.7](BAO_CAO_KET_QUA_HOLDOUT.md).

Dataset có 23 cột `FEATURES`, một cột `label` và 7 cột `META`: `entry_time`, `label_end_time`, `origin_bar`, `entry_bar`, `entry_price`, `leg`, `entry_vs_base_R`. `row_id` chỉ được thêm sau khi sort để định danh dòng và phá hòa top-k. `META`, nhãn và giá tuyệt đối không đi vào ma trận `X`. Danh sách feature chính xác lấy từ [`build_features.py`](../src/citd_ml/features/build_features.py), không suy từ mọi cột CSV:

| Nhóm | Cột |
|---|---|
| Hàng rào | `breakeven_R` |
| Biến động | `atr14_pct`, `atr_ratio_14_90`, `vol20`, `vol200`, `vol_ratio_20_200`, `range_pct` |
| Giá và xu hướng | `dist_ema20_atr`, `dist_ema50_atr`, `dist_ema200_atr`, `ret20_atr`, `ret50_atr`, `pos_in_range50`, `dist_hh20_atr` |
| Tín hiệu | `mom14`, `mom_diff`, `vwap_dist_atr`, `vwap_slope_atr`, `rsi14` |
| Khối lượng và entry | `vol_ratio_volume`, `gap_open_atr` |
| Thời điểm | `hour`, `dow` |

Feature thị trường của mỗi leg đọc bar M15 đã đóng (`entry_bar - 1`); `gap_open_atr` còn đọc giá mở cửa bar entry, đã có lúc vào lệnh. `leg` và `entry_vs_base_R` được lưu để truy vết gia đình lệnh nhưng không train. Lý do chọn/loại cột ở [báo cáo feature](bao_cao_feature.md).

Nhãn triple barrier khác kết quả tài chính. Lower là giá vào trừ 0,6%; upper là giá vào cộng `3,8 × ATR(90)` đóng băng trước entry; horizon là 50 bar M15. [`label_one_entry`](../src/citd_ml/labeling/triple_barrier.py) duyệt M1 theo thứ tự: lower chạm trước hoặc cả hai mốc chạm trong cùng nến M1 thì nhãn 0; upper chạm và không có lower chạm trong bar M15 đó thì nhãn 1; hết horizon chưa chạm thì nhãn 0. Horizon thiếu dữ liệu bị loại khi xuất dataset. Nhãn không áp dụng lệnh đóng thứ Sáu của chiến lược; sơ đồ quy tắc ở [`thong_so_triple_barrier.png`](thong_so_triple_barrier.png).

## 3. Code và điểm kiểm soát

Các bảng dưới tập trung vào hàm nằm trên đường chạy chính. Hàm trả DataFrame/mảng chỉ tạo giá trị trong bộ nhớ; file chỉ xuất hiện tại entrypoint hoặc hàm ghi được nêu rõ.

### 3.1. Từ M1 đến dataset

| File / hàm | Nhận và xử lý; điều kiện chặn | Trả hoặc ghi |
|---|---|---|
| [`pyramid_strategy.py`](../src/citd_ml/strategy/pyramid_strategy.py) · `prepare` | OHLCV M15 → Momentum(14), VWAP(142), ATR(90) | Mảng chỉ báo |
| `PyramidStrategy.entry_signal` | Bar hiện tại, chỉ báo, `base_open`; cần đủ warm-up, không còn lệnh gốc mở, Momentum giảm và VWAP tăng trên bar đã đóng | Có/không mở họ lệnh |
| `PyramidStrategy.open_bar` | Mở leg đến hạn, kiểm tín hiệu mới, hẹn ba leg kế tiếp; `_open` dùng giá mở bar entry và ATR của bar trước | Cập nhật `positions`/`pending`; không trả giá trị |
| `Position.trail`; `PyramidStrategy.scan_bar`, `friday_close` | Dời stop chỉ theo hướng tăng; duyệt M1 để thoát stop/target; đóng vị thế còn lại vào thứ Sáu từ 20:40 | Cập nhật vị thế; hai hàm đóng lệnh trả sự kiện cho tradelist |
| [`triple_barrier.py`](../src/citd_ml/labeling/triple_barrier.py) · `prepare_data` | Kiểm cột M1, parse thời gian/giá, cắt tại mốc được truyền, resample M15 và lập lát M1 | M1/M15, mảng giá M1 và biên lát `lo/hi` |
| `calculate_barriers` | Giá vào, ATR, tham số stop/target → tính hai ngưỡng | Upper, lower |
| `label_one_entry` | Hai ngưỡng và M1 trong tối đa 50 bar M15 → xét thứ tự chạm | Nhãn, số bar, thời điểm chạm, cờ incomplete |
| `run_strategy_and_label` | Replay entry, gán nhãn từng leg và chạy logic thoát lệnh gốc để giữ đúng state | Bảng nhãn thô và số lệnh đã đóng trong bộ nhớ |
| `prepare_handoff_output`; `main` | Loại horizon chưa đủ, ép schema, kiểm nhãn nhị phân rồi ghi file | [`triple_barrier_labels.csv`](../data/processed/triple_barrier_labels.csv) |
| [`build_features.py`](../src/citd_ml/features/build_features.py) · `resolve_positions` | Replay đúng entry của chiến lược; gọi cùng hàm hàng rào/nhãn cho từng vị thế | Bảng vị thế, nhãn và khóa lệnh |
| `build_features` | Tính 23 feature tại entry; chặn `entry_bar < 1` để không đọc nhầm bar cuối chuỗi | Bảng feature và metadata |
| `build_features.main` | Tạo `label_end_time`, loại horizon thiếu, sort ổn định, chọn `FEATURES + label + META` | [`dataset_catboost.csv`](../data/processed/dataset_catboost.csv) theo `--output` |

### 3.2. Train và OOF

[`pre_train.py`](../src/citd_ml/training/pre_train.py) kiểm schema trước khi train: đúng 25.008 dòng, 23 feature số hữu hạn, nhãn 0/1, thời gian nhãn không qua holdout, mỗi `origin_bar` có đủ leg 0–3. `prepare_dataset` sort ổn định theo `entry_time`, thêm `row_id`, tách `X/y/meta` và năm khúc theo chỉ số dòng: `[0, 5001, 10003, 15004, 20006, 25008]`. Code train hiện đọc file dataset đóng băng theo đường dẫn mặc định; `--output-dir` chỉ đổi nơi ghi kết quả.

| Hàm | Nhận và kiểm | Trả hoặc ghi |
|---|---|---|
| [`split_data.py`](../src/citd_ml/training/split_data.py) · `make_random_kfold` | `X`; 5-fold, shuffle, seed 0 | Fold train/test theo dòng |
| `make_grouped_kfold` | `X/y/meta`; 5-fold, `groups=origin_bar` | Fold không tách bốn leg cùng họ |
| `make_walk_forward` | Năm khúc liên tiếp | Bốn fold: train quá khứ, test khúc 2–5 |
| `make_purged_walk_forward` | Walk-fold và thời gian nhãn; giữ train khi `label_end_time < test_start_time` và `entry_bar < test_start_bar - 50` | Bốn fold đã lọc train; test giữ nguyên |
| `validate_all_folds` | Bốn bộ fold; kiểm train/test rời nhau, coverage, thứ tự thời gian và điều kiện purge | PASS hoặc lỗi trước train |
| [`train_catboost.py`](../src/citd_ml/training/train_catboost.py) · `train_method` | Fit model mới cho từng fold; cần đủ hai lớp, xác suất hữu hạn trong `[0,1]`, không chấm trùng test | Metric fold và bảng OOF |
| `_validate_oof_coverage` | Random/Grouped phải có xác suất cho mọi dòng; Walk-forward/Purged chỉ cho khúc 2–5 | PASS hoặc lỗi trước khi lưu |
| `build_summary`, `save_outputs` | Kiểm số fold 5/5/4/4, tính mean metric, serialize CSV | `metrics_by_fold.csv`, `metrics_summary.csv`, bốn `oof_<method>.csv` |

Cấu hình [`MODEL_PARAMS`](../src/citd_ml/training/train_catboost.py): `iterations=1000`, `learning_rate=0.05`, `depth=6`, `l2_leaf_reg=3.0`, `auto_class_weights="Balanced"`, `eval_metric="AUC"`, `random_seed=42`, `thread_count=1`, `allow_writing_files=False`. Không early stopping, không khai báo categorical feature. `thread_count=1` là thiết lập thực thi bổ sung sau bản bàn giao đầu; chi tiết thí nghiệm số luồng ở [`thread_count_sensitivity/README.md`](../outputs/thread_count_sensitivity/README.md).

### 3.3. Backtest bước 4

| Hàm trong [`backtest_pyramid_local.py`](../src/citd_ml/backtest/backtest_pyramid_local.py) | Nhận và kiểm | Trả hoặc ghi |
|---|---|---|
| `load`, `backtest` | M1 trước holdout → M15/lát M1; replay chiến lược chưa lọc, kiểm khóa lệnh không trùng | `baseline_tradelist.csv` qua `save_outputs` |
| `validate_against_reference` | So replay với [tradelist bàn giao](../data/processed/tradelist_pyramid_local.csv) | Chặn khi baseline lệch |
| `load_score_tables` | Bốn OOF; kiểm 25.008 `row_id`, khóa, nhãn/metadata, `fold` và vùng thiếu xác suất khúc 1 | Bốn bảng điểm đã kiểm |
| `build_scored_universe` | Ghép tradelist và OOF one-to-one theo `(origin_bar, entry_bar, leg)`; đối chiếu thời điểm/giá vào | `backtest_scored_universe.csv` |
| `select_top_percent` | Trên 20.007 dòng khúc 2–5, sort xác suất giảm rồi `row_id` tăng; lấy `ceil(n × keep_pct/100)` | Tập `row_id` giữ ở từng mức 20–80% |
| `backtest_filtered` | `shadow` duy trì tín hiệu gốc; `executor` nhận lệnh đã chọn tại thời điểm mở; đối chiếu tập thực thi với top-k | Tradelist đã lọc; mức 50% ghi `trades_top50_<method>.csv` |
| `calculate_metrics`, `build_reports` | Sort theo `close_time, ticket`; tính R, MaxDD đã chốt, PF, win rate; kiểm đúng số lệnh | `backtest_summary_top50.csv` và `backtest_retention_sweep.csv` |

Tỷ lệ giữ cố định giúp bốn nhánh có cùng số lệnh khi so tài chính. Đây là xếp hạng *offline trên toàn khúc 2–5*; code không dựng ngưỡng xác suất khả dụng tuần tự tại mỗi thời điểm. OOF của Random/Grouped có 25.008 dòng được chấm; Walk-forward/Purged để trống xác suất khúc 1 có chủ đích. AUC/F1 trong bảng so sánh được tính trên khúc 2–5 chung, theo từng fold rồi lấy mean; không lấy `metrics_summary.csv` của toàn bộ fold Random/Grouped thay cho [`metrics_chunk2_5.csv`](../outputs/step4_thread1/catboost_training/metrics_chunk2_5.csv).

`R = pl_pct / 0,6` khi `pl_pct` tính bằng điểm phần trăm. Net R là tổng R; MaxDD lấy từ equity đã chốt, bắt đầu 0R; PF là tổng R dương chia trị tuyệt đối tổng R âm; win rate đếm `R > 0`. Không có mark-to-market, phí, spread hay slippage trong các số này.

### 3.4. Holdout và kiểm chứng

| Điểm chạy | Việc chính và điều kiện chặn | Artifact |
|---|---|---|
| [`holdout_stage1_split.py`](../scripts/holdout_stage1_split.py) · `main` | Tách bản tái sinh theo mốc; kiểm 25.008 khóa train cũ và 575.184 ô feature khớp. Chỉ ghi holdout khi mọi check đạt. | `stage1_validation_report.json`, `dataset_catboost_holdout.csv` |
| [`holdout_stage2_train.py`](../scripts/holdout_stage2_train.py) · `precheck`, `run_training`, `verify` | Kiểm purge/embargo tại mốc (đều loại 0 dòng), train hai model độc lập trên 25.008 dòng; so prediction trên train và hash file holdout trước/sau. | Hai `.cbm`, hai `.npy`, `train_run*_report.json`, `stage2_reproducibility_report.json` |
| [`holdout_stage3_backtest.py`](../scripts/holdout_stage3_backtest.py) · `replay_baseline`, `validate_holdout_baseline`, `main` | Replay toàn lịch sử để giữ state, chỉ đo từ mốc holdout; so 5.028 lệnh với tradelist tham chiếu (số thực: `atol=5e-4`), ghép xác suất one-to-one và chạy sweep. | `holdout_scored.csv`, `holdout_fixed_trade_universe_scored.csv`, bảy `holdout_trades_top*.csv`, summary/report |
| [`holdout_evidence.py`](../src/citd_ml/verification/holdout_evidence.py) · `validate_evidence`; [`holdout_stage4_report.py`](../scripts/holdout_stage4_report.py) · `build_table1..4`, `write_chart`, `write_png` | Chặn báo cáo PASS nếu chứng cứ Stage 1–3, manifest canonical hoặc Stage 5 sai; lập bốn bảng và hai đường vốn từ artifact đã kiểm. | [Bảng và hình Stage 4](../outputs/holdout/stage4/), [báo cáo holdout](BAO_CAO_KET_QUA_HOLDOUT.md) |
| [`holdout_stage5_repro_check.py`](../scripts/holdout_stage5_repro_check.py) · `main` | So xác suất, summary, top-k, bảng, HTML/PNG và manifest giữa hai lượt Stage 3–4. | [`stage5_repro_report.json`](../outputs/holdout/repro/stage5_repro_report.json) |
| [`verify_pipeline.py`](../src/citd_ml/verification/verify_pipeline.py) · `check_repeated_runs`, `main` | Bắt hai lượt train/backtest trong bộ nhớ, so frame chưa làm tròn và byte CSV đã lưu; hash nguồn và ghi môi trường. | [`reproducibility.json`](../outputs/step4_thread1/verification/reproducibility.json) |

Stage 4 tạo bảng/hình từ Stage 3; Stage 5 so hai cây Stage 3–4; báo cáo Stage 4 cuối cùng đọc thêm PASS của Stage 5 trước khi công bố. Holdout đã được quan sát trong quá trình làm đồ án, vì vậy các script Stage 1–5 ở đây là bản đồ thực thi lịch sử, không phải quy trình chọn lại tham số.

## 4. Kết quả dùng để kiểm report

Bảng dưới lấy từ [`table1_classification_metrics.csv`](../outputs/holdout/stage4/table1_classification_metrics.csv). Bốn nhánh là mean theo fold trên khúc 2–5; dòng holdout là phép đo riêng của model cuối.

| Phạm vi | ROC-AUC | F1 @ 0,5 |
|---|---:|---:|
| Random K-Fold | 0,8582 | 0,6494 |
| Grouped K-Fold | 0,7454 | 0,5077 |
| Walk-forward | 0,5875 | 0,3288 |
| Purged Walk-forward | 0,5895 | 0,3247 |
| Holdout | 0,6046 | 0,4022 |

Tài chính top 50% lấy từ [`table2_financial_metrics_top50.csv`](../outputs/holdout/stage4/table2_financial_metrics_top50.csv). Baseline luôn so với bộ lọc **trong cùng giai đoạn**; tổng R của khúc 2–5 và holdout không có cùng số lệnh hoặc thời kỳ.

| Phạm vi / bộ lọc | Lệnh | Net R | MaxDD R | PF |
|---|---:|---:|---:|---:|
| Baseline khúc 2–5 | 20.007 | +3.358,25 | 236,13 | 1,2940 |
| Random top 50% | 10.004 | +6.937,53 | 73,76 | 2,5437 |
| Grouped top 50% | 10.004 | +4.752,04 | 99,71 | 1,9500 |
| Walk-forward top 50% | 10.004 | +2.148,31 | 211,29 | 1,4117 |
| Purged top 50% | 10.004 | +2.119,48 | 142,36 | 1,4074 |
| Baseline holdout | 5.028 | +245,93 | 305,32 | 1,0992 |
| Holdout top 50% | 2.514 | −36,55 | 263,25 | 0,9718 |

Sweep đầy đủ 20–80% nằm trong [bảng bốn nhánh](../outputs/holdout/stage4/table3_branch_sweep_20_80.csv) và [bảng holdout](../outputs/holdout/stage4/holdout_sweep_20_80.csv). Mức 50% đã cố định để so sánh; các mức khác trên holdout chỉ là phân tích độ nhạy sau quan sát, không phải chính sách mới được kiểm chứng.

![Baseline và bốn nhánh trên khúc 2–5](../outputs/holdout/stage4/equity-curve-chunk2-5-top50.png)

*Hình 1. Equity đã chốt trên khúc 2–5; [bản HTML](../outputs/holdout/stage4/equity-curve-chunk2-5-top50.html).*

![Baseline và top 50% trên holdout](../outputs/holdout/stage4/equity-curve-holdout-top50.png)

*Hình 2. Equity holdout bắt đầu lại từ 0R; [bản HTML](../outputs/holdout/stage4/equity-curve-holdout-top50.html).*

Random cho AUC cao hơn rõ so với các nhánh theo thời gian. Thiết kế này đồng thời thay đổi cấu trúc gia đình, thứ tự thời gian và kích thước train; không thể quy toàn bộ chênh lệch cho một loại leakage. Purged Walk-forward chỉ loại 21, 3, 8, 2 dòng train ở bốn biên bước 4, nên gần Walk-forward trong lần chạy này. Holdout top 50% thấp hơn baseline holdout 282,48 R; AUC gần holdout hơn không đồng nghĩa bộ lọc có lợi nhuận hơn.

## 5. Bằng chứng, phiên bản và giới hạn

| Phép kiểm | Bằng chứng hiện có |
|---|---|
| Dataset tái sinh | Stage 1: đủ 25.008 khóa đóng băng, 575.184/575.184 ô feature khớp; 4 lệnh biên không vào train/holdout |
| Ranh giới model cuối | Stage 2 chạy purge và embargo, mỗi điều kiện loại 0 dòng trong 25.008 dòng train; file holdout không đổi |
| Baseline holdout | Stage 3 đối chiếu 5.028/5.028 lệnh với tradelist tham chiếu theo validator |
| Bước 4 lặp lại | [Manifest canonical](../outputs/step4_thread1/verification/reproducibility.json) ghi hai lượt train/backtest, so frame chưa làm tròn và CSV đã lưu: 6 CSV train + 8 CSV backtest |
| Holdout lặp lại | [Stage 5](../outputs/holdout/repro/stage5_repro_report.json) PASS: 5.028 xác suất bằng nhau, top-k/bảng và bốn biểu đồ khớp giữa hai lượt trên cùng máy |

Artifact được tạo trên Linux, Python 3.12.14, CatBoost 1.2.10, scikit-learn 1.9.0, pandas 3.0.5, NumPy 2.5.2, Plotly 7.0.0 và Matplotlib 3.11.2; phiên bản đầy đủ nằm trong hai manifest trên. Byte-identical ở đây có phạm vi **cùng máy/cùng mã**. [Thí nghiệm `thread_count`](../outputs/thread_count_sensitivity/README.md) cho thấy đổi số luồng lúc train có thể đổi xác suất và top-k; nó không chứng minh mọi khác biệt liên máy đều do số luồng. Số của bản bàn giao chưa ghim luồng được lưu làm [khối tham chiếu](../outputs/step4_thread1/comparison_report.json).

[Sổ lịch sử holdout](../outputs/verification/holdout_run_history.md) truy xuất được ba phiên bản kết quả trong Git; nó không chứng minh tổng số lần chạy. Phiên bản holdout canonical đã xuất hiện trong commit `7748828` ngày 12/09/2026. Manifest bước 4 và Stage 5 ghi lần kiểm chứng ngày 15/09/2026; [hướng dẫn bàn giao](../deliverables/Huong_dan_su_dung.md) ghi holdout đóng từ 18/09/2026. Vì đã được xem nhiều lần, holdout hiện là đối chiếu hồi cứu, không còn đáp ứng giả định “chỉ mở một lần” của [đề cương](tom_tat_idea_goc.md).

Các kết quả còn bị giới hạn bởi một tài sản, một chiến lược long-only, một giai đoạn, top-k xếp hạng toàn đoạn, train size khác nhau giữa K-Fold và expanding window, chưa có phí/spread/slippage và chưa có khoảng tin cậy cho metric. Muốn đánh giá chính sách giao dịch live cần backtest động và một giai đoạn chưa từng dùng để chọn quyết định.

## 6. Chạy lại và bàn giao

Artifact chính để lần theo một kết quả:

| Cần kiểm | File/thư mục |
|---|---|
| Dataset train và tradelist gốc | [`data/processed/`](../data/processed/) |
| Fold metric và xác suất OOF | [`outputs/step4_thread1/catboost_training/`](../outputs/step4_thread1/catboost_training/) |
| Vũ trụ lệnh, top 50%, sweep bước 4 | [`outputs/step4_thread1/backtest/`](../outputs/step4_thread1/backtest/) |
| Tách holdout, model cuối, backtest holdout | [`outputs/holdout/`](../outputs/holdout/) (Stage 1–3) |
| Bảng và equity dùng trong báo cáo | [`outputs/holdout/stage4/`](../outputs/holdout/stage4/) |
| Manifest kiểm chứng | [bước 4](../outputs/step4_thread1/verification/reproducibility.json), [holdout](../outputs/holdout/repro/stage5_repro_report.json) |

Các lệnh sau kiểm lại **bước 4**, ghi ra thư mục mới. Chọn tên thư mục chưa tồn tại cho mỗi lượt; ví dụ `outputs/recheck_step4_run01/`. `verify_dataset.py` và train đọc dataset đóng băng. Từ backtest trở đi phải có file M1 thô đúng đường dẫn trong [README dữ liệu](../data/raw/README.md).

```bash
uv run python scripts/verify_dataset.py
uv run python scripts/train_models.py --output-dir outputs/recheck_step4_run01/train
uv run python scripts/run_backtest.py --oof-dir outputs/recheck_step4_run01/train --output-dir outputs/recheck_step4_run01/backtest
uv run python scripts/verify_pipeline.py --train-dir outputs/recheck_step4_run01/train --backtest-dir outputs/recheck_step4_run01/backtest --manifest outputs/recheck_step4_run01/verification/reproducibility.json
```

Nếu cần tái sinh dataset từ M1, ghi ra một file khác sau khi thư mục trên đã được tạo; đối chiếu file đó với dataset đóng băng trước khi diễn giải khác biệt:

```bash
uv run python scripts/build_dataset.py --output outputs/recheck_step4_run01/dataset_regenerated.csv
```

`train_models.py` hiện vẫn đọc `data/processed/dataset_catboost.csv`, không tự lấy file `dataset_regenerated.csv`. Vì vậy lệnh trên là bước kiểm **tái sinh dataset**, không biến chuỗi lệnh thành một lần train end-to-end trên dataset mới. Các script holdout Stage 1–5 chỉ được nêu để truy vết lần chạy lịch sử; không dùng lại holdout để chọn feature, tham số hay tỷ lệ giữ.
