# Báo cáo kết quả bước 4 — train và backtest trên 80% đầu

> **Phạm vi.** Báo cáo này là phần **diễn giải** kết quả bốn cách chia trên 80% dữ liệu
> đầu (`outputs/catboost_training/` và `outputs/backtest/`), gồm kết luận về leakage (§7)
> và cảnh báo về cơ chế shadow strategy (§8).
>
> Nó **khác** [`BAO_CAO_KET_QUA_HOLDOUT.md`](BAO_CAO_KET_QUA_HOLDOUT.md): báo cáo holdout
> chỉ trình bày số liệu, biểu đồ và kiểm chứng, cố ý không diễn giải (xem Giai đoạn 5 của
> [`quy_trinh_lam_viec.md`](quy_trinh_lam_viec.md)). Hai tài liệu bổ sung cho nhau.
>
> **Nguồn và trạng thái.** Viết trước khi mã nguồn được tái cấu trúc thành package
> `src/citd_ml/`. Chỉ đường dẫn tham chiếu được cập nhật; phần phân tích giữ nguyên.
> Số liệu ứng với khối **bàn giao** (không ghim `thread_count`).

## 1. Công việc đã thực hiện

Pipeline đã chạy đủ bốn cách chia dữ liệu trong `BAN_GIAO`: Random K-Fold, Grouped K-Fold, Walk-forward và Purged Walk-forward. Kết quả gồm metric của 18 fold, bốn bảng xác suất ngoài mẫu huấn luyện (OOF), bảng backtest top 50% và bảng khảo sát tỷ lệ giữ lệnh từ 20% đến 80%.

Mọi chỉ số backtest đều tính trên 20.007 lệnh thuộc chunk 2–5. Ở mức giữ 50%, Random và Grouped có tổng lợi nhuận cao hơn baseline. Hai nhánh theo thời gian có Profit factor và Win rate cao hơn baseline nhưng tổng lợi nhuận thấp hơn. Các bảng dưới đây trình bày riêng từng chỉ số để tránh đánh giá bộ lọc chỉ qua tổng lợi nhuận.

Phần chiến lược hiện giữ chuỗi tín hiệu baseline bằng một chiến lược mô phỏng song song, gọi là _shadow_. Leader chưa có yêu cầu bổ sung về cách xử lý sau khi CatBoost từ chối lệnh gốc, nên cơ chế này được giữ nguyên theo chỉ dẫn hiện tại. Mục 8 ghi rõ cách xử lý và giới hạn của kết quả để người nhận bàn giao có thể đối chiếu.

## 2. Dữ liệu, nhãn và phạm vi đánh giá

### 2.1. Dữ liệu dùng cho CatBoost

Dataset có 25.008 dòng và 31 cột. Mỗi dòng ứng với một lệnh. Một tín hiệu gốc, được nhận diện bằng `origin_bar`, tạo thành một gia đình gồm bốn lệnh: lệnh gốc `leg = 0` và ba lệnh nhồi `leg = 1, 2, 3`. Ba lệnh nhồi được lên lịch ở ba bar M15 tiếp theo. Các dòng cùng gia đình có thời điểm và bối cảnh thị trường gần nhau, nên không thể xem như bốn quan sát hoàn toàn độc lập.

Đầu vào của CatBoost gồm đúng 23 cột trong hằng `FEATURES` của [build_features.py](../src/citd_ml/features/build_features.py). Bảy cột metadata — `entry_time`, `label_end_time`, `origin_bar`, `entry_bar`, `entry_price`, `leg`, `entry_vs_base_R` — dùng để chia dữ liệu, kiểm tra nhãn chồng lấn và ghép lệnh với backtest; không đưa vào model. `row_id` được thêm sau khi sắp xếp để nhận diện dòng, cũng không phải feature.

Target là cột `label` đã có sẵn. Model trả về `probability`, tức điểm xác suất dự đoán `label = 1`. Backtest dùng điểm này để xếp hạng lệnh. Lời/lỗ thực tế của từng lệnh trong mô phỏng vẫn do Stop, Target và Friday close của chiến lược quyết định; không lấy `label` thay cho kết quả giao dịch.

### 2.2. Năm chunk thời gian

Dataset được sắp xếp tăng dần theo riêng `entry_time`, dùng `kind="stable"` để giữ thứ tự các dòng trùng thời gian, rồi reset index. Năm chunk được chia theo chỉ số dòng với công thức `edges[k] = floor(n × k / 5)`:

| Chunk |      `row_id` | Số dòng |
| ----: | ------------: | ------: |
|     1 |       0–5.000 |   5.001 |
|     2 |  5.001–10.002 |   5.002 |
|     3 | 10.003–15.003 |   5.001 |
|     4 | 15.004–20.005 |   5.002 |
|     5 | 20.006–25.007 |   5.002 |

Random và Grouped có dự đoán OOF cho cả 25.008 dòng. Walk-forward và Purged Walk-forward dùng chunk 1 để train lần đầu, nên không có dự đoán cho chunk này. Vì vậy, baseline và cả bốn nhánh đều loại chunk 1 khi tính chỉ số backtest; phạm vi so sánh là các dòng có `row_id >= 5001`.

### 2.3. Giới hạn holdout

Backtest chỉ sử dụng dữ liệu có thời gian:

```text
dt < 2025-02-08 15:30:00
```

Dữ liệu từ đúng mốc này trở đi không tham gia mô phỏng. Trước training, code cũng kiểm tra cả `entry_time` và `label_end_time`: nếu một trong hai chạm mốc holdout, chương trình báo lỗi. File giá, dataset và tradelist bàn giao được giữ nguyên; kết quả chạy nằm trong `outputs/`.

---

## 3. Thiết kế thí nghiệm

### 3.1. Bốn cách chia dữ liệu

| Phương pháp         | Cách hoạt động                                                      | Rủi ro leakage được kiểm soát                                                    |
| ------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Random K-Fold       | Trộn ngẫu nhiên từng dòng rồi chia 5 fold                           | Không kiểm soát quan hệ gia đình hoặc thứ tự thời gian; dùng làm nhánh đối chứng |
| Grouped K-Fold      | Chia 5 fold nhưng giữ toàn bộ lệnh cùng `origin_bar` trong một nhóm | Giảm leakage giữa leg 0–3 của cùng một gia đình                                  |
| Walk-forward        | Train trên các chunk quá khứ, test trên chunk kế tiếp               | Không để mô hình học tương lai rồi dự đoán quá khứ                               |
| Purged Walk-forward | Walk-forward kết hợp purging và embargo 50 bar M15                  | Loại thêm các nhãn chồng lấn qua biên test và vùng sát biên                      |

Random dùng `KFold(n_splits=5, shuffle=True, random_state=0)`. Grouped dùng `GroupKFold` với cùng ba tham số và `groups=origin_bar`. Walk-forward có bốn lượt: train chunk 1 → test 2; train 1–2 → test 3; train 1–3 → test 4; train 1–4 → test 5.

Với Purged Walk-forward, loại khỏi train các dòng có `label_end_time >= test_start_time` hoặc `entry_bar >= test_start_bar - 50`. Phần test giữ nguyên. Mốc 50 ở đây là số bar M15 quan sát, không thay bằng một khoảng giờ cố định vì chuỗi giá có khoảng trống.

### 3.2. Cấu hình CatBoost

Toàn bộ fold và phương pháp dùng cùng một cấu hình cố định:

| Hyperparameter       |    Giá trị |
| -------------------- | ---------: |
| `iterations`         |       1000 |
| `learning_rate`      |       0.05 |
| `depth`              |          6 |
| `l2_leaf_reg`        |        3.0 |
| `auto_class_weights` | `Balanced` |
| `eval_metric`        |      `AUC` |
| `random_seed`        |         42 |

Các tham số trên áp dụng cho cả 18 fold. Không early stopping, không `cat_features` và không tuning. `hour` và `dow` giữ dạng numeric. Môi trường được kiểm chứng dùng `catboost==1.2.10` và `scikit-learn==1.9.0`, đáp ứng yêu cầu `scikit-learn>=1.6`.

### 3.3. Tạo và sử dụng xác suất OOF

Trong mỗi fold, model chỉ train trên phần train và dự đoán phần test. Các dự đoán này được ghép thành một bảng OOF cho từng phương pháp. Mỗi dòng có điểm được dự đoán đúng một lần, bởi model không dùng chính dòng đó để train.

Bảng OOF được ghép với lệnh backtest theo khóa:

```text
origin_bar + entry_bar + leg
```

Cả bốn bảng ghép đủ 25.008 dòng, không trùng khóa. Walk-forward và Purged Walk-forward vẫn để trống điểm ở 5.001 dòng đầu. Trong mô phỏng, chunk 1 được chạy không lọc để khởi tạo trạng thái chiến lược, sau đó loại khỏi phần tính chỉ số.

Tại mỗi lần shadow sinh một lệnh dự kiến, bộ thực thi (_executor_) tra `row_id` và chỉ mở lệnh thuộc nhóm được chọn. Cách sinh tín hiệu bằng shadow được trình bày ở mục 8, vì nó ảnh hưởng đến cách hiểu toàn bộ bảng backtest.

### 3.4. Quy tắc top-k

Với mỗi phương pháp, code xếp chung 20.007 lệnh của chunk 2–5 theo xác suất giảm dần. Top 50% là một nửa số lệnh đứng đầu bảng này. Ngưỡng `probability >= 0.5` chỉ dùng khi tính F1, không phải điều kiện chọn top 50%. Nếu bằng điểm ở biên, lệnh có `row_id` nhỏ hơn được chọn trước.

Do tập đánh giá có 20.007 dòng là số lẻ, số lệnh giữ lại được làm tròn lên bằng `ceil`:

| Tỷ lệ yêu cầu | Số lệnh giữ | Tỷ lệ thực tế |
| ------------: | ----------: | ------------: |
|           20% |       4.002 |      20,0030% |
|           30% |       6.003 |      30,0045% |
|           40% |       8.003 |      40,0010% |
|           50% |      10.004 |      50,0025% |
|           60% |      12.005 |      60,0040% |
|           70% |      14.005 |      70,0005% |
|           80% |      16.006 |      80,0020% |

---

## 4. Kết quả phân loại trước backtest

Nguồn số liệu: [`outputs/catboost_training/metrics_summary.csv`](../outputs/catboost_training/metrics_summary.csv).

ROC-AUC và F1 được tính riêng từng fold rồi lấy trung bình cộng. F1 dùng ngưỡng 0,5. Bảng này không phải metric tính gộp một lần trên toàn bộ OOF.

| Phương pháp         | Số fold | ROC-AUC trung bình | F1 trung bình |
| ------------------- | ------: | -----------------: | ------------: |
| Random K-Fold       |       5 |           0,862978 |      0,662366 |
| Grouped K-Fold      |       5 |           0,738060 |      0,507709 |
| Walk-forward        |       4 |           0,586725 |      0,326728 |
| Purged Walk-forward |       4 |           0,594776 |      0,330422 |

ROC-AUC giảm từ 0,862978 ở Random xuống 0,738060 ở Grouped. Khi train bằng các chunk quá khứ và test trên chunk kế tiếp, AUC còn khoảng 0,59. Vì tham số model được giữ nguyên, đây là chênh lệch quan sát được giữa các cách chia. Các tập train/test khác nhau, nên chưa thể quy toàn bộ phần chênh lệch cho riêng một loại leakage.

Purged đạt AUC 0,594776 và F1 0,330422, nhỉnh hơn Walk-forward trong lần chạy này. Việc áp dụng thêm purge/embargo không bắt buộc làm metric giảm; kết quả phụ thuộc vào những mẫu train bị loại. Mục 7 nêu rõ phần nào của leakage đã được kiểm tra trực tiếp.

---

## 5. Kết quả backtest top 50%

Nguồn số liệu: [`outputs/backtest/backtest_summary_top50.csv`](../outputs/backtest/backtest_summary_top50.csv).

| Kịch bản            | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate |
| ------------------- | ------: | -------------: | --------: | ------------: | -------: |
| Baseline            |  20.007 |       3.358,25 |    236,13 |         1,294 |   31,71% |
| Random K-Fold       |  10.004 |       6.998,30 |     75,71 |         2,563 |   44,88% |
| Grouped K-Fold      |  10.004 |       4.724,34 |     99,50 |         1,942 |   39,24% |
| Walk-forward        |  10.004 |       2.207,25 |    205,31 |         1,423 |   34,86% |
| Purged Walk-forward |  10.004 |       2.253,72 |    154,11 |         1,435 |   35,17% |

Các chỉ số được tính trên kết quả của từng lệnh:

- `R = pl_pct / (0,006 × 100)`, nên lỗ 0,6% tương ứng −1R;
- `Net profit` là tổng `R` của các lệnh được giữ;
- `Profit factor` là tổng R dương chia cho trị tuyệt đối của tổng R âm;
- `Win rate` là tỷ lệ lệnh có `R > 0`;
- `MaxDD` là drawdown lớn nhất của equity đã chốt, bắt đầu từ 0R, sắp theo `close_time` rồi `ticket`. Đây là drawdown realized theo thứ tự đóng lệnh; báo cáo chưa tính drawdown mark-to-market của vị thế còn mở.

### 5.1. Mức thay đổi so với baseline

| Phương pháp         | Chênh lệch Net profit | Thay đổi Net profit | Giảm MaxDD | Tăng Profit factor | Tăng Win rate |
| ------------------- | --------------------: | ------------------: | ---------: | -----------------: | ------------: |
| Random K-Fold       |            +3.640,05R |            +108,39% |     67,94% |             +1,269 | +13,17 điểm % |
| Grouped K-Fold      |            +1.366,09R |             +40,68% |     57,86% |             +0,648 |  +7,53 điểm % |
| Walk-forward        |            −1.151,00R |             −34,27% |     13,05% |             +0,129 |  +3,14 điểm % |
| Purged Walk-forward |            −1.104,53R |             −32,89% |     34,73% |             +0,141 |  +3,45 điểm % |

### 5.2. Diễn giải

Random giữ 10.004 lệnh nhưng tạo ra 6.998,30R, hơn gấp đôi tổng lợi nhuận baseline. MaxDD giảm từ 236,13R xuống 75,71R. Các con số này cao nhất hoặc tốt nhất trong bảng, nhưng phải đọc cùng cấu trúc split: model có thể học dữ liệu tương lai và các lệnh cùng gia đình với lệnh test.

Grouped vẫn vượt baseline về cả bốn chỉ số. So với Random, tổng lợi nhuận giảm còn 4.724,34R và Profit factor còn 1,942. Việc gom cùng gia đình đã loại đường rò rỉ trực tiếp giữa các leg, nhưng chưa ngăn model dùng dữ liệu tương lai để dự đoán quá khứ.

Hai nhánh theo thời gian cho thấy rõ sự đánh đổi ở mức giữ 50%: tỷ lệ thắng và Profit factor tăng, còn tổng lợi nhuận giảm. Riêng Purged giữ khoảng một nửa số lệnh và giữ được khoảng 67,11% lợi nhuận baseline; MaxDD giảm 34,73%. Bộ lọc đã chọn được một nhóm có hiệu quả trung bình tốt hơn, nhưng nhóm đó vẫn chưa tạo ra tổng lợi nhuận bằng toàn bộ tập lệnh.

So trực tiếp hai nhánh theo thời gian, Purged có thêm 46,47R lợi nhuận và ít hơn 51,20R MaxDD so với Walk-forward. Đây là kết quả tại mức 50% trên dataset này. Các tỷ lệ giữ lệnh khác không phải lúc nào cũng giữ nguyên thứ tự đó.

---

## 6. Phân tích bảng giữ lệnh 20–80%

Nguồn số liệu đầy đủ: [`outputs/backtest/backtest_retention_sweep.csv`](../outputs/backtest/backtest_retention_sweep.csv).

Baseline không dùng bộ lọc nên giữ nguyên 20.007 lệnh ở mọi hàng so sánh:

| Net profit |   MaxDD | Profit factor | Win rate |
| ---------: | ------: | ------------: | -------: |
|  3.358,25R | 236,13R |         1,294 |   31,71% |

### 6.1. Net profit theo tỷ lệ giữ lệnh

| Giữ top | Random K-Fold | Grouped K-Fold |  Walk-forward | Purged Walk-forward |
| ------: | ------------: | -------------: | ------------: | ------------------: |
|     20% |     5.291,57R |      2.820,09R |     1.004,50R |           1.232,38R |
|     30% |     6.627,08R |      3.755,75R |     1.298,00R |           1.401,76R |
|     40% | **7.005,91R** |      4.402,78R |     1.759,23R |           1.702,32R |
|     50% |     6.998,30R |      4.724,34R |     2.207,25R |           2.253,72R |
|     60% |     6.439,83R |  **4.795,38R** |     2.500,98R |           2.411,01R |
|     70% |     5.761,04R |      4.623,85R |     2.708,99R |           2.867,13R |
|     80% |     4.872,57R |      4.391,34R | **3.126,84R** |       **3.138,78R** |

Random đạt tổng lợi nhuận cao nhất trong bảng tại 40%: 7.005,91R, chỉ hơn mức 50% khoảng 7,61R. Grouped đạt cao nhất tại 60% với 4.795,38R. Với hai nhánh này, mở rộng nhóm được giữ qua mức đó làm tổng lợi nhuận giảm; phần lệnh được thêm vào có tổng R âm.

Ở cả Walk-forward và Purged, tổng lợi nhuận tăng qua từng mức giữ lệnh từ 20% đến 80%. Điều này có nghĩa các nhóm lệnh được thêm vào vẫn đóng góp R dương trong lần chạy này. Dù giữ đến 80%, Walk-forward và Purged vẫn thấp hơn baseline lần lượt 231,41R và 219,47R. Không mức nào trong dải khảo sát giúp hai nhánh theo thời gian vượt baseline về tổng lợi nhuận.

### 6.2. MaxDD theo tỷ lệ giữ lệnh

| Giữ top | Random K-Fold | Grouped K-Fold | Walk-forward | Purged Walk-forward |
| ------: | ------------: | -------------: | -----------: | ------------------: |
|     20% |        16,00R |         36,56R |      103,86R |              97,67R |
|     30% |        21,63R |         63,19R |      162,58R |             165,93R |
|     40% |        40,79R |         70,36R |      184,68R |             141,49R |
|     50% |        75,71R |         99,50R |      205,31R |             154,11R |
|     60% |        93,66R |        165,47R |      239,98R |             210,79R |
|     70% |       138,96R |        195,98R |      292,27R |             221,72R |
|     80% |       188,26R |        213,01R |      273,52R |             242,51R |

MaxDD tăng ở phần lớn các bước mở rộng tập lệnh, nhưng không tăng đều. Chẳng hạn, Purged giảm từ 165,93R ở mức 30% xuống 141,49R ở mức 40%; Walk-forward giảm từ 292,27R ở mức 70% xuống 273,52R ở mức 80%. Drawdown phụ thuộc vào thứ tự lời/lỗ của các lệnh, không chỉ số lượng lệnh được giữ.

Trong khoảng 40–70%, Purged có MaxDD thấp hơn Walk-forward ở từng mức tương ứng. Tuy nhiên, tại 80%, cả hai đều có MaxDD cao hơn baseline 236,13R. Vì vậy, thêm lệnh để tăng tổng lợi nhuận cũng có thể làm mất phần cải thiện drawdown của bộ lọc.

### 6.3. Profit factor theo tỷ lệ giữ lệnh

| Giữ top | Random K-Fold | Grouped K-Fold | Walk-forward | Purged Walk-forward |
| ------: | ------------: | -------------: | -----------: | ------------------: |
|     20% |         6,410 |          2,798 |        1,520 |               1,651 |
|     30% |         4,429 |          2,439 |        1,432 |               1,473 |
|     40% |         3,223 |          2,171 |        1,430 |               1,417 |
|     50% |         2,563 |          1,942 |        1,423 |               1,435 |
|     60% |         2,090 |          1,762 |        1,392 |               1,379 |
|     70% |         1,783 |          1,608 |        1,358 |               1,383 |
|     80% |         1,552 |          1,493 |        1,359 |               1,361 |

Cả bốn phương pháp có Profit factor cao nhất tại 20%. Khi giữ thêm lệnh, chỉ số này nhìn chung giảm, dù có vài bước tăng nhẹ ở hai nhánh theo thời gian. Ví dụ, Purged tăng từ 1,417 ở mức 40% lên 1,435 ở mức 50%.

Walk-forward và Purged có Profit factor cao hơn baseline 1,294 ở toàn bộ bảy mức. Đây là bằng chứng quan sát được rằng điểm model giúp chọn một nhóm có tỷ lệ tổng lời trên tổng lỗ tốt hơn trong tập đánh giá. Bảng Net profit cho thấy cải thiện này chưa đủ để bù phần lợi nhuận mất đi khi bỏ bớt lệnh.

### 6.4. Win rate theo tỷ lệ giữ lệnh

| Giữ top | Random K-Fold | Grouped K-Fold | Walk-forward | Purged Walk-forward |
| ------: | ------------: | -------------: | -----------: | ------------------: |
|     20% |        65,12% |         48,08% |       37,38% |              38,81% |
|     30% |        57,44% |         44,88% |       36,33% |              36,71% |
|     40% |        50,41% |         41,98% |       35,56% |              35,60% |
|     50% |        44,88% |         39,24% |       34,86% |              35,17% |
|     60% |        40,37% |         37,02% |       34,30% |              34,31% |
|     70% |        37,14% |         35,23% |       33,62% |              33,99% |
|     80% |        34,43% |         33,73% |       33,47% |              33,54% |

Win rate giảm qua từng mức giữ lệnh ở cả bốn phương pháp. Nhóm 20% có điểm cao nhất cũng là nhóm có tỷ lệ thắng cao nhất trong các tập được xét. Khi giữ đến 80%, Win rate của hai nhánh theo thời gian còn khoảng 33,5%, gần hơn với baseline 31,71%.

Random đạt 65,12% ở mức 20%, trong khi Walk-forward và Purged chỉ đạt 37,38% và 38,81%. Chênh lệch này cần được đọc cùng kết quả kiểm tra leakage ở mục 7; không thể lấy Win rate của Random làm kỳ vọng cho các lệnh tương lai.

### 6.5. Cách sử dụng bảng tỷ lệ giữ lệnh

Mức 50% là mốc báo cáo chính theo `BAN_GIAO`. Bảy mức 20–80% dùng lại cùng bảng OOF, không train lại; chúng cho biết kết quả thay đổi thế nào khi nhận nhiều hoặc ít lệnh hơn.

Các mức 40% của Random và 60% của Grouped chỉ là điểm có Net profit cao nhất trong bảng đã quan sát. Chúng chưa phải tỷ lệ tối ưu để triển khai. Nếu chọn tỷ lệ sau khi xem bảng rồi đánh giá lại ngay trên bảng này, kết quả sẽ chịu thêm thiên lệch do lựa chọn trên tập test.

---

## 7. Kết luận về leakage

### 7.1. Random K-Fold cho kết quả quá lạc quan

Random chia ngẫu nhiên từng dòng trên toàn bộ giai đoạn 2018–2025. Cách chia này cho phép dữ liệu tương lai nằm trong train của một mẫu test ở quá khứ. Nó cũng tách các leg cùng `origin_bar` sang hai phía.

Kiểm tra từng fold cho thấy 99,04–99,52% dòng test có ít nhất một lệnh cùng gia đình nằm trong train. Đây là đường rò rỉ thông tin có thể xác định trực tiếp từ cách chia. Trong điều kiện đó, Random đạt AUC 0,862978, Profit factor top 50% là 2,563 và Net profit cao hơn baseline 108,39%.

Với mục tiêu dự đoán lệnh tương lai, kết quả Random là đánh giá quá lạc quan. Tuy nhiên, khoảng cách metric giữa các nhánh không phải phép đo chính xác lượng leakage: thành phần và kích thước tập train/test cũng khác nhau.

### 7.2. Grouped K-Fold giảm leakage giữa các lệnh cùng gia đình

Grouped giữ bốn leg cùng `origin_bar` trong cùng một fold. Tỷ lệ dòng test có lệnh cùng gia đình ở phía train bằng 0% trong cả năm fold. Như vậy, đường rò rỉ trực tiếp vừa nêu ở Random đã được loại bỏ.

AUC giảm xuống 0,738060 và Net profit top 50% còn 4.724,34R. Kết quả phù hợp với việc quan hệ giữa các leg góp phần làm Random được đánh giá cao. Dù vậy, Grouped vẫn phân bố các gia đình ngẫu nhiên giữa các giai đoạn thời gian, nên chưa giải quyết việc học tương lai để dự đoán quá khứ.

### 7.3. Walk-forward phản ánh tổng quát hóa theo thời gian thực tế hơn

Walk-forward train bằng các dòng ở chunk trước và test trên chunk kế tiếp. Cách chia này sát với thứ tự dự đoán theo thời gian hơn hai cách chia ngẫu nhiên. AUC đạt 0,586725 và Net profit top 50% là 2.207,25R.

Điểm còn thiếu của Walk-forward thông thường là thời điểm biết nhãn. Một dòng có thể vào lệnh trước biên test nhưng chỉ có nhãn sau khi test đã bắt đầu. Nếu vẫn đưa dòng đó vào train, thứ tự `entry_time` đã đúng nhưng thông tin nhãn vẫn tràn qua biên. Đây là phần được xử lý thêm ở nhánh Purged.

### 7.4. Purged Walk-forward kiểm soát biên train/test nghiêm ngặt nhất

Nhánh này bỏ khỏi train các dòng có `label_end_time` bằng hoặc sau thời điểm bắt đầu test, rồi loại thêm vùng 50 bar M15 sát biên theo quy tắc embargo. Số dòng train bị loại qua bốn fold lần lượt là 21, 3, 8 và 2; test không thay đổi.

Trong cả bốn fold của dataset hiện tại, các dòng thỏa điều kiện purge cũng nằm trong vùng embargo. Vì vậy, kết quả chỉ cho biết tác động của hai điều kiện khi áp dụng cùng nhau; chưa tách được lợi ích riêng của từng điều kiện.

Purged đạt AUC 0,594776, Net profit top 50% là 2.253,72R và MaxDD 154,11R. Đây là nhánh kiểm soát biên train/test nghiêm ngặt nhất trong bốn cách được giao. Tính nghiêm ngặt đến từ quy tắc loại dữ liệu, không phải việc metric của nó cao hay thấp hơn Walk-forward. Việc chia chặt hơn cũng không xóa các giới hạn của cách chọn top-k và mô phỏng chiến lược ở mục sau.

---

## 8. Vấn đề chiến lược cần ghi nhận khi bàn giao

### 8.1. Bỏ lệnh gốc có thể làm thay đổi tín hiệu tiếp theo

Trong [pyramid_strategy.py](../src/citd_ml/strategy/pyramid_strategy.py), hàm `entry_signal()` chỉ cho phép tín hiệu gốc mới khi không còn lệnh gốc đang mở. Trạng thái này được lưu trong `base_open`: mở lệnh gốc thì chuyển thành `True`, lệnh gốc đóng thì chuyển về `False`.

Vì vậy, việc CatBoost từ chối một lệnh gốc có thể ảnh hưởng tới các tín hiệu về sau. Nếu trạng thái đi theo vị thế thực tế, lệnh bị từ chối không làm `base_open` chuyển thành `True`, nên chiến lược vẫn có thể nhận tín hiệu gốc mới. Những lệnh mới xuất hiện theo cách này có thể không nằm trong dataset bàn giao và chưa có điểm OOF.

Ví dụ dưới đây chỉ để minh họa cách hoạt động, không phải một giao dịch trích từ dữ liệu. Giả sử baseline mở lệnh gốc lúc 10:00 và đóng lúc 10:45, nhưng CatBoost từ chối lệnh 10:00:

| Thời điểm                             | Giữ chuỗi tín hiệu bằng shadow như code hiện tại     | Sinh tín hiệu theo vị thế thực tế được nhận               |
| ------------------------------------- | ---------------------------------------------------- | --------------------------------------------------------- |
| 10:00                                 | Không mở ở executor, nhưng shadow vẫn mở một lệnh ảo | Không có lệnh gốc được mở                                 |
| 10:15, nếu điều kiện chỉ báo lại thỏa | Shadow còn lệnh gốc nên chặn tín hiệu gốc mới        | Có thể sinh lệnh gốc mới; lệnh này có thể chưa có OOF     |
| 10:45                                 | Lệnh gốc của shadow đóng, `base_open` trở về `False` | Trạng thái phụ thuộc các lệnh thực sự được nhận sau 10:00 |

`BAN_GIAO` yêu cầu tra điểm khi chiến lược định mở lệnh và giữ nguyên các phần còn lại. Tài liệu chưa quy định rõ trạng thái `base_open` phải theo vị thế thực tế hay theo một chiến lược shadow khi lệnh gốc bị từ chối. Đây là điểm còn mở về cách mô phỏng chiến lược.

### 8.2. Cách code hiện tại xử lý

[backtest_pyramid_local.py](../src/citd_ml/backtest/backtest_pyramid_local.py) dùng hai đối tượng chiến lược. Shadow chạy theo baseline để sinh các lệnh dự kiến và giữ lịch của ba leg nhồi. Executor kiểm tra lệnh có thuộc top-k không trước khi mở, rồi quản lý Stop, Target và Friday close cho các lệnh được nhận.

Nhờ shadow, các lệnh được xét vẫn thuộc đúng tập 25.008 dòng đã có trong dataset. Cả bốn phương pháp có thể được so sánh trên cùng tập này. Đổi lại, một lệnh bị từ chối ở executor vẫn có thể chặn tín hiệu mới trong shadow. Một leg nhồi cũng có thể được nhận dù leg gốc cùng gia đình bị từ chối, vì mỗi lệnh được lọc theo điểm riêng.

Các bảng kết quả trong báo cáo được tính theo cơ chế đó. Chưa có kết quả cho phương án để toàn bộ tín hiệu thay đổi theo các vị thế thực tế được nhận.

### 8.3. Những gì đã kiểm tra trên kết quả hiện tại

Tại mức top 50%, số lệnh gốc bị từ chối trong chunk 2–5 là:

| Phương pháp         | Số lệnh `leg = 0` bị từ chối |
| ------------------- | ---------------------------: |
| Random K-Fold       |                        2.426 |
| Grouped K-Fold      |                        2.432 |
| Walk-forward        |                        2.407 |
| Purged Walk-forward |                        2.383 |

Nguồn đối chiếu là [tập lệnh đã ghép điểm](../outputs/backtest/backtest_scored_universe.csv) và bốn file `trades_top50_<method>.csv` trong [outputs/backtest](../outputs/backtest/).

Với mọi lệnh được nhận, giá vào, giá ra, bar thoát và R đều bằng lệnh tương ứng trong baseline. Điều này giải thích vì sao chuyển từ lọc sau khi hoàn tất lệnh sang kiểm tra tại thời điểm mở không làm thay đổi các con số: chuỗi tín hiệu vẫn cố định và các vị thế được quản lý độc lập. Kết quả bằng nhau không chứng minh rằng phương án sinh tín hiệu theo vị thế thực tế cũng sẽ cho cùng lợi nhuận.

### 8.4. Phạm vi sử dụng kết quả và phần cần thống nhất

Báo cáo này dùng kết quả hiện tại để so sánh bốn bảng OOF trên tập lệnh baseline cố định. Theo chỉ dẫn hiện tại, giữ nguyên cơ chế shadow vì leader chưa yêu cầu thay đổi. Giả định này được ghi trong phạm vi bàn giao; kết quả chạy lặp xác nhận cách triển khai hiện có, không thay thế việc đánh giá một chính sách sinh tín hiệu khác nếu có yêu cầu sau này.

Nếu yêu cầu tiếp theo là để việc bỏ lệnh thay đổi tín hiệu về sau, cần xác định thêm cách tạo feature và chọn model chấm điểm cho các lệnh mới tại thời điểm chúng xuất hiện. Không thể tra OOF của một lệnh không có trong bảng điểm, hoặc gán điểm của lệnh khác cho nó. Đây là phần công việc bổ sung; chưa được triển khai trong kết quả đang bàn giao.

### 8.5. Các giới hạn khác khi đọc bảng backtest

- Top-k được xếp trên toàn bộ xác suất OOF của chunk 2–5. Đây là phép so sánh offline; ngưỡng nhận lệnh ở một thời điểm thực tế phải được xác định từ dữ liệu đã có trước thời điểm đó.
- Net profit và MaxDD tính theo R của từng lệnh. Mô phỏng chưa đặt giới hạn vốn cho các vị thế mở đồng thời; MaxDD chỉ tính phần lời/lỗ đã chốt.
- Kết quả chưa cộng phí giao dịch, spread và slippage vào giá vào/ra của chiến lược.
- Dataset chỉ có một tài sản trong một giai đoạn lịch sử. Kết quả hiện tại chưa kiểm chứng trên tài sản khác hoặc một giai đoạn độc lập.

---

## 9. Kết quả kiểm tra sau khi sửa

Lần kiểm chứng ngày 07/09 chạy lại 18 fold training hai lượt và toàn bộ backtest hai lượt. Mỗi lượt backtest gồm baseline cùng 28 cấu hình, tương ứng bốn phương pháp nhân bảy tỷ lệ giữ lệnh.

| Nội dung kiểm tra                  | Kết quả                                                                                                      |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Dataset và phép chia               | Đúng 23 feature; stable sort và ranh giới năm chunk đúng yêu cầu; test giữ nguyên ở nhánh Purged             |
| Bảng OOF                           | Cùng `row_id` 0–25.007; metadata, label và fold khớp dataset và phép chia hiện tại                           |
| Số dòng có dự đoán                 | Random/Grouped: 25.008; Walk/Purged: 20.007, chỉ thiếu điểm ở chunk 1                                        |
| Baseline so với tradelist bàn giao | Khớp đủ 25.008 lệnh đã mở và đóng trước holdout; giá/R dùng dung sai `5e-4` do file tham chiếu làm tròn      |
| Ghép OOF với backtest              | Không trùng hoặc thiếu khóa `origin_bar + entry_bar + leg`                                                   |
| Phạm vi lọc lệnh                   | Mỗi nhánh top 50% có 10.004 lệnh thuộc chunk 2–5; sweep có 35 dòng gồm cả baseline                           |
| Hai lượt training                  | Giá trị chưa làm tròn giống hệt nhau; sáu CSV tái tạo khớp từng byte với file đang báo cáo                   |
| Hai lượt backtest                  | Giá trị chưa làm tròn giống hệt nhau; tám CSV tái tạo khớp từng byte với file đang báo cáo                   |
| Số liệu trong báo cáo              | Bốn metric top 50% được tính lại độc lập; 112 ô số trong bốn bảng sweep khớp CSV theo độ chính xác trình bày |

Code cũng đã được thử với dữ liệu sai có chủ đích. Các trường hợp bị chặn gồm: thời gian bằng đúng mốc holdout; metadata OOF bị sửa giống nhau ở mọi nhánh; fold OOF sai; train bị xóa thêm ngoài purge/embargo; thiếu file tham chiếu; thiếu hoặc thừa lệnh khi đối chiếu. Một lệnh dự kiến ở cuối chuỗi chưa có OOF chỉ được bỏ qua nếu chưa đóng trước holdout; nếu đã đóng, chương trình báo lỗi.

Bằng chứng của hai lượt chạy nằm trong [outputs/verification/reproducibility.json](../outputs/verification/reproducibility.json). File này lưu cấu hình model, phiên bản Python/thư viện và SHA-256 của 11 file đầu vào/code cùng 14 CSV kết quả. Dữ liệu gốc và các CSV kết quả không bị ghi đè trong quá trình kiểm chứng.

[Manifest ngày 06/09](../outputs/backtest/reproducibility_manifest.txt) được giữ để đối chiếu lịch sử; nó chỉ ghi nhận backtest ở phiên bản trước. File JSON ngày 07/09 chứa bằng chứng đầy đủ hơn cho cả training và backtest tại thời điểm chạy.

Lượt kiểm chứng mới nhất chạy từ 09:57:48 đến 10:03:20 ngày 07/09/2026, giờ Việt Nam, và kết thúc với trạng thái `passed`. Cả training và backtest đều hoàn tất hai lượt; giá trị chưa làm tròn giống nhau và kết quả khớp từng byte với 14 CSV hiện có. Đối chiếu lại sau lượt chạy cho thấy toàn bộ 11 file đầu vào/code, gồm `verify_pipeline.py`, cùng 14 CSV đều khớp hash trong manifest mới. Phần bằng chứng tái lập đã được đồng bộ; không còn chênh lệch hash đã ghi nhận trước đó.

Kết quả chạy lặp xác nhận tính tái lập của phiên bản đã kiểm chứng. Lựa chọn cách mô phỏng chiến lược vẫn là vấn đề riêng ở mục 8.

---

## 10. File bàn giao

### 10.1. Code

| File                                                   | Công việc                                                                |
| ------------------------------------------------------ | ------------------------------------------------------------------------ |
| [pre_train.py](../src/citd_ml/training/pre_train.py)                           | Đọc dataset, kiểm tra dữ liệu và xác định năm chunk                      |
| [split_data.py](../src/citd_ml/training/split_data.py)                         | Tạo và kiểm tra bốn cách chia train/test                                 |
| [train_catboost.py](../src/citd_ml/training/train_catboost.py)                 | Train 18 model theo fold, tính metric và lưu OOF                         |
| [backtest_pyramid_local.py](../src/citd_ml/backtest/backtest_pyramid_local.py) | Chạy baseline, ghép OOF và backtest các mức giữ lệnh                     |
| [pyramid_strategy.py](../src/citd_ml/strategy/pyramid_strategy.py)             | Logic chiến lược gốc; file này không bị sửa                              |
| [verify_pipeline.py](../src/citd_ml/verification/verify_pipeline.py)               | Chạy lặp training/backtest, so sánh kết quả và lưu bằng chứng kiểm chứng |

### 10.2. Kết quả training

| File                                                                                      | Nội dung                                                  |
| ----------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| [metrics_by_fold.csv](../outputs/catboost_training/metrics_by_fold.csv)                      | Kích thước train/test, tỷ lệ nhãn và AUC/F1 của từng fold |
| [metrics_summary.csv](../outputs/catboost_training/metrics_summary.csv)                      | AUC/F1 trung bình của từng phương pháp                    |
| Bốn file `oof_<method>.csv` trong [outputs/catboost_training](../outputs/catboost_training/) | Xác suất OOF, fold, nhãn và metadata của từng dòng        |

### 10.3. Kết quả backtest và kiểm chứng

| File                                                                             | Nội dung                                                 |
| -------------------------------------------------------------------------------- | -------------------------------------------------------- |
| [backtest_summary_top50.csv](../outputs/backtest/backtest_summary_top50.csv)        | Baseline và bốn nhánh giữ top 50%                        |
| [backtest_retention_sweep.csv](../outputs/backtest/backtest_retention_sweep.csv)    | Kết quả tại bảy mức giữ lệnh 20–80%                      |
| [backtest_scored_universe.csv](../outputs/backtest/backtest_scored_universe.csv)    | 25.008 lệnh baseline đã ghép điểm của bốn phương pháp    |
| [baseline_tradelist.csv](../outputs/backtest/baseline_tradelist.csv)                | Các lệnh baseline hoàn tất trước holdout                 |
| Bốn file `trades_top50_<method>.csv` trong [outputs/backtest](../outputs/backtest/) | Chi tiết các lệnh được giữ ở mức 50%                     |
| [reproducibility.json](../outputs/verification/reproducibility.json)                | Bằng chứng chạy lặp training và backtest ngày 07/09      |
| [reproducibility_manifest.txt](../outputs/backtest/reproducibility_manifest.txt)    | Bằng chứng backtest ngày 06/09, giữ để đối chiếu lịch sử |

---

## 11. Hướng dẫn chạy lại

Tại thư mục dự án, chạy theo thứ tự dưới đây. Các lệnh gọi trực tiếp Python trong `.venv`, nên không cần kích hoạt môi trường bằng `Activate.ps1`:

```powershell
.\.venv\Scripts\python.exe train_catboost.py
.\.venv\Scripts\python.exe backtest_pyramid_local.py
```

Lệnh đầu tạo lại sáu CSV training; lệnh sau tạo lại tám CSV backtest. Các file kết quả cùng tên trong `outputs/` sẽ được ghi đè. Không chạy lại `build_features.py` hoặc tạo lại dataset cho công việc này. Bảng 20–80% dùng lại OOF, không train một model riêng cho mỗi tỷ lệ.

Để kiểm chứng tính tái lập của cả hai giai đoạn mà không ghi đè CSV kết quả:

```powershell
.\.venv\Scripts\python.exe -u verify_pipeline.py
```

Lệnh này chạy lại training và backtest hai lượt, nên mất thời gian hơn việc chỉ đọc CSV. Khi hai lượt giống nhau và khớp các CSV hiện có, script lưu bằng chứng vào `outputs/verification/reproducibility.json`. Nếu có khác biệt, script báo lỗi để kiểm tra, không tự thay CSV đang được báo cáo.

---

## 12. Kết luận bàn giao

Phần training đã tạo đủ OOF và metric theo bốn cách chia được yêu cầu. Các bảng backtest đã có đủ mức top 50% và dải 20–80%, được kiểm tra lại bằng hai lượt chạy giống nhau.

Về kết quả, hai nhánh theo thời gian chọn được nhóm lệnh có Profit factor và Win rate cao hơn baseline. Tuy nhiên, không mức giữ lệnh nào từ 20% đến 80% giúp chúng vượt baseline về tổng lợi nhuận. Ở mốc chính 50%, Purged giảm MaxDD xuống 154,11R nhưng tổng lợi nhuận chỉ còn 2.253,72R, so với 3.358,25R của baseline.

`BAN_GIAO` yêu cầu thực hiện đúng cách chia dữ liệu, cấu hình model, metric và backtest; không đặt mức AUC tối thiểu hoặc yêu cầu lợi nhuận phải vượt baseline. Vì vậy, tổng lợi nhuận của Walk-forward thấp hơn baseline là kết quả của thí nghiệm, không phải lỗi triển khai hay một tiêu chí nghiệm thu bị thiếu. Việc nâng hiệu quả giao dịch sẽ cần mục tiêu và phép đánh giá riêng nếu có yêu cầu tiếp theo.

Các con số này áp dụng cho tập tín hiệu baseline được giữ cố định bằng shadow. Cơ chế shadow được giữ nguyên theo chỉ dẫn hiện tại, khi leader chưa có yêu cầu bổ sung. Code, kết quả và bằng chứng chạy lặp đã được đối chiếu; các giới hạn của cách mô phỏng vẫn được giữ trong mục 8 để tránh diễn giải kết quả vượt quá phạm vi đã thực hiện.
