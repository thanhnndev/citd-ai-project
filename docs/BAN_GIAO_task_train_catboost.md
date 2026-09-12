# TASK — Chia train/test, train CatBoost, backtest

## File giao

| File | |
|---|---|
| `dataset_catboost.csv` | 25,008 dòng × 31 cột |
| `build_features.py` | Lấy hằng `FEATURES` (23 cột) và `META` (7 cột) |
| `BTCUSD_m1_2018_to_now.csv` | Dữ liệu BTC M1, để backtest |
| `pyramid_strategy.py` | Logic chiến lược |
| `backtest_pyramid_local.py` | Logic backtest |
| `tradelist_pyramid_local.csv` | Kết quả backtest chưa lọc, để đối chiếu |

`catboost==1.2.10`, `scikit-learn>=1.6`.
Biến `SRC` trong `backtest_pyramid_local.py` đang trỏ tên file cũ — sửa cho trỏ đúng `BTCUSD_m1_2018_to_now.csv`.

---

## 1. Dataset

`X` = 23 cột của hằng `FEATURES`. `y` = cột `label`.
7 cột `META` **không đưa vào `X`** — chúng chỉ dùng để chia dữ liệu và ghép backtest.

Sắp xếp theo `entry_time`, dùng **sort ổn định** (`kind="stable"`) — dataset có dòng trùng `entry_time`, sort không ổn định sẽ cho thứ tự khác nhau giữa các máy và làm lệch ranh giới khúc. Sau đó reset index về 0…25007.

Chia 5 khúc theo **chỉ số dòng** (không chia theo ngày): `edges[k] = floor(n × k / 5)`, `k = 0..5`.

| Khúc | Dòng | Số dòng |
|---|---|---|
| 1 | 0 – 5,000 | 5,001 |
| 2 | 5,001 – 10,002 | 5,002 |
| 3 | 10,003 – 15,003 | 5,001 |
| 4 | 15,004 – 20,005 | 5,002 |
| 5 | 20,006 – 25,007 | 5,002 |

---

## 2. Bốn cách chia

**Cách 1 — Random K-Fold**
`KFold(n_splits=5, shuffle=True, random_state=0)`
Trộn ngẫu nhiên 25,008 dòng, chia đều vào 5 rổ, lần lượt lấy 1 rổ làm test và 4 rổ làm train.

**Cách 1b — Grouped K-Fold**
`GroupKFold(n_splits=5, shuffle=True, random_state=0)`, `groups = origin_bar`
Giống Cách 1, khác đúng một điểm: 4 lệnh có cùng `origin_bar` luôn rơi vào cùng một rổ, không bị tách sang hai phía.

**Cách 2 — Walk-forward, expanding window**
Học quá khứ, dự đoán khúc kế tiếp:
train khúc 1 → test khúc 2 · train 1+2 → test 3 · train 1+2+3 → test 4 · train 1+2+3+4 → test 5.
Khúc 1 không được dự đoán.

**Cách 3 — Walk-forward + Purging & Embargo**
Giống Cách 2, thêm bước dọn phần train nằm sát ranh giới. Mỗi lệnh có hai mốc: `entry_time` (lúc vào lệnh) và `label_end_time` (lúc *biết được* nhãn — khi giá chạm hàng rào, hoặc hết 50 bar).

- **Purging** — Một lệnh có thể vào trước ranh giới nhưng chỉ chạm hàng rào *sau* khi khúc test đã bắt đầu. Nhãn của nó khi đó do diễn biến giá **bên trong khúc test** quyết định; cho mô hình học nó là cho nó biết trước tương lai. → Bỏ khỏi train mọi lệnh có `label_end_time ≥` thời điểm khúc test bắt đầu.
- **Embargo** — Purging chỉ chặn lệnh có nhãn tràn qua ranh giới. Những lệnh kết thúc *ngay sát* trước ranh giới thì nhãn không tràn qua, nhưng vẫn phản ánh trạng thái thị trường đang tiếp diễn sang vùng test. → Bỏ thêm khỏi train mọi lệnh vào trong **50 bar M15 cuối** trước khúc test (50 = bằng độ dài hàng rào thời gian).

**Chỉ xoá bên train. Khúc test giữ nguyên 100%** — bỏ mẫu khỏi test thì 4 cách chia đánh giá trên tập lệnh khác nhau, hết so sánh được.

---

## 3. Hyperparameter — cố định cho mọi nhánh, mọi fold

| | |
|---|---|
| `iterations` | 1000 |
| `learning_rate` | 0.05 |
| `depth` | 6 |
| `l2_leaf_reg` | 3.0 |
| `auto_class_weights` | `"Balanced"` |
| `eval_metric` | `"AUC"` |
| `random_seed` | 42 |

Không early stopping, không `cat_features`.

---

## 4. Chỉ số phân loại

**ROC-AUC** và **F1** (ngưỡng 0.5) cho từng cách chia. Tính riêng từng fold rồi lấy trung bình cộng.

---

## 5. Backtest

Mỗi cách chia cho ra một **bảng điểm**: mỗi lệnh một xác suất. Mỗi lệnh nằm trong tập test đúng một lần, nên lấy xác suất ở chính lần đó — tức từ mô hình **chưa từng thấy lệnh đó khi huấn luyện**. Ghép các fold lại thành một cột phủ hết dataset, lưu ra file.

Chạy `backtest_pyramid_local.py` trên `BTCUSD_m1_2018_to_now.csv`, thêm một bộ lọc: chiến lược định mở lệnh nào thì tra điểm của lệnh đó — nằm trong **top 50%** thì cho vào, không thì bỏ qua. Mọi thứ còn lại giữ nguyên.

Chạy 5 lần: **baseline** (không lọc) và **4 nhánh**.

Chỉ tính chỉ số trên **khúc 2–5**, bỏ khúc 1 ở cả 5 lần chạy — vì Cách 2 và 3 không có dự đoán cho khúc 1.

| | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---|---|---|---|
| Baseline | | | | |
| Cách 1 | | | | |
| Cách 1b | | | | |
| Cách 2 | | | | |
| Cách 3 | | | | |

---

## 6. Bảng tỷ lệ giữ lệnh

Lặp lại §5 với **20 / 30 / 40 / 50 / 60 / 70 / 80%**, dùng lại bảng điểm đã lưu, không train lại.

---

## Lưu ý

1. Không đổi hyperparameter, không thêm/bớt feature, không đưa `META` vào `X`.
2. Không bỏ mẫu khỏi tập test ở bất kỳ cách chia nào.
3. Không đụng dữ liệu từ `2025-02-08 15:30:00` trở đi.
4. Cố định seed; chạy hai lần phải ra kết quả giống hệt nhau.
