# Báo cáo chốt hyperparameter gán nhãn — Pyramid Strategy

> **Nguồn và trạng thái.** Viết ở giai đoạn chốt tham số gán nhãn, trước khi mã nguồn
> được tái cấu trúc thành package `src/citd_ml/`. Chỉ đường dẫn và tên file tham chiếu
> được cập nhật cho khớp cấu trúc hiện tại; toàn bộ phần phân tích giữ nguyên.
>
> **Đã đối chiếu với mã nguồn và artifact trong repo này:** `sl_pct=0.006`,
> `atr_mult=3.8`, `atr_period=90`, `k=3` trong
> [`pyramid_strategy.py`](../src/citd_ml/strategy/pyramid_strategy.py); `VERTICAL_BARS=50`
> và mốc holdout `2025-02-08 15:30:00` (bar 199,968) trong
> [`paths.py`](../src/citd_ml/paths.py); 25,008 dòng / 6,252 gia đình × 4 leg và tỷ lệ
> nhãn 1 = 26.16% do [`scripts/verify_dataset.py`](../scripts/verify_dataset.py) in ra;
> baseline holdout +245.93 R trong
> [`holdout_run_history.py`](../scripts/holdout_run_history.py). Sáu quy tắc gán nhãn ở
> §1/§5 được vẽ đầy đủ trong [`thong_so_triple_barrier.png`](thong_so_triple_barrier.png)
> và cài trong [`triple_barrier.py`](../src/citd_ml/labeling/triple_barrier.py).
>
> **Đã lỗi thời:** §6.1 và §7 viết "ba cách chia". Đồ án sau đó chốt **bốn** nhánh —
> Cách 1b (Grouped K-Fold) được bổ sung; lý do ghi ở §6.2 của
> [`bao_cao_feature.md`](bao_cao_feature.md).

Mọi tham số chốt **trước** khi huấn luyện, không số nào chọn bằng AUC hay chỉ số hiệu năng mô hình. Số liệu tính trên **25,012 lệnh thô của 80% dữ liệu đầu** (6,253 tín hiệu × 4 leg: gốc + 3 nhồi). Holdout không được chạm.

> **Ghi chú phương pháp (bản cập nhật):** Toàn bộ số liệu gán nhãn dưới đây tính lại bằng đúng hàm `calculate_barriers` / `label_one_entry` của `src/citd_ml/labeling/triple_barrier.py` (bản Tuấn đã sửa lỗi chỉ gán nhãn lệnh gốc), chạy cho **cả 4 leg** chứ không chỉ leg 0. Các đặc trưng dùng để kiểm tra near-duplicate (ATR%, volatility, khoảng cách EMA, lợi suất, RSI) được **tính lại từ OHLCV M15 gốc** tại bar ngay trước entry của từng lệnh — **không tái sử dụng** cột `momentum_prev`/`vwap_prev`/`atr_frozen`-cho-feature trong file gắn nhãn, vì các cột đó đóng băng tại bar tín hiệu gốc (`origin_bar`) để phục vụ vào lệnh của strategy, không phản ánh đúng trạng thái thị trường tại thời điểm entry riêng của từng lệnh nhồi.

---

## Bảng tham số chốt

| Tham số | Giá trị | Suy ra từ |
|---|---|---|
| **Định nghĩa nhãn** | `1` = lệnh không lỗ (mô phỏng trailing), `0` = lỗ hoặc hết hạn | §1 |
| **Ngưỡng trên** | `entry + 3.8 × ATR(90)`, đóng băng tại bar trước entry | §2 |
| **Ngưỡng dưới** | 0.6% dưới entry (= stop-loss chiến lược) | §2 |
| **Chặn thời gian H** | **50 bar M15** | §3 |
| **Số leg k** | **3** (tính từ xác suất leak) | §4 |
| Phân giải chạm mốc | duyệt nến M1 trong bar M15 | §5 |
| Tỷ lệ nhãn 1 (H=50, toàn bộ 4 leg) | **26.2%** | §3 |
| Ranh giới holdout | 2025-02-08 15:30 (bar 199,968) | 80% dữ liệu |
| Lệnh dùng được ở H=50 (đã bỏ horizon cụt) | **25,008** / 25,012 (6,252 tín hiệu × 4 leg, mất đúng 1 tín hiệu sát biên holdout) | §3 |

### Kết quả chiến lược (80% đầu)

Bảng này lấy trực tiếp từ `tradelist_pyramid_local.csv` (kết quả trailing-stop thật của chiến lược, độc lập với mô phỏng triple-barrier ở trên) — không đổi so với bản trước vì không liên quan tới lỗi gán nhãn của Tuấn.

| | Toàn bộ (25,012) | Chỉ leg 0 (6,253) | Holdout (5,028) |
|---|---|---|---|
| Net profit | +4,613.2R | +1,251.7R | +245.9R |
| Profit factor | 1.318 | 1.351 | 1.099 |
| Win rate | 31.0% | 31.9% | 35.3% |
| Expectancy | +0.184R | +0.200R | +0.049R |

---

## §1. Định nghĩa nhãn

Nhãn 1 = lệnh thoát ≥ hoà vốn. Nhãn 0 = lỗ hoặc hết hạn H bar.

Hàm gán nhãn **mô phỏng đúng luật trailing** của chiến lược: đóng băng ATR tại entry, dời stop mỗi bar, duyệt M1 để xác định thứ tự chạm mốc. Không mô phỏng luật cắt thứ Sáu vì đó là ràng buộc vận hành, không phải thuộc tính tín hiệu. Áp dụng **y hệt cho lệnh gốc lẫn 3 lệnh nhồi** — mỗi lệnh dùng đúng `entry_price`/`atr_frozen` của chính nó.

---

## §2. Ngưỡng trên và dưới

**Ngưỡng dưới = 0.6%** — stop-loss chiến lược, không có lựa chọn khác.

**Ngưỡng trên = entry + 3.8 × ATR(90), đóng băng tại bar i−1.**

Trailing stop = `đỉnh giá − 3.8 × ATR`. Khi giá lên đúng 3.8×ATR → stop dâng tới entry → hoà vốn. Đi xa hơn → không thể lỗ. Đây là **điểm không thể quay đầu** — mốc tự nhiên cho nhãn 1. Đóng băng ATR để mốc này là một con số cố định biết ngay lúc vào lệnh.

**Không dùng ngưỡng cố định theo R** vì breakeven quy ra R dao động cực mạnh (p5=0.99R, p95=6.23R), tính trên toàn bộ 25,012 lệnh (base + nhồi):

| Ngưỡng cố định | % thời gian thật sự đảm bảo không lỗ |
|---|---|
| 2.0R | 33.3% |
| 2.5R | 52.4% |
| 3.0R | 66.2% |
| 4.0R | 82.5% |

Không mức nào đạt 100%. Ngưỡng theo ATR đạt 100% theo định nghĩa. (Số liệu mục này không đổi so với bản trước — breakeven-R chỉ phụ thuộc `entry_price`/`atr_frozen`, không phụ thuộc lỗi gán nhãn của Tuấn.)

---

## §3. Chặn thời gian H = 50 bar

Van an toàn cho lệnh treo lâu bất thường → gán 0.

**Phân phối độ dài lệnh** (số bar tới khi chạm mốc thật sự, gộp cả 4 leg, loại các lệnh chưa từng chạm trong 120 bar): p50 = 10 bar, p80 = 28 bar, p90 = 46 bar.

> **Tương quan nhãn** đo mức độ "giống nhau thật sự" giữa nhãn lệnh gốc và nhãn lệnh nhồi, trên thang −1 đến +1. Khác với tỷ lệ trùng nhãn đơn thuần: nếu phần lớn mẫu là nhãn 0 thì đoán bừa "luôn 0" cũng trùng phần lớn, nhưng tương quan = 0. Tương quan cao nghĩa là biết nhãn lệnh gốc thì đoán được nhãn lệnh nhồi — tức hai lệnh mang cùng thông tin.

| H | Lệnh dùng được | % lệnh xong | % nhãn 1 | Tương quan nhãn (avg leg 1–3) |
|---|---|---|---|---|
| 20 | 25,012 | 70.0% | 16.9% | 0.615 |
| 30 | 25,012 | 80.2% | 21.7% | 0.644 |
| **50** | **25,008** | **89.8%** | **26.2%** | **0.656** |
| 80 | 25,008 | 96.1% | 29.1% | 0.659 |
| 120 | 25,008 | 98.3% | 30.2% | 0.657 |

Quét thêm lưới mịn hơn (bước 10 bar) cho thấy tương quan tăng nhanh tới H≈50–60 (đỉnh tại H=60: 0.662), sau đó đi ngang/dao động nhẹ quanh 0.657–0.659 tới H=120 — không tiếp tục tăng thêm đáng kể.

**Chọn H = 50** vì tương quan nhãn đã gần sát đỉnh (0.656 so với đỉnh 0.662 tại H=60) — phần tăng thêm khi kéo dài horizon quá điểm này là không đáng kể (0.656 → 0.659 → 0.657 khi H đi từ 50 lên 120, gần như đi ngang). Số lệnh bị cắt vì thiếu horizon **không đổi** khi tăng H từ 50 lên 120 (vẫn 25,008/25,012, chỉ mất đúng 1 gia đình sát biên holdout ở cả hai mức), nên không có lý do đánh đổi để kéo dài H. H=50 vẫn cover 89.8% lệnh và cho tỷ lệ nhãn 1 = 26.2% (chấp nhận được, không quá lệch) — chọn mức ngắn nhất đã nằm trên vùng bão hoà của tương quan.

---

## §4. Số leg k = 3

### Ý tưởng

Mỗi tín hiệu sinh một "gia đình" gồm lệnh gốc + k lệnh nhồi = k+1 thành viên. Các thành viên gần giống nhau (near-duplicate). Khi random 5-fold chia dữ liệu, ta muốn **hầu như mọi mẫu test đều có ít nhất một anh em trong train** — để leak xảy ra gần như chắc chắn.

### Tính toán

Mỗi thành viên có xác suất 1/5 rơi vào cùng fold test. Xác suất **cả k anh em đều rơi vào test** (= không ai trong train):

$$P(\text{thoát}) = \left(\frac{1}{5}\right)^{k}$$

| k | P(thoát) | P(có ≥ 1 anh em trong train) |
|---|---|---|
| 1 | 20.00% | 80.00% |
| 2 | 4.00% | 96.00% |
| **3** | **0.80%** | **99.20%** |
| 4 | 0.16% | 99.84% |

k = 3 đưa P(thoát) xuống dưới 1% — thêm nữa không cải thiện đáng kể. Đây là tiêu chí **thuần xác suất**, tính được trước khi chạy mô hình. (Bảng này không đổi — không phụ thuộc dữ liệu gán nhãn.)

### Kiểm tra near-duplicate

Xác suất cao chưa đủ — phải xác nhận anh em thật sự giống nhau.

**Vế nhãn** (H = 50, tính trên 25,008 lệnh hợp lệ, khớp cặp theo `origin_bar`):

| Leg | Trùng nhãn với leg 0 | Tương quan |
|---|---|---|
| 1 | 89.5% | 0.731 |
| 2 | 86.8% | 0.657 |
| 3 | 83.7% | 0.578 |

**Vế đặc trưng** (tương quan với leg 0, đặc trưng tính lại từ OHLCV M15 tại bar trước entry riêng của mỗi lệnh — không dùng cột đóng băng trong file gắn nhãn):

| Đặc trưng | leg 1 | leg 2 | leg 3 |
|---|---|---|---|
| ATR(14) % giá | 0.991 | 0.980 | 0.966 |
| Biến động 50 bar (std lợi suất) | 0.985 | 0.976 | 0.969 |
| Khoảng cách EMA(50) | 0.939 | 0.899 | 0.863 |
| Lợi suất 20 bar | 0.949 | 0.907 | 0.865 |
| RSI(14) | 0.919 | 0.856 | 0.797 |

Trùng nhãn 83.7–89.5%, đặc trưng tương quan 0.80–0.99 → near-duplicate rõ ràng, còn mạnh hơn ước tính trước đó. Random CV chia gia đình ra hai phía thì mô hình tra bảng chứ không dự báo.

---

## §5. Phân giải M1

Nến M15 không cho biết thứ tự giá chạm stop hay mốc trên trước. Duyệt 15 nến M1 bên trong theo thứ tự thời gian để xác định chính xác.

---

## §6. Hạn chế

1. **Mất cân bằng 26.2%.** Class weight cố định, dùng chung cả ba cách chia.
2. **Nhồi khuếch đại confound kích thước tập.** 4× hàng nhưng thông tin thật tăng ít hơn nhiều. Ước lượng bằng design effect: tương quan trung bình giữa mọi cặp leg tại H=50 là **ρ ≈ 0.691** (trung bình 6 cặp: 0–1, 0–2, 0–3, 1–2, 1–3, 2–3), với cụm 4 thành viên/gia đình → hệ số phóng đại phương sai `DEFF = 1 + (4−1)×ρ ≈ 3.07`, tức kích thước mẫu **hiệu dụng** chỉ tăng khoảng **1.3×** so với chỉ dùng leg 0 (4/DEFF ≈ 1.30), không phải 4×. Chênh lệch giữa các cách chia dữ liệu sẽ lớn hơn phần do leak thuần vì cỡ mẫu danh nghĩa bị thổi phồng.
3. **Holdout rơi giai đoạn thị trường khác.** Sụt giảm gồm cả loại leak lẫn thay đổi thị trường.

---

## §7. Bước tiếp theo

1. Chốt bộ đặc trưng, cố định cho cả ba cách chia
2. Sinh tập mẫu với k = 3, khử trùng theo timestamp
3. Chạy ba cách chia trên 80% đầu
4. Mở holdout đúng một lần
