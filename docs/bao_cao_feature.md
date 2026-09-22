# Báo cáo chốt bộ đặc trưng cho CatBoost — Pyramid Strategy

> **Đây là tài liệu mà [`build_features.py`](../src/citd_ml/features/build_features.py)
> trỏ tới** khi liệt kê các cột bị loại khỏi bộ feature (§3 dưới đây).
>
> **Nguồn và trạng thái.** Viết ở giai đoạn chốt bộ đặc trưng, trước khi mã nguồn được
> tái cấu trúc thành package `src/citd_ml/`. Chỉ đường dẫn và tên file tham chiếu được
> cập nhật; toàn bộ phần phân tích giữ nguyên.
>
> **Đã đối chiếu với artifact trong repo này:** bảng §6 khớp từng số với
> [`outputs/catboost_training/metrics_summary.csv`](../outputs/catboost_training/metrics_summary.csv)
> (0.8630 / 0.7381 / 0.5867 / 0.5948); số dòng bị purge/embargo ở §6.2 (21/3/8/2) khớp
> kết quả chạy [`split_data.py`](../src/citd_ml/training/split_data.py); cấu hình CatBoost
> ở §4 khớp `MODEL_PARAMS` trong
> [`train_catboost.py`](../src/citd_ml/training/train_catboost.py); danh sách loại ở §3
> khớp khối `EXCLUDED` trong `build_features.py`.
>
> **Lưu ý khi đọc §6:** đây là khối **bàn giao**, không ghim `thread_count`. Cây canonical
> `thread_count=1` cho số hơi khác (0.8628 / 0.7367 / 0.5875 / 0.5895) — xem
> [`outputs/step4_thread1/`](../outputs/step4_thread1/) và Phụ lục A của
> [`BAO_CAO_KET_QUA_HOLDOUT.md`](BAO_CAO_KET_QUA_HOLDOUT.md).

Bộ đặc trưng khởi điểm: **23 feature**, sinh bởi `src/citd_ml/features/build_features.py` → `dataset_catboost.csv` (25,008 dòng, 6,252 gia đình tín hiệu, tỷ lệ nhãn 1 = 26.16%). Nhãn lấy nguyên từ `src/citd_ml/labeling/triple_barrier.py`, không sửa.

---

## §0. Bốn nguyên tắc chi phối toàn bộ lựa chọn

**1. Mọi feature tính tại bar entry của CHÍNH lệnh đó, chỉ dùng bar đã đóng.**
Với lệnh vào tại bar `i`, feature lấy từ bar `i−1` trở về trước, cộng thêm giá mở cửa của bar `i` (đã biết tại thời điểm vào lệnh). Không tái sử dụng `momentum_prev`/`vwap_prev`/`atr_frozen` trong file gán nhãn — các cột đó đóng băng tại `origin_bar` nên 4 lệnh cùng gia đình sẽ có giá trị y hệt nhau một cách giả tạo. Riêng `atr_frozen` được dùng **đúng một lần** để tính `breakeven_R`, vì ở đó nó không phải chỉ báo thị trường mà là **tham số hình học của chính hàng rào** (barrier được định nghĩa bằng ATR đóng băng — xem §2 báo cáo hyperparameter).

**2. Không feature nào được mã hoá mức giá tuyệt đối hay vị trí lịch của mẫu.**
Mọi đại lượng giá đều chuẩn hoá theo ATR hoặc theo % giá. Lý do định lượng ở §3.

**3. Không đưa giàn giáo của thí nghiệm vào mô hình.**
Chiến lược kim tự tháp k=3 được chọn vì nó **cực đoan có chủ đích** — nó phóng đại chồng lấn nhãn lên mức quan sát được bằng mắt thường, đóng vai mô hình thu nhỏ cho hiện tượng vốn xảy ra kín đáo ở mọi chiến lược. Hệ quả: **mọi đại lượng chỉ tồn tại nhờ cái giàn giáo đó đều bị loại khỏi feature.** Chiến lược thật không có khái niệm "lệnh nhồi thứ mấy trong gia đình" hay "vào cao hơn lệnh gốc bao nhiêu". Đưa vào thì (a) mách thẳng cho mô hình một mảnh cấu trúc gia đình, (b) làm mức thiên lệch đo được **không suy rộng ra được cho chiến lược nào khác** — đủ để phá giá trị của đồ án. Xem §3.

**4. Feature KHÔNG được chọn bằng AUC hay hiệu năng mô hình.**
Giữ đúng cam kết ở [`bao_cao_hyperparameter_triple_barrier.md`](bao_cao_hyperparameter_triple_barrier.md). Bộ 23 feature chốt bằng **lý lẽ nghiệp vụ + kiểm tra dư thừa (Spearman giữa các feature) + nguyên tắc 3** — cả ba đều **không đụng tới nhãn**. Cột AUC đơn biến trong §1 là **thông tin tham khảo, công bố một lần để chứng minh dataset không suy biến**, không dùng để thêm/bớt feature nào. Bằng chứng: các feature AUC yếu nhất (`dow` 0.4956, `mom_diff` 0.5075, `gap_open_atr` 0.5075) **vẫn được giữ**, vì chúng có vai trò nghiệp vụ rõ ràng.

---

## §1. Bộ 23 feature

`sib_corr` = tương quan giữa lệnh nhồi và lệnh gốc cùng gia đình (đo mức độ near-duplicate — **đây là kênh dẫn leak**, tính không cần nhãn). `AUC` = AUC đơn biến, tham khảo (xem §0.3); AUC < 0.5 nghĩa là quan hệ nghịch, với cây quyết định thì tương đương về sức mạnh.

### A. Hình học hàng rào — quyết định độ khó của nhãn

| Feature | Công thức | sib_corr | AUC |
|---|---|---|---|
| `breakeven_R` | `3.8 × ATR(90) / (entry × 0.006)` | 0.999 | **0.386** |

Khoảng cách từ entry tới hàng rào trên, quy ra **đơn vị stop-loss**. Đây là feature mạnh nhất bộ và mạnh cách biệt: hàng rào trên càng xa (tính theo R) thì càng khó chạm → nhãn 1 càng hiếm. Biến thiên cực rộng (p5 = 0.99R, p95 = 6.23R), nên nó thật sự phân biệt được lệnh dễ và lệnh khó.

### B. Chế độ biến động

| Feature | Công thức | sib_corr | AUC |
|---|---|---|---|
| `atr14_pct` | `ATR(14) / close` | 0.979 | 0.419 |
| `atr_ratio_14_90` | `ATR(14) / ATR(90)` | 0.952 | 0.533 |
| `vol20` | `std(lợi suất, 20)` | 0.958 | 0.430 |
| `vol200` | `std(lợi suất, 200)` | 0.998 | 0.403 |
| `vol_ratio_20_200` | `vol20 / vol200` | 0.946 | 0.516 |
| `range_pct` | `(high − low) / close` bar trước | 0.571 | 0.445 |

`atr_ratio_14_90` gần như trực giao với `breakeven_R` (Spearman −0.066) → mang thông tin thật sự mới: biến động đang giãn hay co so với nền dài hạn.

### C. Xu hướng và vị trí trong biên độ — chuẩn hoá theo ATR

| Feature | Công thức | sib_corr | AUC |
|---|---|---|---|
| `dist_ema20_atr` | `(close − EMA20) / ATR(14)` | 0.819 | 0.527 |
| `dist_ema50_atr` | `(close − EMA50) / ATR(14)` | 0.901 | 0.541 |
| `dist_ema200_atr` | `(close − EMA200) / ATR(14)` | 0.959 | 0.527 |
| `ret20_atr` | `(close − close[−20]) / ATR(14)` | 0.866 | 0.521 |
| `ret50_atr` | `(close − close[−50]) / ATR(14)` | 0.944 | 0.538 |
| `pos_in_range50` | vị trí trong `[min low, max high]` 50 bar | 0.884 | 0.539 |
| `dist_hh20_atr` | `(close − max high 20) / ATR(14)` | 0.804 | 0.515 |

Chia cho ATR chứ không để đơn vị giá — điều kiện bắt buộc để feature còn ý nghĩa khi BTC đi từ 3k lên 100k.

### D. Tín hiệu của chính chiến lược, tính lại tươi

| Feature | Công thức | sib_corr | AUC |
|---|---|---|---|
| `mom14` | `momentum(14)` tại bar `i−1` | 0.876 | 0.538 |
| `mom_diff` | `mom14[i−1] − mom14[i−2]` | **0.018** | 0.508 |
| `vwap_dist_atr` | `(close − VWAP142) / ATR(14)` | 0.955 | 0.530 |
| `vwap_slope_atr` | `(VWAP142[i−1] − VWAP142[i−2]) / ATR(14)` | **0.157** | 0.515 |
| `rsi14` | RSI(14) | 0.857 | 0.528 |

`mom_diff` và `vwap_slope_atr` chính là hai vế của điều kiện vào lệnh gốc. Tại bar tín hiệu chúng luôn thoả (mom giảm, vwap tăng); tại các bar nhồi thì không ràng buộc gì → **sib_corr rất thấp (0.018 và 0.157)**. Đây là các feature *phân biệt* anh em, đối trọng với nhóm B/C vốn gần như đồng nhất.

### E. Khối lượng, vi cấu trúc, phiên

| Feature | Công thức | sib_corr | AUC |
|---|---|---|---|
| `vol_ratio_volume` | `volume / MA(volume, 50)` | 0.559 | 0.537 |
| `gap_open_atr` | `(open[i] − close[i−1]) / ATR(14)` | **0.0005** | 0.508 |
| `hour` | giờ trong ngày (0–23) | 0.835 | 0.488 |
| `dow` | thứ trong tuần (0–4) | 0.976 | 0.496 |

Cả bốn đều là thông tin mà bất kỳ chiến lược thật nào cũng có. Không feature nào trong bộ mô tả cấu trúc gia đình lệnh — xem nguyên tắc §0.3 và danh sách loại §3.

**Anh em vẫn phân biệt được với nhau**, nhưng bằng khác biệt thị trường thật chứ không bằng nhãn định danh: `gap_open_atr` (0.0005), `mom_diff` (0.018), `vwap_slope_atr` (0.157). Đây đúng là cách các mẫu chồng lấn khác nhau trong một chiến lược bình thường.

---

## §2. Vì sao bộ này "leak tốt"

Điều đáng chú ý: **nhóm feature mạnh nhất cũng chính là nhóm có sib_corr cao nhất.** `breakeven_R` (0.999), `vol200` (0.998), `atr14_pct` (0.979) vừa là ba feature dẫn đầu về sức phân biệt, vừa gần như đồng nhất giữa 4 anh em. Cộng với tương quan nhãn 0.58–0.73 (§4 báo cáo hyperparameter), kênh leak hình thành **tự nhiên**, không cần và không nên cố ý dựng thêm.

Mật độ chồng lấn đo trên `dataset_catboost.csv`: **77.3%** số mẫu có cửa sổ nhãn phủ lên mẫu kế tiếp; độ dài cửa sổ nhãn trung bình 20.3 bar M15.

---

## §3. Danh sách loại trừ và lý do

| Loại bỏ | Lý do |
|---|---|
| `atr90_pct` | Spearman **1.000** với `breakeven_R` — cùng một feature, chỉ khác hằng số nhân (`breakeven_R = atr90_pct × 3.8/0.006`). Giữ cả hai là nhân đôi trọng số cho một chiều thông tin. |
| `entry_price` (thô) | sib_corr = **0.9999** — gần như là vân tay định danh gia đình, và với BTC nó xấp xỉ một hàm đơn điệu của thời gian. Kiểm chứng thực nghiệm: thêm vào làm **Cách 1 tăng 0.7944 → 0.8091** trong khi **Cách 2 gần như đứng yên (0.5952 → 0.5976)**. Tức nó bơm thêm leak **chỉ cho** nhánh random CV, qua cơ chế *không phải* chồng lấn nhãn → làm nhiễu chính đại lượng đồ án muốn đo. (Ghi chú: nó **không** phá walk-forward như lo ngại ban đầu — cây không ngoại suy được nhưng cũng không sụp; lý do loại là nhiễu quy kết, không phải sụp hiệu năng.) |
| `upper_barrier`, `lower_barrier`, `atr_frozen` (thô) | Cùng vấn đề: đều xấp xỉ mức giá tuyệt đối. Thông tin hữu ích của chúng đã được chắt vào `breakeven_R` ở dạng không thứ nguyên. |
| `leg`, `entry_vs_base_R` | **Giàn giáo của thí nghiệm, không phải thông tin thị trường** (nguyên tắc §0.3). "Lệnh nhồi thứ mấy" và "vào cao hơn lệnh gốc bao nhiêu" chỉ tồn tại vì đồ án cố ý chọn một chiến lược cực đoan để phóng đại chồng lấn; chiến lược thật không có hai khái niệm này. Kiểm chứng: bỏ ra **không đổi kết quả** — chênh lệch rò rỉ thuần +0.0902 (còn giữ: +0.0907), rò rỉ tổng hợp +0.2029 (còn giữ: +0.1992). Khớp với tỷ lệ nhãn 1 phẳng theo leg (26.5 / 26.7 / 25.9 / 25.7%). Hai cột vẫn nằm trong CSV ở nhóm `META` để phân tích kết quả và dựng lại đường vốn. |
| `vol50`, `ret5_atr`, `pos_in_range20` | Dư thừa với feature cùng họ đã có trong bộ (Spearman 0.84–0.88). Tiêu chí loại là tương quan **giữa các feature**, không đụng nhãn. |
| `bars_to_label`, `exit_touch_time`, `bars_held`, `exit_reason`, `pl_pct`, `R` | Thông tin tương lai. Nhìn thấy nhãn. |

---

## §4. Cấu hình CatBoost đề xuất

```python
CatBoostClassifier(
    iterations=1000,          # CỐ ĐỊNH, không early stopping — xem cảnh báo bên dưới
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    auto_class_weights="Balanced",   # 26.16% lớp 1
    eval_metric="AUC",
    random_seed=42,
    verbose=200,
)
```

**Không cần chuẩn hoá / one-hot.** CatBoost xử lý thang đo và NaN nội bộ. Dataset hiện tại 0 NaN.

**Cảnh báo 1 — không dùng early stopping.** [`bao_cao_hyperparameter_triple_barrier.md`](bao_cao_hyperparameter_triple_barrier.md) cam kết siêu tham số cố định cho cả ba cách chia. Early stopping làm số vòng lặp khác nhau giữa ba cách → siêu tham số không còn cố định, và tập validation dùng để dừng lại là một kênh rò rỉ nữa. Chốt cứng `iterations` cho cả ba.

**Cảnh báo 2 — khoan dùng `cat_features` ở vòng đầu.** Đây là điểm phản trực giác với thế mạnh quảng cáo của CatBoost. Ordered target statistics mã hoá biến phân loại **bằng chính nhãn của tập train**; ở Cách 1, tập train chứa anh em của mẫu test → thống kê đó có dính nhãn anh em, tức **kênh leak thứ hai** nằm ngoài chồng lấn nhãn. Với `hour`/`dow` lực leak này bị pha loãng qua hàng nghìn mẫu nên nhỏ, nhưng vẫn làm bẩn quy kết. Giữ `hour`/`dow` ở dạng **số** để đồ án chỉ còn **đúng một kênh leak**. Muốn thử `cat_features` thì để thành một nhánh phụ có đối chứng, sau khi đã có kết quả chính.

**Cảnh báo 3 — bảy cột `META` tuyệt đối không đưa vào `X`.** Gồm `entry_time`, `label_end_time`, `origin_bar`, `entry_bar`, `entry_price`, `leg`, `entry_vs_base_R`. `build_features.py` để chúng cuối file và tách riêng qua hằng `META`; 23 feature nằm ở hằng `FEATURES`. Cứ lấy đúng `FEATURES` là an toàn.

---

## §5. Cột metadata phục vụ chia dữ liệu

| Cột | Dùng cho |
|---|---|
| `entry_time` | t₀ — mốc bắt đầu; sắp xếp thời gian cho Cách 2/3 |
| `label_end_time` | **t₁ — mốc nhãn được xác định. Bắt buộc phải có thì Cách 3 mới purge được.** Bằng `exit_touch_time` nếu chạm hàng rào, ngược lại bằng thời điểm kết thúc bar thứ 50. |
| `origin_bar` | Định danh gia đình. Dùng cho `GroupKFold` ở chẩn đoán §6 và để khử trùng. |
| `entry_bar`, `entry_price` | Đối chiếu ngược với backtest / dựng equity curve |

Không có `label_end_time` thì purging ở Cách 3 không thực hiện được — đây là thứ dễ bỏ sót nhất khi dựng dataset.

---

## §6. Kết quả nền — CatBoost 1.2.10 (`run_catboost_baseline.py`, nay là [`scripts/train_models.py`](../scripts/train_models.py))

Cùng một mô hình, **siêu tham số giống hệt nhau** ở cả bốn nhánh (1000 vòng, lr=0.05, depth=6, `auto_class_weights='Balanced'`, seed=42). Chỉ đổi luật chia dữ liệu.

| Cách chia | ROC-AUC | F1 | AUC từng fold |
|---|---|---|---|
| Cách 1 — random 5-fold, **xé** anh em | **0.8630** | 0.662 | 0.853 · 0.869 · 0.870 · 0.862 · 0.861 |
| Cách 1b — random 5-fold, **giữ** gia đình | **0.7381** | 0.508 | 0.724 · 0.729 · 0.746 · 0.729 · 0.762 |
| Cách 2 — walk-forward expanding | **0.5867** | 0.327 | 0.612 · 0.551 · 0.550 · 0.634 |
| Cách 3 — WF + purge/embargo | **0.5948** | 0.330 | 0.619 · 0.559 · 0.562 · 0.640 |

| Chênh lệch | Giá trị | Diễn giải |
|---|---|---|
| Cách 1 − Cách 1b | **+0.1249** | Leak **thuần** do xé anh em — cùng thời gian, cùng cỡ tập train |
| Cách 1 − Cách 2 | **+0.2763** | Leak tổng hợp (lẫn cả khác biệt cỡ tập train) |
| Cách 1 − Cách 3 | **+0.2682** | Tổng mức thổi phồng |
| Cách 2 − Cách 3 | −0.0081 | **Nhiễu, không phải hiệu ứng** — xem §6.2 |

**Dataset khả dụng:** Cách 2 đạt 0.587 > 0.5, mô hình học được thật chứ không chỉ ghi nhớ. Mức thổi phồng +0.276 AUC đủ lớn để làm kết quả chính.

**CatBoost khai thác leak mạnh hơn GBDT thường.** Cùng dataset, `HistGradientBoostingClassifier` (300 vòng) cho Cách 1 = 0.7988 và leak thuần +0.090; CatBoost (1000 vòng) cho 0.8630 và leak thuần **+0.1249**. Mô hình càng mạnh càng giỏi ghi nhớ mẫu gần trùng — bản thân quan sát này đáng đưa vào báo cáo.

**Bỏ giàn giáo không mất gì:** loại `leg` và `entry_vs_base_R` (25→23 feature) làm các con số dịch ở mức nhiễu, xác nhận hai cột đó không mang tín hiệu thật mà chỉ mang rủi ro phương pháp.

### §6.2. Phát hiện ngoài dự kiến: purge/embargo gần như không có tác dụng

Bảng 1 trong [`tom_tat_idea_goc.md`](tom_tat_idea_goc.md) dự đoán Cách 3 **thấp hơn** Cách 2. Thực tế hai cách gần như bằng nhau. Nguyên nhân đã truy được, và **không phải lỗi cài đặt**:

| Fold | Mẫu train | Bị purge/embargo cắt | Tỷ lệ |
|---|---|---|---|
| 1 | 5,001 | 21 | 0.42% |
| 2 | 10,003 | 3 | 0.03% |
| 3 | 15,004 | 8 | 0.05% |
| 4 | 20,006 | 2 | 0.01% |

Kiểm chứng độc lập: số mẫu train có `label_end_time` ≥ thời điểm test bắt đầu đúng bằng số mẫu bị purge (21/2/1/2). Cài đặt chuẩn.

**Lý do mang tính bản chất:** trong walk-forward, mỗi fold chỉ có **một ranh giới**. Cửa sổ nhãn dài 50 bar M15 = 12.5 giờ, trong đó chỉ có vài tín hiệu. Vùng nhiễm bẩn vì thế chỉ là một vệt mỏng so với hàng nghìn mẫu train. Ngược lại khi chia ngẫu nhiên thì **mọi mẫu đều nằm cạnh một ranh giới** — chồng lấn lan khắp dữ liệu.

**Xác nhận −0.0081 là nhiễu:** chạy lại Cách 2 ba lần trên cùng dữ liệu, chỉ đổi `random_seed` → AUC = 0.5867 / 0.5930 / 0.5877, **biên độ dao động 0.0063**. Chênh lệch Cách 2−Cách 3 cùng bậc độ lớn nên không quy cho purge/embargo được.

**Hệ quả — cần sửa kỳ vọng ở Bảng 1 trước khi viết báo cáo:**

1. Kết luận đúng là **Cách 3 ≈ Cách 2**, không phải "Cách 3 thấp hơn". Đây không phải thất bại mà là kết quả có ý nghĩa: **thiệt hại do nhãn chồng lấn là vấn đề của chia ngẫu nhiên, gần như không phải vấn đề của walk-forward.**
2. **Cách 1b trở thành nhánh quan trọng nhất.** Nó là nhánh duy nhất cô lập được leak chồng lấn ở dạng sạch (+0.1249, cùng thời gian cùng cỡ tập train), trong khi cặp Cách 2 vs Cách 3 mà thiết kế gốc kỳ vọng lại gần như không nói lên điều gì. Chi phí thêm nhánh này gần bằng 0: chỉ đổi `KFold` thành `GroupKFold(groups=origin_bar)`.
3. Mọi chênh lệch nhỏ hơn **≈0.007 AUC** trong đồ án này phải coi là nhiễu. Nên công bố kèm nền nhiễu này để không diễn giải quá tay.

---

## §7. Bước tiếp theo

1. Chốt siêu tham số một lần bằng walk-forward + purge/embargo trên 80% đầu
2. Cố định tỷ lệ giữ lệnh 50%, dựng equity curve cho cả bốn nhánh
3. Mở holdout đúng một lần
4. Khi viết báo cáo: sửa kỳ vọng Bảng 1 theo §6.2, và công bố nền nhiễu 0.0063 kèm mọi so sánh
