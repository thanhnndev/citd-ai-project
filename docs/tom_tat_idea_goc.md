# Tóm tắt idea gốc

> **Nguồn và trạng thái.** Đây là bản đề cương gốc của đồ án, viết **trước** khi chạy
> bất kỳ thí nghiệm nào. Giữ nguyên để đối chiếu kỳ vọng ban đầu với kết quả thực tế.
>
> **Hai chỗ thực tế đã đi khác đề cương:**
> 1. Đề cương dùng **ba** cách chia. Thực tế chốt **bốn** — Cách 1b (Grouped K-Fold)
>    được bổ sung và trở thành nhánh quan trọng nhất.
> 2. Bảng 1 dự đoán Cách 3 (purge/embargo) thấp hơn Cách 2. Thực tế hai cách gần bằng
>    nhau, chênh lệch nằm trong nền nhiễu.
>
> Cả hai được truy nguyên ở §6.2 của [`bao_cao_feature.md`](bao_cao_feature.md).

## Tên đề tài
**Xây dựng hệ thống giao dịch crypto bằng CatBoost. Thông qua đó đánh giá tác động của phương pháp chia dữ liệu đến hiệu năng mô hình học máy lọc tín hiệu giao dịch (khai thác rò rỉ overlaping window do non-IID)**

---

## Bối cảnh và vấn đề

Trong giao dịch tài chính, một cách ứng dụng học máy phổ biến là dùng mô hình để **lọc bớt tín hiệu** từ một chiến lược gốc: tín hiệu nào mô hình cho là sắp thua thì bỏ qua, không vào lệnh. Mô hình không quyết định mua/bán, chỉ quyết định có nghe theo chiến lược gốc hay không.

Vấn đề nằm ở khâu **đo xem việc lọc đó có thật sự hiệu quả hay không**. Có hai sai sót thường gặp:

1. **Nhãn của các mẫu chồng lấn nhau về thời gian (không độc lập)**. Nhãn "thắng/thua" được xác định bằng cách nhìn về tương lai một khoảng thời gian nhất định, nên các tín hiệu gần nhau có nhãn được quyết định bởi cùng một đoạn thị trường. Khi chia dữ liệu ngẫu nhiên, các mẫu chồng lấn bị tách về hai phía huấn luyện/kiểm tra, khiến mô hình đã gián tiếp thấy trước đáp án — làm hiệu năng bị thổi phồng.

2. **So sánh không có mốc đối chứng.** Sau khi lọc, số lệnh giảm đi, nên mức sụt giảm vốn cũng giảm. Nhưng điều này đúng với bất kỳ cách lọc nào, kể cả lọc bừa. Nếu chỉ so "trước lọc" với "sau lọc", không thể tách được phần cải thiện do mô hình thông minh khỏi phần cải thiện chỉ vì vào ít lệnh hơn.

Đề tài này **không nhằm xây dựng mô hình dự báo tốt hơn**, mà nhằm **đo lường mức độ sai lệch** khi đánh giá mô hình trên dữ liệu có tính chất chuỗi thời gian và nhãn chồng lấn.

---

## Mục tiêu

- Xây dựng mô hình CatBoost lọc tín hiệu từ một nhóm chiến lược chưa tối ưu.
- So sánh kết quả đánh giá của **ba cách chia dữ liệu** trên cùng một mô hình.
- Dùng **một giai đoạn dữ liệu độc lập** (holdout) để xác định cách chia nào ít thiên lệch nhất.

---

## Dữ liệu

- **Nguồn dữ liệu:** dữ liệu K-line (OHLCV) của tài sản có biến động cao, ưu tiên tiền mã hóa (BTC, ETH hoặc tương tự) vì dữ liệu miễn phí, đầy đủ, giao dịch liên tục 24/7 không có khoảng trống cuối tuần. Khung thời gian nhỏ để đảm bảo đủ số lượng tín hiệu.
- **Quy mô:** tính toán lấy sao cho đủ entry point làm dữ liệu huấn luyện cho CatBoost, đồng thời đủ dài để tách được một giai đoạn kiểm chứng độc lập ở cuối chuỗi.
- **Chiến lược sinh tín hiệu:** nhóm các chiến lược mua bán (trading strategy) chưa tối ưu. Yêu cầu: sinh ra nhiều tín hiệu, mật độ dày, hiện tượng chồng lấn nhãn lên ngưỡng cực đoan có thể quan sát được. Chiến lược này không cần hiệu quả — nó chỉ đóng vai trò nguồn tín hiệu để lọc.
- **Gán nhãn:** với mỗi tín hiệu, đặt ba mốc — chốt lời, cắt lỗ, và hạn chót thời gian. Giá chạm mốc nào trước thì tín hiệu kết thúc tại đó. Thắng/lãi gán nhãn 1, thua/lỗ gán nhãn 0. *(Phương pháp triple-barrier)* Tuy nhiên cần cân nhắc thêm về trường hợp chạm biên thời gian.
- **Đặc trưng đầu vào cho CatBoost:** chọn các chỉ báo tính tại thời điểm phát sinh tín hiệu — ví dụ như volatility, trend, distance to EMA,... Toàn bộ chỉ chứa thông tin có sẵn tại thời điểm đó, không dùng dữ liệu tương lai.

---

## Phương pháp

### Bước 0: Tách holdout trước tiên

Tách riêng **20% dữ liệu cuối chuỗi** và niêm phong lại. Toàn bộ các bước huấn luyện, lựa chọn tham số, đánh giá bằng ba cách chia chỉ thực hiện trên **80% dữ liệu còn lại**. Holdout chỉ được mở đúng một lần ở bước cuối.

### Bước 1: Huấn luyện mô hình lọc tín hiệu

Sử dụng **cùng một kiến trúc CatBoost** và cùng bộ đặc trưng. Trong mỗi cách chia, mô hình được huấn luyện lại trên từng fold tương ứng. So sánh kết quả là so sánh cách chia dữ liệu, không phải so sánh các mô hình khác nhau.

*Lưu ý về siêu tham số:* Để đảm bảo so sánh công bằng, **toàn bộ siêu tham số của CatBoost được cố định trước** và áp dụng cho cả ba cách chia. Quy trình chọn siêu tham số sẽ được thực hiện một lần duy nhất trên tập 80% đầu bằng phương pháp walk‑forward có purge/embargo (Cách 3) để tránh rò rỉ, hoặc dùng giá trị mặc định phổ biến nếu thời gian hạn chế. Mô hình không được tối ưu siêu tham số riêng cho từng cách chia.

*Lưu ý về mất cân bằng lớp:* Kiểm tra tỷ lệ lớp thắng/thua. Nếu mất cân bằng, có thể dùng trọng số lớp trong CatBoost để tránh mô hình thiên vị lớp đa số. Trọng số lớp cũng được cố định cho cả ba cách.

### Bước 2: Đánh giá bằng ba cách chia dữ liệu

| Cách chia | Mô tả | Rò rỉ còn lại |
|---|---|---|
| **Cách 1 – Chia ngẫu nhiên (Random 5‑fold CV)** | Trộn toàn bộ tín hiệu, bốc ngẫu nhiên vào 5 fold. Lần lượt lấy từng fold làm kiểm tra, 4 fold còn lại làm huấn luyện. | Rò rỉ dữ liệu tương lai + rò rỉ nhãn chồng lấn lan rộng khắp dữ liệu |
| **Cách 2 – Walk‑Forward 5 fold (expanding window)** | Cắt dữ liệu thành 5 khúc liên tiếp theo thời gian. Học khúc 1 → kiểm tra khúc 2; học khúc 1+2 → kiểm tra khúc 3; ... Luôn chỉ dùng quá khứ để dự đoán tương lai. | Chỉ còn rò rỉ do nhãn chồng lấn tại ranh giới |
| **Cách 3 – Walk‑Forward + Purging & Embargo (expanding window)** | Giống Cách 2, thêm: **purging** — loại khỏi phần huấn luyện mọi tín hiệu có khoảng thời gian nhãn chồng lên phần kiểm tra; **embargo** — sau khi purge, loại thêm khỏi phần huấn luyện các mẫu nằm trong vùng đệm ngay sau ranh giới để tránh những mẫu có nhãn kết thúc sát ranh giới vẫn còn ảnh hưởng. Cả purge và embargo **chỉ tác động lên tập huấn luyện, không loại bỏ tín hiệu khỏi phần kiểm tra**. | Giảm thiểu tối đa rò rỉ do chồng lấn nhãn (không còn rò rỉ đáng kể trong phạm vi nghiên cứu) |

**Diễn giải sự khác biệt giữa ba cách:**

- Sự khác biệt giữa Cách 1 và Cách 2 phản ánh **tổng hợp của hai yếu tố**: (1) việc chia ngẫu nhiên cho phép mô hình học dữ liệu tương lai, và (2) mức độ rò rỉ do chồng lấn nhãn — ở Cách 1, chồng lấn lan rộng khắp dữ liệu; ở Cách 2, chỉ giới hạn tại các ranh giới. Tuy nhiên, do sự khác biệt về kích thước tập huấn luyện (Cách 1 train trên ~80% dữ liệu, Cách 2/3 train trên trung bình ~50%), chênh lệch giữa Cách 1 và Cách 2 không chỉ do leakage mà còn do lượng dữ liệu học. Hạn chế này được thừa nhận trong phần Hạn chế.

- Sự khác biệt giữa Cách 2 và Cách 3 phản ánh **chủ yếu tác động của việc xử lý nhãn chồng lấn tại ranh giới**, vì đây là điểm khác biệt chính giữa hai cách. Tuy nhiên, Cách 3 huấn luyện trên ít mẫu hơn Cách 2 (do purge/embargo loại bỏ một số mẫu huấn luyện), nên một phần chênh lệch có thể đến từ việc giảm kích thước tập huấn luyện, không hoàn toàn do loại bỏ rò rỉ. 

**Lưu ý khi thực hiện:**

- Purging và embargo chỉ tác động lên **phần huấn luyện**, không loại bỏ tín hiệu khỏi phần kiểm tra.
- Cách 2 và Cách 3 không dự đoán được khúc đầu tiên (vì chưa có quá khứ để học), trong khi Cách 1 dự đoán toàn bộ. Để so sánh công bằng, **chỉ tính toán chỉ số trên phần thời gian chung của cả ba cách** — tức bỏ khúc đầu ở cả ba.
- **Hạn chế cần ghi nhận:** Cách 1 luôn huấn luyện trên ~80% dữ liệu (4 fold), trong khi Cách 2/3 sử dụng expanding window nên kích thước tập huấn luyện trung bình khoảng 50%. Sự khác biệt này có thể làm cho Cách 2/3 kém hơn không chỉ vì không có rò rỉ mà còn vì học từ ít dữ liệu hơn. Do đó, khi so sánh Cách 1 với Cách 2/3, cần thận trọng: chênh lệch quan sát được không hoàn toàn chỉ do leakage, mà có thể bao gồm một phần do kích thước tập huấn luyện. Đây là một hạn chế của thiết kế thực nghiệm.

### Bước 3: Đo hiệu năng

- **Cấp độ phân loại:** ROC‑AUC, F1‑Score.  

- **Cấp độ tài chính:** ghép các dự đoán thành đường vốn, tính lợi nhuận ròng, mức sụt giảm vốn sâu nhất, số lệnh còn lại sau lọc.  
  

### Bước 4: Chọn ngưỡng xác suất — cố định tỷ lệ giữ lệnh (selection rate)

Để chuyển từ xác suất dự đoán sang quyết định giữ/bỏ, **không cố định một ngưỡng cứng** (ví dụ 0.5) cho cả ba cách, vì điều đó dẫn đến số lệnh giữ lại khác nhau giữa các cách, gây nhiễu cho so sánh tài chính. Đây chính là lỗi thứ hai mà đề tài muốn vạch ra.

Thay vào đó, **cố định tỷ lệ giữ lệnh** (selection rate) — ví dụ chọn giữ lại đúng **50%** số lệnh có xác suất dự đoán cao nhất cho mỗi cách chia. Khi đó:

- Số lệnh giữ lại là như nhau giữa ba cách.
- Sự khác biệt về lợi nhuận/drawdown chỉ còn phản ánh **chất lượng chọn lệnh**, không phải do vào ít lệnh hơn.


### Bước 5: Đối chiếu holdout

Sau khi đã hoàn tất mọi lựa chọn (đặc trưng, tỷ lệ giữ lệnh, tham số mô hình) trên 80% dữ liệu đầu, huấn luyện mô hình trên toàn bộ 80% đó rồi chạy một lần duy nhất trên holdout 20%.

*Làm sạch ranh giới cho holdout:* Trước khi huấn luyện mô hình cuối cùng, **phải loại bỏ khỏi tập huấn luyện (80% đầu) tất cả các mẫu có nhãn chồng lấn với khoảng thời gian của holdout**, tương tự như purge/embargo đã áp dụng trong Cách 3. Điều này đảm bảo holdout không bị rò rỉ từ quá khứ. Vùng đệm (embargo) có thể chừa thêm một đoạn ngắn sau ranh giới 80/20 để an toàn. Tập holdout vẫn giữ nguyên, không loại bỏ mẫu nào.

*Vì sao holdout đáng tin hơn:* chỉ có một chiều thời gian, chỉ có một ranh giới cần làm sạch, và chỉ được sử dụng một lần (tránh overfitting do thử nhiều lần).

*Lưu ý:* So sánh giữa ba cách chia (trên 80% đầu) với holdout (20% cuối) có thể bị nhiễu khá nhiều do khác biệt thị trường (non‑stationarity trong non-IID phần không cùng phân phối). Do đó, holdout đóng vai trò tham chiếu tương đối, còn kết luận chính vẫn dựa trên chênh lệch giữa ba cách chia. Trình bày kết quả cần tách bạch: các chỉ số của ba cách chia trên 80% đầu, và riêng kết quả holdout, không đưa ra cột “sai lệch so với holdout” như một thước đo tuyệt đối.

---

## Kết quả dự kiến

### Bảng 1 – Chỉ số phân loại theo từng cách chia (trên 80% đầu)

| Cách chia | ROC‑AUC | F1‑Score (ngưỡng 0.5) |
|---|---|---|
| Cách 1 – Chia ngẫu nhiên | cao hơn đáng kể | cao hơn đáng kể |
| Cách 2 – Walk‑Forward | trung bình | trung bình |
| Cách 3 – Walk‑Forward + Purging/Embargo | thấp hơn, gần với holdout hơn | thấp hơn, gần với holdout hơn |

*Các mô tả trên chỉ phản ánh kỳ vọng tương đối dựa trên lý thuyết rò rỉ dữ liệu, không phải kết quả thực tế.*

### Bảng 2 – Chỉ số tài chính (tỷ lệ giữ lệnh cố định, tính trên phần thời gian chung)

| Cách chia | Số lệnh giữ lại | Lợi nhuận ròng | Sụt giảm sâu nhất |
|---|---|---|---|
| Cách 1 – Chia ngẫu nhiên | cố định 50% | cao hơn | thấp hơn |
| Cách 2 – Walk‑Forward | cố định 50% | trung bình | trung bình |
| Cách 3 – Walk‑Forward + Purging/Embargo | cố định 50% | thấp hơn | cao hơn |
| **Baseline (không lọc)** | 100% | tham chiếu | tham chiếu |

*Baseline không lọc được tính trên đúng phần thời gian chung của cả ba cách (tức từ fold 2 đến fold 5).*

### Sản phẩm chính

1. **Bảng so sánh ba cách chia và holdout.**
2. **Biểu đồ đường vốn (equity curve) của ba cách chia vẽ chồng lên nhau** trên cùng một khoảng thời gian, cùng với baseline (chiến lược gốc không lọc).

---

## Hạn chế đã lường trước

- **Holdout chỉ là một mẫu duy nhất.** Nếu giai đoạn đó rơi vào một trạng thái thị trường đặc thù, con số thu được có thể lệch. Báo cáo sẽ giữ đúng mức độ: holdout là mốc đối chiếu tương đối, không phải hiệu năng tuyệt đối.
- **Kết quả về hiệu suất giao dịch gắn với nhóm chiến lược gốc và một tài sản cụ thể.** Mức độ thổi phồng phụ thuộc trực tiếp vào mật độ tín hiệu và độ dài hạn chót của nhãn.
- **Khác biệt kích thước tập huấn luyện giữa các cách chia.** Cách 1 train trên ~80% dữ liệu, Cách 2/3 train trên trung bình ~50% do expanding window. Điều này có thể làm chênh lệch kết quả giữa các cách không chỉ do leakage mà còn do lượng dữ liệu học. Đây là hạn chế cố hữu của walk‑forward nhiều fold và được chấp nhận trong phạm vi đề tài.
- **Khác biệt kích thước huấn luyện giữa Cách 2 và Cách 3.** Do purge/embargo loại bỏ một số mẫu huấn luyện, Cách 3 huấn luyện trên ít dữ liệu hơn Cách 2. Điều này có thể ảnh hưởng đến so sánh giữa hai cách. 
---
