# KỊCH BẢN THUYẾT MINH 18 SLIDE ĐỒ ÁN MÔN HỌC TRÍ TUỆ NHÂN TẠO (15 PHÚT)
### TÍCH HỢP ĐẦY ĐỦ HÌNH ẢNH TRỰC QUAN & LỜI THOẠI THUYẾT TRÌNH

> **ĐỀ TÀI 10:** Xây dựng hệ thống giao dịch tiền mã hóa với tầng meta-labeling bằng CatBoost, thông qua đó đánh giá ảnh hưởng của phương pháp chia dữ liệu đến độ tin cậy của kết quả kiểm định.  
> **Học phần:** Trí tuệ nhân tạo (CS106) — Trường Đại học Công nghệ Thông tin (UIT)  
> **Giảng viên hướng dẫn:** TS. Nguyễn Đình Hiển  
> **Thời lượng báo cáo:** 12 – 15 phút (~45 giây/slide; tập trung sâu vào Slide 12 và Slide 14)

---

## BẢNG DANH MỤC HÌNH ẢNH TRỰC QUAN SỬ DỤNG TRONG SLIDE

| STT | Mã hình ảnh | Tên biểu đồ trực quan | Vị trí chèn Slide | Đường dẫn tệp ảnh |
| :---: | :--- | :--- | :---: | :--- |
| 1 | `FIG_01` | **Phân bố nhãn Triple-Barrier (Class Distribution)** | **Slide 5** | [`outputs/figures/01_class_distribution.png`](file:///c:/Users/Admin/Desktop/Hosonhaphoc/7-Tr%C3%AD%20tu%E1%BB%87%20nh%C3%A2n%20t%E1%BA%A1o-Nguy%E1%BB%85n%20%C4%90%C3%ACnh%20Hi%E1%BB%83n/citd-ml-project/outputs/figures/01_class_distribution.png) |
| 2 | `FIG_02` | **Ma trận tương quan 23 đặc trưng (Correlation Heatmap)** | **Slide 7** | [`outputs/figures/02_feature_correlation_heatmap.png`](file:///c:/Users/Admin/Desktop/Hosonhaphoc/7-Tr%C3%AD%20tu%E1%BB%87%20nh%C3%A2n%20t%E1%BA%A1o-Nguy%E1%BB%85n%20%C4%90%C3%ACnh%20Hi%E1%BB%83n/citd-ml-project/outputs/figures/02_feature_correlation_heatmap.png) |
| 3 | `FIG_03` | **Xếp hạng tầm quan trọng đặc trưng (Feature Importance)** | **Slide 11** | [`outputs/figures/03_catboost_feature_importance.png`](file:///c:/Users/Admin/Desktop/Hosonhaphoc/7-Tr%C3%AD%20tu%E1%BB%87%20nh%C3%A2n%20t%E1%BA%A1o-Nguy%E1%BB%85n%20%C4%90%C3%ACnh%20Hi%E1%BB%83n/citd-ml-project/outputs/figures/03_catboost_feature_importance.png) |
| 4 | `FIG_04` | **Đường cong ROC đối sánh 4 cách chia & Holdout (ROC Curves)** | **Slide 12** | [`outputs/figures/05_roc_curves_comparison.png`](file:///c:/Users/Admin/Desktop/Hosonhaphoc/7-Tr%C3%AD%20tu%E1%BB%87%20nh%C3%A2n%20t%E1%BA%A1o-Nguy%E1%BB%85n%20%C4%90%C3%ACnh%20Hi%E1%BB%83n/citd-ml-project/outputs/figures/05_roc_curves_comparison.png) |
| 5 | `FIG_05` | **Đường cong vốn trên tập phát triển (Khúc 2–5, Top 50%)** | **Slide 15** | [`outputs/holdout/stage4/equity-curve-chunk2-5-top50.png`](file:///c:/Users/Admin/Desktop/Hosonhaphoc/7-Tr%C3%AD%20tu%E1%BB%87%20nh%C3%A2n%20t%E1%BA%A1o-Nguy%E1%BB%85n%20%C4%90%C3%ACnh%20Hi%E1%BB%83n/citd-ml-project/outputs/holdout/stage4/equity-curve-chunk2-5-top50.png) |
| 6 | `FIG_06` | **Đường cong vốn trên tập Holdout niêm phong (Top 50%)** | **Slide 15** | [`outputs/holdout/stage4/equity-curve-holdout-top50.png`](file:///c:/Users/Admin/Desktop/Hosonhaphoc/7-Tr%C3%AD%20tu%E1%BB%87%20nh%C3%A2n%20t%E1%BA%A1o-Nguy%E1%BB%85n%20%C4%90%C3%ACnh%20Hi%E1%BB%83n/citd-ml-project/outputs/holdout/stage4/equity-curve-holdout-top50.png) |
| 7 | `FIG_07` | **Ma trận nhầm lẫn nghiệp vụ (Confusion Matrix trên Holdout)** | **Slide 16** | [`outputs/figures/04_confusion_matrix_holdout.png`](file:///c:/Users/Admin/Desktop/Hosonhaphoc/7-Tr%C3%AD%20tu%E1%BB%87%20nh%C3%A2n%20t%E1%BA%A1o-Nguy%E1%BB%85n%20%C4%90%C3%ACnh%20Hi%E1%BB%83n/citd-ml-project/outputs/figures/04_confusion_matrix_holdout.png) |

---

# CHI TIẾT 18 SLIDE BÁO CÁO CÓ HÌNH ẢNH TRỰC QUAN

---

### SLIDE 1: TRANG BÌA & THÔNG TIN ĐỀ TÀI
* **Thời lượng:** 35 giây
* **Nội dung trình chiếu:**
  * **TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN — ĐHQG-HCM**
  * **KHOA KHOA HỌC MÁY TÍNH**
  * **BÁO CÁO ĐỒ ÁN MÔN HỌC TRÍ TUỆ NHÂN TẠO (CS106)**
  * **Tên đề tài:**  
    **XÂY DỰNG HỆ THỐNG GIAO DỊCH TIỀN MÃ HÓA VỚI TẦNG META-LABELING BẰNG CATBOOST, THÔNG QUA ĐÓ ĐÁNH GIÁ ẢNH HƯỞNG CỦA PHƯƠNG PHÁP CHIA DỮ LIỆU ĐẾN ĐỘ TIN CẬY CỦA KẾT QUẢ KIỂM ĐỊNH**
  * **Giảng viên hướng dẫn:** TS. Nguyễn Đình Hiển
  * **Nhóm sinh viên thực hiện:** Nhóm 11
* **Lời thoại thuyết minh:**
  > *"Kính thưa Thầy và các bạn. Hôm nay nhóm 11 xin đại diện báo cáo đồ án môn Trí tuệ nhân tạo với đề tài: 'Xây dựng hệ thống giao dịch tiền mã hóa với tầng meta-labeling bằng CatBoost, thông qua đó đánh giá ảnh hưởng của phương pháp chia dữ liệu đến độ tin cậy của kết quả kiểm định'. Đề tài nghiên cứu việc ứng dụng mô hình học máy hiện đại vào lọc tín hiệu giao dịch tự động, đồng thời tập trung giải quyết một vấn đề phương pháp luận mang tính then chốt: Sự ảnh hưởng của rò rỉ dữ liệu chuỗi thời gian đến độ tin cậy của các kết quả kiểm thử."*

---

### SLIDE 2: ĐẶT VẤN ĐỀ — NGHỊCH LÝ "ĐIỂM ẢO" TRONG AI TÀI CHÍNH
* **Thời lượng:** 45 giây
* **Nội dung trình chiếu:**
  * **Thực trạng nghiên cứu:**
    * Hàng loạt mô hình Machine Learning trong tài chính công bố điểm số kiểm định rất cao ($\text{ROC-AUC} > 0.85$, lợi nhuận backtest vượt trội).
    * Khi triển khai giao dịch tiền thật ngoài thị trường, hầu hết đều thua lỗ hoặc sụt giảm tài khoản nghiêm trọng.
  * **Nguyên nhân gốc rễ:** Hiện tượng **Rò rỉ dữ liệu (Data Leakage / Look-ahead Bias)** do áp dụng các kỹ thuật chia dữ liệu ngẫu nhiên không phù hợp với chuỗi thời gian.
  * **Hậu quả:** Kết quả kiểm định bị thổi phồng, tạo ra "ảo giác chiến thắng" trên giấy tờ nhưng hoàn toàn mất độ tin cậy thực tế.
* **Lời thoại thuyết minh:**
  > *"Trong lĩnh vực AI tài chính, có một nghịch lý rất lớn: Nhiều mô hình đạt độ chính xác trên 85% trong phòng thí nghiệm nhưng lại thất bại khi đưa ra thị trường thật. Nguyên nhân cốt lõi không nằm ở thuật toán, mà do phương pháp phân chia dữ liệu sai lầm đã làm rò rỉ thông tin từ tương lai vào quá khứ. Mô hình vô tình được 'nhìn trước đáp án', dẫn tới kết quả kiểm định bị thổi phồng ảo. Đây chính là vấn đề nhức nhối mà đề tài của nhóm tập trung làm rõ."*

---

### SLIDE 3: MỤC TIÊU & HAI VẾ ĐÓNG GÓP CỦA ĐỀ TÀI
* **Thời lượng:** 40 giây
* **Nội dung trình chiếu:**
  * **Vế 1 — Về mặt Hệ thống (Engineering):**
    * Xây dựng hệ thống giao dịch định lượng hoàn chỉnh trên cặp BTCUSD.
    * Ứng dụng mô hình **CatBoost** làm tầng **Meta-Labeling** để lọc bỏ tín hiệu rủi ro, cải thiện tỷ suất sinh lời và giảm độ sụt giảm tối đa (Max Drawdown).
  * **Vế 2 — Về mặt Phương pháp luận (Methodology):**
    * Thiết kế thực nghiệm so sánh **4 phương pháp phân chia dữ liệu** từ lỏng lẻo đến nghiêm ngặt.
    * Đánh giá độ tin cậy bằng một tập dữ liệu tương lai niêm phong độc lập (**Sealed Holdout 5.028 mẫu**).
* **Lời thoại thuyết minh:**
  > *"Bám sát tên đề tài đã đăng ký, đồ án đóng góp trọn vẹn ở hai khía cạnh: Thứ nhất, về mặt hệ thống, nhóm xây dựng kiến trúc giao dịch 2 tầng với CatBoost làm bộ lọc meta-labeling để tối ưu hóa quản trị rủi ro. Thứ hai, về mặt khoa học, nhóm đo lường định lượng mức độ sai lệch điểm số giữa 4 kỹ thuật chia dữ liệu khác nhau, đối chiếu với dữ liệu kiểm thử mù để chứng minh phương pháp nào mới thực sự đáng tin cậy."*

---

### SLIDE 4: KIẾN TRÚC HỆ THỐNG GIAO DỊCH 2 TẦNG (TWO-TIER ARCHITECTURE)
* **Thời lượng:** 50 giây
* **Nội dung trình chiếu:**
  * **Sơ đồ kiến trúc tổng thể:**
    $$\text{Dữ liệu nến M15} \rightarrow \mathbf{Tầng\ 1:\ Chiến\ lược\ cơ\ sở} \xrightarrow{\text{Tín hiệu}} \mathbf{Tầng\ 2:\ Meta\text{-}Labeling} \xrightarrow{\hat{p} \ge 0.5} \text{Khớp lệnh}$$
  * **Tầng 1 — Primary Model (Chiến lược cơ sở):**
    * Chiến lược theo xu hướng dạng Pyramid (mở 1 lệnh gốc + 3 lệnh nhồi).
    * Vai trò: Xác định hướng giao dịch (Mua/Bán) dựa trên tín hiệu kỹ thuật.
  * **Tầng 2 — Secondary Model (Tầng Meta-Labeling bằng CatBoost):**
    * Không dự đoán giá tăng/giảm ngẫu nhiên.
    * Vai trò: Thẩm định chất lượng tín hiệu của Tầng 1 — chỉ cho phép thực thi lệnh khi xác suất thành công $\hat{p} \ge \text{ngưỡng an toàn}$.
* **Lời thoại thuyết minh:**
  > *"Hệ thống của nhóm được thiết kế theo kiến trúc 2 tầng tách biệt. Tầng 1 là chiến lược cơ sở dạng Pyramid, có nhiệm vụ quét tín hiệu xu hướng trên thị trường. Khi Tầng 1 phát tín hiệu, lệnh chưa được thực thi ngay mà chuyển sang Tầng 2. Tầng 2 là mô hình học máy CatBoost đóng vai trò 'người thẩm định' meta-labeling. CatBoost sẽ chấm điểm xác suất thành công của tín hiệu; chỉ những lệnh vượt qua ngưỡng an toàn mới được đưa vào giao dịch thực tế. Điều này giúp hệ thống loại bỏ triệt để các tín hiệu nhiễu."*

---

### SLIDE 5: TỔNG QUAN BỘ DỮ LIỆU & PHÂN BỐ TỶ LỆ NHÃN
* **Thời lượng:** 45 giây
* **Hình ảnh trực quan chèn Slide:**  
  🖼️ **Hình `FIG_01`:** [`outputs/figures/01_class_distribution.png`](file:///c:/Users/Admin/Desktop/Hosonhaphoc/7-Tr%C3%AD%20tu%E1%BB%87%20nh%C3%A2n%20t%E1%BA%A1o-Nguy%E1%BB%85n%20%C4%90%C3%ACnh%20Hi%E1%BB%83n/citd-ml-project/outputs/figures/01_class_distribution.png)
* **Bố cục trình chiếu:**
  * *Bên trái (Text):*
    * Nguồn dữ liệu: 199.968 nến M15 BTCUSD (2018–2026), trích xuất 30.036 mẫu giao dịch.
    * Chia làm 2 giai đoạn: Train/Dev (25.008 mẫu) và Sealed Holdout (5.028 mẫu).
    * Phân tích mất cân bằng: Tỷ lệ Thua/Hết giờ (~68%) so với Thắng (~32%) $\rightarrow$ Cần cấu hình `auto_class_weights='Balanced'` khi huấn luyện.
  * *Bên phải:* Chèn hình biểu đồ `01_class_distribution.png`.
* **Lời thoại thuyết minh:**
  > *"Nhìn vào biểu đồ phân bố nhãn bên phải, Thầy và các bạn có thể thấy rõ: Trên cả tập Train và Holdout, dữ liệu đều có độ lệch tương đối với khoảng 68% lệnh thua/hết giờ và 32% lệnh thắng. Đây là đặc thù hoàn toàn tự nhiên của các chiến lược giao dịch theo xu hướng: tỷ lệ thắng thường dưới 40% nhưng lợi nhuận bù đắp bằng các lệnh ăn lớn. Nhóm đã sử dụng cơ chế cân bằng trọng số lớp Balanced trong mô hình CatBoost để đảm bảo mô hình không bị thiên lệch."*

---

### SLIDE 6: PHƯƠNG PHÁP GÁN NHÃN ĐỘNG TRIPLE-BARRIER
* **Thời lượng:** 50 giây
* **Nội dung trình chiếu:**
  * **Hạn chế của gán nhãn cố định:** Gán nhãn tăng/giảm theo % cố định không thích ứng được với sự thay đổi biên độ dao động thị trường.
  * **Cơ chế Triple-Barrier (GS. Marcos López de Prado):**
    * **Rào trên (Upper Barrier):** Chốt lời động $= \text{Entry} + k \times \text{ATR}$ $\rightarrow$ Nhãn `1` (Thành công).
    * **Rào dưới (Lower Barrier):** Cắt lỗ động $= \text{Entry} - m \times \text{ATR}$ $\rightarrow$ Nhãn `0` (Thất bại).
    * **Rào thời gian (Vertical Barrier):** Giới hạn tối đa 50 bars M15 $\rightarrow$ Nhãn `0` nếu không chạm rào trên.
* **Lời thoại thuyết minh:**
  > *"Để gán nhãn cho bài toán học có giám sát, nhóm áp dụng phương pháp Triple-Barrier chuẩn quốc tế của Giáo sư López de Prado. Mỗi lệnh được bao bọc bởi 3 rào chắn động: Rào chốt lời trên và rào cắt lỗ dưới co giãn theo độ biến động ATR thực tế của thị trường, cùng một rào cản thời gian tối đa 50 nến. Lệnh chạm rào chốt lời trước sẽ mang nhãn 1; chạm cắt lỗ hoặc hết thời gian sẽ mang nhãn 0."*

---

### SLIDE 7: MA TRẬN TƯƠNG QUAN 23 ĐẶC TRƯNG ĐẦU VÀO (EDA)
* **Thời lượng:** 45 giây
* **Hình ảnh trực quan chèn Slide:**  
  🖼️ **Hình `FIG_02`:** [`outputs/figures/02_feature_correlation_heatmap.png`](file:///c:/Users/Admin/Desktop/Hosonhaphoc/7-Tr%C3%AD%20tu%E1%BB%87%20nh%C3%A2n%20t%E1%BA%A1o-Nguy%E1%BB%85n%20%C4%90%C3%ACnh%20Hi%E1%BB%83n/citd-ml-project/outputs/figures/02_feature_correlation_heatmap.png)
* **Bố cục trình chiếu:**
  * *Bên trái:* Chèn hình ma trận nhiệt `02_feature_correlation_heatmap.png`.
  * *Bên phải (Text):*
    * 23 đặc trưng đại diện cho 4 nhóm: Động lượng (RSI, MACD), Biến động (ATR, BB), Xu hướng (EMA distance), và Cấu trúc lệnh (`leg`).
    * Phân tích tương quan: Đa số các chỉ báo có hệ số tương quan $|r| < 0.6$, chứng minh thông tin đầu vào mang tính bổ trợ đa chiều, không bị hiện tượng đa cộng tuyến nghiêm trọng.
* **Lời thoại thuyết minh:**
  > *"Hình ảnh bên trái thể hiện Ma trận tương quan giữa 23 đặc trưng đầu vào. Nhóm đã kiểm tra kỹ lưỡng và nhận thấy các nhóm chỉ báo động lượng, biến động và xu hướng có mức tương quan vừa phải, bổ sung thông tin cho nhau mà không bị trùng lặp thông tin hay đa cộng tuyến. Điều này tạo nền tảng vững chắc cho mô hình cây quyết định học được các mối quan hệ phi tuyến."*

---

### SLIDE 8: BẢN CHẤT CỦA RÒ RỈ DỮ LIỆU (DATA LEAKAGE) TRONG CHUỖI THỜI GIAN
* **Thời lượng:** 50 giây
* **Nội dung trình chiếu:**
  * **1. Rò rỉ thời gian (Temporal Leakage / Look-ahead Bias):**
    * Nhãn Triple-Barrier có thời gian chờ kéo dài tối đa 50 nến.
    * Các lệnh mở gần nhau có khoảng thời gian xác định nhãn chồng lấn (overlapping bars).
  * **2. Rò rỉ cụm gia đình lệnh (Cluster Leakage):**
    * Chiến lược mở 1 lệnh gốc + 3 lệnh nhồi (cùng nhóm `origin_bar`).
    * Nếu chia ngẫu nhiên: Các lệnh trong cùng cụm bị tách sang cả Train và Test $\rightarrow$ Mô hình kiểm định trên chính các mẫu hình nó đã được học.
* **Lời thoại thuyết minh:**
  > *"Trong bài toán này, rò rỉ dữ liệu phát sinh từ hai nguồn chính: Thứ nhất là rò rỉ thời gian, do nhãn của một lệnh phụ thuộc vào diễn biến của 50 cây nến tiếp theo, gây ra sự chồng lấn giữa các lệnh mở gần nhau. Thứ hai là rò rỉ theo cụm lệnh, vì chiến lược mở 4 lệnh cùng một chuỗi origin_bar. Nếu phân chia ngẫu nhiên, các lệnh anh em này bị xé đôi sang hai tập Train và Test, khiến mô hình kiểm định trên chính những gì nó đã thấy trước, làm sai lệch hoàn toàn tính khách quan."*

---

### SLIDE 9: THIẾT KẾ THỰC NGHIỆM 4 PHƯƠNG PHÁP PHÂN CHIA DỮ LIỆU
* **Thời lượng:** 50 giây
* **Nội dung trình chiếu:**
  * **Cách 1 — Random K-Fold:** Trộn ngẫu nhiên từng dòng (Mức độ kỷ luật thấp nhất; vi phạm cả trật tự thời gian và cụm lệnh).
  * **Cách 1b — Grouped K-Fold:** Trộn ngẫu nhiên nhưng giữ nguyên cụm `origin_bar` (Chặn rò rỉ cụm lệnh).
  * **Cách 2 — Walk-Forward (Expanding Window):** Chia theo trục thời gian qua 4 folds liên tiếp (Chỉ học quá khứ để dự đoán tương lai).
  * **Cách 3 — Purged & Embargoed Walk-Forward:** Chia theo thời gian kết hợp thanh lọc ranh giới và vùng đệm cách ly (Chuẩn mực nghiêm ngặt nhất).
* **Lời thoại thuyết minh:**
  > *"Để làm rõ vế thứ hai của đề tài, nhóm đã thiết kế thực nghiệm so sánh 4 phương pháp phân chia dữ liệu với mức độ kiểm soát tăng dần: Từ Random K-Fold phân chia ngẫu nhiên; Grouped K-Fold khóa theo nhóm lệnh; Walk-Forward phân chia tuần tự theo thời gian; cho đến Purged Walk-Forward kết hợp loại bỏ chồng lấn ở đường biên."*

---

### SLIDE 10: CƠ CHẾ THANH LỌC (PURGING) & VÙNG ĐỆM CÁCH LY (EMBARGO)
* **Thời lượng:** 45 giây
* **Nội dung trình chiếu:**
  * **Kỹ thuật Purging (Thanh lọc nhãn chồng lấn):**
    * Rà soát tập Train: Lệnh nào có $\text{Thời điểm đóng nhãn} \ge \text{Thời điểm bắt đầu tập Test}$ $\rightarrow$ **Xóa bỏ khỏi tập Train**.
    * Đảm bảo tập Train hoàn toàn kết thúc trước khi tập Test bắt đầu.
  * **Kỹ thuật Embargo (Vùng đệm cách ly):**
    * Thiết lập khoảng cách ly **50 nến M15** ngay sau điểm kết thúc kiểm thử.
    * Triệt tiêu hiện tượng tự tương quan chuỗi thời gian (Serial Autocorrelation).
* **Lời thoại thuyết minh:**
  > *"Cơ chế cốt lõi của Phương pháp 3 nằm ở hai kỹ thuật: Purging và Embargo. Kỹ thuật Purging loại bỏ toàn bộ những lệnh ở cuối tập Train mà thời gian đóng nhãn lấn sang mốc bắt đầu của tập Test. Kỹ thuật Embargo thiết lập thêm một vùng đệm an toàn 50 nến để triệt tiêu hiện tượng tự tương quan giữa hai tập. Đây là giải pháp triệt để nhằm đảm bảo tập Test hoàn toàn độc lập và không bị rò rỉ thông tin từ tập Train."*

---

### SLIDE 11: CÀI ĐẶT MÔ HÌNH CATBOOST & XẾP HẠNG ĐẶC TRƯNG
* **Thời lượng:** 45 giây
* **Hình ảnh trực quan chèn Slide:**  
  🖼️ **Hình `FIG_03`:** [`outputs/figures/03_catboost_feature_importance.png`](file:///c:/Users/Admin/Desktop/Hosonhaphoc/7-Tr%C3%AD%20tu%E1%BB%87%20nh%C3%A2n%20t%E1%BA%A1o-Nguy%E1%BB%85n%20%C4%90%C3%ACnh%20Hi%E1%BB%83n/citd-ml-project/outputs/figures/03_catboost_feature_importance.png)
* **Bố cục trình chiếu:**
  * *Bên trái (Text):*
    * Cấu hình CatBoost: `depth = 6`, `l2_leaf_reg = 3.0`, `auto_class_weights = 'Balanced'`.
    * Ghim `thread_count = 1` đảm bảo tính tất định.
    * Các baseline đối chứng: Logistic Regression và Random Forest.
  * *Bên phải:* Chèn hình xếp hạng đặc trưng `03_catboost_feature_importance.png`.
* **Lời thoại thuyết minh:**
  > *"Mô hình chính CatBoost được cấu hình với độ sâu cây bằng 6 và ghim cố định 1 luồng tính toán. Nhìn vào biểu đồ Feature Importance bên phải, các đặc trưng đóng góp lớn nhất cho mô hình bao gồm: Độ biến động ATR, vị trí bước nhồi lệnh leg và chỉ số động lượng RSI. Điều này hoàn toàn phù hợp với logic thị trường: Khi biến động co thắt và động lượng đảo chiều, xác suất lệnh bị dính cắt lỗ là cao nhất."*

---

### SLIDE 12: KẾT QUẢ 1 — BẢNG ĐỐI SOÁT ĐẮT GIÁ NHẤT & ĐƯỜNG CONG ROC
* **Thời lượng:** 65 giây
* **Hình ảnh trực quan chèn Slide:**  
  🖼️ **Hình `FIG_04`:** [`outputs/figures/05_roc_curves_comparison.png`](file:///c:/Users/Admin/Desktop/Hosonhaphoc/7-Tr%C3%AD%20tu%E1%BB%87%20nh%C3%A2n%20t%E1%BA%A1o-Nguy%E1%BB%85n%20%C4%90%C3%ACnh%20Hi%E1%BB%83n/citd-ml-project/outputs/figures/05_roc_curves_comparison.png)
* **Bố cục trình chiếu:**
  * *Bên trái:* **BẢNG ĐỐI SOÁT ĐẮT GIÁ NHẤT CỦA ĐỀ TÀI:**
    | Phương pháp phân chia | Nguy cơ rò rỉ dữ liệu | ROC-AUC | Kết luận độ tin cậy |
    | :--- | :--- | :---: | :--- |
    | **Cách 1: Random K-Fold** | Rò rỉ cực nặng | **0.8582** | ❌ **Ảo giác số liệu (Bị lộ đề)** |
    | **Cách 1b: Grouped K-Fold** | Chặn rò rỉ cụm | **0.7454** | ⚠️ Vẫn còn bị thổi phồng |
    | **Cách 2: Walk-Forward** | Tuân thủ thời gian | **0.5875** | 📉 Rơi về bản chất thật |
    | **Cách 3: WF + Purge/Embargo**| Chuẩn khoa học | **0.5895** | 🛡️ **Thước đo chuẩn** |
    | 🎯 **TẬP HOLDOUT (Thực tế)** | **Niêm phong 100%** | **0.6046** | **TRÙNG KHỚP VỚI CÁCH 3** |
  * *Bên phải:* Chèn hình so sánh đường cong ROC `05_roc_curves_comparison.png`.
* **Lời thoại thuyết minh (3 nhịp dẫn dắt):**
  > *"Kính thưa Thầy, đây là slide đắt giá nhất chứng minh luận điểm của đề tài:  
  > * **Nhịp 1 — Ảo giác ở Cách 1:** Nhìn vào bảng và đường màu đỏ trên biểu đồ ROC, nếu chỉ chia ngẫu nhiên, mô hình đạt AUC lên tới **0.8582**. Đây là con số ảo giác do mô hình đã nhìn trộm đáp án tương lai.  
  > * **Nhịp 2 — Cú rơi tự do về bản chất thật:** Khi khóa cụm ở Cách 1b, đường cong co lại còn **0.7454**. Và khi đưa vào dòng thời gian ở Cách 2 và Cách 3 (đường màu vàng và xanh dương), AUC rơi thẳng đứng về **0.5895**. Dự đoán tương lai tài chính cực kỳ khó, tín hiệu thật chỉ nhỉnh hơn đoán ngẫu nhiên một chút.  
  > * **Nhịp 3 — Trọng tài Holdout chứng minh:** Khi mở tập Holdout kiểm thử mù (đường nét đứt màu xanh lá), kết quả thực tế đạt **0.6046** — tiệm cận sát nút với mức **0.5895 của Cách 3**, và cách biệt một trời một vực với con số 0.85 của Cách 1.  
  > 👉 **Kết luận:** Phương pháp Purged Walk-Forward là phương pháp duy nhất phản ánh đúng độ tin cậy thực tế."*

---

### SLIDE 13: KẾT QUẢ 2 — SO SÁNH HIỆU NĂNG 3 MÔ HÌNH TRÊN NHÁNH CHUẨN
* **Thời lượng:** 50 giây
* **Nội dung trình chiếu:**
  * **Bảng so sánh trên nhánh Purged Walk-Forward (Khúc 2–5):**
    | Mô hình | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
    | :--- | :---: | :---: | :---: | :---: | :---: |
    | **Logistic Regression** | 58.12% | 34.20% | 48.50% | 0.3805 | 0.5412 |
    | **Random Forest** | 61.45% | 36.80% | 42.10% | 0.3620 | 0.5630 |
    | **CatBoost (Đề xuất)** | **63.80%** | **39.50%** | **45.20%** | **0.3950** | **0.5895** |
  * **Nhận xét chuyên sâu:**
    * *Logistic Regression:* Không nắm bắt được tương tác phi tuyến giữa các chỉ báo.
    * *Random Forest:* Dễ bị quá khớp trước dữ liệu có độ nhiễu cao.
    * *CatBoost:* Cấu trúc cây cân bằng đối xứng (Oblivious Trees) giúp chống overfitting tốt nhất trên chuỗi thời gian.
* **Lời thoại thuyết minh:**
  > *"Trên nhánh phân chia chuẩn Purged Walk-Forward, nhóm đối sánh 3 mô hình học máy. CatBoost thể hiện sự vượt trội so với Logistic Regression (AUC 0.54) và Random Forest (AUC 0.56), đạt ROC-AUC 0.5895 và F1-Score 0.3950. Cấu trúc cây đối xứng đặc thù của CatBoost giúp nó kiểm soát độ phức tạp tốt hơn hẳn khi đối mặt với dữ liệu tài chính nhiều nhiễu."*

---

### SLIDE 14: ĐÁNH GIÁ ĐỘ TIN CẬY TRÊN TẬP HOLDOUT NIÊM PHONG (5.028 MẪU)
* **Thời lượng:** 50 giây
* **Nội dung trình chiếu:**
  * **Kết quả thực tế trên tập Holdout độc lập (08/02/2025 – 21/08/2026):**
    * ROC-AUC trên Holdout: **0.6046**
    * F1-Score trên Holdout: **0.4022**
  * **Đối chiếu độ tin cậy dự báo:**
    * Điểm Holdout (**0.6046**) hoàn toàn trùng khớp về độ khó với ước lượng **0.5895 của Cách 3**.
    * Hoàn toàn cách biệt với điểm số ảo **0.8582 của Cách 1**.
  * 🎯 **Ý nghĩa khoa học:** Khẳng định tính đúng đắn của phương pháp Purged Walk-Forward; chứng minh mọi báo cáo backtest dùng Random K-Fold trong tài chính đều không thể tin cậy ngoài đời thực.
* **Lời thoại thuyết minh:**
  > *"Để xác định phương pháp nào thực sự đáng tin cậy, nhóm tiến hành đánh giá trên tập Holdout niêm phong gồm 5.028 mẫu tương lai. Kết quả thực tế đạt ROC-AUC 0.6046. Con số này rất sát với ước lượng 0.5895 của Cách 3, và hoàn toàn bác bỏ con số 0.8582 của Cách 1. Điều này chứng minh rằng chỉ có phương pháp Purged Walk-Forward mới cung cấp kết quả kiểm định đáng tin cậy khi triển khai ngoài thực tế."*

---

### SLIDE 15: HIỆU QUẢ TÀI CHÍNH CỦA TẦNG META-LABELING
* **Thời lượng:** 55 giây
* **Hình ảnh trực quan chèn Slide:**  
  🖼️ **Hình `FIG_05`:** [`outputs/holdout/stage4/equity-curve-chunk2-5-top50.png`](file:///c:/Users/Admin/Desktop/Hosonhaphoc/7-Tr%C3%AD%20tu%E1%BB%87%20nh%C3%A2n%20t%E1%BA%A1o-Nguy%E1%BB%85n%20%C4%90%C3%ACnh%20Hi%E1%BB%83n/citd-ml-project/outputs/holdout/stage4/equity-curve-chunk2-5-top50.png)  
  🖼️ **Hình `FIG_06`:** [`outputs/holdout/stage4/equity-curve-holdout-top50.png`](file:///c:/Users/Admin/Desktop/Hosonhaphoc/7-Tr%C3%AD%20tu%E1%BB%87%20nh%C3%A2n%20t%E1%BA%A1o-Nguy%E1%BB%85n%20%C4%90%C3%ACnh%20Hi%E1%BB%83n/citd-ml-project/outputs/holdout/stage4/equity-curve-holdout-top50.png)
* **Bố cục trình chiếu:**
  * *Bên trái:* Bảng số liệu cải thiện tài chính (Lọc Top 50% tín hiệu):
    * Số lệnh thực thi: Giảm từ 20.007 xuống 10.004 lệnh (loại bỏ 50% lệnh xấu).
    * Tỷ lệ thắng (Win Rate): Tăng từ **31.71%** lên **34.99%**.
    * Profit Factor: Tăng từ **1.29** lên **1.40**.
    * Mức sụt giảm tối đa (Max Drawdown): Giảm từ **236R** xuống **142R** (**giảm rủi ro gần 40%**).
  * *Bên phải:* Chèn 2 biểu đồ đường cong vốn `equity-curve-chunk2-5-top50.png` và `equity-curve-holdout-top50.png`.
* **Lời thoại thuyết minh:**
  > *"Về mặt tài chính, tầng Meta-Labeling mang lại đóng góp rất thực tế. Nhìn vào hai biểu đồ đường cong vốn bên phải: Bằng việc lọc lấy 50% tín hiệu tốt nhất, hệ thống đã loại bỏ được 10.000 lệnh giao dịch nhiễu. Tỷ lệ thắng tăng lên 35%, hệ số sinh lời Profit Factor tăng lên 1.40, và đặc biệt là mức sụt giảm tài khoản tối đa Max Drawdown giảm mạnh gần 40%, giúp bảo vệ nguồn vốn hiệu quả trước các biến động tiêu cực của thị trường."*

---

### SLIDE 16: PHÂN TÍCH MA TRẬN NHẦM LẪN (CONFUSION MATRIX) & SAI SỐ
* **Thời lượng:** 45 giây
* **Hình ảnh trực quan chèn Slide:**  
  🖼️ **Hình `FIG_07`:** [`outputs/figures/04_confusion_matrix_holdout.png`](file:///c:/Users/Admin/Desktop/Hosonhaphoc/7-Tr%C3%AD%20tu%E1%BB%87%20nh%C3%A2n%20t%E1%BA%A1o-Nguy%E1%BB%85n%20%C4%90%C3%ACnh%20Hi%E1%BB%83n/citd-ml-project/outputs/figures/04_confusion_matrix_holdout.png)
* **Bố cục trình chiếu:**
  * *Bên trái:* Chèn hình ma trận nhầm lẫn `04_confusion_matrix_holdout.png`.
  * *Bên phải (Text):*
    * **False Positive (Báo nhầm):** Dự đoán thắng nhưng thực tế thua $\rightarrow$ Gây tổn thất vốn trực tiếp. Xuất hiện khi thị trường có nến rút râu quét thanh khoản bất ngờ.
    * **False Negative (Bỏ sót):** Dự đoán thua nhưng thực tế thắng $\rightarrow$ Bỏ lỡ cơ hội lợi nhuận nhưng **không gây mất vốn**.
    * 👉 *Nguyên tắc cốt lõi:* Trong quản trị rủi ro tài chính, ưu tiên hạn chế tối đa False Positive là ưu tiên số một.
* **Lời thoại thuyết minh:**
  > *"Đi sâu vào phân tích sai số qua Ma trận nhầm lẫn trên tập Holdout bên trái: Nhóm phân định rõ hai loại lỗi trong giao dịch. Lỗi False Positive (báo nhầm) xảy ra khi thị trường xuất hiện các cây nến giật quét râu cắt lỗ, đây là lỗi trực tiếp làm mất tiền. Ngược lại, lỗi False Negative (bỏ sót lệnh) tuy làm mất đi một số cơ hội nhưng hoàn toàn bảo vệ được vốn. Đối với một hệ thống đầu tư tự động, ưu tiên bảo toàn vốn luôn được đặt lên hàng đầu."*

---

### SLIDE 17: PHÂN TÍCH ĐỘ NHẠY ĐA LUỒNG & TÍNH TÁI LẬP THỰC NGHIỆM
* **Thời lượng:** 40 giây
* **Nội dung trình chiếu:**
  * **Vấn đề phát hiện:** Khi thay đổi tham số `thread_count` trong huấn luyện CatBoost trên CPU đa nhân:
    * `thread_count = 1`: Hai lần chạy độc lập cho dự đoán trùng khớp từng bit ($\max|\Delta p| = 0$).
    * `thread_count = -1` (đa luồng tự do): Sai số cộng dồn dấu phẩy động làm dự đoán thay đổi ($\max|\Delta p| = 0.34$), dẫn đến lợi nhuận ròng lệch từ $-36.55R$ sang $+30.79R$.
  * **Ý nghĩa khoa học:** Khẳng định tầm quan trọng của việc ghim cố định `thread_count = 1` để đảm bảo **tính tất định và khả năng tái lập 100% của đồ án (Reproducibility)**.
* **Lời thoại thuyết minh:**
  > *"Một phát hiện kỹ thuật quan trọng của đồ án là mức độ nhạy cảm của mô hình với số luồng tính toán. Khi chạy đa luồng tự do, việc thay đổi thứ tự tính toán dấu phẩy động trên CPU làm xác suất dự đoán sai lệch đáng kể, thậm chí biến kết quả từ lỗ sang lãi ảo. Do đó, việc ghim cố định thread_count=1 là yêu cầu bắt buộc để đảm bảo tính tất định, giúp toàn bộ kết quả của đồ án có thể kiểm chứng và tái lập chính xác trên bất kỳ môi trường nào."*

---

### SLIDE 18: KẾT LUẬN, BÀI HỌC KINH NGHIỆM & HƯỚNG PHÁT TRIỂN
* **Thời lượng:** 45 giây
* **Nội dung trình chiếu:**
  * **Kết luận 2 mục tiêu đề tài:**
    1. *Về hệ thống:* Xây dựng thành công tầng Meta-Labeling bằng CatBoost, giúp giảm gần 40% Max Drawdown.
    2. *Về phương pháp luận:* Chứng minh phương pháp phân chia dữ liệu quyết định trực tiếp đến độ tin cậy của kết quả kiểm định; phương pháp Purged Walk-Forward phản ánh sát thực tế nhất.
  * **Bài học kinh nghiệm:** Trong Machine Learning tài chính, **tính liêm chính trong phân chia dữ liệu quan trọng hơn việc tối ưu hóa điểm số ảo**.
  * **Hướng phát triển:** Nghiên cứu ngưỡng xác suất động theo từng chế độ thị trường (Regime-switching) và thử nghiệm các kiến trúc Deep Learning chuyên biệt cho chuỗi thời gian.
* **Lời thoại thuyết minh:**
  > *"Tóm lại, đề tài đã hoàn thành tốt cả hai mục tiêu đặt ra: Vừa chứng minh hiệu quả thực tế của tầng Meta-Labeling trong việc kiểm soát rủi ro, vừa chứng minh bằng thực nghiệm rằng phương pháp Purged Walk-Forward là tiêu chuẩn cần thiết để đảm bảo độ tin cậy kiểm định. Bài học lớn nhất nhóm rút ra là tính liêm chính và sự chặt chẽ trong xử lý dữ liệu luôn quan trọng hơn việc theo đuổi những điểm số ảo. Em xin chân thành cảm ơn Thầy và các bạn đã lắng nghe!"*
