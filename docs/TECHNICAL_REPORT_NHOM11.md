# BÁO CÁO KỸ THUẬT — HỆ THỐNG PYRAMID BTCUSD VÀ BỘ LỌC CATBOOST

> **Tài liệu làm việc cho nhóm.** Bố cục, câu hỏi nghiên cứu, thuật ngữ và cách ghi số của tài liệu này theo [báo cáo khoa học của nhóm trưởng](../deliverables/Scientific_report_Nhom11.pdf). Đây là bản giải thích triển khai và bản đồ bằng chứng để các thành viên viết phần kỹ thuật; khi trích dẫn kết quả, dùng cây canonical `thread_count=1` và ghi rõ phạm vi kiểm chứng. Các bảng trong báo cáo khoa học là bản trình bày chính thức.

## TÓM TẮT

Đồ án đo ảnh hưởng của **cách chia dữ liệu** lên độ tin cậy của phép kiểm định một bộ lọc giao dịch. Chiến lược pyramid trên BTCUSD M15 sinh lệnh ứng viên; CatBoost chỉ chấm xác suất một lệnh có tín hiệu tốt. Bốn cách chia giữ nguyên dữ liệu, đặc trưng và cấu hình mô hình: Random K-Fold, Grouped K-Fold, Walk-forward, Purged Walk-forward kèm embargo. Kết quả được đối chiếu với 5.028 lệnh holdout sau mốc đã chốt. Trên khúc 2–5, ROC-AUC đi từ 0,8582 (Random) xuống 0,5895 (Purged Walk-forward), còn holdout đạt 0,6046. Bộ lọc top 50% trên holdout đạt −36,55 R so với baseline +245,93 R. Kết luận kỹ thuật là Random K-Fold tạo ước lượng lạc quan trong thiết kế có họ lệnh và nhãn chồng lấn; kết quả holdout không xác nhận khả năng cải thiện lợi nhuận của bộ lọc.

**Từ khóa:** meta-labeling, rò rỉ nhãn, triple barrier, walk-forward, purging, embargo, CatBoost, holdout niêm phong, tái lập kết quả.

## CHƯƠNG 1. GIỚI THIỆU VÀ PHÁT BIỂU VẤN ĐỀ

### 1.1. Bối cảnh và câu hỏi nghiên cứu

Một tín hiệu gốc của chiến lược mở một lệnh nền và ba lệnh chồng (`leg` 0–3). Bốn lệnh dùng chung `origin_bar`, có cửa sổ quan sát nhãn gần nhau và các đặc trưng tương tự. Nếu chia từng dòng ngẫu nhiên, thành viên một họ có thể nằm cả trong train lẫn test. Mô hình khi đó hưởng lợi từ thông tin rất gần mẫu kiểm tra. Đồ án trả lời ba câu hỏi của báo cáo khoa học: (RQ1) cách chia làm thay đổi chỉ số bao nhiêu; (RQ2) điểm số có hội tụ khi kiểm soát gia đình và thời gian không; (RQ3) cách chia nào gần quan sát holdout hơn.

### 1.2. Phạm vi hệ thống

CatBoost là **tầng meta-labeling**, không tạo tín hiệu mua/bán. Chiến lược sơ cấp sinh vũ trụ lệnh; mô hình xếp hạng các lệnh trong vũ trụ đó. Backtest dùng vũ trụ ứng viên cố định: loại một lệnh không thay đổi tín hiệu chiến lược xuất hiện về sau. Vì vậy các chỉ số đo chất lượng phép **lọc trên cùng danh sách ứng viên**, không phải hiệu năng của hệ thống giao dịch chạy lại động sau mỗi quyết định bỏ lệnh. Mã chiến lược: [`pyramid_strategy.py`](../src/citd_ml/strategy/pyramid_strategy.py); mã backtest: [`backtest_pyramid_local.py`](../src/citd_ml/backtest/backtest_pyramid_local.py).

## CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ QUY TẮC GÁN NHÃN

### 2.1. Triple barrier theo giao dịch

Nhãn 1 biểu thị lệnh đạt trạng thái không lỗ theo mô phỏng trailing; nhãn 0 biểu thị lỗ hoặc hết chặn thời gian. Ngưỡng dưới là stop-loss 0,6% dưới giá vào. Ngưỡng trên là giá vào cộng `3,8 × ATR(90)` đóng băng trước entry; khi chạm mức này, trailing stop có thể dời lên hòa vốn. Chặn thời gian là 50 bar M15. Thứ tự chạm mốc được phân giải bằng nến M1 bên trong M15. Cùng một luật được áp dụng riêng cho cả bốn leg, dùng giá vào và ATR của từng leg. Sơ đồ sáu quy tắc và phân tích chọn tham số: [`thong_so_triple_barrier.png`](thong_so_triple_barrier.png), [`bao_cao_hyperparameter_triple_barrier.md`](bao_cao_hyperparameter_triple_barrier.md); mã: [`triple_barrier.py`](../src/citd_ml/labeling/triple_barrier.py).

Chọn ba leg có cơ sở xác suất: trong random 5-fold, xác suất cả ba “anh em” của một lệnh đều nằm trong fold test là `(1/5)^3 = 0,8%`. Phép tính này cho thấy khả năng cùng họ xuất hiện phía train rất cao; các số đo tương quan nhãn và đặc trưng ở báo cáo tham số xác nhận tính gần trùng của các leg. Đây là cơ chế khiến Random K-Fold dễ lạc quan, không phải bằng chứng rằng mọi chênh lệch AUC đều do một nguồn rò rỉ duy nhất.

### 2.2. Bốn cách chia

| Cách chia trong báo cáo khoa học | Ràng buộc chính | Rủi ro được kiểm soát |
|---|---|---|
| Cách 1 — Random K-Fold | Chia dòng ngẫu nhiên | Mốc tham chiếu lạc quan; có thể tách cùng họ và đảo chiều thời gian |
| Cách 1b — Grouped K-Fold | Cùng `origin_bar` ở một phía | Giảm thông tin từ các leg cùng họ; vẫn có thể dùng tương lai để dự đoán quá khứ |
| Cách 2 — Walk-forward | Train quá khứ, test tương lai | Tôn trọng thứ tự thời gian |
| Cách 3 — Purged Walk-forward | Walk-forward, thêm kiểm tra chồng lấn nhãn và embargo 50 bar | Siết điều kiện gần biên train/test |

Mã chia fold: [`split_data.py`](../src/citd_ml/training/split_data.py). Trên bốn biên fold của bước 4, số dòng bị loại là **21, 3, 8, 2** (tổng 34); vì vậy tác dụng bổ sung của purge/embargo trong thiết lập này nhỏ. Phần holdout train cuối có kiểm tra purge/embargo nhưng loại **0 dòng**. Hai con số thuộc **hai giai đoạn khác nhau**, không được nhập làm một.

### 2.3. Chỉ số

ROC-AUC đo thứ hạng xác suất giữa hai lớp; F1 ở ngưỡng 0,5 đo cân bằng precision/recall tại ngưỡng đó. Với chiến lược giao dịch, báo cáo thêm net profit tính theo **R** (bội số rủi ro ban đầu), MaxDD của đường vốn, profit factor (tổng lãi / trị tuyệt đối tổng lỗ), win rate và số lệnh. AUC tốt không tự động suy ra lợi nhuận tốt: hai phép đo có đối tượng và hàm mục tiêu khác nhau.

## CHƯƠNG 3. DỮ LIỆU VÀ ĐẶC TRƯNG

### 3.1. Nguồn và ranh giới

Dữ liệu giá gốc là BTCUSD M1; chiến lược và đặc trưng vận hành trên M15, dùng M1 để xác định thứ tự sự kiện trong nến. File thô khoảng 209 MB không được commit; quy cách đặt file xem [`data/raw/README.md`](../data/raw/README.md). Dataset đã xử lý và kết quả đã commit, nên team có thể kiểm tra bảng kết quả và train từ dataset sẵn có; sinh lại dataset hoặc backtest cần file M1 thô.

Mốc tách holdout là **2025-02-08 15:30:00**, bar M15 số **199.968**. Dataset trước holdout có **25.008 lệnh**, gồm **6.252 họ × 4 leg**, tỷ lệ nhãn 1 khoảng **26,2%**. Holdout có **5.028 lệnh**. Có **4 lệnh biên** vào trước mốc nhưng cửa sổ nhãn vượt qua mốc: không thuộc train hoặc holdout. Toàn lịch sử tái sinh vì vậy gồm `25.008 + 4 + 5.028 = 30.040` lệnh. Bằng chứng ranh giới và bốn khóa lệnh biên: [`BAO_CAO_KET_QUA_HOLDOUT.md`](BAO_CAO_KET_QUA_HOLDOUT.md), mục 1.7.

### 3.2. Đặc trưng và metadata

Mô hình dùng **23 đặc trưng** đã xác định tại thời điểm vào lệnh, chia thành nhóm hình học hàng rào, chế độ biến động, xu hướng/vị trí giá, tín hiệu chiến lược và khối lượng/phiên. **Bảy cột metadata** chỉ phục vụ định danh, gán nhãn, chia fold hoặc đối chiếu, không vào ma trận feature. Đặc biệt `leg` và `entry_vs_base_R` mô tả giàn giáo pyramid, có thể tiết lộ cấu trúc họ. Danh sách và lý do chọn/loại từng cột: [`bao_cao_feature.md`](bao_cao_feature.md); định nghĩa được code sử dụng: [`build_features.py`](../src/citd_ml/features/build_features.py). Khi viết bảng đặc trưng, lấy tên cột từ mã và báo cáo này, không suy từ toàn bộ cột CSV.

Ranh giới năm khúc của 25.008 dòng là `[0, 5001, 10003, 15004, 20006, 25008]`. **Khúc 1 được loại khỏi phần so sánh chung** để cả bốn nhánh có cùng phạm vi đánh giá là khúc 2–5 (**20.007 lệnh**). Không so AUC của một nhánh trên năm khúc với nhánh khác trên bốn khúc.

## CHƯƠNG 4. PHƯƠNG PHÁP THỰC NGHIỆM VÀ TRIỂN KHAI

### 4.1. Kiểm soát biến

Giữ cố định dataset, 23 feature, nhãn, danh sách ứng viên, CatBoost và quy tắc top-k; biến so sánh chính là cách chia train/test. Cấu hình CatBoost: `iterations=1000`, `learning_rate=0.05`, `depth=6`, `l2_leaf_reg=3.0`, `auto_class_weights=Balanced`, `eval_metric=AUC`, seed 42. Bản kết quả canonical ghim `thread_count=1` để giảm sai lệch số học trên máy đã kiểm. `thread_count` là tham số kỹ thuật được thêm sau bản bàn giao ban đầu, không phải một siêu tham số chọn theo holdout.

### 4.2. Luồng dữ liệu và artifact

| Giai đoạn | Điểm chạy / mã chính | Đầu ra để đối chiếu |
|---|---|---|
| Sinh và kiểm tra dataset | [`build_dataset.py`](../scripts/build_dataset.py), [`verify_dataset.py`](../scripts/verify_dataset.py) | `data/processed/dataset_catboost.csv`, tradelist, labels |
| Chia fold và train bốn nhánh | [`train_models.py`](../scripts/train_models.py), [`train_catboost.py`](../src/citd_ml/training/train_catboost.py) | [`outputs/step4_thread1/catboost_training/`](../outputs/step4_thread1/catboost_training/) — OOF, metric từng fold |
| Backtest và sweep 20–80% | [`run_backtest.py`](../scripts/run_backtest.py) | [`outputs/step4_thread1/backtest/`](../outputs/step4_thread1/backtest/) — vũ trụ có điểm, danh sách giữ, chỉ số |
| Tái lập bước 4 | [`verify_pipeline.py`](../scripts/verify_pipeline.py) | [`reproducibility.json`](../outputs/step4_thread1/verification/reproducibility.json) |
| Holdout giai đoạn 1–5 | `scripts/holdout_stage*.py` | [`outputs/holdout/`](../outputs/holdout/) và [`BAO_CAO_KET_QUA_HOLDOUT.md`](BAO_CAO_KET_QUA_HOLDOUT.md) |

Xác suất đánh giá trên khúc 2–5 là **out-of-fold (OOF)**: mỗi lệnh được chấm bởi mô hình không train trên lệnh đó. Quy tắc giữ top-k là xếp `probability` giảm dần, phá hòa bằng `row_id` tăng dần, lấy `ceil(n × keep_pct / 100)`. Top 50% của 20.007 lệnh là **10.004**; top 50% của 5.028 lệnh là **2.514**. Equity cộng R tại `close_time`; các lệnh đóng cùng thời điểm được gộp trước khi cập nhật đường vốn. Cách làm và biểu đồ: [`implementation_decisions.md`](../outputs/holdout/stage4/implementation_decisions.md).

### 4.3. Quy trình holdout

Giai đoạn 1 tái sinh lịch sử và tách holdout; giai đoạn 2 train mô hình cuối hai lần và đối chiếu dự đoán; giai đoạn 3 chấm điểm/backtest vũ trụ cố định; giai đoạn 4 tổng hợp bảng, biểu đồ; giai đoạn 5 so kết quả của hai lần chạy độc lập. Các lệnh lịch sử và vị trí file đầu ra nằm trong [`README_VI.md`](../README_VI.md#quy-trình-holdout-niêm-phong). Đánh giá holdout đã đóng ngày **18/09/2026**. Các bảng sweep holdout dùng để kiểm tra độ nhạy, không để chọn lại tỷ lệ 50% sau khi đã xem kết quả.

## CHƯƠNG 5. KẾT QUẢ THỬ NGHIỆM

### 5.1. Phân loại trên khúc 2–5 và holdout

| Cách chia | ROC-AUC | F1 @ 0,5 |
|---|---:|---:|
| Random K-Fold | 0,8582 | 0,6494 |
| Grouped K-Fold | 0,7454 | 0,5077 |
| Walk-forward | 0,5875 | 0,3288 |
| Purged Walk-forward | 0,5895 | 0,3247 |
| **Holdout** | **0,6046** | **0,4022** |

Nguồn: [`table1_classification_metrics.csv`](../outputs/holdout/stage4/table1_classification_metrics.csv). Sai lệch AUC giữa Random và holdout xấp xỉ **+0,2536**; giữa Purged Walk-forward và holdout xấp xỉ **−0,0151**. Holdout là một phép đo riêng trên mô hình cuối, không phải một fold thứ năm của bảng OOF.

### 5.2. Backtest top 50%

| Phương án | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate |
|---|---:|---:|---:|---:|---:|
| Baseline, khúc 2–5 | 20.007 | +3.358,25 | 236,13 | 1,2940 | 31,71% |
| Random K-Fold | 10.004 | +6.937,53 | 73,76 | 2,5437 | 44,65% |
| Grouped K-Fold | 10.004 | +4.752,04 | 99,71 | 1,9500 | 39,26% |
| Walk-forward | 10.004 | +2.148,31 | 211,29 | 1,4117 | 34,76% |
| Purged Walk-forward | 10.004 | +2.119,48 | 142,36 | 1,4074 | 34,99% |
| Baseline holdout | 5.028 | +245,93 | 305,32 | 1,0992 | 35,28% |
| **Holdout top 50%** | **2.514** | **−36,55** | **263,25** | **0,9718** | **34,81%** |

Nguồn: [`table2_financial_metrics_top50.csv`](../outputs/holdout/stage4/table2_financial_metrics_top50.csv). **Không so trực tiếp tổng R của khúc 2–5 với holdout như thể cùng thời kỳ hoặc cùng số lệnh**; so từng phương án với baseline tương ứng và dùng holdout để kiểm tra kết luận về khả năng khái quát. Bảng sweep đầy đủ của bốn nhánh và holdout: [`table3_branch_sweep_20_80.csv`](../outputs/holdout/stage4/table3_branch_sweep_20_80.csv), [`holdout_sweep_20_80.csv`](../outputs/holdout/stage4/holdout_sweep_20_80.csv). Trên holdout, top 20–50% đều âm; từ top 60% trở lên net R dương, nhưng đó là **phân tích sau quan sát**, không phải tỷ lệ được chốt lại.

### 5.3. Biểu đồ và tái lập

Đường vốn của bốn nhánh và holdout nằm trong [`outputs/holdout/stage4/`](../outputs/holdout/stage4/) ở dạng PNG và HTML. Cả hai bắt đầu tại 0 R, chỉ nhảy khi lệnh đóng. Bản chạy lại canonical khớp từng byte **8 CSV** của artifact `thread_count=1` đã commit; đối chiếu run 1 và run 2 của holdout giai đoạn 3–4 **PASS**, mọi delta 0,0 và bốn biểu đồ giống hệt từng byte ([`stage5_repro_report.json`](../outputs/holdout/repro/stage5_repro_report.json)). Hai file model `.cbm` có thể khác SHA-256 do metadata tuần tự hóa dù mảng dự đoán trùng; không dùng hash `.cbm` một mình để kết luận sai lệch dự đoán.

## CHƯƠNG 6. BÀN LUẬN

**RQ1.** Cách chia thay đổi mạnh điểm đánh giá: AUC Random cao hơn Purged Walk-forward **0,2687**; top 50% Random báo +6.937,53 R trong khi Purged Walk-forward báo +2.119,48 R. Chênh lệch là bằng chứng rằng phép đánh giá nhạy với thiết kế split trên tập dữ liệu này.

**RQ2.** Điểm giảm rõ khi chuyển từ Random sang Grouped rồi Walk-forward; Walk-forward và Purged Walk-forward gần nhau (0,5875 và 0,5895). Purge/embargo chỉ loại 34 dòng ở bốn biên fold bước 4, nên khác biệt nhỏ là hợp lý trong cấu hình hiện tại. Không suy rộng rằng purge/embargo vô ích với dữ liệu hoặc horizon khác.

**RQ3.** Hai cách chia theo thời gian gần AUC holdout hơn Random/Grouped. Nhưng holdout top 50% **kém baseline 282,48 R**, cho thấy “ước lượng phân loại gần hơn” vẫn không bảo đảm bộ lọc sinh lợi khi triển khai. Tránh diễn giải điểm ROC-AUC 0,6046 thành bằng chứng thành công tài chính.

Thí nghiệm thay riêng `thread_count` cho thấy hai lần train `tc=1` trên máy kiểm giống dự đoán từng bit; train `tc=2` và mặc định thay đổi dự đoán và net R holdout top 50% (lần lượt **−12,26 R** và **+30,79 R**, so với **−36,55 R** của `tc=1`). Cùng model đã fit, thay số luồng khi **inference** không thay dự đoán. Đây là quan sát trên một máy, một build, một dataset và một seed; không quy mọi sai lệch liên máy cho số luồng. Xem [`outputs/thread_count_sensitivity/README.md`](../outputs/thread_count_sensitivity/README.md).

## CHƯƠNG 7. HẠN CHẾ VÀ HƯỚNG PHÁT TRIỂN

1. **Vũ trụ lệnh cố định:** backtest không mô phỏng lại tương tác động của chiến lược sau khi bỏ lệnh; một chiến lược live có thể sinh chuỗi lệnh khác.
2. **Một thị trường và một cấu hình:** BTCUSD M15, một họ lệnh bốn leg, một bộ feature và CatBoost. Kết luận định lượng không đại diện cho mọi tài sản hoặc thời kỳ.
3. **Lịch sử mở holdout:** ba phiên bản kết quả có thể truy xuất từ Git, nhưng sổ không chứng minh tổng số lần thực thi hoặc rằng mọi quyết định trước đó hoàn toàn độc lập với việc xem holdout. Các lần kiểm chứng và thí nghiệm số luồng đã truy cập holdout thêm. Xem [`holdout_run_history.md`](../outputs/verification/holdout_run_history.md).
4. **Tái lập liên máy:** bằng chứng byte-identical áp dụng cho các lượt kiểm trên máy tạo artifact; tương đương liên máy chưa được chứng minh. Các artifact bàn giao trước khi ghim `thread_count` được giữ như **khối tham chiếu**, khác một số giá trị so với canonical.

Hướng phát triển phù hợp là backtest chiến lược động sau lọc, kiểm định thêm giai đoạn/tài sản mới, dùng nhiều đường kiểm định thời gian và đánh giá độ bền trên nhiều môi trường. Những thử nghiệm đó cần một tập dữ liệu mới để tránh tiếp tục tối ưu theo holdout đã xem.

## CHƯƠNG 8. KẾT LUẬN VÀ HƯỚNG DẪN VIẾT BÁO CÁO NHÓM

Kết luận đã được dữ liệu hiện có hỗ trợ: cách chia dữ liệu là biến phương pháp luận ảnh hưởng lớn đến điểm kiểm định của bộ lọc CatBoost trên tập lệnh pyramid có nhãn chồng lấn; đánh giá theo thời gian gần AUC holdout hơn đánh giá ngẫu nhiên; bộ lọc top 50% không cải thiện kết quả tài chính trên holdout. Phạm vi kết luận là **độ tin cậy của đánh giá**, không phải một cam kết lợi nhuận.

Để thống nhất với báo cáo nhóm trưởng, team nên dùng tên chương 1–8 và cách gọi “Cách 1 / 1b / 2 / 3”, phân biệt rõ **khúc 2–5 OOF**, **holdout** và **khối tham chiếu bàn giao**. Mọi bảng kết quả chính lấy từ `outputs/holdout/stage4/` và cây `outputs/step4_thread1/`; ghi số theo quy ước báo cáo: AUC/F1/PF bốn chữ số, R/MaxDD hai chữ số, số lệnh nguyên. Khi diễn giải, dẫn cả nguồn artifact và giới hạn tương ứng. Với mô tả chi tiết phương pháp, đối chiếu [README tiếng Việt](../README_VI.md), [báo cáo bước 4](BAO_CAO_KET_QUA_STEP4.md), [báo cáo holdout](BAO_CAO_KET_QUA_HOLDOUT.md) và [báo cáo khoa học](../deliverables/Scientific_report_Nhom11.pdf) trước khi chốt câu chữ.
