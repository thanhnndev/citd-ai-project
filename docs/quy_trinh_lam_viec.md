Giai đoạn 0 — Setup & đọc hiểu (bắt buộc trước khi code)
AI đọc kỹ BAN_GIAO_task_holdout.md, README.md, và BAN_GIAO_task_train_catboost.md (bước trước). Cài đúng version: catboost==1.2.10, scikit-learn>=1.6, Python theo README (3.12). KHÔNG được sắp xếp lại thư mục chay/ vì import bằng đường dẫn tương đối. Yêu cầu AI liệt kê lại bằng lời của nó các con số cố định (25,008 dòng cũ, mốc holdout, bar 199,968, hyperparameter) để xác nhận đã hiểu đúng trước khi viết dòng code nào.
2
Giai đoạn 1 — Sinh & kiểm dataset holdout
Viết file MỚI (ví dụ build_features_holdout_check.py hoặc dùng --holdout có sẵn) để sinh lại toàn bộ dữ liệu, kiểm tra: (1) phần trước mốc ra đúng 25,012 dòng, (2) toàn bộ 25,008 khóa (origin_bar, entry_bar, leg) của dataset_catboost.csv cũ có mặt trong bản mới, (3) 23 cột FEATURES trùng từng ô. Chỉ báo 'xong' khi cả 3 kiểm tra này pass và có log/print rõ ràng con số thật. Nếu lệch — dừng lại, báo cụ thể lệch ở đâu, không tự sửa để 'cho khớp'.
3
Giai đoạn 2 — Train model cuối + Purge/Embargo
Viết file mới train model cuối cùng trên dataset_catboost.csv, hyperparameter y nguyên (không đổi 1 số nào). Áp purge + embargo, in ra số dòng bị cắt (kỳ vọng 0 — nếu khác 0 phải dừng và báo, không tự động 'fix' bằng cách đổi ngưỡng). Cố định seed 42, chạy 2 lần đối chiếu ra y hệt nhau (hash hoặc so từng dòng).
4
Giai đoạn 3 — Chấm điểm & backtest holdout
Viết file mới (không sửa backtest_pyramid_local.py cũ) để: chạy backtest full từ 2018 nhưng chỉ tính chỉ số từ mốc holdout, giữ vũ trụ lệnh cố định, baseline + sweep 20-80%. Đối chiếu baseline holdout với tradelist_pyramid_local.csv để đảm bảo logic chiến lược không bị lệch. Log rõ số lệnh, net profit, PF, win rate mỗi mức %.
5
Giai đoạn 4 — Bảng tổng hợp + biểu đồ
Điền dòng Holdout vào Bảng 1 và Bảng 2 theo đúng format có sẵn trong BAN_GIAO_task_holdout.md (copy nguyên khung bảng, chỉ điền số). Vẽ 2 biểu đồ đường vốn theo đúng mô tả mục 5 (trục ngang close_time, trục dọc R cộng dồn). Xuất bản HTML cho biểu đồ.
6
Giai đoạn 5 — Viết báo cáo kết quả (5 phần)
Viết đúng 5 phần theo mục 6: Số liệu / Biểu đồ / Quyết định triển khai (liệt kê những chỗ phải tự quyết và lý do) / Kiểm chứng (liệt kê từng phép kiểm + kết quả, kể cả version Python/thư viện) / Bảng file sinh ra. KHÔNG viết phần diễn giải hay kết luận — nhắc AI rõ ràng vì dễ bị lố sang phần phân tích.
7
Giai đoạn 6 — Tự rà soát trước khi nộp (self-QA, tránh gửi qua gửi lại)
Trước khi báo hoàn thành, bắt AI tự chạy lại checklist: (1) không sửa dataset_catboost.csv/outputs/ gốc, (2) kết quả mới nằm thư mục riêng, (3) chạy lại 2 lần ra giống hệt, (4) mọi con số trong báo cáo đúng với log thực tế đã chạy — không phải số đoán/ước lượng, (5) đối chiếu lại với 3 chỗ code hay vướng đã liệt kê. Chỉ khi checklist này pass hết mới gửi cho bạn.