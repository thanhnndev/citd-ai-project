# Quyết định triển khai

1. Trục thời gian của cả hai biểu đồ dùng `close_time`, vì R của một lệnh chỉ hoàn tất tại thời điểm đóng lệnh.
2. Các lệnh có cùng `close_time` được sắp xếp ổn định theo `ticket`, cộng R của chúng thành một điểm thời gian duy nhất; điểm đó là equity sau toàn bộ lệnh đóng cùng lúc.
3. Mỗi đường được thêm điểm R = 0 ngay trước `close_time` đầu tiên, để đường vốn bắt đầu từ 0 mà không thay đổi mốc dữ liệu giao dịch.
4. Biểu đồ 1 chỉ nhận các dòng `chunk` 2–5 của cây canonical `outputs/step4_thread1`; top 50% của từng nhánh dùng `ceil(n × 50%)`, xếp xác suất giảm dần rồi `row_id` tăng dần khi hòa điểm — đúng luật đã chốt ở bước trước.
5. Biểu đồ 2 bắt đầu lại từ R = 0 tại holdout, dùng trực tiếp vũ trụ baseline và danh sách Top 50% đã lưu từ Giai đoạn 3; không nối với equity của khúc 2–5.
6. Đường vốn dùng dạng bậc thang ngang-rồi-dọc (`hv`) cho HTML; PNG xuất bằng matplotlib từ đúng dữ liệu đó với `drawstyle="steps-post"`, `dpi=110`, figsize cố định và không có timestamp trong phần chữ nên byte ổn định giữa các lần chạy.
7. Bảng 1–3 dùng khối canonical `thread_count=1`; số bàn giao không ghim `thread_count` được giữ nguyên trong khối tham chiếu dán nhãn và không trộn vào khối chính.
8. Bảng 3 lấy trực tiếp từ `outputs/step4_thread1/backtest/backtest_retention_sweep.csv`; baseline chỉ nêu một lần phía trên bảng.
9. Bốn lệnh biên giữa dataset tái sinh và dataset đóng băng được liệt kê trong mục 1.7; chúng không thuộc train và không thuộc holdout.
10. Purge và embargo đều trả về 0 dòng, nên train giữ nguyên 25,008 dòng; điều kiện lọc được chạy trước khi quyết định không loại dòng nào.
11. So khớp giá baseline dùng sai số tuyệt đối `5e-4`, theo validator bàn giao; giá vào, giá ra và R được kiểm dưới cùng ngưỡng này.
12. Chấm điểm holdout lấy danh sách 23 `FEATURES` trực tiếp từ `build_features.py`; không tự liệt kê cột.
13. Hai thư mục bàn giao `outputs/catboost_training` và `outputs/backtest` chỉ được đọc, không bị ghi đè; mọi bảng canonical, sweep và biểu đồ đều lấy từ cây `outputs/step4_thread1`.
