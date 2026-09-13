# Quyết định triển khai

1. Trục thời gian của cả hai biểu đồ dùng `close_time`, vì R của một lệnh chỉ hoàn tất tại thời điểm đóng lệnh.
2. Các lệnh có cùng `close_time` được sắp xếp ổn định theo `ticket`, cộng R của chúng thành một điểm thời gian duy nhất; điểm đó là equity sau toàn bộ lệnh đóng cùng lúc.
3. Mỗi đường được thêm điểm R = 0 ngay trước `close_time` đầu tiên, để đường vốn bắt đầu từ 0 mà không thay đổi mốc dữ liệu giao dịch.
4. Biểu đồ 1 chỉ nhận các dòng `chunk` 2–5. Top 50% của từng nhánh dùng `ceil(20007 × 50%) = 10004`, xếp xác suất giảm dần rồi `row_id` tăng dần khi hòa điểm — đúng luật đã chốt ở bước trước.
5. Biểu đồ 2 bắt đầu lại từ R = 0 tại holdout, dùng trực tiếp vũ trụ baseline và danh sách Top 50% đã lưu từ Giai đoạn 3; không nối với equity của khúc 2–5.
6. Đường vốn dùng dạng bậc thang ngang-rồi-dọc (`hv`): giữ nguyên giữa hai `close_time` và chỉ nhảy tại lúc lệnh đóng; không chèn giao dịch hoặc điểm equity giả vào khoảng trống.
7. Purge và embargo đều trả về 0 dòng, nên train giữ nguyên 25,008 dòng; điều kiện lọc được chạy trước khi quyết định không loại dòng nào.
8. So khớp giá baseline dùng sai số tuyệt đối `5e-4`, theo validator bàn giao; giá vào, giá ra và R được kiểm dưới cùng ngưỡng này.
9. Chấm điểm holdout lấy danh sách 23 `FEATURES` trực tiếp từ `build_features.py`; không tự liệt kê cột.
10. CatBoost được ghim thêm `thread_count=1` — đây là tham số kỹ thuật (không thuộc danh sách hyperparameter mô hình đã chốt) để loại số luồng CPU như một nguồn sai lệch đã biết. Phép kiểm Stage 2 chỉ kết luận hai lượt trên cùng máy có prediction giống hệt; không dùng nó để khẳng định giống từng byte giữa mọi máy. Bốn dòng đầu Bảng 1 và năm dòng đầu Bảng 2 vẫn lấy nguyên từ bàn giao; chỉ dòng Holdout và bảng sweep được tính mới.
