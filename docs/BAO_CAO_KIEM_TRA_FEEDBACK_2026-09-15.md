# Báo cáo kiểm tra sau bổ sung trao đổi của nhóm

> **Cập nhật sau vòng sửa cuối:** Báo cáo chính gửi Nhi là [BAO_CAO_KET_QUA_HOLDOUT.md](BAO_CAO_KET_QUA_HOLDOUT.md). Đã sửa script và sinh lại báo cáo, README/README_VI: giới hạn sổ ở ba phiên bản lịch sử truy xuất được, phân biệt nghi ngờ 12/09 với thí nghiệm 15/09, bỏ suy luận kết quả xấu hơn chứng minh không cherry-pick, làm rõ replay bốn lệnh biên. Bảng đối chiếu sáu feedback nằm trong Phần 4 báo cáo chính. Nội dung audit bên dưới là snapshot trước sửa; các mục “cần sửa” đã được xử lý. Giới hạn lịch sử quyết định và tái lập liên máy vẫn còn; không đọc kết luận PASS cũ như bằng chứng loại trừ cherry-pick hoặc đã tuân thủ chỉ một lần thực thi holdout.


## 1. Kết luận

**Các thiếu sót về bảng, biểu đồ và bằng chứng tái lập đã được bổ sung. Tuy nhiên, chưa nên kết luận “đã xử lý trọn vẹn mọi feedback”: phần diễn đạt lịch sử mở holdout, lập luận không cherry-pick và mô tả bốn lệnh biên vẫn cần sửa.**

Kiểm tra tại HEAD `cdc16b6`, ngày 15/09/2026; working tree sạch trước kiểm tra. Đã đọc yêu cầu bàn giao, phản hồi Nhi và chuỗi tin nhắn Thịnh do người dùng cung cấp, đối chiếu code, Git và artifact. Tin nhắn là nguồn người dùng cung cấp, chưa phải log thực thi hay artifact của máy Windows.

Đây là báo cáo kiểm tra riêng để Thành nắm tình trạng. Không thay phần biện luận của Nhi, không sửa số liệu hoặc code, không train lại, không chọn lại tỷ lệ giữ lệnh. Lần kiểm tra này có chấm lại holdout bằng hai model đã lưu; cần phân biệt việc đó với một lần train mới.

## 2. Đối chiếu sáu feedback của Nhi

| Mục | Tình trạng tại HEAD hiện tại | Căn cứ và phần còn lại |
|---|---|---|
| 1. Nhất quán số bốn cách chia và holdout; lý do ghim thread | Đã bổ sung số và thí nghiệm; cần làm rõ trình tự quyết định | Bảng chính dùng `outputs/step4_thread1`, có khối tham chiếu bàn giao. Thí nghiệm thay đổi số luồng có kết quả lưu. Cần ghi rõ ngày 12/09 là nghi ngờ ban đầu; thí nghiệm xác nhận được bổ sung ngày 15/09. |
| 2. Số các lần holdout trước, minh bạch chạy lại | Đã truy xuất được số; kết luận còn quá mạnh | Git có ba phiên bản kết quả lịch sử. Điều này không chứng minh chỉ có đúng ba lần thực thi hoặc chưa từng có lần không commit. Không thể lấy việc kết quả hiện tại xấu hơn để chứng minh chắc chắn không cherry-pick. |
| 3. Bảng 20–80% của bốn cách chia | Đã có | Bảng 3 có 28 dòng, đặt trước bảng holdout 7 dòng; nguồn canonical được ghi rõ. |
| 4. Chèn hai hình | Đã có | Hai ảnh PNG được nhúng trong báo cáo, kèm hai HTML. Kiểm tra 42 liên kết nội bộ: không có đích bị thiếu. |
| 5. Tái lập đủ Stage 3–4; OS và Plotly | Đã có bằng chứng trên cùng máy | Chạy lại bộ đối chiếu Stage 5: PASS; dự đoán, chỉ số, lựa chọn top-k, bảng và biểu đồ hai run khớp. OS, Plotly, Matplotlib có trong metadata. Không suy rộng thành bảo đảm liên máy. |
| 6. Tách bảng, số lẻ, thời gian, bốn lệnh biên | Đã bổ sung cấu trúc và ghi chú; còn một câu sai phạm vi | Có khối bốn cách chia/holdout riêng, thời gian và phép cộng 25.008 + 4 + 5.028 = 30.040. Câu “không được dùng để train hay backtest” cần sửa vì replay chạy toàn lịch sử. |

Nguồn chính: [báo cáo kết quả](BAO_CAO_KET_QUA_HOLDOUT.md), [yêu cầu holdout](BAN_GIAO_task_holdout.md), [script sinh báo cáo](../scripts/holdout_stage4_report.py).

## 3. Thread count: bằng chứng nói được đến đâu?

### 3.1. Kết quả đã lưu của thí nghiệm có kiểm soát

Thí nghiệm giữ cùng dữ liệu, feature, seed và cấu hình còn lại, thay số luồng khi train. Artifact hiện tại ghi thời điểm `2026-09-15T13:45:45+00:00` (20:45:45 giờ Việt Nam), CPU count 24.

| Số luồng khi train | Top 50% net R | Số lệnh top 50% trùng với tc=1 | Holdout max abs(Δp) so với tc=1 |
|---|---:|---:|---:|
| 1 | −36,55 | 2.514/2.514 | 0,000000 |
| 2 | −12,26 | 2.252/2.514 | 0,346856 |
| −1 | +30,79 | 2.254/2.514 | 0,335194 |

Artifact còn ghi: lặp train với cùng số luồng cho dự đoán giống hệt trong các phép thử; chấm cùng model với prediction thread count 1/2/−1 không đổi dự đoán.

**Kết luận được hỗ trợ:** thay số luồng khi train làm thay đổi model output trong môi trường thí nghiệm này. Ảnh hưởng đến tập lệnh được chọn và net R là có thật; không thể gọi toàn bộ hiện tượng này là sai số 1e-15.

**Chưa được chứng minh:** mọi khác biệt Windows–Linux đều do số luồng; cơ chế bên trong chính xác là thứ tự cộng floating-point; ghim một luồng sẽ luôn tạo output giống từng byte trên mọi máy. Thí nghiệm chỉ bao phủ một máy, một build, một dataset, một seed. Lần audit này đọc và kiểm tra cấu trúc phép thử, không train lại thí nghiệm.

Nguồn: [script thí nghiệm](../scripts/thread_count_sensitivity.py), [kết quả JSON](../outputs/thread_count_sensitivity/thread_count_sensitivity.json).

### 3.2. Hai hiện tượng không được nhập làm một

- **Tin nhắn 12/09:** Thịnh báo hai run trên Windows giống nhau, prediction train giữa hai máy lệch trong 1e-15. Không có artifact Windows được commit để kiểm lại, chưa có kết quả đầy đủ trên holdout.
- **Thí nghiệm 15/09:** đổi số luồng train trên cùng máy tạo chênh lệch xác suất tối đa khoảng 0,35 và đổi hơn 250 lệnh trong tập top 50%.

Vì vậy, “sai số bé tí” chỉ phù hợp với phép so cụ thể Thịnh đã báo. Câu “pull code mới về train từ đầu chắc chắn khác report” cũng quá mạnh: hiện chưa có đối chứng liên máy đầy đủ cho cấu hình hiện tại. File CSV bị Git đánh dấu thay đổi tự nó chưa cho biết thay đổi là số học, định dạng, metadata hay nội dung khác.

## 4. Lịch sử holdout và thời điểm quyết định

Các thời điểm dưới đây là **thời điểm commit**, không phải log xác nhận chính xác thời điểm chạy.

| Mốc Git, giờ Việt Nam | Dữ kiện truy xuất được | Top 50% net R |
|---|---|---:|
| `dc25cd3`, 12/09 14:52:41 | Phiên bản gốc; Windows suy từ đường dẫn; chưa ghi số luồng | −23,83 |
| `5f46e41`, 12/09 15:06:19 | Phiên bản Linux suy từ đường dẫn; chưa ghi số luồng | +30,79 |
| `dbf8dea`, 12/09 15:20:38 | Commit thêm `thread_count=1` | — |
| `7748828`, 12/09 15:20:39 | Lưu kết quả với `thread_count=1` | −36,55 |
| `b278617`, 15/09 16:49:14 | Commit thí nghiệm có kiểm soát sau feedback | — |
| `de36762`, 15/09 21:17:06 | Sửa trình bày, cấu hình ghi file và làm mới bằng chứng | Không đổi số headline |

Tin nhắn Thành lúc 12/09 15:45 mô tả việc ghim luồng và nhờ kiểm tra xảy ra sau commit lưu kết quả tc=1. Nội dung lúc 22:00 nói đang tìm nguyên nhân và nghe giải thích từ AI phù hợp với việc **ban đầu là giả thuyết**. Đây không phải bằng chứng rằng thí nghiệm kiểm soát đã tồn tại trước khi đổi cấu hình.

Sổ Git khắc phục thiếu số cũ: có thể kiểm lại các phiên bản đã lưu, cùng baseline +245,93 R. Nhưng script lịch sử dùng danh sách ba commit định sẵn (`RUN_SPECS`), không phải bộ đếm mọi lần holdout từng được đọc/chấm. Repo còn có run2 và thí nghiệm chấm holdout sau đó; audit này cũng chấm lại bằng model đã lưu.

**Cách diễn đạt đúng:** “Ba phiên bản kết quả holdout lịch sử truy xuất được từ Git.” Sau đó liệt kê riêng các lần kiểm chứng kỹ thuật và thí nghiệm bổ sung.

Việc giữ kết quả −36,55 R thay vì +30,79 R là dữ kiện hỗ trợ minh bạch; nó không xác nhận toàn bộ lịch sử ra quyết định. Tỷ lệ top 50% đã được yêu cầu bàn giao chốt trước; không đổi sang 60–80% vì thấy sweep holdout có số thuận lợi hơn. Việc công khai các lần xem lại không khôi phục trạng thái holdout chưa từng được xem.

Nguồn: [sổ lịch sử](../outputs/verification/holdout_run_history.json), [script truy xuất](../scripts/holdout_run_history.py), [task chốt top 50%](BAN_GIAO_task_holdout.md).

## 5. Những chỗ còn cần chỉnh trước khi gửi lại

| Mức | Vị trí | Vấn đề | Câu hoặc hành động đề xuất |
|---|---|---|---|
| Cần sửa | `docs/BAO_CAO_KET_QUA_HOLDOUT.md:9`; `scripts/holdout_stage4_report.py:873`; README phần lịch sử | “toàn bộ ba lần mở holdout” suy rộng hơn Git | Đổi thành “Ba phiên bản kết quả holdout lịch sử truy xuất được từ Git”; ghi rõ sổ không xác nhận tổng số lần thực thi. |
| Cần sửa | `README.md:441`; `README_VI.md:430`; `docs/FEEDBACK_COMPLIANCE_AUDIT.md` mục 5 | Dùng kết quả xấu hơn làm lập luận khẳng định không cherry-pick | “Kết quả được giữ lại có net R thấp hơn hai phiên bản lịch sử; đây là dữ kiện hỗ trợ, không thay thế lịch sử quyết định và không tự chứng minh không cherry-pick.” |
| Cần sửa | `scripts/holdout_stage4_report.py:951`; báo cáo kết quả, Phần 3 mục 11 | “không được dùng để train hay backtest” | “Bốn lệnh biên không thuộc tập train và không được đưa vào chỉ số đánh giá holdout. Replay toàn lịch sử vẫn xử lý giai đoạn này để duy trì trạng thái chiến lược.” |
| Cần bổ sung | Báo cáo kết quả, lý do chạy lại và mục 1.6 | Chưa nối rõ hai mốc nghi ngờ và thí nghiệm | Ghi: “Ngày 12/09 ghim một luồng theo giả thuyết kỹ thuật; ngày 15/09 bổ sung thí nghiệm đối chứng xác nhận ảnh hưởng trong môi trường hiện tại.” |
| Đã sửa | `scripts/holdout_stage4_report.py:597,688` và báo cáo sinh ra | Hai lỗi dấu `|` Thịnh nêu | HEAD hiện tại đã escape cả hai. Quét số cột bảng Markdown: không có dòng lệch cột. Không cần báo lại đây là lỗi còn mở. |

Sửa nội dung ở script sinh report rồi sinh lại các bản liên quan để tránh tái phát. Audit cũ nên được bổ sung ghi chú cập nhật, vì kết luận “cả sáu mục PASS” chưa phản ánh đầy đủ giới hạn bằng chứng nêu trên.

## 6. Kiểm tra thực sự chạy trong phiên này

| Phép kiểm | Kết quả |
|---|---|
| `.venv/bin/python -m pytest -q` | 13 passed, 0,06 giây |
| Stage 5 đối chiếu artifact run1/run2, xuất vào `/tmp/citd-context-audit-stage5.json` | PASS; prediction và mọi metric delta 0; top-k trùng; bảng, biểu đồ và manifest khớp |
| Load hai `.cbm` hiện có và chấm lại 5.028 dòng holdout | Hai model có max abs(Δp) = 0 |
| So prediction model với CSV lưu | Max abs diff ≈ 4,9993e-13; phù hợp CSV ghi `%.12g` |
| Tính AUC/F1 trực tiếp từ prediction model | 0,6045544538928682 / 0,40220723482526055; khớp báo cáo 0,6046 / 0,4022 |
| Tự sắp xếp probability giảm, row_id tăng, lấy 2.514 lệnh | Bộ khóa top 50% khớp 2.514/2.514 |
| Tính tổng R từ CSV | Baseline +245,92968265920814; top 50% −36,55275499584241; khớp mức làm tròn công bố |
| Đọc `stage3_report.json` bằng `git show` ở ba commit lịch sử | AUC/F1 khớp sổ lịch sử |
| Liên kết và bảng Markdown báo cáo | 42 liên kết có đích; không có dòng bảng sai số cột |
| Quy mô bảng sweep Stage 4 | Bốn cách chia: 28 dòng; holdout: 7 dòng |

**Giới hạn phiên kiểm tra:** Stage 5 là bộ đối chiếu file có sẵn, không tự train hoặc replay lại. Phiên này không chạy lại toàn pipeline từ raw M1, không tái thực hiện training bốn split, không thử trên Windows. Bằng chứng chạy lại toàn pipeline trước đó nằm trong [audit tái lập](INDEPENDENT_REPRO_AUDIT.md) và manifest đã commit; phải phân biệt với các kiểm tra mới ở bảng trên. 13 unit test cũng không tự chứng minh mọi tính chất thống kê hoặc kiến trúc của đồ án.

## 7. Về đề xuất “đóng gói model cho Nhi dùng”

Repo **đã có** model `outputs/holdout/stage2/catboost_final_holdout_run1.cbm` và run2. Chấm lại hai model trong phiên này đã khớp. Có thể bàn giao model run1 cùng dữ liệu đầu vào, thứ tự 23 feature, phiên bản môi trường, checksum và output kỳ vọng để Nhi kiểm tra kết quả chấm điểm mà không phải train lại.

Tuy nhiên, dùng model đã lưu và tái huấn luyện là hai phép kiểm khác nhau. Bàn giao model giúp cố định model được dùng để chấm; nó không giải quyết thay lịch sử mở holdout và không bảo đảm mọi môi trường sẽ tạo byte output giống nhau. Hai file model run1/run2 có hash khác cũng không tự chứng minh dự đoán khác: phép chấm lại ở đây cho dự đoán giống hệt.

## 8. Nội dung tóm tắt có thể trao đổi lại với nhóm

> Anh đã đối chiếu code tại `cdc16b6`. Số bốn cách chia đã dùng bản ghim một luồng; đủ bảng sweep, hai ảnh và bằng chứng tái lập Stage 3–4 trên cùng máy. Thí nghiệm bổ sung ngày 15/09 cho thấy đổi số luồng lúc train làm đổi kết quả trong môi trường này, nhưng không chứng minh mọi lệch Windows–Linux đều do luồng. Ngày 12/09 việc ghim luồng xuất phát từ nghi ngờ kỹ thuật, thí nghiệm xác nhận có sau đó. Git truy xuất được ba phiên bản kết quả holdout, không xác nhận tổng số lần chạy. Hai lỗi dấu gạch đứng đã sửa ở code hiện tại. Còn cần chỉnh câu về lịch sử holdout, lập luận không cherry-pick và bốn lệnh biên. Giữ nguyên top 50% và số hiện tại: AUC 0,6046, F1 0,4022, net R −36,55; không chọn lại theo sweep holdout.

Đoạn trên là bản nháp để người dùng tham khảo, chưa được gửi cho bất kỳ thành viên nào.


## 9. Kiểm tra sau sửa và chuẩn bị bàn giao

Vòng tiếp theo đã sửa báo cáo chính cho Nhi, hai README, script sinh report và sổ lịch sử; sinh lại cả bản run1 và run2. Đây là cập nhật sau snapshot ở các mục trên.

- `pytest -q`: 13 passed.
- `verify_pipeline` trên cây canonical: **PASS**, chạy train hai lượt và backtest hai lượt; 6 CSV training và 8 CSV backtest khớp nhau và khớp bản đã lưu từng byte. Manifest được cập nhật với hash script hiện tại.
- Stage 5: **PASS**, mọi delta số học bằng 0, tập top-k giống nhau, bảng và bốn biểu đồ byte-identical giữa run1/run2.
- Mỗi báo cáo: 42 liên kết hợp lệ, 71 dòng số liệu cũ giữ nguyên, bảng không sai số cột; kích thước file trong bảng inventory khớp file thực tế.
- Dataset: 25.008 train + 4 biên + 5.028 holdout = 30.040; train và holdout không giao khóa. Khoảng holdout khớp yêu cầu.
- Hai PNG đã kiểm tra trực quan; nhãn, chú giải và đường vốn hiển thị rõ. Không đổi file model, prediction, bảng số hoặc artifact bàn giao.

Báo cáo cho Nhi giữ đúng năm phần yêu cầu và thêm bảng đối chiếu sáu feedback trong Phần 4. Những phần còn giới hạn là lịch sử holdout đã được truy cập nhiều lần và chưa có đối chứng đầy đủ liên máy; không đánh dấu hai giới hạn này là đã được chứng minh chỉ bằng việc sửa tài liệu.
