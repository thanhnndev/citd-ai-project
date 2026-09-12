# TASK — Chạy holdout, tổng hợp, đối chiếu

4 cách chia cho 4 kết quả khác nhau. Giờ mở phần dữ liệu niêm phong từ đầu đồ án ra, xem con số nào sát thật.

Mốc holdout: **`2025-02-08 15:30:00`**. Từ mốc này trở đi chưa ai đụng vào.

**Chỉ chạy đúng một lần.** Không thử nhiều kiểu rồi lấy cái đẹp nhất. Feature, hyperparameter, tỷ lệ giữ lệnh đã chốt ở bước trước, không đổi.

`catboost==1.2.10`, `scikit-learn>=1.6`.

---

## Đồ án đang đo cái gì — nói ngắn để nắm

Model CatBoost ở đây làm **bộ lọc tín hiệu**. Chiến lược định vào lệnh nào thì model chấm điểm lệnh đó, điểm thấp thì bỏ qua. Model không quyết định mua bán, chỉ quyết định có nghe theo chiến lược hay không.

Chỗ khó nằm ở nhãn. Nhãn thắng/thua của một lệnh được xác định bằng cách **nhìn tới tương lai** — chờ giá chạm chốt lời, cắt lỗ, hoặc hết 50 bar. Nên hai lệnh vào gần nhau có nhãn do cùng một đoạn thị trường quyết định, tức là nhãn của chúng **chồng lấn nhau**. Chiến lược ở đây còn cố ý nhồi thêm 3 lệnh ngay sau mỗi tín hiệu, nên 4 lệnh cùng một gia đình (`origin_bar`) gần như chung số phận.

Hệ quả: nếu chia dữ liệu ngẫu nhiên, 4 lệnh cùng gia đình bị tách hai phía train/test. Model học 2 lệnh rồi đi đoán 2 lệnh gần như y hệt → coi như đã thấy trước đáp án → điểm đánh giá bị thổi phồng.

Bốn cách chia là bốn mức chặt khác nhau:

| | |
|---|---|
| **Cách 1** — Random K-Fold | Trộn ngẫu nhiên từng dòng, chia 5 rổ. Lỏng nhất, gia đình bị xé đôi thoải mái |
| **Cách 1b** — Grouped K-Fold | Vẫn trộn ngẫu nhiên, nhưng 4 lệnh cùng gia đình luôn nằm chung một rổ |
| **Cách 2** — Walk-forward | Cắt theo thời gian, học quá khứ đoán tương lai. Không còn chuyện học tương lai đoán quá khứ |
| **Cách 3** — WF + Purge/Embargo | Như Cách 2, dọn thêm các lệnh nằm sát ranh giới train/test |

Đồ án đo xem bốn cách cho kết quả chênh nhau bao nhiêu. **Holdout ở bước này là giai đoạn dữ liệu chưa ai đụng, dùng để đối chiếu.**

Chỉ cần nắm tới đây. Phần diễn giải sâu hơn Nhi lo.

---

## File giao

### `tai_lieu/`

| File | |
|---|---|
| `BAN_GIAO_task_train_catboost.md` | Task bước ngay trước. Tra lại cách chia 4 nhánh, hyperparameter, luật lọc top-k |

### `chay/` — thư mục làm việc, chạy được luôn

| File | |
|---|---|
| `BTCUSD_m1_2018_to_now.csv` | BTC M1, 2018-01 → 2026-08. Dữ liệu gốc duy nhất |
| `pyramid_strategy.py` | Logic chiến lược |
| `tradelist_pyramid_local.csv` | Backtest chưa lọc, có cả phần holdout. Dùng làm mốc đối chiếu |
| `label_triple_barrier_mới.py` | Bộ gán nhãn → `triple_barrier_labels_mới.csv` |
| `build_features.py` | Sinh dataset. Có hằng `FEATURES` (23) và `META` (7) |
| `verify_dataset.py` | Kiểm tra dataset sau khi sinh |
| `dataset_catboost.csv` | 25,008 dòng, phần 80% đầu |
| `pre_train.py`, `split_data.py`, `train_catboost.py`, `backtest_pyramid_local.py`, `verify_pipeline.py` | Code bước trước, chạy được, dùng lại |
| `outputs/` | Kết quả 4 nhánh: 4 bảng OOF, metric từng fold, backtest, sweep, bằng chứng chạy lặp |

Code trong `chay/` import lẫn nhau bằng đường dẫn tương đối. **Để nguyên thư mục phẳng như vậy, đừng xếp lại vào thư mục con** thì mới chạy được.

### Ba chỗ code hiện tại sẽ vướng

Code bước trước viết cho đúng 25,008 dòng và cho phần trước mốc holdout, nên chạy thẳng cho holdout là gãy. Đây là ba chỗ phải sửa, biết trước cho đỡ mất thời gian dò:

1. `pre_train.py` — hằng `EXPECTED_ROWS = 25_008` và `EXPECTED_EDGES` đóng cứng. Dataset holdout ~5,028 dòng sẽ bị nó chặn.
2. `backtest_pyramid_local.py` — hàm `load()` cắt dữ liệu tại `dt < HOLDOUT_START`, nên không backtest được giai đoạn holdout.
3. `verify_pipeline.py` — hằng `SOURCE_FILES` liệt kê tên file cố định để băm SHA-256. Thêm script mới thì nhớ thêm vào danh sách.

Đừng sửa đè lên file cũ. Viết file mới cho bước holdout, để nguyên code bước trước còn chạy lại được.

---

## 1. Sinh dataset holdout

`build_features.py` có tham số `--holdout`. Truyền mốc xa (`2100-01-01`) để nó chạy hết dữ liệu, rồi cắt đôi theo `entry_time`.

**Cẩn thận chỗ này:** phần trước mốc sinh lại sẽ ra **25,012** dòng, không phải 25,008. Lý do: 4 lệnh vào sát mốc, trước đây cắt dữ liệu nên nhãn chưa xong, bị loại. Giờ đủ dữ liệu thì nhãn xong, chúng chui vào.

- Phần trước mốc: **xài nguyên `dataset_catboost.csv` cũ**. Đừng thay bằng bản mới sinh. Ranh giới 5 khúc và kết quả bước trước dựa vào đúng 25,008 dòng đó.
- Kiểm tra: 25,008 khóa `(origin_bar, entry_bar, leg)` của file cũ phải nằm đủ trong bản sinh lại, 23 cột `FEATURES` trùng từng ô. Lệch thì dừng, báo.
- **Dataset holdout** = dòng có `entry_time >= 2025-02-08 15:30:00`. Khoảng 5,028 dòng (2025-02-09 → 2026-08-21). Báo số thật.

---

## 2. Train model cuối

Một model duy nhất, train trên `dataset_catboost.csv`. Hyperparameter y nguyên bước trước:

`iterations` 1000 · `learning_rate` 0.05 · `depth` 6 · `l2_leaf_reg` 3.0 · `auto_class_weights` Balanced · `eval_metric` AUC · `random_seed` 42. Không early stopping, không `cat_features`.

Trước khi train, dọn ranh giới holdout giống Cách 3 đã làm:

- **Purge** — bỏ dòng có `label_end_time >= 2025-02-08 15:30:00`
- **Embargo** — bỏ dòng có `entry_bar >= 199968 - 50` (199968 là bar M15 của mốc holdout)

**Hai bước này dự kiến cắt 0 dòng.** Dataset vốn đã sạch: `entry_bar` lớn nhất 199873, cách mốc 95 bar; `label_end_time` lớn nhất `2025-02-06 16:44`. Vẫn phải chạy và ghi số vào báo cáo. Ra khác 0 là có gì sai ở bước trước, dừng lại báo.

Holdout giữ nguyên 100%, không bỏ dòng nào.

---

## 3. Chấm điểm và backtest holdout

Model cuối chấm xác suất cho từng lệnh holdout. Model chưa từng thấy một bar nào của giai đoạn này.

Backtest: chạy `backtest_pyramid_local.py` từ 2018 cho chiến lược khởi động đủ, nhưng **chỉ tính chỉ số từ mốc holdout trở đi**. Giống hệt cách bước trước bỏ khúc 1.

Giữ nguyên luật lọc của bước trước: **vũ trụ lệnh cố định**. Chiến lược sinh tín hiệu y hệt baseline. Lệnh không lọt top thì bỏ ra khỏi phần tính chỉ số, **không** để việc bỏ lệnh làm đổi chuỗi tín hiệu về sau. Có vậy holdout mới so được với 4 nhánh.

Chạy **baseline** (không lọc) và **top 50%**. Rồi làm thêm 20/30/40/50/60/70/80%, xài lại bảng điểm, không train lại.

---

## 4. Bảng tổng hợp

**Bảng 1 — Chỉ số phân loại, đã bỏ khúc 1 ở cả 4 nhánh**

| | ROC-AUC | F1 @0.5 |
|---|---|---|
| Cách 1 — Random K-Fold | 0.8595 | 0.6525 |
| Cách 1b — Grouped K-Fold | 0.7475 | 0.5105 |
| Cách 2 — Walk-forward | 0.5867 | 0.3267 |
| Cách 3 — WF + Purge/Embargo | 0.5948 | 0.3304 |
| **Holdout** | | |

4 dòng đầu có sẵn, chỉ điền dòng holdout.

Đề cương chốt là bỏ khúc 1 ở mọi nhánh, để 4 cách được đo trên cùng khoảng thời gian. Cách 2/3 vốn không có khúc 1 nên số không đổi. Cách 1/1b phải bỏ khúc 1 ra rồi tính lại — làm từ 4 file `oof_*.csv`, không train lại. Bảng `metrics_summary.csv` của bước trước là bản **chưa** bỏ khúc 1 (0.8630 / 0.7381), đừng bê thẳng vào.

**Bảng 2 — Chỉ số tài chính, giữ top 50%**

| | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---|---|---|---|---|
| Baseline (khúc 2–5) | 20,007 | +3,358.3 | 236.1 | 1.294 | 31.7 |
| Cách 1 | 10,004 | +6,998.3 | 75.7 | 2.563 | 44.9 |
| Cách 1b | 10,004 | +4,724.3 | 99.5 | 1.942 | 39.2 |
| Cách 2 | 10,004 | +2,207.3 | 205.3 | 1.423 | 34.9 |
| Cách 3 | 10,004 | +2,253.7 | 154.1 | 1.435 | 35.2 |
| **Baseline holdout** | | | | | |
| **Holdout, top 50%** | | | | | |

Thêm bảng 20–80% cho holdout, đặt cạnh bảng cũ.

Để riêng hai khối: 4 nhánh trên 80% đầu, và holdout. **Không thêm cột "sai lệch so với holdout".**

---

## 5. Biểu đồ đường vốn

Hai biểu đồ:

1. Baseline + 4 nhánh ở mức 50%, vẽ chồng lên nhau, trên khúc 2–5. Dựng từ `backtest_scored_universe.csv` (có sẵn `R`, `close_time`, 4 cột xác suất).
2. Baseline + holdout ở mức 50%, vẽ riêng.

Trục ngang `close_time`, trục dọc R cộng dồn từ 0.

---

## 6. Báo cáo kết quả

Bước này nộp **báo cáo kết quả** — gom số lại cho gọn gàng để Nhi biện luận ở báo cáo chính. 5 phần:

**1. Số liệu** — Bảng 1, Bảng 2, bảng 20–80% ở mục 4. Điền số, không bình luận.

**2. Biểu đồ** — 2 biểu đồ ở mục 5, kèm bản HTML.

**3. Quyết định triển khai** — chỗ nào trong task này để mở, các bạn chọn thế nào, vì sao chọn vậy. Ví dụ: làm tròn số lệnh top-k lên hay xuống, dựng equity curve theo mốc thời gian nào.

**4. Kiểm chứng** — liệt kê từng phép kiểm và kết quả:
- Phần trước mốc của bản sinh lại có chứa đủ 25,008 khóa cũ không, 23 cột `FEATURES` có trùng từng ô không
- Purge và embargo tại ranh giới holdout cắt bao nhiêu dòng
- Backtest chạy lại có khớp `tradelist_pyramid_local.csv` không
- Chạy 2 lần có ra kết quả giống nhau từng byte không
- Phiên bản Python và thư viện

**5. Bảng file sinh ra** — mỗi file chứa gì, bao nhiêu dòng.

**Không viết phần diễn giải, biện luận, kết luận.** Phần đó Nhi làm ở báo cáo chính. Việc ở đây là ra số cho đúng, vẽ biểu đồ cho đúng, ghi rõ đã làm những gì và đã kiểm những gì. Không cần đoán vì sao holdout cao hay thấp, cũng không cần so sánh xem cách chia nào tốt hơn.

**Báo cáo technical** — file nào làm gì, hàm nào làm gì — là **task riêng ở cuối đồ án**, không phải bây giờ. Bước này cứ code cho chạy và trả số là đủ.

---

## Lưu ý

1. Không đổi hyperparameter, không thêm bớt feature, không đưa `META` vào `X`.
2. Không bỏ dòng nào khỏi holdout.
3. Không sửa `dataset_catboost.csv` và `outputs/`. Kết quả mới ghi ra thư mục riêng.
4. Cố định seed. Chạy hai lần phải ra y hệt nhau.
5. Holdout mở một lần. Phải chạy lại vì lỗi kỹ thuật thì ghi rõ trong báo cáo.
6. Có gì không rõ thì hỏi trước khi tự quyết, đừng đoán.
