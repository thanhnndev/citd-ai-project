# AUDIT ĐỐI CHIẾU FEEDBACK REVIEW — CITD ML (holdout)

- **Ngày audit:** 2026-09-15
- **Phạm vi:** read-only. Không sửa file nào; tài liệu này là file duy nhất được tạo.
  Không chạy `git add/commit/push`.
- **HEAD khi audit:** `93cdf503f5e2ddc1aeceec41389be569d1987d3b` (`93cdf50`).
  Working tree có thay đổi chưa commit của vòng "best-practice" (xem G5).
- **Môi trường:** `.venv/bin/python` — Python 3.12.14 · catboost 1.2.10 · scikit-learn 1.9.0 ·
  pandas 3.0.5 · numpy 2.5.2 · plotly 7.0.0 · matplotlib 3.11.2 ·
  OS `Linux-7.2.5-1-cachyos-x86_64-with-glibc2.44`.
- **Test suite:** `.venv/bin/python -m pytest -q` → **13 passed** (0 failed, 0 skipped).
- **Kết quả tổng quát:** cả 6 mục feedback đều đạt về nội dung; **0 mismatch số học**;
  còn 7 gap nhỏ về trình bày/biên soạn (G1–G7, mục 7) không làm sai số hay kết luận.

---

## 1. Bảng phán quyết theo từng mục feedback

| # | Nội dung feedback (tóm tắt) | Phán quyết | Bằng chứng chính (file:line / lệnh) | Gap còn lại |
|---|---|---|---|---|
| 1 | Báo cáo vẫn giữ số bàn giao (chưa ghim `thread_count`) thay vì số đã chạy lại trên Linux; so sánh 4 cách chia với holdout không nhất quán; không nói có thật sự lệch do `thread_count` hay không | **PASS** | `docs/BAO_CAO_KET_QUA_HOLDOUT.md:60-68` (khối chính canonical `thread_count=1`), `:70-77` (khối bàn giao có dán nhãn), `:79-86` (bảng chênh lệch canonical − bàn giao), `:88-94` (nguồn và luật tính), `:172-183` (thí nghiệm có kiểm soát 1.6); `outputs/step4_thread1/catboost_training/metrics_chunk2_5.csv:2-5`; `outputs/thread_count_sensitivity/thread_count_sensitivity.json`; lệnh recompute OOF (mục 6) cho kết quả khớp 8/8 chỉ số ở 4 chữ số | Bảng 1.6 không render đúng do chưa escape `\|` ở header (`docs/BAO_CAO_KET_QUA_HOLDOUT.md:176`) — G1 |
| 2 | Changelog chỉ "saved" là đã chạy holdout nhiều lần mà không có số các lần trước; cần chứng minh không cherry-pick; teammate chạy lại nếu trùng số thì phải ghi rõ là trùng số | **PASS** | `docs/BAO_CAO_KET_QUA_HOLDOUT.md:13-17` (Bảng A đủ 3 lần mở holdout kèm ROC-AUC/F1/top-50/PF), `:19` (baseline giống nhau cả 3 lần), `:21-28` (xác minh teammate + "không ghi lại metric nào"), `:30-32` (artifact canonical + lý do chạy lại); `outputs/verification/holdout_run_history.json` (`runs[0..2]`, `pairwise_deltas`, `external_verification`, `assertions` 12/12 `passed=true`); `README.md:481-489`; `README_VI.md:470-478` | Cụm từ "opened once / nobody touched" ở `README.md:337,15` và `README_VI.md:332,13-14` chưa được định lượng so với 3 lần mở — G3 |
| 3 | Thiếu bảng 20–80% của 4 cách chia đặt cạnh bảng 20–80% của holdout | **PASS** | `docs/BAO_CAO_KET_QUA_HOLDOUT.md:125-158` (Bảng 3: 4 cách chia × 7 mức = 28 dòng) và `:160-170` (Bảng 4: holdout 20–80%, 7 dòng); `outputs/holdout/stage4/table3_branch_sweep_20_80.csv`; nguồn `outputs/step4_thread1/backtest/backtest_retention_sweep.csv` | Không |
| 4 | Báo cáo chỉ có link HTML, chưa chèn 2 hình biểu đồ | **PASS** | `docs/BAO_CAO_KET_QUA_HOLDOUT.md:202-204` (2 ảnh PNG nhúng bằng cú pháp Markdown `![...]`), `:206-207` (link HTML tương tác); kiểm 32/32 dòng bảng file `:248-283` khớp kích thước file thật; PNG không còn tag `Software` | Không |
| 5 | Phép kiểm "chạy 2 lần trên cùng máy" chưa xong (mới so điểm train), thiếu holdout/backtest/bảng/biểu đồ; môi trường thiếu hệ điều hành và phiên bản plotly | **PASS** | `outputs/holdout/repro/stage5_repro_report.json`: `status=PASS`, chấm holdout 5.028 dòng `max_abs_diff=0.0`, `stage3_backtest_summary` 8/8 cột `0.0`, `stage3_top_k_selections` 7/7 `keys_identical=true`, 4 bảng Stage 4 `dataframes_exactly_equal=true`, 4 biểu đồ `bytes_identical=true`; `docs/BAO_CAO_KET_QUA_HOLDOUT.md:240-246` (PASS đủ chuỗi + OS + Plotly + Matplotlib); `outputs/holdout/stage2/train_run1_report.json` `versions` có `platform`, `plotly`, `matplotlib` | Dòng kiểm chứng `docs/BAO_CAO_KET_QUA_HOLDOUT.md:241` bị cắt khi render do chứa `max \|Δprobability\|` chưa escape — G2 |
| 6 | 4 ý nhỏ: (a) Bảng 2 chưa tách khối 4 cách chia vs holdout; (b) chữ số thập phân không đều; (c) thiếu giai đoạn holdout 2025-02-09 → 2026-08-21; (d) chưa ghi 4 lệnh biên và phép cộng 25,008 + 4 + 5,028 = 30,040 | **PASS** | (a) `docs/BAO_CAO_KET_QUA_HOLDOUT.md:96-123` tách 3 khối có tiêu đề riêng (canonical/`thread_count=1`, tham chiếu bàn giao, holdout); (b) `:198` ghi quy ước chữ số, script kiểm decimal **0 cột lệch**; (c) `:187` ghi `entry_time` 2025-02-09 16:00:00 → 2026-08-21 04:30:00, `label_end_time` muộn nhất 2026-08-21 07:15:00; (d) `:188-197` liệt kê 4 lệnh biên và tổng 30,040 | Không có gap nội dung cho 4 ý; còn lỗi trình bày chung ở G1/G2 |

**Kết luận mục 1:** Không mục nào FAIL. Cả 6 mục PASS về số liệu và nội dung; các gap G1–G7 là lỗi trình bày/biên soạn, không đổi kết luận.

---

## 2. Cross-check số liệu bằng chương trình

Tất cả so khớp đều chạy bằng `.venv/bin/python` với `decimal.Decimal` (không dùng so sánh chuỗi mờ),
trừ các phép recompute dùng `pandas` + `scikit-learn`. Dấu `+`, dấu trừ Unicode (`−`) và dấu phẩy
phân cách hàng nghìn được chuẩn hóa trước khi so.

| # | Phép kiểm | Nguồn đối chiếu | Số giá trị so | Kết quả |
|---|---|---|---|---|
| 2.1 | Bảng 1 khối canonical (4 dòng ROC-AUC/F1) | `outputs/step4_thread1/catboost_training/metrics_chunk2_5.csv:2-5` | 8/8 | **0 mismatch** (khớp ở 4 chữ số: 0.8582/0.6494, 0.7454/0.5077, 0.5875/0.3288, 0.5895/0.3247) |
| 2.2 | Bảng 2 khối holdout (2 dòng × 5 cột) | `outputs/holdout/stage3/stage3_backtest_summary.csv` (keep 100 và 50) | 10/10 | **0 mismatch** (5.028/+245.93/305.32/1.0992/35.28; 2.514/−36.55/263.25/0.9718/34.81) |
| 2.3 | Bảng 3 (4 cách chia × 7 mức × 5 cột) | `outputs/step4_thread1/backtest/backtest_retention_sweep.csv` | 140/140 | **0 mismatch**; `outputs/holdout/stage4/table3_branch_sweep_20_80.csv` cũng khớp canonical sweep 28/28 dòng |
| 2.4 | Bảng 4 (holdout 20–80%, 7 dòng × 5 cột) | `outputs/holdout/stage3/stage3_backtest_summary.csv` | 35/35 | **0 mismatch** |
| 2.5 | Bảng 1.6 `thread_count` (4 cấu hình × thread_count + 6 số + overlap) | `outputs/thread_count_sensitivity/thread_count_sensitivity.json` (`configs`, `deltas_vs_tc1_a`, `conclusion_facts`) | 32/32 | **0 mismatch** (train max 0.0/0.0/0.283158/0.299853; holdout max 0.0/0.0/0.346856/0.335194; Pearson 1.0000/1.0000/0.9331/0.9409; net R −36.55/−36.55/−12.26/+30.79; overlap 2514/2514/2252/2254) |
| 2.6 | Bảng A lịch sử 3 lần mở holdout (3 dòng × 4 số) | `outputs/verification/holdout_run_history.json` (`runs[0..2].classification`, `.top50_holdout`) | 12/12 | **0 mismatch** (0.6050/0.4017/−23.83/0.9815; 0.6023/0.4065/+30.79/1.0239; 0.6046/0.4022/−36.55/0.9718) |
| 2.6b | Bảng sweep 3 lần chạy + bảng pairwise trong `holdout_run_history.md` | `holdout_run_history.json` | 21 + 18 | **0 mismatch số học** (1 khác biệt format `0.31904` vs `0.319040` do số 0 cuối, không phải lệch số) |
| 2.7 | Bảng môi trường báo cáo (8 dòng) | `outputs/holdout/stage2/train_run1_report.json` `versions` | 8/8 | **0 mismatch**, gồm OS (`Linux-7.2.5-1-cachyos-x86_64-with-glibc2.44`), Plotly `7.0.0`, Matplotlib `3.11.2`; `stage5_repro_report.json` `environment` cũng khớp đủ 8 khóa |

**Tuyên bố:** không phát hiện mismatch số học nào trong toàn bộ 6 phép kiểm bắt buộc và các phép kiểm bổ sung ở mục 2.

### 2.8. Phép kiểm bổ sung đã chạy

| Phép kiểm | Kết quả |
|---|---|
| Recompute Bảng 1 từ 4 file OOF (canonical và bàn giao) theo đúng luật `chunk=1+searchsorted([5001,10003,15004,20006,25008], row_id, 'right')`, bỏ `chunk==1`, `fold>=1`, lấy trung bình fold | canonical 0.8581816112/0.6493677055, 0.7453547557/0.5077137634, 0.5874589748/0.3287537694, 0.5895341560/0.3246870069; bàn giao 0.8594955970/0.6524771196, 0.7474686023/0.5105026762, 0.5867245511/0.3267281763, 0.5947758032/0.3304223166 — làm tròn 4 chữ số khớp đủ 8/8 giá trị báo cáo/README |
| Bảng delta "canonical − bàn giao" tính trên giá trị chưa làm tròn | Khớp 8/8: −0.0013/−0.0031, −0.0021/−0.0028, +0.0007/+0.0020, −0.0052/−0.0057 |
| 4 lệnh biên + phép cộng | Full regenerated 30.040 dòng = 25.008 train + **4 lệnh biên** (origin_bar 199930, entry_bar 199930–199933, leg 0–3, entry_time 2025-02-08 06:00–06:45) + 5.028 holdout; giao khóa train∩holdout = 0; khớp chính xác bảng `docs/BAO_CAO_KET_QUA_HOLDOUT.md:190-195` và tổng ở `:197` |
| Giai đoạn holdout | `dataset_catboost_holdout.csv` entry_time 2025-02-09 16:00:00 → 2026-08-21 04:30:00, label_end_time max 2026-08-21 07:15:00 — khớp `:187` |
| 8 file CSV canonical vs commit `7748828` | SHA-256 giống hệt 8/8 (metrics_by_fold, metrics_summary, 4 OOF, backtest_summary_top50, backtest_retention_sweep); `outputs/step4_thread1/comparison_report.json` `vs_commit_7748828.all_files_bytes_identical=true` |
| Biểu đồ (dữ liệu nhúng Plotly) | Biểu đồ 1 dùng dữ liệu canonical: endpoint lần lượt +6937.53476724, +4752.04217114, +2148.30521167, +2119.47819179 (không phải số bàn giao); biểu đồ 2: +245.92968266 và −36.55275499; run1 và run2 HTML byte-identical |
| PNG không phụ thuộc phiên bản | `PIL.Image.open(...).info` chỉ còn `dpi`, không có key `Software` |
| Bảng "Phần 5 — Bảng file sinh ra" | 32/32 dòng khớp kích thước file thật (gồm 2 PNG 93.985 B / 81.443 B) |
| Decimal convention | Quét mọi cột số của 16 bảng trong báo cáo: **không cột nào trộn số chữ số thập phân** (ROC/F1/PF 4; net R/MaxDD/win 2; 1.6 max\|Δp\| 6) |

---

## 3. Ràng buộc task-spec (a)–(h)

| Mã | Ràng buộc | Phán quyết | Bằng chứng |
|---|---|---|---|
| (a) | Không có cột "sai lệch so với holdout" | **PASS** | `docs/BAO_CAO_KET_QUA_HOLDOUT.md` không có cột nào tên "sai lệch so với holdout"; bảng chênh lệch duy nhất là "canonical − bàn giao" (`:79-86`) — đúng thứ feedback yêu cầu bổ sung, không phải cột deviation-vs-holdout |
| (b) | Bảng 2 tách rõ khối 4 cách chia và holdout | **PASS** | `:96-106` "Khối chính — 4 cách chia …", `:108-116` "Khối tham chiếu — bàn giao …", `:118-123` "Khối holdout — giữ top 50%" |
| (c) | Ghi giai đoạn holdout + 4 lệnh biên + 25.008 + 4 + 5.028 = 30.040 | **PASS** | `:187` (giai đoạn), `:188-196` (4 lệnh biên), `:197` (tổng); kiểm độc lập bằng pandas khớp 30.040/25.008/5.028/4 |
| (d) | Quy ước chữ số thập phân đều | **PASS** | `:198` quy ước; script decimal 0 cột lệch; số dòng trong mỗi bảng dùng đúng số chữ số của cột đó |
| (e) | Phép kiểm 2 lần chạy hoàn tất, **không còn "CHƯA XÁC NHẬN"**, phủ holdout + backtest + bảng + biểu đồ | **PASS** | `grep` ba tài liệu (báo cáo + 2 README) **0 kết quả** cho "CHƯA XÁC NHẬN"/"chưa xác nhận"; `stage5_repro_report.json` PASS đủ 7 nhóm item + hash 20 file (16 file CSV/JSON/MD + 4 biểu đồ); `docs/BAO_CAO…:240-246` |
| (f) | Môi trường có OS + Plotly (+ Matplotlib) | **PASS** | `:45-56` bảng môi trường (OS, Plotly 7.0.0, Matplotlib 3.11.2), `:246` footer; `train_run1_report.json`/`train_run2_report.json` `versions` và `stage5_repro_report.json` `environment` |
| (g) | Changelog ghi số các lần holdout trước + ghi chú teammate "trùng số" | **PASS** | `README.md:481-489` và `README_VI.md:470-478` ghi đủ 3 lần (`dc25cd3` 0.60502/0.40171, top-50 −23.83 R; `5f46e41` 0.60232/0.40647, +30.79 R; `7748828` 0.60455/0.40221, −36.55 R) và ghi chú teammate Bùi Quốc Thịnh "prediction trùng trong `1e-15`, không byte-identical, không ghi metric"; `external_verification` trong JSON |
| (h) | Thay đổi code xem được từ repo (script + commit + comment) | **PASS*** | Commit: `54eed41` (output-dir + canonical rerun), `b278617` (thí nghiệm thread_count), `a3ff712` (run history), `8003c82` (run2 + env + manifest), `ee24520` (sinh báo cáo), `bdf25eb`/`93cdf50` (changelog + head). Comment: `src/citd_ml/training/train_catboost.py:22-29`; `scripts/holdout_stage4_report.py:308-310`; `scripts/holdout_stage2_train.py:21-23,31-32`. *Vòng best-practice mới nhất đang ở working tree chưa commit (đã ghi rõ trong `docs/CODE_BEST_PRACTICE_AUDIT.md:143-194` và changelog) — xem G5 |

---

## 4. Quét câu stale / mâu thuẫn còn sót

Đã quét `docs/BAO_CAO_KET_QUA_HOLDOUT.md`, `README.md`, `README_VI.md` cho các mẫu
`CHƯA XÁC NHẬN`, `chưa xác nhận`, `chưa có thí nghiệm thread_count`, `lấy nguyên từ bàn giao`,
`opened once`, `nobody touched`, `chưa ai đụng`, `đã được thay thế`, v.v.

| File:line | Câu còn sót (rút gọn) | Phân loại | Đánh giá |
|---|---|---|---|
| `README.md:15` | "…evaluates the model on a **sealed holdout period** that nobody touched during development." | **Mâu thuẫn thật (nhẹ) về diễn đạt** | Sổ lịch sử ghi 3 lần mở holdout trong quá trình phát triển. Nên định lượng: "no model training used the holdout; the three documented openings are in `holdout_run_history.json`" |
| `README_VI.md:13-14` | "…giai đoạn holdout niêm phong mà chưa ai đụng tới trong suốt quá trình phát triển." | **Mâu thuẫn thật (nhẹ) về diễn đạt** | Cùng lý do trên |
| `README.md:337` | "The holdout is opened once, in four stages, then compared by a fifth check." | **Mâu thuẫn thật (nhẹ) về diễn đạt** | Chính README `:111` ghi "All three sealed-holdout openings". Nên sửa thành "the canonical workflow opens the holdout once; historically three openings…" |
| `README_VI.md:332` | "Holdout chỉ mở một lần, qua 4 giai đoạn, rồi được đối chiếu ở phép kiểm thứ 5." | **Mâu thuẫn thật (nhẹ) về diễn đạt** | Cùng lý do trên |
| `README.md:569-571` | "Fixed chart 1 … it now builds from the restored handed-over `backtest_scored_universe.csv`, so its 4 branch endpoints match … (+6,998.3 / +4,724.3 / +2,207.3 / +2,253.7 R)." | **Entry lịch sử nay đã cũ** | Là changelog 2026-09-12, nhưng biểu đồ 1 hiện dựng từ cây canonical (endpoint +6.937,53 / +4.752,04 / +2.148,31 / +2.119,48 R; `implementation_decisions.md:4`, `docs/BAO_CAO…:227`). Supersession note ở `:566-568` chỉ nói Table 1/2, chưa nói chart 1 — nên chú thích thêm |
| `README_VI.md:557-559` | "**Sửa** biểu đồ 1 … khớp Bảng 2 (+6.998,3 / +4.724,3 / +2.207,3 / +2.253,7 R)." | **Entry lịch sử nay đã cũ** | Cùng lý do trên |
| `README_VI.md:516` | "ghi rõ toàn pipeline tái lập từng byte **lúc đó** chưa được chứng minh — đã được thay thế ngày 2026-09-15: chuỗi run1-vs-run2 trên cùng máy nay PASS" | **Entry lịch sử, đã tự chú thích supersession** | Không phải mâu thuẫn: câu nói về trạng thái 2026-09-13 và đã ghi rõ thay thế |
| `README.md:527-529` | "marked full-pipeline byte reproducibility as unproven **at the time** — superseded on 2026-09-15…" | **Entry lịch sử, đã tự chú thích supersession** | Không phải mâu thuẫn |
| `README.md:579-586` | "at the time the repository contained no controlled experiment changing only `thread_count`… (superseded on 2026-09-15…)" | **Entry lịch sử, đã tự chú thích supersession** | Không phải mâu thuẫn |
| `docs/BAO_CAO_KET_QUA_HOLDOUT.md:244` | "Kiểm chứng liên máy \| **GIỚI HẠN** — … không thể tái tạo từ git…" | **Giới hạn trung thực, không phải mâu thuẫn** | Phù hợp với `stage2_reproducibility_report.json` `cross_machine_claim = not established by this command` |

**Kết luận mục 4:** không còn câu "CHƯA XÁC NHẬN" hay tuyên bố sai số liệu. Có 4 câu diễn đạt
chưa định lượng về số lần mở holdout (G3) và 2 entry changelog lịch sử chưa chú thích thay thế
cho biểu đồ 1 (G4); tất cả là lỗi biên soạn, không ảnh hưởng số.

---

## 5. "Chứng minh trong sạch" — hướng của các con số

Yêu cầu: kết quả canonical `thread_count=1` **không được** tốt hơn bản đa luồng theo kiểu
cherry-pick. Kiểm tra hướng số bằng `outputs/verification/holdout_run_history.json` và
`outputs/step4_thread1/comparison_report.json`:

| So sánh | `thread_count=1` (canonical) | Không ghim / đa luồng | Hướng |
|---|---|---|---|
| Holdout top 50% net R | **−36.55 R** (`7748828`) | +30.79 R (`5f46e41`, Linux đa luồng); −23.83 R (`dc25cd3`, Windows) | tc=1 **xấu nhất trong cả 3 lần mở holdout** |
| Holdout ROC-AUC | 0.6046 | 0.6023 (`5f46e41`); 0.6050 (`dc25cd3`) | tốt hơn Linux đa luồng +0.0022, **kém hơn** bản Windows |
| Holdout F1 @0.5 | 0.4022 | 0.4065 (`5f46e41`); 0.4017 (`dc25cd3`) | **kém hơn** bản đa luồng |
| Thí nghiệm có kiểm soát top-50 net R | **−36.55 R** (`tc=1`) | −12.26 R (`tc=2`); **+30.79 R** (mặc định `−1`) | tc=1 là cấu hình **tệ nhất** |
| 4 cách chia chunk 2–5, net R canonical − bàn giao | −60.77 / +27.70 / −58.94 / −134.24 R | — | 3/4 nhánh **xấu hơn**; nhánh Grouped tốt hơn nhẹ +27.70 R (~0,6%) |
| 4 cách chia chunk 2–5, ΔROC-AUC canonical − bàn giao | −0.0013 / −0.0021 / +0.0007 / −0.0052 | — | 3/4 **thấp hơn** |

**Kết luận:** hướng số liệu nhất quán với tuyên bố "không cherry-pick" ở `README.md:436-438`
và `README_VI.md:425-428`. Lựa chọn `thread_count=1` cho kết quả holdout **xấu hơn** cả bản
Linux đa luồng lẫn bản Windows ở chỉ số tài chính headline (top 50% net R), và xấu hơn 3/4
nhánh so với bàn giao; vì vậy không có dấu hiệu chọn cấu hình theo hướng có lợi. Toàn bộ số
của các lần trước vẫn được giữ nguyên trong sổ lịch sử (`outputs/verification/holdout_run_history.json`),
không bị thay thế.

---

## 6. Lệnh / vị trí đã dùng cho từng phán quyết

| Phán quyết | Lệnh hoặc file:line |
|---|---|
| Mục 1 — bảng canonical + thí nghiệm | `.venv/bin/python - <<'PY'` (parse bảng Markdown báo cáo + `decimal` so với `metrics_chunk2_5.csv` và `thread_count_sensitivity.json`); recompute OOF bằng `.venv/bin/python - <<'PY'` dùng `searchsorted` + `roc_auc_score`/`f1_score` |
| Mục 2 — lịch sử 3 lần mở | `.venv/bin/python - <<'PY'` so Bảng A với `holdout_run_history.json`; kiểm `assertions.checks` 12/12 `passed=true` |
| Mục 3 — bảng 20–80% | `.venv/bin/python - <<'PY'` so 140 giá trị Bảng 3 với `outputs/step4_thread1/backtest/backtest_retention_sweep.csv`; so 28 dòng `table3_branch_sweep_20_80.csv` |
| Mục 4 — ảnh biểu đồ | Kiểm 32/32 dòng bảng file (`docs/BAO_CAO_KET_QUA_HOLDOUT.md:248-283`) bằng `Path.stat().st_size`; `PIL.Image.open` đọc `info`; decode base64 `y` nhúng trong HTML để đọc endpoint |
| Mục 5 — chuỗi 2 lần chạy | `.venv/bin/python -c` đọc `stage5_repro_report.json` (`items.*.pass`, `file_hashes.*.bytes_identical`); `grep` "CHƯA XÁC NHẬN" 3 tài liệu → 0 |
| Mục 6 — 4 ý nhỏ | `.venv/bin/python - <<'PY'` đếm số thập phân từng cột; kiểm 4 lệnh biên bằng pandas trên `dataset_catboost_full_regenerated.csv` + `data/processed/dataset_catboost.csv` + `dataset_catboost_holdout.csv` |
| Byte-identity 8 CSV | vòng `sha256sum` so file hiện tại với `git show 7748828:outputs/...` (8/8 IDENTICAL) |
| pytest | `.venv/bin/python -m pytest -q` → `13 passed in 0.03s` |

---

## 7. Gap cần sửa (không có mismatch số)

| Mã | Mức | Mô tả | Vị trí | Đề xuất |
|---|---|---|---|---|
| G1 | Trình bày | Header bảng 1.6 chứa `Train max\|Δp\|` / `Holdout max\|Δp\|` chưa escape `\|`: header 13 ô nhưng delimiter 9 ô → GFM không nhận diện bảng, hiển thị dạng text | `docs/BAO_CAO_KET_QUA_HOLDOUT.md:176` | Escape thành `Train max\|Δp\|`, `Holdout max\|Δp\|` (hoặc bỏ ký tự `\|` trong tên cột) |
| G2 | Trình bày | Dòng kiểm chứng run1-vs-run2 chứa `max \|Δprobability\|` chưa escape → dòng có 4 ô so với 2 ô header; phần "ΔROC-AUC… biểu đồ byte-identical" bị cắt khi render | `docs/BAO_CAO_KET_QUA_HOLDOUT.md:241` | Escape `\|` |
| G3 | Biên soạn | "nobody touched during development"/"chưa ai đụng" và "holdout is opened once"/"chỉ mở một lần" mâu thuẫn nhẹ với sổ 3 lần mở | `README.md:15,337`; `README_VI.md:13-14,332` | Thêm vế định lượng: "canonical workflow mở một lần; lịch sử 3 lần mở ghi ở …" |
| G4 | Biên soạn | Changelog 2026-09-12 nói biểu đồ 1 khớp endpoint bàn giao (+6.998,3…); biểu đồ 1 hiện là canonical (+6.937,53…) | `README.md:569-571`; `README_VI.md:557-559` | Thêm "(superseded on 2026-09-15: chart 1 was rebuilt from the canonical tree)" |
| G5 | Quy trình | Thay đổi vòng best-practice (`MODEL_PARAMS` + `allow_writing_files`, import chung, pin matplotlib, PNG metadata, `include_plotlyjs=True`) và 2 audit doc `docs/CODE_BEST_PRACTICE_AUDIT.md`, `docs/INDEPENDENT_REPRO_AUDIT.md` **chưa commit**, mới ở working tree | `git status --short` | Commit theo cùng chuẩn các commit `54eed41…ee24520`; hiện tại đã được mô tả trong audit §7 + changelog nên vẫn review được |
| G6 | Nhãn | `holdout_stage4_report.py` hardcode "(run 1)"; báo cáo run2 ghi artifact lấy từ `repro/run2/stage3` nhưng lại nhãn "(run 1)" | `scripts/holdout_stage4_report.py:773`; `outputs/holdout/repro/run2/BAO_CAO_holdout_run2.md:30` | Tham số hóa nhãn run theo `--stage3-dir` |
| G7 | Biên soạn | `docs/INDEPENDENT_REPRO_AUDIT.md` liệt kê kích thước file của HEAD (PNG 94.055/81.513 B; `.cbm` 1.129.392 B; `train_run*_report.json` 1.949 B; sensitivity JSON 15.560 B; stage5 28.153 B; manifest 5.286 B) — khớp commit cũ nhưng không còn khớp working tree sau vòng best-practice (93.985/81.443 B; 1.129.416 B; 2.011 B; 16.782 B; 28.181 B; 5.320 B) | `docs/INDEPENDENT_REPRO_AUDIT.md:180-216` | Ghi chú rõ đây là snapshot tại HEAD `93cdf50`, hoặc cập nhật sau khi commit vòng best-practice |

Không gap nào trong G1–G7 làm sai số liệu, sai phép kiểm hay thay đổi kết luận.

---

## 8. Kết luận

- Cả 6 mục feedback review đều **PASS** về nội dung và số liệu; không mục nào FAIL hay PARTIAL.
- **0 mismatch** trên tất cả cross-check bắt buộc: Bảng 1 canonical, Bảng 2 holdout, Bảng 3 canonical sweep,
  Bảng 1.6 thread-count, Bảng A lịch sử, môi trường OS/Plotly/Matplotlib.
- Không còn câu "CHƯA XÁC NHẬN" hay tuyên bố sai số; phép kiểm 2 lần chạy đã phủ đủ
  chấm holdout + backtest + 4 bảng + 4 biểu đồ (`stage5_repro_report.json` PASS).
- Yêu cầu "chứng minh trong sạch" thỏa: canonical `thread_count=1` **xấu hơn** bản đa luồng ở
  chỉ số headline (holdout top 50% net R −36.55 R so với +30.79 R) và xấu hơn 3/4 nhánh so với bàn giao;
  số của cả 3 lần mở holdout được lưu đầy đủ, không thay thế.
- Việc cần làm để hoàn thiện: sửa G1–G2 (escape `|` trong bảng Markdown), định lượng G3,
  chú thích G4, commit vòng best-practice G5 và sửa nhãn run G6, refresh G7.

---

## 9. Addendum 2026-09-15 — trạng thái các gap sau khi sửa

> Mục này được thêm sau khi sửa; nội dung audit gốc ở mục 1–8 giữ nguyên.
> Không chạy `git add/commit/push`; thay đổi nằm trong working tree.

| Mã | Trạng thái | Đã sửa ở đâu | Bằng chứng xác minh |
|---|---|---|---|
| G1 | **FIXED** | `scripts/holdout_stage4_report.py` — escape `\|` trong header Bảng 1.6 | `docs/BAO_CAO_KET_QUA_HOLDOUT.md:176` và `outputs/holdout/repro/run2/BAO_CAO_holdout_run2.md:176` nay đủ 9 ô; script kiểm bảng thấy 16/16 bảng mỗi báo cáo có header = delimiter |
| G2 | **FIXED** | `scripts/holdout_stage4_report.py` — escape `\|` ở dòng kiểm chứng Stage 5 | `docs/BAO_CAO_KET_QUA_HOLDOUT.md:241` (và bản run 2) nay đúng 2 ô; nội dung ΔROC-AUC/ΔF1/biểu đồ không còn bị cắt khi render |
| G3 | **FIXED** | `README.md:14-18`, `README.md:340-343`; `README_VI.md:13-17`, `README_VI.md:335-338` | Định lượng đúng sổ lịch sử: holdout được giữ kín cho mọi quyết định phát triển; không feature, hyperparameter, cách chia train/holdout hay luật top-k nào thay đổi giữa các lần mở; mọi lần mở ghi trong `outputs/verification/holdout_run_history.md`; quy trình 4 giai đoạn + phép kiểm thứ 5, mọi lần mở thêm đều được log |
| G4 | **FIXED** | `README.md:574-579`; `README_VI.md:562-566` | Claim lịch sử 2026-09-12 giữ nguyên (+6.998,3 / +4.724,3 / +2.207,3 / +2.253,7 R) và thêm ghi chú superseded: từ 2026-09-15 biểu đồ 1 dựng từ `outputs/step4_thread1/backtest/backtest_scored_universe.csv` (canonical `thread_count=1`) với endpoint +6,937.53 / +4,752.04 / +2,148.31 / +2,119.48 |
| G5 | **PENDING COMMIT** | — | Vòng best-practice + 3 audit doc vẫn ở working tree; theo yêu cầu không chạy `git add/commit/push` |
| G6 | **FIXED** | `scripts/holdout_stage4_report.py` — thêm `run_label_from_stage3()`, nhãn suy từ `--stage3-dir` | Stage 4 sinh lại cả hai run; `outputs/holdout/repro/run2/BAO_CAO_holdout_run2.md:30` nay ghi `(run 2)`; báo cáo run 1 vẫn `(run 1)` |
| G7 | **FIXED** | `docs/INDEPENDENT_REPRO_AUDIT.md` — mục "Addendum 2026-09-15" | Ghi rõ vòng best-practice cố ý đổi byte `.cbm` và metadata/kích thước PNG (93.985 B / 81.443 B), mọi số liệu giữ nguyên; `outputs/holdout/repro/stage5_repro_report.json` (28.181 B) và `outputs/step4_thread1/verification/reproducibility.json` (5.320 B) đã refresh; các bảng cũ giữ nguyên như snapshot HEAD `93cdf50` |

Kiểm chứng sau khi sửa:

- `.venv/bin/python scripts/holdout_stage5_repro_check.py` → **PASS**: mọi delta số học `0.0`,
  4/4 biểu đồ (HTML + PNG) byte-identical, `manifests_exactly_equal = true`.
- Parse mọi bảng Markdown của hai báo cáo: **16/16 bảng mỗi file** có số ô header = delimiter
  (và mọi dòng dữ liệu khớp số ô).
- Spot-check dòng đầu Bảng 1/2/3/4 của cả hai báo cáo khớp artifact nguồn
  (`metrics_chunk2_5.csv`, `backtest_summary_top50.csv`, `backtest_retention_sweep.csv`,
  `stage3_backtest_summary.csv`).
- Hash 4 bảng CSV + 2 file MD Stage 4 của run 1 và run 2 **không đổi** sau khi sinh lại;
  chỉ 2 dòng trình bày của mỗi báo cáo đổi (G1/G2) và nhãn run 2 (G6).
- `.venv/bin/python -m pytest -q` → **13 passed**.
- Không file nào thuộc danh sách đóng băng bị sửa; không số liệu nào thay đổi.
