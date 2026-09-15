# AUDIT CODE THEO CHUẨN THƯ VIỆN — CITD ML

- **Ngày audit:** 2026-09-15
- **Phạm vi:** read-only. Không sửa bất kỳ file nào; tài liệu này là file duy nhất được tạo.
- **Môi trường xác minh:** Python 3.12.14 · catboost 1.2.10 · scikit-learn 1.9.0 · pandas 3.0.5 · numpy 2.5.2 · plotly 7.0.0 · matplotlib 3.11.2 (đọc từ metadata PNG committed).
- **Trạng thái git trước audit:** sạch (`git status --porcelain` rỗng). Không chạy `git add/commit/push`.
- **Không chạm artifact đóng băng:** `data/processed`, `outputs/catboost_training`, `outputs/backtest`, `outputs/verification/reproducibility.json`, `outputs/holdout/stage1-3`, `docs/BAN_GIAO_*`, `docs/BAO_CAO_KET_QUA_HOLDOUT.md`.

---

## 1. Scope + phương pháp

### 1.1. File đã đọc (đầy đủ)

| Nhóm | File |
|---|---|
| Training | `src/citd_ml/training/train_catboost.py`, `src/citd_ml/training/pre_train.py`, `src/citd_ml/training/split_data.py` |
| Backtest | `src/citd_ml/backtest/backtest_pyramid_local.py` |
| Features / strategy | `src/citd_ml/features/build_features.py`, `src/citd_ml/features/verify_dataset.py`, `src/citd_ml/strategy/pyramid_strategy.py`, `src/citd_ml/labeling/triple_barrier.py` |
| Verification | `src/citd_ml/verification/verify_pipeline.py`, `src/citd_ml/verification/holdout_evidence.py` |
| Scripts holdout | `scripts/holdout_stage2_train.py`, `scripts/holdout_stage3_backtest.py`, `scripts/holdout_stage4_report.py`, `scripts/holdout_stage5_repro_check.py` |
| Scripts thí nghiệm / lịch sử | `scripts/thread_count_sensitivity.py`, `scripts/holdout_run_history.py` |
| Scripts tiện ích | `scripts/train_models.py`, `scripts/run_backtest.py`, `scripts/build_dataset.py`, `scripts/verify_pipeline.py`, `scripts/verify_dataset.py` |
| Packaging / tests | `pyproject.toml`, `requirements.txt`, `uv.lock` (đọc phần liên quan), `tests/test_package.py`, `src/citd_ml/paths.py`, `src/citd_ml/__init__.py` |

### 1.2. Cách kiểm tra

- **ctx7 CLI** (`npx -y ctx7@latest library ...` + `npx -y ctx7@latest docs ...`) cho 5 library ID (mục 1.3).
- **Kiểm tra runtime read-only:** `inspect.signature` và docstring của package đã cài (catboost, sklearn, plotly, matplotlib); gọi Plotly `to_html` trong bộ nhớ (không ghi file); chạy `pre_train` + `split_data` với `-W error::FutureWarning -W error::DeprecationWarning`.
- **Không chạy** full train/backtest/holdout để tránh ghi artifact; các kết luận về API dựa trên static + runtime read-only.
- **Test suite:** `.venv/bin/python -m pytest -q` → `13 passed in 0.03s`.

### 1.3. Library ID đã query và fact tài liệu đã dùng

| Library ID | Fact đã dùng (quote ngắn) |
|---|---|
| `/catboost/catboost` | `thread_count` (training): *"The `thread_count` parameter specifies the number of threads used during training. For CPU, it optimizes execution speed and does not affect results."* · `predict_proba`: *"The number of threads to use for computation. Optimizes the speed of execution."*; `staged_predict_proba`: *"This parameter doesn't affect results."* · `random_seed`: *"The random seed (random_seed or random_state) is used for training to ensure reproducibility."* · `allow_writing_files`: *"The `allow_writing_files` parameter determines whether analytical and snapshot files can be written during training. Setting it to `False` disables tools like snapshotting and data visualization."* · `train_dir`: *"The directory for output files is specified using the `train-dir` (`train_dir`) parameter. The default value for this parameter is the current directory."* (ctx7 còn trích source `catboost_logger_helpers.cpp`: nếu `AllowWriteFiles()==false` thì không tạo file; ngược lại tạo thư mục `trainDir`.) · `silent`: *"Controls the logging level during training. Setting to True enables silent logging."* · `save_model`: *"Saves the trained model to a file"* (signature `save_model(fname, format="cbm", export_parameters=None, pool=None)`) · `auto_class_weights`: *"'Balanced' weights classes as `maxSumWeightInClass / sumWeightInClass`"*; *"Do not use with `class_weights` or `scale_pos_weight`."* · early stopping (tutorial): *"Training stops if AUC on the evaluation set does not improve for `early_stopping_rounds`"* — cần `eval_set`. |
| `/scikit-learn/scikit-learn` | `roc_auc_score` binary case: *"the probability of the class with the 'greater label' ... should be provided as the `y_score` parameter."* · `GroupKFold`: `shuffle` và `random_state` *".. versionadded:: 1.6"*; *"Pass an int for reproducible output across multiple function calls."* · glossary `random_state`: *"ensuring reproducibility of results"*. |
| `/websites/pandas_pydata` | `to_csv`: `float_format` = *"Format string for floating point numbers"*, `date_format` = *"Type of ... format"* cho cột datetime · `read_csv`: *"Use `parse_dates` to identify columns for date conversion"* · ghi chú deprecation: `date_parser` bị thay bằng `date_format` từ pandas 2.0 (repo không dùng). |
| `/websites/matplotlib_stable` | Agg: *"non-interactive backend ... particularly useful for generating plots in environments without a display"* · `savefig(..., metadata=...)`: *"Key/value pairs to store in the image metadata."* · docstring PNG (`FigureCanvasAgg.print_png`, bản 3.11.2 đã cài): *"If 'Software' is not given, an autogenerated value for Matplotlib will be used. This can be removed by setting it to None."* |
| `/plotly/plotly.py` | `to_html` signature có `include_plotlyjs`, `full_html`, `div_id` · `include_plotlyjs`: *"If True, a script tag containing the plotly.js source code (~3MB) is included in the output. HTML files generated with this option are fully self-contained and can be used offline."* (các giá trị docs liệt kê: `True`, `'cdn'`, `'directory'`, chuỗi kết thúc `.js`, `False`) · `div_id`: *"If provided, this is the value of the id attribute of the div tag. If None, the id attribute is a UUID."* |

### 1.4. Kiểm chứng tại chỗ (runtime, read-only)

- `include_plotlyjs="inline"` và `include_plotlyjs=True` cho **cùng một chuỗi HTML byte-for-byte**; gọi lặp lại cùng tham số cũng cho cùng bytes; bỏ `div_id` thì Plotly sinh UUID ngẫu nhiên (xác nhận giá trị của `div_id` cố định trong repo).
- PNG committed có chunk `tEXt Software = "Matplotlib version3.11.2, https://matplotlib.org/"`.
- `catboost_info/` đang tồn tại ở gốc repo: `catboost_training.json`, `learn_error.tsv`, `time_left.tsv`, `learn/`, `tmp/` (bị `.gitignore:19` bỏ qua).
- Đối chiếu JSON thí nghiệm `outputs/thread_count_sensitivity/thread_count_sensitivity.json`: `tc1_a` vs `tc1_b` = `max|Δp| 0.0` (train + holdout); `tc2`/`tc_default` khác baseline `25.008/25.008` dòng train và `5.028/5.028` dòng holdout ở ngưỡng `>1e-12`; `prediction_thread_count_changed_predictions = false`; `fixed_thread_count_training_is_repeatable = true`.
- `outputs/holdout/stage2/stage2_reproducibility_report.json`: `predictions_exactly_equal = true`, `max_prediction_abs_difference = 0.0`, `model_hashes_equal = false`.

---

## 2. Bảng findings

Thang mức độ dùng trong audit:

- **BLOCKER** — sai kết quả / crash / corrupt dữ liệu.
- **MAJOR** — có thể âm thầm làm sai bằng chứng tái lập hoặc phá pipeline tài liệu, chưa gây lỗi số ngay.
- **MINOR** — vệ sinh mã / cài đặt / khả năng chuyển máy, không ảnh hưởng số hiện tại.
- **NIT** — cải thiện nhỏ, gần như không rủi ro.

| ID | Mức | File:line | Tài liệu nói gì | Code hiện tại | Đổi số liệu? | Khuyến nghị |
|---|---|---|---|---|---|---|
| CAT-01 | MINOR | `src/citd_ml/training/train_catboost.py:14` (dict), `:97` (fit); `scripts/holdout_stage2_train.py:30`, `:93`; `scripts/thread_count_sensitivity.py:64`, `:373`, `:386` | `train_dir` mặc định là **thư mục hiện tại**; `allow_writing_files=False` tắt ghi file phân tích/snapshot | Không đặt `allow_writing_files`/`train_dir` → mỗi lần `fit` sinh `catboost_info/` ở gốc repo (đang tồn tại; `.gitignore:19` chỉ che, không ngăn ghi) | Không | Thêm `"allow_writing_files": False` vào `MODEL_PARAMS` (một nguồn, xem CAT-02) hoặc trỏ `train_dir` vào thư mục scratch. Nếu muốn snapshot thì dùng `train_dir` thay vì tắt hẳn |
| CAT-02 | MAJOR | `scripts/holdout_stage2_train.py:30-42`; `scripts/thread_count_sensitivity.py:64-72`; so với `src/citd_ml/training/train_catboost.py:14-26` | `random_seed` dùng để *"ensure reproducibility"*; evidence chỉ đáng tin khi tham số train là một nguồn duy nhất | `PARAMS` (stage 2) và `BASE_PARAMS` (thí nghiệm) **sao chép tay** dict của `MODEL_PARAMS`; hiện tại giá trị giống nhau, nhưng `verify_pipeline.py:179` ghi `training.MODEL_PARAMS` vào manifest còn `train_run1_report.json` ghi `PARAMS` riêng | Không (nếu giá trị giữ nguyên) | Import `MODEL_PARAMS` từ `citd_ml.training.train_catboost`; stage 2 dùng `PARAMS = {**MODEL_PARAMS}`; thí nghiệm dùng `{**MODEL_PARAMS, "thread_count": tc}`. Nếu muốn tách biệt hoàn toàn, thêm assert equality khi khởi động |
| CAT-03 | NIT | `scripts/holdout_stage2_train.py:97`; `scripts/holdout_stage3_backtest.py:138`; `scripts/thread_count_sensitivity.py:376-377, 475-476` | `thread_count` lúc predict *"Optimizes the speed of execution"* / *"doesn't affect results"* | Prediction dùng mặc định `-1` (tất cả core), không ghim; thí nghiệm đo trên máy: `prediction_thread_count_changed_predictions = false` | Không | Tùy chọn: truyền `thread_count=1` tường minh ở call predict để ý định rõ ràng. Không bắt buộc vì docs và đo đạc đều nói không đổi kết quả |
| SKL-01 | NIT | `src/citd_ml/training/train_catboost.py:102-103`; `scripts/holdout_stage3_backtest.py:140`; `scripts/holdout_stage4_report.py:169-170` | `f1_score(y_true, y_pred, *, ..., zero_division='warn')`; nhãn dương ngầm định `pos_label=1` | Dùng đúng F1 nhị phân; mỗi fold đã validate đủ 2 lớp ở `y` (`train_catboost.py:93`) nên không có cảnh báo thực tế; ngưỡng 0.5 là quyết định đã ghi tài liệu | Không | Tùy chọn: thêm `zero_division=0` cho tường minh. AUC dùng `predict_proba[:,1]` đúng chuẩn binary của docs |
| PND-01 | NIT | `src/citd_ml/training/pre_train.py:105-109` | *"Use `parse_dates` to identify columns for date conversion"* | `pd.read_csv(path)` không `parse_dates` rồi `pd.to_datetime` từng cột ở dòng 108-109 (các loader khác như `holdout_stage2_train.py:50-52`, `holdout_stage3_backtest.py:135` dùng `parse_dates`) | Không | Đọc 1 lần: `pd.read_csv(path, parse_dates=["entry_time", "label_end_time"])`, bỏ vòng lặp convert |
| MPL-01 | MINOR | `scripts/holdout_stage4_report.py:20-22` (import) và `:296-309` (savefig); `pyproject.toml:10-16`; `requirements.txt:3-7` | Package dùng trực tiếp phải được khai báo; PNG là artifact được đối chiếu byte (`tests/test_package.py:203-209`) | `matplotlib` được import trực tiếp nhưng **không** có trong `pyproject.toml`/`requirements.txt`; hiện chỉ có nhờ catboost kéo theo (`uv.lock:19` trong deps của catboost, package ở `uv.lock:375`, bản 3.11.2). Không file evidence nào ghi version matplotlib (grep `outputs/holdout/stage2/train_run1_report.json`, `outputs/holdout/repro/stage5_repro_report.json`, `outputs/verification/*` → 0 kết quả) | Không | Thêm `matplotlib==3.11.2` vào cả `pyproject.toml` và `requirements.txt`; ghi version matplotlib vào `versions` của stage 2/stage 5. Môi trường hiện tại vẫn 3.11.2 nên byte PNG/HTML không đổi |
| MPL-02 | NIT | `scripts/holdout_stage4_report.py:308` | `metadata` lưu key/value vào ảnh; docstring PNG: *"If 'Software' is not given, an autogenerated value ... can be removed by setting it to None."* | `figure.savefig(path, format="png", dpi=110)` → PNG chứa `Software = "Matplotlib version3.11.2..."`; byte ổn định trong cùng version, khác khi đổi version | Không đổi số; **đổi byte PNG** nếu sửa | Nếu muốn PNG ổn định qua version: `metadata={"Software": None}`. Việc này làm đổi hash PNG → phải chạy lại stage 4/5 và cập nhật hash nếu thực hiện |
| PLT-01 | NIT | `scripts/holdout_stage4_report.py:288` | Docs liệt kê `include_plotlyjs` = `True`, `'cdn'`, `'directory'`, `'<...>.js'`, `False`; không có `'inline'` | Dùng `include_plotlyjs="inline"`. Kiểm tra runtime: source plotly chỉ đặc biệt hoá `'cdn'`/`'directory'`/`.js`, còn lại truthy → nhánh inline; output **giống hệt** `include_plotlyjs=True` | Không (đã kiểm chứng byte-equal) | Đổi sang `include_plotlyjs=True` cho đúng docs; giá trị hiện tại vẫn hoạt động và không hồi quy |

**Tổng hợp mức độ:** BLOCKER 0 · MAJOR 1 · MINOR 2 · NIT 5.

---

## 3. Sai lệch có chủ đích / đã ghi tài liệu

| ID | Nội dung | Bằng chứng | Trạng thái |
|---|---|---|---|
| D1 | **Ghim `thread_count=1` dù docs nói CPU thread count không ảnh hưởng kết quả.** Trên máy này đo được đổi training `thread_count` (1 → 2 / −1) làm **toàn bộ** prediction đổi (`25.008/25.008` train, `5.028/5.028` holdout lệch `>1e-12`; `max|Δp|` train 0.2832/0.2999, holdout 0.3469/0.3352), trong khi hai lần cùng `tc=1` giống hệt từng bit; prediction-stage không đổi. | `src/citd_ml/training/train_catboost.py:22-25`; `scripts/holdout_stage2_train.py:38-41`; `outputs/thread_count_sensitivity/thread_count_sensitivity.json` (`conclusion_facts`); `README.md:398-405`; `README_VI.md:392-400` | **Documented** — giữ nguyên. Repo chủ động nêu docs mâu thuẫn với đo đạc và giới hạn kết luận ở một máy/một build. Ghi chú thêm: `tc=1` cho top-50 net R xấu hơn bản không ghim (−36.55 R vs +30.79 R) nên không phải cherry-pick (`README.md:436-439`) |
| D2 | **Không ghim `thread_count` ở bước predict** (dùng mặc định −1). Docs nói *"doesn't affect results"*; thí nghiệm xác nhận ở prediction (`prediction_thread_count_changed_predictions = false`), nên không cần ghim để tái lập. | `scripts/holdout_stage2_train.py:97`; `scripts/holdout_stage3_backtest.py:138`; JSON thí nghiệm khối `prediction_thread_count_sensitivity` | **Documented** — không bắt buộc sửa (xem NIT CAT-03 nếu muốn tường minh) |
| D3 | **File `.cbm` không byte-identical giữa 2 lần chạy cùng máy** dù prediction giống hệt: `model_hashes_equal = false`, `predictions_exactly_equal = true`. Tiêu chí PASS chỉ yêu cầu prediction byte-identical + holdout không đổi; docs `save_model` không hứa determinism byte. | `outputs/holdout/stage2/stage2_reproducibility_report.json:3`; `scripts/holdout_stage2_train.py:134,142-147`; `README.md:425-431` | **Documented** — không dùng hash `.cbm` làm tiêu chí PASS. Không cần sửa |
| D4 | **Không dùng `eval_set` / early stopping; `eval_metric="AUC"` chỉ còn vai trò metric học.** Docs minh hoạ early stopping cần `eval_set` + `od_type`/`early_stopping_rounds`; bộ hyperparameter là hợp đồng đóng băng của bàn giao nên không được đổi. | `src/citd_ml/training/train_catboost.py:14-26,97` (không có `eval_set`); `docs/BAN_GIAO_task_holdout.md` (hyperparameter cố định) | **Documented/frozen** — giữ nguyên |
| D5 | **Khác biệt liên máy trong `1e-15`, không byte-identical.** Chỉ là báo cáo miệng của teammate Windows, không có artifact để tái tạo từ git; báo cáo ghi rõ đây là thông tin tham khảo, không thay bằng chứng PASS. | `scripts/holdout_run_history.py:618-636`; `scripts/holdout_stage4_report.py:689-694`; `scripts/holdout_stage2_train.py:143` (`cross_machine_claim: "not established by this command"`) | **Documented** — giới hạn phạm vi, không phải lỗi code |

Các cấu hình còn lại của CatBoost đều **khớp docs**, không có sai lệch:

- `random_seed=42` có ở cả 3 nơi train (`train_catboost.py:21`, `holdout_stage2_train.py:37`, `thread_count_sensitivity.py:71`).
- `auto_class_weights="Balanced"` không đi kèm `class_weights`/`scale_pos_weight` (đúng cảnh báo docs).
- Không truyền `cat_features` vì toàn bộ 23 feature numeric — `pre_train.py:56-59` chủ động validate numeric.
- `task_type` để mặc định CPU; docs xác nhận CPU là mặc định và GPU chỉ dùng khi khai báo.
- `verbose=False` khi `fit` tương đương silent logging theo docs; chỉ khác là **file log vẫn được ghi** (CAT-01).

---

## 4. Verdict theo khu vực

| Khu vực | Verdict | Căn cứ |
|---|---|---|
| **CatBoost** | Có findings: 1 MAJOR (CAT-02), 1 MINOR (CAT-01), 1 NIT (CAT-03) | `thread_count` semantics, `random_seed`, `auto_class_weights`, `eval_metric`, no `cat_features`, CPU `task_type`, `save_model`/`load_model`, `predict_proba` đều dùng đúng docs. Sai lệch D1 là intentional + documented. Rủi ro chính là tham số train bị nhân bản (CAT-02) và file log không được tắt (CAT-01) |
| **scikit-learn** | Best practice OK (chỉ NIT SKL-01) | `roc_auc_score(y, predict_proba[:,1])` đúng binary case; `f1_score` nhị phân đúng; `GroupKFold(shuffle=True, random_state=0)` hợp lệ từ sklearn 1.6 và tái lập được (đang chạy 1.9.0); không dùng tham số deprecated (`multi_class` không truyền) |
| **pandas** | Best practice OK (chỉ NIT PND-01) | `to_csv(float_format="%.12g", date_format=...)` đúng API docs; `read_csv(parse_dates=...)` dùng ở hầu hết loader; không phát hiện API pandas bị removed/deprecated (grep + chạy `-W error::FutureWarning -W error::DeprecationWarning` trên `pre_train`/`split_data`: sạch); không có chained assignment/Copy-on-Write violation |
| **matplotlib** | Có findings: 1 MINOR (MPL-01), 1 NIT (MPL-02) | `matplotlib.use("Agg")` đặt trước khi import `pyplot` — đúng chuẩn headless; figsize/dpi cố định, không timestamp. Thiếu khai báo/pin dependency và version không nằm trong evidence; PNG mang tag `Software` phụ thuộc version |
| **plotly** | Best practice OK (chỉ NIT PLT-01) | `div_id` cố định để tránh UUID ngẫu nhiên (đúng docs; có test `tests/test_package.py:109-114`); `include_plotlyjs="inline"` hoạt động và byte-equal với `True` nhưng không nằm trong danh sách giá trị docs; HTML self-contained, hash byte được stage 5 kiểm (`tests/test_package.py:203-209`) |
| **Python tổng quát** | Best practice OK | `argparse` + `pathlib` + `typing` hiện đại (`list[...]`, `Path \| None`); `hashlib.file_digest` (3.11+), `importlib.metadata.version`, `datetime.now(timezone.utc)` — không có `utcnow`, không `distutils`/`pkg_resources`, không API deprecated. Các script thêm `src` vào `sys.path` là pattern nhất quán của repo, không gây lỗi |

---

## 5. Kiểm thử & bằng chứng trước khi kết luận

- **pytest:** `.venv/bin/python -m pytest -q` → **13 passed in 0.03s** (0 failed, 0 skipped).
- **Runtime deprecation check:** `pre_train.prepare_dataset()` + `split_data.make_all_folds/validate_all_folds` chạy với `-W error::FutureWarning -W error::DeprecationWarning` → không warning; `(25.008, 32)`, edges `[0, 5001, 10003, 15004, 20006, 25008]`; 4 cách chia đủ 5/5/4/4 fold.
- **Greps:** không có `DataFrame.append`, `fillna(method=...)`, `date_parser`, `infer_datetime_format`, `delim_whitespace`, `iteritems`, `applymap`, `np.NaN`/`np.float_`... trong `src/`, `scripts/`, `tests/`.
- **Plotly in-memory:** `inline` == `True` byte-for-byte; gọi lặp lại deterministic; bỏ `div_id` sinh UUID.
- **Artifact evidence:** `catboost_info/` tồn tại ở gốc repo (JUSTIFICATION cho CAT-01); `stage2_reproducibility_report.json` `model_hashes_equal=false`; sensitivity JSON xác nhận D1/D2.

---

## 6. Fix nào cần regenerate artifact?

| Finding | Có đổi số classification/backtest/holdout? | Có cần regenerate artifact? |
|---|---|---|
| CAT-01 thêm `allow_writing_files` | Không | Chỉ khi muốn `outputs/verification/reproducibility.json` + `train_run*_report.json` phản ánh key mới (nội dung JSON đổi, số không đổi). Artifact đóng băng không được tự ý ghi lại trong audit này |
| CAT-02 gộp nguồn tham số | Không (giá trị hiện giống nhau) | Không, nếu giữ nguyên giá trị |
| CAT-03 ghim predict `thread_count=1` | Không (đo được `max\|Δp\| = 0`) | Không |
| SKL-01 thêm `zero_division=0` | Không | Không |
| PND-01 `parse_dates` | Không | Không |
| MPL-01 khai báo/pin matplotlib + ghi version vào evidence | Không (môi trường vẫn 3.11.2) | Không nếu chỉ sửa packaging; **có** nếu ghi thêm version vào stage 2/stage 5 JSON (2 file evidence này sẽ đổi nội dung) |
| MPL-02 `metadata={"Software": None}` | Không đổi số | **Có** — byte PNG đổi, phải chạy lại stage 4/5 và cập nhật hash/biểu đồ committed |
| PLT-01 đổi `"inline"` → `True` | Không | Không (output byte-equal đã kiểm chứng) |

**3 khuyến nghị quan trọng nhất (ưu tiên sửa):**

1. **CAT-02** — Một nguồn tham số duy nhất: `scripts/holdout_stage2_train.py:30` và `scripts/thread_count_sensitivity.py:64` import `MODEL_PARAMS` từ `src/citd_ml/training/train_catboost.py:14`.
2. **CAT-01** — Thêm `"allow_writing_files": False` vào `src/citd_ml/training/train_catboost.py:14` (kèm hai script, hoặc tự động theo CAT-02) để không còn sinh `catboost_info/`.
3. **MPL-01** — Khai báo `matplotlib==3.11.2` trong `pyproject.toml:10` + `requirements.txt:3` và ghi version matplotlib vào evidence (`scripts/holdout_stage2_train.py:107`, `scripts/holdout_stage5_repro_check.py:448`).

> Ghi chú giới hạn audit: không chạy lại train/backtest/holdout nên không xác nhận lại số của artifact; các kết luận API dựa trên docs ctx7 + introspection package đã cài + chạy read-only phần đọc dữ liệu. Không có BLOCKER nào bị phát hiện.

---

## 7. Resolution (bổ sung 2026-09-15, sau khi áp dụng fix)

> Mục này được thêm sau khi sửa; nội dung audit gốc ở mục 1–6 giữ nguyên. Không
> có số liệu classification/backtest/holdout nào thay đổi. Không chạy
> `git add/commit/push`; thay đổi nằm trong working tree.

### 7.1. F1–F5 = FIXED

| Fix | Finding | Nội dung đã sửa | Lệnh / bằng chứng | Kết quả |
|---|---|---|---|---|
| F1 | CAT-02 (MAJOR) | `scripts/holdout_stage2_train.py` và `scripts/thread_count_sensitivity.py` import `MODEL_PARAMS` từ `src/citd_ml/training/train_catboost.py`; stage 2 dùng `PARAMS = dict(MODEL_PARAMS)`, thí nghiệm dùng `params_for(thread_count)` | `.venv/bin/python scripts/thread_count_sensitivity.py` (đã chạy lại trước đó); `.venv/bin/python scripts/holdout_stage2_train.py --run-id {1,2}` | Một nguồn tham số duy nhất; `conclusion_facts` của thí nghiệm không đổi; `train_run*_report.json` ghi đủ `allow_writing_files` |
| F2 | CAT-01 (MINOR) | Thêm `"allow_writing_files": False` vào `MODEL_PARAMS`; xóa `catboost_info/` ở gốc repo | Toàn bộ lượt chạy ở mục 7.2 | Không lượt chạy nào tái tạo `catboost_info/` |
| F3 | MPL-01 (MINOR) | `matplotlib==3.11.2` trong `pyproject.toml`, `requirements.txt` và `uv.lock`; stage 2 ghi `versions.matplotlib`; stage 5 ghi `environment.matplotlib`; `holdout_evidence.py` bắt buộc key `matplotlib` | `train_run1_report.json` → `"matplotlib": "3.11.2"`; `stage5_repro_report.json` → `"matplotlib": "3.11.2"`; báo cáo Stage 4 in "Matplotlib 3.11.2" | Dependency được khai báo/pin và phiên bản nằm trong evidence |
| F4 | MPL-02 (NIT) | `figure.savefig(..., metadata={"Software": None})` trong `scripts/holdout_stage4_report.py` | `PIL.Image.open(...).info.get("Software")` → `None` cho cả 4 PNG Stage 4 | PNG không còn tag phiên bản matplotlib; kích thước đổi đúng dự kiến (94,055 → 93,985 B; 81,513 → 81,443 B), bảng số không đổi |
| F5 | PLT-01 (NIT) | `include_plotlyjs=True` thay cho `"inline"` trong `scripts/holdout_stage4_report.py` | `.venv/bin/python scripts/holdout_stage5_repro_check.py` → cả 4 biểu đồ `bytes_identical = true` | Khớp danh sách giá trị docs; hành vi/output không đổi |

### 7.2. Bằng chứng làm mới (không đổi số)

- **Stage 2 — byte-identity prediction:** copy `train_predictions_run{1,2}.npy` sang
  `/tmp/opencode/audit_finish/stage2_before/` trước khi chạy; chạy
  `.venv/bin/python scripts/holdout_stage2_train.py --run-id 1`, `--run-id 2`,
  `--verify`. `cmp` trước/sau và run1-vs-run2: **IDENTICAL**; SHA-256 cả trước
  lẫn sau đều là `2b5d473c35036aa979a033ecdf4355b018c62362b70092d7055b1c6844785e8a`.
  `.cbm` đổi hash (`09ac7b61…/3b6db983…` → `90a6c179…/00dc6f07…`) — đúng ghi
  nhận D3, không phải tiêu chí PASS.
- **Stage 3:** chạy lại `.venv/bin/python scripts/holdout_stage3_backtest.py` và
  `.venv/bin/python scripts/holdout_stage3_backtest.py --run-id 2 --out-dir outputs/holdout/repro/run2/stage3`;
  `git diff --stat -- outputs/holdout/stage3 outputs/holdout/repro/run2/stage3` **rỗng**.
- **Stage 4:** chạy lại cả hai nhánh (mặc định + `--stage3-dir outputs/holdout/repro/run2/stage3 --out-dir outputs/holdout/repro/run2/stage4 --report outputs/holdout/repro/run2/BAO_CAO_holdout_run2.md`);
  hai báo cáo nay liệt kê `Matplotlib 3.11.2` và `allow_writing_files=False`;
  bảng số giữ nguyên, chỉ kích thước/hash file đổi (PNG, `.cbm`, `train_run*_report.json`).
- **Stage 5:** `.venv/bin/python scripts/holdout_stage5_repro_check.py` → **PASS**;
  `probability_max_abs_diff = 0.0`, `summary_max_abs_diff` toàn bộ `0.0`,
  `classification_abs_diff` `0.0`, `top_k_all_identical = true`,
  `stage4_tables_exact = true`, `charts_bytes_identical = true`.
- **`verify_pipeline` canonical:**
  `.venv/bin/python scripts/verify_pipeline.py --train-dir outputs/step4_thread1/catboost_training --backtest-dir outputs/step4_thread1/backtest --manifest outputs/step4_thread1/verification/reproducibility.json`
  → **PASS**; `allow_writing_files=False` không làm đổi byte CSV (8/8 bảng
  "PASS exact repeat + saved CSV", hai lượt train/backtest khớp file đang báo
  cáo); chỉ manifest JSON đổi (`allow_writing_files`, `source_sha256` của các
  script đã sửa, timestamp).
- **pytest:** `.venv/bin/python -m pytest -q` → **13 passed**.
- **`catboost_info/`:** xóa trước khi chạy; kiểm tra sau từng lệnh Stage 2/3/4/5
  và `verify_pipeline` → không tái tạo.

### 7.3. CAT-03 / SKL-01 / PND-01 = giữ nguyên có chủ đích

| ID | Lý do giữ nguyên (một dòng) |
|---|---|
| CAT-03 | Predict dùng mặc định `thread_count=-1`: docs nói không đổi kết quả và thí nghiệm committed đo được `prediction_thread_count_changed_predictions = false`; ghim thêm không tạo thêm bằng chứng. |
| SKL-01 | Không thêm `zero_division` vào `f1_score`: mỗi fold đã validate đủ hai lớp nên không có cảnh báo thực tế; ngưỡng 0.5 là quyết định đã ghi tài liệu. |
| PND-01 | `pre_train.py` vẫn convert ngày sau `read_csv`: hành vi tương đương `parse_dates`, sửa chỉ để đồng bộ style sẽ đổi hash nguồn canonical mà không thêm giá trị bằng chứng. |
