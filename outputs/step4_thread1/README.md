# Rerun canonical thread_count=1 (bốn split)

Cây output này là bản chạy lại **canonical** của pipeline bốn split
(Random / Grouped / Walk-forward / Purged Walk-forward) với `thread_count=1`,
dùng để đối chiếu số liệu trong báo cáo. Đây **không phải** thư mục bàn giao:
hai thư mục đã đóng băng `outputs/catboost_training/` và `outputs/backtest/`
không bị ghi đè.

## Cấu hình canonical

- `MODEL_PARAMS` trong `src/citd_ml/training/train_catboost.py` giữ
  `thread_count=1`, `random_seed=42`; dữ liệu, feature, cách chia split và
  tham số mô hình giữ nguyên.
- Chỉ thư mục output được đổi qua cờ dòng lệnh (`--output-dir`, `--oof-dir`).

## Lệnh tái lập

```bash
.venv/bin/python scripts/train_models.py --output-dir outputs/step4_thread1/catboost_training
.venv/bin/python scripts/run_backtest.py --output-dir outputs/step4_thread1/backtest --oof-dir outputs/step4_thread1/catboost_training
```

Không chạy hai lệnh trên mà thiếu `--output-dir` / `--oof-dir`, vì khi đó
pipeline sẽ ghi vào artifacts bàn giao.

## Nội dung

- `catboost_training/`: `metrics_by_fold.csv`, `metrics_summary.csv`,
  4 file `oof_*.csv` và `metrics_chunk2_5.csv` (ROC-AUC/F1 chỉ tính chunk 2-5,
  làm tròn 4 chữ số thập phân).
- `backtest/`: `backtest_summary_top50.csv`, `backtest_retention_sweep.csv`,
  `backtest_scored_universe.csv`, `baseline_tradelist.csv` và 4 file
  `trades_top50_*.csv`.
- `comparison_report.json`: so khớp từng file với bàn giao đã đóng băng và với
  commit `7748828` (lần chạy `thread_count=1` ngày 2026-09-12), kèm hash,
  phiên bản môi trường và lệnh đã dùng.

## Kết quả chính

Phân loại chunk 2-5 (trung bình theo fold, xem `metrics_chunk2_5.csv`):

| method | ROC-AUC | F1@0.5 |
|---|---|---|
| random_kfold | 0.8582 | 0.6494 |
| grouped_kfold | 0.7454 | 0.5077 |
| walk_forward | 0.5875 | 0.3288 |
| purged_walk_forward | 0.5895 | 0.3247 |

Tài chính chunk 2-5, giữ top 50%:

| method | net R | max DD (R) | profit factor | win rate (%) |
|---|---|---|---|---|
| baseline | +3358.25107471 | 236.126175 | 1.29397333 | 31.713900 |
| random_kfold | +6937.53476724 | 73.764787 | 2.54373656 | 44.652139 |
| grouped_kfold | +4752.04217114 | 99.709315 | 1.95000124 | 39.264294 |
| walk_forward | +2148.30521167 | 211.293790 | 1.41165125 | 34.756098 |
| purged_walk_forward | +2119.47819179 | 142.364326 | 1.40735113 | 34.986006 |

Cả 8 file được đối chiếu (metrics + OOF + hai bảng backtest) **giống hệt từng
byte** với commit `7748828`; so với bàn giao thì chênh lệch net R top 50% là
−60.77 / +27.70 / −58.94 / −134.24 R cho bốn nhánh theo thứ tự trên. Chi tiết
đầy đủ nằm trong `comparison_report.json`.
