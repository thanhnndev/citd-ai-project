# Bảng 1 — Chỉ số phân loại

**Khối chính — 4 cách chia chạy lại với `thread_count=1` (cây canonical):**

| Cách chia | ROC-AUC | F1 @0.5 |
|---|---|---|
| Cách 1 — Random K-Fold | 0.8582 | 0.6494 |
| Cách 1b — Grouped K-Fold | 0.7454 | 0.5077 |
| Cách 2 — Walk-forward | 0.5875 | 0.3288 |
| Cách 3 — WF + Purge/Embargo | 0.5895 | 0.3247 |
| Holdout | 0.6046 | 0.4022 |

**Khối tham chiếu — bàn giao (không ghim `thread_count`):**

| Cách chia | ROC-AUC | F1 @0.5 |
|---|---|---|
| Cách 1 — Random K-Fold | 0.8595 | 0.6525 |
| Cách 1b — Grouped K-Fold | 0.7475 | 0.5105 |
| Cách 2 — Walk-forward | 0.5867 | 0.3267 |
| Cách 3 — WF + Purge/Embargo | 0.5948 | 0.3304 |

**Chênh lệch canonical − bàn giao (tính trên giá trị chưa làm tròn):**

| Cách chia | ΔROC-AUC | ΔF1 @0.5 |
|---|---|---|
| Cách 1 — Random K-Fold | -0.0013 | -0.0031 |
| Cách 1b — Grouped K-Fold | -0.0021 | -0.0028 |
| Cách 2 — Walk-forward | +0.0007 | +0.0020 |
| Cách 3 — WF + Purge/Embargo | -0.0052 | -0.0057 |


# Bảng 2 — Chỉ số tài chính, giữ top 50%

**Khối chính — 4 cách chia trên khúc 2–5, giữ top 50%, `thread_count=1`:**

|  | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---|---|---|---|---|
| Baseline (khúc 2–5) | 20,007 | +3,358.25 | 236.13 | 1.2940 | 31.71 |
| Cách 1 — Random K-Fold | 10,004 | +6,937.53 | 73.76 | 2.5437 | 44.65 |
| Cách 1b — Grouped K-Fold | 10,004 | +4,752.04 | 99.71 | 1.9500 | 39.26 |
| Cách 2 — Walk-forward | 10,004 | +2,148.31 | 211.29 | 1.4117 | 34.76 |
| Cách 3 — WF + Purge/Embargo | 10,004 | +2,119.48 | 142.36 | 1.4074 | 34.99 |

**Khối tham chiếu — bàn giao (không ghim `thread_count`):**

|  | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---|---|---|---|---|
| Baseline (khúc 2–5) | 20,007 | +3,358.25 | 236.13 | 1.2940 | 31.71 |
| Cách 1 — Random K-Fold | 10,004 | +6,998.30 | 75.71 | 2.5627 | 44.88 |
| Cách 1b — Grouped K-Fold | 10,004 | +4,724.34 | 99.50 | 1.9418 | 39.24 |
| Cách 2 — Walk-forward | 10,004 | +2,207.25 | 205.31 | 1.4226 | 34.86 |
| Cách 3 — WF + Purge/Embargo | 10,004 | +2,253.72 | 154.11 | 1.4350 | 35.17 |

**Khối holdout — giữ top 50%:**

|  | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---|---|---|---|---|
| Baseline holdout | 5,028 | +245.93 | 305.32 | 1.0992 | 35.28 |
| Holdout, top 50% | 2,514 | -36.55 | 263.25 | 0.9718 | 34.81 |


# Bảng 3 — Sweep 20–80% của 4 cách chia (thread_count=1)

Baseline (không lọc, khúc 2–5): 20,007 lệnh, +3,358.25 R, MaxDD 236.13 R, PF 1.2940, win rate 31.71%.

| Cách chia | Mức giữ | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---|---|---|---|---|---|
| Cách 1 — Random K-Fold | Top 20% | 4,002 | +5,248.16 | 13.49 | 6.3257 | 64.82 |
| Cách 1 — Random K-Fold | Top 30% | 6,003 | +6,640.38 | 23.91 | 4.4036 | 57.02 |
| Cách 1 — Random K-Fold | Top 40% | 8,003 | +7,136.31 | 34.59 | 3.2781 | 50.58 |
| Cách 1 — Random K-Fold | Top 50% | 10,004 | +6,937.53 | 73.76 | 2.5437 | 44.65 |
| Cách 1 — Random K-Fold | Top 60% | 12,005 | +6,451.10 | 110.82 | 2.0872 | 40.32 |
| Cách 1 — Random K-Fold | Top 70% | 14,005 | +5,800.28 | 136.66 | 1.7880 | 37.14 |
| Cách 1 — Random K-Fold | Top 80% | 16,006 | +4,980.00 | 178.18 | 1.5665 | 34.61 |
| Cách 1b — Grouped K-Fold | Top 20% | 4,002 | +2,803.94 | 33.27 | 2.7979 | 48.23 |
| Cách 1b — Grouped K-Fold | Top 30% | 6,003 | +3,696.16 | 47.29 | 2.4148 | 44.76 |
| Cách 1b — Grouped K-Fold | Top 40% | 8,003 | +4,325.22 | 67.22 | 2.1492 | 41.83 |
| Cách 1b — Grouped K-Fold | Top 50% | 10,004 | +4,752.04 | 99.71 | 1.9500 | 39.26 |
| Cách 1b — Grouped K-Fold | Top 60% | 12,005 | +4,772.24 | 150.80 | 1.7581 | 37.08 |
| Cách 1b — Grouped K-Fold | Top 70% | 14,005 | +4,521.23 | 191.83 | 1.5925 | 35.08 |
| Cách 1b — Grouped K-Fold | Top 80% | 16,006 | +4,297.70 | 193.95 | 1.4817 | 33.63 |
| Cách 2 — Walk-forward | Top 20% | 4,002 | +1,056.60 | 103.86 | 1.5463 | 37.21 |
| Cách 2 — Walk-forward | Top 30% | 6,003 | +1,327.33 | 159.63 | 1.4409 | 36.27 |
| Cách 2 — Walk-forward | Top 40% | 8,003 | +1,695.46 | 157.93 | 1.4115 | 35.34 |
| Cách 2 — Walk-forward | Top 50% | 10,004 | +2,148.31 | 211.29 | 1.4117 | 34.76 |
| Cách 2 — Walk-forward | Top 60% | 12,005 | +2,575.06 | 240.18 | 1.4051 | 34.37 |
| Cách 2 — Walk-forward | Top 70% | 14,005 | +2,760.58 | 283.51 | 1.3664 | 33.75 |
| Cách 2 — Walk-forward | Top 80% | 16,006 | +3,120.69 | 273.52 | 1.3581 | 33.50 |
| Cách 3 — WF + Purge/Embargo | Top 20% | 4,002 | +1,266.26 | 86.49 | 1.6598 | 38.26 |
| Cách 3 — WF + Purge/Embargo | Top 30% | 6,003 | +1,381.02 | 149.36 | 1.4623 | 36.40 |
| Cách 3 — WF + Purge/Embargo | Top 40% | 8,003 | +1,674.66 | 144.96 | 1.4107 | 35.67 |
| Cách 3 — WF + Purge/Embargo | Top 50% | 10,004 | +2,119.48 | 142.36 | 1.4074 | 34.99 |
| Cách 3 — WF + Purge/Embargo | Top 60% | 12,005 | +2,323.85 | 208.79 | 1.3642 | 34.12 |
| Cách 3 — WF + Purge/Embargo | Top 70% | 14,005 | +2,688.96 | 221.19 | 1.3578 | 33.77 |
| Cách 3 — WF + Purge/Embargo | Top 80% | 16,006 | +3,061.29 | 240.51 | 1.3510 | 33.38 |


# Bảng 4 — Sweep holdout 20–80%

| Lọc | Số lệnh | Net profit (R) | MaxDD (R) | Profit factor | Win rate % |
|---|---|---|---|---|---|
| Top 20% | 1,006 | -30.01 | 125.38 | 0.9385 | 36.18 |
| Top 30% | 1,509 | -50.01 | 210.77 | 0.9343 | 35.52 |
| Top 40% | 2,012 | -53.17 | 251.56 | 0.9485 | 35.04 |
| Top 50% | 2,514 | -36.55 | 263.25 | 0.9718 | 34.81 |
| Top 60% | 3,017 | +73.16 | 233.95 | 1.0474 | 35.60 |
| Top 70% | 3,520 | +186.92 | 197.24 | 1.1040 | 35.77 |
| Top 80% | 4,023 | +259.91 | 219.02 | 1.1278 | 35.52 |
