# GHI CHÚ CHUẨN BỊ BẢO VỆ — NHÓM 11

> Tài liệu ngắn để tra cứu trước khi thầy hỏi. Ưu tiên câu trả lời theo **code và artifact canonical**, không dùng slide/demo cũ.

## 1. Bản chất đồ án

CatBoost là **bộ lọc tín hiệu**, không phải mô hình tự tạo tín hiệu mua/bán:

```text
Pyramid tạo lệnh ứng viên → CatBoost chấm điểm → giữ/loại lệnh → backtest
```

Mục tiêu chính là đo **độ nhạy của phương pháp kiểm định** khi dữ liệu có họ lệnh và nhãn chồng lấn; không phải chứng minh CatBoost sinh lợi nhuận.

## 2. Các số phải nhớ

| Hạng mục | Giá trị |
|---|---:|
| Mốc holdout | `2025-02-08 15:30:00` / M15 bar `199.968` |
| Train đóng băng | 25.008 lệnh = 6.252 họ × 4 leg |
| Phạm vi so sánh chung | Khúc 2–5 = 20.007 lệnh |
| Holdout | 5.028 lệnh |
| Lệnh biên bị loại | 4 |
| Feature / metadata | 23 / 7 |
| Holdout ROC-AUC / F1 | 0,6046 / 0,4022 |
| Baseline holdout | +245,93 R, PF 1,0992 |
| Holdout top 50% | −36,55 R, PF 0,9718 |
| Purge + embargo ở 4 fold phát triển | 21 + 3 + 8 + 2 = 34 dòng |
| Purge + embargo ở final train | 0 dòng |

AUC bốn nhánh trên khúc 2–5: `0,8582 → 0,7454 → 0,5875 → 0,5895`.

## 3. Câu trả lời cốt lõi

> “Đồ án giữ nguyên dữ liệu, 23 feature, nhãn và cấu hình CatBoost, chỉ thay đổi cách chia train/test. Random K-Fold có thể tách các leg cùng `origin_bar` và đảo chiều thời gian, nên AUC 0,8582 bị lạc quan. Grouped K-Fold khóa family nhưng vẫn trộn thời gian; Walk-forward giữ đúng chiều thời gian; Purged Walk-forward thêm kiểm tra biên. Hai cách theo thời gian gần holdout hơn, nhưng top 50% vẫn làm xấu kết quả tài chính.”

## 4. Các điểm phải nói thật

- Code hiện tại của chiến lược là **long-only**; base signal mở leg 0 và schedule leg 1–3 cho ba bar M15 kế tiếp.
- Hàm nhãn hiện là **fixed barrier**: `upper = entry + 3,8 × ATR(90)`, `lower = entry × 0,994`, horizon 50 M15. Nó không replay dynamic trailing stop.
- Target thật của strategy là 33,7%, trailing stop là `peak − 3,8 × ATR`, và có Friday exit; vì vậy `label=1` không bảo đảm R thực tế dương.
- Top 50% được chọn từ toàn bộ kỳ đánh giá: đây là **offline ranking**, chưa phải quy tắc live biết trước tương lai.
- Holdout đã được mở lại trong các kiểm tra kỹ thuật và thí nghiệm `thread_count`; không nên nói “chỉ mở đúng một lần”.
- Reproducibility PASS là trên **cùng máy/cùng code**; chưa chứng minh byte-identical giữa các máy.
- Backtest không có phí, spread, slippage, capital constraint hoặc mark-to-market drawdown.

## 5. Số phải trích dẫn

- Tổng quan và kết luận: [`TECHNICAL_REPORT_NHOM11.md`](TECHNICAL_REPORT_NHOM11.md).
- Bối cảnh, cách chạy và giới hạn: [`README.md`](../README.md).
- Số liệu và artifact holdout: [`BAO_CAO_KET_QUA_HOLDOUT.md`](BAO_CAO_KET_QUA_HOLDOUT.md).
- Chiến lược: [`pyramid_strategy.py`](../src/citd_ml/strategy/pyramid_strategy.py).
- Nhãn: [`triple_barrier.py`](../src/citd_ml/labeling/triple_barrier.py).
- Feature: [`build_features.py`](../src/citd_ml/features/build_features.py).
- Split: [`split_data.py`](../src/citd_ml/training/split_data.py).
- CatBoost: [`train_catboost.py`](../src/citd_ml/training/train_catboost.py).
- Lịch sử mở holdout: [`holdout_run_history.md`](../outputs/verification/holdout_run_history.md).
- Kết quả kiểm lặp: [`stage5_repro_report.json`](../outputs/holdout/repro/stage5_repro_report.json).

## 6. Cấu trúc câu trả lời khi bị hỏi

1. **Định nghĩa đối tượng**: CatBoost lọc lệnh do Pyramid tạo.
2. **Nêu cơ chế**: family overlap, nhãn 50 bar, đảo chiều thời gian.
3. **Nêu cách kiểm soát**: Grouped → Walk-forward → Purge/Embargo.
4. **Nêu số**: `0,8582`, `0,5895`, holdout `0,6046`.
5. **Nêu giới hạn**: holdout top 50% `−36,55 R`; chưa chứng minh lợi nhuận/live.
