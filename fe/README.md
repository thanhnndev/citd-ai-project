# CITD ML — frontend demo offline

Bản frontend này chạy độc lập bằng Vite. Nó **không gọi FastAPI `/api/*`**, không
cần model `.cbm`, không cần file raw và không đọc `outputs/` lúc chạy.

## Chạy demo

Từ thư mục `fe/`:

```bash
npm ci
npm run dev
```

Mở URL Vite in ra (mặc định `http://127.0.0.1:5173`). Build tĩnh:

```bash
npm run build
npm run preview
```

`package-lock.json` đang dùng Vite 8, vì vậy Node.js cần đáp ứng engine của
Vite (`^20.19.0` hoặc `>=22.12.0`).

## Dữ liệu nào là số thật?

Các số canonical được chép tĩnh từ artifact đã commit của repository:

- `outputs/holdout/stage3/stage3_report.json`
- `outputs/holdout/stage3/stage3_backtest_summary.csv`
- `outputs/holdout/stage3/holdout_fixed_trade_universe_scored.csv`
- `outputs/holdout/stage4/table1_classification_metrics.csv`
- `outputs/holdout/repro/stage5_repro_report.json`

Bản demo dùng 24 dòng holdout đầu để hiển thị `entry`, `exit`, `R`, xác suất
CatBoost đã chấm sẵn và một số feature thật. Holdout đầy đủ vẫn là **5.028 dòng**;
UI không cố tải toàn bộ file vào browser. Feature importance toàn model được lấy
từ báo cáo kỹ thuật đã commit.

## Phần nào là mock/offline?

- **Không có CatBoost live**: `predictSignal` chỉ trả về trạng thái
  `available: false`; UI dùng xác suất precomputed trong artifact.
- **OHLC minh họa**: nến được suy ra từ `entry`/`exit` của các dòng holdout để
  vẽ giao diện. Đây không phải nến M15 nguyên bản và không phải dữ liệu realtime.
- **Đường vốn minh họa**: có điểm cuối khớp `net_profit_R` của từng mức top-k,
  nhưng đường giữa các điểm là nội suy trình duyệt. KPI và bảng sweep mới là số
  canonical.
- **ROC minh họa**: AUC/F1 là số canonical; đường ROC được nội suy từ AUC để
  minh họa, không phải raw OOF ROC curve.
- **TP/SL minh họa**: vùng `+2R/-1R` chỉ để giải thích giao diện, không phải
  dynamic barrier hay quyết định đặt lệnh.
- **Không mô phỏng** phí, spread, slippage, trạng thái tài khoản, đặt lệnh,
  execution hay dữ liệu thị trường realtime.

## Backend cũ

Backend FastAPI vẫn tồn tại trong `be/`, với các route được khai báo trong
`be/app/api/endpoints.py` và prefix `/api` trong `be/app/main.py`:
`/health`, `/market/sample`, `/predict`, `/features/importances`,
`/backtest/summary`, `/backtest/equity-curve`, `/splits/comparison`.

Trước khi triển khai bản offline này, `src/api.js` gọi toàn bộ route trên và
Vite proxy `/api` tới `127.0.0.1:8000`; vì vậy demo cũ không chạy độc lập. Bản
hiện tại đã bỏ proxy và thay bằng adapter tĩnh trong `src/api.js` + `src/demo-data.js`.
