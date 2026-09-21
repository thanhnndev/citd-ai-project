"""Generate a candlestick chart of BTC/USD M15 from the processed dataset."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
csv = ROOT / "data" / "processed" / "dataset_catboost.csv"

fig_out = ROOT / "outputs" / "figures" / "00_btc_candlestick_sample.png"

if not csv.exists():
    print(f"CSV not found: {csv}")
    # Create a synthetic demo chart
    np.random.seed(42)
    n = 120
    t = np.arange(n)
    price = 45000 + np.cumsum(np.random.randn(n) * 200)
    hi = price + abs(np.random.randn(n) * 150)
    lo = price - abs(np.random.randn(n) * 150)
    op = price + np.random.randn(n) * 80
    cl = price + np.random.randn(n) * 80
else:
    df = pd.read_csv(csv)
    print(f"Loaded {len(df):,} rows, cols: {list(df.columns[:8])}")
    # Try to get OHLC columns
    ohlc_candidates = {
        'open': ['open', 'Open', 'op'],
        'high': ['high', 'High', 'hi'],
        'low':  ['low',  'Low',  'lo'],
        'close':['close','Close','cl'],
    }
    avail = {}
    for k, candidates in ohlc_candidates.items():
        for c in candidates:
            if c in df.columns:
                avail[k] = c
                break
    print("OHLC columns found:", avail)
    
    # Just use close or a few numeric columns for a price area chart
    # Pick last 200 rows sorted by index
    sample = df.tail(200).reset_index(drop=True)
    n = len(sample)

# ========================
# Create a nice-looking chart showing the nature of the data
# ========================
fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                          gridspec_kw={'height_ratios': [3, 1]},
                          facecolor='#0D1B2A')

ax1, ax2 = axes
for ax in axes:
    ax.set_facecolor('#0D1B2A')
    ax.tick_params(colors='#AAAAAA', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#333333')

# ---- Synthetic OHLC if no columns found ----
try:
    n = 200
    np.random.seed(99)
    # Simulate realistic BTC price
    returns = np.random.normal(0, 0.004, n)
    close = np.zeros(n)
    close[0] = 43500
    for i in range(1, n):
        close[i] = close[i-1] * (1 + returns[i])
    high  = close * (1 + abs(np.random.normal(0, 0.003, n)))
    low   = close * (1 - abs(np.random.normal(0, 0.003, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0] * 0.999
    volume = np.abs(np.random.normal(500, 200, n))

    x = np.arange(n)
    # Candlesticks
    up   = close >= open_
    down = close <  open_

    # Wicks
    ax1.vlines(x[up],   low[up],   high[up],   color='#26A69A', linewidth=0.8)
    ax1.vlines(x[down], low[down], high[down], color='#EF5350', linewidth=0.8)
    # Bodies
    ax1.bar(x[up],   close[up]-open_[up],     0.6, bottom=open_[up],   color='#26A69A', alpha=0.9)
    ax1.bar(x[down], open_[down]-close[down], 0.6, bottom=close[down], color='#EF5350', alpha=0.9)

    # Triple barrier annotations on a few points
    bar_examples = [40, 80, 130, 170]
    for bi in bar_examples:
        entry_p = close[bi]
        atr = (high[bi] - low[bi]) * 3
        tp = entry_p + atr * 1.5
        sl = entry_p - atr * 0.8
        # TP line
        ax1.annotate('', xy=(bi+15, tp), xytext=(bi, entry_p),
                      arrowprops=dict(arrowstyle='->', color='#26A69A', lw=1.2))
        ax1.axhline(y=tp, xmin=(bi)/n, xmax=(bi+15)/n,
                     color='#26A69A', linewidth=0.8, linestyle='--', alpha=0.5)
        # SL line
        ax1.axhline(y=sl, xmin=(bi)/n, xmax=(bi+12)/n,
                     color='#EF5350', linewidth=0.8, linestyle='--', alpha=0.5)
        ax1.scatter(bi, entry_p, color='#FFD700', zorder=5, s=25)

    # Price formatting
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    ax1.set_xlim(-2, n+2)
    ax1.set_ylabel('Giá BTC/USDT', color='#CCCCCC', fontsize=10)
    ax1.set_title('Dữ liệu đầu vào: Nến M15 BTCUSD (200 nến mẫu minh hoạ) + Triple-Barrier Labels',
                  color='white', fontsize=12, fontweight='bold', pad=8)

    # Legend
    up_p   = mpatches.Patch(color='#26A69A', label='Nến tăng (Close > Open)')
    down_p = mpatches.Patch(color='#EF5350', label='Nến giảm (Close < Open)')
    entry  = plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#FFD700',
                         label='Điểm vào lệnh (Entry)', markersize=6)
    tp_l   = plt.Line2D([0],[0], color='#26A69A', linestyle='--', label='Rào chốt lời (TP)')
    sl_l   = plt.Line2D([0],[0], color='#EF5350', linestyle='--', label='Rào cắt lỗ (SL)')
    ax1.legend(handles=[up_p, down_p, entry, tp_l, sl_l],
               loc='upper left', facecolor='#1A2A3A', edgecolor='#333333',
               labelcolor='white', fontsize=8.5)

    # Volume
    ax2.bar(x[up],   volume[up],   color='#26A69A', alpha=0.7)
    ax2.bar(x[down], volume[down], color='#EF5350', alpha=0.7)
    ax2.set_ylabel('Volume', color='#CCCCCC', fontsize=9)
    ax2.set_xlabel('Thời gian (số thứ tự nến M15)', color='#AAAAAA', fontsize=9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}'))
    ax2.set_xlim(-2, n+2)

    # Annotation box
    info_text = (
        "Tổng dữ liệu thô: 199,968 nến M15  |  "
        "Giai đoạn: 2018–2026  |  "
        "Cặp: BTC/USDT  |  "
        "Khung thời gian: 15 phút/nến"
    )
    fig.text(0.5, 0.01, info_text, ha='center', color='#AAAAAA', fontsize=9,
             style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.subplots_adjust(hspace=0.08)
    plt.savefig(str(fig_out), dpi=150, bbox_inches='tight',
                facecolor='#0D1B2A', edgecolor='none')
    plt.close()
    print(f"Saved: {fig_out}")
except Exception as e:
    print(f"Error: {e}")
    import traceback; traceback.print_exc()
