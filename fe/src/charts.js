import { Chart, registerables } from 'chart.js';
Chart.register(...registerables);

/**
 * Authentic Japanese Candlestick & Triple-Barrier Canvas Renderer
 */
export class CandlestickRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.candles = [];
    this.signals = [];
    this.selectedSignal = null;
    this.setupResize();
  }

  setupResize() {
    const resize = () => {
      if (!this.canvas) return;
      const rect = this.canvas.getBoundingClientRect();
      this.canvas.width = rect.width * (window.devicePixelRatio || 1);
      this.canvas.height = rect.height * (window.devicePixelRatio || 1);
      this.ctx.resetTransform();
      this.ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
      this.render();
    };
    window.addEventListener('resize', resize);
    setTimeout(resize, 40);
  }

  setData(candles, signals) {
    this.candles = candles;
    this.signals = signals;
    if (signals.length > 0 && !this.selectedSignal) {
      this.selectedSignal = signals[0];
    }
    this.render();
  }

  setSelectedSignal(signal) {
    this.selectedSignal = signal;
    this.render();
  }

  render() {
    const ctx = this.ctx;
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    ctx.clearRect(0, 0, width, height);
    if (!this.candles || this.candles.length === 0) return;

    // Price scaling
    let minP = Infinity;
    let maxP = -Infinity;
    let maxVol = 0;
    for (const c of this.candles) {
      if (c.low < minP) minP = c.low;
      if (c.high > maxP) maxP = c.high;
      if (c.volume > maxVol) maxVol = c.volume;
    }

    const pad = (maxP - minP) * 0.18;
    minP -= pad;
    maxP += pad;

    const chartBottom = height - 50; // Space for volume bars
    const priceToY = (p) => chartBottom - ((p - minP) / (maxP - minP)) * (chartBottom - 30);
    const indexToX = (i) => 25 + (i / (this.candles.length - 1)) * (width - 90);

    // Subtle horizontal price gridlines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 6; i++) {
      const p = minP + (i / 6) * (maxP - minP);
      const y = priceToY(p);
      ctx.beginPath();
      ctx.moveTo(20, y);
      ctx.lineTo(width - 70, y);
      ctx.stroke();

      // Price labels on right
      ctx.fillStyle = '#64748B';
      ctx.font = '10px Inter, sans-serif';
      ctx.fillText(p.toFixed(0), width - 62, y + 3);
    }

    const candleW = Math.max(3, ((width - 90) / this.candles.length) * 0.72);

    // Draw Volume Bars at bottom
    this.candles.forEach((c, i) => {
      const x = indexToX(i);
      const vH = (c.volume / (maxVol || 1)) * 38;
      const isUp = c.close >= c.open;
      ctx.fillStyle = isUp ? 'rgba(16, 185, 129, 0.25)' : 'rgba(239, 68, 68, 0.25)';
      ctx.fillRect(x - candleW / 2, height - vH - 6, candleW, vH);
    });

    // Draw Candlesticks (Wick + Body)
    this.candles.forEach((c, i) => {
      const x = indexToX(i);
      const openY = priceToY(c.open);
      const closeY = priceToY(c.close);
      const highY = priceToY(c.high);
      const lowY = priceToY(c.low);
      const isUp = c.close >= c.open;
      const col = isUp ? '#38BDF8' : '#64748B'; // Aesthetic slate/cyan candles matching mockup

      ctx.strokeStyle = col;
      ctx.lineWidth = 1.2;

      // Upper & Lower Wick
      ctx.beginPath();
      ctx.moveTo(x, highY);
      ctx.lineTo(x, lowY);
      ctx.stroke();

      // Body
      const bodyY = Math.min(openY, closeY);
      const bodyH = Math.max(2, Math.abs(closeY - openY));
      ctx.fillStyle = isUp ? '#1E293B' : col; // Hollow style for bullish, filled for bearish
      ctx.fillRect(x - candleW / 2, bodyY, candleW, bodyH);
      ctx.strokeRect(x - candleW / 2, bodyY, candleW, bodyH);
    });

    // Draw Triple-Barrier for Selected Signal
    if (this.selectedSignal) {
      const s = this.selectedSignal;
      const sX = indexToX(s.index);
      const tpY = priceToY(s.tp);
      const slY = priceToY(s.sl);
      const endX = Math.min(width - 70, indexToX(s.index + s.max_bars));

      // Upper Barrier (Take Profit +2 ATR - Green dashed line)
      ctx.save();
      ctx.setLineDash([6, 5]);
      ctx.strokeStyle = '#10B981';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sX, tpY);
      ctx.lineTo(endX, tpY);
      ctx.stroke();

      ctx.fillStyle = '#10B981';
      ctx.font = 'bold 11px Inter';
      ctx.fillText(`Take Profit: $${s.tp.toFixed(0)}`, endX - 120, tpY - 6);

      // Lower Barrier (Stop Loss -1 ATR - Red dashed line)
      ctx.strokeStyle = '#EF4444';
      ctx.beginPath();
      ctx.moveTo(sX, slY);
      ctx.lineTo(endX, slY);
      ctx.stroke();

      ctx.fillStyle = '#EF4444';
      ctx.fillText(`Stop Loss: $${s.sl.toFixed(0)}`, endX - 110, slY + 14);

      // Vertical Time Barrier (24 bars)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(endX, tpY);
      ctx.lineTo(endX, slY);
      ctx.stroke();
      ctx.restore();

      // Glow effect around selected entry signal
      ctx.save();
      ctx.shadowColor = s.decision === 'PASS' ? '#10B981' : '#EF4444';
      ctx.shadowBlur = 12;
      ctx.strokeStyle = s.decision === 'PASS' ? '#10B981' : '#EF4444';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(sX, priceToY(s.price), 8, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    // Draw Entry Signal Dots (Green for Pass, Red/Orange for Skip)
    this.signals.forEach((s) => {
      const x = indexToX(s.index);
      const y = priceToY(s.price);

      ctx.save();
      ctx.fillStyle = s.decision === 'PASS' ? '#10B981' : '#EF4444';
      ctx.shadowColor = s.decision === 'PASS' ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)';
      ctx.shadowBlur = 6;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });
  }
}

/**
 * Chart.js Equity Curve Manager with Glowing Neon Line
 */
export class EquityCurveChart {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.chart = null;
  }

  render(data) {
    if (this.chart) {
      this.chart.destroy();
    }

    const ctx = this.canvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 200);
    gradient.addColorStop(0, 'rgba(56, 189, 248, 0.35)');
    gradient.addColorStop(1, 'rgba(56, 189, 248, 0.0)');

    this.chart = new Chart(this.canvas, {
      type: 'line',
      data: {
        labels: data.times,
        datasets: [
          {
            label: `CatBoost Filtered (Top ${data.keep_pct}%)`,
            data: data.filtered,
            borderColor: '#38BDF8',
            backgroundColor: gradient,
            fill: true,
            borderWidth: 2.2,
            pointRadius: 0,
            tension: 0.25
          },
          {
            label: 'Baseline (Chiến lược gốc)',
            data: data.baseline,
            borderColor: '#64748B',
            borderWidth: 1.4,
            pointRadius: 0,
            tension: 0.15,
            borderDash: [4, 4]
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        plugins: {
          legend: {
            position: 'top',
            labels: { color: '#94A3B8', font: { family: 'Inter', size: 11 }, boxWidth: 10 }
          },
          tooltip: {
            backgroundColor: '#10192D',
            borderColor: 'rgba(255, 255, 255, 0.1)',
            borderWidth: 1,
            titleColor: '#FFF',
            bodyColor: '#38BDF8'
          }
        },
        scales: {
          x: { display: false },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.04)' },
            ticks: { color: '#64748B', font: { size: 10 }, callback: (v) => `${v} R` }
          }
        }
      }
    });
  }
}

/**
 * Chart.js ROC Curves Manager
 */
export class RocChart {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.chart = null;
  }

  render(rocData) {
    if (this.chart) {
      this.chart.destroy();
    }

    const datasets = rocData.map((d) => ({
      label: `${d.name} (AUC = ${d.auc.toFixed(3)})`,
      data: d.points.map((p) => ({ x: p.fpr, y: p.tpr })),
      borderColor: d.color,
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.2
    }));

    // Diagonal
    datasets.push({
      label: 'Đoán ngẫu nhiên (AUC = 0.500)',
      data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      borderColor: '#64748B',
      borderDash: [5, 5],
      borderWidth: 1.5,
      pointRadius: 0
    });

    this.chart = new Chart(this.canvas, {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'right',
            labels: { color: '#94A3B8', font: { family: 'Inter', size: 11 }, boxWidth: 10 }
          }
        },
        scales: {
          x: {
            type: 'linear', min: 0, max: 1,
            title: { display: true, text: 'False Positive Rate (Báo động giả)', color: '#94A3B8' },
            grid: { color: 'rgba(255, 255, 255, 0.04)' },
            ticks: { color: '#94A3B8' }
          },
          y: {
            min: 0, max: 1,
            title: { display: true, text: 'True Positive Rate (Bắt đúng)', color: '#94A3B8' },
            grid: { color: 'rgba(255, 255, 255, 0.04)' },
            ticks: { color: '#94A3B8' }
          }
        }
      }
    });
  }
}
