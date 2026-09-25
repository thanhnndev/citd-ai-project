import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

const COLORS = {
  accent: '#2f5c6d',
  accentSoft: 'rgba(47, 92, 109, 0.12)',
  green: '#2f7355',
  red: '#a64b49',
  text: '#25292b',
  muted: '#697271',
  line: '#9aa4a1',
  lineStrong: '#5b6867',
  grid: 'rgba(99, 110, 107, 0.16)',
  volumeUp: 'rgba(47, 92, 109, 0.15)',
  volumeDown: 'rgba(105, 114, 113, 0.14)',
  surface: '#fffefa',
  tooltipBorder: '#d6d5ce',
};

const MONO = 'ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace';

Chart.defaults.color = COLORS.muted;
Chart.defaults.borderColor = COLORS.grid;
Chart.defaults.font.family = MONO;

function formatPrice(value) {
  return `$${Number(value).toLocaleString('vi-VN', { maximumFractionDigits: 0 })}`;
}

/** Canvas renderer for the browser-friendly, trade-event OHLC illustration. */
export class CandlestickRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas?.getContext('2d');
    this.candles = [];
    this.signals = [];
    this.selectedSignal = null;
    this.setupResize();
  }

  setupResize() {
    if (!this.canvas) return;

    const resize = () => {
      const rect = this.canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.round(rect.width));
      const height = Math.max(1, Math.round(rect.height));
      this.canvas.width = width * ratio;
      this.canvas.height = height * ratio;
      this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      this.render(width, height);
    };

    if ('ResizeObserver' in window) {
      this.resizeObserver = new ResizeObserver(resize);
      this.resizeObserver.observe(this.canvas);
    }
    window.addEventListener('resize', resize);
    requestAnimationFrame(resize);
  }

  setData(candles, signals) {
    this.candles = Array.isArray(candles) ? candles : [];
    this.signals = Array.isArray(signals) ? signals : [];
    if (this.signals.length && !this.selectedSignal) this.selectedSignal = this.signals[0];
    this.render();
  }

  setSelectedSignal(signal) {
    this.selectedSignal = signal;
    this.render();
  }

  render(forcedWidth, forcedHeight) {
    if (!this.ctx || !this.candles.length) return;

    const width = forcedWidth || this.canvas.getBoundingClientRect().width;
    const height = forcedHeight || this.canvas.getBoundingClientRect().height;
    if (!width || !height) return;

    const ctx = this.ctx;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = COLORS.surface;
    ctx.fillRect(0, 0, width, height);

    let minPrice = Math.min(...this.candles.map((candle) => candle.low));
    let maxPrice = Math.max(...this.candles.map((candle) => candle.high));
    const selected = this.selectedSignal;
    if (selected) {
      minPrice = Math.min(minPrice, selected.sl);
      maxPrice = Math.max(maxPrice, selected.tp);
    }

    const rawRange = Math.max(1, maxPrice - minPrice);
    minPrice -= rawRange * 0.12;
    maxPrice += rawRange * 0.12;

    const plotLeft = 24;
    const plotRight = width - 68;
    const plotTop = 28;
    const volumeBottom = height - 30;
    const priceBottom = height - 54;
    const plotHeight = Math.max(1, priceBottom - plotTop);
    const priceToY = (price) => priceBottom - ((price - minPrice) / (maxPrice - minPrice)) * plotHeight;
    const indexToX = (index) => plotLeft + (index / Math.max(1, this.candles.length - 1)) * (plotRight - plotLeft);

    // Horizontal grid and right-hand price labels.
    ctx.font = `10px ${MONO}`;
    ctx.textAlign = 'left';
    for (let index = 0; index <= 5; index += 1) {
      const price = minPrice + (index / 5) * (maxPrice - minPrice);
      const y = priceToY(price);
      ctx.beginPath();
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      ctx.moveTo(plotLeft, y);
      ctx.lineTo(plotRight, y);
      ctx.stroke();
      ctx.fillStyle = COLORS.muted;
      ctx.fillText(formatPrice(price), plotRight + 8, y + 3);
    }

    const candleWidth = Math.max(3, Math.min(14, ((plotRight - plotLeft) / this.candles.length) * 0.68));
    const maxVolume = Math.max(...this.candles.map((candle) => candle.volume || 0), 1);

    // Volume is part of the illustration, not a market volume series.
    this.candles.forEach((candle, index) => {
      const x = indexToX(index);
      const volumeHeight = ((candle.volume || 0) / maxVolume) * 24;
      ctx.fillStyle = candle.close >= candle.open ? COLORS.volumeUp : COLORS.volumeDown;
      ctx.fillRect(x - candleWidth / 2, volumeBottom - volumeHeight, candleWidth, volumeHeight);
    });

    this.candles.forEach((candle, index) => {
      const x = indexToX(index);
      const openY = priceToY(candle.open);
      const closeY = priceToY(candle.close);
      const highY = priceToY(candle.high);
      const lowY = priceToY(candle.low);
      const isUp = candle.close >= candle.open;
      const color = isUp ? COLORS.accent : COLORS.muted;

      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, highY);
      ctx.lineTo(x, lowY);
      ctx.stroke();

      const bodyY = Math.min(openY, closeY);
      const bodyHeight = Math.max(2, Math.abs(closeY - openY));
      ctx.fillStyle = isUp ? COLORS.accentSoft : 'rgba(105, 114, 113, 0.12)';
      ctx.fillRect(x - candleWidth / 2, bodyY, candleWidth, bodyHeight);
      ctx.strokeStyle = color;
      ctx.strokeRect(x - candleWidth / 2, bodyY, candleWidth, bodyHeight);
    });

    this.drawSignals(indexToX, priceToY, plotRight, plotTop, priceBottom);
    if (selected) this.drawSelectedSignal(selected, indexToX, priceToY, plotRight, plotTop, priceBottom);

    ctx.fillStyle = COLORS.muted;
    ctx.font = `9px ${MONO}`;
    ctx.textAlign = 'left';
    ctx.fillText('Illustrative trade-event candles', plotLeft, height - 10);
  }

  drawSignals(indexToX, priceToY, plotRight, plotTop, priceBottom) {
    const ctx = this.ctx;
    this.signals.forEach((signal) => {
      const x = indexToX(signal.index);
      const y = priceToY(signal.price);
      const color = signal.decision === 'PASS' ? COLORS.green : COLORS.red;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    });

    if (this.signals.length) {
      ctx.fillStyle = COLORS.muted;
      ctx.font = `9px ${MONO}`;
      ctx.textAlign = 'right';
      ctx.fillText(`${this.signals.length} scored trades`, plotRight, plotTop - 9);
    }
  }

  drawSelectedSignal(signal, indexToX, priceToY, plotRight, plotTop, priceBottom) {
    const ctx = this.ctx;
    const startX = indexToX(signal.index);
    const endX = Math.min(plotRight, indexToX(signal.index + signal.max_bars));
    const tpY = priceToY(signal.tp);
    const slY = priceToY(signal.sl);
    const entryY = priceToY(signal.price);

    ctx.save();
    ctx.setLineDash([5, 5]);
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = COLORS.green;
    ctx.beginPath();
    ctx.moveTo(startX, tpY);
    ctx.lineTo(endX, tpY);
    ctx.stroke();
    ctx.strokeStyle = COLORS.red;
    ctx.beginPath();
    ctx.moveTo(startX, slY);
    ctx.lineTo(endX, slY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = COLORS.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(endX, tpY);
    ctx.lineTo(endX, slY);
    ctx.stroke();
    ctx.restore();

    const color = signal.decision === 'PASS' ? COLORS.green : COLORS.red;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(startX, entryY, 7, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    ctx.font = `9px ${MONO}`;
    ctx.textAlign = 'right';
    ctx.fillStyle = COLORS.green;
    ctx.fillText(`TP ${formatPrice(signal.tp)}`, endX - 6, tpY - 5);
    ctx.fillStyle = COLORS.red;
    ctx.fillText(`SL ${formatPrice(signal.sl)}`, endX - 6, slY + 13);
  }
}

export class EquityCurveChart {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.chart = null;
  }

  render(data) {
    if (!this.canvas) return;
    this.destroy();

    this.chart = new Chart(this.canvas, {
      type: 'line',
      data: {
        labels: data.times,
        datasets: [
          {
            label: `Top ${data.keep_pct}% · minh họa`,
            data: data.filtered,
            borderColor: COLORS.accent,
            backgroundColor: 'transparent',
            borderWidth: 2,
            fill: false,
            pointRadius: 0,
            stepped: 'after',
            tension: 0,
          },
          {
            label: 'Baseline · minh họa',
            data: data.baseline,
            borderColor: COLORS.lineStrong,
            borderWidth: 1.3,
            borderDash: [5, 5],
            pointRadius: 0,
            stepped: 'after',
            tension: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 260 },
        interaction: { intersect: false, mode: 'index' },
        plugins: {
          legend: {
            position: 'top',
            align: 'end',
            labels: {
              color: COLORS.text,
              boxWidth: 18,
              font: { size: 10, family: MONO },
            },
          },
          tooltip: {
            backgroundColor: COLORS.surface,
            borderColor: COLORS.tooltipBorder,
            borderWidth: 1,
            titleColor: COLORS.text,
            bodyColor: COLORS.accent,
            padding: 10,
            callbacks: {
              label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(2)} R`,
            },
          },
        },
        scales: {
          x: { display: false },
          y: {
            grid: { color: COLORS.grid },
            ticks: {
              color: COLORS.muted,
              font: { size: 9, family: MONO },
              callback: (value) => `${value} R`,
            },
          },
        },
      },
    });
  }

  resize() {
    this.chart?.resize();
  }

  destroy() {
    this.chart?.destroy();
    this.chart = null;
  }
}

export class RocChart {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.chart = null;
  }

  render(curves) {
    if (!this.canvas) return;
    this.chart?.destroy();

    const datasets = curves.map((curve) => {
      const isHoldout = curve.id === 'holdout';
      return {
        label: `${curve.name} · AUC ${curve.auc.toFixed(4)}`,
        data: curve.points.map((point) => ({ x: point.fpr, y: point.tpr })),
        borderColor: isHoldout ? COLORS.accent : COLORS.muted,
        borderWidth: isHoldout ? 2.6 : 1.4,
        borderDash: curve.id === 'random_kfold' ? [5, 4] : undefined,
        pointRadius: 0,
        tension: 0.12,
      };
    });

    datasets.push({
      label: 'Random · AUC 0.500',
      data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      borderColor: COLORS.line,
      borderDash: [5, 5],
      borderWidth: 1,
      pointRadius: 0,
    });

    this.chart = new Chart(this.canvas, {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 280 },
        interaction: { intersect: false, mode: 'nearest' },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              color: COLORS.text,
              boxWidth: 16,
              font: { size: 9, family: MONO },
              padding: 13,
            },
          },
          tooltip: {
            backgroundColor: COLORS.surface,
            borderColor: COLORS.tooltipBorder,
            borderWidth: 1,
            titleColor: COLORS.text,
            bodyColor: COLORS.accent,
            padding: 10,
          },
        },
        scales: {
          x: {
            type: 'linear',
            min: 0,
            max: 1,
            title: {
              display: true,
              text: 'False positive rate',
              color: COLORS.muted,
              font: { size: 10, family: MONO },
            },
            grid: { color: COLORS.grid },
            ticks: { color: COLORS.muted, font: { size: 9, family: MONO } },
          },
          y: {
            type: 'linear',
            min: 0,
            max: 1,
            title: {
              display: true,
              text: 'True positive rate',
              color: COLORS.muted,
              font: { size: 10, family: MONO },
            },
            grid: { color: COLORS.grid },
            ticks: { color: COLORS.muted, font: { size: 9, family: MONO } },
          },
        },
      },
    });
  }
}
