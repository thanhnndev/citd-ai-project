import {
  fetchHealth,
  fetchMarketSample,
  predictSignal,
  fetchBacktestSummary,
  fetchEquityCurve,
  fetchSplitsComparison,
  fetchFeatureImportances
} from './api.js';

import {
  CandlestickRenderer,
  EquityCurveChart,
  RocChart
} from './charts.js';

// Global instances
let candlestickRenderer = null;
let equityChart = null;
let rocChart = null;
let backtestSummaryData = [];
let marketSignals = [];

// Navigation View Switcher (Terminal, Leakage, Analogy)
function initViewSwitcher() {
  const pills = document.querySelectorAll('.nav-pill');
  const views = {
    terminal: document.getElementById('view-terminal'),
    leakage: document.getElementById('view-leakage'),
    analogy: document.getElementById('view-analogy')
  };

  pills.forEach((pill) => {
    pill.addEventListener('click', () => {
      pills.forEach((p) => p.classList.remove('active'));
      pill.classList.add('active');

      const target = pill.getAttribute('data-view');
      Object.entries(views).forEach(([k, el]) => {
        if (el) el.style.display = (k === target) ? 'flex' : 'none';
      });

      window.dispatchEvent(new Event('resize'));
    });
  });
}

// Update AI Decision Panel for selected signal
function updateDecisionPanel(signal) {
  const gaugePct = document.getElementById('gaugePct');
  const circleFill = document.getElementById('circleFill');
  const verdictBanner = document.getElementById('verdictBanner');
  const currentPriceDisplay = document.getElementById('currentPriceDisplay');
  const featureBarsContainer = document.getElementById('featureBarsContainer');

  const pct = Math.round(signal.probability * 100);
  gaugePct.innerText = `${pct}%`;
  currentPriceDisplay.innerText = `$${signal.price.toFixed(2)}`;

  // SVG dashoffset calculation: 2 * PI * 68 = 427.25
  const perimeter = 427.25;
  const offset = perimeter - (perimeter * pct) / 100;
  circleFill.style.strokeDashoffset = offset;

  if (signal.decision === 'PASS') {
    circleFill.style.stroke = 'var(--accent-green)';
    verdictBanner.className = 'verdict-banner pass';
    verdictBanner.innerText = 'PASS (EXECUTE)';
  } else {
    circleFill.style.stroke = 'var(--accent-red)';
    verdictBanner.className = 'verdict-banner skip';
    verdictBanner.innerText = 'SKIP (REJECT)';
  }

  // Render Top 5 Feature Bars
  featureBarsContainer.innerHTML = '';
  const feats = signal.features || {};
  const featureEntries = [
    { name: 'breakeven_R', val: feats.breakeven_R ?? 1.5, weight: 9.3 },
    { name: 'vol200', val: feats.vol200 ?? 0.008, weight: 8.8 },
    { name: 'dist_ema200', val: feats.dist_ema200_atr ?? 1.2, weight: 6.0 },
    { name: 'atr_ratio', val: 1.15, weight: 5.8 },
    { name: 'rsi14', val: feats.rsi14 ?? 52.4, weight: 5.2 }
  ];

  featureEntries.forEach((f) => {
    const row = document.createElement('div');
    row.className = 'feature-bar-row';
    const fillWidth = Math.min(100, Math.max(15, f.weight * 9.5));
    row.innerHTML = `
      <span style="color:var(--text-muted);">${f.name}</span>
      <div style="display:flex; align-items:center; gap:8px;">
        <div class="feature-bar-track">
          <div class="feature-bar-fill" style="width:${fillWidth}%;"></div>
        </div>
        <strong style="color:var(--text-white); min-width:32px; text-align:right;">${f.val}</strong>
      </div>
    `;
    featureBarsContainer.appendChild(row);
  });
}

// Populate Signal Buttons
function populateSignalButtons(signals) {
  const container = document.getElementById('signalPillContainer');
  container.innerHTML = '';

  signals.forEach((s, idx) => {
    const btn = document.createElement('button');
    btn.className = `signal-btn ${idx === 0 ? 'active' : ''}`;
    btn.innerText = `#${idx + 1} (${s.decision})`;
    btn.style.color = s.decision === 'PASS' ? 'var(--accent-green)' : 'var(--accent-red)';

    btn.addEventListener('click', () => {
      document.querySelectorAll('.signal-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      candlestickRenderer.setSelectedSignal(s);
      updateDecisionPanel(s);
    });

    container.appendChild(btn);
  });
}

// Initialize Terminal View (Candlestick & Top K% Slider)
async function initTerminal() {
  candlestickRenderer = new CandlestickRenderer('candlestickCanvas');
  equityChart = new EquityCurveChart('equityCanvas');
  const slider = document.getElementById('filterSlider');
  const sliderLabel = document.getElementById('sliderLabel');

  try {
    // 1. Load Market & Signals
    const marketData = await fetchMarketSample(140);
    marketSignals = marketData.signals;
    candlestickRenderer.setData(marketData.candles, marketData.signals);

    if (marketSignals.length > 0) {
      populateSignalButtons(marketSignals);
      updateDecisionPanel(marketSignals[0]);
    }

    // 2. Load Backtest Summary & Equity Curve
    backtestSummaryData = await fetchBacktestSummary();

    const updateSlider = async (pct) => {
      sliderLabel.innerText = `Filter Threshold: Top ${pct}%`;
      const eqData = await fetchEquityCurve(pct);
      equityChart.render(eqData);

      const row = backtestSummaryData.find((r) => r.keep_pct === parseInt(pct)) || backtestSummaryData[0];
      if (row) {
        document.getElementById('kpiWinRate').innerText = `${row.win_rate_pct.toFixed(2)}%`;
        document.getElementById('kpiNetProfit').innerText = `+${row.net_profit_R.toFixed(1)} R`;
        document.getElementById('kpiProfitFactor').innerText = `${row.profit_factor.toFixed(2)}`;
        document.getElementById('kpiDrawdown').innerText = `-${row.max_dd_R.toFixed(1)} R`;
      }
    };

    slider.addEventListener('input', (e) => {
      updateSlider(e.target.value);
    });

    // Default Top 50%
    updateSlider(50);

  } catch (err) {
    console.error('Error loading terminal data:', err);
  }
}

// Initialize Data Leakage Lab
async function initLeakageLab() {
  rocChart = new RocChart('rocCanvas');
  const container = document.getElementById('splitsListContainer');

  try {
    const data = await fetchSplitsComparison();
    rocChart.render(data.roc_curves);

    container.innerHTML = '';
    data.splits.forEach((s) => {
      const entry = document.createElement('div');
      entry.className = 'split-entry';
      entry.innerHTML = `
        <div>
          <div style="font-weight:700; font-size:12px; color:${s.color};">${s.name}</div>
          <div style="font-size:11px; color:var(--text-muted); margin-top:2px;">
            AUC: <strong>${s.roc_auc.toFixed(4)}</strong> | Net: <strong>+${s.net_profit_r} R</strong>
          </div>
        </div>
        <span style="font-size:11px; font-weight:700; padding:3px 8px; border-radius:4px; background:${s.color}22; color:${s.color};">${s.reliability}</span>
      `;
      container.appendChild(entry);
    });
  } catch (err) {
    console.error('Error loading leakage lab:', err);
  }
}

// Check Backend Health
async function checkHealth() {
  const statusText = document.getElementById('status-text');
  try {
    const h = await fetchHealth();
    statusText.innerText = `Model: CatBoost Online (${h.holdout_samples} samples)`;
  } catch (e) {
    statusText.innerText = 'Model: Offline';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initViewSwitcher();
  checkHealth();
  initTerminal();
  initLeakageLab();
});
