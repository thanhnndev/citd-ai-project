import {
  fetchBacktestSummary,
  fetchEquityCurve,
  fetchFeatureImportances,
  fetchHealth,
  fetchMarketSample,
  fetchSplitsComparison,
} from './api.js';
import { DEMO_META } from './demo-data.js';
import { CandlestickRenderer, EquityCurveChart, RocChart } from './charts.js';

const state = {
  candlestickRenderer: null,
  equityChart: null,
  rocChart: null,
  marketSignals: [],
  backtestSummary: [],
  featureImportances: [],
  leakageReady: false,
  activeView: 'terminal',
};

const numberFormatter = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 2 });

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) element.textContent = value;
}

function signed(value, digits = 2) {
  const safeValue = Number(value) || 0;
  return `${safeValue >= 0 ? '+' : ''}${safeValue.toLocaleString('vi-VN', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;
}

function colorForProfit(value) {
  const element = document.getElementById('kpiNetProfit');
  if (element) element.className = value >= 0 ? 'positive' : 'negative';
}

function featureValue(feature, value) {
  if (feature === 'vol200') return `${(Number(value) * 100).toFixed(2)}%`;
  if (feature === 'hour') return Number(value).toFixed(0);
  return Number(value).toFixed(2);
}

function updateDecisionPanel(signal) {
  const probabilityPct = Math.round(signal.probability * 100);
  const gaugePerimeter = 2 * Math.PI * 74;
  const gaugeFill = document.getElementById('circleFill');
  const verdict = document.getElementById('verdictBanner');

  setText('gaugePct', `${probabilityPct}%`);
  setText('currentPriceDisplay', `$${numberFormatter.format(signal.price)}`);
  setText('selectedRowId', `#${signal.id}`);
  setText('selectedTime', signal.time);
  setText('selectedBars', String(signal.barsHeld));
  setText('selectedResult', signal.actualResult);

  const result = document.getElementById('selectedResult');
  if (result) result.className = signal.r >= 0 ? 'positive' : 'negative';

  if (gaugeFill) {
    gaugeFill.style.strokeDashoffset = String(gaugePerimeter * (1 - signal.probability));
    gaugeFill.style.stroke = signal.decision === 'PASS' ? 'var(--green)' : 'var(--red)';
  }

  if (verdict) {
    verdict.className = `verdict ${signal.decision === 'PASS' ? 'pass' : 'skip'}`;
    verdict.textContent = signal.decision === 'PASS' ? 'PASS · GIỮ ỨNG VIÊN' : 'SKIP · BỎ ỨNG VIÊN';
  }

  const featureContainer = document.getElementById('featureBarsContainer');
  if (!featureContainer) return;

  featureContainer.replaceChildren();
  state.featureImportances.forEach((importance) => {
    const value = signal.features[importance.feature];
    const row = document.createElement('div');
    row.className = 'feature-row';

    const label = document.createElement('span');
    label.className = 'feature-label';
    label.textContent = importance.label;
    label.title = `${importance.feature} · ${importance.importance_pct.toFixed(2)}% importance`;

    const track = document.createElement('span');
    track.className = 'feature-track';
    track.setAttribute('aria-hidden', 'true');
    const fill = document.createElement('span');
    fill.className = 'feature-fill';
    fill.style.display = 'block';
    fill.style.width = `${Math.min(100, (importance.importance_pct / 10) * 100)}%`;
    track.appendChild(fill);

    const valueElement = document.createElement('span');
    valueElement.className = 'feature-value';
    valueElement.textContent = featureValue(importance.feature, value);

    row.append(label, track, valueElement);
    featureContainer.appendChild(row);
  });
}

function populateSignalButtons(signals) {
  const container = document.getElementById('signalPillContainer');
  if (!container) return;

  container.replaceChildren();
  signals.forEach((signal, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `signal-btn ${signal.decision === 'PASS' ? 'pass' : 'skip'} ${index === 0 ? 'active' : ''}`;
    button.textContent = `#${signal.id}`;
    button.setAttribute('aria-label', `Lệnh ${signal.id}, xác suất ${(signal.probability * 100).toFixed(1)}%, ${signal.decision}`);
    button.setAttribute('aria-pressed', index === 0 ? 'true' : 'false');

    button.addEventListener('click', () => {
      container.querySelectorAll('.signal-btn').forEach((item) => {
        item.classList.remove('active');
        item.setAttribute('aria-pressed', 'false');
      });
      button.classList.add('active');
      button.setAttribute('aria-pressed', 'true');
      state.candlestickRenderer.setSelectedSignal(signal);
      updateDecisionPanel(signal);
    });

    container.appendChild(button);
  });
}

function renderRetentionTable(activePct) {
  const body = document.getElementById('retentionTableBody');
  if (!body) return;

  body.replaceChildren();
  const rows = [...state.backtestSummary].sort((a, b) => {
    if (a.keep_pct === 100) return -1;
    if (b.keep_pct === 100) return 1;
    return a.keep_pct - b.keep_pct;
  });

  rows.forEach((row) => {
    const tr = document.createElement('tr');
    if (row.keep_pct === activePct) tr.className = 'is-active';
    const cells = [
      row.filter,
      numberFormatter.format(row.trades),
      `${signed(row.net_profit_R)} R`,
      `${numberFormatter.format(row.max_dd_R)} R`,
      row.profit_factor.toFixed(4),
      `${row.win_rate_pct.toFixed(2)}%`,
    ];
    cells.forEach((value, index) => {
      const td = document.createElement('td');
      td.textContent = value;
      if (index === 2) td.className = row.net_profit_R >= 0 ? 'positive' : 'negative';
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}

async function updateRetention(pct) {
  const activePct = Number(pct);
  const row = state.backtestSummary.find((item) => item.keep_pct === activePct);
  if (!row) return;

  setText('retentionValue', `${activePct}%`);
  setText('kpiNetProfit', `${signed(row.net_profit_R)} R`);
  colorForProfit(row.net_profit_R);
  setText('kpiTrades', `${numberFormatter.format(row.trades)} / 5.028 lệnh`);
  setText('kpiWinRate', `${row.win_rate_pct.toFixed(2)}%`);
  setText('kpiProfitFactor', row.profit_factor.toFixed(4));
  setText('kpiDrawdown', `${numberFormatter.format(row.max_dd_R)} R`);

  const equity = await fetchEquityCurve(activePct);
  state.equityChart.render(equity);
  renderRetentionTable(activePct);
}

async function initTerminal() {
  state.candlestickRenderer = new CandlestickRenderer('candlestickCanvas');
  state.equityChart = new EquityCurveChart('equityCanvas');

  const [marketData, summaries, importances] = await Promise.all([
    fetchMarketSample(24),
    fetchBacktestSummary(),
    fetchFeatureImportances(),
  ]);

  state.marketSignals = marketData.signals;
  state.backtestSummary = summaries;
  state.featureImportances = importances;

  state.candlestickRenderer.setData(marketData.candles, marketData.signals);
  populateSignalButtons(state.marketSignals);
  if (state.marketSignals.length) updateDecisionPanel(state.marketSignals[0]);

  const slider = document.getElementById('filterSlider');
  slider?.addEventListener('input', (event) => updateRetention(event.currentTarget.value));
  await updateRetention(slider?.value || 50);
}

function renderSplitTable(splits) {
  const body = document.getElementById('splitsTableBody');
  if (!body) return;

  body.replaceChildren();
  splits.forEach((split) => {
    const tr = document.createElement('tr');
    const cells = [
      split.name,
      split.roc_auc.toFixed(4),
      split.f1.toFixed(4),
      `${signed(split.net_profit_r)} R`,
      split.profit_factor.toFixed(4),
      split.note,
    ];
    cells.forEach((value, index) => {
      const td = document.createElement('td');
      td.textContent = value;
      if (index === 0) td.className = 'method-cell';
      if (index === 3) td.className = split.net_profit_r >= 0 ? 'positive' : 'negative';
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}

async function initLeakageLab() {
  if (state.leakageReady) return;
  state.leakageReady = true;

  state.rocChart = new RocChart('rocCanvas');
  const data = await fetchSplitsComparison();
  state.rocChart.render(data.roc_curves);
  renderSplitTable(data.splits);
}

function renderSourceList() {
  const list = document.getElementById('sourceList');
  if (!list) return;

  list.replaceChildren();
  DEMO_META.sources.forEach((source) => {
    const li = document.createElement('li');
    li.textContent = source;
    list.appendChild(li);
  });
}

async function checkDataMode() {
  const status = await fetchHealth();
  const title = status.status === 'offline_demo' ? 'OFFLINE DEMO' : 'STATIC DATA';
  document.querySelectorAll('.offline-status strong').forEach((element) => {
    element.textContent = title;
  });
}

function switchView(target) {
  const validTarget = ['terminal', 'leakage', 'scope'].includes(target) ? target : 'terminal';
  state.activeView = validTarget;

  document.querySelectorAll('[data-view-panel]').forEach((panel) => {
    const active = panel.dataset.viewPanel === validTarget;
    panel.hidden = !active;
    panel.classList.toggle('active', active);
  });

  document.querySelectorAll('.nav-pill').forEach((button) => {
    const active = button.dataset.view === validTarget;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
  });

  if (validTarget === 'leakage') initLeakageLab();
  if (validTarget === 'terminal') requestAnimationFrame(() => state.equityChart?.resize());
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function initViewSwitcher() {
  document.querySelectorAll('.nav-pill').forEach((button) => {
    button.addEventListener('click', () => switchView(button.dataset.view));
  });

  document.querySelectorAll('[data-view]:not(.nav-pill)').forEach((button) => {
    button.addEventListener('click', () => switchView(button.dataset.view));
  });
}

document.addEventListener('DOMContentLoaded', async () => {
  initViewSwitcher();
  renderSourceList();
  checkDataMode();

  try {
    await initTerminal();
  } catch (error) {
    console.error('Không thể khởi tạo dữ liệu demo tĩnh:', error);
    const main = document.querySelector('.page-shell');
    if (main) {
      const warning = document.createElement('div');
      warning.className = 'noscript-warning';
      warning.textContent = 'Không thể khởi tạo frontend demo. Kiểm tra lại bundle trong src/.';
      main.prepend(warning);
    }
  }
});
