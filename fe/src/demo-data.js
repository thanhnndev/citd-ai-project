/**
 * Offline demo data for the CITD ML frontend.
 *
 * The canonical metrics below are transcribed from committed project artifacts.
 * Browser-friendly visual approximations (trade-event OHLC, equity paths and
 * ROC curves) are intentionally generated in memory and are labelled as such
 * in the UI. No file outside fe/ is read at runtime and no API is requested.
 */

export const DEMO_META = {
  project: 'CITD ML — Pyramid BTCUSD + CatBoost',
  dataMode: 'offline-demo',
  dataModeLabel: 'Offline demo · artifact tĩnh',
  modelMode: 'precomputed',
  modelModeLabel: 'Không chạy CatBoost live',
  holdoutRows: 5028,
  trainRows: 25008,
  features: 23,
  holdoutStart: '2025-02-08 15:30:00',
  holdoutEnd: '2026-08-21 04:30:00',
  reproducibility: 'PASS · run 1 = run 2 trên cùng máy',
  sources: [
    'outputs/holdout/stage3/stage3_report.json',
    'outputs/holdout/stage3/stage3_backtest_summary.csv',
    'outputs/holdout/stage3/holdout_fixed_trade_universe_scored.csv',
    'outputs/holdout/stage4/table1_classification_metrics.csv',
    'outputs/holdout/repro/stage5_repro_report.json',
  ],
};

/**
 * A 24-row subset of committed holdout rows (row_id 0–23). Values are used to
 * illustrate the precomputed probability and feature snapshot in the browser.
 * These are real scored holdout records, not newly generated predictions.
 */
const HOLDOUT_SAMPLE = [
  [0, '2025-02-09 16:00:00', 16, 1.20568106, 0.00164538, -1.20740903, 1.30865999, 0.734416, 0, 96444.0, 95865.336, -1.0, 14],
  [1, '2025-02-09 16:15:00', 16, 1.2064869, 0.00164252, -0.88825088, 1.30883677, 0.810087, 0, 96518.4, 95939.2896, -1.0, 5],
  [2, '2025-02-09 16:30:00', 16, 1.20895417, 0.00163973, -1.37905366, 1.3209388, 0.826179, 0, 96393.6, 96007.6873, -0.667252, 19],
  [3, '2025-02-09 16:45:00', 16, 1.20714064, 0.00163623, -1.2960136, 1.3294367, 0.827954, 0, 96409.0, 96008.6247, -0.692147, 18],
  [4, '2025-02-10 20:45:00', 20, 1.26774401, 0.00147318, -1.01089902, 1.21247843, 0.649301, 0, 96396.9, 95818.5186, -1.0, 3],
  [5, '2025-02-10 21:00:00', 21, 1.2700656, 0.00147531, -1.57611572, 1.20415938, 0.593248, 0, 96276.1, 95698.4434, -1.0, 2],
  [6, '2025-02-10 21:15:00', 21, 1.26639307, 0.00147127, -1.98816628, 1.15499641, 0.593288, 0, 96193.0, 95615.842, -1.0, 1],
  [7, '2025-02-10 21:30:00', 21, 1.28467623, 0.00148665, -3.20013341, 1.16822701, 0.481548, 0, 95905.3, 95329.8682, -1.0, 0],
  [8, '2025-02-10 00:00:00', 0, 1.57059637, 0.00170741, -0.22699787, 2.05436289, 0.57533, 0, 96439.8, 96199.792, -0.41478, 5],
  [9, '2025-02-10 00:15:00', 0, 1.59719901, 0.00169235, 0.76286325, 2.09514525, 0.699215, 0, 96940.4, 96358.7576, -1.0, 4],
  [10, '2025-02-10 00:30:00', 0, 1.60486971, 0.00167783, 0.27951783, 2.15015463, 0.786452, 0, 96696.3, 96116.1222, -1.0, 3],
  [11, '2025-02-10 00:45:00', 0, 1.61070251, 0.00167841, 0.38165312, 2.19422999, 0.82644, 0, 96767.3, 96186.6962, -1.0, 2],
  [12, '2025-02-10 01:30:00', 1, 1.65688345, 0.00166101, -0.8027979, 1.88536566, 0.789981, 0, 96173.8, 95596.7572, -1.0, 0],
  [13, '2025-02-10 01:45:00', 1, 1.71894378, 0.0017192, -1.71623061, 1.87833522, 0.53665, 1, 95543.9, 97327.5924, 3.111471, 51],
  [14, '2025-02-10 02:00:00', 2, 1.74823028, 0.00171446, -1.94629199, 1.89581824, 0.647222, 1, 95563.9, 97310.5938, 3.046293, 50],
  [15, '2025-02-10 02:15:00', 2, 1.75279176, 0.00171345, -1.56696638, 1.72464967, 0.693438, 1, 95477.4, 97308.888, 3.197071, 49],
  [16, '2025-02-10 03:15:00', 3, 1.79701824, 0.00173257, -0.29419216, 1.59250643, 0.497734, 0, 96016.4, 97277.7407, 2.189454, 45],
  [17, '2025-02-10 03:30:00', 3, 1.80893004, 0.0017515, -0.29562854, 1.62271948, 0.54877, 1, 96369.9, 97267.0416, 1.551559, 44],
  [18, '2025-02-10 03:45:00', 3, 1.83827008, 0.001808, 1.05317898, 1.59666818, 0.611945, 1, 96970.9, 97243.4478, 0.468436, 43],
  [19, '2025-02-10 04:00:00', 4, 1.86128282, 0.00181155, 0.61213786, 1.58472403, 0.512795, 1, 96785.2, 97232.1322, 0.87561, 43],
  [20, '2025-02-10 14:45:00', 14, 2.23023959, 0.00189827, 1.84794426, 0.99296182, 0.304068, 0, 97468.9, 96884.0866, -1.0, 3],
  [21, '2025-02-10 15:00:00', 15, 2.26117283, 0.00190241, 1.1390153, 1.04291628, 0.431528, 0, 97276.1, 97165.9516, -0.188722, 91],
  [22, '2025-02-10 15:15:00', 15, 2.2676194, 0.00190703, -0.81745055, 1.08119999, 0.525271, 0, 97487.7, 96902.7738, -1.0, 1],
  [23, '2025-02-10 15:30:00', 15, 2.31121139, 0.0019338, 0.48044109, 1.12782096, 0.711851, 0, 97052.6, 97139.8456, 0.149825, 89],
];

/** Global model importance reported in the committed technical report. */
export const FEATURE_IMPORTANCES = [
  { feature: 'breakeven_R', importance_pct: 9.32, label: 'breakeven R' },
  { feature: 'vol200', importance_pct: 8.84, label: 'volatility 200' },
  { feature: 'dist_ema200_atr', importance_pct: 6.02, label: 'distance / EMA200' },
  { feature: 'atr_ratio_14_90', importance_pct: 5.85, label: 'ATR ratio 14/90' },
  { feature: 'hour', importance_pct: 5.85, label: 'hour (UTC)' },
];

/** Exact summary rows from outputs/holdout/stage3/stage3_backtest_summary.csv. */
const BACKTEST_SUMMARY = [
  { keep_pct: 100, filter: 'Baseline', trades: 5028, net_profit_R: 245.929682659, max_dd_R: 305.31742867, profit_factor: 1.09915259483, win_rate_pct: 35.2824184566 },
  { keep_pct: 20, filter: 'Top 20%', trades: 1006, net_profit_R: -30.013369591, max_dd_R: 125.376298026, profit_factor: 0.938466410654, win_rate_pct: 36.1829025845 },
  { keep_pct: 30, filter: 'Top 30%', trades: 1509, net_profit_R: -50.0085631644, max_dd_R: 210.765309903, profit_factor: 0.934293529676, win_rate_pct: 35.520212061 },
  { keep_pct: 40, filter: 'Top 40%', trades: 2012, net_profit_R: -53.1654255481, max_dd_R: 251.555737069, profit_factor: 0.948544713299, win_rate_pct: 35.0397614314 },
  { keep_pct: 50, filter: 'Top 50%', trades: 2514, net_profit_R: -36.5527549959, max_dd_R: 263.246077697, profit_factor: 0.971794851726, win_rate_pct: 34.8050914877 },
  { keep_pct: 60, filter: 'Top 60%', trades: 3017, net_profit_R: 73.1608503384, max_dd_R: 233.952436014, profit_factor: 1.04738635284, win_rate_pct: 35.5982764335 },
  { keep_pct: 70, filter: 'Top 70%', trades: 3520, net_profit_R: 186.924927125, max_dd_R: 197.237275172, profit_factor: 1.10400028476, win_rate_pct: 35.7670454545 },
  { keep_pct: 80, filter: 'Top 80%', trades: 4023, net_profit_R: 259.911296757, max_dd_R: 219.024029419, profit_factor: 1.12779311196, win_rate_pct: 35.520755655 },
];

/** Four development splits plus the sealed holdout; metrics from table1. */
const SPLITS = [
  { id: 'random_kfold', short_name: 'Random K-Fold', name: 'Cách 1 — Random K-Fold', roc_auc: 0.8582, f1: 0.6494, trades: 10004, net_profit_r: 6937.53, max_dd_r: 73.76, profit_factor: 2.5437, win_rate_pct: 44.65, reliability: 'Rất thấp', color: '#ef6b73', note: 'Không bảo toàn thứ tự thời gian' },
  { id: 'grouped_kfold', short_name: 'Grouped K-Fold', name: 'Cách 1b — Grouped K-Fold', roc_auc: 0.7454, f1: 0.5077, trades: 10004, net_profit_r: 4752.04, max_dd_r: 99.71, profit_factor: 1.95, win_rate_pct: 39.26, reliability: 'Thấp', color: '#f2b84b', note: 'Tách theo tháng, vẫn có phụ thuộc lân cận' },
  { id: 'walk_forward', short_name: 'Walk-forward', name: 'Cách 2 — Walk-forward', roc_auc: 0.5875, f1: 0.3288, trades: 10004, net_profit_r: 2148.31, max_dd_r: 211.29, profit_factor: 1.4117, win_rate_pct: 34.76, reliability: 'Khá', color: '#61d6a7', note: 'Mở rộng cửa sổ theo thời gian' },
  { id: 'purged_walk_forward', short_name: 'Purged WF', name: 'Cách 3 — Purged WF + Embargo', roc_auc: 0.5895, f1: 0.3247, trades: 10004, net_profit_r: 2119.48, max_dd_r: 142.36, profit_factor: 1.4074, win_rate_pct: 34.99, reliability: 'Cao', color: '#62a8f7', note: 'Có kiểm biên purge và embargo' },
  { id: 'holdout', short_name: 'Holdout', name: 'Holdout — kiểm định ngoài mẫu', roc_auc: 0.6046, f1: 0.4022, trades: 5028, net_profit_r: 245.929682659, max_dd_r: 305.31742867, profit_factor: 1.09915259483, win_rate_pct: 35.2824184566, reliability: 'Độc lập', color: '#a98af5', note: 'Baseline 5.028 lệnh; AUC/F1 tính một lần' },
];

function rounded(value, digits = 4) {
  return Number(value.toFixed(digits));
}

/** Convert committed trade records to clearly labelled illustrative OHLC bars. */
export function buildMarketSample(limit = 24) {
  const safeLimit = Math.max(6, Math.min(Number(limit) || 24, HOLDOUT_SAMPLE.length));
  const sourceRows = HOLDOUT_SAMPLE.slice(0, safeLimit);

  const signals = sourceRows.map((row, index) => {
    const [id, time, hour, breakevenR, vol200, distEma, atrRatio, probability, label, entryPrice, exitPrice, resultR, barsHeld] = row;
    const decision = probability >= 0.5 ? 'PASS' : 'SKIP';
    const atrProxy = Math.max(Math.abs(entryPrice - exitPrice), entryPrice * 0.0035);

    return {
      id,
      index,
      time,
      hour,
      type: 'LONG',
      price: entryPrice,
      exitPrice,
      probability,
      decision,
      label,
      r: resultR,
      barsHeld,
      actualResult: resultR >= 0 ? `WIN ${resultR >= 0 ? '+' : ''}${resultR.toFixed(2)} R` : `LOSS ${resultR.toFixed(2)} R`,
      tp: entryPrice + atrProxy * 2,
      sl: entryPrice - atrProxy,
      max_bars: Math.max(3, Math.min(16, barsHeld || 8)),
      features: { breakeven_R: breakevenR, vol200, dist_ema200_atr: distEma, atr_ratio_14_90: atrRatio, hour },
    };
  });

  const candles = signals.map((signal, index) => {
    const move = signal.exitPrice - signal.price;
    const range = Math.max(Math.abs(move), signal.price * 0.0025);
    const open = signal.price - move * 0.18;
    const close = signal.exitPrice;
    const high = Math.max(open, close) + range * 0.3;
    const low = Math.min(open, close) - range * 0.24;
    const wave = Math.abs(Math.sin((signal.id + 1) * 1.71));

    return {
      time: signal.time,
      open: rounded(open, 2),
      high: rounded(high, 2),
      low: rounded(low, 2),
      close: rounded(close, 2),
      volume: Math.round(180 + wave * 620 + (signal.label ? 110 : 0)),
      illustrative: true,
    };
  });

  return {
    candles,
    signals,
    meta: {
      title: 'Replay 24 lệnh holdout đã chấm điểm',
      note: 'OHLC minh họa suy ra từ entry/exit; không phải nến M15 nguyên bản.',
      source: 'holdout_fixed_trade_universe_scored.csv · row_id 0–23',
    },
  };
}

/** A browser-friendly path with an exact final value from the artifact. */
function buildIllustrativePath(endpoint, points = 36) {
  const amplitude = Math.min(18, Math.max(4, Math.abs(endpoint) * 0.075));
  const values = [];

  for (let i = 0; i < points; i += 1) {
    const x = i / (points - 1);
    const eased = 1 - Math.pow(1 - x, 1.7);
    const ripple = Math.sin(x * Math.PI * 5.2) * amplitude * Math.sin(Math.PI * x) * (0.35 + (1 - x) * 0.65);
    values.push(rounded(endpoint * eased + ripple, 2));
  }

  values[0] = 0;
  values[values.length - 1] = rounded(endpoint, 2);
  return values;
}

export function buildEquityCurve(keepPct = 50) {
  const pct = Math.max(10, Math.min(100, Number(keepPct) || 50));
  const summary = BACKTEST_SUMMARY.find((row) => row.keep_pct === pct) || BACKTEST_SUMMARY.find((row) => row.keep_pct === 50);
  const baseline = BACKTEST_SUMMARY.find((row) => row.keep_pct === 100);
  const pointCount = 36;
  const baselineValues = buildIllustrativePath(baseline.net_profit_R, pointCount);
  const filteredValues = buildIllustrativePath(summary.net_profit_R, pointCount);

  return {
    times: Array.from({ length: pointCount }, (_, index) => `M${index}`),
    baseline: baselineValues,
    filtered: filteredValues,
    keep_pct: summary.keep_pct,
    illustrative: true,
    note: 'Đường hình minh họa; điểm cuối khớp Net R trong artifact.',
  };
}

/** Generate a readable ROC-shaped curve from each committed AUC value. */
function buildIllustrativeRoc(auc) {
  const exponent = (1 - auc) / auc;
  return Array.from({ length: 41 }, (_, index) => {
    const fpr = index / 40;
    let tpr = Math.pow(fpr, exponent);
    if (index === 0) tpr = 0;
    if (index === 40) tpr = 1;
    return { fpr, tpr: rounded(tpr, 4) };
  });
}

export function getOfflineHealth() {
  return {
    status: 'offline_demo',
    mode: 'static_artifacts',
    model: 'CatBoost precomputed scores only',
    model_loaded: false,
    live_inference: false,
    features_count: DEMO_META.features,
    feature_importance_preview_count: FEATURE_IMPORTANCES.length,
    holdout_samples: DEMO_META.holdoutRows,
    train_samples: DEMO_META.trainRows,
    framework: 'Vite static demo',
  };
}

export function getBacktestSummary() {
  return BACKTEST_SUMMARY.map((row) => ({ ...row }));
}

export function getSplitsComparison() {
  return {
    splits: SPLITS.map((split) => ({ ...split })),
    roc_curves: SPLITS.map((split) => ({
      id: split.id,
      name: split.short_name,
      color: split.color,
      auc: split.roc_auc,
      illustrative: true,
      points: buildIllustrativeRoc(split.roc_auc),
    })),
    roc_note: 'Đường ROC được nội suy để minh họa AUC đã commit; không phải điểm OOF gốc.',
  };
}
