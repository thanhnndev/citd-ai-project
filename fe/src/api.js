import {
  buildEquityCurve,
  buildMarketSample,
  FEATURE_IMPORTANCES,
  getBacktestSummary,
  getOfflineHealth,
  getSplitsComparison,
} from './demo-data.js';

/**
 * Local data adapter.
 *
 * The original adapter called /api/* on the FastAPI service. This implementation
 * deliberately keeps the same function contract but resolves data from committed
 * artifacts copied into src/demo-data.js. No browser request leaves the Vite app.
 */

export async function fetchHealth() {
  return getOfflineHealth();
}

export async function fetchMarketSample(limit = 24) {
  return buildMarketSample(limit);
}

export async function predictSignal(_features, threshold = 0.5) {
  return {
    available: false,
    live_inference: false,
    threshold,
    decision: 'UNAVAILABLE',
    reason: 'Bản demo offline không nạp hoặc chạy CatBoost. Chỉ hiển thị xác suất đã tính sẵn trong artifact holdout.',
  };
}

export async function fetchBacktestSummary() {
  return getBacktestSummary();
}

export async function fetchEquityCurve(keepPct = 50) {
  return buildEquityCurve(keepPct);
}

export async function fetchSplitsComparison() {
  return getSplitsComparison();
}

export async function fetchFeatureImportances() {
  return FEATURE_IMPORTANCES.map((item) => ({ ...item }));
}
