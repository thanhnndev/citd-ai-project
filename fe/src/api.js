/**
 * API Client for Backend Endpoints
 */
const BASE_URL = '/api';

export async function fetchHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  return res.json();
}

export async function fetchMarketSample(limit = 120) {
  const res = await fetch(`${BASE_URL}/market/sample?limit=${limit}`);
  return res.json();
}

export async function predictSignal(features, threshold = 0.50) {
  const res = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features, threshold })
  });
  return res.json();
}

export async function fetchBacktestSummary() {
  const res = await fetch(`${BASE_URL}/backtest/summary`);
  return res.json();
}

export async function fetchEquityCurve(keepPct = 50) {
  const res = await fetch(`${BASE_URL}/backtest/equity-curve?keep_pct=${keepPct}`);
  return res.json();
}

export async function fetchSplitsComparison() {
  const res = await fetch(`${BASE_URL}/splits/comparison`);
  return res.json();
}

export async function fetchFeatureImportances() {
  const res = await fetch(`${BASE_URL}/features/importances`);
  return res.json();
}
