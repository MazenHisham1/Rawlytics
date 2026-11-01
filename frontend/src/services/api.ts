const DEFAULT_API_URL = '';

// One-time cleanup: drop any URL saved under the old key so the field starts empty.
localStorage.removeItem('smart-cleaner-api-url');

let currentSessionId: string | null = null;

export function getSessionId(): string | null {
  return currentSessionId;
}

export function setSessionId(id: string | null): void {
  currentSessionId = id;
}

const API_URL_KEY = 'rawlytics-api-url';

export function getApiUrl(): string {
  return localStorage.getItem(API_URL_KEY) || DEFAULT_API_URL;
}

export function clearApiUrl(): void {
  localStorage.removeItem(API_URL_KEY);
}

export function setApiUrl(url: string): void {
  const trimmed = url.trim();
  if (trimmed) localStorage.setItem(API_URL_KEY, trimmed);
  else clearApiUrl();
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  if (!getApiUrl()) throw new Error('No API URL set. Open API settings and enter your backend URL.');
  const baseUrl = getApiUrl().replace(/\/+$/, '');
  const headers = new Headers(options?.headers);
  headers.set('ngrok-skip-browser-warning', 'true');

  const separator = path.includes('?') ? '&' : '?';
  const sessionParam = currentSessionId ? `session_id=${encodeURIComponent(currentSessionId)}&` : '';
  const fullPath = `${baseUrl}${path}${separator}${sessionParam}args=&kwargs=`;

  const res = await fetch(fullPath, {
    ...options,
    headers,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error');
    throw new Error(`API error (${res.status}): ${text}`);
  }
  return res.json();
}

// ========================= Upload =========================
export async function uploadFile(file: File): Promise<{ session_id: string; filename: string; rows: number; columns: number; column_names: string[]; size_mb: number }> {
  const formData = new FormData();
  formData.append('file', file);
  const result = await apiFetch<{ session_id: string; filename: string; rows: number; columns: number; column_names: string[]; size_mb: number }>('/upload', { method: 'POST', body: formData });
  setSessionId(result.session_id);
  return result;
}

export async function deleteSession(sessionId: string): Promise<{ deleted: string }> {
  return apiFetch(`/session/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
}

// ========================= Basic data =========================
export async function fetchInfo(): Promise<{ rows: number; columns: number; dtypes: Record<string, string> }> {
  return apiFetch('/info', { method: 'POST' });
}

export async function fetchDescribe(): Promise<Record<string, Record<string, number>>> {
  return apiFetch('/describe', { method: 'POST' });
}

export async function fetchNulls(): Promise<Record<string, number>> {
  return apiFetch('/nulls', { method: 'POST' });
}

export async function fetchDuplicates(): Promise<{ duplicates: number }> {
  return apiFetch('/duplicates', { method: 'POST' });
}

export async function fetchShape(): Promise<{ rows: number; columns: number }> {
  return apiFetch('/shape', { method: 'POST' });
}

export async function fetchColumnUnique(col: string): Promise<{ column: string; unique: number }> {
  return apiFetch(`/column/unique/${encodeURIComponent(col)}`, { method: 'POST' });
}

// ========================= Charts =========================
export async function fetchChartDistribution(col: string): Promise<{ label: string; value: number }[]> {
  return apiFetch(`/chart/distribution/${encodeURIComponent(col)}`, { method: 'POST' });
}

export async function fetchChartBar(col: string): Promise<{ label: string; value: number }[]> {
  return apiFetch(`/chart/bar/${encodeURIComponent(col)}`, { method: 'POST' });
}

export async function fetchChartPie(col: string): Promise<{ label: string; value: number }[]> {
  return apiFetch(`/chart/pie/${encodeURIComponent(col)}`, { method: 'POST' });
}

export async function fetchChartHist(col: string, bins = 10): Promise<{ label: string; value: number }[]> {
  return apiFetch(`/chart/hist/${encodeURIComponent(col)}?bins=${bins}`, { method: 'POST' });
}

export async function fetchChartScatter(col1: string, col2: string): Promise<{ x: number; y: number }[]> {
  return apiFetch(`/chart/scatter/${encodeURIComponent(col1)}/${encodeURIComponent(col2)}`, { method: 'POST' });
}

export async function fetchChartBoxplot(col: string): Promise<{ min: number; q1: number; median: number; q3: number; max: number; outliers: number[] }> {
  return apiFetch(`/chart/boxplot/${encodeURIComponent(col)}`, { method: 'POST' });
}

// ========================= Statistical Depth =========================
export async function fetchColumnSkewness(col: string): Promise<{ column: string; skewness: number; type: string }> {
  return apiFetch(`/column/skewness/${encodeURIComponent(col)}`, { method: 'POST' });
}

export async function fetchColumnKurtosis(col: string): Promise<{ column: string; kurtosis: number | null; type: string }> {
  return apiFetch(`/column/kurtosis/${encodeURIComponent(col)}`, { method: 'POST' });
}

export async function fetchNormality(col: string): Promise<{ column: string; test: string; statistic: number; p_value: number; alpha: number; verdict: string; n: number }> {
  return apiFetch(`/stats/normality/${encodeURIComponent(col)}`, { method: 'POST' });
}

export async function fetchEntropy(col: string): Promise<{ column: string; entropy_bits: number; max_possible_bits: number; normalized_entropy: number; interpretation: string }> {
  return apiFetch(`/column/entropy/${encodeURIComponent(col)}`, { method: 'POST' });
}

export interface CorrelationResult {
  pearson: Record<string, Record<string, number>>;
  cramers_v: Record<string, Record<string, number | null>>;
  numeric_columns: string[];
  categorical_columns: string[];
}

export async function fetchCorrelation(): Promise<CorrelationResult> {
  return apiFetch('/correlation', { method: 'POST' });
}

// ========================= Outlier Detection =========================
export interface OutlierDetectionResult {
  method: string;
  threshold: string;
  lower_fence: number | null;
  upper_fence: number | null;
  n_outliers: number;
  pct_outliers: number;
  outlier_indices: number[];
  outlier_values: number[];
  n_total: number;
  n_valid: number;
}

export async function fetchOutliers(col: string, method: 'iqr' | 'zscore' = 'iqr'): Promise<OutlierDetectionResult> {
  return apiFetch(`/outliers/${encodeURIComponent(col)}?method=${method}`, { method: 'POST' });
}

// ========================= Imputation =========================
export interface ImputationReport {
  missing_rate: number;
  skewness: number;
  has_outliers: boolean;
  dataset_size: number;
  strong_correlations: Record<string, number>;
  recommended_method: string;
}

export async function fetchImputation(col: string): Promise<ImputationReport> {
  return apiFetch(`/imputation/${encodeURIComponent(col)}/recommend`, { method: 'POST' });
}

export interface ImputationApplyResult {
  column: string;
  method_applied: string;
  fill_value: string;
  n_missing_before: number;
  n_missing_after: number;
  n_filled: number;
}

export async function applyImputation(col: string, method: string): Promise<ImputationApplyResult> {
  return apiFetch(`/imputation/${encodeURIComponent(col)}/apply`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ method }),
  });
}

// ========================= Data Quality =========================
export interface DtypeSuggestion {
  current_dtype: string;
  suggested_dtype: string;
  reason: string;
}

export async function fetchDtypeSuggestions(): Promise<Record<string, DtypeSuggestion>> {
  return apiFetch('/suggest/dtypes', { method: 'POST' });
}

// ========================= ML Readiness =========================
export interface EncodingSuggestion {
  n_unique: number;
  cardinality_ratio: number;
  recommended_encoding: string;
  reason: string;
}

export async function fetchEncodingSuggestions(): Promise<Record<string, EncodingSuggestion>> {
  return apiFetch('/ml/encoding_suggestions', { method: 'POST' });
}

export interface FeatureImportanceResult {
  target: string;
  task_type: 'classification' | 'regression';
  ranking: { feature: string; mutual_info_score: number; importance: 'high' | 'medium' | 'low' }[];
}

export async function fetchFeatureImportance(targetCol: string): Promise<FeatureImportanceResult> {
  return apiFetch(`/ml/feature_importance/${encodeURIComponent(targetCol)}`, { method: 'POST' });
}

export interface ClassBalanceResult {
  column: string;
  n_classes: number;
  total_samples: number;
  class_distribution: { class: string; count: number; ratio: number; pct: number }[];
  is_imbalanced: boolean;
  minority_ratio: number;
  imbalance_warning: string | null;
}

export async function fetchClassBalance(col: string): Promise<ClassBalanceResult> {
  return apiFetch(`/ml/class_balance/${encodeURIComponent(col)}`, { method: 'POST' });
}

// ========================= Auto Report =========================
export async function fetchFullReport(): Promise<Record<string, unknown>> {
  return apiFetch('/report', { method: 'POST' });
}

// ========================= Health =========================
export async function checkConnection(): Promise<boolean> {
  if (!getApiUrl()) return false;
  try {
    const baseUrl = getApiUrl().replace(/\/+$/, '');
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    const res = await fetch(`${baseUrl}/ping`, {
      method: 'POST',
      signal: controller.signal,
      headers: { 'ngrok-skip-browser-warning': 'true' },
    });
    clearTimeout(timeout);
    return res.ok;
  } catch {
    return false;
  }
}
