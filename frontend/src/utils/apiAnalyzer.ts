import {
  fetchInfo, fetchNulls, fetchDuplicates, fetchDescribe,
  fetchColumnUnique, fetchColumnSkewness, fetchImputation,
  fetchColumnKurtosis, fetchNormality, fetchEntropy,
  fetchCorrelation, fetchOutliers, fetchDtypeSuggestions,
  fetchEncodingSuggestions
} from '@/services/api';
import type { DatasetAnalysis, ColumnInfo, DatasetStats, OutlierInfo, SkewnessInfo, ImputationRecommendation } from '@/types/dataset';

function mapPandasDtype(dtype: string): ColumnInfo['type'] {
  if (dtype.includes('int') || dtype.includes('float')) return 'numeric';
  if (dtype.includes('bool')) return 'boolean';
  if (dtype.includes('datetime')) return 'datetime';
  if (dtype.includes('object') || dtype.includes('str') || dtype.includes('category')) return 'categorical';
  return 'unknown';
}

export async function analyzeFromApi(): Promise<DatasetAnalysis> {
  // Fetch all basic stats + new endpoints in parallel
  const [info, nulls, dups, describe, correlationData, dtypeSuggestions, encodingSuggestions] = await Promise.all([
    fetchInfo(),
    fetchNulls(),
    fetchDuplicates(),
    fetchDescribe(),
    fetchCorrelation().catch(() => null),
    fetchDtypeSuggestions().catch(() => null),
    fetchEncodingSuggestions().catch(() => null),
  ]);

  const columnNames = Object.keys(info.dtypes);
  const numericCols = columnNames.filter(name => mapPandasDtype(info.dtypes[name]) === 'numeric');
  const categoricalCols = columnNames.filter(name => mapPandasDtype(info.dtypes[name]) === 'categorical');

  // Fetch per-column data in parallel
  const [uniqueResults, skewnessResults, kurtosisResults, normalityResults, entropyResults, outlierResults] = await Promise.all([
    // Unique counts for all columns
    Promise.all(
      columnNames.map(col =>
        fetchColumnUnique(col)
          .then(r => ({ col, unique: r.unique }))
          .catch(() => ({ col, unique: 0 }))
      )
    ),
    // Skewness for numeric columns
    Promise.all(
      numericCols.map(col =>
        fetchColumnSkewness(col)
          .then(r => ({ col, skewness: r.skewness, type: r.type }))
          .catch(() => ({ col, skewness: null as number | null, type: null as string | null }))
      )
    ),
    // Kurtosis for numeric columns
    Promise.all(
      numericCols.map(col =>
        fetchColumnKurtosis(col)
          .then(r => ({ col, kurtosis: r.kurtosis, type: r.type }))
          .catch(() => ({ col, kurtosis: null as number | null, type: null as string | null }))
      )
    ),
    // Normality for numeric columns
    Promise.all(
      numericCols.map(col =>
        fetchNormality(col)
          .then(r => ({ col, verdict: r.verdict, p_value: r.p_value }))
          .catch(() => ({ col, verdict: null as string | null, p_value: null as number | null }))
      )
    ),
    // Entropy for categorical columns
    Promise.all(
      categoricalCols.map(col =>
        fetchEntropy(col)
          .then(r => ({ col, entropy: r.normalized_entropy, interpretation: r.interpretation }))
          .catch(() => ({ col, entropy: null as number | null, interpretation: null as string | null }))
      )
    ),
    // Outlier detection for numeric columns
    Promise.all(
      numericCols.map(col =>
        fetchOutliers(col, 'iqr')
          .then(r => ({ col, ...r }))
          .catch(() => null)
      )
    ),
  ]);

  const uniqueMap = Object.fromEntries(uniqueResults.map(u => [u.col, u.unique]));
  const skewnessMap = Object.fromEntries(skewnessResults.map(s => [s.col, { skewness: s.skewness, type: s.type }]));
  const kurtosisMap = Object.fromEntries(kurtosisResults.map(k => [k.col, { kurtosis: k.kurtosis, type: k.type }]));
  const normalityMap = Object.fromEntries(normalityResults.map(n => [n.col, { verdict: n.verdict, p_value: n.p_value }]));
  const entropyMap = Object.fromEntries(entropyResults.map(e => [e.col, { entropy: e.entropy, interpretation: e.interpretation }]));

  // Build column info
  const columns: ColumnInfo[] = columnNames.map(name => {
    const nullPercent = nulls[name] || 0;
    const missingCount = Math.round((nullPercent / 100) * info.rows);
    const skewData = skewnessMap[name];
    const kurtData = kurtosisMap[name];
    const normData = normalityMap[name];
    const entData = entropyMap[name];
    return {
      name,
      type: mapPandasDtype(info.dtypes[name]),
      missingCount,
      missingPercent: nullPercent,
      uniqueCount: uniqueMap[name] || 0,
      sampleValues: [],
      skewness: skewData?.skewness ?? null,
      skewnessType: skewData?.type ?? null,
      kurtosis: kurtData?.kurtosis ?? null,
      kurtosisType: kurtData?.type ?? null,
      normalityVerdict: normData?.verdict ?? null,
      normalityPValue: normData?.p_value ?? null,
      entropy: entData?.entropy ?? null,
      entropyInterpretation: entData?.interpretation ?? null,
    };
  });

  // Build outliers from real backend data
  const outliers: OutlierInfo[] = outlierResults
    .filter((r): r is NonNullable<typeof r> => r !== null && r.n_outliers > 0)
    .map(r => ({
      column: r.col,
      count: r.n_outliers,
      percentage: r.pct_outliers,
      lowerBound: r.lower_fence ?? 0,
      upperBound: r.upper_fence ?? 0,
    }));

  // Build skewness info
  const skewness: SkewnessInfo[] = [];
  for (const col of columns.filter(c => c.type === 'numeric')) {
    if (col.skewness == null) continue;
    const skew = col.skewness;
    let interpretation: 'symmetric' | 'moderately_skewed' | 'highly_skewed' = 'symmetric';
    let direction: 'left' | 'right' | 'none' = 'none';
    if (Math.abs(skew) > 1) {
      interpretation = 'highly_skewed';
      direction = skew > 0 ? 'right' : 'left';
    } else if (Math.abs(skew) > 0.5) {
      interpretation = 'moderately_skewed';
      direction = skew > 0 ? 'right' : 'left';
    }
    if (interpretation !== 'symmetric') {
      skewness.push({ column: col.name, skewness: skew, interpretation, direction });
    }
  }

  // Fetch imputation recommendations for numeric columns with missing values
  const colsWithMissing = columns.filter(c => c.type === 'numeric' && c.missingCount > 0);
  const imputationResults = await Promise.all(
    colsWithMissing.map(col =>
      fetchImputation(col.name)
        .then(r => ({
          column: col.name,
          missing_rate: r.missing_rate,
          skewness: r.skewness,
          has_outliers: r.has_outliers,
          dataset_size: r.dataset_size,
          strong_correlations: r.strong_correlations,
          recommended_method: r.recommended_method,
        } as ImputationRecommendation))
        .catch(() => null)
    )
  );
  const imputations: ImputationRecommendation[] = imputationResults.filter((r): r is ImputationRecommendation => r !== null);

  const totalMissing = columns.reduce((sum, col) => sum + col.missingCount, 0);
  const totalCells = info.rows * info.columns;

  const stats: DatasetStats = {
    rowCount: info.rows,
    columnCount: info.columns,
    duplicateRows: dups.duplicates,
    totalMissing,
    missingPercent: totalCells > 0 ? (totalMissing / totalCells) * 100 : 0,
    memoryUsage: 'Server-side'
  };

  return {
    stats,
    columns,
    outliers,
    skewness,
    recommendations: [],
    imputations,
    correlation: correlationData ?? undefined,
    dtypeSuggestions: dtypeSuggestions ?? undefined,
    encodingSuggestions: encodingSuggestions ?? undefined,
  };
}
