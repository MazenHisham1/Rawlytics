export interface ColumnInfo {
  name: string;
  type: 'numeric' | 'categorical' | 'datetime' | 'boolean' | 'unknown';
  missingCount: number;
  missingPercent: number;
  uniqueCount: number;
  sampleValues: (string | number | boolean | null)[];
  skewness?: number | null;
  skewnessType?: string | null;
  kurtosis?: number | null;
  kurtosisType?: string | null;
  normalityVerdict?: string | null;
  normalityPValue?: number | null;
  entropy?: number | null;
  entropyInterpretation?: string | null;
}

export interface DatasetStats {
  rowCount: number;
  columnCount: number;
  duplicateRows: number;
  totalMissing: number;
  missingPercent: number;
  memoryUsage: string;
}

export interface OutlierInfo {
  column: string;
  count: number;
  percentage: number;
  lowerBound: number;
  upperBound: number;
}

export interface SkewnessInfo {
  column: string;
  skewness: number;
  interpretation: 'symmetric' | 'moderately_skewed' | 'highly_skewed';
  direction: 'left' | 'right' | 'none';
}

export interface Recommendation {
  id: string;
  type: 'warning' | 'error' | 'success' | 'info';
  category: 'missing' | 'duplicates' | 'outliers' | 'skewness' | 'types' | 'general' | 'imputation';
  title: string;
  description: string;
  action?: string;
  affectedColumns?: string[];
}

export interface ImputationRecommendation {
  column: string;
  missing_rate: number;
  skewness: number;
  has_outliers: boolean;
  dataset_size: number;
  strong_correlations: Record<string, number>;
  recommended_method: string;
}

export interface CorrelationData {
  pearson: Record<string, Record<string, number>>;
  cramers_v: Record<string, Record<string, number | null>>;
  numeric_columns: string[];
  categorical_columns: string[];
}

export interface DtypeSuggestionData {
  [column: string]: {
    current_dtype: string;
    suggested_dtype: string;
    reason: string;
  };
}

export interface EncodingSuggestionData {
  [column: string]: {
    n_unique: number;
    cardinality_ratio: number;
    recommended_encoding: string;
    reason: string;
  };
}

export interface DatasetAnalysis {
  stats: DatasetStats;
  columns: ColumnInfo[];
  outliers: OutlierInfo[];
  skewness: SkewnessInfo[];
  recommendations: Recommendation[];
  imputations: ImputationRecommendation[];
  correlation?: CorrelationData;
  dtypeSuggestions?: DtypeSuggestionData;
  encodingSuggestions?: EncodingSuggestionData;
}

export type ChartType = 'histogram' | 'bar' | 'box' | 'scatter' | 'pie' | 'heatmap';
