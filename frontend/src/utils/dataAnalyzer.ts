import type { ColumnInfo, DatasetStats, OutlierInfo, SkewnessInfo, Recommendation, DatasetAnalysis } from '@/types/dataset';

export function parseCSV(content: string): Record<string, unknown>[] {
  const lines = content.trim().split('\n');
  if (lines.length < 2) return [];

  const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
  const data: Record<string, unknown>[] = [];

  for (let i = 1; i < lines.length; i++) {
    const values = parseCSVLine(lines[i]);
    if (values.length === headers.length) {
      const row: Record<string, unknown> = {};
      headers.forEach((header, index) => {
        row[header] = parseValue(values[index]);
      });
      data.push(row);
    }
  }

  return data;
}

function parseCSVLine(line: string): string[] {
  const values: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      values.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }
  values.push(current.trim());

  return values;
}

function parseValue(value: string): unknown {
  const trimmed = value.trim().replace(/^"|"$/g, '');
  
  if (trimmed === '' || trimmed.toLowerCase() === 'null' || trimmed.toLowerCase() === 'na' || trimmed.toLowerCase() === 'nan') {
    return null;
  }
  
  if (trimmed.toLowerCase() === 'true') return true;
  if (trimmed.toLowerCase() === 'false') return false;
  
  const num = Number(trimmed);
  if (!isNaN(num) && trimmed !== '') return num;
  
  return trimmed;
}

function detectColumnType(values: unknown[]): ColumnInfo['type'] {
  const nonNullValues = values.filter(v => v !== null && v !== undefined);
  if (nonNullValues.length === 0) return 'unknown';

  const types = nonNullValues.map(v => {
    if (typeof v === 'boolean') return 'boolean';
    if (typeof v === 'number') return 'numeric';
    if (typeof v === 'string') {
      if (/^\d{4}-\d{2}-\d{2}/.test(v) || /^\d{2}\/\d{2}\/\d{4}/.test(v)) {
        return 'datetime';
      }
      return 'categorical';
    }
    return 'unknown';
  });

  const typeCounts = types.reduce((acc, type) => {
    acc[type] = (acc[type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const dominantType = Object.entries(typeCounts).sort((a, b) => b[1] - a[1])[0];
  return dominantType ? (dominantType[0] as ColumnInfo['type']) : 'unknown';
}

function calculateOutliers(values: number[]): { count: number; lowerBound: number; upperBound: number } {
  if (values.length < 4) return { count: 0, lowerBound: 0, upperBound: 0 };

  const sorted = [...values].sort((a, b) => a - b);
  const q1Index = Math.floor(sorted.length * 0.25);
  const q3Index = Math.floor(sorted.length * 0.75);
  const q1 = sorted[q1Index];
  const q3 = sorted[q3Index];
  const iqr = q3 - q1;
  const lowerBound = q1 - 1.5 * iqr;
  const upperBound = q3 + 1.5 * iqr;

  const outlierCount = values.filter(v => v < lowerBound || v > upperBound).length;

  return { count: outlierCount, lowerBound, upperBound };
}

function calculateSkewness(values: number[]): number {
  if (values.length < 3) return 0;

  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / n;
  const std = Math.sqrt(variance);

  if (std === 0) return 0;

  const skewness = values.reduce((sum, v) => sum + Math.pow((v - mean) / std, 3), 0) / n;
  return skewness;
}

export function analyzeDataset(data: Record<string, unknown>[]): DatasetAnalysis {
  if (data.length === 0) {
    return {
      stats: { rowCount: 0, columnCount: 0, duplicateRows: 0, totalMissing: 0, missingPercent: 0, memoryUsage: '0 KB' },
      columns: [],
      outliers: [],
      skewness: [],
      recommendations: [],
      imputations: []
    };
  }

  const columnNames = Object.keys(data[0]);
  const rowCount = data.length;
  const columnCount = columnNames.length;

  // Calculate duplicates
  const stringifiedRows = data.map(row => JSON.stringify(row));
  const uniqueRows = new Set(stringifiedRows);
  const duplicateRows = rowCount - uniqueRows.size;

  // Analyze columns
  const columns: ColumnInfo[] = columnNames.map(name => {
    const values = data.map(row => row[name]);
    const missingCount = values.filter(v => v === null || v === undefined || v === '').length;
    const nonNullValues = values.filter(v => v !== null && v !== undefined && v !== '');
    const uniqueCount = new Set(nonNullValues.map(v => String(v))).size;

    return {
      name,
      type: detectColumnType(values),
      missingCount,
      missingPercent: (missingCount / rowCount) * 100,
      uniqueCount,
      sampleValues: nonNullValues.slice(0, 5) as (string | number | boolean | null)[]
    };
  });

  const totalMissing = columns.reduce((sum, col) => sum + col.missingCount, 0);
  const totalCells = rowCount * columnCount;
  const missingPercent = totalCells > 0 ? (totalMissing / totalCells) * 100 : 0;

  // Calculate outliers for numeric columns
  const outliers: OutlierInfo[] = columns
    .filter(col => col.type === 'numeric')
    .map(col => {
      const numericValues = data
        .map(row => row[col.name])
        .filter((v): v is number => typeof v === 'number');
      
      const outlierInfo = calculateOutliers(numericValues);
      return {
        column: col.name,
        count: outlierInfo.count,
        percentage: numericValues.length > 0 ? (outlierInfo.count / numericValues.length) * 100 : 0,
        lowerBound: outlierInfo.lowerBound,
        upperBound: outlierInfo.upperBound
      };
    })
    .filter(o => o.count > 0);

  // Calculate skewness for numeric columns
  const skewness: SkewnessInfo[] = columns
    .filter(col => col.type === 'numeric')
    .map(col => {
      const numericValues = data
        .map(row => row[col.name])
        .filter((v): v is number => typeof v === 'number');
      
      const skew = calculateSkewness(numericValues);
      let interpretation: SkewnessInfo['interpretation'] = 'symmetric';
      let direction: SkewnessInfo['direction'] = 'none';

      if (Math.abs(skew) > 1) {
        interpretation = 'highly_skewed';
        direction = skew > 0 ? 'right' : 'left';
      } else if (Math.abs(skew) > 0.5) {
        interpretation = 'moderately_skewed';
        direction = skew > 0 ? 'right' : 'left';
      }

      return { column: col.name, skewness: skew, interpretation, direction };
    })
    .filter(s => s.interpretation !== 'symmetric');

  // Generate recommendations
  const recommendations: Recommendation[] = [];
  let recId = 1;

  // Missing value recommendations
  const columnsWithMissing = columns.filter(c => c.missingPercent > 0);
  if (columnsWithMissing.length > 0) {
    const highMissing = columnsWithMissing.filter(c => c.missingPercent > 50);
    const moderateMissing = columnsWithMissing.filter(c => c.missingPercent > 10 && c.missingPercent <= 50);
    const lowMissing = columnsWithMissing.filter(c => c.missingPercent <= 10 && c.missingPercent > 0);

    if (highMissing.length > 0) {
      recommendations.push({
        id: `rec-${recId++}`,
        type: 'error',
        category: 'missing',
        title: 'High Missing Values Detected',
        description: `${highMissing.length} column(s) have more than 50% missing values. Consider dropping these columns or investigating data collection issues.`,
        action: 'Drop columns or investigate source',
        affectedColumns: highMissing.map(c => c.name)
      });
    }

    if (moderateMissing.length > 0) {
      recommendations.push({
        id: `rec-${recId++}`,
        type: 'warning',
        category: 'missing',
        title: 'Moderate Missing Values',
        description: `${moderateMissing.length} column(s) have 10-50% missing values. Consider imputation strategies based on column type.`,
        action: 'Use mean/median for numeric, mode for categorical',
        affectedColumns: moderateMissing.map(c => c.name)
      });
    }

    if (lowMissing.length > 0) {
      recommendations.push({
        id: `rec-${recId++}`,
        type: 'info',
        category: 'missing',
        title: 'Low Missing Values',
        description: `${lowMissing.length} column(s) have less than 10% missing values. Simple imputation or row removal may be appropriate.`,
        action: 'Impute or drop rows with missing values',
        affectedColumns: lowMissing.map(c => c.name)
      });
    }
  }

  // Duplicate recommendations
  if (duplicateRows > 0) {
    const duplicatePercent = (duplicateRows / rowCount) * 100;
    recommendations.push({
      id: `rec-${recId++}`,
      type: duplicatePercent > 10 ? 'warning' : 'info',
      category: 'duplicates',
      title: 'Duplicate Rows Detected',
      description: `Found ${duplicateRows} duplicate rows (${duplicatePercent.toFixed(1)}% of data). Consider removing duplicates unless they represent valid repeated observations.`,
      action: 'Review and remove duplicates if appropriate'
    });
  }

  // Outlier recommendations
  if (outliers.length > 0) {
    const significantOutliers = outliers.filter(o => o.percentage > 5);
    if (significantOutliers.length > 0) {
      recommendations.push({
        id: `rec-${recId++}`,
        type: 'warning',
        category: 'outliers',
        title: 'Significant Outliers Detected',
        description: `${significantOutliers.length} numeric column(s) have more than 5% outliers. Investigate if these are data errors or genuine extreme values.`,
        action: 'Winsorize, cap, or remove based on domain knowledge',
        affectedColumns: significantOutliers.map(o => o.column)
      });
    }
  }

  // Skewness recommendations
  const highlySkewed = skewness.filter(s => s.interpretation === 'highly_skewed');
  if (highlySkewed.length > 0) {
    recommendations.push({
      id: `rec-${recId++}`,
      type: 'info',
      category: 'skewness',
      title: 'Highly Skewed Features',
      description: `${highlySkewed.length} column(s) are highly skewed. Consider applying log, sqrt, or Box-Cox transformation for better model performance.`,
      action: 'Apply appropriate transformation',
      affectedColumns: highlySkewed.map(s => s.column)
    });
  }

  // Type-based recommendations
  const categoricalWithHighCardinality = columns.filter(
    c => c.type === 'categorical' && c.uniqueCount > 50
  );
  if (categoricalWithHighCardinality.length > 0) {
    recommendations.push({
      id: `rec-${recId++}`,
      type: 'info',
      category: 'types',
      title: 'High Cardinality Categorical Features',
      description: `${categoricalWithHighCardinality.length} categorical column(s) have more than 50 unique values. Consider grouping rare categories or using target encoding.`,
      action: 'Group categories or use advanced encoding',
      affectedColumns: categoricalWithHighCardinality.map(c => c.name)
    });
  }

  // Success recommendation if data is clean
  if (recommendations.length === 0) {
    recommendations.push({
      id: `rec-${recId++}`,
      type: 'success',
      category: 'general',
      title: 'Dataset Looks Good!',
      description: 'No major data quality issues detected. Your dataset appears to be clean and ready for analysis.',
      action: 'Proceed with modeling'
    });
  }

  const stats: DatasetStats = {
    rowCount,
    columnCount,
    duplicateRows,
    totalMissing,
    missingPercent,
    memoryUsage: `${(JSON.stringify(data).length / 1024).toFixed(1)} KB`
  };

  return { stats, columns, outliers, skewness, recommendations, imputations: [] };
}

export function getNumericColumns(columns: ColumnInfo[]): ColumnInfo[] {
  return columns.filter(c => c.type === 'numeric');
}

export function getCategoricalColumns(columns: ColumnInfo[]): ColumnInfo[] {
  return columns.filter(c => c.type === 'categorical' || c.type === 'boolean');
}
