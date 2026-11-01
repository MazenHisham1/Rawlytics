import { useState } from 'react';
import { AlertTriangle, Search } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { fetchOutliers, type OutlierDetectionResult } from '@/services/api';
import type { ColumnInfo } from '@/types/dataset';
import { cn } from '@/lib/utils';

interface OutlierPanelProps {
  columns: ColumnInfo[];
}

export function OutlierPanel({ columns }: OutlierPanelProps) {
  const numericCols = columns.filter(c => c.type === 'numeric');
  const [selectedCol, setSelectedCol] = useState(numericCols[0]?.name || '');
  const [method, setMethod] = useState<'iqr' | 'zscore'>('iqr');
  const [result, setResult] = useState<OutlierDetectionResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleDetect = async () => {
    if (!selectedCol) return;
    setLoading(true);
    try {
      const data = await fetchOutliers(selectedCol, method);
      setResult(data);
    } catch {
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Select value={selectedCol} onValueChange={setSelectedCol}>
          <SelectTrigger className="w-[180px] bg-secondary border-border">
            <SelectValue placeholder="Select column" />
          </SelectTrigger>
          <SelectContent>
            {numericCols.map(col => (
              <SelectItem key={col.name} value={col.name}>{col.name}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={method} onValueChange={(v) => setMethod(v as 'iqr' | 'zscore')}>
          <SelectTrigger className="w-[140px] bg-secondary border-border">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="iqr">IQR Method</SelectItem>
            <SelectItem value="zscore">Z-Score</SelectItem>
          </SelectContent>
        </Select>

        <Tooltip>
          <TooltipTrigger asChild>
            <button
              onClick={handleDetect}
              disabled={loading || !selectedCol}
              className={cn(
                'px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2',
                'bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50'
              )}
            >
              <Search className="w-4 h-4" />
              {loading ? 'Detecting...' : 'Detect Outliers'}
            </button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Run outlier detection on selected column</p>
          </TooltipContent>
        </Tooltip>
      </div>

      {result && (
        <div className="glass-card p-5 space-y-4 animate-fade-in">
          <div className="flex items-center gap-2">
            <AlertTriangle className={cn('w-5 h-5', result.n_outliers > 0 ? 'text-warning' : 'text-success')} />
            <h4 className="font-semibold text-foreground">
              {result.n_outliers} outliers found ({result.pct_outliers}%)
            </h4>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
            <div className="bg-secondary/50 rounded-lg p-3">
              <span className="text-muted-foreground block text-xs">Method</span>
              <span className="font-medium text-foreground">{result.method.toUpperCase()}</span>
            </div>
            <div className="bg-secondary/50 rounded-lg p-3">
              <span className="text-muted-foreground block text-xs">Threshold</span>
              <span className="font-mono text-foreground text-xs">{result.threshold}</span>
            </div>
            <div className="bg-secondary/50 rounded-lg p-3">
              <span className="text-muted-foreground block text-xs">Valid Values</span>
              <span className="font-medium text-foreground">{result.n_valid.toLocaleString()}</span>
            </div>
            <div className="bg-secondary/50 rounded-lg p-3">
              <span className="text-muted-foreground block text-xs">Total Rows</span>
              <span className="font-medium text-foreground">{result.n_total.toLocaleString()}</span>
            </div>
          </div>

          {result.outlier_values.length > 0 && (
            <div>
              <span className="text-xs text-muted-foreground block mb-2">
                Outlier values (showing up to 20):
              </span>
              <div className="flex flex-wrap gap-1.5">
                {result.outlier_values.slice(0, 20).map((val, i) => (
                  <span key={i} className="px-2 py-0.5 bg-warning/10 text-warning rounded text-xs font-mono">
                    {val.toFixed(2)}
                  </span>
                ))}
                {result.outlier_values.length > 20 && (
                  <span className="px-2 py-0.5 text-muted-foreground text-xs">
                    +{result.outlier_values.length - 20} more
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
