import { useState } from 'react';
import { Wand2, BarChart3, Target } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { fetchFeatureImportance, fetchClassBalance, type FeatureImportanceResult, type ClassBalanceResult } from '@/services/api';
import type { ColumnInfo, EncodingSuggestionData } from '@/types/dataset';
import { cn } from '@/lib/utils';

interface MLReadinessProps {
  columns: ColumnInfo[];
  encodingSuggestions?: EncodingSuggestionData;
}

export function MLReadiness({ columns, encodingSuggestions }: MLReadinessProps) {
  const [targetCol, setTargetCol] = useState('');
  const [fiResult, setFiResult] = useState<FeatureImportanceResult | null>(null);
  const [cbResult, setCbResult] = useState<ClassBalanceResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!targetCol) return;
    setLoading(true);
    try {
      const [fi, cb] = await Promise.all([
        fetchFeatureImportance(targetCol).catch(() => null),
        fetchClassBalance(targetCol).catch(() => null),
      ]);
      setFiResult(fi);
      setCbResult(cb);
    } finally {
      setLoading(false);
    }
  };

  const importanceColor = (imp: string) => {
    if (imp === 'high') return 'text-success bg-success/10';
    if (imp === 'medium') return 'text-warning bg-warning/10';
    return 'text-muted-foreground bg-muted/30';
  };

  return (
    <div className="space-y-6">
      {/* Encoding Suggestions */}
      {encodingSuggestions && Object.keys(encodingSuggestions).length > 0 && (
        <div className="glass-card p-5 space-y-3">
          <h4 className="font-semibold text-foreground flex items-center gap-2">
            <Wand2 className="w-4 h-4 text-primary" />
            Encoding Suggestions
          </h4>
          <div className="space-y-2">
            {Object.entries(encodingSuggestions).map(([col, info]) => (
              <div key={col} className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 p-2 bg-secondary/30 rounded-lg text-sm">
                <span className="font-mono text-primary">{col}</span>
                <span className="text-foreground font-medium">{info.recommended_encoding}</span>
                <span className="text-xs text-muted-foreground">{info.reason}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Feature Importance */}
      <div className="glass-card p-5 space-y-4">
        <h4 className="font-semibold text-foreground flex items-center gap-2">
          <Target className="w-4 h-4 text-primary" />
          Feature Importance
        </h4>
        <div className="flex flex-wrap items-center gap-3">
          <Select value={targetCol} onValueChange={setTargetCol}>
            <SelectTrigger className="w-[200px] bg-secondary border-border">
              <SelectValue placeholder="Select target column" />
            </SelectTrigger>
            <SelectContent>
              {columns.map(col => (
                <SelectItem key={col.name} value={col.name}>{col.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={handleAnalyze}
                disabled={loading || !targetCol}
                className={cn(
                  'px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2',
                  'bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50'
                )}
              >
                <BarChart3 className="w-4 h-4" />
                {loading ? 'Analyzing...' : 'Analyze'}
              </button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Analyze feature importance for target column</p>
            </TooltipContent>
          </Tooltip>
        </div>

        {fiResult && (
          <div className="space-y-2 animate-fade-in">
            <p className="text-xs text-muted-foreground">
              Task type: <span className="font-medium text-foreground">{fiResult.task_type}</span>
            </p>
            <div className="space-y-1.5">
              {fiResult.ranking.map((item, i) => (
                <div key={item.feature} className="flex items-center gap-3 animate-slide-up" style={{ animationDelay: `${i * 50}ms` }}>
                  <span className="font-mono text-sm text-muted-foreground w-[140px] truncate" title={item.feature}>
                    {item.feature}
                  </span>
                  <div className="flex-1 h-6 bg-secondary/50 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary/60 rounded-full transition-all duration-500"
                      style={{ width: `${Math.min(item.mutual_info_score * 100, 100)}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono w-[50px] text-right">{item.mutual_info_score.toFixed(3)}</span>
                  <span className={cn('px-2 py-0.5 rounded text-xs font-medium', importanceColor(item.importance))}>
                    {item.importance}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {cbResult && (
          <div className="space-y-3 animate-fade-in border-t border-border pt-4">
            <h5 className="text-sm font-medium text-foreground flex items-center gap-2">
              Class Balance
              {cbResult.is_imbalanced && (
                <span className="px-2 py-0.5 bg-warning/10 text-warning text-xs rounded-full">Imbalanced</span>
              )}
            </h5>
            <div className="space-y-1.5">
              {cbResult.class_distribution.map(cls => (
                <div key={cls.class} className="flex items-center gap-3 text-sm">
                  <span className="font-mono text-muted-foreground w-[100px] truncate">{cls.class}</span>
                  <div className="flex-1 h-5 bg-secondary/50 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-info/50 rounded-full"
                      style={{ width: `${cls.pct}%` }}
                    />
                  </div>
                  <span className="text-xs w-[80px] text-right text-muted-foreground">
                    {cls.count.toLocaleString()} ({cls.pct}%)
                  </span>
                </div>
              ))}
            </div>
            {cbResult.imbalance_warning && (
              <p className="text-xs text-warning bg-warning/5 p-2 rounded">{cbResult.imbalance_warning}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
