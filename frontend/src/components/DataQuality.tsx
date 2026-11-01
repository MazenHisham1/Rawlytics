import { ArrowRightLeft, CheckCircle } from 'lucide-react';
import type { DtypeSuggestionData } from '@/types/dataset';
import { cn } from '@/lib/utils';

interface DataQualityProps {
  dtypeSuggestions?: DtypeSuggestionData;
}

export function DataQuality({ dtypeSuggestions }: DataQualityProps) {
  if (!dtypeSuggestions || Object.keys(dtypeSuggestions).length === 0) {
    return (
      <div className="glass-card p-6 text-center text-muted-foreground">
        No data type suggestions available.
      </div>
    );
  }

  const actionable = Object.entries(dtypeSuggestions).filter(
    ([, info]) => info.suggested_dtype !== 'keep as-is'
  );
  const keepAsIs = Object.entries(dtypeSuggestions).filter(
    ([, info]) => info.suggested_dtype === 'keep as-is'
  );

  return (
    <div className="space-y-4">
      {actionable.length > 0 && (
        <div className="glass-card p-5 space-y-3">
          <h4 className="font-semibold text-foreground flex items-center gap-2">
            <ArrowRightLeft className="w-4 h-4 text-warning" />
            Suggested Type Changes ({actionable.length})
          </h4>
          <div className="space-y-2">
            {actionable.map(([col, info]) => (
              <div
                key={col}
                className="flex flex-col sm:flex-row sm:items-center gap-2 p-3 bg-warning/5 border border-warning/20 rounded-lg text-sm"
              >
                <span className="font-mono text-primary font-medium min-w-[120px]">{col}</span>
                <div className="flex items-center gap-2 flex-1">
                  <span className="px-2 py-0.5 bg-secondary rounded text-xs font-mono">{info.current_dtype}</span>
                  <span className="text-muted-foreground">→</span>
                  <span className="px-2 py-0.5 bg-primary/10 text-primary rounded text-xs font-mono">{info.suggested_dtype}</span>
                </div>
                <span className="text-xs text-muted-foreground">{info.reason}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {keepAsIs.length > 0 && (
        <div className="glass-card p-5 space-y-3">
          <h4 className="font-semibold text-foreground flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-success" />
            Columns with Correct Types ({keepAsIs.length})
          </h4>
          <div className="flex flex-wrap gap-2">
            {keepAsIs.map(([col, info]) => (
              <span key={col} className="px-3 py-1 bg-secondary/50 rounded-lg text-xs font-mono text-muted-foreground">
                {col} <span className={cn('text-foreground')}>({info.current_dtype})</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
