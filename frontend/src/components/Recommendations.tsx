import { Beaker } from 'lucide-react';
import type { ImputationRecommendation } from '@/types/dataset';
import { cn } from '@/lib/utils';

interface RecommendationsProps {
  imputations: ImputationRecommendation[];
}

const styleMap = {
  warning: 'recommendation-warning',
  error: 'recommendation-error',
  info: 'recommendation-info'
};

const iconColorMap = {
  warning: 'text-warning',
  error: 'text-destructive',
  info: 'text-info'
};

export function Recommendations({ imputations }: RecommendationsProps) {
  if (imputations.length === 0) {
    return (
      <div className="recommendation-card recommendation-success">
        <div className="flex items-start gap-3">
          <Beaker className="w-5 h-5 flex-shrink-0 mt-0.5 text-success" />
          <div>
            <h4 className="font-medium text-foreground">No Imputation Needed</h4>
            <p className="text-sm text-muted-foreground">No numeric columns with missing values were found.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {imputations.map((imp, index) => {
        const correlationEntries = Object.entries(imp.strong_correlations);
        const severity: 'warning' | 'error' | 'info' =
          imp.missing_rate >= 0.3 ? 'error' : imp.missing_rate >= 0.05 ? 'warning' : 'info';

        return (
          <div
            key={imp.column}
            className={cn('recommendation-card animate-slide-up', styleMap[severity])}
            style={{ animationDelay: `${index * 100}ms` }}
          >
            <div className="flex items-start gap-3">
              <Beaker className={cn('w-5 h-5 flex-shrink-0 mt-0.5', iconColorMap[severity])} />
              <div className="flex-1 space-y-2">
                <h4 className="font-medium text-foreground">
                  <span className="font-mono text-primary">{imp.column}</span>
                  <span className="mx-2">→</span>
                  <span className="text-accent-foreground">{imp.recommended_method}</span>
                </h4>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                  <div className="bg-secondary/50 rounded px-2 py-1">
                    <span className="text-muted-foreground">Missing: </span>
                    <span className="font-medium text-foreground">{(imp.missing_rate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="bg-secondary/50 rounded px-2 py-1">
                    <span className="text-muted-foreground">Skew: </span>
                    <span className="font-medium text-foreground">{imp.skewness.toFixed(3)}</span>
                  </div>
                  <div className="bg-secondary/50 rounded px-2 py-1">
                    <span className="text-muted-foreground">Outliers: </span>
                    <span className={cn('font-medium', imp.has_outliers ? 'text-warning' : 'text-success')}>
                      {imp.has_outliers ? 'Yes' : 'No'}
                    </span>
                  </div>
                  <div className="bg-secondary/50 rounded px-2 py-1">
                    <span className="text-muted-foreground">N: </span>
                    <span className="font-medium text-foreground">{imp.dataset_size}</span>
                  </div>
                </div>
                {correlationEntries.length > 0 && (
                  <div className="space-y-1">
                    <span className="text-xs text-muted-foreground">Strong correlations:</span>
                    <div className="flex flex-wrap gap-1">
                      {correlationEntries.map(([col, val]) => (
                        <span
                          key={col}
                          className="px-2 py-0.5 bg-primary/10 rounded text-xs font-mono text-primary"
                        >
                          {col}: {(val as number).toFixed(2)}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
