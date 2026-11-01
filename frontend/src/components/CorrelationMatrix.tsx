import { useState } from 'react';
import type { CorrelationData } from '@/types/dataset';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';

interface CorrelationMatrixProps {
  data: CorrelationData;
}

type MatrixType = 'pearson' | 'cramers_v';

function getColor(value: number | null): string {
  if (value === null) return 'bg-muted/30';
  const abs = Math.abs(value);
  if (abs > 0.7) return value > 0 ? 'bg-primary/70 text-primary-foreground' : 'bg-destructive/70 text-destructive-foreground';
  if (abs > 0.4) return value > 0 ? 'bg-primary/40' : 'bg-destructive/40';
  if (abs > 0.2) return value > 0 ? 'bg-primary/20' : 'bg-destructive/20';
  return 'bg-muted/30';
}

export function CorrelationMatrix({ data }: CorrelationMatrixProps) {
  const [matrixType, setMatrixType] = useState<MatrixType>('pearson');

  const hasPearson = data.numeric_columns.length >= 2;
  const hasCramers = data.categorical_columns.length >= 2;

  const matrix = matrixType === 'pearson' ? data.pearson : data.cramers_v;
  const columns = matrixType === 'pearson' ? data.numeric_columns : data.categorical_columns;

  if (!hasPearson && !hasCramers) {
    return (
      <div className="glass-card p-6 text-center text-muted-foreground">
        Not enough columns for correlation analysis (need at least 2 numeric or 2 categorical columns).
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        {hasPearson && (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={() => setMatrixType('pearson')}
                className={cn(
                  'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
                  matrixType === 'pearson' ? 'bg-primary text-primary-foreground' : 'bg-secondary text-muted-foreground hover:text-foreground'
                )}
              >
                Pearson (Numeric)
              </button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Show numeric correlations</p>
            </TooltipContent>
          </Tooltip>
        )}
        {hasCramers && (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={() => setMatrixType('cramers_v')}
                className={cn(
                  'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
                  matrixType === 'cramers_v' ? 'bg-primary text-primary-foreground' : 'bg-secondary text-muted-foreground hover:text-foreground'
                )}
              >
                Cramér's V (Categorical)
              </button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Show categorical associations</p>
            </TooltipContent>
          </Tooltip>
        )}
      </div>

      <div className="glass-card p-4 overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr>
              <th className="p-1 text-left text-muted-foreground font-mono"></th>
              {columns.map(col => (
                <th key={col} className="p-1 text-center text-muted-foreground font-mono truncate max-w-[80px]" title={col}>
                  {col.length > 8 ? col.slice(0, 8) + '…' : col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {columns.map(row => (
              <tr key={row}>
                <td className="p-1 font-mono text-muted-foreground truncate max-w-[100px]" title={row}>
                  {row.length > 10 ? row.slice(0, 10) + '…' : row}
                </td>
                {columns.map(col => {
                  const val = matrix[row]?.[col] ?? null;
                  return (
                    <td
                      key={col}
                      className={cn('p-1 text-center rounded-sm font-mono', getColor(val))}
                      title={`${row} × ${col}: ${val?.toFixed(4) ?? 'N/A'}`}
                    >
                      {val !== null ? val.toFixed(2) : '—'}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center gap-4 text-xs text-muted-foreground">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded bg-destructive/70 inline-block" /> Strong negative
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded bg-muted/30 inline-block" /> Weak / none
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded bg-primary/70 inline-block" /> Strong positive
        </span>
      </div>
    </div>
  );
}
