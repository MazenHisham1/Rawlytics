import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import type { ColumnInfo } from '@/types/dataset';
import { cn } from '@/lib/utils';

interface ColumnTableProps {
  columns: ColumnInfo[];
  onColumnSelect?: (column: ColumnInfo) => void;
  selectedColumn?: string;
}

const typeBadgeClass: Record<ColumnInfo['type'], string> = {
  numeric: 'badge-numeric',
  categorical: 'badge-categorical',
  datetime: 'badge-datetime',
  boolean: 'badge-boolean',
  unknown: 'bg-muted text-muted-foreground'
};

export function ColumnTable({ columns, onColumnSelect, selectedColumn }: ColumnTableProps) {
  return (
    <div className="glass-card overflow-hidden animate-fade-in">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="table-header">
              <th className="px-4 py-3 text-left">Column Name</th>
              <th className="px-4 py-3 text-left">Type</th>
              <th className="px-4 py-3 text-left">Missing</th>
              <th className="px-4 py-3 text-left">Unique</th>
              <th className="px-4 py-3 text-left hidden md:table-cell">Skewness</th>
              <th className="px-4 py-3 text-left hidden lg:table-cell">Kurtosis</th>
              <th className="px-4 py-3 text-left hidden lg:table-cell">Normality</th>
            </tr>
          </thead>
          <tbody>
            {columns.map((column, index) => (
              <tr
                key={column.name}
                className={cn(
                  'table-row cursor-pointer',
                  selectedColumn === column.name && 'bg-primary/10'
                )}
                onClick={() => onColumnSelect?.(column)}
                style={{ animationDelay: `${index * 30}ms` }}
              >
                <td className="px-4 py-3">
                  <span className="font-mono text-sm text-foreground">{column.name}</span>
                </td>
                <td className="px-4 py-3">
                  <Badge className={cn('text-xs', typeBadgeClass[column.type])}>
                    {column.type}
                  </Badge>
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2 min-w-[120px]">
                    <Progress 
                      value={column.missingPercent} 
                      className="h-2 w-16 bg-secondary"
                    />
                    <span className={cn(
                      'text-sm font-mono',
                      column.missingPercent > 50 ? 'text-destructive' :
                      column.missingPercent > 10 ? 'text-warning' :
                      column.missingPercent > 0 ? 'text-muted-foreground' : 'text-success'
                    )}>
                      {column.missingPercent.toFixed(1)}%
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3">
                  <span className="text-sm text-muted-foreground font-mono">
                    {column.uniqueCount.toLocaleString()}
                  </span>
                </td>
                <td className="px-4 py-3 hidden md:table-cell">
                  {column.skewness != null ? (
                    <div className="flex items-center gap-2">
                      <span className={cn(
                        'font-mono text-sm font-medium',
                        column.skewnessType === 'left-skewed' ? 'text-info' :
                        column.skewnessType === 'right-skewed' ? 'text-warning' :
                        'text-success'
                      )}>
                        {column.skewness.toFixed(3)}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {column.skewnessType ?? '—'}
                      </span>
                    </div>
                  ) : column.type === 'categorical' && column.entropy != null ? (
                    <div className="flex items-center gap-1">
                      <span className="font-mono text-sm text-info">{column.entropy.toFixed(3)}</span>
                      <span className="text-xs text-muted-foreground">entropy</span>
                    </div>
                  ) : (
                    <span className="text-xs text-muted-foreground">N/A</span>
                  )}
                </td>
                <td className="px-4 py-3 hidden lg:table-cell">
                  {column.kurtosis != null ? (
                    <div className="flex items-center gap-1">
                      <span className="font-mono text-sm text-foreground">{column.kurtosis.toFixed(3)}</span>
                      <span className="text-xs text-muted-foreground truncate max-w-[100px]" title={column.kurtosisType ?? ''}>
                        {column.kurtosisType?.split(' ')[0] ?? ''}
                      </span>
                    </div>
                  ) : (
                    <span className="text-xs text-muted-foreground">N/A</span>
                  )}
                </td>
                <td className="px-4 py-3 hidden lg:table-cell">
                  {column.normalityVerdict ? (
                    <span className={cn(
                      'px-2 py-0.5 rounded text-xs font-medium',
                      column.normalityVerdict === 'likely normal'
                        ? 'bg-success/10 text-success'
                        : 'bg-warning/10 text-warning'
                    )}>
                      {column.normalityVerdict}
                    </span>
                  ) : (
                    <span className="text-xs text-muted-foreground">N/A</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
