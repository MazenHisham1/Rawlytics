import { Database, Columns, Copy, AlertTriangle, HardDrive } from 'lucide-react';
import type { DatasetStats } from '@/types/dataset';

interface StatsOverviewProps {
  stats: DatasetStats;
}

export function StatsOverview({ stats }: StatsOverviewProps) {
  const statCards = [
    {
      icon: Database,
      label: 'Total Rows',
      value: stats.rowCount.toLocaleString(),
      color: 'text-primary'
    },
    {
      icon: Columns,
      label: 'Columns',
      value: stats.columnCount.toString(),
      color: 'text-info'
    },
    {
      icon: Copy,
      label: 'Duplicates',
      value: stats.duplicateRows.toLocaleString(),
      color: stats.duplicateRows > 0 ? 'text-warning' : 'text-success'
    },
    {
      icon: AlertTriangle,
      label: 'Missing Values',
      value: `${stats.missingPercent.toFixed(1)}%`,
      color: stats.missingPercent > 10 ? 'text-destructive' : stats.missingPercent > 0 ? 'text-warning' : 'text-success'
    },
    {
      icon: HardDrive,
      label: 'Memory',
      value: stats.memoryUsage,
      color: 'text-muted-foreground'
    }
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
      {statCards.map((stat, index) => (
        <div
          key={stat.label}
          className="stat-card animate-slide-up"
          style={{ animationDelay: `${index * 50}ms` }}
        >
          <div className="flex items-center gap-2">
            <stat.icon className={`w-4 h-4 ${stat.color}`} />
            <span className="stat-label">{stat.label}</span>
          </div>
          <span className={`stat-value ${stat.color}`}>{stat.value}</span>
        </div>
      ))}
    </div>
  );
}
