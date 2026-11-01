import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, ScatterChart, Scatter, CartesianGrid, Legend,
  AreaChart, Area
} from 'recharts';
import { BarChart3, PieChart as PieIcon, ScatterChart as ScatterIcon, TrendingUp, BoxSelect } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tooltip as TooltipUI, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { BoxPlotChart } from '@/components/BoxPlotChart';
import {
  fetchChartHist, fetchChartBar, fetchChartPie,
  fetchChartScatter, fetchChartBoxplot
} from '@/services/api';
import type { ColumnInfo, ChartType } from '@/types/dataset';
import { cn } from '@/lib/utils';

interface ChartPanelProps {
  data: Record<string, unknown>[];
  columns: ColumnInfo[];
}

const COLORS = [
  'hsl(174, 72%, 50%)',
  'hsl(199, 89%, 48%)',
  'hsl(142, 70%, 45%)',
  'hsl(38, 92%, 50%)',
  'hsl(280, 65%, 60%)',
  'hsl(350, 80%, 60%)'
];

const TOOLTIP_STYLE = {
  backgroundColor: 'hsl(222, 47%, 9%)',
  border: '1px solid hsl(222, 30%, 18%)',
  borderRadius: '8px',
  color: 'hsl(210, 40%, 98%)'
};

export function ChartPanel({ columns }: ChartPanelProps) {
  const numericColumns = columns.filter(c => c.type === 'numeric');

  const [chartType, setChartType] = useState<ChartType>('histogram');
  const [selectedColumn, setSelectedColumn] = useState<string>(columns[0]?.name || '');
  const [secondColumn, setSecondColumn] = useState<string>(numericColumns[1]?.name || numericColumns[0]?.name || '');
  const [chartData, setChartData] = useState<unknown[]>([]);
  const [boxData, setBoxData] = useState<{ min: number; q1: number; median: number; q3: number; max: number; outliers: number[] } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch chart data from backend whenever chart type or columns change
  useEffect(() => {
    if (!selectedColumn) return;

    let cancelled = false;
    setLoading(true);
    setError(null);
    setBoxData(null);

    const fetchData = async () => {
      try {
        let result: unknown;

        switch (chartType) {
          case 'histogram':
            result = await fetchChartHist(selectedColumn, 20);
            break;
          case 'bar':
            result = await fetchChartBar(selectedColumn);
            break;
          case 'pie':
            result = await fetchChartPie(selectedColumn);
            break;
          case 'scatter':
            if (!secondColumn) return;
            result = await fetchChartScatter(selectedColumn, secondColumn);
            break;
          case 'box':
            result = await fetchChartBoxplot(selectedColumn);
            break;
          default:
            result = [];
        }

        if (cancelled) return;

        if (chartType === 'box') {
          setBoxData(result as typeof boxData);
          setChartData([]);
        } else {
          setChartData(result as unknown[]);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to fetch chart data');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    fetchData();
    return () => { cancelled = true; };
  }, [chartType, selectedColumn, secondColumn]);

  const chartButtons = [
    { type: 'histogram' as ChartType, icon: TrendingUp, label: 'Histogram' },
    { type: 'bar' as ChartType, icon: BarChart3, label: 'Bar Chart' },
    { type: 'pie' as ChartType, icon: PieIcon, label: 'Pie Chart' },
    { type: 'scatter' as ChartType, icon: ScatterIcon, label: 'Scatter' },
    { type: 'box' as ChartType, icon: BoxSelect, label: 'Box Plot' }
  ];

  const availableColumns = (chartType === 'histogram' || chartType === 'scatter' || chartType === 'box')
    ? numericColumns
    : columns;

  return (
    <div className="space-y-4 animate-fade-in">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex gap-1 p-1 bg-secondary/50 rounded-lg">
          {chartButtons.map(({ type, icon: Icon, label }) => (
            <TooltipUI key={type}>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className={cn(
                    'gap-2 transition-all',
                    chartType === type && 'bg-primary text-primary-foreground hover:bg-primary/90'
                  )}
                  onClick={() => setChartType(type)}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{label}</span>
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Switch to {label.toLowerCase()}</p>
              </TooltipContent>
            </TooltipUI>
          ))}
        </div>

        <Select value={selectedColumn || undefined} onValueChange={setSelectedColumn}>
          <SelectTrigger className="w-[180px] bg-secondary border-border">
            <SelectValue placeholder="Select column" />
          </SelectTrigger>
          <SelectContent>
            {availableColumns.filter(col => col.name).map(col => (
              <SelectItem key={col.name} value={col.name}>
                {col.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {chartType === 'scatter' && numericColumns.length > 0 && (
          <Select value={secondColumn || undefined} onValueChange={setSecondColumn}>
            <SelectTrigger className="w-[180px] bg-secondary border-border">
              <SelectValue placeholder="Y-axis column" />
            </SelectTrigger>
            <SelectContent>
              {numericColumns.map(col => (
                <SelectItem key={col.name} value={col.name}>
                  {col.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </div>

      <div className="chart-container">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-full text-destructive text-sm">
            {error}
          </div>
        ) : chartType === 'box' && boxData ? (
          <BoxPlotChart data={boxData} columnName={selectedColumn} />
        ) : chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            {chartType === 'histogram' ? (
              <AreaChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                <defs>
                  <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(174, 72%, 50%)" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="hsl(174, 72%, 50%)" stopOpacity={0.1} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(222, 30%, 18%)" />
                <XAxis dataKey="label" stroke="hsl(215, 20%, 55%)" fontSize={11} angle={-45} textAnchor="end" height={60} />
                <YAxis stroke="hsl(215, 20%, 55%)" fontSize={12} />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Area type="monotone" dataKey="value" stroke="hsl(174, 72%, 50%)" strokeWidth={2} fillOpacity={1} fill="url(#colorCount)" />
              </AreaChart>
            ) : chartType === 'bar' ? (
              <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(222, 30%, 18%)" />
                <XAxis dataKey="label" stroke="hsl(215, 20%, 55%)" fontSize={11} angle={-45} textAnchor="end" height={60} />
                <YAxis stroke="hsl(215, 20%, 55%)" fontSize={12} />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Bar dataKey="value" fill="hsl(199, 89%, 48%)" radius={[4, 4, 0, 0]} />
              </BarChart>
            ) : chartType === 'pie' ? (
              <PieChart>
                <Pie
                  data={chartData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                  nameKey="label"
                  label={({ label, percent }: { label: string; percent: number }) => `${label} (${(percent * 100).toFixed(0)}%)`}
                >
                  {(chartData as { label: string; value: number }[]).map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Legend />
              </PieChart>
            ) : chartType === 'scatter' ? (
              <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(222, 30%, 18%)" />
                <XAxis type="number" dataKey="x" name={selectedColumn} stroke="hsl(215, 20%, 55%)" fontSize={12} />
                <YAxis type="number" dataKey="y" name={secondColumn} stroke="hsl(215, 20%, 55%)" fontSize={12} />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Scatter data={chartData} fill="hsl(174, 72%, 50%)" />
              </ScatterChart>
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground">
                Select a column to visualize
              </div>
            )}
          </ResponsiveContainer>
        ) : (
          <div className="flex items-center justify-center h-full text-muted-foreground">
            Select a column to visualize
          </div>
        )}
      </div>
    </div>
  );
}
