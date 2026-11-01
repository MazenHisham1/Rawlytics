import { useMemo } from 'react';

interface BoxPlotData {
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  outliers: number[];
}

interface BoxPlotChartProps {
  data: BoxPlotData;
  columnName: string;
}

export function BoxPlotChart({ data, columnName }: BoxPlotChartProps) {
  const layout = useMemo(() => {
    const padding = 40;
    const width = 800;
    const height = 400;
    const plotLeft = padding + 60;
    const plotRight = width - padding;
    const plotWidth = plotRight - plotLeft;
    const centerY = height / 2;
    const boxHeight = 120;

    // Scale values to pixel positions
    const outliers = data.outliers ?? [];
    const allValues = [data.min, data.max, ...outliers];
    const dataMin = Math.min(...allValues);
    const dataMax = Math.max(...allValues);
    const range = dataMax - dataMin || 1;
    const margin = range * 0.1;

    const scale = (val: number) =>
      plotLeft + ((val - (dataMin - margin)) / (range + 2 * margin)) * plotWidth;

    return {
      width,
      height,
      centerY,
      boxHeight,
      minX: scale(data.min),
      q1X: scale(data.q1),
      medianX: scale(data.median),
      q3X: scale(data.q3),
      maxX: scale(data.max),
      outlierXs: (data.outliers ?? []).map(o => scale(o)),
      scale,
      dataMin: dataMin - margin,
      dataMax: dataMax + margin,
      plotLeft,
      plotRight,
      plotWidth,
    };
  }, [data]);

  // Generate tick values
  const ticks = useMemo(() => {
    const count = 6;
    const range = layout.dataMax - layout.dataMin;
    const step = range / (count - 1);
    return Array.from({ length: count }, (_, i) => {
      const val = layout.dataMin + i * step;
      return {
        value: val,
        x: layout.scale(val),
        label: val.toFixed(1),
      };
    });
  }, [layout]);

  return (
    <div className="flex flex-col items-center gap-3 w-full h-full justify-center">
      <p className="text-base font-semibold text-muted-foreground">{columnName}</p>
      <svg
        viewBox={`0 0 ${layout.width} ${layout.height}`}
        className="w-full"
        style={{ minHeight: '400px' }}
      >
        {/* Whisker line (min to max) */}
        <line
          x1={layout.minX}
          y1={layout.centerY}
          x2={layout.maxX}
          y2={layout.centerY}
          stroke="hsl(215, 20%, 55%)"
          strokeWidth={1.5}
        />
        {/* Min whisker cap */}
        <line
          x1={layout.minX}
          y1={layout.centerY - 20}
          x2={layout.minX}
          y2={layout.centerY + 20}
          stroke="hsl(215, 20%, 55%)"
          strokeWidth={1.5}
        />
        {/* Max whisker cap */}
        <line
          x1={layout.maxX}
          y1={layout.centerY - 20}
          x2={layout.maxX}
          y2={layout.centerY + 20}
          stroke="hsl(215, 20%, 55%)"
          strokeWidth={1.5}
        />
        {/* Box (Q1 to Q3) */}
        <rect
          x={layout.q1X}
          y={layout.centerY - layout.boxHeight / 2}
          width={layout.q3X - layout.q1X}
          height={layout.boxHeight}
          fill="hsl(199, 89%, 48%)"
          fillOpacity={0.3}
          stroke="hsl(199, 89%, 48%)"
          strokeWidth={2}
          rx={4}
        />
        {/* Median line */}
        <line
          x1={layout.medianX}
          y1={layout.centerY - layout.boxHeight / 2}
          x2={layout.medianX}
          y2={layout.centerY + layout.boxHeight / 2}
          stroke="hsl(174, 72%, 50%)"
          strokeWidth={3}
        />
        {/* Outlier dots */}
        {layout.outlierXs.map((x, i) => (
          <circle
            key={i}
            cx={x}
            cy={layout.centerY}
            r={4}
            fill="hsl(350, 80%, 60%)"
            fillOpacity={0.7}
          />
        ))}
        {/* Axis ticks */}
        {ticks.map((tick, i) => (
          <g key={i}>
            <line
              x1={tick.x}
              y1={layout.centerY + layout.boxHeight / 2 + 10}
              x2={tick.x}
              y2={layout.centerY + layout.boxHeight / 2 + 16}
              stroke="hsl(215, 20%, 55%)"
              strokeWidth={1}
            />
            <text
              x={tick.x}
              y={layout.centerY + layout.boxHeight / 2 + 30}
              textAnchor="middle"
              fill="hsl(215, 20%, 55%)"
              fontSize={11}
            >
              {tick.label}
            </text>
          </g>
        ))}
        {/* Labels */}
        <text x={layout.minX} y={layout.centerY - 30} textAnchor="middle" fill="hsl(215, 20%, 55%)" fontSize={10}>
          Min: {data.min.toFixed(2)}
        </text>
        <text x={layout.q1X} y={layout.centerY - layout.boxHeight / 2 - 8} textAnchor="middle" fill="hsl(199, 89%, 48%)" fontSize={10}>
          Q1: {data.q1.toFixed(2)}
        </text>
        <text x={layout.medianX} y={layout.centerY - layout.boxHeight / 2 - 22} textAnchor="middle" fill="hsl(174, 72%, 50%)" fontSize={10} fontWeight="bold">
          Median: {data.median.toFixed(2)}
        </text>
        <text x={layout.q3X} y={layout.centerY - layout.boxHeight / 2 - 8} textAnchor="middle" fill="hsl(199, 89%, 48%)" fontSize={10}>
          Q3: {data.q3.toFixed(2)}
        </text>
        <text x={layout.maxX} y={layout.centerY - 30} textAnchor="middle" fill="hsl(215, 20%, 55%)" fontSize={10}>
          Max: {data.max.toFixed(2)}
        </text>
      </svg>
    </div>
  );
}
