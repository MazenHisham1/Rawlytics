import type { DatasetAnalysis } from '@/types/dataset';

interface ReportData {
  analysis: DatasetAnalysis;
  fileName: string;
  imputations: DatasetAnalysis['imputations'];
}

function generateColumnsHtml(columns: DatasetAnalysis['columns']): string {
  return columns.map(col => {
    const missingBadge = col.missingPercent > 50
      ? `<span class="badge badge-error">${col.missingPercent.toFixed(1)}%</span>`
      : col.missingPercent > 10
      ? `<span class="badge badge-warning">${col.missingPercent.toFixed(1)}%</span>`
      : `${col.missingPercent.toFixed(1)}%`;
    const typeBadge = col.type === 'numeric' ? 'numeric' : 'categorical';
    const kurtosis = col.kurtosis != null ? col.kurtosis.toFixed(3) : '—';
    const kurtosisType = col.kurtosisType ?? '—';
    const normality = col.normalityVerdict ?? '—';
    const normalityBadge = col.normalityVerdict === 'normal'
      ? `<span class="badge badge-success">${normality}</span>`
      : col.normalityVerdict === 'not_normal'
      ? `<span class="badge badge-warning">${normality}</span>`
      : normality;
    const entropy = col.entropy != null ? col.entropy.toFixed(3) : '—';
    const entropyInterp = col.entropyInterpretation ?? '';
    return `<tr>
      <td><code>${col.name}</code></td>
      <td><span class="badge badge-${typeBadge}">${col.type}</span></td>
      <td>${missingBadge}</td>
      <td>${col.uniqueCount.toLocaleString()}</td>
      <td>${kurtosis}${kurtosisType !== '—' ? ` <span class="badge badge-info">${kurtosisType}</span>` : ''}</td>
      <td>${normalityBadge}</td>
      <td>${entropy}${entropyInterp ? ` <span class="text-muted">(${entropyInterp})</span>` : ''}</td>
    </tr>`;
  }).join('');
}

function generateCorrelationHtml(correlation?: DatasetAnalysis['correlation']): string {
  if (!correlation) return '';

  const renderMatrix = (matrix: Record<string, Record<string, number | null>>, cols: string[], title: string) => {
    if (cols.length === 0) return '';
    const headerCells = cols.map(c => `<th class="rotate-header"><div><span>${c.length > 12 ? c.slice(0, 12) + '…' : c}</span></div></th>`).join('');
    const rows = cols.map(row => {
      const cells = cols.map(col => {
        const val = matrix[row]?.[col];
        if (val == null) return `<td class="corr-cell">—</td>`;
        const abs = Math.abs(val);
        const hue = val >= 0 ? 170 : 0;
        const alpha = Math.min(abs * 0.7, 0.6);
        return `<td class="corr-cell" style="background: hsla(${hue}, 70%, 50%, ${alpha});" title="${row} × ${col}: ${val.toFixed(3)}">${val.toFixed(2)}</td>`;
      }).join('');
      return `<tr><td class="row-label"><code>${row.length > 15 ? row.slice(0, 15) + '…' : row}</code></td>${cells}</tr>`;
    }).join('');
    return `<div class="corr-block">
      <h3>${title}</h3>
      <div class="corr-scroll">
        <table class="corr-table">
          <thead><tr><th></th>${headerCells}</tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </div>`;
  };

  const pearsonHtml = renderMatrix(correlation.pearson, correlation.numeric_columns, '📐 Pearson Correlation (Numeric)');
  const cramersHtml = renderMatrix(correlation.cramers_v, correlation.categorical_columns, '🔤 Cramér\'s V (Categorical)');

  if (!pearsonHtml && !cramersHtml) return '';

  return `
    <div class="section">
      <div class="section-header">
        <div class="section-icon icon-purple">🔗</div>
        <h2>Correlation Analysis <span>Relationships between features</span></h2>
      </div>
      ${pearsonHtml}
      ${cramersHtml}
    </div>`;
}

function generateDtypeSuggestionsHtml(suggestions?: DatasetAnalysis['dtypeSuggestions']): string {
  if (!suggestions || Object.keys(suggestions).length === 0) return '';
  const rows = Object.entries(suggestions).map(([col, info]) => `<tr>
    <td><code>${col}</code></td>
    <td><span class="badge badge-warning">${info.current_dtype}</span></td>
    <td><span class="badge badge-success">${info.suggested_dtype}</span></td>
    <td>${info.reason}</td>
  </tr>`).join('');

  return `
    <div class="section">
      <div class="section-header">
        <div class="section-icon icon-amber">🔧</div>
        <h2>Data Type Suggestions <span>${Object.keys(suggestions).length} columns can be optimized</span></h2>
      </div>
      <table>
        <thead><tr><th>Column</th><th>Current Type</th><th>Suggested Type</th><th>Reason</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

function generateEncodingSuggestionsHtml(suggestions?: DatasetAnalysis['encodingSuggestions']): string {
  if (!suggestions || Object.keys(suggestions).length === 0) return '';
  const rows = Object.entries(suggestions).map(([col, info]) => `<tr>
    <td><code>${col}</code></td>
    <td>${info.n_unique}</td>
    <td>${(info.cardinality_ratio * 100).toFixed(1)}%</td>
    <td><span class="badge badge-info">${info.recommended_encoding}</span></td>
    <td>${info.reason}</td>
  </tr>`).join('');

  return `
    <div class="section">
      <div class="section-header">
        <div class="section-icon icon-green">🧠</div>
        <h2>Encoding Suggestions <span>ML-ready feature engineering</span></h2>
      </div>
      <table>
        <thead><tr><th>Column</th><th>Unique</th><th>Cardinality</th><th>Encoding</th><th>Reason</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

export function generateReportHtml({ analysis, fileName, imputations }: ReportData): string {
  const { stats, columns, outliers, skewness, recommendations } = analysis;
  const dateStr = new Date().toLocaleDateString('en-US', { dateStyle: 'long' });
  const year = new Date().getFullYear();

  const columnsHtml = generateColumnsHtml(columns);
  const correlationHtml = generateCorrelationHtml(analysis.correlation);
  const dtypeHtml = generateDtypeSuggestionsHtml(analysis.dtypeSuggestions);
  const encodingHtml = generateEncodingSuggestionsHtml(analysis.encodingSuggestions);

  const outliersHtml = outliers.length > 0 ? `
    <div class="section">
      <div class="section-header">
        <div class="section-icon icon-amber">📈</div>
        <h2>Outlier Analysis <span>${outliers.length} columns with outliers</span></h2>
      </div>
      <table>
        <thead><tr><th>Column</th><th>Outlier Count</th><th>Percentage</th><th>Bounds (IQR)</th></tr></thead>
        <tbody>${outliers.map(o => `<tr>
          <td><code>${o.column}</code></td>
          <td>${o.count}</td>
          <td><span class="badge ${o.percentage > 5 ? 'badge-warning' : 'badge-info'}">${o.percentage.toFixed(1)}%</span></td>
          <td>${o.lowerBound.toFixed(2)} — ${o.upperBound.toFixed(2)}</td>
        </tr>`).join('')}</tbody>
      </table>
    </div>` : '';

  const skewnessHtml = skewness.length > 0 ? `
    <div class="section">
      <div class="section-header">
        <div class="section-icon icon-purple">📐</div>
        <h2>Skewness Analysis <span>Distribution shape insights</span></h2>
      </div>
      <table>
        <thead><tr><th>Column</th><th>Skewness</th><th>Direction</th><th>Interpretation</th></tr></thead>
        <tbody>${skewness.map(s => `<tr>
          <td><code>${s.column}</code></td>
          <td>${s.skewness.toFixed(3)}</td>
          <td>${s.direction === 'right' ? '→ Right' : s.direction === 'left' ? '← Left' : '↔ Symmetric'}</td>
          <td><span class="badge ${s.interpretation === 'highly_skewed' ? 'badge-warning' : 'badge-info'}">${s.interpretation.replace('_', ' ')}</span></td>
        </tr>`).join('')}</tbody>
      </table>
    </div>` : '';

  const imputationsHtml = imputations.length > 0 ? `
    <div class="section">
      <div class="section-header">
        <div class="section-icon icon-green">🧪</div>
        <h2>Imputation Recommendations <span>Smart fill strategies</span></h2>
      </div>
      ${imputations.map(imp => {
        const severity = imp.missing_rate >= 0.3 ? 'error' : imp.missing_rate >= 0.05 ? 'warning' : 'info';
        const corrEntries = Object.entries(imp.strong_correlations);
        const corrHtml = corrEntries.length > 0
          ? `<div class="imp-corr">Strong correlations: ${corrEntries.map(([col, val]) => `<code>${col}: ${(val as number).toFixed(2)}</code>`).join(', ')}</div>` : '';
        return `<div class="rec-card rec-${severity}">
          <div class="rec-title"><code>${imp.column}</code> → ${imp.recommended_method}</div>
          <div class="imp-stats">
            <span>Missing: <strong>${(imp.missing_rate * 100).toFixed(1)}%</strong></span>
            <span>Skew: <strong>${imp.skewness.toFixed(3)}</strong></span>
            <span>Outliers: <strong>${imp.has_outliers ? 'Yes' : 'No'}</strong></span>
            <span>N: <strong>${imp.dataset_size}</strong></span>
          </div>
          ${corrHtml}
        </div>`;
      }).join('')}
    </div>` : '';

  const recommendationsHtml = recommendations.map(rec => `
    <div class="rec-card rec-${rec.type}">
      <div class="rec-title">${rec.title}</div>
      <div class="rec-desc">${rec.description}</div>
      ${rec.action ? `<div class="rec-action">➤ ${rec.action}</div>` : ''}
      ${rec.affectedColumns?.length ? `<div class="rec-affected">Affected: ${rec.affectedColumns.join(', ')}</div>` : ''}
    </div>
  `).join('');

  // Count sections for TOC
  const sectionsList = [
    { id: 'overview', icon: '📊', label: 'Dataset Overview' },
    { id: 'columns', icon: '📋', label: `Column Details (${columns.length})` },
    ...(outliers.length > 0 ? [{ id: 'outliers', icon: '📈', label: 'Outlier Analysis' }] : []),
    ...(skewness.length > 0 ? [{ id: 'skewness', icon: '📐', label: 'Skewness Analysis' }] : []),
    ...(analysis.correlation ? [{ id: 'correlation', icon: '🔗', label: 'Correlation Analysis' }] : []),
    ...(analysis.dtypeSuggestions && Object.keys(analysis.dtypeSuggestions).length > 0 ? [{ id: 'dtype', icon: '🔧', label: 'Data Type Suggestions' }] : []),
    ...(analysis.encodingSuggestions && Object.keys(analysis.encodingSuggestions).length > 0 ? [{ id: 'encoding', icon: '🧠', label: 'Encoding Suggestions' }] : []),
    ...(imputations.length > 0 ? [{ id: 'imputation', icon: '🧪', label: 'Imputation Recommendations' }] : []),
    { id: 'recommendations', icon: '💡', label: 'Recommendations' },
  ];

  const tocHtml = `
    <div class="toc">
      <div class="toc-title">📑 Table of Contents</div>
      <div class="toc-grid">
        ${sectionsList.map((s, i) => `<a href="#${s.id}" class="toc-item"><span class="toc-num">${String(i + 1).padStart(2, '0')}</span><span class="toc-icon">${s.icon}</span>${s.label}</a>`).join('')}
      </div>
    </div>`;

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Rawlytics Report - ${fileName}</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(30px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes shimmer {
      0% { background-position: -200% center; }
      100% { background-position: 200% center; }
    }
    @keyframes float {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-6px); }
    }
    body {
      font-family: 'Inter', system-ui, sans-serif;
      background: #080d1a;
      color: #e2e8f0;
      min-height: 100vh;
      overflow-x: hidden;
    }
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background:
        radial-gradient(ellipse 80% 60% at 50% -20%, rgba(45, 212, 191, 0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(14, 165, 233, 0.06) 0%, transparent 50%),
        radial-gradient(ellipse 50% 40% at 10% 60%, rgba(168, 85, 247, 0.05) 0%, transparent 50%);
      pointer-events: none;
      z-index: 0;
    }
    .container { max-width: 1100px; margin: 0 auto; padding: 2.5rem 1.5rem; position: relative; z-index: 1; }
    .header {
      text-align: center;
      margin-bottom: 2rem;
      padding: 3rem 2rem 2.5rem;
      background: linear-gradient(145deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.7));
      border-radius: 24px;
      border: 1px solid rgba(45, 212, 191, 0.15);
      position: relative;
      overflow: hidden;
      animation: fadeInUp 0.6s ease-out;
    }
    .header::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg, transparent, #2dd4bf, #0ea5e9, #a855f7, transparent);
      animation: shimmer 3s linear infinite;
      background-size: 200% 100%;
    }
    .logo-icon {
      width: 52px; height: 52px;
      background: linear-gradient(135deg, #2dd4bf, #0ea5e9);
      border-radius: 16px;
      display: inline-flex; align-items: center; justify-content: center;
      font-size: 1.5rem;
      box-shadow: 0 8px 24px -4px rgba(45, 212, 191, 0.3);
      animation: float 4s ease-in-out infinite;
      margin-bottom: 1rem;
    }
    .header h1 {
      font-size: 2.75rem;
      font-weight: 900;
      letter-spacing: -0.03em;
      background: linear-gradient(135deg, #2dd4bf 0%, #0ea5e9 50%, #a855f7 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .header .tagline { color: #64748b; font-size: 0.9rem; font-style: italic; margin-top: 0.25rem; }
    .meta-info { display: flex; justify-content: center; gap: 1rem; margin-top: 1.25rem; flex-wrap: wrap; }
    .meta-chip {
      display: inline-flex; align-items: center; gap: 0.4rem;
      padding: 0.4rem 1rem;
      background: rgba(45, 212, 191, 0.08);
      border: 1px solid rgba(45, 212, 191, 0.15);
      border-radius: 100px;
      font-size: 0.8rem; color: #94a3b8;
    }
    .meta-chip strong { color: #e2e8f0; }

    /* Table of Contents */
    .toc {
      background: linear-gradient(145deg, rgba(15, 23, 42, 0.8), rgba(20, 30, 50, 0.6));
      border-radius: 20px;
      padding: 1.5rem 2rem;
      margin-bottom: 1.5rem;
      border: 1px solid rgba(148, 163, 184, 0.08);
      animation: fadeInUp 0.6s ease-out 0.05s both;
    }
    .toc-title { font-weight: 700; font-size: 1rem; margin-bottom: 1rem; color: #f1f5f9; }
    .toc-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; }
    .toc-item {
      display: inline-flex; align-items: center; gap: 0.4rem;
      padding: 0.4rem 0.9rem;
      background: rgba(45, 212, 191, 0.06);
      border: 1px solid rgba(148, 163, 184, 0.1);
      border-radius: 10px;
      font-size: 0.78rem; color: #94a3b8;
      text-decoration: none;
      transition: all 0.2s;
    }
    .toc-item:hover { background: rgba(45, 212, 191, 0.12); border-color: rgba(45, 212, 191, 0.3); color: #e2e8f0; }
    .toc-num { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #2dd4bf; font-weight: 600; }
    .toc-icon { font-size: 0.85rem; }

    .section {
      background: linear-gradient(145deg, rgba(15, 23, 42, 0.8), rgba(20, 30, 50, 0.6));
      border-radius: 20px;
      padding: 2rem;
      margin-bottom: 1.5rem;
      border: 1px solid rgba(148, 163, 184, 0.08);
      backdrop-filter: blur(10px);
      animation: fadeInUp 0.6s ease-out both;
      transition: border-color 0.3s, box-shadow 0.3s;
    }
    .section:hover {
      border-color: rgba(45, 212, 191, 0.2);
      box-shadow: 0 8px 40px -12px rgba(45, 212, 191, 0.1);
    }
    .section-header {
      display: flex; align-items: center; gap: 0.75rem;
      margin-bottom: 1.5rem; padding-bottom: 1rem;
      border-bottom: 1px solid rgba(148, 163, 184, 0.1);
    }
    .section-icon {
      width: 40px; height: 40px;
      border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      font-size: 1.2rem; flex-shrink: 0;
    }
    .icon-teal { background: rgba(45, 212, 191, 0.15); }
    .icon-blue { background: rgba(14, 165, 233, 0.15); }
    .icon-purple { background: rgba(168, 85, 247, 0.15); }
    .icon-amber { background: rgba(245, 158, 11, 0.15); }
    .icon-green { background: rgba(34, 197, 94, 0.15); }
    .section h2 { font-size: 1.2rem; font-weight: 700; color: #f1f5f9; letter-spacing: -0.01em; }
    .section h2 span { color: #64748b; font-weight: 400; font-size: 0.85rem; margin-left: 0.5rem; }
    .section h3 { font-size: 1rem; font-weight: 600; color: #cbd5e1; margin-bottom: 1rem; margin-top: 0.5rem; }

    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; }
    .stat-card {
      background: linear-gradient(145deg, rgba(15, 23, 42, 0.9), rgba(20, 30, 50, 0.7));
      padding: 1.5rem; border-radius: 16px; text-align: center;
      border: 1px solid rgba(148, 163, 184, 0.06);
      transition: all 0.3s ease; position: relative; overflow: hidden;
    }
    .stat-card::before {
      content: ''; position: absolute; top: 0; left: 0; right: 0;
      height: 2px; background: linear-gradient(90deg, #2dd4bf, #0ea5e9);
      opacity: 0; transition: opacity 0.3s;
    }
    .stat-card:hover::before { opacity: 1; }
    .stat-card:hover { transform: translateY(-2px); border-color: rgba(45, 212, 191, 0.2); }
    .stat-value {
      font-size: 2rem; font-weight: 800; letter-spacing: -0.02em;
      background: linear-gradient(135deg, #f1f5f9, #cbd5e1);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .stat-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.5rem; font-weight: 600; }

    table { width: 100%; border-collapse: separate; border-spacing: 0; }
    thead tr { background: rgba(15, 23, 42, 0.5); }
    th {
      padding: 0.85rem 1rem; text-align: left; color: #64748b;
      font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
      font-weight: 600; border-bottom: 1px solid rgba(148, 163, 184, 0.1);
    }
    th:first-child { border-radius: 10px 0 0 0; }
    th:last-child { border-radius: 0 10px 0 0; }
    td { padding: 0.85rem 1rem; border-bottom: 1px solid rgba(148, 163, 184, 0.05); font-size: 0.9rem; }
    tr { transition: background 0.2s; }
    tbody tr:hover { background: rgba(45, 212, 191, 0.03); }
    .text-muted { color: #64748b; font-size: 0.75rem; }

    code {
      font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
      background: rgba(45, 212, 191, 0.08); padding: 0.2rem 0.5rem;
      border-radius: 6px; color: #2dd4bf; border: 1px solid rgba(45, 212, 191, 0.1);
    }
    .badge {
      display: inline-flex; align-items: center; padding: 0.3rem 0.7rem;
      border-radius: 100px; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.02em;
    }
    .badge-numeric { background: rgba(14, 165, 233, 0.12); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.2); }
    .badge-categorical { background: rgba(168, 85, 247, 0.12); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.2); }
    .badge-warning { background: rgba(245, 158, 11, 0.12); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.2); }
    .badge-error { background: rgba(239, 68, 68, 0.12); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.2); }
    .badge-success { background: rgba(34, 197, 94, 0.12); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.2); }
    .badge-info { background: rgba(14, 165, 233, 0.12); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.2); }

    /* Correlation matrix */
    .corr-block { margin-bottom: 1.5rem; }
    .corr-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; padding-bottom: 0.5rem; }
    .corr-table { border-collapse: collapse; font-size: 0.75rem; }
    .corr-table th, .corr-table td { padding: 0.5rem 0.6rem; text-align: center; white-space: nowrap; border: 1px solid rgba(148, 163, 184, 0.06); }
    .corr-table th { font-size: 0.65rem; color: #64748b; }
    .corr-cell { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; min-width: 50px; transition: transform 0.15s; }
    .corr-cell:hover { transform: scale(1.1); z-index: 1; }
    .row-label { text-align: right !important; padding-right: 0.75rem !important; }
    .rotate-header div { writing-mode: vertical-rl; transform: rotate(180deg); height: 80px; display: flex; align-items: center; }

    .rec-card {
      padding: 1.25rem 1.5rem; border-radius: 14px; margin-bottom: 0.75rem;
      border-left: 4px solid; background: rgba(15, 23, 42, 0.5);
      transition: all 0.2s ease;
    }
    .rec-card:hover { transform: translateX(4px); background: rgba(15, 23, 42, 0.7); }
    .rec-warning { background: rgba(245, 158, 11, 0.05); border-color: #f59e0b; }
    .rec-error { background: rgba(239, 68, 68, 0.05); border-color: #ef4444; }
    .rec-success { background: rgba(34, 197, 94, 0.05); border-color: #22c55e; }
    .rec-info { background: rgba(14, 165, 233, 0.05); border-color: #0ea5e9; }
    .rec-title { font-weight: 700; margin-bottom: 0.35rem; font-size: 0.95rem; }
    .rec-desc { font-size: 0.85rem; color: #94a3b8; line-height: 1.6; }
    .rec-action { margin-top: 0.5rem; color: #2dd4bf; font-size: 0.85rem; font-weight: 500; }
    .rec-affected { margin-top: 0.5rem; font-size: 0.75rem; color: #475569; }
    .imp-stats { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.5rem; font-size: 0.85rem; color: #94a3b8; }
    .imp-corr { margin-top: 0.5rem; font-size: 0.75rem; color: #94a3b8; }

    .footer { text-align: center; padding: 3rem 2rem 2rem; }
    .footer::before {
      content: ''; display: block; width: 120px; height: 2px;
      background: linear-gradient(90deg, transparent, rgba(45, 212, 191, 0.3), transparent);
      margin: 0 auto 1.5rem;
    }
    .footer-brand {
      font-size: 1.1rem; font-weight: 700;
      background: linear-gradient(135deg, #2dd4bf, #0ea5e9);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .footer p { color: #475569; font-size: 0.8rem; margin-top: 0.5rem; }
    .footer .powered { margin-top: 1rem; font-size: 0.7rem; color: #334155; letter-spacing: 0.05em; }

    /* Responsive */
    @media (max-width: 768px) {
      .container { padding: 1rem; }
      .header { padding: 2rem 1.25rem 1.75rem; border-radius: 16px; }
      .header h1 { font-size: 1.75rem; }
      .meta-info { flex-direction: column; align-items: center; gap: 0.5rem; }
      .meta-chip { font-size: 0.7rem; }
      .section { padding: 1.25rem; border-radius: 14px; }
      .section h2 { font-size: 1rem; }
      .section h2 span { display: block; margin-left: 0; margin-top: 0.25rem; }
      .stats-grid { grid-template-columns: repeat(2, 1fr); gap: 0.75rem; }
      .stat-value { font-size: 1.5rem; }
      .stat-card { padding: 1rem; border-radius: 12px; }
      table { display: block; overflow-x: auto; -webkit-overflow-scrolling: touch; }
      th, td { padding: 0.6rem 0.75rem; font-size: 0.8rem; white-space: nowrap; }
      .rec-card { padding: 1rem 1.25rem; }
      .imp-stats { flex-direction: column; gap: 0.25rem; }
      .toc-grid { flex-direction: column; }
    }
    @media (max-width: 480px) {
      .header h1 { font-size: 1.4rem; }
      .stats-grid { grid-template-columns: 1fr 1fr; }
      .stat-value { font-size: 1.25rem; }
      .stat-label { font-size: 0.6rem; }
      .section-header { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
      .logo-icon { width: 40px; height: 40px; font-size: 1.2rem; border-radius: 12px; }
    }
    @media (min-width: 769px) and (max-width: 1024px) {
      .container { padding: 1.5rem; }
      .stats-grid { grid-template-columns: repeat(3, 1fr); }
    }
    @media print {
      body { background: #fff; color: #1e293b; }
      body::before { display: none; }
      .section { border: 1px solid #e2e8f0; background: #fff; box-shadow: none; }
      .header { background: #f8fafc; }
      .stat-card { background: #f8fafc; border: 1px solid #e2e8f0; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="logo-icon">📊</div>
      <h1>Rawlytics Report</h1>
      <p class="tagline">From raw data to real insights</p>
      <div class="meta-info">
        <div class="meta-chip">📁 <strong>${fileName}</strong></div>
        <div class="meta-chip">📅 ${dateStr}</div>
        <div class="meta-chip">🔢 <strong>${stats.rowCount.toLocaleString()}</strong> rows × <strong>${stats.columnCount}</strong> cols</div>
        <div class="meta-chip">📊 <strong>${sectionsList.length}</strong> analysis sections</div>
      </div>
    </div>

    ${tocHtml}

    <div class="section" id="overview">
      <div class="section-header">
        <div class="section-icon icon-teal">📊</div>
        <h2>Dataset Overview <span>Key metrics at a glance</span></h2>
      </div>
      <div class="stats-grid">
        <div class="stat-card"><div class="stat-value">${stats.rowCount.toLocaleString()}</div><div class="stat-label">Total Rows</div></div>
        <div class="stat-card"><div class="stat-value">${stats.columnCount}</div><div class="stat-label">Columns</div></div>
        <div class="stat-card"><div class="stat-value">${stats.duplicateRows.toLocaleString()}</div><div class="stat-label">Duplicates</div></div>
        <div class="stat-card"><div class="stat-value">${stats.missingPercent.toFixed(1)}%</div><div class="stat-label">Missing Values</div></div>
        <div class="stat-card"><div class="stat-value">${stats.memoryUsage}</div><div class="stat-label">Memory Usage</div></div>
      </div>
    </div>

    <div class="section" id="columns">
      <div class="section-header">
        <div class="section-icon icon-blue">📋</div>
        <h2>Column Details <span>${columns.length} columns analyzed</span></h2>
      </div>
      <table>
        <thead><tr><th>Column</th><th>Type</th><th>Missing %</th><th>Unique</th><th>Kurtosis</th><th>Normality</th><th>Entropy</th></tr></thead>
        <tbody>${columnsHtml}</tbody>
      </table>
    </div>

    ${outliersHtml}
    ${skewnessHtml}
    ${correlationHtml}
    ${dtypeHtml}
    ${encodingHtml}
    ${imputationsHtml}

    <div class="section" id="recommendations">
      <div class="section-header">
        <div class="section-icon icon-teal">💡</div>
        <h2>Recommendations <span>Actionable next steps</span></h2>
      </div>
      ${recommendationsHtml}
    </div>

    <div class="footer">
      <div class="footer-brand">Rawlytics</div>
      <p>From raw data to real insights</p>
      <div class="powered">Generated automatically • ${year}</div>
    </div>
  </div>
</body>
</html>`;
}
