import { useState, useCallback } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { Table, BarChart3, Lightbulb, FileSpreadsheet, GitCompare, AlertTriangle, Cpu, Wrench } from 'lucide-react';
import { Header } from '@/components/Header';
import { FileUpload } from '@/components/FileUpload';
import { StatsOverview } from '@/components/StatsOverview';
import { ColumnTable } from '@/components/ColumnTable';
import { ChartPanel } from '@/components/ChartPanel';
import { Recommendations } from '@/components/Recommendations';
import { ReportGenerator } from '@/components/ReportGenerator';
import { CorrelationMatrix } from '@/components/CorrelationMatrix';
import { OutlierPanel } from '@/components/OutlierPanel';
import { MLReadiness } from '@/components/MLReadiness';
import { DataQuality } from '@/components/DataQuality';
import { analyzeFromApi } from '@/utils/apiAnalyzer';
import { deleteSession, getSessionId } from '@/services/api';
import type { DatasetAnalysis } from '@/types/dataset';
import { toast } from '@/hooks/use-toast';

const Index = () => {
  const [data, setData] = useState<Record<string, unknown>[] | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [analysis, setAnalysis] = useState<DatasetAnalysis | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleFileLoaded = useCallback(async (loadedData: Record<string, unknown>[], name: string) => {
    setData(loadedData);
    setFileName(name);
    setIsAnalyzing(true);

    try {
      const result = await analyzeFromApi();
      setAnalysis(result);
    } catch (err) {
      toast({
        title: 'Analysis Failed',
        description: err instanceof Error ? err.message : 'Failed to analyze dataset from backend',
        variant: 'destructive'
      });
      setData(null);
      setFileName('');
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const handleReset = useCallback(async () => {
    const sessionId = getSessionId();
    if (sessionId) {
      deleteSession(sessionId).catch(() => {});
    }
    setData(null);
    setFileName('');
    setAnalysis(null);
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        {!data && !isAnalyzing ? (
          <div className="max-w-4xl mx-auto space-y-12">
            <div className="text-center space-y-4 animate-fade-in">
              <h2 className="text-4xl md:text-5xl font-bold">
                <span className="gradient-text">Analyze</span> Your Data
              </h2>
              <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                Upload your dataset to discover insights, detect issues, and get smart 
                recommendations for data cleaning and preprocessing.
              </p>
            </div>
            
            <FileUpload onFileLoaded={handleFileLoaded} />
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 animate-fade-in" style={{ animationDelay: '200ms' }}>
              {[
                { icon: Table, label: 'Column Analysis', desc: 'Types & missing values' },
                { icon: BarChart3, label: 'Visualizations', desc: 'Charts & distributions' },
                { icon: Lightbulb, label: 'Smart Tips', desc: 'Cleaning recommendations' },
                { icon: FileSpreadsheet, label: 'Reports', desc: 'Export analysis' },
              ].map((feature, i) => (
                <div key={feature.label} className="glass-card p-4 text-center space-y-2" style={{ animationDelay: `${(i + 3) * 100}ms` }}>
                  <feature.icon className="w-8 h-8 mx-auto text-primary" />
                  <h3 className="font-medium text-foreground">{feature.label}</h3>
                  <p className="text-xs text-muted-foreground">{feature.desc}</p>
                </div>
              ))}
            </div>
          </div>
        ) : isAnalyzing ? (
          <div className="flex flex-col items-center justify-center py-24 space-y-6 animate-fade-in">
            <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin" />
            <div className="text-center space-y-2">
              <h3 className="text-xl font-semibold text-foreground">Analyzing Dataset...</h3>
              <p className="text-muted-foreground">Fetching analysis from backend</p>
            </div>
          </div>
        ) : analysis ? (
          <div className="space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center">
                  <FileSpreadsheet className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <h2 className="font-semibold text-foreground">{fileName}</h2>
                  <p className="text-sm text-muted-foreground">
                    {analysis.stats.rowCount.toLocaleString()} rows × {analysis.stats.columnCount} columns
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <ReportGenerator analysis={analysis} fileName={fileName} imputations={analysis.imputations} />
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      onClick={handleReset}
                      className="text-sm text-muted-foreground hover:text-foreground transition-colors"
                    >
                      Upload new file
                    </button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Start over with a new file</p>
                  </TooltipContent>
                </Tooltip>
              </div>
            </div>

            <StatsOverview stats={analysis.stats} />

            <Tabs defaultValue="columns" className="space-y-4">
            <TabsList className="bg-secondary/50 p-1 flex-wrap h-auto gap-1">
              <Tooltip>
                <TooltipTrigger asChild>
                  <TabsTrigger value="columns" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                    <Table className="w-4 h-4" />
                    <span className="hidden sm:inline">Columns</span>
                  </TabsTrigger>
                </TooltipTrigger>
                <TooltipContent>
                  <p>View column analysis</p>
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <TabsTrigger value="visualize" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                    <BarChart3 className="w-4 h-4" />
                    <span className="hidden sm:inline">Visualize</span>
                  </TabsTrigger>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Visualize your data</p>
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <TabsTrigger value="correlation" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                    <GitCompare className="w-4 h-4" />
                    <span className="hidden sm:inline">Correlation</span>
                  </TabsTrigger>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Explore correlations</p>
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <TabsTrigger value="outliers" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                    <AlertTriangle className="w-4 h-4" />
                    <span className="hidden sm:inline">Outliers</span>
                  </TabsTrigger>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Detect outliers</p>
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <TabsTrigger value="quality" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                    <Wrench className="w-4 h-4" />
                    <span className="hidden sm:inline">Data Quality</span>
                  </TabsTrigger>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Review data quality</p>
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <TabsTrigger value="ml" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                    <Cpu className="w-4 h-4" />
                    <span className="hidden sm:inline">ML Readiness</span>
                  </TabsTrigger>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Check ML readiness</p>
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <TabsTrigger value="recommendations" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                    <Lightbulb className="w-4 h-4" />
                    <span className="hidden sm:inline">Recommendations</span>
                    {analysis.imputations.length > 0 && (
                      <span className="ml-1 px-1.5 py-0.5 text-xs bg-warning/20 text-warning rounded-full">
                        {analysis.imputations.length}
                      </span>
                    )}
                  </TabsTrigger>
                </TooltipTrigger>
                <TooltipContent>
                  <p>View cleaning recommendations</p>
                </TooltipContent>
              </Tooltip>
            </TabsList>

              <TabsContent value="columns" className="mt-4">
                <ColumnTable columns={analysis.columns} />
              </TabsContent>

              <TabsContent value="visualize" className="mt-4">
                <ChartPanel data={[]} columns={analysis.columns} />
              </TabsContent>

              <TabsContent value="correlation" className="mt-4">
                {analysis.correlation ? (
                  <CorrelationMatrix data={analysis.correlation} />
                ) : (
                  <div className="glass-card p-6 text-center text-muted-foreground">
                    Correlation data not available.
                  </div>
                )}
              </TabsContent>

              <TabsContent value="outliers" className="mt-4">
                <OutlierPanel columns={analysis.columns} />
              </TabsContent>

              <TabsContent value="quality" className="mt-4">
                <DataQuality dtypeSuggestions={analysis.dtypeSuggestions} />
              </TabsContent>

              <TabsContent value="ml" className="mt-4">
                <MLReadiness columns={analysis.columns} encodingSuggestions={analysis.encodingSuggestions} />
              </TabsContent>

              <TabsContent value="recommendations" className="mt-4">
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h3 className="font-medium text-foreground flex items-center gap-2">
                      <Lightbulb className="w-4 h-4 text-primary" />
                      Imputation Recommendations
                    </h3>
                    <Recommendations imputations={analysis.imputations} />
                  </div>
                  
                  <div className="space-y-4">
                    {analysis.outliers.length > 0 && (
                      <div className="glass-card p-4 space-y-3">
                        <h4 className="font-medium text-foreground">Outlier Summary</h4>
                        <div className="space-y-2">
                          {analysis.outliers.slice(0, 5).map((o) => (
                            <div key={o.column} className="flex items-center justify-between text-sm">
                              <span className="font-mono text-muted-foreground">{o.column}</span>
                              <span className="text-warning">{o.count} outliers ({o.percentage.toFixed(1)}%)</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {analysis.skewness.length > 0 && (
                      <div className="glass-card p-4 space-y-3">
                        <h4 className="font-medium text-foreground">Skewness Summary</h4>
                        <div className="space-y-2">
                          {analysis.skewness.slice(0, 5).map((s) => (
                            <div key={s.column} className="flex items-center justify-between text-sm">
                              <span className="font-mono text-muted-foreground">{s.column}</span>
                              <span className={s.interpretation === 'highly_skewed' ? 'text-warning' : 'text-info'}>
                                {s.skewness.toFixed(2)} ({s.direction})
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        ) : null}
      </main>

      <footer className="py-6 border-t border-border/50 mt-12">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          Rawlytics turns messy data into clarity — no code required.
        </div>
      </footer>
    </div>
  );
};

export default Index;
