import { useState } from 'react';
import { FileText, Download, Loader2, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import type { DatasetAnalysis } from '@/types/dataset';
import { generateReportHtml } from '@/utils/reportTemplate';

interface ReportGeneratorProps {
  analysis: DatasetAnalysis;
  fileName: string;
  imputations: DatasetAnalysis['imputations'];
}

export function ReportGenerator({ analysis, fileName, imputations }: ReportGeneratorProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [isComplete, setIsComplete] = useState(false);

  const generateReport = () => {
    setIsGenerating(true);

    const reportContent = generateReportHtml({ analysis, fileName, imputations });

    setTimeout(() => {
      const blob = new Blob([reportContent], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `rawlytics-report-${fileName.replace(/\.[^.]+$/, '')}.html`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setIsGenerating(false);
      setIsComplete(true);
      setTimeout(() => setIsComplete(false), 3000);
    }, 1500);
  };

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          onClick={generateReport}
          disabled={isGenerating}
          className="gap-2 bg-gradient-to-r from-primary to-info text-primary-foreground hover:opacity-90 transition-opacity"
          size="lg"
        >
          {isGenerating ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Generating Report...
            </>
          ) : isComplete ? (
            <>
              <CheckCircle className="w-5 h-5" />
              Report Downloaded!
            </>
          ) : (
            <>
              <FileText className="w-5 h-5" />
              Generate Report
              <Download className="w-4 h-4" />
            </>
          )}
        </Button>
      </TooltipTrigger>
      <TooltipContent>
        <p>Download analysis as HTML report</p>
      </TooltipContent>
    </Tooltip>
  );
}
