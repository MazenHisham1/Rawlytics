import { useState, useCallback } from 'react';
import { Upload, FileSpreadsheet, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { uploadFile } from '@/services/api';
import { parseCSV } from '@/utils/dataAnalyzer';

interface FileUploadProps {
  onFileLoaded: (data: Record<string, unknown>[], fileName: string) => void;
}

export function FileUpload({ onFileLoaded }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState<string>('');

  const handleFile = useCallback(async (file: File) => {
    setError(null);
    setIsLoading(true);
    setStatus('');

    try {
      if (!file.name.endsWith('.csv') && !file.name.endsWith('.json')) {
        throw new Error('Please upload a CSV or JSON file');
      }

      // Step 1: Upload to FastAPI backend
      setStatus('Uploading to server...');
      try {
        await uploadFile(file);
      } catch (err) {
        throw new Error(
          `Failed to upload to backend. Make sure your FastAPI server is running with CORS enabled. ${err instanceof Error ? err.message : ''}`
        );
      }

      // Step 2: Parse locally for chart data (raw rows needed for scatter plots)
      setStatus('Parsing local data...');
      const text = await file.text();
      let data: Record<string, unknown>[];

      if (file.name.endsWith('.json')) {
        const parsed = JSON.parse(text);
        data = Array.isArray(parsed) ? parsed : [parsed];
      } else {
        data = parseCSV(text);
      }

      if (data.length === 0) {
        throw new Error('No data found in file');
      }

      onFileLoaded(data, file.name);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process file');
    } finally {
      setIsLoading(false);
      setStatus('');
    }
  }, [onFileLoaded]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  return (
    <div className="w-full max-w-2xl mx-auto animate-fade-in">
      <label
        className={cn(
          'upload-zone min-h-[280px]',
          isDragging && 'upload-zone-active',
          isLoading && 'opacity-70 cursor-wait'
        )}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <input
          type="file"
          accept=".csv,.json"
          onChange={handleInputChange}
          className="hidden"
          disabled={isLoading}
        />
        
        <div className={cn(
          'w-20 h-20 rounded-2xl flex items-center justify-center transition-all duration-300',
          isDragging ? 'bg-primary/20 scale-110' : 'bg-secondary'
        )}>
          {isLoading ? (
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          ) : (
            <Upload className={cn(
              'w-10 h-10 transition-colors',
              isDragging ? 'text-primary' : 'text-muted-foreground'
            )} />
          )}
        </div>

        <div className="text-center space-y-2">
          <p className="text-lg font-medium text-foreground">
            {isLoading ? status || 'Processing...' : 'Drop your dataset here'}
          </p>
          <p className="text-sm text-muted-foreground">
            or click to browse • CSV, JSON supported
          </p>
        </div>

        <div className="flex items-center gap-3 px-4 py-2 bg-secondary/50 rounded-lg">
          <FileSpreadsheet className="w-5 h-5 text-primary" />
          <span className="text-sm text-muted-foreground">
            Uploads to FastAPI backend for analysis
          </span>
        </div>
      </label>

      {error && (
        <div className="mt-4 flex items-center gap-3 p-4 bg-destructive/10 border border-destructive/30 rounded-lg animate-scale-in">
          <AlertCircle className="w-5 h-5 text-destructive flex-shrink-0" />
          <p className="text-sm text-destructive">{error}</p>
        </div>
      )}
    </div>
  );
}
