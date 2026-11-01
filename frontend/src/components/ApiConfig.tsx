import { useState, useEffect } from 'react';
import { Settings, Wifi, WifiOff, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { getApiUrl, setApiUrl, clearApiUrl, checkConnection } from '@/services/api';
import { cn } from '@/lib/utils';

interface ApiConfigProps {
  onConnectionChange?: (connected: boolean) => void;
}

export function ApiConfig({ onConnectionChange }: ApiConfigProps) {
  const [url, setUrl] = useState(getApiUrl());
  const [isOpen, setIsOpen] = useState(false);
  const [status, setStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');

  const testConnection = async () => {
    setStatus('checking');
    const ok = await checkConnection();
    setStatus(ok ? 'connected' : 'disconnected');
    onConnectionChange?.(ok);
  };

  useEffect(() => {
    testConnection();
    const interval = setInterval(testConnection, 30000);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSave = () => {
    setApiUrl(url);
    testConnection();
  };

  return (
    <div className="relative">
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsOpen(!isOpen)}
            className="gap-2 text-muted-foreground hover:text-foreground"
          >
            {status === 'checking' ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : status === 'connected' ? (
              <Wifi className="w-4 h-4 text-success" />
            ) : (
              <WifiOff className="w-4 h-4 text-destructive" />
            )}
            <Settings className="w-4 h-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>Configure API connection</p>
        </TooltipContent>
      </Tooltip>

      {isOpen && (
        <div className="absolute right-0 top-full mt-2 w-80 glass-card p-4 space-y-3 z-50 animate-scale-in">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium text-foreground">API Configuration</h4>
            <span className={cn(
              'text-xs px-2 py-0.5 rounded-full',
              status === 'connected' ? 'bg-success/20 text-success' :
              status === 'disconnected' ? 'bg-destructive/20 text-destructive' :
              'bg-muted text-muted-foreground'
            )}>
              {status === 'connected' ? 'Connected' : status === 'disconnected' ? 'Disconnected' : 'Checking...'}
            </span>
          </div>

          <div className="space-y-2">
            <label className="text-xs text-muted-foreground">FastAPI Backend URL</label>
            <Input
              value={url}
              onChange={e => setUrl(e.target.value)}
              placeholder="http://localhost:8000"
              className="bg-secondary border-border text-sm"
            />
          </div>

          <div className="flex gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button size="sm" onClick={handleSave} className="flex-1">
                  Save & Test
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Save URL and test connection</p>
              </TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button size="sm" variant="outline" onClick={testConnection}>
                  Test
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Test API connection</p>
              </TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    setUrl('');
                    clearApiUrl();
                    setStatus('checking');
                    onConnectionChange?.(false);
                  }}
                  className="px-3"
                >
                  Clear
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Clear the saved URL and start fresh</p>
              </TooltipContent>
            </Tooltip>
          </div>

          <p className="text-xs text-muted-foreground">
            Make sure your FastAPI server has CORS enabled and is running.
          </p>
        </div>
      )}
    </div>
  );
}
