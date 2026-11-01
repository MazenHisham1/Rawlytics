import { useEffect, useState } from 'react';
import { Sparkles } from 'lucide-react';
import { ApiConfig } from '@/components/ApiConfig';
import { cn } from '@/lib/utils';

interface HeaderProps {
  onConnectionChange?: (connected: boolean) => void;
}

export function Header({ onConnectionChange }: HeaderProps) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <header
      className={cn(
        'sticky top-0 z-50 px-4 border-b transition-all duration-300 supports-[backdrop-filter]:bg-background/80 bg-background backdrop-blur-xl',
        scrolled
          ? 'py-3 border-border shadow-lg shadow-background/40'
          : 'py-6 border-border/50'
      )}
    >
      <div className="container mx-auto flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div
            className={cn(
              'rounded-xl bg-gradient-to-br from-primary to-info flex items-center justify-center shadow-glow transition-all duration-300',
              scrolled ? 'w-8 h-8' : 'w-10 h-10'
            )}
          >
            <Sparkles className={cn('text-primary-foreground transition-all duration-300', scrolled ? 'w-4 h-4' : 'w-5 h-5')} />
          </div>
          <div>
            <h1 className={cn('font-bold gradient-text leading-tight transition-all duration-300', scrolled ? 'text-lg' : 'text-xl')}>
              Rawlytics
            </h1>
            <p
              className={cn(
                'text-xs text-muted-foreground overflow-hidden transition-all duration-300',
                scrolled ? 'max-h-0 opacity-0' : 'max-h-5 opacity-100'
              )}
            >
              From raw data to real insights.
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-2 text-xs text-muted-foreground">
            <span className="px-2 py-1 bg-secondary/50 rounded border border-border/50">CSV</span>
            <span className="px-2 py-1 bg-secondary/50 rounded border border-border/50">JSON</span>
            <span className="text-primary font-medium">Supported</span>
          </div>
          <ApiConfig onConnectionChange={onConnectionChange} />
        </div>
      </div>
    </header>
  );
}
