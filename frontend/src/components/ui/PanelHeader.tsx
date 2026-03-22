import { ReactNode } from "react";
import { LucideIcon } from "lucide-react";

interface PanelHeaderProps {
  title: string;
  subtitle?: string;
  action?: ReactNode;
  icon?: LucideIcon;
  className?: string;
}

export default function PanelHeader({ title, subtitle, action, icon: Icon, className = "" }: PanelHeaderProps) {
  return (
    <div className={`flex items-start justify-between gap-4 ${className}`}>
      <div>
        <div className="flex items-center gap-2">
          {Icon && <Icon className="h-4 w-4 text-primary" />}
          <h2 className="text-sm uppercase tracking-[0.14em] text-text-muted">{title}</h2>
        </div>
        {subtitle && <p className="mt-2 text-sm text-text-muted">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}
