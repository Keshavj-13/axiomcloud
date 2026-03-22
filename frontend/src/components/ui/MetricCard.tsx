import { LucideIcon } from "lucide-react";

interface MetricCardProps {
  label: string;
  value: string | number;
  icon: LucideIcon;
  helper?: string;
}

export default function MetricCard({ label, value, icon: Icon, helper }: MetricCardProps) {
  return (
    <div className="panel rounded-lg p-5">
      <div className="mb-4 inline-flex h-9 w-9 items-center justify-center rounded-lg border border-outline/20 bg-surface-variant/70">
        <Icon className="h-4 w-4 text-primary" />
      </div>
      <div className="font-mono text-2xl text-text-primary">{value}</div>
      <div className="mt-1 text-[11px] uppercase tracking-[0.12em] text-text-muted">{label}</div>
      {helper && <div className="mt-2 text-xs text-text-muted">{helper}</div>}
    </div>
  );
}
