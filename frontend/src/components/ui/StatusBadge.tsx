import { ReactNode } from "react";
import clsx from "clsx";

type StatusTone = "healthy" | "processing" | "idle" | "error" | "primary";

interface StatusBadgeProps {
  label: string;
  tone?: StatusTone;
  icon?: ReactNode;
  className?: string;
}

const toneClasses: Record<StatusTone, string> = {
  healthy: "bg-emerald-500/10 text-emerald-300 border-emerald-500/20",
  processing: "bg-amber-500/10 text-amber-300 border-amber-500/20",
  idle: "bg-outline/10 text-text-muted border-outline/20",
  error: "bg-red-500/10 text-red-300 border-red-500/20",
  primary: "bg-primary/15 text-primary border-primary/30",
};

export default function StatusBadge({ label, tone = "idle", icon, className }: StatusBadgeProps) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-full border px-2 py-1 text-[10px] uppercase tracking-[0.14em]",
        toneClasses[tone],
        className
      )}
    >
      {icon}
      {label}
    </span>
  );
}
