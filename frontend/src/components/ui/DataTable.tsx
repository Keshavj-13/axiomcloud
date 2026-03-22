import { ReactNode } from "react";
import clsx from "clsx";

interface TableColumn<T> {
  key: string;
  header: string;
  className?: string;
  align?: "left" | "center" | "right";
  render: (row: T) => ReactNode;
}

interface DataTableProps<T> {
  columns: TableColumn<T>[];
  data: T[];
  rowKey: (row: T) => string | number;
  emptyState?: ReactNode;
}

export default function DataTable<T>({ columns, data, rowKey, emptyState }: DataTableProps<T>) {
  if (data.length === 0) {
    return <>{emptyState}</>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[760px] text-sm">
        <thead>
          <tr className="border-b border-outline/20">
            {columns.map((col) => (
              <th
                key={col.key}
                className={clsx(
                  "pb-3 text-[10px] uppercase tracking-[0.14em] text-text-muted",
                  col.align === "right" && "text-right",
                  col.align === "center" && "text-center",
                  col.align !== "right" && col.align !== "center" && "text-left",
                  col.className
                )}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row) => (
            <tr key={rowKey(row)} className="border-b border-outline/10 transition-colors hover:bg-surface-variant/30">
              {columns.map((col) => (
                <td
                  key={col.key}
                  className={clsx(
                    "py-3 text-text-primary",
                    col.align === "right" && "text-right",
                    col.align === "center" && "text-center",
                    col.align !== "right" && col.align !== "center" && "text-left",
                    col.className
                  )}
                >
                  {col.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
