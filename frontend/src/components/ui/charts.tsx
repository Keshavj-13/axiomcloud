"use client";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  LineChart,
  Line,
} from "recharts";

export function HorizontalBarChart({
  data,
  xKey,
  yKey,
  color = "#3b6ef6",
  height = 280,
}: {
  data: Record<string, number | string>[];
  xKey: string;
  yKey: string;
  color?: string;
  height?: number;
}) {
  if (!data?.length) return <div className="py-8 text-xs text-text-muted">No data available.</div>;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ left: 24, right: 12, top: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,110,246,0.1)" />
        <XAxis type="number" tick={{ fill: "#6b7a9e", fontSize: 11 }} />
        <YAxis type="category" dataKey={yKey} width={170} tick={{ fill: "#8b97b8", fontSize: 11 }} />
        <Tooltip contentStyle={{ background: "#0d1526", border: "1px solid rgba(59,110,246,0.3)", borderRadius: "8px", color: "#e8edf8" }} />
        <Bar dataKey={xKey} fill={color} radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function RocLineChart({
  data,
  height = 280,
}: {
  data: { fpr: number; tpr: number }[];
  height?: number;
}) {
  if (!data?.length) return <div className="py-8 text-xs text-text-muted">No ROC data available.</div>;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,110,246,0.1)" />
        <XAxis dataKey="fpr" type="number" domain={[0, 1]} tick={{ fill: "#6b7a9e", fontSize: 11 }} />
        <YAxis domain={[0, 1]} tick={{ fill: "#6b7a9e", fontSize: 11 }} />
        <Tooltip contentStyle={{ background: "#0d1526", border: "1px solid rgba(59,110,246,0.3)", borderRadius: "8px", color: "#e8edf8" }} />
        <Line type="monotone" dataKey="tpr" stroke="#00f5ff" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function MatrixHeatmap({
  labels,
  matrix,
}: {
  labels: string[];
  matrix: number[][];
}) {
  if (!matrix?.length) return <div className="py-8 text-xs text-text-muted">No matrix data available.</div>;

  const max = Math.max(...matrix.flat(), 0);

  return (
    <div className="overflow-x-auto">
      <table className="mx-auto text-center text-xs font-mono">
        <thead>
          <tr>
            <th className="p-2 text-text-muted">Actual\Predicted</th>
            {labels.map((label) => (
              <th key={label} className="p-2 text-text-muted">{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="p-2 text-text-muted">{labels[i] ?? `Class ${i}`}</td>
              {row.map((val, j) => {
                const intensity = max > 0 ? val / max : 0;
                return (
                  <td
                    key={`${i}-${j}`}
                    className="rounded p-2 font-semibold text-white"
                    style={{
                      background:
                        i === j
                          ? `rgba(0,255,148,${0.15 + intensity * 0.45})`
                          : `rgba(239,68,68,${0.08 + intensity * 0.35})`,
                    }}
                  >
                    {val}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
