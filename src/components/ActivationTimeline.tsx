"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";
import { TokenActivation } from "@/lib/types";

const COLORS = [
  "#3b82f6", "#8b5cf6", "#06b6d4", "#10b981",
  "#f59e0b", "#ef4444", "#ec4899", "#f97316",
];

interface ActivationTimelineProps {
  tokens: TokenActivation[];
  featureIds: number[];
  featureLabels: Record<number, string>;
}

/** Line chart showing feature activation values over token positions. */
export default function ActivationTimeline({
  tokens,
  featureIds,
  featureLabels,
}: ActivationTimelineProps) {
  if (tokens.length === 0 || featureIds.length === 0) return null;

  const featureKeys = featureIds.map(String);

  const data = tokens.map((tok, i) => {
    const point: Record<string, number | string> = {
      position: i,
      token: tok.token.replace(/\n/g, "\u21b5").replace(/ /g, "\u00b7"),
    };
    for (const key of featureKeys) {
      point[`f_${key}`] = tok.activations[key] ?? 0;
    }
    return point;
  });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null;
    const tokenText = data[label as number]?.token ?? "";
    return (
      <div className="rounded-lg border border-zinc-200 bg-white px-3 py-2 shadow-lg text-[11px]">
        <div className="font-mono text-zinc-500 mb-1">
          pos {label}: <span className="text-zinc-800">{tokenText}</span>
        </div>
        {payload.map((entry: { color: string; name: string; value: number }) => {
          const id = entry.name.replace("f_", "");
          return (
            <div key={entry.name} className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full flex-shrink-0" style={{ backgroundColor: entry.color }} />
              <span className="text-zinc-500 truncate max-w-[120px]">{featureLabels[Number(id)] ?? `#${id}`}</span>
              <span className="font-mono ml-auto" style={{ color: entry.color }}>
                {entry.value > 0 ? "+" : ""}{(entry.value as number).toFixed(4)}
              </span>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="w-full" style={{ height: 280 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
          <XAxis
            dataKey="position"
            tick={{ fill: "#a1a1aa", fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: "#e4e4e7" }}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fill: "#a1a1aa", fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: "#e4e4e7" }}
            width={48}
          />
          <ReferenceLine y={0} stroke="#e4e4e7" strokeDasharray="3 3" />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            formatter={(value: string) => {
              const id = value.replace("f_", "");
              return featureLabels[Number(id)] ?? `#${id}`;
            }}
            wrapperStyle={{ fontSize: 11, color: "#71717a" }}
          />
          {featureIds.map((id, i) => (
            <Line
              key={id}
              type="monotone"
              dataKey={`f_${id}`}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 3, strokeWidth: 0 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
