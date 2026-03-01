"use client";

import { MonotonicityData } from "@/lib/types";

interface SparklineProps {
  data: MonotonicityData;
  width?: number;
  height?: number;
}

/**
 * Tiny inline SVG showing monotonicity: 3 points (neg, baseline, pos)
 * connected by a line with filled area. Green if monotonic, gray if not.
 */
export default function Sparkline({ data, width = 48, height = 16 }: SparklineProps) {
  const values = [data.neg_avg, data.baseline, data.pos_avg];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const pad = 2;
  const plotW = width - pad * 2;
  const plotH = height - pad * 2;

  const points = values.map((v, i) => ({
    x: pad + (i / 2) * plotW,
    y: pad + plotH - ((v - min) / range) * plotH,
  }));

  const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
  const areaPath = `${linePath} L ${points[2].x} ${pad + plotH} L ${points[0].x} ${pad + plotH} Z`;

  const color = data.is_monotonic ? "#22c55e" : "#71717a";
  const fillColor = data.is_monotonic ? "#22c55e" : "#71717a";

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className="inline-block flex-shrink-0"
      aria-label={`${data.is_monotonic ? "Monotonic" : "Non-monotonic"}: ${data.effect_size > 0 ? "+" : ""}${data.effect_size.toFixed(3)}`}
    >
      <path d={areaPath} fill={fillColor} opacity={0.15} />
      <path d={linePath} fill="none" stroke={color} strokeWidth={1.5} strokeLinecap="round" />
      {points.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={1.5} fill={color} />
      ))}
    </svg>
  );
}
