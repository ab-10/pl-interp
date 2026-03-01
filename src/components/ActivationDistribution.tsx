"use client";

import { TokenActivation } from "@/lib/types";

const COLORS = [
  "#3b82f6", "#8b5cf6", "#06b6d4", "#10b981",
  "#f59e0b", "#ef4444", "#ec4899", "#f97316",
];

interface ActivationDistributionProps {
  tokens: TokenActivation[];
  featureIds: number[];
  featureLabels: Record<number, string>;
}

interface FeatureStats {
  id: number;
  label: string;
  color: string;
  firingRate: number;
  mean: number;
  max: number;
  values: number[];
  histogram: number[];
}

const NUM_BINS = 24;
const FIRE_THRESHOLD = 0.01;

function computeStats(
  tokens: TokenActivation[],
  featureId: number,
  label: string,
  color: string,
): FeatureStats {
  const key = String(featureId);
  const values: number[] = [];
  let firingCount = 0;
  let sum = 0;
  let max = 0;

  for (const tok of tokens) {
    const v = tok.activations[key] ?? 0;
    values.push(v);
    if (Math.abs(v) > FIRE_THRESHOLD) firingCount++;
    sum += v;
    if (Math.abs(v) > Math.abs(max)) max = v;
  }

  const mean = tokens.length > 0 ? sum / tokens.length : 0;
  const firingRate = tokens.length > 0 ? firingCount / tokens.length : 0;

  // Build histogram of absolute values
  const absMax = Math.abs(max) || 1;
  const histogram = new Array(NUM_BINS).fill(0);
  for (const v of values) {
    const bin = Math.min(Math.floor((Math.abs(v) / absMax) * NUM_BINS), NUM_BINS - 1);
    histogram[bin]++;
  }

  return { id: featureId, label, color, firingRate, mean, max, values, histogram };
}

/** Per-feature activation distribution with histogram + summary stats. */
export default function ActivationDistribution({
  tokens,
  featureIds,
  featureLabels,
}: ActivationDistributionProps) {
  if (tokens.length === 0 || featureIds.length === 0) return null;

  const stats = featureIds.map((id, i) =>
    computeStats(tokens, id, featureLabels[id] ?? `#${id}`, COLORS[i % COLORS.length]),
  );

  const histMax = Math.max(...stats.flatMap((s) => s.histogram));

  return (
    <div className="flex flex-col gap-3">
      {stats.map((s) => (
        <div key={s.id} className="flex flex-col gap-1.5 rounded-lg border border-zinc-100 bg-zinc-50/50 p-3">
          {/* Header */}
          <div className="flex items-center gap-2">
            <span
              className="h-2.5 w-2.5 rounded-full flex-shrink-0"
              style={{ backgroundColor: s.color }}
            />
            <span className="text-[12px] font-medium text-zinc-700 truncate">{s.label}</span>
            <span className="text-[10px] text-zinc-400 ml-auto font-mono">#{s.id}</span>
          </div>

          {/* Histogram */}
          <div className="flex items-end gap-px h-10">
            {s.histogram.map((count, i) => {
              const h = histMax > 0 ? (count / histMax) * 100 : 0;
              return (
                <div
                  key={i}
                  className="flex-1 rounded-t-sm transition-colors"
                  style={{
                    height: `${Math.max(h, 1)}%`,
                    backgroundColor: h > 1 ? s.color : "#e4e4e7",
                    opacity: h > 1 ? 0.6 : 0.3,
                  }}
                />
              );
            })}
          </div>
          <div className="flex items-center justify-between text-[10px] text-zinc-400">
            <span>0</span>
            <span>|{Math.abs(s.max).toFixed(3)}|</span>
          </div>

          {/* Stats row */}
          <div className="flex items-center gap-4 text-[11px]">
            <div>
              <span className="text-zinc-400">Firing: </span>
              <span className="font-mono font-medium" style={{ color: s.firingRate > 0.5 ? s.color : "#71717a" }}>
                {(s.firingRate * 100).toFixed(0)}%
              </span>
            </div>
            <div>
              <span className="text-zinc-400">Mean: </span>
              <span className="font-mono text-zinc-600">
                {s.mean > 0 ? "+" : ""}{s.mean.toFixed(4)}
              </span>
            </div>
            <div>
              <span className="text-zinc-400">Peak: </span>
              <span className="font-mono font-medium" style={{ color: s.color }}>
                {s.max > 0 ? "+" : ""}{s.max.toFixed(4)}
              </span>
            </div>
            <div>
              <span className="text-zinc-400">Tokens: </span>
              <span className="font-mono text-zinc-600">{tokens.length}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
