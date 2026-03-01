"use client";

import { useState } from "react";
import { TokenActivation } from "@/lib/types";

const FEATURE_COLORS = [
  "#3b82f6", "#8b5cf6", "#06b6d4", "#10b981",
  "#f59e0b", "#ef4444", "#ec4899", "#f97316",
];

interface AnnotatedCodeProps {
  tokens: TokenActivation[];
  featureIds: number[];
  featureLabels: Record<number, string>;
}

/** Decode sentencepiece tokenizer tokens back to readable text. */
function decodeToken(token: string): string {
  let s = token;
  s = s.replace(/▁/g, " ");
  s = s.replace(/<0x([0-9A-Fa-f]{2})>/g, (_, hex) =>
    String.fromCharCode(parseInt(hex, 16)),
  );
  return s;
}

/** Interpolate between color stops based on normalized value [0, 1]. */
function heatColor(t: number): string {
  // 5-stop gradient: dark blue → blue → cyan → yellow → red
  const stops = [
    [30, 58, 138],    // #1e3a8a deep blue
    [59, 130, 246],   // #3b82f6 blue
    [6, 182, 212],    // #06b6d4 cyan
    [250, 204, 21],   // #facc15 yellow
    [220, 38, 38],    // #dc2626 red
  ];
  const clamped = Math.max(0, Math.min(1, t));
  const seg = clamped * (stops.length - 1);
  const i = Math.min(Math.floor(seg), stops.length - 2);
  const frac = seg - i;
  const r = Math.round(stops[i][0] + (stops[i + 1][0] - stops[i][0]) * frac);
  const g = Math.round(stops[i][1] + (stops[i + 1][1] - stops[i][1]) * frac);
  const b = Math.round(stops[i][2] + (stops[i + 1][2] - stops[i][2]) * frac);
  return `rgb(${r}, ${g}, ${b})`;
}

/** Renders generated code with per-token activation coloring and hover tooltips. */
export default function AnnotatedCode({
  tokens,
  featureIds,
  featureLabels,
}: AnnotatedCodeProps) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ top: 0, left: 0 });

  if (tokens.length === 0) return null;

  const featureKeys = featureIds.map(String);

  // Compute per-token total activation + min/max for normalization
  const totals: number[] = [];
  for (const tok of tokens) {
    let sum = 0;
    for (const key of featureKeys) {
      sum += tok.activations[key] ?? 0;
    }
    totals.push(sum);
  }
  const minAct = Math.min(...totals);
  const maxAct = Math.max(...totals);
  const range = maxAct - minAct || 1;

  // Map feature IDs to colors for the tooltip
  const featureColorMap: Record<string, string> = {};
  featureIds.forEach((id, i) => {
    featureColorMap[String(id)] = FEATURE_COLORS[i % FEATURE_COLORS.length];
  });

  return (
    <>
    <pre className="overflow-auto rounded-lg border border-zinc-100 bg-zinc-50 p-4 text-[13px] leading-relaxed font-mono">
      <code>
        {tokens.map((tok, i) => {
          const t = (totals[i] - minAct) / range; // 0 = min, 1 = max
          const hasActivation = Math.abs(totals[i]) > 1e-6;
          const color = heatColor(t);
          // Opacity scales with how far from the midpoint (more saturated at extremes)
          const opacity = hasActivation ? 0.12 + t * 0.28 : 0;

          return (
            <span
              key={i}
              className="relative cursor-default"
              style={{
                backgroundColor: opacity > 0.01 ? color : undefined,
                opacity: opacity > 0.01 ? undefined : undefined,
                ...(opacity > 0.01 ? {
                  backgroundColor: `color-mix(in srgb, ${color} ${Math.round(opacity * 100)}%, transparent)`,
                } : {}),
              }}
              onMouseEnter={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                setTooltipPos({ top: rect.top - 4, left: rect.left + rect.width / 2 });
                setHoveredIdx(i);
              }}
              onMouseLeave={() => setHoveredIdx(null)}
            >
              {decodeToken(tok.token)}
            </span>
          );
        })}
      </code>
    </pre>
    {hoveredIdx !== null && (() => {
      const tok = tokens[hoveredIdx];
      const activeEntries = featureKeys
        .map((key) => ({ key, val: tok.activations[key] }))
        .filter((e) => e.val !== undefined && Math.abs(e.val) > 1e-6);
      if (activeEntries.length === 0) return null;
      return (
        <div
          className="fixed z-[200] -translate-x-1/2 -translate-y-full whitespace-nowrap rounded-lg border border-zinc-200 bg-white px-2.5 py-1.5 text-[11px] text-zinc-700 shadow-lg pointer-events-none"
          style={{ top: tooltipPos.top, left: tooltipPos.left }}
        >
          {activeEntries.map((e) => (
            <div key={e.key} className="flex items-center gap-2">
              <span
                className="h-1.5 w-1.5 rounded-full flex-shrink-0"
                style={{ backgroundColor: featureColorMap[e.key] ?? "#a1a1aa" }}
              />
              <span className="text-zinc-400 truncate max-w-[100px]">
                {featureLabels[Number(e.key)] ?? `#${e.key}`}
              </span>
              <span
                className="font-mono ml-1"
                style={{ color: e.val! > 0 ? "#3b82f6" : e.val! < 0 ? "#ef4444" : "#a1a1aa" }}
              >
                {e.val! > 0 ? "+" : ""}{e.val!.toFixed(4)}
              </span>
            </div>
          ))}
        </div>
      );
    })()}
    </>
  );
}
