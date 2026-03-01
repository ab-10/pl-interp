"use client";

import { useState } from "react";
import { TokenActivation } from "@/lib/types";

interface TokenHeatmapProps {
  tokens: TokenActivation[];
  activeFeatureIds: number[];
}

/** Render code tokens with background color intensity proportional to SAE activation. */
export default function TokenHeatmap({ tokens, activeFeatureIds }: TokenHeatmapProps) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (tokens.length === 0) return null;

  // Compute max activation for normalization
  const featureKeys = activeFeatureIds.map(String);
  let maxAct = 0;
  for (const tok of tokens) {
    for (const key of featureKeys) {
      const v = tok.activations[key];
      if (v !== undefined && Math.abs(v) > maxAct) maxAct = Math.abs(v);
    }
  }
  if (maxAct === 0) maxAct = 1;

  return (
    <div className="relative">
      <pre className="overflow-auto rounded-lg bg-zinc-900 p-4 text-sm leading-relaxed">
        <code>
          {tokens.map((tok, i) => {
            // Sum activation across active features
            let totalAct = 0;
            for (const key of featureKeys) {
              totalAct += tok.activations[key] ?? 0;
            }
            const intensity = Math.min(Math.abs(totalAct) / maxAct, 1);
            const isPositive = totalAct >= 0;

            // Blue for positive activation, red-orange for negative
            const bg = isPositive
              ? `rgba(59, 130, 246, ${intensity * 0.6})`
              : `rgba(239, 68, 68, ${intensity * 0.6})`;

            return (
              <span
                key={i}
                className="relative cursor-default"
                style={{ backgroundColor: intensity > 0.01 ? bg : undefined }}
                onMouseEnter={() => setHoveredIdx(i)}
                onMouseLeave={() => setHoveredIdx(null)}
              >
                {tok.token}
                {hoveredIdx === i && Object.keys(tok.activations).length > 0 && (
                  <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 z-50 whitespace-nowrap rounded bg-zinc-800 border border-zinc-600 px-2 py-1 text-xs text-zinc-200 shadow-lg pointer-events-none">
                    {featureKeys.map((key) => {
                      const v = tok.activations[key];
                      if (v === undefined) return null;
                      return (
                        <div key={key}>
                          <span className="text-zinc-400">#{key}:</span>{" "}
                          <span className={v > 0 ? "text-blue-400" : v < 0 ? "text-red-400" : "text-zinc-400"}>
                            {v > 0 ? "+" : ""}{v.toFixed(4)}
                          </span>
                        </div>
                      );
                    })}
                  </span>
                )}
              </span>
            );
          })}
        </code>
      </pre>
    </div>
  );
}
