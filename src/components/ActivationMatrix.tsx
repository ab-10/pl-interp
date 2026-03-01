"use client";

import { useState } from "react";
import { TokenActivation } from "@/lib/types";

const COLORS = [
  "#3b82f6", "#8b5cf6", "#06b6d4", "#10b981",
  "#f59e0b", "#ef4444", "#ec4899", "#f97316",
];

interface ActivationMatrixProps {
  tokens: TokenActivation[];
  featureIds: number[];
  featureLabels: Record<number, string>;
}

/** Grid heatmap: rows = tokens, columns = features, cell color = activation intensity. */
export default function ActivationMatrix({
  tokens,
  featureIds,
  featureLabels,
}: ActivationMatrixProps) {
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);

  if (tokens.length === 0 || featureIds.length === 0) return null;

  const MAX_TOKENS = 150;
  const displayTokens = tokens.slice(0, MAX_TOKENS);
  const featureKeys = featureIds.map(String);

  // Compute max activation per feature for independent normalization
  const maxPerFeature: Record<string, number> = {};
  for (const key of featureKeys) {
    let m = 0;
    for (const tok of displayTokens) {
      const v = Math.abs(tok.activations[key] ?? 0);
      if (v > m) m = v;
    }
    maxPerFeature[key] = m || 1;
  }

  const formatToken = (t: string) => {
    return t
      .replace(/\n/g, "\u21b5")
      .replace(/\t/g, "\u2192")
      .replace(/ /g, "\u00b7");
  };

  return (
    <div className="overflow-auto rounded-lg border border-zinc-200 bg-white" style={{ maxHeight: 480 }}>
      <table className="border-collapse text-[11px]">
        <thead>
          <tr>
            <th className="sticky left-0 top-0 z-20 bg-zinc-50 border-b border-r border-zinc-200 px-2 py-1.5 text-left font-medium text-zinc-400 min-w-[60px]">
              Token
            </th>
            {featureIds.map((id, i) => (
              <th
                key={id}
                className="sticky top-0 z-10 bg-zinc-50 border-b border-zinc-200 px-2 py-1.5 text-center font-medium whitespace-nowrap"
                style={{ color: COLORS[i % COLORS.length] }}
              >
                <span className="max-w-[100px] block truncate" title={featureLabels[id] ?? `#${id}`}>
                  {featureLabels[id] ?? `#${id}`}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {displayTokens.map((tok, rowIdx) => (
            <tr key={rowIdx} className="group">
              <td className="sticky left-0 z-10 bg-white group-hover:bg-zinc-50 border-r border-zinc-100 px-2 py-0.5 font-mono text-zinc-600 whitespace-nowrap max-w-[80px] truncate">
                {formatToken(tok.token)}
              </td>
              {featureKeys.map((key, colIdx) => {
                const val = tok.activations[key] ?? 0;
                const norm = Math.abs(val) / maxPerFeature[key];
                const color = COLORS[colIdx % COLORS.length];
                const isHovered = hoveredCell?.row === rowIdx && hoveredCell?.col === colIdx;

                return (
                  <td
                    key={key}
                    className="relative px-0 py-0 text-center cursor-default"
                    style={{
                      backgroundColor: norm > 0.01 ? `${color}${Math.round(norm * 0.45 * 255).toString(16).padStart(2, "0")}` : undefined,
                      minWidth: 48,
                      height: 22,
                    }}
                    onMouseEnter={() => setHoveredCell({ row: rowIdx, col: colIdx })}
                    onMouseLeave={() => setHoveredCell(null)}
                  >
                    {isHovered && Math.abs(val) > 1e-6 && (
                      <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 z-30 whitespace-nowrap rounded border border-zinc-200 bg-white px-2 py-1 text-[10px] text-zinc-700 shadow-lg pointer-events-none">
                        <span className="font-mono">{val > 0 ? "+" : ""}{val.toFixed(4)}</span>
                        <span className="text-zinc-400 ml-1">{formatToken(tok.token)}</span>
                      </span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {tokens.length > MAX_TOKENS && (
        <p className="text-center text-[10px] text-zinc-400 py-1.5 border-t border-zinc-100">
          Showing {MAX_TOKENS} of {tokens.length} tokens
        </p>
      )}
    </div>
  );
}
