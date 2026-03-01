"use client";

import { SAEDecompositionEntry } from "@/lib/types";

interface SAEDecompositionBarProps {
  token: string;
  decomposition: SAEDecompositionEntry[];
  reconstructionError: number;
  highlightedFeatureId: number;
  featureActivation: number;
}

const SEGMENT_COLORS = [
  "#3b82f6", // blue-500
  "#8b5cf6", // violet-500
  "#ec4899", // pink-500
  "#f59e0b", // amber-500
  "#14b8a6", // teal-500
  "#ef4444", // red-500
  "#6366f1", // indigo-500
  "#84cc16", // lime-500
  "#f97316", // orange-500
  "#06b6d4", // cyan-500
];

export default function SAEDecompositionBar({
  token,
  decomposition,
  reconstructionError,
  highlightedFeatureId,
  featureActivation,
}: SAEDecompositionBarProps) {
  const total =
    decomposition.reduce((sum, d) => sum + Math.abs(d.activation), 0) +
    reconstructionError;

  if (total === 0) {
    return (
      <div data-testid="sae-decomposition">
        <h3 className="text-sm font-medium text-zinc-700 mb-2">
          SAE Decomposition at{" "}
          <span data-testid="detail-token-label" className="font-mono text-orange-600">
            &ldquo;{token}&rdquo;
          </span>
        </h3>
        <p className="text-xs text-zinc-500">No active features at this position.</p>
      </div>
    );
  }

  return (
    <div data-testid="sae-decomposition">
      <h3 className="text-sm font-medium text-zinc-700 mb-2">
        SAE Decomposition at{" "}
        <span data-testid="detail-token-label" className="font-mono text-orange-600">
          &ldquo;{token}&rdquo;
        </span>
      </h3>

      {/* Stacked bar */}
      <div className="flex h-8 rounded overflow-hidden">
        {decomposition.map((entry, i) => {
          const isHighlighted = entry.feature_id === highlightedFeatureId;
          const widthPct = (Math.abs(entry.activation) / total) * 100;
          return (
            <div
              key={entry.feature_id}
              data-testid="sae-segment"
              data-feature-id={String(entry.feature_id)}
              data-highlighted={isHighlighted ? "true" : "false"}
              className="relative flex items-center justify-center text-xs text-white overflow-hidden"
              style={{
                width: `${widthPct}%`,
                minWidth: widthPct > 0 ? "2px" : "0",
                backgroundColor: isHighlighted
                  ? "transparent"
                  : SEGMENT_COLORS[i % SEGMENT_COLORS.length],
                backgroundImage: isHighlighted
                  ? `repeating-linear-gradient(45deg, #f97316, #f97316 4px, #ea580c 4px, #ea580c 8px)`
                  : undefined,
                borderRight: "1px solid rgba(255,255,255,0.3)",
              }}
              title={`Feature ${entry.feature_id}: ${entry.label} (${entry.activation.toFixed(3)})`}
            >
              {widthPct > 8 && (
                <span className="truncate px-1 text-[10px] font-mono">
                  {isHighlighted ? `#${entry.feature_id}` : entry.feature_id}
                </span>
              )}
            </div>
          );
        })}
        {/* Reconstruction error segment */}
        {reconstructionError > 0 && (
          <div
            className="flex items-center justify-center text-[10px] text-zinc-500"
            style={{
              width: `${(reconstructionError / total) * 100}%`,
              minWidth: "2px",
              backgroundColor: "#d4d4d8", // zinc-300
            }}
            title={`Reconstruction error: ${reconstructionError.toFixed(3)}`}
          >
            {(reconstructionError / total) * 100 > 5 && "err"}
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        {decomposition.map((entry, i) => {
          const isHighlighted = entry.feature_id === highlightedFeatureId;
          return (
            <div key={entry.feature_id} className="flex items-center gap-1.5">
              <div
                className="w-3 h-3 rounded-sm flex-shrink-0"
                style={{
                  backgroundColor: isHighlighted
                    ? "#f97316"
                    : SEGMENT_COLORS[i % SEGMENT_COLORS.length],
                }}
              />
              <span className="text-zinc-500 font-mono">#{entry.feature_id}</span>
              <span className="text-zinc-700 truncate">{entry.label}</span>
              <span className="text-zinc-400 ml-auto">{entry.activation.toFixed(2)}</span>
            </div>
          );
        })}
        {reconstructionError > 0 && (
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-sm flex-shrink-0 bg-zinc-300" />
            <span className="text-zinc-500">recon. error</span>
            <span className="text-zinc-400 ml-auto">
              {reconstructionError.toFixed(2)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
