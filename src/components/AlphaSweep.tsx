"use client";

import { useCallback, useEffect } from "react";
import { SweepResult } from "@/lib/types";
import TokenHeatmap from "./TokenHeatmap";

interface AlphaSweepProps {
  results: SweepResult[];
  selectedIndex: number;
  onIndexChange: (index: number) => void;
  activeFeatureIds: number[];
  showHeatmap: boolean;
}

/** Discrete scrubber for browsing code at different steering alpha levels. */
export default function AlphaSweep({
  results,
  selectedIndex,
  onIndexChange,
  activeFeatureIds,
  showHeatmap,
}: AlphaSweepProps) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft" && selectedIndex > 0) {
        onIndexChange(selectedIndex - 1);
      } else if (e.key === "ArrowRight" && selectedIndex < results.length - 1) {
        onIndexChange(selectedIndex + 1);
      }
    },
    [selectedIndex, results.length, onIndexChange],
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  if (results.length === 0) return null;

  const current = results[selectedIndex];

  return (
    <div className="flex flex-col gap-3">
      {/* Scrubber bar */}
      <div className="flex items-center gap-1">
        <span className="text-xs text-zinc-500 mr-2">Alpha:</span>
        {results.map((r, i) => (
          <button
            key={i}
            onClick={() => onIndexChange(i)}
            className={`px-2 py-0.5 text-xs rounded font-mono transition-colors ${
              i === selectedIndex
                ? "bg-blue-600 text-white"
                : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200"
            }`}
          >
            {r.alpha > 0 ? "+" : ""}{r.alpha.toFixed(1)}x
          </button>
        ))}
      </div>

      {/* Code at selected alpha */}
      {showHeatmap && current?.token_activations ? (
        <TokenHeatmap
          tokens={current.token_activations}
          activeFeatureIds={activeFeatureIds}
        />
      ) : (
        <pre className="overflow-auto rounded-lg bg-zinc-900 p-4 text-sm leading-relaxed">
          <code className="text-zinc-300">{current?.text ?? ""}</code>
        </pre>
      )}
    </div>
  );
}
