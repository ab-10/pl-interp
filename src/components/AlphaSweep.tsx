"use client";

import { useCallback, useEffect } from "react";
import { SweepResult } from "@/lib/types";
import { diffLines, Change } from "diff";

interface AlphaSweepProps {
  results: SweepResult[];
  baselineText: string;
  selectedIndex: number;
  onIndexChange: (index: number) => void;
}

/** Discrete scrubber showing diff vs baseline at different steering alpha levels. */
export default function AlphaSweep({
  results,
  baselineText,
  selectedIndex,
  onIndexChange,
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
  const changes = diffLines(baselineText, current?.text ?? "");

  return (
    <div className="flex flex-col gap-3">
      {/* Scrubber bar */}
      <div className="flex items-center gap-1">
        <span className="text-xs text-zinc-400 mr-2">Alpha:</span>
        {results.map((r, i) => (
          <button
            key={i}
            onClick={() => onIndexChange(i)}
            className={`px-2 py-0.5 text-xs rounded font-mono transition-colors ${
              i === selectedIndex
                ? "bg-zinc-900 text-white"
                : "bg-zinc-100 text-zinc-500 hover:bg-zinc-200 hover:text-zinc-700"
            }`}
          >
            {r.alpha > 0 ? "+" : ""}{r.alpha.toFixed(1)}x
          </button>
        ))}
      </div>

      {/* Diff view at selected alpha */}
      <pre className="overflow-auto rounded-lg border border-zinc-200 bg-zinc-50 p-4 text-[13px] leading-relaxed font-mono">
        <code>
          {changes.map((change: Change, i: number) => {
            const lines = change.value.replace(/\n$/, "").split("\n");
            return lines.map((line: string, j: number) => {
              if (change.added) {
                return (
                  <div key={`${i}-${j}`} className="bg-emerald-50 text-emerald-800">
                    <span className="mr-2 inline-block w-4 select-none text-emerald-400">+</span>
                    {line}
                  </div>
                );
              }
              if (change.removed) {
                return (
                  <div key={`${i}-${j}`} className="bg-red-50 text-red-800">
                    <span className="mr-2 inline-block w-4 select-none text-red-400">-</span>
                    {line}
                  </div>
                );
              }
              return (
                <div key={`${i}-${j}`} className="text-zinc-600">
                  <span className="mr-2 inline-block w-4 select-none text-zinc-300">&nbsp;</span>
                  {line}
                </div>
              );
            });
          })}
        </code>
      </pre>
    </div>
  );
}
