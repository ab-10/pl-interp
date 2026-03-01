"use client";

import { diffLines, Change } from "diff";

interface ResultsPanelProps {
  baseline: string | null;
  steered: string | null;
  loading: boolean;
}

export default function ResultsPanel({ baseline, steered, loading }: ResultsPanelProps) {
  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-zinc-700 border-t-blue-500" />
          <p className="text-xs text-zinc-500">Generating...</p>
        </div>
      </div>
    );
  }

  if (baseline === null && steered === null) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-zinc-600">
          Enter a prompt and click Generate to see results.
        </p>
      </div>
    );
  }

  const changes = diffLines(baseline ?? "", steered ?? "");

  return (
    <div className="flex h-full flex-col gap-2">
      <div className="flex items-baseline gap-3">
        <h3 className="text-xs font-medium text-zinc-400">Diff</h3>
        <p className="text-[11px] text-zinc-600">
          <span className="text-red-400/70">baseline</span>
          {" → "}
          <span className="text-emerald-400/70">steered</span>
        </p>
      </div>
      <pre className="flex-1 overflow-auto rounded-lg border border-zinc-800/50 bg-zinc-950 p-4 text-[13px] leading-relaxed font-mono">
        <code>
          {changes.map((change: Change, i: number) => {
            const lines = change.value.replace(/\n$/, "").split("\n");
            return lines.map((line: string, j: number) => {
              if (change.added) {
                return (
                  <div key={`${i}-${j}`} className="bg-emerald-500/10 text-emerald-300">
                    <span className="mr-2 inline-block w-4 select-none text-emerald-500/50">+</span>
                    {line}
                  </div>
                );
              }
              if (change.removed) {
                return (
                  <div key={`${i}-${j}`} className="bg-red-500/10 text-red-300">
                    <span className="mr-2 inline-block w-4 select-none text-red-500/50">-</span>
                    {line}
                  </div>
                );
              }
              return (
                <div key={`${i}-${j}`} className="text-zinc-400">
                  <span className="mr-2 inline-block w-4 select-none text-zinc-700">&nbsp;</span>
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
