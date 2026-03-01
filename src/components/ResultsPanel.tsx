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
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-zinc-300 border-t-blue-600" />
          <p className="text-sm text-zinc-500">Generating...</p>
        </div>
      </div>
    );
  }

  if (baseline === null && steered === null) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-zinc-400 dark:text-zinc-500">
          Enter a prompt and click Generate to see results.
        </p>
      </div>
    );
  }

  const changes = diffLines(baseline ?? "", steered ?? "");

  return (
    <div className="flex h-full flex-col gap-2">
      <div className="flex items-baseline gap-3">
        <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
          Diff
        </h3>
        <p className="text-xs text-zinc-400 dark:text-zinc-500">
          <span className="text-red-400">baseline</span>
          {" → "}
          <span className="text-green-400">steered</span>
        </p>
      </div>
      <pre className="flex-1 overflow-auto rounded-lg bg-zinc-900 p-4 text-sm leading-relaxed">
        <code>
          {changes.map((change: Change, i: number) => {
            const lines = change.value.replace(/\n$/, "").split("\n");
            return lines.map((line: string, j: number) => {
              if (change.added) {
                return (
                  <div key={`${i}-${j}`} className="bg-green-900/40 text-green-300">
                    <span className="mr-2 inline-block w-4 select-none text-green-500">+</span>
                    {line}
                  </div>
                );
              }
              if (change.removed) {
                return (
                  <div key={`${i}-${j}`} className="bg-red-900/40 text-red-300">
                    <span className="mr-2 inline-block w-4 select-none text-red-500">-</span>
                    {line}
                  </div>
                );
              }
              return (
                <div key={`${i}-${j}`} className="text-zinc-400">
                  <span className="mr-2 inline-block w-4 select-none text-zinc-600">&nbsp;</span>
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
