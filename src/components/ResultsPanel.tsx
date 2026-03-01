"use client";

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

  return (
    <div className="grid h-full grid-cols-2 gap-4">
      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">Baseline</h3>
        <pre className="flex-1 overflow-auto rounded-lg bg-zinc-900 p-4 text-sm text-zinc-100">
          <code>{baseline}</code>
        </pre>
      </div>
      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">Steered</h3>
        <pre className="flex-1 overflow-auto rounded-lg bg-zinc-900 p-4 text-sm text-zinc-100">
          <code>{steered}</code>
        </pre>
      </div>
    </div>
  );
}
