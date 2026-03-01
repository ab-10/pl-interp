"use client";

interface PromptInputProps {
  value: string;
  onChange: (value: string) => void;
  onGenerate: () => void;
  loading: boolean;
}

export default function PromptInput({ value, onChange, onGenerate, loading }: PromptInputProps) {
  return (
    <div className="flex flex-col gap-2">
      <label className="text-[11px] font-medium uppercase tracking-wider text-zinc-400">
        Prompt
      </label>
      <textarea
        placeholder="def fibonacci(n):"
        rows={3}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            onGenerate();
          }
        }}
        className="w-full rounded-md border border-zinc-200 bg-white px-3 py-2 font-mono text-sm text-zinc-900 placeholder:text-zinc-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500/30 resize-none"
      />
      <button
        onClick={onGenerate}
        disabled={loading || !value.trim()}
        className="w-full rounded-md bg-zinc-900 px-4 py-1.5 text-xs font-medium text-white transition-all hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-40 active:scale-[0.98]"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <span className="h-3 w-3 animate-spin rounded-full border border-white/30 border-t-white" />
            Generating...
          </span>
        ) : (
          "Generate \u2318\u21B5"
        )}
      </button>
    </div>
  );
}
