"use client";

interface PromptInputProps {
  value: string;
  onChange: (value: string) => void;
  onGenerate: () => void;
  loading: boolean;
}

export default function PromptInput({
  value,
  onChange,
  onGenerate,
  loading,
}: PromptInputProps) {
  return (
    <div className="flex flex-col gap-2">
      <label htmlFor="prompt" className="text-sm font-medium text-zinc-700">
        Prompt
      </label>
      <textarea
        id="prompt"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            onGenerate();
          }
        }}
        placeholder="def fibonacci(n):"
        rows={4}
        className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 font-mono text-sm text-zinc-900 placeholder:text-zinc-400 focus:border-orange-500 focus:outline-none focus:ring-1 focus:ring-orange-500"
      />
      <button
        onClick={onGenerate}
        disabled={loading || !value.trim()}
        className="w-full rounded-lg bg-orange-500 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-orange-600 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading ? "Generating..." : "Generate (Ctrl+Enter)"}
      </button>
    </div>
  );
}
