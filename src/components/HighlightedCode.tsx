/** Render code context with >>>token<<< markers highlighted. */
export default function HighlightedCode({ text }: { text: string }) {
  const parts = text.split(/(>>>.*?<<<)/g);
  return (
    <code className="text-[11px] leading-relaxed whitespace-pre-wrap break-all">
      {parts.map((part, i) =>
        part.startsWith(">>>") && part.endsWith("<<<") ? (
          <span key={i} className="bg-amber-200 text-amber-900 rounded px-0.5">
            {part.slice(3, -3)}
          </span>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </code>
  );
}
