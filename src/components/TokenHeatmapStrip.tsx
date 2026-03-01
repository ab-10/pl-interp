"use client";

interface TokenHeatmapStripProps {
  tokens: string[];
  activations: number[];
  featureId: number;
  selectedIndex: number | null;
  onTokenClick: (index: number) => void;
}

function activationToColor(value: number, maxValue: number): string {
  if (maxValue <= 0 || value <= 0) return "rgb(39, 39, 42)"; // zinc-800
  const t = Math.min(value / maxValue, 1);
  // Interpolate from zinc-800 to green-500
  const r = Math.round(39 + (34 - 39) * t);
  const g = Math.round(39 + (197 - 39) * t);
  const b = Math.round(42 + (94 - 42) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

export default function TokenHeatmapStrip({
  tokens,
  activations,
  featureId,
  selectedIndex,
  onTokenClick,
}: TokenHeatmapStripProps) {
  const maxAct = Math.max(...activations, 0);

  return (
    <div data-testid="heatmap-strip" data-feature-id={String(featureId)}>
      <div className="flex overflow-x-auto gap-px pb-2">
        {tokens.map((token, i) => (
          <button
            key={i}
            data-testid="heatmap-cell"
            data-activation={activations[i]}
            className={`flex-shrink-0 px-1.5 py-1 font-mono text-xs text-white cursor-pointer border-2 transition-colors ${
              selectedIndex === i
                ? "border-yellow-400"
                : "border-transparent"
            }`}
            style={{ backgroundColor: activationToColor(activations[i], maxAct) }}
            onClick={() => onTokenClick(i)}
            title={`${token} (activation: ${activations[i].toFixed(3)})`}
          >
            {token || "\u00A0"}
          </button>
        ))}
      </div>
      {/* Color scale legend */}
      <div className="flex items-center gap-2 mt-1 text-xs text-zinc-400">
        <span>0</span>
        <div
          className="h-3 w-32 rounded"
          style={{
            background: `linear-gradient(to right, rgb(39, 39, 42), rgb(34, 197, 94))`,
          }}
        />
        <span>{maxAct.toFixed(1)}</span>
      </div>
    </div>
  );
}
