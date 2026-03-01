"use client";

interface TokenHeatmapStripProps {
  tokens: string[];
  activations: number[];
  featureId: number;
  selectedIndex: number | null;
  onTokenClick: (index: number) => void;
}

function activationToColor(value: number, maxValue: number): string {
  if (maxValue <= 0 || value <= 0) return "rgb(244, 244, 245)"; // zinc-100
  const t = Math.min(value / maxValue, 1);
  // Interpolate from zinc-100 to orange-500 (#f97316)
  const r = Math.round(244 + (249 - 244) * t);
  const g = Math.round(244 + (115 - 244) * t);
  const b = Math.round(245 + (22 - 245) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

function textColorForBg(value: number, maxValue: number): string {
  if (maxValue <= 0) return "text-zinc-700";
  const t = Math.min(value / maxValue, 1);
  return t > 0.45 ? "text-white" : "text-zinc-700";
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
        {tokens.map((token, i) => {
          const act = activations[i] ?? 0;
          return (
          <button
            key={i}
            data-testid="heatmap-cell"
            data-activation={act}
            className={`flex-shrink-0 px-1.5 py-1 font-mono text-xs cursor-pointer border-2 transition-colors ${
              textColorForBg(act, maxAct)
            } ${
              selectedIndex === i
                ? "border-orange-600"
                : "border-transparent"
            }`}
            style={{ backgroundColor: activationToColor(act, maxAct) }}
            onClick={() => onTokenClick(i)}
            title={`${token} (activation: ${act.toFixed(3)})`}
          >
            {token || "\u00A0"}
          </button>
          );
        })}
      </div>
      {/* Color scale legend */}
      <div className="flex items-center gap-2 mt-1 text-xs text-zinc-500">
        <span>0</span>
        <div
          className="h-3 w-32 rounded"
          style={{
            background: `linear-gradient(to right, rgb(244, 244, 245), rgb(249, 115, 22))`,
          }}
        />
        <span>{maxAct.toFixed(1)}</span>
      </div>
    </div>
  );
}
