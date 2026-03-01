"use client";

import { Feature } from "@/lib/types";
import FeatureSlider from "./FeatureSlider";
import CustomFeatureInput from "./CustomFeatureInput";

/** Composite key for per-feature strength state: "layer:id" */
function featureKey(layer: number, id: number): string {
  return `${layer}:${id}`;
}

interface FeaturePanelProps {
  features: Feature[];
  strengths: Record<string, number>;
  onStrengthChange: (layer: number, id: number, strength: number) => void;
  loading: boolean;
  customFeatures: { id: number; layer: number; strength: number }[];
  onCustomAdd: (id: number, layer: number, strength: number) => void;
  onCustomRemove: (idx: number) => void;
  onCustomChange: (idx: number, strength: number) => void;
}

export default function FeaturePanel({
  features,
  strengths,
  onStrengthChange,
  loading,
  customFeatures,
  onCustomAdd,
  onCustomRemove,
  onCustomChange,
}: FeaturePanelProps) {
  if (loading) {
    return (
      <div className="flex flex-col gap-3">
        <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">Features</h3>
        {[...Array(5)].map((_, i) => (
          <div key={i} className="flex flex-col gap-1 animate-pulse">
            <div className="h-4 w-3/4 rounded bg-zinc-200 dark:bg-zinc-700" />
            <div className="h-5 w-full rounded bg-zinc-200 dark:bg-zinc-700" />
          </div>
        ))}
      </div>
    );
  }

  // Group features by layer
  const byLayer = new Map<number, Feature[]>();
  for (const f of features) {
    const group = byLayer.get(f.layer) ?? [];
    group.push(f);
    byLayer.set(f.layer, group);
  }
  const layers = [...byLayer.keys()].sort((a, b) => a - b);

  return (
    <div className="flex flex-col gap-3">
      <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">Features</h3>

      {layers.map((layer) => (
        <div key={layer} className="flex flex-col gap-2">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-400 dark:text-zinc-500">
            Layer {layer}
          </h4>
          {byLayer.get(layer)!.map((feature) => (
            <FeatureSlider
              key={featureKey(feature.layer, feature.id)}
              label={`#${feature.id} ${feature.label}`}
              value={strengths[featureKey(feature.layer, feature.id)] ?? 0}
              onChange={(v) => onStrengthChange(feature.layer, feature.id, v)}
            />
          ))}
        </div>
      ))}

      {features.length === 0 && (
        <p className="text-xs text-zinc-400 dark:text-zinc-500">
          No labeled features yet. Use custom features below.
        </p>
      )}

      <hr className="border-zinc-200 dark:border-zinc-700" />

      <CustomFeatureInput
        customFeatures={customFeatures}
        onAdd={onCustomAdd}
        onRemove={onCustomRemove}
        onChange={onCustomChange}
      />
    </div>
  );
}
