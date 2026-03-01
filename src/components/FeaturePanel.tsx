"use client";

import { Feature } from "@/lib/types";
import FeatureSlider from "./FeatureSlider";
import CustomFeatureInput from "./CustomFeatureInput";

interface FeaturePanelProps {
  features: Feature[];
  strengths: Record<number, number>;
  onStrengthChange: (id: number, strength: number) => void;
  loading: boolean;
  customStrengths: Record<number, number>;
  onCustomAdd: (id: number, strength: number) => void;
  onCustomRemove: (id: number) => void;
  onCustomChange: (id: number, strength: number) => void;
}

export default function FeaturePanel({
  features,
  strengths,
  onStrengthChange,
  loading,
  customStrengths,
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

  return (
    <div className="flex flex-col gap-3">
      <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">Features</h3>
      {features.map((feature) => (
        <FeatureSlider
          key={feature.id}
          label={`#${feature.id} ${feature.label}`}
          value={strengths[feature.id] ?? 0}
          onChange={(v) => onStrengthChange(feature.id, v)}
        />
      ))}

      <hr className="border-zinc-200 dark:border-zinc-700" />

      <CustomFeatureInput
        customStrengths={customStrengths}
        onAdd={onCustomAdd}
        onRemove={onCustomRemove}
        onChange={onCustomChange}
      />
    </div>
  );
}
