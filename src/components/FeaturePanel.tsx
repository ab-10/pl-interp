"use client";

import { Feature } from "@/lib/types";
import FeatureSlider from "./FeatureSlider";

interface FeaturePanelProps {
  features: Feature[];
  strengths: Record<number, number>;
  onStrengthChange: (id: number, strength: number) => void;
  loading: boolean;
}

export default function FeaturePanel({
  features,
  strengths,
  onStrengthChange,
  loading,
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
          label={feature.label}
          value={strengths[feature.id] ?? 0}
          onChange={(v) => onStrengthChange(feature.id, v)}
        />
      ))}
    </div>
  );
}
