"use client";

import { Feature } from "@/lib/types";

interface FeatureSelectorProps {
  features: Feature[];
  selectedFeatureId: number;
  onChange: (featureId: number) => void;
}

export default function FeatureSelector({
  features,
  selectedFeatureId,
  onChange,
}: FeatureSelectorProps) {
  return (
    <div className="flex items-center gap-2">
      <label className="text-sm font-medium text-zinc-300">Feature:</label>
      <select
        data-testid="feature-selector"
        value={selectedFeatureId}
        onChange={(e) => onChange(Number(e.target.value))}
        className="rounded border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
      >
        {features.map((f) => (
          <option key={f.id} value={f.id}>
            {f.id} — {f.label}
          </option>
        ))}
      </select>
    </div>
  );
}
