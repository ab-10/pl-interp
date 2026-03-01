"use client";

import { useState } from "react";
import { EnrichedFeature } from "@/lib/types";
import FeatureSlider from "./FeatureSlider";
import CustomFeatureInput from "./CustomFeatureInput";

interface FeaturePanelProps {
  features: EnrichedFeature[];
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
  const [controlsOpen, setControlsOpen] = useState(false);

  if (loading) {
    return (
      <div className="flex flex-col gap-3">
        <h3 className="text-[11px] font-medium uppercase tracking-wider text-zinc-400">
          Features
        </h3>
        {[...Array(4)].map((_, i) => (
          <div key={i} className="flex flex-col gap-1.5 animate-pulse">
            <div className="h-3 w-2/3 rounded bg-zinc-200" />
            <div className="h-[6px] w-full rounded-full bg-zinc-200" />
          </div>
        ))}
      </div>
    );
  }

  const steeringFeatures = features.filter((f) => f.category !== "control");
  const controlFeatures = features.filter((f) => f.category === "control");

  const getBestSparkline = (f: EnrichedFeature) => {
    if (!f.monotonicity) return undefined;
    const entries = Object.values(f.monotonicity);
    if (entries.length === 0) return undefined;
    return entries.reduce((best, curr) =>
      Math.abs(curr.effect_size) > Math.abs(best.effect_size) ? curr : best
    );
  };

  return (
    <div className="flex flex-col gap-3">
      <h3 className="text-[11px] font-medium uppercase tracking-wider text-zinc-400">
        Steering
      </h3>
      {steeringFeatures.map((feature) => (
        <FeatureSlider
          key={feature.id}
          label={feature.label}
          value={strengths[feature.id] ?? 0}
          onChange={(v) => onStrengthChange(feature.id, v)}
          slider={feature.slider}
          sparkline={getBestSparkline(feature)}
          description={feature.description}
          confidence={feature.confidence}
          codeExamples={feature.code_examples}
        />
      ))}

      {controlFeatures.length > 0 && (
        <>
          <button
            onClick={() => setControlsOpen(!controlsOpen)}
            className="flex items-center gap-1.5 text-[11px] text-zinc-400 hover:text-zinc-600 transition-colors mt-1"
          >
            <svg
              className={`h-3 w-3 transition-transform ${controlsOpen ? "rotate-90" : ""}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
            </svg>
            Controls ({controlFeatures.length})
          </button>
          {controlsOpen && (
            <div className="flex flex-col gap-3 ml-1 pl-3 border-l border-zinc-200">
              {controlFeatures.map((feature) => (
                <FeatureSlider
                  key={feature.id}
                  label={feature.label}
                  value={strengths[feature.id] ?? 0}
                  onChange={(v) => onStrengthChange(feature.id, v)}
                  slider={feature.slider}
                  description={feature.description}
                  confidence={feature.confidence}
                  codeExamples={feature.code_examples}
                />
              ))}
            </div>
          )}
        </>
      )}

      <div className="h-px bg-zinc-200 mt-1" />

      <CustomFeatureInput
        customStrengths={customStrengths}
        onAdd={onCustomAdd}
        onRemove={onCustomRemove}
        onChange={onCustomChange}
      />
    </div>
  );
}
