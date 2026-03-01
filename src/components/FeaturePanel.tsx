"use client";

import { useState } from "react";
import { Feature } from "@/lib/types";
import FeatureSlider from "./FeatureSlider";

interface FeaturePanelProps {
  features: Feature[];
  activeFeatureId: number | null;
  strength: number;
  onToggle: (id: number) => void;
  onStrengthChange: (strength: number) => void;
  loading: boolean;
  customFeatureIds: number[];
  onAddCustom: (id: number) => void;
  onRemoveCustom: (id: number) => void;
}

export default function FeaturePanel({
  features,
  activeFeatureId,
  strength,
  onToggle,
  onStrengthChange,
  loading,
  customFeatureIds,
  onAddCustom,
  onRemoveCustom,
}: FeaturePanelProps) {
  const [newId, setNewId] = useState("");

  if (loading) {
    return (
      <div className="flex flex-col gap-3">
        <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">Features</h3>
        {[...Array(2)].map((_, i) => (
          <div key={i} className="flex flex-col gap-1 animate-pulse">
            <div className="h-8 w-full rounded bg-zinc-200 dark:bg-zinc-700" />
          </div>
        ))}
      </div>
    );
  }

  const allFeatures: Feature[] = [
    ...features,
    ...customFeatureIds.map((id) => ({ id, label: `Feature #${id}` })),
  ];

  const activeFeature = allFeatures.find((f) => f.id === activeFeatureId);

  const handleAdd = () => {
    const id = parseInt(newId, 10);
    if (isNaN(id) || id < 0) return;
    if (allFeatures.some((f) => f.id === id)) return;
    onAddCustom(id);
    setNewId("");
  };

  return (
    <div className="flex flex-col gap-3">
      <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">Features</h3>

      {allFeatures.map((feature) => {
        const isActive = feature.id === activeFeatureId;
        const isCustom = customFeatureIds.includes(feature.id);
        return (
          <div key={feature.id} className="flex flex-col gap-2">
            <div className="flex items-center gap-1">
              <button
                onClick={() => onToggle(feature.id)}
                className={`flex-1 rounded px-3 py-2 text-left text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-blue-600 text-white"
                    : "bg-zinc-200 text-zinc-700 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
                }`}
              >
                {feature.label} <span className="font-mono text-xs opacity-50">#{feature.id}</span>
              </button>
              {isCustom && (
                <button
                  onClick={() => onRemoveCustom(feature.id)}
                  className="rounded px-1.5 py-2 text-zinc-400 hover:text-red-500 transition-colors"
                  title="Remove"
                >
                  &times;
                </button>
              )}
            </div>
            {isActive && activeFeature && (
              <FeatureSlider
                label={<>{activeFeature.label} <span className="font-mono text-xs opacity-50">#{activeFeature.id}</span></>}
                value={strength}
                onChange={onStrengthChange}
              />
            )}
          </div>
        );
      })}

      {allFeatures.length === 0 && (
        <p className="text-xs text-zinc-400 dark:text-zinc-500">
          No features available. Add a custom feature below.
        </p>
      )}

      <div className="flex items-center gap-2">
        <input
          type="number"
          min={0}
          value={newId}
          onChange={(e) => setNewId(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAdd()}
          placeholder="Feature ID"
          className="flex-1 rounded border border-zinc-300 bg-white px-2 py-1.5 text-sm text-zinc-900 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-100"
        />
        <button
          onClick={handleAdd}
          disabled={newId === "" || isNaN(parseInt(newId, 10))}
          className="rounded bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40"
        >
          Add
        </button>
      </div>
    </div>
  );
}
