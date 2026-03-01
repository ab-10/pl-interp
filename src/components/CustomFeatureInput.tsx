"use client";

import { useState } from "react";
import FeatureSlider from "./FeatureSlider";

interface CustomFeatureInputProps {
  customFeatures: { id: number; strength: number }[];
  onAdd: (id: number, strength: number) => void;
  onRemove: (idx: number) => void;
  onChange: (idx: number, strength: number) => void;
}

export default function CustomFeatureInput({
  customFeatures,
  onAdd,
  onRemove,
  onChange,
}: CustomFeatureInputProps) {
  const [featureIndex, setFeatureIndex] = useState("");
  const [strength, setStrength] = useState(5);

  const handleAdd = () => {
    const idx = parseInt(featureIndex, 10);
    if (isNaN(idx) || idx < 0) return;
    onAdd(idx, strength);
    setFeatureIndex("");
    setStrength(5);
  };

  return (
    <div className="flex flex-col gap-3">
      <h3 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
        Custom Features
      </h3>

      <div className="flex items-end gap-2">
        <div className="flex flex-1 flex-col gap-1">
          <label className="text-xs text-zinc-500 dark:text-zinc-400">
            Feature ID
          </label>
          <input
            type="number"
            min={0}
            value={featureIndex}
            onChange={(e) => setFeatureIndex(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAdd()}
            placeholder="e.g. 1234"
            className="rounded border border-zinc-300 bg-white px-2 py-1 text-sm text-zinc-900 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-100"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-zinc-500 dark:text-zinc-400">
            Strength
          </label>
          <input
            type="number"
            min={-10}
            max={10}
            step={0.5}
            value={strength}
            onChange={(e) => setStrength(parseFloat(e.target.value))}
            onKeyDown={(e) => e.key === "Enter" && handleAdd()}
            className="w-16 rounded border border-zinc-300 bg-white px-2 py-1 text-sm text-zinc-900 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-100"
          />
        </div>
        <button
          onClick={handleAdd}
          disabled={featureIndex === "" || isNaN(parseInt(featureIndex, 10))}
          className="rounded bg-blue-600 px-3 py-1 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40"
        >
          Add
        </button>
      </div>

      {customFeatures.map((cf, idx) => (
        <div key={idx} className="flex items-center gap-2">
          <div className="flex-1">
            <FeatureSlider
              label={`#${cf.id}`}
              value={cf.strength}
              onChange={(v) => onChange(idx, v)}
            />
          </div>
          <button
            onClick={() => onRemove(idx)}
            className="mt-3 text-zinc-400 hover:text-red-500"
            title="Remove"
          >
            &times;
          </button>
        </div>
      ))}
    </div>
  );
}
