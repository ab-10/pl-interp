"use client";

import { useState } from "react";
import FeatureSlider from "./FeatureSlider";

interface CustomFeatureInputProps {
  customStrengths: Record<number, number>;
  onAdd: (id: number, strength: number) => void;
  onRemove: (id: number) => void;
  onChange: (id: number, strength: number) => void;
}

export default function CustomFeatureInput({
  customStrengths,
  onAdd,
  onRemove,
  onChange,
}: CustomFeatureInputProps) {
  const [featureIndex, setFeatureIndex] = useState("");
  const [strength, setStrength] = useState(1.0);

  const handleAdd = () => {
    const idx = parseInt(featureIndex, 10);
    if (isNaN(idx) || idx < 0 || idx > 131071) return;
    onAdd(idx, strength);
    setFeatureIndex("");
    setStrength(1.0);
  };

  const entries = Object.entries(customStrengths).map(([id, s]) => ({
    id: Number(id),
    strength: s,
  }));

  return (
    <div className="flex flex-col gap-3">
      <h3 className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
        Custom Features
      </h3>

      <div className="flex items-end gap-1.5">
        <div className="flex flex-1 flex-col gap-1">
          <label className="text-[10px] text-zinc-600">Index</label>
          <input
            type="number"
            min={0}
            max={131071}
            value={featureIndex}
            onChange={(e) => setFeatureIndex(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAdd()}
            placeholder="42000"
            className="rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1 text-xs font-mono text-zinc-100 placeholder:text-zinc-700 focus:border-blue-500/50 focus:outline-none focus:ring-1 focus:ring-blue-500/30"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[10px] text-zinc-600">Strength</label>
          <input
            type="number"
            min={-5}
            max={5}
            step={0.5}
            value={strength}
            onChange={(e) => setStrength(parseFloat(e.target.value))}
            onKeyDown={(e) => e.key === "Enter" && handleAdd()}
            className="w-14 rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1 text-xs font-mono text-zinc-100 focus:border-blue-500/50 focus:outline-none focus:ring-1 focus:ring-blue-500/30"
          />
        </div>
        <button
          onClick={handleAdd}
          disabled={featureIndex === "" || isNaN(parseInt(featureIndex, 10))}
          className="rounded-md bg-zinc-800 px-2.5 py-1 text-[11px] font-medium text-zinc-300 transition-colors hover:bg-zinc-700 hover:text-zinc-100 disabled:opacity-30 disabled:cursor-not-allowed"
        >
          Add
        </button>
      </div>

      {entries.map(({ id, strength: s }) => (
        <div key={id} className="flex items-center gap-1.5">
          <div className="flex-1">
            <FeatureSlider
              label={`#${id}`}
              value={s}
              onChange={(v) => onChange(id, v)}
            />
          </div>
          <button
            onClick={() => onRemove(id)}
            className="mt-2 text-zinc-600 hover:text-red-400 transition-colors text-sm"
            title="Remove"
          >
            &times;
          </button>
        </div>
      ))}
    </div>
  );
}
