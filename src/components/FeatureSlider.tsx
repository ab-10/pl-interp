"use client";

import { ReactNode } from "react";

interface FeatureSliderProps {
  label: ReactNode;
  value: number;
  onChange: (value: number) => void;
}

export default function FeatureSlider({ label, value, onChange }: FeatureSliderProps) {
  const color =
    value > 0
      ? "text-green-600"
      : value < 0
        ? "text-red-600"
        : "text-zinc-500";

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-sm text-zinc-700">{label}</span>
        <span className={`text-sm font-mono font-medium ${color}`}>
          {value > 0 ? "+" : ""}
          {value.toFixed(1)}
        </span>
      </div>
      <input
        type="range"
        min={-10}
        max={10}
        step={0.5}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-orange-500"
      />
    </div>
  );
}
