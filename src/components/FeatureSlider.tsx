"use client";

import { MonotonicityData, SliderConfig } from "@/lib/types";
import Sparkline from "./Sparkline";

const DEFAULT_SLIDER: SliderConfig = { min: -5, max: 5, step: 0.5, default: 0 };

interface FeatureSliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  slider?: SliderConfig;
  sparkline?: MonotonicityData;
}

export default function FeatureSlider({
  label,
  value,
  onChange,
  slider = DEFAULT_SLIDER,
  sparkline,
}: FeatureSliderProps) {
  const color =
    value > 0
      ? "text-emerald-600"
      : value < 0
        ? "text-rose-600"
        : "text-zinc-400";

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between gap-2">
        <span
          className="min-w-0 truncate text-[12px] text-zinc-700"
          title={label}
        >
          {label}
        </span>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          {sparkline && <Sparkline data={sparkline} />}
          <span className={`font-mono text-[11px] font-medium w-9 text-right ${color}`}>
            {value > 0 ? "+" : ""}{value.toFixed(1)}
          </span>
        </div>
      </div>
      <input
        type="range"
        min={slider.min}
        max={slider.max}
        step={slider.step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
    </div>
  );
}
