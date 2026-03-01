"use client";

interface FeatureSliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
}

export default function FeatureSlider({ label, value, onChange }: FeatureSliderProps) {
  const color =
    value > 0
      ? "text-green-600 dark:text-green-400"
      : value < 0
        ? "text-red-600 dark:text-red-400"
        : "text-zinc-500 dark:text-zinc-400";

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-sm text-zinc-700 dark:text-zinc-300">{label}</span>
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
        className="w-full accent-blue-600"
      />
    </div>
  );
}
