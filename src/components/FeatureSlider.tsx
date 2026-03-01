"use client";

import { useState } from "react";
import { CodeExample, MonotonicityData, SliderConfig } from "@/lib/types";
import Sparkline from "./Sparkline";

const DEFAULT_SLIDER: SliderConfig = { min: -5, max: 5, step: 0.5, default: 0 };

const CONFIDENCE_DOT: Record<string, string> = {
  high: "bg-emerald-500",
  medium: "bg-amber-500",
  low: "bg-rose-400",
};

/** Render code context with >>>token<<< markers highlighted. */
function HighlightedCode({ text }: { text: string }) {
  const parts = text.split(/(>>>.*?<<<)/g);
  return (
    <code className="text-[11px] leading-relaxed whitespace-pre-wrap break-all">
      {parts.map((part, i) =>
        part.startsWith(">>>") && part.endsWith("<<<") ? (
          <span key={i} className="bg-amber-200 text-amber-900 rounded px-0.5">
            {part.slice(3, -3)}
          </span>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </code>
  );
}

interface FeatureSliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  slider?: SliderConfig;
  sparkline?: MonotonicityData;
  description?: string;
  confidence?: "high" | "medium" | "low";
  codeExamples?: CodeExample[];
}

export default function FeatureSlider({
  label,
  value,
  onChange,
  slider = DEFAULT_SLIDER,
  sparkline,
  description,
  confidence,
  codeExamples,
}: FeatureSliderProps) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const hasDetails = !!(description || (codeExamples && codeExamples.length > 0));

  const color =
    value > 0
      ? "text-emerald-600"
      : value < 0
        ? "text-rose-600"
        : "text-zinc-400";

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          {confidence && (
            <span
              className={`h-1.5 w-1.5 rounded-full flex-shrink-0 ${CONFIDENCE_DOT[confidence]}`}
              title={`Confidence: ${confidence}`}
            />
          )}
          <span
            className="min-w-0 truncate text-[12px] text-zinc-700"
            title={label}
          >
            {label}
          </span>
        </div>
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
      {hasDetails && (
        <>
          <button
            onClick={() => setDetailsOpen(!detailsOpen)}
            className="self-start text-[10px] text-zinc-400 hover:text-zinc-600 transition-colors"
          >
            {detailsOpen ? "Hide details" : "Details"}
          </button>
          {detailsOpen && (
            <div className="flex flex-col gap-2 rounded bg-zinc-50 p-2 text-[11px] text-zinc-600">
              {description && <p>{description}</p>}
              {codeExamples && codeExamples.length > 0 && (
                <div className="flex flex-col gap-1.5">
                  <span className="text-[10px] font-medium uppercase tracking-wider text-zinc-400">
                    Top activating examples
                  </span>
                  {codeExamples.slice(0, 3).map((ex, i) => (
                    <div
                      key={i}
                      className="rounded border border-zinc-200 bg-white p-1.5 overflow-x-auto"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[10px] text-zinc-400">{ex.task_id}</span>
                        <span className="text-[10px] font-mono text-zinc-400">
                          act: {ex.activation.toFixed(4)}
                        </span>
                      </div>
                      <HighlightedCode text={ex.code_context} />
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
