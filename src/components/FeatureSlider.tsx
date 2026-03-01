"use client";

import { useState, useRef, useEffect } from "react";
import { CodeExample, MonotonicityData, SliderConfig } from "@/lib/types";
import Sparkline from "./Sparkline";
import HighlightedCode from "./HighlightedCode";

const DEFAULT_SLIDER: SliderConfig = { min: -5, max: 5, step: 0.5, default: 0 };

const CONFIDENCE_DOT: Record<string, string> = {
  high: "bg-emerald-500",
  medium: "bg-amber-500",
  low: "bg-rose-400",
};

const VARIANT_DISPLAY: Record<string, string> = {
  type_annotations: "Type Annotations",
  error_handling: "Error Handling",
  control_flow: "Control Flow",
  decomposition: "Decomposition",
  functional_style: "Functional Style",
  recursion: "Recursion",
  verbose_documentation: "Documentation",
};

interface FeatureSliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  slider?: SliderConfig;
  sparkline?: MonotonicityData;
  description?: string;
  confidence?: "high" | "medium" | "low";
  codeExamples?: CodeExample[];
  primaryVariant?: string;
  cohensD?: number;
  bestEffect?: { property: string; effectSize: number; isMonotonic: boolean };
  featureId?: number;
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
  primaryVariant,
  cohensD,
  bestEffect,
  featureId,
}: FeatureSliderProps) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [infoOpen, setInfoOpen] = useState(false);
  const infoButtonRef = useRef<HTMLButtonElement>(null);
  const infoPopoverRef = useRef<HTMLDivElement>(null);
  const [popoverPos, setPopoverPos] = useState({ top: 0, left: 0 });
  const hasDetails = !!(description || (codeExamples && codeExamples.length > 0));
  const hasInfo = !!(primaryVariant || cohensD !== undefined);

  const handleInfoToggle = () => {
    if (!infoOpen && infoButtonRef.current) {
      const rect = infoButtonRef.current.getBoundingClientRect();
      setPopoverPos({ top: rect.bottom + 4, left: Math.max(8, rect.left) });
    }
    setInfoOpen(!infoOpen);
  };

  // Close info popover on outside click
  useEffect(() => {
    if (!infoOpen) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as Node;
      if (
        infoPopoverRef.current && !infoPopoverRef.current.contains(target) &&
        infoButtonRef.current && !infoButtonRef.current.contains(target)
      ) {
        setInfoOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [infoOpen]);

  const color =
    value > 0
      ? "text-emerald-600"
      : value < 0
        ? "text-rose-600"
        : "text-zinc-400";

  const variantDisplay = primaryVariant
    ? VARIANT_DISPLAY[primaryVariant] ?? primaryVariant.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())
    : null;

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
          {hasInfo && (
            <div className="relative flex-shrink-0">
              <button
                ref={infoButtonRef}
                onClick={handleInfoToggle}
                className="flex items-center justify-center h-3.5 w-3.5 rounded-full border border-zinc-300 text-[9px] font-medium text-zinc-400 hover:border-zinc-400 hover:text-zinc-600 transition-colors leading-none"
                title="Feature info"
              >
                i
              </button>
              {infoOpen && (
                <div
                  ref={infoPopoverRef}
                  className="fixed z-[200] w-56 rounded-lg border border-zinc-200 bg-white p-3 shadow-lg text-[11px] text-zinc-600"
                  style={{ top: popoverPos.top, left: popoverPos.left }}
                >
                  {featureId !== undefined && (
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-mono text-[10px] text-zinc-400">#{featureId}</span>
                      {confidence && (
                        <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${
                          confidence === "high" ? "bg-emerald-50 text-emerald-700" :
                          confidence === "medium" ? "bg-amber-50 text-amber-700" :
                          "bg-rose-50 text-rose-700"
                        }`}>
                          {confidence} confidence
                        </span>
                      )}
                    </div>
                  )}

                  {description && (
                    <p className="mb-2 text-zinc-700">{description}</p>
                  )}

                  <div className="flex flex-col gap-1.5">
                    {variantDisplay && (
                      <div className="flex items-center justify-between">
                        <span className="text-zinc-400">Probe direction</span>
                        <span className="font-medium text-zinc-700">{variantDisplay}</span>
                      </div>
                    )}
                    {cohensD !== undefined && (
                      <div className="flex items-center justify-between">
                        <span className="text-zinc-400">Task impact</span>
                        <span className={`font-mono font-medium ${
                          Math.abs(cohensD) > 1 ? "text-emerald-600" :
                          Math.abs(cohensD) > 0.5 ? "text-amber-600" :
                          "text-zinc-500"
                        }`}>
                          d={cohensD > 0 ? "+" : ""}{cohensD.toFixed(2)}
                        </span>
                      </div>
                    )}
                    {bestEffect && (
                      <div className="flex items-center justify-between">
                        <span className="text-zinc-400">Strongest effect</span>
                        <span className="text-zinc-700">
                          {VARIANT_DISPLAY[bestEffect.property] ?? bestEffect.property}
                          {" "}
                          <span className="font-mono text-[10px]">
                            ({bestEffect.effectSize > 0 ? "+" : ""}{bestEffect.effectSize.toFixed(3)})
                          </span>
                        </span>
                      </div>
                    )}
                    {bestEffect && (
                      <div className="flex items-center justify-between">
                        <span className="text-zinc-400">Monotonic</span>
                        <span className={bestEffect.isMonotonic ? "text-emerald-600" : "text-zinc-400"}>
                          {bestEffect.isMonotonic ? "Yes" : "No"}
                        </span>
                      </div>
                    )}
                  </div>

                  {cohensD !== undefined && (
                    <p className="mt-2 text-[10px] text-zinc-400 border-t border-zinc-100 pt-1.5">
                      Cohen&apos;s d measures effect on code task pass rate.
                      {Math.abs(cohensD) > 1 ? " Large effect." :
                       Math.abs(cohensD) > 0.5 ? " Medium effect." :
                       " Small effect."}
                    </p>
                  )}
                </div>
              )}
            </div>
          )}
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
