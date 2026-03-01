"use client";

import { useState, useEffect } from "react";
import { EnrichedFeature, FeatureSuccessData, RelabelResponse } from "@/lib/types";
import { fetchDefaultPrompt, fetchFeatureSuccess, relabelFeature } from "@/lib/api";
import HighlightedCode from "./HighlightedCode";
import SuccessAnalysisPanel from "./SuccessAnalysisPanel";

const CONFIDENCE_COLORS: Record<string, string> = {
  high: "bg-emerald-50 text-emerald-700",
  medium: "bg-amber-50 text-amber-700",
  low: "bg-rose-50 text-rose-700",
};

const CONFIDENCE_DOT: Record<string, string> = {
  high: "bg-emerald-500",
  medium: "bg-amber-500",
  low: "bg-rose-400",
};

interface LLMAnalysisTabProps {
  features: EnrichedFeature[];
}

export default function LLMAnalysisTab({ features }: LLMAnalysisTabProps) {
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [promptText, setPromptText] = useState("");
  const [promptLoading, setPromptLoading] = useState(false);
  const [useCustom, setUseCustom] = useState(false);
  const [relabelLoading, setRelabelLoading] = useState(false);
  const [relabelResult, setRelabelResult] = useState<RelabelResponse | null>(null);
  const [relabelError, setRelabelError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [successData, setSuccessData] = useState<FeatureSuccessData | null>(null);

  // Load success analysis data on mount
  useEffect(() => {
    fetchFeatureSuccess().then(setSuccessData);
  }, []);

  // Filter to features with code examples
  const labeledFeatures = features.filter(
    (f) => f.code_examples && f.code_examples.length > 0,
  );

  const filtered = labeledFeatures.filter((f) => {
    if (!search.trim()) return true;
    const q = search.toLowerCase();
    return (
      f.label.toLowerCase().includes(q) ||
      (f.llm_label ?? "").toLowerCase().includes(q) ||
      (f.description ?? "").toLowerCase().includes(q) ||
      String(f.id).includes(q)
    );
  });

  const selected = labeledFeatures.find((f) => f.id === selectedId) ?? null;

  // Load default prompt when feature is selected
  useEffect(() => {
    if (selectedId === null) return;
    setPromptLoading(true);
    setRelabelResult(null);
    setRelabelError(null);
    setSaved(false);
    setUseCustom(false);

    fetchDefaultPrompt(selectedId)
      .then((data) => {
        if (data) setPromptText(data.prompt);
      })
      .finally(() => setPromptLoading(false));
  }, [selectedId]);

  const handleRelabel = async () => {
    if (selectedId === null || relabelLoading) return;
    setRelabelLoading(true);
    setRelabelError(null);
    setRelabelResult(null);
    setSaved(false);

    try {
      const result = await relabelFeature(
        selectedId,
        useCustom ? promptText : undefined,
        false,
      );
      setRelabelResult(result);
    } catch (err) {
      setRelabelError(err instanceof Error ? err.message : "Relabeling failed");
    } finally {
      setRelabelLoading(false);
    }
  };

  const handleSave = async () => {
    if (selectedId === null || !relabelResult) return;
    try {
      await relabelFeature(
        selectedId,
        useCustom ? promptText : undefined,
        true,
      );
      setSaved(true);
    } catch (err) {
      setRelabelError(err instanceof Error ? err.message : "Save failed");
    }
  };

  return (
    <div className="flex h-full">
      {/* Left: Feature browser */}
      <div className="flex w-[280px] flex-shrink-0 flex-col border-r border-zinc-200">
        <div className="border-b border-zinc-200 p-3">
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search features..."
            className="w-full rounded border border-zinc-200 bg-white px-2.5 py-1.5 text-[12px] text-zinc-700 placeholder:text-zinc-400 focus:border-blue-400 focus:outline-none"
          />
          <div className="mt-1.5 text-[10px] text-zinc-400">
            {filtered.length} of {labeledFeatures.length} labeled features
          </div>
        </div>
        <div className="flex-1 overflow-y-auto">
          {filtered.map((f) => (
            <button
              key={f.id}
              onClick={() => setSelectedId(f.id)}
              className={`w-full text-left px-3 py-2.5 border-b border-zinc-100 transition-colors ${
                selectedId === f.id
                  ? "bg-blue-50 border-l-2 border-l-blue-500"
                  : "hover:bg-zinc-50 border-l-2 border-l-transparent"
              }`}
            >
              <div className="flex items-center gap-1.5">
                {f.confidence && (
                  <span
                    className={`h-1.5 w-1.5 rounded-full flex-shrink-0 ${CONFIDENCE_DOT[f.confidence]}`}
                  />
                )}
                <span className="font-mono text-[10px] text-zinc-400">
                  #{f.id}
                </span>
                {(f.success_verdict ?? successData?.features?.[String(f.id)]?.verdict) &&
                  !(f.success_verdict ?? successData?.features?.[String(f.id)]?.verdict)?.startsWith("[") && (
                  <span className={`ml-auto text-[9px] px-1.5 py-0.5 rounded-full font-medium ${
                    (f.success_verdict ?? successData?.features?.[String(f.id)]?.verdict) === "contributes"
                      ? "bg-emerald-50 text-emerald-600"
                      : (f.success_verdict ?? successData?.features?.[String(f.id)]?.verdict) === "hinders"
                        ? "bg-rose-50 text-rose-600"
                        : "bg-zinc-100 text-zinc-500"
                  }`}>
                    {f.success_verdict ?? successData?.features?.[String(f.id)]?.verdict}
                  </span>
                )}
              </div>
              <div className="text-[12px] font-medium text-zinc-700 mt-0.5 truncate">
                {f.llm_label || f.label}
              </div>
              {f.description && (
                <div className="text-[10px] text-zinc-400 mt-0.5 truncate">
                  {f.description}
                </div>
              )}
            </button>
          ))}
          {filtered.length === 0 && (
            <div className="p-4 text-center text-[11px] text-zinc-400">
              No matching features
            </div>
          )}
        </div>
      </div>

      {/* Right: Detail + relabeling */}
      <div className="flex-1 overflow-y-auto p-5">
        {!selected ? (
          <div className="flex h-full items-center justify-center">
            <p className="text-[13px] text-zinc-400">
              Select a feature to view details and re-label
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-5 max-w-2xl">
            {/* Header */}
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="font-mono text-[11px] text-zinc-400">
                  Feature #{selected.id}
                </span>
                {selected.confidence && (
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded-full ${CONFIDENCE_COLORS[selected.confidence]}`}
                  >
                    {selected.confidence}
                  </span>
                )}
              </div>
              <h3 className="text-[15px] font-semibold text-zinc-900">
                {selected.llm_label || selected.label}
              </h3>
              {selected.description && (
                <p className="mt-1 text-[12px] text-zinc-600">
                  {selected.description}
                </p>
              )}
            </div>

            {/* Code examples */}
            {selected.code_examples && selected.code_examples.length > 0 && (
              <div>
                <h4 className="text-[10px] font-medium uppercase tracking-wider text-zinc-400 mb-2">
                  Top activating examples ({selected.code_examples.length})
                </h4>
                <div className="flex flex-col gap-2">
                  {selected.code_examples.slice(0, 5).map((ex, i) => (
                    <div
                      key={i}
                      className="rounded border border-zinc-200 bg-white p-2.5 overflow-x-auto"
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-[10px] text-zinc-400">
                          {ex.task_id}
                        </span>
                        <span className="text-[10px] font-mono text-zinc-400">
                          activation: {ex.activation.toFixed(4)}
                        </span>
                      </div>
                      <HighlightedCode text={ex.code_context} />
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Success Analysis */}
            <SuccessAnalysisPanel feature={selected} successData={successData} />

            {/* Re-label section */}
            <div className="border-t border-zinc-200 pt-4">
              <h4 className="text-[10px] font-medium uppercase tracking-wider text-zinc-400 mb-3">
                Re-label with LLM
              </h4>

              {/* Toggle */}
              <label className="flex items-center gap-2 mb-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useCustom}
                  onChange={(e) => setUseCustom(e.target.checked)}
                  className="rounded border-zinc-300"
                />
                <span className="text-[11px] text-zinc-600">
                  Use custom prompt
                </span>
              </label>

              {/* Prompt textarea */}
              {useCustom && (
                <div className="mb-3">
                  {promptLoading ? (
                    <div className="h-48 rounded border border-zinc-200 bg-zinc-50 flex items-center justify-center text-[11px] text-zinc-400">
                      Loading prompt...
                    </div>
                  ) : (
                    <textarea
                      value={promptText}
                      onChange={(e) => setPromptText(e.target.value)}
                      rows={12}
                      className="w-full rounded border border-zinc-200 bg-white p-3 font-mono text-[11px] text-zinc-700 focus:border-blue-400 focus:outline-none resize-y"
                    />
                  )}
                  <button
                    onClick={() => {
                      if (selectedId !== null) {
                        setPromptLoading(true);
                        fetchDefaultPrompt(selectedId)
                          .then((data) => {
                            if (data) setPromptText(data.prompt);
                          })
                          .finally(() => setPromptLoading(false));
                      }
                    }}
                    className="mt-1 text-[10px] text-zinc-400 hover:text-zinc-600 transition-colors"
                  >
                    Reset to default
                  </button>
                </div>
              )}

              {/* Action buttons */}
              <button
                onClick={handleRelabel}
                disabled={relabelLoading}
                className="rounded bg-blue-500 px-4 py-1.5 text-[12px] font-medium text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {relabelLoading ? (
                  <span className="flex items-center gap-1.5">
                    <svg
                      className="h-3 w-3 animate-spin"
                      viewBox="0 0 24 24"
                      fill="none"
                    >
                      <circle
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                        className="opacity-25"
                      />
                      <path
                        d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                        fill="currentColor"
                        className="opacity-75"
                      />
                    </svg>
                    Labeling...
                  </span>
                ) : (
                  "Re-label"
                )}
              </button>

              {relabelError && (
                <div className="mt-3 rounded border border-rose-200 bg-rose-50 p-3 text-[11px] text-rose-700">
                  {relabelError}
                </div>
              )}

              {/* Result */}
              {relabelResult && (
                <div className="mt-4 rounded border border-zinc-200 bg-zinc-50 p-4">
                  <h4 className="text-[10px] font-medium uppercase tracking-wider text-zinc-400 mb-2">
                    New label
                  </h4>
                  <div className="flex flex-col gap-1.5">
                    <div className="flex items-center gap-2">
                      <span className="text-[13px] font-semibold text-zinc-900">
                        {relabelResult.label}
                      </span>
                      <span
                        className={`text-[10px] px-1.5 py-0.5 rounded-full ${
                          CONFIDENCE_COLORS[relabelResult.confidence] ??
                          "bg-zinc-100 text-zinc-600"
                        }`}
                      >
                        {relabelResult.confidence}
                      </span>
                    </div>
                    <p className="text-[12px] text-zinc-600">
                      {relabelResult.description}
                    </p>
                  </div>
                  <button
                    onClick={handleSave}
                    disabled={saved}
                    className={`mt-3 rounded px-3 py-1 text-[11px] font-medium transition-colors ${
                      saved
                        ? "bg-emerald-50 text-emerald-700 cursor-default"
                        : "bg-zinc-200 text-zinc-700 hover:bg-zinc-300"
                    }`}
                  >
                    {saved ? "Saved to session" : "Save to session"}
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
