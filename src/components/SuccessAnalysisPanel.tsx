"use client";

import { useState } from "react";
import { EnrichedFeature, AnalyzeSuccessResponse, FeatureSuccessData } from "@/lib/types";
import { analyzeFeatureSuccess } from "@/lib/api";
import HighlightedCode from "./HighlightedCode";

const VERDICT_STYLE: Record<string, { bg: string; text: string; dot: string }> = {
  contributes: { bg: "bg-emerald-50", text: "text-emerald-700", dot: "bg-emerald-500" },
  neutral: { bg: "bg-zinc-100", text: "text-zinc-600", dot: "bg-zinc-400" },
  hinders: { bg: "bg-rose-50", text: "text-rose-700", dot: "bg-rose-500" },
};

const CONFIDENCE_COLORS: Record<string, string> = {
  high: "text-emerald-600",
  medium: "text-amber-600",
  low: "text-rose-500",
};

function VerdictBadge({ verdict }: { verdict: string }) {
  const style = VERDICT_STYLE[verdict] ?? VERDICT_STYLE.neutral;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium ${style.bg} ${style.text}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${style.dot}`} />
      {verdict}
    </span>
  );
}

function StatBar({ label, passVal, failVal }: { label: string; passVal: number; failVal: number }) {
  const max = Math.max(passVal, failVal, 0.001);
  return (
    <div className="flex items-center gap-2 text-[10px]">
      <span className="w-20 text-zinc-400 text-right">{label}</span>
      <div className="flex-1 flex items-center gap-1">
        <div className="flex-1 h-2 bg-zinc-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-emerald-400 rounded-full"
            style={{ width: `${(passVal / max) * 100}%` }}
          />
        </div>
        <span className="w-14 text-right font-mono text-emerald-600">
          {typeof passVal === "number" && passVal < 1 ? (passVal * 100).toFixed(1) + "%" : passVal.toFixed(4)}
        </span>
      </div>
      <div className="flex-1 flex items-center gap-1">
        <div className="flex-1 h-2 bg-zinc-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-rose-400 rounded-full"
            style={{ width: `${(failVal / max) * 100}%` }}
          />
        </div>
        <span className="w-14 text-right font-mono text-rose-500">
          {typeof failVal === "number" && failVal < 1 ? (failVal * 100).toFixed(1) + "%" : failVal.toFixed(4)}
        </span>
      </div>
    </div>
  );
}

interface SuccessAnalysisPanelProps {
  feature: EnrichedFeature;
  successData: FeatureSuccessData | null;
}

export default function SuccessAnalysisPanel({ feature, successData }: SuccessAnalysisPanelProps) {
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalyzeSuccessResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showPassExamples, setShowPassExamples] = useState(false);
  const [showFailExamples, setShowFailExamples] = useState(false);

  const fid = String(feature.id);
  const entry = successData?.features?.[fid];

  const verdict = result?.verdict ?? feature.success_verdict ?? entry?.verdict;
  const mechanism = result?.mechanism ?? feature.success_mechanism ?? entry?.mechanism;
  const confidence = result?.confidence ?? feature.success_confidence ?? entry?.llm_confidence;

  const stats = feature.success_stats ?? (entry ? {
    cohens_d: entry.cohens_d,
    mean_pass: entry.mean_pass,
    mean_fail: entry.mean_fail,
    fire_rate_pass: entry.fire_rate_pass,
    fire_rate_fail: entry.fire_rate_fail,
  } : null);

  const passExamples = entry?.pass_examples ?? [];
  const failExamples = entry?.fail_examples ?? [];

  const handleAnalyze = async () => {
    setAnalyzing(true);
    setError(null);
    setResult(null);
    try {
      const res = await analyzeFeatureSuccess(feature.id);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setAnalyzing(false);
    }
  };

  if (!entry && !feature.success_verdict) {
    return (
      <div className="rounded-lg border border-dashed border-zinc-200 bg-zinc-50 p-4">
        <p className="text-[11px] text-zinc-400 mb-2">
          No success analysis available for this feature.
        </p>
        <p className="text-[10px] text-zinc-300">
          Run analyze_success on the VM to generate analysis data.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Verdict + mechanism */}
      <div className="rounded-lg border border-zinc-200 bg-white p-4">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-[10px] font-medium uppercase tracking-wider text-zinc-400">
            Success Analysis
          </h4>
          {verdict && !verdict.startsWith("[") && <VerdictBadge verdict={verdict} />}
        </div>

        {mechanism && !mechanism.startsWith("[") ? (
          <p className="text-[12px] text-zinc-700 leading-relaxed">
            {mechanism}
          </p>
        ) : (
          <p className="text-[11px] text-zinc-400 italic">
            {verdict?.startsWith("[") ? "Analysis pending — run LLM analysis" : "No mechanism available"}
          </p>
        )}

        {confidence && !confidence.startsWith("[") && (
          <div className="mt-2 flex items-center gap-1.5">
            <span className="text-[10px] text-zinc-400">Confidence:</span>
            <span className={`text-[10px] font-medium ${CONFIDENCE_COLORS[confidence] ?? "text-zinc-500"}`}>
              {confidence}
            </span>
          </div>
        )}
      </div>

      {/* Statistical signal */}
      {stats && (
        <div className="rounded-lg border border-zinc-200 bg-white p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-[10px] font-medium uppercase tracking-wider text-zinc-400">
              Statistical Signal
            </h4>
            <span className="font-mono text-[10px] text-zinc-400">
              Cohen&apos;s d = {(stats.cohens_d ?? 0) >= 0 ? "+" : ""}{(stats.cohens_d ?? 0).toFixed(4)}
            </span>
          </div>

          {/* Legend */}
          <div className="flex items-center gap-4 mb-2 text-[10px]">
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-emerald-400" /> Pass
            </span>
            <span className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-rose-400" /> Fail
            </span>
          </div>

          <div className="flex flex-col gap-1.5">
            <StatBar
              label="Mean act."
              passVal={stats.mean_pass ?? 0}
              failVal={stats.mean_fail ?? 0}
            />
            <StatBar
              label="Fire rate"
              passVal={stats.fire_rate_pass ?? 0}
              failVal={stats.fire_rate_fail ?? 0}
            />
          </div>
        </div>
      )}

      {/* Code examples (collapsible) */}
      {(passExamples.length > 0 || failExamples.length > 0) && (
        <div className="rounded-lg border border-zinc-200 bg-white p-4">
          <h4 className="text-[10px] font-medium uppercase tracking-wider text-zinc-400 mb-3">
            Code Examples (Pass vs Fail)
          </h4>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            {/* Pass examples */}
            <div>
              <button
                onClick={() => setShowPassExamples(!showPassExamples)}
                className="flex items-center gap-1.5 text-[11px] font-medium text-emerald-700 mb-2 hover:text-emerald-800"
              >
                <svg
                  className={`h-3 w-3 transition-transform ${showPassExamples ? "rotate-90" : ""}`}
                  fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                </svg>
                Passing code ({passExamples.length})
              </button>
              {showPassExamples && passExamples.map((ex, i) => (
                <div key={i} className="rounded border border-emerald-200 bg-emerald-50/50 p-2 mb-1.5 overflow-x-auto">
                  <div className="flex justify-between mb-1">
                    <span className="text-[9px] text-emerald-600">{ex.task_id}</span>
                    <span className="text-[9px] font-mono text-emerald-500">{ex.activation.toFixed(4)}</span>
                  </div>
                  <HighlightedCode text={ex.code_context} />
                </div>
              ))}
            </div>

            {/* Fail examples */}
            <div>
              <button
                onClick={() => setShowFailExamples(!showFailExamples)}
                className="flex items-center gap-1.5 text-[11px] font-medium text-rose-700 mb-2 hover:text-rose-800"
              >
                <svg
                  className={`h-3 w-3 transition-transform ${showFailExamples ? "rotate-90" : ""}`}
                  fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                </svg>
                Failing code ({failExamples.length})
              </button>
              {showFailExamples && failExamples.map((ex, i) => (
                <div key={i} className="rounded border border-rose-200 bg-rose-50/50 p-2 mb-1.5 overflow-x-auto">
                  <div className="flex justify-between mb-1">
                    <span className="text-[9px] text-rose-600">{ex.task_id}</span>
                    <span className="text-[9px] font-mono text-rose-500">{ex.activation.toFixed(4)}</span>
                  </div>
                  <HighlightedCode text={ex.code_context} />
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Re-analyze button */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleAnalyze}
          disabled={analyzing}
          className="rounded bg-zinc-800 px-3 py-1.5 text-[11px] font-medium text-white hover:bg-zinc-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {analyzing ? (
            <span className="flex items-center gap-1.5">
              <svg className="h-3 w-3 animate-spin" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25" />
                <path d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" fill="currentColor" className="opacity-75" />
              </svg>
              Analyzing...
            </span>
          ) : (
            "Re-analyze with LLM"
          )}
        </button>
        {error && (
          <span className="text-[10px] text-rose-600">{error}</span>
        )}
      </div>
    </div>
  );
}
