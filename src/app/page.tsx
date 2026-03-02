"use client";

import { useState, useEffect, useMemo } from "react";
import dynamic from "next/dynamic";
import {
  EnrichedFeature,
  EnrichedGenerateResponse,
  FeatureMapPoint,
  FeatureOverride,
  ServerCapabilities,
  ServerInfo,
} from "@/lib/types";
import sampleGeneration from "@/lib/sampleGeneration.json";
import {
  fetchFeatures,
  fetchServerInfo,
  fetchFeatureMap,
  generateCompletion,
} from "@/lib/api";
import PromptInput from "@/components/PromptInput";
import FeaturePanel from "@/components/FeaturePanel";
import ResultsPanel from "@/components/ResultsPanel";
import ErrorBanner from "@/components/ErrorBanner";
import AlphaSweep from "@/components/AlphaSweep";
import ActivationMatrix from "@/components/ActivationMatrix";
import ActivationTimeline from "@/components/ActivationTimeline";
import ActivationDistribution from "@/components/ActivationDistribution";
import AnnotatedCode from "@/components/AnnotatedCode";
import LLMAnalysisTab from "@/components/LLMAnalysisTab";

const FeatureMap = dynamic(() => import("@/components/FeatureMap"), {
  ssr: false,
  loading: () => (
    <div className="flex h-64 items-center justify-center text-zinc-400 text-sm">
      Loading feature map...
    </div>
  ),
});

type Tab = "code" | "analysis" | "llm-analysis" | "features";

const DEFAULT_CAPS: ServerCapabilities = {
  token_activations: false,
  alpha_sweep: false,
  feature_map: false,
  enriched_features: false,
  density: false,
  llm_analysis: false,
};

const SWEEP_ALPHAS = [-2, -1, 0, 1, 2, 3];

/**
 * Extract code from model output — mirrors experiments/evaluation/extractor.py.
 * Strategy: 1) markdown code blocks, 2) bare function def, 3) full text fallback.
 */
function extractCode(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) return "";

  // Strategy 1: markdown code blocks ([\s\S] instead of dotall flag)
  const blockPattern = /```(?:python)?\s*\n([\s\S]*?)```/g;
  const blocks: string[] = [];
  let m;
  while ((m = blockPattern.exec(trimmed)) !== null) {
    blocks.push(m[1].trim());
  }
  if (blocks.length > 0) return blocks[0];

  // Strategy 2: bare function def
  const funcMatch = trimmed.match(/^(def \w+[\s\S]*)/m);
  if (funcMatch) return funcMatch[1].trim();

  // Strategy 3: full text fallback
  return trimmed;
}

export default function Home() {
  const [prompt, setPrompt] = useState("Write a Python fibonacci function. Only output code.");
  const [features, setFeatures] = useState<EnrichedFeature[]>([]);
  const [strengths, setStrengths] = useState<Record<number, number>>({});
  const [customStrengths, setCustomStrengths] = useState<Record<number, number>>({});
  const [temperature, setTemperature] = useState(0.3);

  const [serverInfo, setServerInfo] = useState<ServerInfo | null>(null);
  const [capabilities, setCapabilities] = useState<ServerCapabilities>(DEFAULT_CAPS);
  const [featureMapPoints, setFeatureMapPoints] = useState<FeatureMapPoint[] | null>(null);

  const [result, setResult] = useState<EnrichedGenerateResponse | null>(
    sampleGeneration as EnrichedGenerateResponse,
  );
  const [loading, setLoading] = useState(false);
  const [featuresLoading, setFeaturesLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [activeTab, setActiveTab] = useState<Tab>("code");
  const [showSweep, setShowSweep] = useState(false);
  const [selectedSweepIndex, setSelectedSweepIndex] = useState(0);

  const activeFeatureIds = useMemo(() => {
    const merged = { ...strengths, ...customStrengths };
    return Object.entries(merged)
      .filter(([, v]) => v !== 0)
      .map(([id]) => Number(id));
  }, [strengths, customStrengths]);

  const selectedFeatureIdSet = useMemo(
    () => new Set(activeFeatureIds),
    [activeFeatureIds],
  );

  const featureLabels = useMemo(() => {
    const labels: Record<number, string> = {};
    features.forEach((f) => { labels[f.id] = f.label; });
    Object.keys(customStrengths).forEach((id) => {
      const numId = Number(id);
      if (!(numId in labels)) labels[numId] = `#${id}`;
    });
    return labels;
  }, [features, customStrengths]);

  const modelName = serverInfo?.model?.split("/").pop() ?? "Model";

  useEffect(() => {
    fetchFeatures()
      .then((feats) => {
        setFeatures(feats);
        const initial: Record<number, number> = {};
        feats.forEach((f) => (initial[f.id] = f.slider?.default ?? 0));
        setStrengths(initial);
      })
      .catch((err) => {
        setError(`Cannot connect to backend: ${err.message}`);
      })
      .finally(() => setFeaturesLoading(false));

    fetchServerInfo().then((info) => {
      if (info) {
        setServerInfo(info);
        setCapabilities(info.capabilities ?? DEFAULT_CAPS);
      }
    });

    fetchFeatureMap().then((points) => {
      if (points) setFeatureMapPoints(points);
    });
  }, []);

  const handleGenerate = async () => {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setError(null);
    setSelectedSweepIndex(Math.max(0, SWEEP_ALPHAS.indexOf(1)));

    const merged = { ...strengths, ...customStrengths };
    const overrides: FeatureOverride[] = Object.entries(merged)
      .filter(([, strength]) => strength !== 0)
      .map(([id, strength]) => ({ id: Number(id), strength }));

    try {
      const res = await generateCompletion(prompt, overrides, temperature, {
        includeActivations: capabilities.token_activations,
        alphas: showSweep && capabilities.alpha_sweep ? SWEEP_ALPHAS : undefined,
      });
      res.baseline = extractCode(res.baseline);
      res.steered = extractCode(res.steered);
      if (res.sweep_results) {
        for (const sr of res.sweep_results) {
          sr.text = extractCode(sr.text);
        }
      }
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setLoading(false);
    }
  };

  const handleFeatureMapSelect = (id: number) => {
    if (!(id in customStrengths) && !features.some((f) => f.id === id)) {
      setCustomStrengths((prev) => ({ ...prev, [id]: 1.0 }));
    }
  };

  const tabs: { key: Tab; label: string; show: boolean }[] = [
    { key: "code", label: "Code", show: true },
    { key: "analysis", label: "Analysis", show: true },
    {
      key: "llm-analysis",
      label: "LLM Analysis",
      show: capabilities.llm_analysis || features.some((f) => f.code_examples && f.code_examples.length > 0),
    },
    {
      key: "features",
      label: "Feature Space",
      show: capabilities.feature_map || featureMapPoints !== null,
    },
  ];

  return (
    <div className="flex h-screen flex-col bg-white">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-zinc-200 px-6 py-3">
        <div className="flex items-center gap-3">
          <div className="h-6 w-6 rounded-md bg-gradient-to-br from-blue-500 to-violet-500" />
          <div>
            <h1 className="text-sm font-semibold tracking-tight text-zinc-900">
              Feature Steering
            </h1>
            <p className="text-[11px] text-zinc-400">
              {modelName} · SAE · Layer {serverInfo?.steer_layer ?? "\u2014"}
            </p>
          </div>
        </div>
        {serverInfo && (
          <div className="flex items-center gap-2">
            <span className="inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[11px] text-zinc-400">Connected</span>
          </div>
        )}
      </header>

      {error && (
        <div className="px-6 pt-3">
          <ErrorBanner message={error} onDismiss={() => setError(null)} />
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        {/* Left sidebar */}
        <aside className="flex w-80 flex-shrink-0 flex-col border-r border-zinc-200 bg-zinc-50/50">
          <div className="flex flex-col gap-5 overflow-y-auto p-4">
            <PromptInput
              value={prompt}
              onChange={setPrompt}
              onGenerate={handleGenerate}
              loading={loading}
            />

            {/* Temperature */}
            <div>
              <div className="mb-2 flex items-center justify-between">
                <label className="text-[11px] font-medium uppercase tracking-wider text-zinc-400">
                  Temperature
                </label>
                <span className="font-mono text-[11px] text-zinc-500">
                  {temperature.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div className="h-px bg-zinc-200" />

            {/* Features */}
            <FeaturePanel
              features={features}
              strengths={strengths}
              onStrengthChange={(id, v) => setStrengths((prev) => ({ ...prev, [id]: v }))}
              loading={featuresLoading}
              customStrengths={customStrengths}
              onCustomAdd={(id, strength) => setCustomStrengths((prev) => ({ ...prev, [id]: strength }))}
              onCustomRemove={(id) =>
                setCustomStrengths((prev) => {
                  const next = { ...prev };
                  delete next[id];
                  return next;
                })
              }
              onCustomChange={(id, strength) => setCustomStrengths((prev) => ({ ...prev, [id]: strength }))}
            />
          </div>
        </aside>

        {/* Main area */}
        <main className="flex flex-1 flex-col overflow-hidden bg-white">
          {/* Tab bar */}
          <div className="flex items-center gap-1 border-b border-zinc-200 px-4 pt-1">
            {tabs
              .filter((t) => t.show)
              .map((t) => (
                <button
                  key={t.key}
                  onClick={() => setActiveTab(t.key)}
                  className={`relative px-3 py-2 text-xs font-medium transition-colors ${
                    activeTab === t.key
                      ? "text-zinc-900"
                      : "text-zinc-400 hover:text-zinc-600"
                  }`}
                >
                  {t.label}
                  {activeTab === t.key && (
                    <span className="absolute bottom-0 left-0 right-0 h-0.5 rounded-full bg-blue-500" />
                  )}
                </button>
              ))}

            {/* Sweep toggle */}
            {activeTab === "code" && capabilities.alpha_sweep && (
              <div className="ml-auto flex items-center gap-3">
                <label className="flex items-center gap-1.5 cursor-pointer group">
                  <div
                    className={`relative h-4 w-7 rounded-full transition-colors ${
                      showSweep ? "bg-blue-500" : "bg-zinc-200"
                    }`}
                    onClick={() => setShowSweep(!showSweep)}
                  >
                    <div
                      className={`absolute top-0.5 h-3 w-3 rounded-full bg-white shadow transition-transform ${
                        showSweep ? "translate-x-3.5" : "translate-x-0.5"
                      }`}
                    />
                  </div>
                  <span className="text-[11px] text-zinc-400 group-hover:text-zinc-600">
                    Sweep
                  </span>
                </label>
              </div>
            )}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-y-auto p-5">
            {/* Loading state — centered in main area */}
            {loading && (
              <div className="flex h-full items-center justify-center">
                <div className="flex flex-col items-center gap-3">
                  <div className="h-6 w-6 animate-spin rounded-full border-2 border-zinc-200 border-t-blue-500" />
                  <p className="text-xs text-zinc-400">Generating...</p>
                </div>
              </div>
            )}

            {/* Code Tab */}
            {activeTab === "code" && !loading && (
              <div className="flex flex-col gap-4">
                {showSweep && result?.sweep_results && result.sweep_results.length > 0 ? (
                  <AlphaSweep
                    results={result.sweep_results}
                    baselineText={result.baseline}
                    selectedIndex={selectedSweepIndex}
                    onIndexChange={setSelectedSweepIndex}
                  />
                ) : (
                  <ResultsPanel
                    baseline={result ? result.baseline : null}
                    steered={result ? result.steered : null}
                    loading={false}
                  />
                )}
              </div>
            )}

            {/* Analysis Tab */}
            {activeTab === "analysis" && !loading && (
              <div className="flex flex-col gap-6">
                {result ? (
                  <>
                    {/* Full Layer Activation — top 32 most active SAE features */}
                    {result.layer_activations && result.layer_top_features && result.layer_top_features.length > 0 && (
                      <section className="rounded-lg border border-zinc-200 bg-white p-5">
                        <h3 className="mb-1 text-xs font-medium uppercase tracking-wider text-zinc-400">
                          Layer Activation
                        </h3>
                        <p className="mb-3 text-[11px] text-zinc-400">
                          Top {result.layer_top_features.length} most active SAE features across all generated tokens. Each column is a neuron.
                        </p>
                        <ActivationMatrix
                          tokens={result.layer_activations}
                          featureIds={result.layer_top_features.map((f) => f.id)}
                          featureLabels={featureLabels}
                        />
                      </section>
                    )}

                    {/* Generated code + distribution side by side */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {/* Generated code with activation overlay */}
                      <section className="rounded-lg border border-zinc-200 bg-white p-5">
                        <h3 className="mb-3 text-xs font-medium uppercase tracking-wider text-zinc-400">
                          Steered Output
                        </h3>
                        {result.token_activations && activeFeatureIds.length > 0 ? (
                          <AnnotatedCode
                            tokens={result.token_activations}
                            featureIds={activeFeatureIds}
                            featureLabels={featureLabels}
                          />
                        ) : (
                          <pre className="overflow-auto rounded-lg border border-zinc-100 bg-zinc-50 p-4 text-[13px] leading-relaxed font-mono text-zinc-700">
                            <code>{result.steered}</code>
                          </pre>
                        )}
                      </section>

                      {/* Distribution per steered feature */}
                      <section className="rounded-lg border border-zinc-200 bg-white p-5">
                        <h3 className="mb-3 text-xs font-medium uppercase tracking-wider text-zinc-400">
                          Activation Distribution
                        </h3>
                        {result.token_activations && activeFeatureIds.length > 0 ? (
                          <ActivationDistribution
                            tokens={result.token_activations}
                            featureIds={activeFeatureIds}
                            featureLabels={featureLabels}
                          />
                        ) : (
                          <p className="text-[11px] text-zinc-400">
                            {activeFeatureIds.length === 0
                              ? "Adjust feature sliders to see distribution"
                              : "No activation data available"}
                          </p>
                        )}
                      </section>
                    </div>

                    {/* Activation Timeline — steered features over token positions */}
                    {result.token_activations && activeFeatureIds.length > 0 && (
                      <section className="rounded-lg border border-zinc-200 bg-white p-5">
                        <h3 className="mb-1 text-xs font-medium uppercase tracking-wider text-zinc-400">
                          Activation Timeline
                        </h3>
                        <p className="mb-3 text-[11px] text-zinc-400">
                          How steered feature activations evolve across generated tokens.
                        </p>
                        <ActivationTimeline
                          tokens={result.token_activations}
                          featureIds={activeFeatureIds}
                          featureLabels={featureLabels}
                        />
                      </section>
                    )}
                  </>
                ) : (
                  <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-zinc-200 bg-zinc-50">
                    <div className="text-center">
                      <p className="text-sm text-zinc-400">
                        Generate code to see activation analysis
                      </p>
                      <p className="text-[11px] text-zinc-300 mt-1">
                        Activation data is captured automatically
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* LLM Analysis Tab */}
            {activeTab === "llm-analysis" && (
              <div className="h-full">
                <LLMAnalysisTab features={features} />
              </div>
            )}

            {/* Feature Space Tab */}
            {activeTab === "features" && (
              <div className="flex flex-col gap-4 h-full">
                <div className="flex items-baseline justify-between">
                  <h3 className="text-xs font-medium uppercase tracking-wider text-zinc-400">
                    Feature Space (UMAP)
                  </h3>
                  <p className="text-[11px] text-zinc-400">
                    Click a feature to add it as a slider
                  </p>
                </div>
                {featureMapPoints ? (
                  <div className="flex-1 min-h-[500px] rounded-lg border border-zinc-200 overflow-hidden">
                    <FeatureMap
                      points={featureMapPoints}
                      selectedIds={selectedFeatureIdSet}
                      onSelect={handleFeatureMapSelect}
                    />
                  </div>
                ) : (
                  <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-zinc-200 bg-zinc-50">
                    <p className="text-sm text-zinc-400">
                      Feature map not available
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
