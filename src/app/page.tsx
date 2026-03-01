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
import TokenHeatmap from "@/components/TokenHeatmap";
import AlphaSweep from "@/components/AlphaSweep";
import DensityRadar from "@/components/DensityRadar";

const FeatureMap = dynamic(() => import("@/components/FeatureMap"), {
  ssr: false,
  loading: () => (
    <div className="flex h-64 items-center justify-center text-zinc-500 text-sm">
      Loading feature map...
    </div>
  ),
});

type Tab = "code" | "analysis" | "features";

const DEFAULT_CAPS: ServerCapabilities = {
  token_activations: false,
  alpha_sweep: false,
  feature_map: false,
  enriched_features: false,
  density: false,
};

const SWEEP_ALPHAS = [-2, -1, 0, 1, 2, 3];

export default function Home() {
  const [prompt, setPrompt] = useState("def fibonacci(n):");
  const [features, setFeatures] = useState<EnrichedFeature[]>([]);
  const [strengths, setStrengths] = useState<Record<number, number>>({});
  const [customStrengths, setCustomStrengths] = useState<Record<number, number>>({});
  const [temperature, setTemperature] = useState(0.3);

  const [serverInfo, setServerInfo] = useState<ServerInfo | null>(null);
  const [capabilities, setCapabilities] = useState<ServerCapabilities>(DEFAULT_CAPS);
  const [featureMapPoints, setFeatureMapPoints] = useState<FeatureMapPoint[] | null>(null);

  const [result, setResult] = useState<EnrichedGenerateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [featuresLoading, setFeaturesLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [activeTab, setActiveTab] = useState<Tab>("code");
  const [showHeatmap, setShowHeatmap] = useState(false);
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
        includeActivations: showHeatmap && capabilities.token_activations,
        alphas: showSweep && capabilities.alpha_sweep ? SWEEP_ALPHAS : undefined,
      });
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
      key: "features",
      label: "Feature Space",
      show: capabilities.feature_map || featureMapPoints !== null,
    },
  ];

  return (
    <div className="flex h-screen flex-col bg-zinc-950">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-zinc-800/50 px-6 py-3">
        <div className="flex items-center gap-3">
          <div className="h-6 w-6 rounded-md bg-gradient-to-br from-blue-500 to-violet-500" />
          <div>
            <h1 className="text-sm font-semibold tracking-tight text-zinc-100">
              Feature Steering
            </h1>
            <p className="text-[11px] text-zinc-500">
              {modelName} · SAE · Layer {serverInfo?.steer_layer ?? "—"}
            </p>
          </div>
        </div>
        {serverInfo && (
          <div className="flex items-center gap-2">
            <span className="inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[11px] text-zinc-500">Connected</span>
          </div>
        )}
      </header>

      {error && (
        <div className="px-6 pt-3">
          <ErrorBanner message={error} onDismiss={() => setError(null)} />
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        {/* ── Left sidebar ─────────────────────────────────── */}
        <aside className="flex w-72 flex-shrink-0 flex-col border-r border-zinc-800/50 bg-zinc-950">
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
                <label className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                  Temperature
                </label>
                <span className="font-mono text-[11px] text-zinc-400">
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

            <div className="h-px bg-zinc-800/50" />

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

        {/* ── Main area ────────────────────────────────────── */}
        <main className="flex flex-1 flex-col overflow-hidden bg-zinc-900/30">
          {/* Tab bar */}
          <div className="flex items-center gap-1 border-b border-zinc-800/50 px-4 pt-1">
            {tabs
              .filter((t) => t.show)
              .map((t) => (
                <button
                  key={t.key}
                  onClick={() => setActiveTab(t.key)}
                  className={`relative px-3 py-2 text-xs font-medium transition-colors ${
                    activeTab === t.key
                      ? "text-zinc-100"
                      : "text-zinc-500 hover:text-zinc-300"
                  }`}
                >
                  {t.label}
                  {activeTab === t.key && (
                    <span className="absolute bottom-0 left-0 right-0 h-0.5 rounded-full bg-blue-500" />
                  )}
                </button>
              ))}

            {/* Visualization toggles — right side of tab bar */}
            {activeTab === "code" && (capabilities.token_activations || capabilities.alpha_sweep) && (
              <div className="ml-auto flex items-center gap-3">
                {capabilities.token_activations && (
                  <label className="flex items-center gap-1.5 cursor-pointer group">
                    <div
                      className={`relative h-4 w-7 rounded-full transition-colors ${
                        showHeatmap ? "bg-blue-600" : "bg-zinc-700"
                      }`}
                      onClick={() => setShowHeatmap(!showHeatmap)}
                    >
                      <div
                        className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-transform ${
                          showHeatmap ? "translate-x-3.5" : "translate-x-0.5"
                        }`}
                      />
                    </div>
                    <span className="text-[11px] text-zinc-500 group-hover:text-zinc-400">
                      Heatmap
                    </span>
                  </label>
                )}
                {capabilities.alpha_sweep && (
                  <label className="flex items-center gap-1.5 cursor-pointer group">
                    <div
                      className={`relative h-4 w-7 rounded-full transition-colors ${
                        showSweep ? "bg-blue-600" : "bg-zinc-700"
                      }`}
                      onClick={() => setShowSweep(!showSweep)}
                    >
                      <div
                        className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-transform ${
                          showSweep ? "translate-x-3.5" : "translate-x-0.5"
                        }`}
                      />
                    </div>
                    <span className="text-[11px] text-zinc-500 group-hover:text-zinc-400">
                      Sweep
                    </span>
                  </label>
                )}
              </div>
            )}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-y-auto p-5">
            {/* ── Code Tab ──────────────────────────────── */}
            {activeTab === "code" && (
              <div className="flex flex-col gap-4">
                {showSweep && result?.sweep_results && result.sweep_results.length > 0 ? (
                  <AlphaSweep
                    results={result.sweep_results}
                    selectedIndex={selectedSweepIndex}
                    onIndexChange={setSelectedSweepIndex}
                    activeFeatureIds={activeFeatureIds}
                    showHeatmap={showHeatmap}
                  />
                ) : showHeatmap && result?.token_activations && result.token_activations.length > 0 ? (
                  <TokenHeatmap
                    tokens={result.token_activations}
                    activeFeatureIds={activeFeatureIds}
                  />
                ) : (
                  <ResultsPanel
                    baseline={result?.baseline ?? null}
                    steered={result?.steered ?? null}
                    loading={loading}
                  />
                )}

                {(showHeatmap || showSweep) && result && (
                  <div className="border-t border-zinc-800/50 pt-4">
                    <ResultsPanel
                      baseline={result.baseline}
                      steered={result.steered}
                      loading={false}
                    />
                  </div>
                )}
              </div>
            )}

            {/* ── Analysis Tab ──────────────────────────── */}
            {activeTab === "analysis" && (
              <div className="flex flex-col gap-6">
                {result?.baseline_density && result?.steered_density ? (
                  <div className="rounded-lg border border-zinc-800/50 bg-zinc-900/50 p-5">
                    <h3 className="mb-4 text-xs font-medium uppercase tracking-wider text-zinc-500">
                      Property Density
                    </h3>
                    <DensityRadar
                      baselineDensity={result.baseline_density}
                      steeredDensity={result.steered_density}
                    />
                  </div>
                ) : (
                  <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-zinc-800 bg-zinc-900/20">
                    <p className="text-sm text-zinc-600">
                      Generate code to see property density analysis
                    </p>
                  </div>
                )}

                {result && (
                  <div className="rounded-lg border border-zinc-800/50 bg-zinc-900/50 p-5">
                    <h3 className="mb-4 text-xs font-medium uppercase tracking-wider text-zinc-500">
                      Code Diff
                    </h3>
                    <ResultsPanel
                      baseline={result.baseline}
                      steered={result.steered}
                      loading={false}
                    />
                  </div>
                )}
              </div>
            )}

            {/* ── Feature Space Tab ─────────────────────── */}
            {activeTab === "features" && (
              <div className="flex flex-col gap-4 h-full">
                <div className="flex items-baseline justify-between">
                  <h3 className="text-xs font-medium uppercase tracking-wider text-zinc-500">
                    Feature Space (UMAP)
                  </h3>
                  <p className="text-[11px] text-zinc-600">
                    Click a feature to add it as a slider
                  </p>
                </div>
                {featureMapPoints ? (
                  <div className="flex-1 min-h-[500px] rounded-lg border border-zinc-800/50 overflow-hidden">
                    <FeatureMap
                      points={featureMapPoints}
                      selectedIds={selectedFeatureIdSet}
                      onSelect={handleFeatureMapSelect}
                    />
                  </div>
                ) : (
                  <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-zinc-800 bg-zinc-900/20">
                    <p className="text-sm text-zinc-600">
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
