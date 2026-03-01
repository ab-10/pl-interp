"use client";

import { useState, useEffect } from "react";
import { AnalyzeResponse, BackendInfo, Feature, FeatureOverride } from "@/lib/types";
import { analyzeFeature, fetchFeatures, fetchInfo, generateCompletion } from "@/lib/api";
import PromptInput from "@/components/PromptInput";
import FeaturePanel from "@/components/FeaturePanel";
import ResultsPanel from "@/components/ResultsPanel";
import ErrorBanner from "@/components/ErrorBanner";
import TokenHeatmapStrip from "@/components/TokenHeatmapStrip";
import SAEDecompositionBar from "@/components/SAEDecompositionBar";
import LayerAttributionChart from "@/components/LayerAttributionChart";

export default function Home() {
  const [prompt, setPrompt] = useState("Write a Python function that merges two sorted lists.");
  const [features, setFeatures] = useState<Feature[]>([]);
  const [activeFeatureId, setActiveFeatureId] = useState<number | null>(null);
  const [activeStrength, setActiveStrength] = useState(3);
  const [baseline, setBaseline] = useState<string | null>(null);
  const [steered, setSteered] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [customFeatureIds, setCustomFeatureIds] = useState<number[]>([]);
  const [featuresLoading, setFeaturesLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [temperature, setTemperature] = useState(0.3);
  const [info, setInfo] = useState<BackendInfo | null>(null);

  // Analyze state
  const [analyzeData, setAnalyzeData] = useState<AnalyzeResponse | null>(null);
  const [analyzeLoading, setAnalyzeLoading] = useState(false);
  const [selectedToken, setSelectedToken] = useState<number | null>(null);

  useEffect(() => {
    fetchInfo()
      .then(setInfo)
      .catch(() => {});
    fetchFeatures()
      .then((feats) => {
        setFeatures(feats);
      })
      .catch((err) => {
        setError(`Cannot connect to backend: ${err.message}`);
      })
      .finally(() => setFeaturesLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const getActiveOverrides = (): FeatureOverride[] => {
    if (activeFeatureId !== null && activeStrength !== 0) {
      return [{ id: activeFeatureId, strength: activeStrength }];
    }
    return [];
  };

  const handleGenerate = async () => {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setError(null);
    setSelectedToken(null);
    setAnalyzeData(null);

    const overrides = getActiveOverrides();

    // Run analysis in parallel if a feature is selected
    if (activeFeatureId !== null) {
      setAnalyzeLoading(true);
      analyzeFeature(prompt, activeFeatureId, overrides)
        .then(setAnalyzeData)
        .catch(() => {})
        .finally(() => setAnalyzeLoading(false));
    }

    try {
      const result = await generateCompletion(prompt, overrides, temperature);
      setBaseline(result.baseline);
      setSteered(result.steered);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setLoading(false);
    }
  };

  const tokenDetail =
    analyzeData && selectedToken !== null
      ? analyzeData.token_details[String(selectedToken)]
      : null;

  return (
    <div className="flex min-h-screen flex-col bg-white">
      <header className="border-b border-zinc-200 px-6 py-4">
        <h1 className="text-xl font-semibold text-zinc-900">
          Feature Steering
        </h1>
        <p className="text-sm text-zinc-500">
          Steer Ministral 8B code generation with SAE feature sliders
        </p>
        {info && (
          <div className="mt-1 flex gap-4 text-xs font-mono text-zinc-400">
            <span>Model: {info.model}</span>
            <span>SAE: {info.sae}{info.layer != null ? ` (layer ${info.layer})` : ""}</span>
          </div>
        )}
      </header>

      {error && <div className="px-6 pt-4"><ErrorBanner message={error} onDismiss={() => setError(null)} /></div>}

      <div className="flex flex-1 gap-0">
        {/* Left sidebar: prompt + features */}
        <aside className="relative z-10 flex w-80 flex-shrink-0 flex-col gap-6 overflow-y-auto border-r border-zinc-200 bg-zinc-50 p-6">
          <PromptInput
            value={prompt}
            onChange={setPrompt}
            onGenerate={handleGenerate}
            loading={loading}
          />
          <div>
            <label className="mb-2 block text-sm font-medium text-zinc-700">
              Temperature: {temperature.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full accent-orange-500"
            />
            <div className="mt-1 flex justify-between text-xs text-zinc-500">
              <span>0 (greedy)</span>
              <span>1.0</span>
            </div>
          </div>
          <FeaturePanel
            features={features}
            activeFeatureId={activeFeatureId}
            strength={activeStrength}
            onToggle={(id) => {
              setActiveFeatureId((prev) => (prev === id ? null : id));
              setActiveStrength(3);
            }}
            onStrengthChange={setActiveStrength}
            loading={featuresLoading}
            customFeatureIds={customFeatureIds}
            onAddCustom={(id) => {
              setCustomFeatureIds((prev) =>
                prev.includes(id) ? prev : [...prev, id]
              );
              setActiveFeatureId(id);
              setActiveStrength(3);
            }}
            onRemoveCustom={(id) => {
              setCustomFeatureIds((prev) => prev.filter((fid) => fid !== id));
              if (activeFeatureId === id) {
                setActiveFeatureId(null);
              }
            }}
          />
        </aside>

        {/* Main area: results (left) + analysis (right) */}
        <main className="grid flex-1 grid-cols-2 gap-0">
          {/* Left column: Diff output */}
          <div className="overflow-y-auto border-r border-zinc-200 p-6">
            <ResultsPanel baseline={baseline} steered={steered} loading={loading} />
          </div>

          {/* Right column: Feature Activation Analysis */}
          <div className="overflow-y-auto p-6">
            <h2 className="text-lg font-semibold text-zinc-900 mb-4">
              Feature Activation Analysis
            </h2>

            {!analyzeData && !analyzeLoading && (
              <p className="text-sm text-zinc-500">
                Select a feature and generate to see activation analysis.
              </p>
            )}

            {analyzeLoading && (
              <div className="flex items-center gap-2 text-sm text-zinc-500">
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-zinc-400 border-t-transparent" />
                Running analysis (generating + forward pass + SAE encode)...
              </div>
            )}

            {analyzeData && !analyzeLoading && (
              <div data-testid="detail-panel" data-feature-activation={
                selectedToken !== null ? analyzeData.feature_activations[selectedToken] : 0
              }>
                {/* Token Heatmap Strip */}
                <TokenHeatmapStrip
                  tokens={analyzeData.tokens}
                  activations={analyzeData.feature_activations}
                  featureId={activeFeatureId!}
                  selectedIndex={selectedToken}
                  onTokenClick={setSelectedToken}
                />

                {/* Detail panel for selected token */}
                {tokenDetail && selectedToken !== null && (
                  <div className="mt-6 space-y-6">
                    <SAEDecompositionBar
                      token={analyzeData.tokens[selectedToken]}
                      decomposition={tokenDetail.sae_decomposition}
                      reconstructionError={tokenDetail.reconstruction_error}
                      highlightedFeatureId={activeFeatureId!}
                      featureActivation={analyzeData.feature_activations[selectedToken]}
                    />
                    <LayerAttributionChart
                      attribution={tokenDetail.layer_attribution}
                      featureId={activeFeatureId!}
                      featureActivation={analyzeData.feature_activations[selectedToken]}
                      token={analyzeData.tokens[selectedToken]}
                    />
                  </div>
                )}

                {selectedToken === null && (
                  <p className="mt-4 text-sm text-zinc-500">
                    Click a token above to see its SAE decomposition and layer attribution.
                  </p>
                )}
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
