"use client";

import { useState, useEffect } from "react";
import { AnalyzeResponse, BackendInfo, Feature, FeatureOverride } from "@/lib/types";
import { analyzeFeature, fetchFeatures, fetchInfo, generateCompletion } from "@/lib/api";
import PromptInput from "@/components/PromptInput";
import FeaturePanel from "@/components/FeaturePanel";
import ResultsPanel from "@/components/ResultsPanel";
import ErrorBanner from "@/components/ErrorBanner";
import FeatureSelector from "@/components/FeatureSelector";
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
  const [selectedFeatureId, setSelectedFeatureId] = useState<number | null>(null);

  useEffect(() => {
    fetchInfo()
      .then(setInfo)
      .catch(() => {});
    fetchFeatures()
      .then((feats) => {
        setFeatures(feats);
        // Default to first feature for analysis
        if (feats.length > 0 && selectedFeatureId === null) {
          setSelectedFeatureId(feats[0].id);
        }
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

    const overrides = getActiveOverrides();

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

  const handleAnalyze = async (featureId?: number) => {
    const fId = featureId ?? selectedFeatureId;
    if (!prompt.trim() || analyzeLoading || fId === null) return;
    setAnalyzeLoading(true);
    setError(null);
    setSelectedToken(null);

    const overrides = getActiveOverrides();

    try {
      const result = await analyzeFeature(prompt, fId, overrides);
      setAnalyzeData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setAnalyzeLoading(false);
    }
  };

  const handleFeatureChange = (featureId: number) => {
    setSelectedFeatureId(featureId);
    // Re-analyze with new feature
    handleAnalyze(featureId);
  };

  const tokenDetail =
    analyzeData && selectedToken !== null
      ? analyzeData.token_details[String(selectedToken)]
      : null;

  return (
    <div className="flex min-h-screen flex-col bg-zinc-50 dark:bg-zinc-950">
      <header className="border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
        <h1 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
          Feature Steering
        </h1>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Steer Ministral 8B code generation with SAE feature sliders
        </p>
        {info && (
          <div className="mt-1 flex gap-4 text-xs font-mono text-zinc-400 dark:text-zinc-500">
            <span>Model: {info.model}</span>
            <span>SAE: {info.sae}{info.layer != null ? ` (layer ${info.layer})` : ""}</span>
          </div>
        )}
      </header>

      {error && <div className="px-6 pt-4"><ErrorBanner message={error} onDismiss={() => setError(null)} /></div>}

      <div className="flex flex-1 gap-0">
        {/* Left sidebar: prompt + features */}
        <aside className="relative z-10 flex w-80 flex-shrink-0 flex-col gap-6 overflow-y-auto border-r border-zinc-200 p-6 dark:border-zinc-800">
          <PromptInput
            value={prompt}
            onChange={setPrompt}
            onGenerate={handleGenerate}
            loading={loading}
          />
          <div>
            <label className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Temperature: {temperature.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full accent-blue-600"
            />
            <div className="mt-1 flex justify-between text-xs text-zinc-400">
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

        {/* Main area: results + analysis */}
        <main className="flex-1 overflow-y-auto p-6">
          <ResultsPanel baseline={baseline} steered={steered} loading={loading} />

          {/* Feature Activation Analysis */}
          <div className="mt-8 border-t border-zinc-200 pt-6 dark:border-zinc-800">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
                Feature Activation Analysis
              </h2>
              <div className="flex items-center gap-4">
                {features.length > 0 && selectedFeatureId !== null && (
                  <FeatureSelector
                    features={features}
                    selectedFeatureId={selectedFeatureId}
                    onChange={handleFeatureChange}
                  />
                )}
                <button
                  onClick={() => handleAnalyze()}
                  disabled={analyzeLoading || !prompt.trim() || selectedFeatureId === null}
                  className="rounded bg-emerald-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {analyzeLoading ? "Analyzing..." : "Analyze"}
                </button>
              </div>
            </div>

            {!analyzeData && !analyzeLoading && (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Select a feature and click Analyze to visualize feature activations on generated text.
              </p>
            )}

            {analyzeLoading && (
              <div className="flex items-center gap-2 text-sm text-zinc-400">
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
                  featureId={selectedFeatureId!}
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
                      highlightedFeatureId={selectedFeatureId!}
                      featureActivation={analyzeData.feature_activations[selectedToken]}
                    />
                    <LayerAttributionChart
                      attribution={tokenDetail.layer_attribution}
                      featureId={selectedFeatureId!}
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
