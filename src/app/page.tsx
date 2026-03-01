"use client";

import { useState, useEffect } from "react";
import { BackendInfo, Feature, FeatureOverride } from "@/lib/types";
import { fetchFeatures, fetchInfo, generateCompletion } from "@/lib/api";
import PromptInput from "@/components/PromptInput";
import FeaturePanel from "@/components/FeaturePanel";
import ResultsPanel from "@/components/ResultsPanel";
import ErrorBanner from "@/components/ErrorBanner";

/** Composite key for per-feature strength state: "layer:id" */
function featureKey(layer: number, id: number): string {
  return `${layer}:${id}`;
}

export default function Home() {
  const [prompt, setPrompt] = useState("def fibonacci(n):");
  const [features, setFeatures] = useState<Feature[]>([]);
  const [strengths, setStrengths] = useState<Record<string, number>>({});
  const [baseline, setBaseline] = useState<string | null>(null);
  const [steered, setSteered] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [customFeatures, setCustomFeatures] = useState<
    { id: number; layer: number; strength: number }[]
  >([]);
  const [featuresLoading, setFeaturesLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [temperature, setTemperature] = useState(0.3);
  const [info, setInfo] = useState<BackendInfo | null>(null);

  useEffect(() => {
    fetchInfo()
      .then(setInfo)
      .catch(() => {});
    fetchFeatures()
      .then((feats) => {
        setFeatures(feats);
        const initial: Record<string, number> = {};
        feats.forEach((f) => (initial[featureKey(f.layer, f.id)] = 0));
        setStrengths(initial);
      })
      .catch((err) => {
        setError(`Cannot connect to backend: ${err.message}`);
      })
      .finally(() => setFeaturesLoading(false));
  }, []);

  const handleGenerate = async () => {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setError(null);

    const overrides: FeatureOverride[] = [];

    // Registry features
    for (const feat of features) {
      const key = featureKey(feat.layer, feat.id);
      const s = strengths[key] ?? 0;
      if (s !== 0) {
        overrides.push({ id: feat.id, layer: feat.layer, strength: s });
      }
    }

    // Custom features
    for (const cf of customFeatures) {
      if (cf.strength !== 0) {
        overrides.push({ id: cf.id, layer: cf.layer, strength: cf.strength });
      }
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

  return (
    <div className="flex min-h-screen flex-col bg-zinc-50 dark:bg-zinc-950">
      <header className="border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
        <h1 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
          Feature Steering
        </h1>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Steer Ministral 8B code generation with dual-layer SAE feature sliders
        </p>
        {info && (
          <div className="mt-1 flex gap-4 text-xs font-mono text-zinc-400 dark:text-zinc-500">
            <span>Model: {info.model}</span>
            {info.saes &&
              Object.entries(info.saes).map(([layer, path]) => (
                <span key={layer}>SAE L{layer}: {path}</span>
              ))}
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
            strengths={strengths}
            onStrengthChange={(layer, id, v) =>
              setStrengths((prev) => ({ ...prev, [featureKey(layer, id)]: v }))
            }
            loading={featuresLoading}
            customFeatures={customFeatures}
            onCustomAdd={(id, layer, strength) =>
              setCustomFeatures((prev) => [...prev, { id, layer, strength }])
            }
            onCustomRemove={(idx) =>
              setCustomFeatures((prev) => prev.filter((_, i) => i !== idx))
            }
            onCustomChange={(idx, strength) =>
              setCustomFeatures((prev) =>
                prev.map((cf, i) => (i === idx ? { ...cf, strength } : cf))
              )
            }
          />
        </aside>

        {/* Main area: results */}
        <main className="flex-1 p-6">
          <ResultsPanel baseline={baseline} steered={steered} loading={loading} />
        </main>
      </div>
    </div>
  );
}
