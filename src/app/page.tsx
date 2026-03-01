"use client";

import { useState, useEffect } from "react";
import { Feature, FeatureOverride } from "@/lib/types";
import { fetchFeatures, generateCompletion } from "@/lib/api";
import PromptInput from "@/components/PromptInput";
import FeaturePanel from "@/components/FeaturePanel";
import ResultsPanel from "@/components/ResultsPanel";
import ErrorBanner from "@/components/ErrorBanner";

export default function Home() {
  const [prompt, setPrompt] = useState("def fibonacci(n):");
  const [features, setFeatures] = useState<Feature[]>([]);
  const [strengths, setStrengths] = useState<Record<number, number>>({});
  const [baseline, setBaseline] = useState<string | null>(null);
  const [steered, setSteered] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [customStrengths, setCustomStrengths] = useState<Record<number, number>>({});
  const [featuresLoading, setFeaturesLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [temperature, setTemperature] = useState(0.3);

  useEffect(() => {
    fetchFeatures()
      .then((feats) => {
        setFeatures(feats);
        const initial: Record<number, number> = {};
        feats.forEach((f) => (initial[f.id] = 0));
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

    const merged = { ...strengths, ...customStrengths };
    const overrides: FeatureOverride[] = Object.entries(merged)
      .filter(([, strength]) => strength !== 0)
      .map(([id, strength]) => ({ id: Number(id), strength }));

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
          Steer Mistral 7B code generation with SAE feature sliders
        </p>
      </header>

      {error && <div className="px-6 pt-4"><ErrorBanner message={error} onDismiss={() => setError(null)} /></div>}

      <div className="flex flex-1 gap-0">
        {/* Left sidebar: prompt + features */}
        <aside className="flex w-80 flex-shrink-0 flex-col gap-6 border-r border-zinc-200 p-6 dark:border-zinc-800">
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
            onStrengthChange={(id, v) =>
              setStrengths((prev) => ({ ...prev, [id]: v }))
            }
            loading={featuresLoading}
            customStrengths={customStrengths}
            onCustomAdd={(id, strength) =>
              setCustomStrengths((prev) => ({ ...prev, [id]: strength }))
            }
            onCustomRemove={(id) =>
              setCustomStrengths((prev) => {
                const next = { ...prev };
                delete next[id];
                return next;
              })
            }
            onCustomChange={(id, strength) =>
              setCustomStrengths((prev) => ({ ...prev, [id]: strength }))
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
