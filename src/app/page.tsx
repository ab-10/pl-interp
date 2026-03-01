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
  const [featuresLoading, setFeaturesLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

    const overrides: FeatureOverride[] = Object.entries(strengths).map(
      ([id, strength]) => ({ id: Number(id), strength })
    );

    try {
      const result = await generateCompletion(prompt, overrides);
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
          <FeaturePanel
            features={features}
            strengths={strengths}
            onStrengthChange={(id, v) =>
              setStrengths((prev) => ({ ...prev, [id]: v }))
            }
            loading={featuresLoading}
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
