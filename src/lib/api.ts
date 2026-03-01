import { AnalyzeResponse, BackendInfo, Feature, FeatureOverride, GenerateResponse } from "./types";

const API_BASE = "/api/backend";

export async function fetchFeatures(): Promise<Feature[]> {
  const res = await fetch(`${API_BASE}/features`);
  if (!res.ok) {
    throw new Error(`Failed to fetch features: ${res.statusText}`);
  }
  const data: Record<string, string> = await res.json();
  return Object.entries(data).map(([id, label]) => ({
    id: Number(id),
    label,
  }));
}

export async function fetchInfo(): Promise<BackendInfo> {
  const res = await fetch(`${API_BASE}/info`);
  if (!res.ok) {
    throw new Error(`Failed to fetch info: ${res.statusText}`);
  }
  return res.json();
}

export async function generateCompletion(
  prompt: string,
  features: FeatureOverride[],
  temperature: number = 0.3
): Promise<GenerateResponse> {
  const activeFeatures = features.filter((f) => f.strength !== 0);
  const res = await fetch(`${API_BASE}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, features: activeFeatures, temperature }),
  });
  if (!res.ok) {
    throw new Error(`Generation failed: ${res.statusText}`);
  }
  return res.json();
}

export async function analyzeFeature(
  prompt: string,
  featureId: number,
  steering: FeatureOverride[] = []
): Promise<AnalyzeResponse> {
  const activeSteering = steering.filter((f) => f.strength !== 0);
  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      feature_id: featureId,
      steering: activeSteering,
    }),
  });
  if (!res.ok) {
    throw new Error(`Analysis failed: ${res.statusText}`);
  }
  return res.json();
}
