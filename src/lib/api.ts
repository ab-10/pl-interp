import { Feature, FeatureOverride, GenerateResponse } from "./types";

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

export async function generateCompletion(
  prompt: string,
  features: FeatureOverride[]
): Promise<GenerateResponse> {
  const activeFeatures = features.filter((f) => f.strength !== 0);
  const res = await fetch(`${API_BASE}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, features: activeFeatures }),
  });
  if (!res.ok) {
    throw new Error(`Generation failed: ${res.statusText}`);
  }
  return res.json();
}
