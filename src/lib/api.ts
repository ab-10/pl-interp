import { BackendInfo, Feature, FeatureOverride, GenerateResponse } from "./types";

const API_BASE = "/api/backend";

export async function fetchFeatures(): Promise<Feature[]> {
  const res = await fetch(`${API_BASE}/features`);
  if (!res.ok) {
    throw new Error(`Failed to fetch features: ${res.statusText}`);
  }
  // Response shape: { "18": { "304": "label", ... }, "27": { ... } }
  const data: Record<string, Record<string, string>> = await res.json();
  const features: Feature[] = [];
  for (const [layer, labels] of Object.entries(data)) {
    for (const [id, label] of Object.entries(labels)) {
      features.push({ id: Number(id), layer: Number(layer), label });
    }
  }
  return features;
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
