import {
  EnrichedFeature,
  EnrichedGenerateResponse,
  FeatureMapPoint,
  FeatureOverride,
  RelabelResponse,
  ServerInfo,
} from "./types";

const API_BASE = "/api/backend";

/** Detect whether the /features response is enriched (objects) or flat (strings). */
function isEnrichedResponse(data: Record<string, unknown>): boolean {
  const first = Object.values(data)[0];
  return typeof first === "object" && first !== null;
}

/** Fetch available steering features. Handles both enriched and flat formats. */
export async function fetchFeatures(): Promise<EnrichedFeature[]> {
  const res = await fetch(`${API_BASE}/features`);
  if (!res.ok) {
    throw new Error(`Failed to fetch features: ${res.statusText}`);
  }
  const data = await res.json();

  if (isEnrichedResponse(data)) {
    return Object.entries(data).map(([id, value]: [string, unknown]) => {
      const v = value as Record<string, unknown>;
      return {
        id: Number(id),
        label: (v.label as string) ?? `Feature ${id}`,
        primary_variant: v.primary_variant as string | undefined,
        cohens_d: v.cohens_d as number | undefined,
        category: v.category as "steering" | "control" | undefined,
        slider: v.slider as EnrichedFeature["slider"],
        monotonicity: v.monotonicity as EnrichedFeature["monotonicity"],
        description: v.description as string | undefined,
        llm_label: v.llm_label as string | undefined,
        confidence: v.confidence as EnrichedFeature["confidence"],
        code_examples: v.code_examples as EnrichedFeature["code_examples"],
      };
    });
  }

  // Community server: flat {id: label} format
  return Object.entries(data).map(([id, label]) => ({
    id: Number(id),
    label: label as string,
  }));
}

/** Generate baseline and steered completions with optional enrichment. */
export async function generateCompletion(
  prompt: string,
  features: FeatureOverride[],
  temperature: number = 0.3,
  options?: { includeActivations?: boolean; alphas?: number[] },
): Promise<EnrichedGenerateResponse> {
  const activeFeatures = features.filter((f) => f.strength !== 0);
  const body: Record<string, unknown> = {
    prompt,
    features: activeFeatures,
    temperature,
  };
  if (options?.includeActivations) body.include_activations = true;
  if (options?.alphas) body.alphas = options.alphas;

  const res = await fetch(`${API_BASE}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`Generation failed: ${res.statusText}`);
  }
  return res.json();
}

/** Fetch server info and capabilities. Returns null if unavailable. */
export async function fetchServerInfo(): Promise<ServerInfo | null> {
  try {
    const res = await fetch(`${API_BASE}/info`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

/** Fetch the default labeling prompt for a feature. */
export async function fetchDefaultPrompt(
  featureId: number,
): Promise<{ feature_id: number; prompt: string } | null> {
  try {
    const res = await fetch(`${API_BASE}/default_labeling_prompt/${featureId}`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

/** Re-label a feature using Bedrock LLM. */
export async function relabelFeature(
  featureId: number,
  customPrompt?: string,
  updateRegistry: boolean = false,
): Promise<RelabelResponse> {
  const body: Record<string, unknown> = {
    feature_id: featureId,
    update_registry: updateRegistry,
  };
  if (customPrompt !== undefined) body.custom_prompt = customPrompt;

  const res = await fetch(`${API_BASE}/relabel_feature`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail || `Relabel failed: ${res.statusText}`);
  }
  return res.json();
}

/** Fetch pre-computed UMAP feature map. Returns null if unavailable. */
export async function fetchFeatureMap(): Promise<FeatureMapPoint[] | null> {
  try {
    const res = await fetch(`${API_BASE}/feature_map`);
    if (!res.ok) return null;
    const data = await res.json();
    return data.features ?? null;
  } catch {
    return null;
  }
}
