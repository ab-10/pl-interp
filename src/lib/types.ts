export interface Feature {
  id: number;
  layer: number;
  label: string;
}

export interface FeatureOverride {
  id: number;
  layer: number;
  strength: number;
}

export interface GenerateRequest {
  prompt: string;
  features: FeatureOverride[];
  temperature?: number;
}

export interface GenerateResponse {
  baseline: string;
  steered: string;
}

export interface BackendInfo {
  model: string;
  saes: Record<string, string>;
}
