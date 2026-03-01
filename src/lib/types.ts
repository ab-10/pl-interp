export interface Feature {
  id: number;
  label: string;
}

export interface FeatureOverride {
  id: number;
  strength: number;
}

export interface GenerateRequest {
  prompt: string;
  features: FeatureOverride[];
}

export interface GenerateResponse {
  baseline: string;
  steered: string;
}
