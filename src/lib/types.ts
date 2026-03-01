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
  temperature?: number;
}

export interface GenerateResponse {
  baseline: string;
  steered: string;
}

export interface BackendInfo {
  model: string;
  sae: string;
  layer?: number;
}

// --- Analyze API types ---

export interface AnalyzeRequest {
  prompt: string;
  feature_id: number;
  steering: FeatureOverride[];
}

export interface SAEDecompositionEntry {
  feature_id: number;
  label: string;
  activation: number;
}

export interface LayerAttribution {
  layer: number;
  attn: number;
  mlp: number;
}

export interface TokenDetail {
  sae_decomposition: SAEDecompositionEntry[];
  layer_attribution: LayerAttribution[];
  reconstruction_error: number;
}

export interface AnalyzeResponse {
  tokens: string[];
  feature_label: string;
  feature_activations: number[];
  token_details: Record<string, TokenDetail>;
}
