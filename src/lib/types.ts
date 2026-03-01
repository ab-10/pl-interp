// ── Core types (unchanged for backward compat) ───────────────────

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

// ── Code example from LLM feature labeling ──────────────────────

export interface CodeExample {
  task_id: string;
  code_context: string;
  activation: number;
}

// ── Enriched feature types ────────────────────────────────────────

export interface SliderConfig {
  min: number;
  max: number;
  step: number;
  default: number;
}

export interface MonotonicityData {
  neg_avg: number;
  baseline: number;
  pos_avg: number;
  is_monotonic: boolean;
  effect_size: number;
}

export interface LogitAttribution {
  promoted: { token: string; logit: number }[];
  suppressed: { token: string; logit: number }[];
}

export interface EnrichedFeature {
  id: number;
  label: string;
  primary_variant?: string;
  cohens_d?: number;
  category?: "steering" | "control";
  slider?: SliderConfig;
  monotonicity?: Record<string, MonotonicityData>;
  logit_attribution?: LogitAttribution;
  description?: string;
  llm_label?: string;
  confidence?: "high" | "medium" | "low";
  code_examples?: CodeExample[];
}

// ── Token activation types ────────────────────────────────────────

export interface TokenActivation {
  token: string;
  activations: Record<string, number>;
}

export interface SweepResult {
  alpha: number;
  text: string;
  token_activations?: TokenActivation[];
}

export interface ActivationStats {
  count: number;
  total_tokens: number;
  sparsity: number;
  mean: number;
  max: number;
  min: number;
}

export interface TopActivatingToken {
  token: string;
  activation: number;
}

export interface EnrichedGenerateResponse extends GenerateResponse {
  token_activations?: TokenActivation[];
  sweep_results?: SweepResult[];
  baseline_density?: Record<string, number>;
  steered_density?: Record<string, number>;
  activation_stats?: Record<string, ActivationStats>;
  top_activating_tokens?: Record<string, TopActivatingToken[]>;
}

// ── Feature map types ─────────────────────────────────────────────

export interface FeatureMapPoint {
  id: number;
  x: number;
  y: number;
  primary_variant?: string;
  cohens_d?: number;
  label?: string;
}

// ── Server info types ─────────────────────────────────────────────

export interface ServerCapabilities {
  token_activations: boolean;
  alpha_sweep: boolean;
  feature_map: boolean;
  enriched_features: boolean;
  density: boolean;
  llm_analysis: boolean;
}

// ── LLM Analysis types ──────────────────────────────────────────

export interface RelabelResponse {
  feature_id: number;
  label: string;
  description: string;
  confidence: string;
  prompt_used: string;
}

export interface ServerInfo {
  model: string;
  steer_layer: number;
  d_sae: number;
  max_new_tokens: number;
  capabilities?: ServerCapabilities;
}
