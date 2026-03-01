"""Pipeline configuration. Single source of truth for all parameters.

Model selection: call set_model("model-name") before using any config values,
or pass --model to scripts. Default: ministral-8b.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    model_id: str
    num_layers: int
    hidden_dim: int = 4096

    @property
    def capture_layers(self) -> list[int]:
        """50% and 75% points — mid-network semantics and late-network refinement."""
        return [self.num_layers // 2, 3 * self.num_layers // 4]

    @property
    def hidden_states_indices(self) -> list[int]:
        """Indices into output_hidden_states (0 = embeddings, i+1 = layer i)."""
        return [layer + 1 for layer in self.capture_layers]

    # Backward compat: primary capture layer (50%)
    @property
    def capture_layer(self) -> int:
        return self.capture_layers[0]

    @property
    def hidden_states_index(self) -> int:
        return self.hidden_states_indices[0]


MODELS: dict[str, ModelConfig] = {
    "ministral-8b": ModelConfig(
        model_id="mistralai/Ministral-8B-Instruct-2410",
        num_layers=36,
    ),
}

# ---------------------------------------------------------------------------
# Active model config (set by set_model(), used by all pipeline modules)
# ---------------------------------------------------------------------------

MODEL_NAME: str = ""
MODEL_ID: str = ""
MODEL_DTYPE: str = "float16"
MODEL_HIDDEN_DIM: int = 4096
MODEL_NUM_LAYERS: int = 0
CAPTURE_LAYERS: list[int] = []
HIDDEN_STATES_INDICES: list[int] = []
# Backward compat aliases (primary = 50% layer)
CAPTURE_LAYER: int = 0
HIDDEN_STATES_INDEX: int = 0

# Storage paths (model-specific)
SCRATCH_ROOT = Path("/scratch")
SCRATCH_DIR = Path("")
GENERATIONS_DIR = Path("")
ACTIVATIONS_DIR = Path("")
SAE_DIR = Path("")
STEERING_DIR = Path("")
ANALYSIS_DIR = Path("")
ALL_SCRATCH_DIRS: list[Path] = []


def set_model(name: str) -> None:
    """Reconfigure all model-dependent settings. Call early, before imports that read config."""
    global MODEL_NAME, MODEL_ID, MODEL_HIDDEN_DIM, MODEL_NUM_LAYERS
    global CAPTURE_LAYERS, HIDDEN_STATES_INDICES, CAPTURE_LAYER, HIDDEN_STATES_INDEX
    global SCRATCH_DIR, GENERATIONS_DIR, ACTIVATIONS_DIR
    global SAE_DIR, STEERING_DIR, ANALYSIS_DIR, ALL_SCRATCH_DIRS

    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODELS.keys())}")

    m = MODELS[name]
    MODEL_NAME = name
    MODEL_ID = m.model_id
    MODEL_HIDDEN_DIM = m.hidden_dim
    MODEL_NUM_LAYERS = m.num_layers
    CAPTURE_LAYERS = m.capture_layers
    HIDDEN_STATES_INDICES = m.hidden_states_indices
    CAPTURE_LAYER = m.capture_layer
    HIDDEN_STATES_INDEX = m.hidden_states_index

    SCRATCH_DIR = SCRATCH_ROOT / name
    GENERATIONS_DIR = SCRATCH_DIR / "generations"
    ACTIVATIONS_DIR = SCRATCH_DIR / "activations"
    SAE_DIR = SCRATCH_DIR / "sae"
    STEERING_DIR = SCRATCH_DIR / "steering"
    ANALYSIS_DIR = SCRATCH_DIR / "analysis"
    ALL_SCRATCH_DIRS = [GENERATIONS_DIR, ACTIVATIONS_DIR, SAE_DIR, STEERING_DIR, ANALYSIS_DIR]


def add_model_arg(parser) -> None:
    """Add --model argument to an argparse parser. Call set_model(args.model) after parsing."""
    parser.add_argument(
        "--model", type=str, default="ministral-8b",
        choices=list(MODELS.keys()),
        help=f"Model config to use (default: ministral-8b)",
    )


# Initialize with default
set_model("ministral-8b")


# ---------------------------------------------------------------------------
# Generation parameters (shared across models)
# ---------------------------------------------------------------------------

TEMPERATURE = 0.7
TOP_P = 0.95
MAX_NEW_TOKENS = 512
NUM_RUNS = 5
BASE_SEED = 42  # seed = BASE_SEED + run_id (42, 43, 44, 45, 46)
NUM_GPUS = 2

# --- Extraction & evaluation ---
EXTRACTION_TIMEOUT = 3  # seconds per test execution
GREEDY_RETRY_TEMP = 0.0  # temperature for extraction retry

# --- Variants ---
UNIVERSAL_CONTRACT = (
    "Return only code. Define exactly the function with the specified signature. "
    "No prints. No markdown."
)

VARIANTS = {
    "baseline": "",  # contract only
    "typed": (
        "Use full Python type hints. "
        "The function must pass static type checking."
    ),
    "invariants": (
        "Clearly state assumptions using assertions. "
        "Handle edge cases explicitly with assert or if+raise."
    ),
    "decomposition": (
        "Use at least 2 helper functions OR ≥3 named intermediate variables. "
        "Avoid long one-liner expressions."
    ),
    "error_handling": (
        "On invalid input, raise ValueError. Handle empty inputs explicitly. "
        "Never silently return wrong results."
    ),
    "control_flow": (
        "Use guard clauses and early returns. Limit nesting depth to 2. "
        "Prefer flat control flow."
    ),
}

VARIANT_IDS = list(VARIANTS.keys())

# --- Failure categories ---
FAILURE_CATEGORIES = [
    "pass", "wrong_answer", "syntax_error", "type_error",
    "runtime_error", "timeout", "extraction_fail",
]

# --- SAE ---
SAE_NUM_FEATURES = 16_384  # 4x expansion (was 32x=131k; reduced to fix dead features)
SAE_K = 64  # TopK sparsity: exactly K active features per token
SAE_TRAINING_TOKENS = 10_000_000  # ~7 epochs over 1.4M unique tokens

# --- Steering ---
STEERING_NUM_FEATURES = 3
STEERING_ALPHAS = [1.0, 3.0, 5.0, -3.0]
STEERING_TASKS = 164  # HumanEval only for steering

# --- Weights & Biases ---
WANDB_PROJECT = "pl-interp"
WANDB_ENTITY = None  # override via WANDB_ENTITY env var
