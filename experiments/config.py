"""Pipeline configuration. Single source of truth for all parameters."""

from pathlib import Path

# --- Model ---
MODEL_ID = "mistralai/Ministral-8B-Instruct-2410"
MODEL_DTYPE = "float16"
MODEL_HIDDEN_DIM = 4096
MODEL_NUM_LAYERS = 36

# --- Activation capture ---
# Layer 18 (50% of 36) = index 19 in output_hidden_states (index 0 = embeddings)
CAPTURE_LAYER = 18
HIDDEN_STATES_INDEX = CAPTURE_LAYER + 1  # off-by-one: hidden_states[0] = embeddings

# --- Generation ---
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_NEW_TOKENS = 512
NUM_RUNS = 3
BASE_SEED = 42  # seed = BASE_SEED + run_id (42, 43, 44)
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
    "pass",
    "wrong_answer",
    "syntax_error",
    "type_error",
    "runtime_error",
    "timeout",
    "extraction_fail",
]

# --- SAE ---
SAE_NUM_FEATURES = 32_768
SAE_K = 64  # TopK sparsity: exactly K active features per token
SAE_TRAINING_TOKENS = 2_000_000

# --- Steering ---
# 3 SAE features × 3 conditions (+α, −α, random) × 164 tasks + 164 baseline
STEERING_NUM_FEATURES = 3
STEERING_ALPHAS = [1.0, 3.0, 5.0, -3.0]
STEERING_TASKS = 164  # HumanEval only for steering

# --- Weights & Biases ---
WANDB_PROJECT = "pl-interp"
WANDB_ENTITY = None  # override via WANDB_ENTITY env var

# --- Storage ---
# NVMe scratch on VM (ephemeral, 7TB, fast sequential writes)
# Model-specific subdirectory to allow multiple model runs
SCRATCH_DIR = Path("/scratch/ministral-8b")
GENERATIONS_DIR = SCRATCH_DIR / "generations"
ACTIVATIONS_DIR = SCRATCH_DIR / "activations"
SAE_DIR = SCRATCH_DIR / "sae"
STEERING_DIR = SCRATCH_DIR / "steering"
ANALYSIS_DIR = SCRATCH_DIR / "analysis"

ALL_SCRATCH_DIRS = [
    GENERATIONS_DIR,
    ACTIVATIONS_DIR,
    SAE_DIR,
    STEERING_DIR,
    ANALYSIS_DIR,
]
