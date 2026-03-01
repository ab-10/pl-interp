# Pipeline Results — March 1, 2026

## Infrastructure

| VM | Model | Size | Region | Stages Completed |
|---|---|---|---|---|
| h100-dev-box-4 | Mistral-7B-Instruct-v0.3 | 2x H100 NVL | Azure WUS3 | 1-5b (5c crashed, no steering) |
| h100-dev-box-5 | Ministral-8B | 2x H100 NVL | Azure WUS3 | 1-7 (full pipeline) |

Total wall clock: ~20 min per model for Stages 1-3. SAE training + analysis + steering added ~90 min on box-5.

---

## Generation Results (Stage 1-2)

### Pass Rates by Variant

| Variant | Mistral-7B | Ministral-8B | Both models trend |
|---|---|---|---|
| **baseline** | **22.2%** | **36.8%** | Best |
| typed | 19.9% (-2.3) | 35.3% (-1.5) | |
| control_flow | 20.6% (-1.6) | 34.1% (-2.7) | |
| error_handling | 20.1% (-2.1) | 33.8% (-3.0) | |
| invariants | 20.1% (-2.1) | 33.9%* (-2.9) | |
| **decomposition** | **14.9% (-7.3)** | **28.6% (-8.2)** | Worst |

*Ministral-8B invariants corrected for run_2 extraction bug (see Data Quality).

**Key finding: Every variant hurts pass rate. Ranking is consistent across both models.** Decomposition is the most harmful (-7 to -8pp). The variants add structural constraints that make generation harder without improving correctness.

### Volume Summary

| Metric | Mistral-7B | Ministral-8B |
|---|---|---|
| Total records | 11,952 | 17,532 |
| Unique tasks | 664 (HumanEval + MBPP) | 974 (MBPP only) |
| Extraction clean rate | 100.0% | 99.9% |
| Cross-run variance | +/-2pp (stable) | +/-2pp (stable, except invariants run_2) |
| Mean generated tokens (baseline) | 136.0 | 49.6 |
| 512-token truncation rate | 2.5% | 0.2% |

### Failure Categories

| Category | Mistral-7B | Ministral-8B |
|---|---|---|
| pass | 19.6% | 32.6% |
| wrong_answer | 39.9% | 36.6% |
| type_error | 18.5% | 20.3% |
| runtime_error | 18.5% | 6.5% |
| syntax_error | 3.1% | 3.8% |
| timeout | 0.4% | 0.2% |

### Task Difficulty (majority-vote across 6 variants)

| Solvable by | Mistral-7B | Ministral-8B |
|---|---|---|
| 0/6 variants | 67.0% | 54.9% |
| 6/6 variants | 5.0% | 15.5% |

Ministral-8B shows a more bimodal distribution — tasks are either unsolvable or solvable by most variants.

---

## SAE Training (Stage 4)

### Mistral-7B SAE

| Parameter | Value |
|---|---|
| Architecture | TopK SAE (d_sae=32,768, k=64) |
| Training tokens | 2,000,000 (balanced pass/fail) |
| Pass token pool | 225,913 |
| Fail token pool | 1,357,988 |
| Final variance explained | 0.627 (at step 250/489) |
| **Dead features post-training** | **30,384 / 32,768 (92.7%)** |
| Alive features | 2,384 (7.3%) |
| Features with positive Cohen's d | 1,082 |

**Critical concern:** 92.7% dead feature rate despite 0 dead features reported during training. The SAE's effective capacity is ~7% of design. Variance explained at 0.627 is moderate.

### Ministral-8B SAE

| Parameter | Value |
|---|---|
| Architecture | TopK SAE (d_sae ~131,072, k=64) |
| Checkpoint size | 4.1 GB (vs 1.1 GB Mistral) |

Ministral-8B used a 32x expansion (vs 8x for Mistral-7B). Dead feature stats not logged but inferred to be high given the 4x larger dictionary.

---

## Feature Candidates (Stage 5)

### Mistral-7B Top 3

| Feature | Variant | Cohen's d | Peak Activation |
|---|---|---|---|
| 3155 | decomposition | +355.67 | 0.1015 |
| 12311 | typed | -324.17 | 0.0946 |
| 27052 | error_handling | +78.06 | 0.0531 |

### Ministral-8B Top 3

| Feature | Variant | Cohen's d | Peak Activation |
|---|---|---|---|
| 76259 | decomposition | +123,289 | 0.1251 |
| 130545 | error_handling | +655.72 | 0.1330 |
| 11736 | control_flow | -440.32 | 0.1335 |

All features show near-perfect variant specificity (activate almost exclusively on their primary variant) and cross-run reproducibility (identical activations at same task/position across runs 0/1/2). Features respond to **prompt structure**, not generation stochasticity.

---

## Steering Results — Ministral-8B (Stages 6-7)

Baseline (unsteered) HumanEval pass rate: **62.2% (102/164)**.

### Contrastive Steering

| Direction | alpha=+3 | alpha=-3 |
|---|---|---|
| control_flow | 48.2% (p=0.014) | 52.4% (p=0.094) |
| decomposition | 46.9% (p=0.008) | 57.9% (p=0.499) |
| error_handling | 53.7% (p=0.146) | 47.6% (p=0.011) |
| invariants | 50.0% (p=0.034) | 54.3% (p=0.179) |
| typed | 59.2% (p=0.651) | 52.4% (p=0.094) |

4/10 conditions significant (p<0.05). **All significant effects are negative.**

### SAE Steering

| Feature (Variant) | alpha=+3 | alpha=-3 |
|---|---|---|
| 76259 (decomposition) | 54.9% (p=0.218) | 57.3% (p=0.431) |
| 130545 (error_handling) | 54.3% (p=0.179) | 57.9% (p=0.499) |
| 11736 (control_flow) | 57.3% (p=0.431) | 56.7% (p=0.368) |
| random_116424 | 59.2% (p=0.651) | 53.0% (p=0.118) |
| random_40235 | 51.2% (p=0.058) | 57.3% (p=0.431) |
| random_94346 | 51.2% (p=0.058) | **61.6%** (p=1.000) |

**0/12 SAE conditions significant.** Random features produce comparable or larger effects than targeted features.

### Summary

| Metric | Contrastive | SAE Targeted | SAE Random |
|---|---|---|---|
| Mean |delta| from baseline | 9.9pp | 5.8pp | 6.6pp |
| Significant results (p<0.05) | 4/10 | 0/6 | 0/6 |
| Any positive delta? | **No** | **No** | **No** |

**Steering is a universal null result.** All 22 conditions reduce pass rate. Not a single condition improves over the 62.2% baseline. The model's coding ability is disrupted by any activation intervention at alpha=+/-3.

---

## Known Issues

### 1. Mistral-7B Stage 5c Crash
Contrastive direction computation failed with `ValueError: No baseline records found` because generation files at `/scratch/generations/` were deleted between stages 5b and 5c. The old script used bare `/scratch/` paths instead of model-specific `/scratch/mistral-7b/`. **Fix:** Re-run stages 5c-7 using the refactored `run_stages_4_7.sh` with `--model mistral-7b`, copying generation files from `~/results/mistral-7b/generations/` back to `/scratch/mistral-7b/generations/`.

### 2. Ministral-8B Invariants Run_2 Extraction Bug
462/974 invariants run_2 records retained backtick fences in `extracted_code` (marked `extraction_clean=True` incorrectly), causing 548 false syntax errors. Inflates syntax_error rate and deflates invariants pass rate by ~7pp. **Fix:** Harden extraction validation to reject code containing backtick fences.

### 3. SAE Dead Feature Rate (92.7% on Mistral-7B)
Training reported 0 dead features, but post-training analysis found 92.7% dead. The dead feature resampling mechanism is not working effectively. Aux loss was 0.0 throughout training.

### 4. Error Hash Uniqueness
Error hashes include temp-file paths in tracebacks, making nearly every error "unique." Not useful for clustering. Use `failure_category` instead.

---

## Data Locations

### On VMs (rsynced to persistent disk)
- `h100-dev-box-4:~/results/mistral-7b/` — generations, SAE checkpoint (1.1GB), analysis
- `h100-dev-box-5:~/results/ministral-8b/` — generations, SAE checkpoint (4.1GB), analysis, steering

### Local
- `results/mistral-7b/` — generations, feature candidates, stages log
- `results/ministral-8b/` — generations, feature candidates, steering results, steering records
