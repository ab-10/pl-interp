# Discovered SAE Features

Results from generation-based feature discovery, run 2026-03-01.

## Setup

| Parameter | Value |
|-----------|-------|
| **Model** | `mistralai/Ministral-8B-Instruct-2410` (bfloat16) |
| **SAE layer 18** | `~/8b_saes/layer_18_sae_checkpoint.pt` on GPU VM |
| **SAE layer 27** | `~/8b_saes/layer_27_sae_checkpoint.pt` on GPU VM |
| **SAE architecture** | Custom BatchTopK, d_model=4096, d_sae=16384, k=64 |
| **SAE training data** | 6 epochs, ~14.4M tokens |
| **GPU VM** | `azureuser@20.38.0.252`, 2x NVIDIA H100 NVL (96 GB each) |
| **Discovery method** | Generation-based contrastive (Cosgrove Method 3) |
| **Discovery script** | `scripts/run_discovery.py` |
| **Runtime** | ~36 minutes on one H100 |

## Verified Features

30 features across 5 code properties, all verified to produce visible steering
effects on 3 neutral prompts. Features are ranked by differential score within
each property. The "best per property" feature (highest specificity or
diff score) is marked with an arrow.

### Type Annotations

| Layer | Feature | Diff Score | Specificity | Demo Strength |
|-------|---------|-----------|-------------|---------------|
| 18 | **13176** | 1.403 | 1499x | 3.0 |
| 18 | 10100 | 0.638 | 300x | 3.0 |
| 18 | 5690 | 0.440 | 1.1x | 3.0 |
| 27 | 15092 | 0.612 | 1.9x | 3.0 |
| 27 | **6538** | 0.435 | 745x | 3.0 |
| 27 | 9865 | 0.431 | 1.6x | 3.0 |

L18:13176 is the standout feature — fires in 41% of target generation tokens
with near-zero activation on control prompts (1499x specificity). L27:6538 is
similarly sharp (745x). Feature 5690 (L18) also appears under recursion (see
below), suggesting it encodes a broader "structured code" concept.

### Error Handling

| Layer | Feature | Diff Score | Specificity | Demo Strength |
|-------|---------|-----------|-------------|---------------|
| 18 | **9742** | 1.024 | 1.5x | 3.0 |
| 18 | 15112 | 0.727 | 3.1x | 5.0 |
| 18 | 853 | 0.623 | 1.7x | 3.0 |
| 27 | **10821** | 0.703 | 1.7x | 3.0 |
| 27 | 9958 | 0.431 | 2.0x | 5.0 |
| 27 | 10731 | 0.409 | 1.6x | 3.0 |

Error handling features have moderate specificity (1.5–3.1x) because
try/except patterns share tokens with general control flow. L18:9742 has the
highest differential score. L18:15112 and L27:9958 require strength 5.0 for a
visible effect.

### Recursive Patterns

| Layer | Feature | Diff Score | Specificity | Demo Strength |
|-------|---------|-----------|-------------|---------------|
| 18 | **16290** | 1.610 | 1.5x | 3.0 |
| 18 | 5690 | 0.932 | 1.3x | 3.0 |
| 18 | 5069 | 0.704 | 1.1x | 3.0 |
| 27 | **10693** | 1.208 | 2.0x | 3.0 |
| 27 | 16218 | 0.989 | 1.4x | 3.0 |
| 27 | 1777 | 0.956 | 1.2x | 3.0 |

L18:16290 has the highest differential score of any recursion feature. L27:10693
has the best specificity at 2.0x. Note that L18:5690 appears here and under type
annotations — it is likely a general "code complexity" feature rather than a
recursion-specific one.

### Verbose Comments & Documentation

| Layer | Feature | Diff Score | Specificity | Demo Strength |
|-------|---------|-----------|-------------|---------------|
| 18 | 914 | 1.315 | 2.7x | 3.0 |
| 18 | 6294 | 1.315 | 2.2x | 3.0 |
| 18 | **9344** | 0.887 | 5.6x | 3.0 |
| 27 | **15092** | 1.855 | 2.7x | 3.0 |
| 27 | 761 | 0.875 | 2.4x | 3.0 |
| 27 | 15358 | 0.484 | 1.3x | 3.0 |

L27:15092 has the highest differential score (1.855) across all verbose-comments
features, and also appears under type annotations. It likely encodes a broader
"code documentation / explanation" behavior. L18:9344 has the highest specificity
(5.6x) and is the most targeted "comments" feature in layer 18.

### Functional Style (map/filter/lambda)

| Layer | Feature | Diff Score | Specificity | Demo Strength |
|-------|---------|-----------|-------------|---------------|
| 18 | **16149** | 2.127 | 11.6x | 3.0 |
| 18 | 8575 | 1.227 | 2.3x | 5.0 |
| 18 | 5766 | 0.820 | 1.1x | 3.0 |
| 27 | **480** | 1.877 | 14.8x | 3.0 |
| 27 | 8557 | 1.096 | 3.9x | 3.0 |
| 27 | 6837 | 0.766 | 3.5x | 8.0 |

Functional style features are the sharpest overall. L18:16149 (11.6x) and
L27:480 (14.8x) are highly specific — they fire almost exclusively on
functional-style code and rarely on imperative code. L18:16149 also has the
highest differential score of any single feature found (2.127).

## Cross-Property Overlap

Two features appear under multiple properties:

| Feature | Properties | Interpretation |
|---------|-----------|----------------|
| L18:5690 | Type annotations, Recursive patterns | Broad "structured code" feature, not property-specific |
| L27:15092 | Type annotations, Verbose comments | Broad "code documentation / explanation" feature |

For the demo, use the property-specific features (not these shared ones) when
you want to show clean single-axis steering.

## Recommended Demo Configuration

**Best single feature per property for the demo** (pick one per property to
keep the slider panel clean):

| Property | Layer | Feature | Strength | Why |
|----------|-------|---------|----------|-----|
| Type annotations | 18 | 13176 | 3.0 | Highest specificity (1499x) |
| Error handling | 18 | 9742 | 3.0 | Highest diff score |
| Recursive patterns | 18 | 16290 | 3.0 | Highest diff score |
| Verbose comments | 18 | 9344 | 3.0 | Highest specificity (5.6x) |
| Functional style | 27 | 480 | 3.0 | Highest specificity (14.8x) |

**Demo prompt:** `Write a Python function that merges two sorted lists.`

**Backup prompts:**
- `Implement a function that checks if a string is a palindrome.`
- `Write a function that counts word frequencies in a string.`

**Settings:** temperature 0.3, max_new_tokens 200, slider range -10 to +10

## Data Files

| File | Location | Contents |
|------|----------|----------|
| `results/demo_features.json` | This repo | All 30 verified features with scores, strengths, labels |
| `results/ranking.json` | This repo | Full Phase 2 ranking (top 5 per layer per property with raw stats) |
| `scripts/run_discovery.py` | This repo | Discovery script (generate, rank, verify) |
| `~/8b_saes/layer_18_sae_checkpoint.pt` | GPU VM | Layer 18 SAE weights |
| `~/8b_saes/layer_27_sae_checkpoint.pt` | GPU VM | Layer 27 SAE weights |
| `~/results/demo_features.json` | GPU VM | Copy of results on the VM |
| `~/results/ranking.json` | GPU VM | Copy of ranking on the VM |
