# Feature Discovery Results (Milestone 2)

Run date: 2026-03-01

## Pipeline Summary

| Script | Runtime | Input | Output |
|--------|---------|-------|--------|
| 01_collect_activations.py | ~1 min | 1,278 code + 300 non-code prompts | activations_code.json, activations_noncode.json |
| 02_find_code_features.py | <1 sec | Activation JSONs | code_features_ranked.json (top 50) |
| 03_label_features.py | ~40 sec | Top 20 features + Mistral API | feature_registry.json |
| 04_verify_steering.py | ~46 sec | Top 5 features | steering_verification.json |

SAE hook point resolved to `blocks.16.hook_mlp_out` (MLP output, not residual stream). These features capture post-MLP computations at layer 16, where the model has already processed syntax into more abstract representations.

---

## Activation Collection

1,278 code prompts across 8 languages (via MultiPL-E + HumanEval) and 300 keyword-filtered Alpaca non-code prompts. Every prompt produced firing features — zero failures.

| Language | Prompts |
|----------|---------|
| py | 164 |
| js | 161 |
| java | 158 |
| cpp | 161 |
| rs | 156 |
| ts | 159 |
| sh | 158 |
| r | 161 |
| **non-code** | **300** |

---

## Differential Ranking

Top features fall into two distinct profiles.

### High-frequency, code-exclusive features

Fire broadly across code, never on prose.

| Feature | Code Freq | Code Act | NonCode Count | Label |
|---------|-----------|----------|---------------|-------|
| 17411 | 81% | 17.8 | 0 | R-style function stubs with doctests |
| 8081 | 74% | 18.6 | 0 | C++ Competitive Programming Stubs |
| 59751 | 68% | 16.5 | 0 | Rust function docstring examples |
| 84869 | 68% | 14.2 | 0 | Rust function docstring with examples |
| 40901 | 72% | 13.2 | 0 | Incomplete or Malformed Coding Problems |
| 20663 | 41% | 11.2 | 0 | Function signature with doctest examples |
| 65699 | 37% | 11.8 | 0 | Incomplete or Truncated Code Examples |
| 123547 | 42% | 9.8 | 0 | Programming Problem Statement Detection |

These detect something very general about code structure — they fire on the vast majority of code and literally zero non-code prompts.

### Low-frequency, high-activation features

Fire on fewer prompts but with extreme activation values.

| Feature | Code Freq | Code Act | NonCode Count | Label |
|---------|-----------|----------|---------------|-------|
| 6133 | 14% | **70.6** | 1 | Function stubs with docstring examples |
| 124724 | 12% | **77.4** | 3 | Bash Function Docstring Math Problems |
| 90078 | 15% | **30.9** | 3 | Java Method Stub with Doctest Examples |

Feature 6133 has a mean activation of 70.6 — roughly 4x higher than the high-frequency features. Feature 124724 hits 77.4. These are highly specialized detectors that fire intensely on specific code patterns.

### Weakest signals

Features at the bottom of the top-50 have low specificity ratios: 24767 (4.3x, fires on 9% of non-code) and 123133 (3.3x, 20 non-code prompts). These would not survive a stricter filter.

---

## Feature Labels

Labels from Mistral API (`mistral-medium-latest`) break down by category:

**Language-specific detectors:**
- 17411, 55116 — R
- 8081, 32104, 18872 — C++
- 59751, 84869, 96334 — Rust
- 32238, 73182, 90078 — Java
- 124724 — Bash

**Cross-language detectors:**
- 6133 — Function stubs with docstring examples
- 20663 — Function signature with doctest examples
- 53300 — Function signature with docstring examples
- 49053 — Doctest-style function documentation

**Structural detectors:**
- 40901 — Incomplete or Malformed Coding Problems
- 65699 — Incomplete or Truncated Code Examples
- 123547 — Programming Problem Statement Detection
- 53942 — Math/CS Problem Statement Detection

Almost all features detect the same high-level concept from different angles: **code problem stubs with doctest-style examples.** This is a dataset artifact — HumanEval and its MultiPL-E translations all share the same structure (docstring + examples + empty function body).

---

## Steering Verification

Prompt: `"Write a Python function that"` | Strength: 3.0 | Temperature: 0.3

**Baseline** (no steering): The model generates a StackOverflow-style Q&A — "I am trying to write a python function that takes in a list of integers and returns the sum..." followed by repetitive comments about `sum(lst)`. The model's default for an incomplete sentence is to role-play a help-seeking user.

### Per-feature results

**Feature 17411** (R-style stubs): Jumps straight to code — `def find_max(lst)` with full implementation and example usage. No Q&A framing. Steering pushes the model from "confused beginner asking a question" toward "here's a clean code implementation."

**Feature 8081** (C++ competitive programming): Opens with `def get_sum(lst)` immediately, then provides a structured answer with code blocks and example calls. More verbose and tutorial-like than 17411.

**Feature 59751** (Rust docstrings): Still generates a Q&A-style response, but with a more complex problem (sum of numbers > 50 with conditional logic). Steering made the code problem more sophisticated but didn't fully escape the Q&A framing.

**Feature 32104** (C++ docstring examples): Generates `def cube_num(num): return num**3` — a terse, minimal function. This feature pushes toward compact implementations.

**Feature 18872** (C++ Kata problems): Generates a more complex problem involving array slicing (`sum_n(n, arr)`), including a fabricated error discussion. Steers toward "coding challenge debugging" territory.

All 5 features produced visible behavioral changes. The dominant axis is **Q&A mode vs direct code generation**.

---

## Assessment

**What works:** The differential analysis cleanly separates code from non-code features. Top features are near-perfect classifiers (81% code, 0% non-code). Steering produces visible behavioral changes.

**The limitation:** Almost all top features detect "HumanEval-style function stubs with doctests" — the same concept, varied by language. This is because the dataset is structurally uniform. The SAE found features for that specific format, not deeper code semantics like error handling, recursion, or type safety.

**For the web UI (Milestone 4):** Current features let you steer along one behavioral axis: "generate code immediately" vs "discuss code conversationally." To discover features for finer-grained code properties (error handling, functional style, verbosity), activation collection needs a more diverse code dataset — not just HumanEval prompts.
