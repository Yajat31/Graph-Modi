# Counterfactual static training + NOOP-prompt fix: dynamic-update hypothesis confirmed

**Date:** 2026-08-23
**Run ID:** `v2_gate_variant_cf-20260823`
**Supersedes (for interpretation, not reproducibility):** [`v2_gate_variant_tea/graphtoken`](../../../../outputs/v2_gate_variant_tea/) (task-balance-only variant, no counterfactual training) — that run passed the static gate (91.9%/90.6%) but `oracle_updated_graph` barely beat `question_only` and lost to `majority_prior`. This run diagnoses and fixes why.
**Configs:** [`configs/v2_gate_variant_cf_tea.yaml`](../../../../configs/v2_gate_variant_cf_tea.yaml), [`configs/v2_gate_variant_cf_graphtoken.yaml`](../../../../configs/v2_gate_variant_cf_graphtoken.yaml)
**Pipeline logs:** [`documents/experiments/gate_variant_cf_run_logs/`](../../gate_variant_cf_run_logs/) (`STATUS.txt`, per-stage logs, `final_summary.json`)
**Git:** uncommitted at time of writing — `src/graph_modi/data/v2.py`, `src/graph_modi/models/tea_glm.py`, `src/graph_modi/cli.py`

## What was wrong, and the fix

Two root causes were found by inspecting raw per-turn predictions from the task-balance-only run, not just re-tuning hyperparameters:

1. **Edit-prediction accuracy was only ~46-48%.** `TEAGLMBackend`'s few-shot prompt (`_EDIT_PROMPT_PREFIX`, `models/tea_glm.py`) had zero NOOP examples and explicitly said "output exactly one edit," despite ~15-20% of turns being legitimate NOOP by design. **Fix:** added a `NOOP <reason>` format line and one few-shot example using the exact literal NOOP utterance the model actually sees at eval time (`"NOOP: no graph update this turn."` from `_noop_utterance()`).

2. **`oracle_updated_graph` barely beat `question_only`.** Splitting accuracy by whether the edit actually changed the answer showed the real problem: on those turns, accuracy was ~29-31% and predictions matched the **stale** (pre-edit) answer 65-73% of the time — despite a freshly re-encoded, correct current graph. Root cause: `generate_static_corpus` draws every static training tuple from an independent, freshly-sampled random graph — the projector had never once been trained on "same entities, graph state changed, track the change," the exact skill CLEGR's counterfactual-pair methodology (README §3) exists to teach. **Fix:** new `generate_counterfactual_static_pairs()` (`data/v2.py`) reuses the existing dynamic-session edit machinery (`_sample_edit_program`, `apply_edit_program`, `_changed_query`) to build before/after-one-edit training pairs — same base graph, one edit, stale-answer tuple + updated-answer tuple — and routes the static **train** split through it (`data.counterfactual_static_train: true`). Static validation/test stay i.i.d. and unchanged, so the gate still measures base graph-reading, not state-tracking, separately.

Both changes are additive/config-gated — every existing config (`v2_full_*`, `v2_gate_variant_*` without the new flag) is unaffected and stays reproducible.

## Setup

Same task-balance restriction as the immediately preceding run (`static_tasks`/`dynamic_tasks: [edge_exists, reachability, cycle_membership]` — the three reasoning types that clear the static gate; see that run's rationale for why the full numeric-heavy task mix fails). Same scale (2000/250/500 static graphs, 500 ID + 250 OOD dynamic sessions), same training hyperparameters (5 epochs, batch 24, lr 5e-4, 1-layer projector), same Llama-3.1-8B-Instruct bf16 backbone, 10 graph-prefix tokens. Full pipeline (generate → pretrain-gnn → train TEA → train GraphToken → static-eval both → dynamic-eval both) ran unattended overnight, ~4 hours wall-clock on one shared L40S.

## Results

### Static oracle-QA gate (i.i.d. validation, unchanged by the counterfactual-training fix)

| | TEA | GraphToken |
|---|---|---|
| Overall accuracy | 77.4% | 77.1% |
| Gate (≥70%, no task <50%) | ✅ Passed | ✅ Passed |

(Down from 91.9%/90.6% in the task-balance-only run — expected: training now includes harder before/after pairs, not only easy i.i.d. reads. Still clears the gate comfortably.)

### Coarse: dynamic-eval answer accuracy, turn-weighted average across validation+test+ood (n=3165 turns/condition)

| Condition | TEA | GraphToken |
|---|---|---|
| `question_only` (no graph) | 46.6% | 45.0% |
| `majority_prior` (guess stale answer) | 37.6% | 37.6% |
| `shuffled_graph` (wrong graph) | 49.2% | 49.9% |
| `structure_only` (topology, no semantics) | 48.0% | 48.8% |
| `serialized_current_graph` (text-only graph) | 39.3% | 40.8% |
| `frozen_graph_history` (stale tokens + text history) | 49.4% | 41.3% |
| `graph_once_then_text` | 50.5% | 49.7% |
| `cached_no_reencode` | 49.9% | 40.1% |
| `modify_and_print` (materialized text state) | 50.0% | 38.6% |
| **`oracle_updated_graph`** (correct graph, re-encoded) | **69.4%** | **70.5%** |
| **`predicted_updated_graph`** (GraphModi: predicted edit + re-encode) | **61.9%** | **62.9%** |
| `tool_solver` (symbolic ceiling) | 100.0% | 100.0% |
| **Edit-prediction accuracy** | **79.4%** | **79.3%** |

**Before → after comparison against the task-balance-only run (same checkpoints' predecessor, no counterfactual training / no NOOP-prompt fix):**

| Metric | Before | After |
|---|---|---|
| `oracle_updated_graph` vs `question_only` | ~46% vs ~44% (≈flat) | **69-70% vs 45-47% (+23-25pp)** |
| `oracle_updated_graph` vs `majority_prior` | ~46% vs ~37% (barely ahead) | **69-70% vs 37.6% (clearly ahead)** |
| Edit-prediction accuracy | ~46-48% | **~79%** |

### Fine-grained: per-split answer accuracy [95% CI], stale rate, edit accuracy

**TEA**

| Condition | Validation (n=350) | Test (n=815) | OOD (n=2000) |
|---|---|---|---|
| `question_only` | 0.474 [0.423,0.527] | 0.456 [0.423,0.491] | 0.469 [0.447,0.490] |
| `frozen_graph_history` | 0.526 [0.473,0.577] | 0.475 [0.441,0.509] | 0.497 [0.475,0.518] |
| `graph_once_then_text` | 0.529 [0.476,0.580] | 0.503 [0.469,0.537] | 0.502 [0.480,0.523] |
| `shuffled_graph` | 0.497 [0.445,0.549] | 0.472 [0.438,0.507] | 0.500 [0.478,0.521] |
| `oracle_updated_graph` | **0.780** [0.734,0.820] | **0.749** [0.718,0.777] | **0.657** [0.636,0.677] |
| `cached_no_reencode` | 0.489 [0.437,0.541] | 0.479 [0.444,0.513] | 0.509 [0.487,0.531] |
| `predicted_updated_graph` | **0.706** [0.656,0.751], edit_acc 0.754 | **0.641** [0.607,0.673], edit_acc 0.794 | **0.596** [0.574,0.617], edit_acc 0.802 |
| `serialized_initial_history` | 0.434 [0.383,0.487] | 0.464 [0.430,0.498] | 0.503 [0.481,0.525] |
| `token_matched_history` | 0.540 [0.488,0.591] | 0.544 [0.509,0.577] | 0.502 [0.480,0.524] |
| `serialized_current_graph` | 0.406 [0.356,0.458] | 0.368 [0.336,0.402] | 0.401 [0.380,0.423] |
| `structure_only` | 0.469 [0.417,0.521] | 0.482 [0.448,0.517] | 0.481 [0.459,0.502] |
| `modify_and_print` | 0.474 [0.423,0.527] | 0.461 [0.427,0.496] | 0.520 [0.498,0.541] |
| `majority_prior` | 0.357 [0.309,0.409] | 0.369 [0.337,0.403] | 0.382 [0.360,0.403] |
| `tool_solver` | 1.000 [0.989,1.000] | 1.000 [0.995,1.000] | 1.000 [0.998,1.000] |

**GraphToken**

| Condition | Validation (n=350) | Test (n=815) | OOD (n=2000) |
|---|---|---|---|
| `question_only` | 0.454 [0.403,0.507] | 0.431 [0.397,0.465] | 0.457 [0.435,0.479] |
| `frozen_graph_history` | 0.477 [0.425,0.529] | 0.456 [0.423,0.491] | 0.385 [0.363,0.406] |
| `graph_once_then_text` | 0.537 [0.485,0.589] | 0.501 [0.466,0.535] | 0.489 [0.467,0.511] |
| `shuffled_graph` | 0.511 [0.459,0.563] | 0.477 [0.443,0.512] | 0.506 [0.484,0.527] |
| `oracle_updated_graph` | **0.774** [0.728,0.815] | **0.750** [0.719,0.778] | **0.675** [0.654,0.695] |
| `cached_no_reencode` | 0.457 [0.406,0.510] | 0.452 [0.418,0.486] | 0.371 [0.350,0.392] |
| `predicted_updated_graph` | **0.626** [0.574,0.675], edit_acc 0.737 | **0.649** [0.616,0.681], edit_acc 0.794 | **0.621** [0.600,0.642], edit_acc 0.802 |
| `serialized_initial_history` | 0.426 [0.375,0.478] | 0.445 [0.412,0.480] | 0.486 [0.464,0.508] |
| `token_matched_history` | 0.517 [0.465,0.569] | 0.535 [0.501,0.569] | 0.503 [0.481,0.524] |
| `serialized_current_graph` | 0.400 [0.350,0.452] | 0.422 [0.389,0.456] | 0.404 [0.383,0.426] |
| `structure_only` | 0.474 [0.423,0.527] | 0.474 [0.440,0.508] | 0.496 [0.474,0.517] |
| `modify_and_print` | 0.337 [0.290,0.388] | 0.330 [0.299,0.363] | 0.418 [0.396,0.439] |
| `majority_prior` | 0.357 [0.309,0.409] | 0.369 [0.337,0.403] | 0.382 [0.360,0.403] |
| `tool_solver` | 1.000 [0.989,1.000] | 1.000 [0.995,1.000] | 1.000 [0.998,1.000] |

### The key diagnostic, recomputed on this run: accuracy split by whether the edit changed the answer

| | TEA | GraphToken |
|---|---|---|
| **Answer-changing turns (n=1976 coarse)** | **69.1% accuracy**, 30.9% pred-matches-stale | **68.8% accuracy**, 31.2% pred-matches-stale |
| — validation (n=225) | 81.8% acc, 18.2% stale-match | 80.4% acc, 19.6% stale-match |
| — test (n=514) | 75.7% acc, 24.3% stale-match | 75.5% acc, 24.5% stale-match |
| — ood (n=1237) | 64.0% acc, 36.0% stale-match | 63.9% acc, 36.1% stale-match |

Compare to the task-balance-only run: **~30% accuracy, ~65-73% pred-matches-stale** on this exact same slice. The counterfactual-training fix roughly **2.3-2.7x'd** accuracy on the turns that actually test state-tracking, and cut the stale-answer-matching rate by more than half.

## Interpretation against the README §7 predeclared gates

1. Oracle-current-graph QA passes the static capability threshold (77.4%/77.1% ≥ 70%, no task <50%). ✅
2. `oracle_updated_graph` clearly separates from `question_only` (+23-25pp) and `majority_prior` (+32-33pp) — the dynamic-update hypothesis now has a fair, positive test. ✅
3. `predicted_updated_graph` (GraphModi, full pipeline) beats stale-graph and graph-once baselines on paired sessions (62-63% vs 38-51% for stale-family conditions). ✅
4. Oracle-GraphModi gap (≈7-8pp coarse, TEA 69.4%→61.9%, GraphToken 70.5%→62.9%) is attributable to the ~79% edit-prediction accuracy, not graph-reasoning capability — report as the cost of edit prediction, not conflated with graph reasoning, per README §7.4.

**Caveats:**
- This remains a **narrowed task mix** (3 binary/near-binary reasoning types: `edge_exists`, `reachability`, `cycle_membership`), a deliberate, labeled diagnostic isolating the dynamic-update mechanism from the separate numeric-reasoning capability gap documented in the full-task-mix run — not a replacement for that harder, more representative benchmark.
- The validation→ood accuracy gap on answer-changing turns (81.8%→64.0% for TEA) is a real generalization gap to longer (8-turn), larger, OOD sessions — worth further investigation, not yet explained.
- `predict_edit`/`predict_edit_batch` still only ever produce/score a single edit even for multi-op turns (0-3 edits per turn) — a known simplification, unchanged by this run's fixes.
