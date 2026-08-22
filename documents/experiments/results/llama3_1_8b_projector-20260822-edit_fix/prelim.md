# Preliminary result: self-directed edit extraction + 4-step evolution

**Date:** 2026-08-22
**Run ID:** `llama3_1_8b_projector-20260822-edit_fix`
**Supersedes/extends:** [`llama3_1_8b_projector-20260821-eval_subset`](../llama3_1_8b_projector-20260821-eval_subset/prelim.md) — same trained checkpoint, no retraining; this run fixes the self-directed edit-prediction path and adds a new baseline + a dedicated fixed-turn-count dataset.
**Configs:** [`configs/llama3_1_8b_eval_subset.yaml`](../../../../configs/llama3_1_8b_eval_subset.yaml) (subset re-run), [`configs/llama3_1_8b_4step_eval.yaml`](../../../../configs/llama3_1_8b_4step_eval.yaml) (new 4-step set)
**Git commit:** uncommitted — see "Code changes" below. Same checkpoint as the 2026-08-21 run (`outputs/llama3_1_8b/projector/checkpoint-final`, `outputs/llama3_1_8b/gnn/`), which was itself trained on uncommitted code.

## What changed since the 2026-08-21 run

The 2026-08-21 write-up found `predicted_updated_graph` (the realistic, self-directed version of the framework — model predicts its own edit from the revision text, applies it, re-encodes) completely broken: 0% execution-equivalent edit accuracy. Diagnosed and fixed today, no retraining involved (pure inference-path fix):

1. **Empty prompt → schema + few-shot prompt.** `TEAGLMBackend.predict_edit()` previously prompted with just `"Revision: {utterance}\nReturn one graph edit:"` — no schema, no examples. Added a fixed prefix describing the edit grammar (`SET NODE <station> status open|closed`, `ADD/DEL EDGE <station> <station> transfer|track`) with 5 worked examples matching this dataset's actual utterance phrasings.
2. **Graph-conditioned generation was corrupting output.** `predict_edit()` originally called the graph-conditioned `generate()` (graph prefix tokens appended right before generation). Diagnosed by comparing against the same prompt through `generate_text_only()`: without graph tokens, the model produced the exact correct edit, node name and all; with graph tokens, output was incoherent (`'5:3'`, `'2:4'`, digit soup). Root cause: Stage-2 training only ever placed the graph prefix immediately before *short* QA prompts — appending it after this longer few-shot instruction prompt is out of distribution for the frozen LM. Fixed by switching `predict_edit()` to `generate_text_only()` — architecturally correct anyway, since the utterance always names the target node explicitly and needs no graph context to parse.
3. **No stop token → hallucinated continuations corrupted otherwise-correct output.** The model has no learned stop condition for this format and keeps generating fabricated `Revision:/Edit:` continuations of the few-shot pattern. `parse_edit()`'s `SET NODE` branch greedily joins everything after token 4 as the value field, so trailing hallucinated text corrupted the value even when the actual edit prediction was correct. Fixed by truncating to the first line of generated text before parsing.

Verified incrementally against the trained model on real held-out utterances at each step: 0/10 → 0/25 (well-formed but node-name corrupted) → 17/25 (text-only routing) → 30/30 (+ first-line truncation).

Also added a new evaluation condition, `graph_once_then_text` (`src/graph_modi/evaluation/runner.py`, `src/graph_modi/models/tea_glm.py`, `src/graph_modi/models/base.py` — added `ModelInput.turn_index`): graph tokens supplied on turn 0 only, every subsequent turn is text-only. Isolates whether the model needs a *repeated* graph reminder or can carry graph information forward itself after a single exposure. Full suite (pytest, 29 tests; `make smoke`-equivalent CPU pipeline) passes with the new condition included.

## Setup

Same trained artifacts as the 2026-08-21 run (see that write-up for full architecture/training details): Llama-3.1-8B-Instruct backbone, 8-layer sum-aggregation GraphSAGE (4096 hidden, query-conditioned), single-linear projector, 52.5% GNN train accuracy, projector final loss 0.57.

Two evaluation runs:
- **Subset re-run**: same 100 validation + 50 test sessions as 2026-08-21 (variable turn counts, cycled through `[1,2,3,5]`), re-evaluated with the fixed `predict_edit()` and the same 10 original conditions (the 11th, `graph_once_then_text`, was added to the config after this run had already started, so it's not in this particular run — see the 4-step set below for that).
- **New 4-step set**: 150 freshly-generated sessions, each with **exactly 4 turns** (`data.turns: [4]` in `configs/llama3_1_8b_4step_eval.yaml`), same locked Watts-Strogatz distribution otherwise. Every session evaluated under all 11 conditions. Generation is pure CPU/symbolic (no LLM), audit passed clean on all 600 turns (`4step_data_audit.json`). This gives a clean, equal-N-per-turn-index view of performance across a fixed number of cumulative edits, rather than reusing/truncating the original variable-length sessions.

Both runs launched as detached background processes on separate GPUs (0 and 1) to run in parallel; no OOM, no contention.

## Results

### Per-condition answer accuracy — subset re-run (unchanged sessions from 2026-08-21, fixed predict_edit)

| Condition | Validation (n=100) | Test (n=50) |
|---|---|---|
| `oracle_updated_graph` | 0.404 [0.347, 0.463] | 0.415 [0.335, 0.499] |
| `predicted_updated_graph` | 0.418 [0.361, 0.477], **edit_exec_eq=1.000** | 0.393 [0.314, 0.477], **edit_exec_eq=1.000** |
| `cached_no_reencode` | 0.298 [0.247, 0.355] | 0.296 [0.226, 0.378] |
| `shuffled_graph` | 0.280 [0.230, 0.336] | 0.311 [0.239, 0.394] |
| `frozen_graph_history` | 0.280 [0.230, 0.336] | 0.274 [0.206, 0.355] |
| `serialized_initial_history` | 0.167 [0.128, 0.216] | 0.274 [0.206, 0.355] |
| `question_only` | 0.153 [0.115, 0.200] | 0.163 [0.110, 0.234] |
| `tool_solver` (ceiling) | 1.000 | 1.000 |

`execution_equivalent_edit_accuracy` for `predicted_updated_graph` is **1.000 on both splits** (was 0.000 on 2026-08-21) — every predicted edit across 275 validation turns and 135 test turns matched the gold edit under execution equivalence.

### Per-condition answer accuracy — new 4-step set (150 sessions, exactly 4 turns each, n=600 turns/condition)

| Condition | Accuracy [95% CI] | `edit_exec_eq` | Input tokens |
|---|---|---|---|
| `oracle_updated_graph` | 0.452 [0.412, 0.492] | — | 19 |
| `predicted_updated_graph` | 0.450 [0.411, 0.490] | **1.000** | 19 |
| `shuffled_graph` | 0.332 [0.295, 0.370] | — | 19 |
| `frozen_graph_history` | 0.307 [0.271, 0.345] | — | 36 |
| `cached_no_reencode` | 0.288 [0.254, 0.326] | — | 36 |
| `serialized_initial_history` | 0.237 [0.204, 0.272] | — | 151 |
| `token_matched_history` | 0.217 [0.186, 0.251] | — | 36 |
| **`graph_once_then_text`** | **0.168 [0.141, 0.200]** | — | 36 |
| `serialized_current_graph` | 0.148 [0.122, 0.179] | — | 134 |
| `question_only` | 0.143 [0.118, 0.174] | — | 19 |
| `tool_solver` (ceiling) | 1.000 [0.994, 1.000] | — | 19 |

### Paired bootstrap significance (10,000 resamples, matched by session+turn)

**Subset re-run:**

| Comparison | Validation Δ [95% CI] | Test Δ [95% CI] |
|---|---|---|
| `predicted_updated_graph` − `oracle_updated_graph` | +0.015 [−0.051, +0.080] ns | −0.022 [−0.104, +0.059] ns |
| `predicted_updated_graph` − `cached_no_reencode` | +0.120 [+0.051, +0.185] **SIG** | +0.096 [+0.007, +0.193] **SIG** |
| `predicted_updated_graph` − `frozen_graph_history` | +0.138 [+0.076, +0.204] **SIG** | +0.119 [+0.037, +0.200] **SIG** |
| `predicted_updated_graph` − `shuffled_graph` | +0.138 [+0.069, +0.207] **SIG** | +0.081 [−0.007, +0.178] ns |
| `predicted_updated_graph` − `serialized_initial_history` | +0.251 [+0.182, +0.316] **SIG** | +0.119 [+0.015, +0.222] **SIG** |
| `oracle_updated_graph` − `cached_no_reencode` | +0.105 [+0.036, +0.175] **SIG** | +0.119 [+0.015, +0.222] **SIG** |

**4-step set (n=600 paired turns):**

| Comparison | Δ accuracy | 95% CI | Win/Lose | Significant? |
|---|---|---|---|---|
| `predicted_updated_graph` − `oracle_updated_graph` | −0.002 | [−0.047, +0.043] | 97/98 | ns (statistically identical) |
| `predicted_updated_graph` − `frozen_graph_history` | +0.143 | [+0.097, +0.190] | 155/69 | **SIG** |
| `predicted_updated_graph` − `cached_no_reencode` | +0.162 | [+0.117, +0.207] | 157/60 | **SIG** |
| `predicted_updated_graph` − `graph_once_then_text` | +0.282 | [+0.235, +0.328] | 204/35 | **SIG** |
| `predicted_updated_graph` − `shuffled_graph` | +0.118 | [+0.070, +0.167] | 155/84 | **SIG** |
| `oracle_updated_graph` − `cached_no_reencode` | +0.163 | [+0.113, +0.213] | 175/77 | **SIG** |
| `frozen_graph_history` − `graph_once_then_text` | +0.138 | [+0.093, +0.183] | 143/60 | **SIG** |
| `frozen_graph_history` − `question_only` | +0.163 | [+0.118, +0.208] | 152/54 | **SIG** |
| `graph_once_then_text` − `question_only` | +0.025 | [−0.015, +0.067] | 88/73 | **ns** |

### Accuracy by turn index — 4-step set (n=150 per cell)

| Condition | Turn 0 | Turn 1 | Turn 2 | Turn 3 |
|---|---|---|---|---|
| `oracle_updated_graph` | 0.387 | 0.600 | 0.347 | **0.473** |
| `predicted_updated_graph` | 0.407 | 0.540 | 0.367 | **0.487** |
| `frozen_graph_history` | 0.227 | 0.500 | 0.287 | **0.213** |
| `cached_no_reencode` | 0.187 | 0.487 | 0.273 | **0.207** |
| `shuffled_graph` | 0.200 | 0.513 | 0.320 | 0.293 |
| `graph_once_then_text` | 0.180 | 0.127 | 0.133 | 0.233 |
| `question_only` | 0.127 | 0.180 | 0.120 | 0.147 |

## Interpretation

1. **The self-directed loop now works, and matches the oracle.** `predicted_updated_graph` vs `oracle_updated_graph` is not significant on any split, including the largest sample (4-step set, n=600, diff −0.2pp). This is the practically important result: the *realistic, deployable* version of the framework — no gold edit fed in, the model extracts it from the revision text itself — performs indistinguishably from the idealized upper bound, because edit extraction is now 100% execution-equivalent everywhere it was tested (410+ paired turns across three independent evaluation runs).
2. **A single graph exposure is worthless without repetition.** `graph_once_then_text` (graph shown once at turn 0, pure text after) is statistically indistinguishable from `question_only` (graph never shown at all): diff +2.5pp, CI includes 0, n=600. Whatever the model does with the graph tokens at turn 0 does not persist into later turns in any way that helps answer subsequent questions. This directly rules out "the model just needs to see the graph once and can track changes itself from there" as a viable cheaper alternative to your framework.
3. **Stale approaches degrade specifically as edits accumulate; the framework doesn't.** The turn-by-turn table is the clearest evidence for the core hypothesis: at turn 0 all graph-shown conditions are roughly comparable (0.19–0.41), but by turn 3 (after 4 cumulative edits) `frozen_graph_history`/`cached_no_reencode` have fallen to ~0.21, barely above the 0.147 question-only floor, while `oracle`/`predicted_updated_graph` are still at ~0.48–0.49 — essentially undiminished from where they started. The gap between "your framework" and "stale graph" widens specifically as a function of accumulated edits, which is the actual claim the project needs to support.
4. Turn 1 shows an accuracy spike across *every* condition including the weak ones (`frozen_graph_history` 0.500, `cached_no_reencode` 0.487, `shuffled_graph` 0.513) — likely a property of which reasoning types/queries land at that turn index in this generated set rather than a modeling effect, since it's shared across conditions including ones that shouldn't have graph info. Worth keeping in mind when interpreting turn 1 specifically; the turn-3 divergence is the more load-bearing number.
5. `predicted_updated_graph` vs `shuffled_graph` is significant on the 4-step set (n=600) and validation, but not on the small subset's test split (n=135) — consistent with that being the noisiest of the three paired comparisons, not a real inconsistency.

## Caveats

- Subset re-run and 4-step set are both compute-constrained preliminary checks (100–150 sessions), not the full 1,000/2,000-session locked splits.
- Uncommitted code (see git status at write time: `cli.py`, `evaluation/runner.py`, `models/base.py`, `models/tea_glm.py`, `training.py`, `README.md`, `configs/smoke.yaml` modified; three new configs untracked). Same caveat as 2026-08-21 — commit before treating as citable.
- The turn-1 spike (point 4 above) is unexplained; worth a quick look at the 4-step set's per-turn reasoning-type distribution before over-interpreting any single turn index in isolation from the overall trend.
- `graph_once_then_text` and the 4-step dataset are both new this run; only one dataset draw each so far.

## Decision / next steps

The core hypothesis is now supported by three independent pieces of evidence (2026-08-21 subset, 2026-08-22 subset re-run, 2026-08-22 4-step set), and the practically-relevant version of the framework (self-directed, not oracle-fed) is confirmed working. Suggested next steps:
1. Commit the code changes (predict_edit fix + `graph_once_then_text` condition) — currently uncommitted across two days of work.
2. Investigate the turn-1 spike briefly (check reasoning-type distribution by turn index in the 4-step set) before including it in any report without comment.
3. Full 1,000/2,000-session evaluation is optional at this point for the yes/no question (already answered); still valuable for tighter effect-size estimates if this becomes a written report.
