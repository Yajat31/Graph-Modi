# Batched evaluation at scale: definitive answer to the core hypothesis

**Date:** 2026-08-22
**Run ID:** `llama3_1_8b_projector-20260822-batched_scale`
**Supersedes:** [`llama3_1_8b_projector-20260822-edit_fix`](../llama3_1_8b_projector-20260822-edit_fix/prelim.md) (same checkpoint, no retraining) — same evaluations, rerun at ~3x the session count under a newly-batched `evaluate_sessions()`, which also changes the numbers slightly from run-to-run bf16 rounding (see "Code changes"), so treat this run's numbers as the current reference, not the previous one.
**Configs:** [`configs/llama3_1_8b_eval_subset.yaml`](../../../../configs/llama3_1_8b_eval_subset.yaml), [`configs/llama3_1_8b_4step_eval.yaml`](../../../../configs/llama3_1_8b_4step_eval.yaml)
**Git commit:** uncommitted — `src/graph_modi/evaluation/runner.py`, `models/base.py`, `models/tea_glm.py`, `cli.py` (batching); see "Code changes" below.

## Code changes: batched evaluation

`evaluate_sessions()` previously made one sequential `generate()` call per `(session, condition, turn)` — batch size 1 throughout, no progress output until the whole run finished.

1. **Batched generation primitives** (`TEAGLM.generate_batch`, `generate_text_only_batch` in `models/tea_glm.py`): left-padded batched forward passes, replacing N sequential single-item `.generate()` calls with one padded batch call. Verified exact-match against the unbatched calls on a small synthetic model. On the real 8B model, **batched and unbatched outputs are not bit-identical** — verified this is bf16 floating-point non-associativity (batched matmul kernels reduce in a different order), not an implementation bug: tried explicit `position_ids`, disabling the KV cache, and forcing eager attention instead of SDPA, none of which closed the gap. This is a documented property of low-precision batched LLM inference. Decision (explicit user call): batch anyway and treat this as the new reference going forward, rerunning prior evaluations rather than mixing batched and unbatched numbers.
2. **`predict_edit_batch` / `answer_batch`** added to `TEAGLMBackend` and `SymbolicMockBackend` (`models/base.py`, `models/tea_glm.py`), routing each item in a batch to the graph-conditioned or text-only path based on its condition (uniform within a batch by construction — see point 3).
3. **`evaluate_sessions()` rewritten** (`evaluation/runner.py`) from per-turn sequential to round-based batching: for each condition, build one `_Task` per session, then advance every task through turn (round) 0, 1, 2, ... in lockstep, batching the `predict_edit`/`answer` calls for all sessions still active at that round (sessions with fewer turns simply drop out of later rounds). This works because turn *index* routing decisions (e.g. `graph_once_then_text`'s "graph only on turn 0") are uniform across all active tasks at a given round, and turns within one `(session, condition)` pair are causally sequential so can't be batched against each other — only across different sessions/conditions at the same round.
4. **Progress logging** added to `evaluate_sessions()` (`progress=True`) and wired through `command_evaluate`: prints sessions/conditions/total-turns up front, then a line per `(condition, round)` completed, with running turn count and elapsed time — `evaluate` previously had zero output until the entire run finished.
5. New `evaluation.batch_size` config key (default 16; both configs here use 32).

**Verified before trusting the numbers below:**
- Structural correctness: on 20 real sessions × 7 conditions, batched run produced exactly the expected row count (385), correct `edit_exec_eq`/`state_exact` values, and a sensible accuracy ordering — see 2026-08-22 session notes.
- Full test suite (29 tests) and the CPU smoke pipeline (`generate` → `pretrain-gnn` → `train-projector` → `evaluate` against the mock backend, all 11 conditions including the new `graph_once_then_text`) both pass clean.
- **Speedup**: 20 sessions × 7 conditions (385 turn-evals) in 27s versus the old unbatched rate (~0.45s/turn-eval, i.e. ~173s) — roughly 6x. The two full runs below (24,475 total turn-evaluations combined) completed in **9 min** and **~9 min** respectively, versus ~50 min for the old 4-step run alone at a third of the session count.

## Setup

Same trained checkpoint as both 2026-08-21 and 2026-08-22 earlier runs (no retraining): Llama-3.1-8B-Instruct, 8-layer sum-aggregation GraphSAGE (52.5% train acc), single-linear projector (final loss 0.57).

Scaled up from the prior ~150-session runs to "a few hundred samples" per the user's request for tighter CIs, particularly to resolve the previously-ambiguous `graph_once_then_text` vs `question_only` comparison:
- **Subset re-run**: 300 validation + 150 test sessions (up from 100+50), drawn as a strict prefix of the full locked `datasets/watts_strogatz_metro_v1` splits, variable turn counts cycling `[1,2,3,5]`.
- **4-step set**: 350 sessions (up from 150), each with exactly 4 turns, freshly generated (`configs/llama3_1_8b_4step_eval.yaml`, `data.turns: [4]`). Audit passed clean on all 1,400 turns.

## Results

### Per-condition accuracy — subset re-run

| Condition | Validation (n=300) | Test (n=150) |
|---|---|---|
| `oracle_updated_graph` | 0.417 [0.384, 0.451] | 0.434 [0.387, 0.483] |
| `predicted_updated_graph` | 0.428 [0.395, 0.462], edit_exec_eq **1.000** | 0.410 [0.363, 0.458], edit_exec_eq **1.000** |
| `shuffled_graph` | 0.287 [0.257, 0.319] | 0.293 [0.251, 0.338] |
| `cached_no_reencode` | 0.290 [0.260, 0.322] | 0.293 [0.251, 0.338] |
| `frozen_graph_history` | 0.285 [0.255, 0.317] | 0.300 [0.258, 0.346] |
| `serialized_initial_history` | 0.222 [0.195, 0.251] | 0.224 [0.187, 0.267] |
| `token_matched_history` | 0.205 [0.179, 0.234] | 0.244 [0.205, 0.288] |
| `graph_once_then_text` | 0.176 [0.151, 0.203] | 0.173 [0.140, 0.213] |
| `serialized_current_graph` | 0.162 [0.139, 0.189] | 0.168 [0.135, 0.208] |
| `question_only` | 0.159 [0.135, 0.185] | 0.178 [0.144, 0.218] |
| `tool_solver` (ceiling) | 1.000 | 1.000 |

### Per-condition accuracy — 4-step set (350 sessions, exactly 4 turns each, n=1400 turns/condition)

| Condition | Accuracy [95% CI] | edit_exec_eq |
|---|---|---|
| `oracle_updated_graph` | 0.447 [0.421, 0.473] | — |
| `predicted_updated_graph` | 0.444 [0.418, 0.470] | **1.000** |
| `cached_no_reencode` | 0.316 [0.293, 0.341] | — |
| `frozen_graph_history` | 0.310 [0.286, 0.335] | — |
| `shuffled_graph` | 0.284 [0.261, 0.308] | — |
| `serialized_initial_history` | 0.243 [0.221, 0.266] | — |
| `token_matched_history` | 0.213 [0.192, 0.235] | — |
| `question_only` | 0.172 [0.153, 0.193] | — |
| `graph_once_then_text` | 0.175 [0.156, 0.196] | — |
| `serialized_current_graph` | 0.151 [0.133, 0.170] | — |
| `tool_solver` (ceiling) | 1.000 [0.997, 1.000] | — |

### Paired bootstrap significance (10,000 resamples)

**4-step set (n=1,400 paired turns — largest, cleanest sample):**

| Comparison | Δ accuracy | 95% CI | Win/Lose | Significant? |
|---|---|---|---|---|
| `predicted_updated_graph` − `oracle_updated_graph` | −0.004 | [−0.031, +0.024] | 192/197 | **ns — statistically identical** |
| `predicted_updated_graph` − `frozen_graph_history` | +0.134 | [+0.104, +0.165] | 351/164 | **SIG** |
| `predicted_updated_graph` − `cached_no_reencode` | +0.127 | [+0.096, +0.159] | 356/178 | **SIG** |
| `predicted_updated_graph` − `graph_once_then_text` | +0.269 | [+0.239, +0.299] | 481/105 | **SIG** |
| `predicted_updated_graph` − `shuffled_graph` | +0.160 | [+0.129, +0.191] | 388/164 | **SIG** |
| `oracle_updated_graph` − `cached_no_reencode` | +0.131 | [+0.099, +0.161] | 348/165 | **SIG** |
| `frozen_graph_history` − `graph_once_then_text` | +0.135 | [+0.105, +0.165] | 340/151 | **SIG** |
| `frozen_graph_history` − `question_only` | +0.138 | [+0.109, +0.168] | 341/148 | **SIG** |
| **`graph_once_then_text` − `question_only`** | **+0.003** | **[−0.024, +0.031]** | 199/195 | **ns — resolved null, tight CI** |

**Subset re-run:**

| Comparison | Validation (n=825) | Test (n=410) |
|---|---|---|
| `predicted_updated_graph` − `oracle_updated_graph` | +0.011 [−0.025, +0.047] ns | −0.024 [−0.076, +0.027] ns |
| `predicted_updated_graph` − `cached_no_reencode` | +0.138 [+0.097, +0.179] **SIG** | +0.117 [+0.059, +0.176] **SIG** |
| `predicted_updated_graph` − `frozen_graph_history` | +0.143 [+0.105, +0.182] **SIG** | +0.110 [+0.054, +0.168] **SIG** |
| `predicted_updated_graph` − `shuffled_graph` | +0.141 [+0.099, +0.182] **SIG** | +0.117 [+0.063, +0.173] **SIG** |
| `graph_once_then_text` − `question_only` | +0.017 [−0.017, +0.051] ns | −0.005 [−0.059, +0.046] ns |
| `frozen_graph_history` − `graph_once_then_text` | +0.109 [+0.074, +0.145] **SIG** | +0.127 [+0.071, +0.183] **SIG** |

### Accuracy by turn index — 4-step set (n=350 per cell)

| Condition | Turn 0 | Turn 1 | Turn 2 | Turn 3 |
|---|---|---|---|---|
| `oracle_updated_graph` | 0.437 | 0.543 | 0.326 | **0.483** |
| `predicted_updated_graph` | 0.374 | 0.563 | 0.346 | **0.491** |
| `frozen_graph_history` | 0.229 | 0.500 | 0.291 | **0.220** |
| `cached_no_reencode` | 0.194 | 0.503 | 0.334 | **0.234** |
| `shuffled_graph` | 0.180 | 0.494 | 0.283 | 0.177 |
| `graph_once_then_text` | 0.197 | 0.154 | 0.103 | 0.246 |
| `question_only` | 0.160 | 0.240 | 0.157 | 0.131 |

## Interpretation

1. **`predicted_updated_graph` = `oracle_updated_graph`, now the most tightly-bounded result in this project.** Diff −0.4pp, CI [−3.1pp, +2.4pp] at n=1,400 — this is as close to a definitive "these are the same" as a preliminary experiment gets. The realistic, self-directed loop (extract edit from text → apply → re-encode) performs identically to being handed the gold edit, because edit extraction is 100% execution-equivalent across every evaluation run in this project so far (410–1,400 turns per run, 5 independent samples total across two days).
2. **The one-time-exposure question is now resolved, not just suggestive.** `graph_once_then_text` vs `question_only`: diff +0.3pp, CI [−2.4pp, +3.1pp] at n=1,400 (was +2.5pp with a wider, less centered CI on the smaller sample). A single early graph exposure provides no detectable lasting benefit through subsequent turns — this was the specific comparison the larger sample was run to firm up, and it now has.
3. **Turn-by-turn divergence holds up at 2x the sample size.** By turn 3, `frozen_graph_history`/`cached_no_reencode` sit at 0.22–0.23 (barely above the 0.131–0.172 question-only-ish floor), while `oracle`/`predicted_updated_graph` hold at 0.48–0.49 — the same qualitative pattern as the smaller 2026-08-22 run, now backed by n=350 per turn-index cell instead of n=150.
4. Turn 1's cross-condition spike (present again here: `frozen_graph_history` 0.500, `cached_no_reencode` 0.503, `shuffled_graph` 0.494, `question_only` 0.240 — all elevated relative to their own turn 0/2/3) replicates from the previous run and is shared even by conditions that shouldn't carry graph information (`shuffled_graph`, `question_only`), reinforcing that it's a property of which reasoning types/queries land at turn index 1 in this generator, not a modeling effect. Still not investigated in depth — flagged again as a caveat.
5. Absolute accuracy numbers shifted a few points from the 2026-08-22 unbatched run (e.g. `oracle_updated_graph` on the 4-step set: 0.452 → 0.447) — expected, and already documented as a consequence of batched bf16 generation not being bit-identical to unbatched (see "Code changes"). The *comparisons* (which is what the hypothesis rests on) are stable across both runs.

## Caveats

- Uncommitted code (batching changes on top of the already-uncommitted predict_edit fix from earlier today).
- Not the full 1,000/2,000-session locked splits — this is "a few hundred samples," chosen as one iteration up from the initial ~150-session subset, in line with the compute-conscious approach agreed for prelim experiments.
- Batched vs. unbatched generation on this model is not bit-reproducible (bf16 non-associativity); treat any single accuracy number as having a rounding-noise floor of a percentage point or two, and rely on the paired comparisons (which are internally consistent within a run) for actual claims.
- Turn-1 cross-condition spike (point 4) still unexplained.

## Decision / next steps

The core hypothesis and both follow-up questions (does self-directed match oracle; does one-time graph exposure help) are now answered with the tightest evidence this project has produced. Next steps:
1. Commit all code changes from today (predict_edit fix + `graph_once_then_text` condition + full batching) — three preliminary runs deep on uncommitted code.
2. Repo cleanup: removed the one-off `outputs/overnight_driver.sh` wait-loop script and its logs, now fully superseded by `evaluate`'s native batching + progress logging.
3. Full 1,000/2,000-session run remains optional — useful only for a final reportable effect size, not for answering any open yes/no question at this point.
