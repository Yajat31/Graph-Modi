# Preliminary result: does updating + re-encoding the graph beat a stale graph?

**Date:** 2026-08-21 (overnight run, completed 2026-08-22)
**Run ID:** `llama3_1_8b_projector-20260821-eval_subset`
**Configs:** [`configs/llama3_1_8b_projector.yaml`](../../../../configs/llama3_1_8b_projector.yaml) (pretrain-gnn + train-projector), [`configs/llama3_1_8b_eval_subset.yaml`](../../../../configs/llama3_1_8b_eval_subset.yaml) (evaluate)
**Git commit:** uncommitted — `src/graph_modi/cli.py`, `src/graph_modi/models/tea_glm.py`, `src/graph_modi/training.py` were modified during this run (see "Code changes" below); the two configs above are new/untracked.

## Purpose / hypothesis

Core project question: does feeding the LLM an **updated, re-encoded graph** for the current turn outperform feeding it a **stale initial graph plus text history** and leaving it to the LLM to track what changed? This is a cheap, compute-constrained preliminary check before committing to a full 3,000-session evaluation.

## Setup

- **Backbone:** `meta-llama/Meta-Llama-3.1-8B-Instruct` (local copy at `/home/arihantr/models/Meta-Llama-3.1-8B-Instruct`), bf16, frozen throughout.
- **Graph encoder:** GraphSAGE, 8 layers, hidden/output dim 4096 (matched to the LLM's hidden size), sum aggregation, LayerNorm after every layer. Pretrained via `pretrain-gnn` for 20 epochs on the full `datasets/watts_strogatz_metro_v1` train split (27,500 turn-examples), conditioned on the query's source/target node (not just a whole-graph mean pool — see "Code changes"). Final training accuracy 52.5% (label space of 13 answers, weighted majority baseline ≈ 33.7%). Checkpoint: `outputs/llama3_1_8b/gnn/` (metadata copied here as `gnn_metadata.json`).
- **Projector:** single linear layer, 4096 → 8×4096 (8 graph prefix tokens), ~134M params — matches the TEA-GLM paper's design (a single linear projector from GNN-output-dim = LLM-hidden-dim), not the repo's earlier 2-layer 256→512→8×4096 MLP.
- **Projector training:** `train-projector`, 5 epochs, effective batch size 16, lr 5e-4, bf16, full answer-sequence loss through `inputs_embeds`. Final mean loss 0.5745. Checkpoint: `outputs/llama3_1_8b/projector/checkpoint-final/` (metadata copied here as `projector_metadata.json`).
- **Evaluation subset:** 100 validation + 50 test sessions (first N lines of the full 1,000/2,000-session splits from `datasets/watts_strogatz_metro_v1`), chosen to fit a compute-constrained overnight check rather than the full 3,000-session set. `evaluation.json` copied here in full (summary + per-turn rows).

## Code changes made to get a working run

The codebase had never actually trained/evaluated a real (non-mock) backend before this run. Several bugs surfaced and were fixed along the way (all in the uncommitted diff):

1. **Projector was a 256-dim-input 2-layer MLP**, not matching the TEA-GLM paper's single-linear, LLM-hidden-dim-input design — fixed via a `projector_num_layers` config knob (default 1) and setting `graph_hidden_size: 4096`.
2. **Stage-1 GNN pretraining never saw the query's source/target node**, only a whole-graph mean pool — made most `shortest_path`/`reachability` labels structurally unpredictable. Fixed by conditioning the classifier on `[pooled, source_node_repr, target_node_repr]`.
3. **Stage-1 pretraining ran on CPU only**; added CUDA device placement, batching (`gnn_batch_size`), and a true block-diagonal batched GraphSAGE forward pass (merges a batch of graphs into one disjoint graph — mathematically identical to per-graph forward, verified numerically, ~5x speedup on top of GPU placement).
4. **Mean aggregation + 4 layers plateaued at ~36% train accuracy** (roughly the per-reasoning-type majority baseline) — switched to sum aggregation (preserves count/distance signal) with 8 layers (covers the observed max shortest-path length of 7 hops) plus LayerNorm after every layer and gradient clipping (sum aggregation without normalization diverged at 8 layers before this fix).
5. **`evaluate`/`run-session` never moved the model to GPU** (`_tea_model()` in `cli.py`) — was silently running on CPU.
6. **Once on GPU, `evaluate`/`run-session` crashed with a dtype mismatch** (fp32 GNN/projector vs bf16 LLM) — training worked because Accelerate's `mixed_precision="bf16"` autocasts the forward pass automatically; the inference-only path has no such wrapper. Fixed with an explicit `torch.autocast` context in `TEAGLM.generate()` / `generate_text_only()` / `forward()`.

## Results

### Per-condition answer accuracy (independent 95% CIs)

| Condition | Validation (n=100 sessions) | Test (n=50 sessions) |
|---|---|---|
| `oracle_updated_graph` | 0.422 [0.365, 0.481] | 0.452 [0.370, 0.536] |
| `predicted_updated_graph` | 0.302 [0.251, 0.358] | 0.356 [0.280, 0.439] |
| `shuffled_graph` | 0.302 [0.251, 0.358] | 0.289 [0.219, 0.370] |
| `frozen_graph_history` | 0.298 | 0.296 |
| `cached_no_reencode` | 0.255 [0.207, 0.309] | 0.237 [0.173, 0.315] |
| `serialized_initial_history` | 0.207 [0.164, 0.259] | 0.252 [0.186, 0.331] |
| `token_matched_history` | 0.193 | 0.200 |
| `question_only` | 0.153 [0.115, 0.200] | 0.156 [0.104, 0.226] |
| `serialized_current_graph` | 0.142 [0.106, 0.188] | 0.163 [0.110, 0.234] |
| `tool_solver` (ceiling) | 1.000 | 1.000 |

`exact_graph_state` = 1.0 for every condition except `predicted_updated_graph` (0.0, expected — see limitations). `execution_equivalent_edit_accuracy` = 0.0 for `predicted_updated_graph` on both splits.

### Paired comparison (the actual test of the hypothesis)

Every session runs under every condition with the same seed, so the honest comparison is **paired**, not independent CIs (which ignore that per-turn difficulty is shared across conditions). Paired bootstrap (10,000 resamples) on `answer_correct`, matched by `(session_id, turn_index)`:

**Validation (n=275 paired turns):**

| Comparison | Δ accuracy | 95% CI | Win / Lose / Tie | Significant? |
|---|---|---|---|---|
| `oracle_updated_graph` − `frozen_graph_history` | +0.124 | [+0.058, +0.189] | 61 / 27 / 187 | **Yes** |
| `oracle_updated_graph` − `cached_no_reencode` | +0.167 | [+0.102, +0.233] | 71 / 25 / 179 | **Yes** |
| `oracle_updated_graph` − `serialized_initial_history` | +0.215 | [+0.142, +0.287] | 87 / 28 / 160 | **Yes** |
| `oracle_updated_graph` − `shuffled_graph` | +0.120 | [+0.047, +0.193] | 69 / 36 / 170 | **Yes** |
| `oracle_updated_graph` − `question_only` | +0.269 | [+0.196, +0.342] | 98 / 24 / 153 | **Yes** |

**Test (n=135 paired turns):**

| Comparison | Δ accuracy | 95% CI | Win / Lose / Tie | Significant? |
|---|---|---|---|---|
| `oracle_updated_graph` − `frozen_graph_history` | +0.156 | [+0.059, +0.252] | 35 / 14 / 86 | **Yes** |
| `oracle_updated_graph` − `cached_no_reencode` | +0.215 | [+0.119, +0.311] | 38 / 9 / 88 | **Yes** |
| `oracle_updated_graph` − `serialized_initial_history` | +0.200 | [+0.104, +0.296] | 38 / 11 / 86 | **Yes** |
| `oracle_updated_graph` − `shuffled_graph` | +0.163 | [+0.074, +0.259] | 33 / 11 / 91 | **Yes** |
| `oracle_updated_graph` − `question_only` | +0.296 | [+0.200, +0.393] | 48 / 8 / 79 | **Yes** |

Every comparison is significant, in the same direction, independently on both splits.

## Interpretation

1. **Core hypothesis supported**: re-encoding the updated graph beats every stale/static alternative tested — beating not just text-based history (`frozen_graph_history`, `serialized_initial_history`) but also `cached_no_reencode`, which applies the edit to the tracked state but doesn't refresh the graph tokens the LLM sees. That isolates the effect specifically to *re-encoding*, not just "an edit happened somewhere."
2. **The trained projector encodes real signal**, not just "having any graph tokens helps" — `oracle_updated_graph` clearly beats `shuffled_graph` (wrong graph's tokens).
3. **`predicted_updated_graph` (self-directed edit extraction) doesn't work yet** — `execution_equivalent_edit_accuracy` is 0.0. This is expected, not a modeling failure: Stage-2 training only taught the projector the question-answering prompt format (`render_question(...) → gold_answer`); the frozen LLM was never shown a single example of the "return one graph edit" format `predict_edit()` uses, and the zero-shot prompt gives it no schema or example to work from. `predicted_updated_graph`'s accuracy (0.30–0.36) sitting between `cached_no_reencode` and `oracle_updated_graph` is consistent with "a mostly-wrong edit still nudges things in the right direction sometimes," not genuine edit-prediction skill.
4. Mildly surprising, worth re-checking at scale rather than reading into at n=150: `serialized_current_graph` (raw graph text) scores below even `question_only` on validation — graph tokens outperforming serialized text was not a given going in.

## Caveats

- Only 150 of 3,000 available sessions (compute-constrained preliminary check, not the full evaluation).
- Run against **uncommitted code** — see git status above. Commit before treating this as a citable/reproducible result.
- `frozen_graph_history` and `token_matched_history` accuracy numbers above don't have CIs recomputed here (pulled from `evaluation.json`'s summary directly); the paired bootstrap table is the number that actually matters for the hypothesis test.

## Decision / next steps

Preliminary evidence clearly supports the core hypothesis at negligible compute cost. Per [`documents/experiments/README.md`](../../README.md), a full 3,000-session run would firm up effect-size estimates for reporting, but is not needed to answer "does this work" — that's already answered. Suggested next steps, in order:
1. Commit the code fixes listed above (currently uncommitted).
2. Decide whether to invest in fixing self-directed edit prediction (`predicted_updated_graph`) — currently the practical, no-oracle version of this framework, and it isn't working yet.
3. When compute allows, run the full 3,000-session evaluation (all 10 conditions, or the 7 core conditions from this run) for a reportable number.
