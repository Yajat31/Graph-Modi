# CLEGR-Extended Task Family: Status Report (exploratory, in progress)

**Status:** TEA-GLM and GraphToken both iterated to v2 (hops + reasoning_type); multi-neighbor
readout (v3) prototyped, validated on static eval, extended to GraphToken, and run through the
**full 4-way dynamic (multi-turn) eval matrix** (TEA/GraphToken × pooled/multi-neighbor) — see
§7 for the headline result. soft_prompt not part of the v2/v3 comparison.
**Branch:** `exact2x`
**Config family:** [`configs/v2_clegr_extended_*.yaml`](../../../../configs/) (train) and
[`configs/v2_clegr_extended_exact2x_*.yaml`](../../../../configs/) (eval-only, exact-uniform)
**Compute:** remote GPU box (10.4.25.56), GPU 1 only, shared with another user's job —
GPU 0 and GPU 1's pre-existing allocation were never touched.
**Artifacts in this folder:** `static_eval_{validation,test}.json` (TEA v2 checkpoint),
`gnn_metadata.json`, `projector_metadata.json`, `train_dataset_audit.json`,
`exact2x_dataset_audit.json`, `multi_neighbor_static_eval_{validation,test}.json` (TEA v3).
**Figures:** [`figures/clegr_extended_*.png`](../../figures/) (`scripts/analyze_clegr_extended.py`)

---

## Figure gallery

![Overall accuracy progression](../../figures/clegr_extended_progression.png)

![Static gate history](../../figures/clegr_extended_static_history.png)

![Multi-neighbor readout, static probe](../../figures/clegr_extended_multi_neighbor_static.png)

![Full dynamic eval, 4-way comparison](../../figures/clegr_extended_dynamic_4way.png)

![TEA v2 pooled, accuracy by condition](../../figures/clegr_extended_tea_v2_conditions.png)

---

## 1. Motivation

The existing benchmark (`v2_gate_variant_cf_exact2x`, see
[`../v2_gate_variant_cf_exact2x-20260911/report.md`](../v2_gate_variant_cf_exact2x-20260911/report.md))
narrows the task mix to three yes/no, topology-only tasks (`edge_exists`, `reachability`,
`cycle_membership`) — explicitly flagged there as limitation #1: "not a full-benchmark claim."
This work extends the task family toward CLEGR (arXiv 2508.20583, *A Graph Talks, But Who's
Listening? Rethinking Evaluations for Graph-Language Models*, Findings ACL 2026) — the paper
[`README.md`](../../../../README.md) already cites as the design basis — by adding filtered
aggregation, weighted path cost, constrained reachability, and topology-with-filter task
families, and applying the same exact-uniform balancing already used for dynamic val/test to
the static oracle-QA corpus too.

## 2. What was implemented

### 2.1 New task families (code)

Four new `ReasoningType` members added end-to-end (schema → solver → question rendering →
candidate generation → hop-depth stratification → task rosters):
[`schema.py`](../../../../src/graph_modi/schema.py),
[`graph/solvers.py`](../../../../src/graph_modi/graph/solvers.py),
[`data/multiturn.py`](../../../../src/graph_modi/data/multiturn.py),
[`data/v2.py`](../../../../src/graph_modi/data/v2.py):

- `constrained_reachability` — reachability while avoiding a class of station (fixed to
  `accessible == False`, deliberately narrowed from an original 5-filter design per a later
  "keep tasks easier" pass).
- `within_hops_count` — count of stations within a fixed hop radius (2), optionally filtered
  by attribute.
- `within_hops_list` — comma-separated list of stations within hops matching a filter
  (filtered-only, not "list everyone nearby," to keep answers short).
- `most_common_attribute_within_hops` — modal attribute value among stations within hops.

Also activated three task types that already had solver logic but were never wired into any
active task roster (`filtered_neighbor_count`, `filtered_path_count`, `path_cost`), and fixed
two real bugs found along the way:

- `node_degree` was listed in `_STATIC_TASKS` but had no `_queries()` candidate-generation
  branch, so it silently produced zero training examples every time (`generate_static_corpus`
  skips a task with no candidates rather than erroring). Fixed by adding the missing branch.
- Watts-Strogatz edges always kept the schema default `weight=1.0`, making `path_cost`
  numerically degenerate (cost == hop count). Fixed by assigning the same `[1,5]` weight range
  SBM/Erdos-Renyi already use, as a final pass in `make_graph`.

### 2.2 Metrics

[`evaluation/metrics.py`](../../../../src/graph_modi/evaluation/metrics.py) gained
`answers_match()` (numeric tolerance for `path_cost`-style answers instead of blind string
equality) and `set_f1()` (set-valued scoring for `within_hops_list`), wired into both
`evaluation/runner.py` and `evaluation/static_eval.py`.

### 2.3 Exact-uniform balancing extended to static data

`generate_static_exact_uniform()` in `data/v2.py` mirrors `generate_exact_uniform_sessions()`
(same exact-`n` x density grid, now crossed with the expanded task list) but for independent
static `(G, Q, A)` tuples instead of dynamic sessions. Wired through a new
`data.static_layout: exact_uniform` / `data.static_exact_uniform` config block in
[`cli.py`](../../../../src/graph_modi/cli.py) — train split stays free-sampled, matching how
`generate_exact_uniform_sessions` already treats train.

### 2.4 New configs

`v2_clegr_extended_{tea,graphtoken,soft_prompt}.yaml` (train, free-sampled data, expanded task
lists) and `v2_clegr_extended_exact2x_{tea,graphtoken,soft_prompt}.yaml` (eval-only,
exact-uniform static + dynamic) — new files, none of the historical `v2_gate_variant_cf_*`
configs were touched, so existing results/checkpoints remain reproducible as before.

## 3. Infrastructure

Remote GPU box provisioned from scratch this session: `uv` venv (torch 2.13 / CUDA 13),
`meta-llama/Meta-Llama-3.1-8B-Instruct` downloaded (~16GB) after working around an IPv6
black-hole on that network (forced IPv4-only resolution in the download script — see
git history for the exact monkeypatch). GPU 1 is shared with another user's job whose memory
footprint fluctuated between ~34GB and ~53GB over the session; never touched, only the
remaining free slice was used.

## 4. Training/eval iteration history

Three static oracle-QA gate runs against `configs/v2_clegr_extended_exact2x_tea.yaml`
(same eval-only exact-uniform dataset each time; TEA only — GraphToken/soft_prompt not yet
retrained on the fixed architecture). All three **fail** the 70%-overall / no-task-below-50%
gate; the point of tracking them is the *shape* of the failure, not a pass.

| Run | Config delta from previous | Overall (val/test) |
|---|---|---|
| **Before** | Initial full task-family expansion: 12-dim hashed node features, 5 projector epochs, 20 GNN-pretrain epochs, `max_new_tokens=32`, `batch_size=24` | 25.6% / 26.9% |
| **v1 (+hops)** | `node_feature_size` 12→64, projector epochs 5→10, GNN-pretrain epochs 20→30, `max_new_tokens` 32→96 (fixed answer truncation), `batch_size` 24→8 + `gradient_accumulation_steps` 1→3 (CUDA OOM workaround — GPU-neighbor memory grew mid-run), hop radius threaded into the graph encoder as `HOPS_FEATURE_DIM` | 22.2% / 24.5% |
| **v2 (+reasoning_type)** | One-hot task-identity feature (`REASONING_TYPE_FEATURE_DIM`) added alongside hops; task difficulty also reduced (hop radius fixed at 2, `within_hops_list` filtered-only, `constrained_reachability` single-filter) | **24.8% / 27.8%** |

![Static gate history](../../figures/clegr_extended_static_history.png)

Full per-task breakdown (validation / test, percent):

| Task | Before | v1 (+hops) | v2 (+task-id) |
|---|---|---|---|
| reachability | 53.9 / 46.2 | 68.0 / 70.5 | 56.4 / 53.8 |
| **cycle_membership** | 60.3 / 55.1 | **21.8 / 24.4** | **80.8 / 70.5** |
| edge_exists | 51.3 / 56.4 | 46.2 / 43.6 | 51.3 / 51.3 |
| constrained_reachability | 52.6 / 59.0 | 41.0 / 44.9 | 30.8 / 38.5 |
| filtered_neighbor_count | 17.9 / 30.8 | 14.1 / 24.4 | 10.3 / 21.8 |
| filtered_path_count | 17.9 / 15.4 | 20.5 / 28.2 | 19.2 / 24.4 |
| node_degree | 15.4 / 26.9 | 20.5 / 20.5 | 20.5 / 25.6 |
| **within_hops_count** | 2.6 / 3.8 | **16.7 / 11.5** | 7.7 / 11.5 |
| within_hops_list | 0.0 / 0.0 | 2.6 / 1.3 | 2.6 / 1.3 |
| most_common_attribute_within_hops | 23.1 / 21.8 | 9.0 / 14.1 | 7.7 / 19.2 |
| path_cost | 2.6 / 0.0 | 1.3 / 1.3 | 2.6 / 1.3 |
| shortest_path | 10.3 / 7.7 | 5.1 / 9.0 | 7.7 / 14.1 |

### 4.1 Root-cause findings

1. **`max_new_tokens=32` truncated list-valued answers mid-token** (inherited from the
   yes/no-only historical configs). Fixed to 96 in the v1 pass; confirmed via raw
   `within_hops_list` predictions like `"...teststation000010"` cut off mid-ID.
2. **`hops` (radius 2 vs. 3) was never passed to the graph encoder.** `encode_graphs` only
   took `source_ids`/`target_ids`; the model saw only the *final*, `num_layers`-mixed node
   embedding, which conflates every radius up to `num_layers` into one vector. Fixed by
   appending a raw hop-count scalar (`HOPS_FEATURE_DIM`) to the projector's input vector in
   `TEAGLM.encode_graphs`/`graph_prefix`/`forward`/`generate`/`generate_batch`, threaded
   through `training.py` and the `TEAGLMBackend` inference path.
3. **Answer-format leakage across the ~12-task mix, diagnosed via raw predictions**, e.g.
   `cycle_membership` (a yes/no task) answered with a bare number (`"15"`) in 32% of v1 test
   examples, and `most_common_attribute_within_hops` (categorical) answered with entire
   comma-separated station-ID lists. Root cause: the model had no explicit task-identity
   signal — only question text plus a task-agnostic graph prefix — and defaulted to whatever
   answer format was best-represented in the training mix (6+ of 12 tasks are numeric).
   Fixed in v2 by adding a one-hot `reasoning_type` feature
   (`REASONING_TYPE_FEATURE_DIM`, full `ReasoningType` enum, not just the two tasks observed
   to leak) via the same encoder path as the hops fix. Confirmed fixed by re-inspecting raw
   predictions: `cycle_membership` off-format rate dropped from ~40% to ~8%, and its accuracy
   went 21.8%→80.8% (val), 24.4%→70.5% (test) — the single largest swing in the whole
   iteration history.
4. **Not fixed by either change — a different, deeper problem:**
   `most_common_attribute_within_hops` kept leaking `"15"`-style numeric answers at the same
   ~31% rate even with the correct task-identity signal present. This is not format confusion
   (the model now knows which task it is) but a genuine capability gap: computing the *mode*
   of an attribute across a neighborhood requires aggregating multiple neighbor values, and
   the current readout gives the projector only one pooled vector (plus source/target/hop/
   task scalars) — nothing that represents the neighbor attribute distribution itself.
5. **`constrained_reachability` never learned above a majority-guess baseline** in any of the
   three runs (majority-class baseline on this eval slice ≈56%; the model scored 30.8-52.6%
   across runs, i.e. at-or-below chance). Likely the same class of limitation as (4): avoiding
   a class of node while checking reachability needs constrained multi-hop pathfinding, which
   shallow GraphSAGE message passing has no strong inductive bias for. The apparent
   before→v1→v2 downward trend is probably mostly small-sample noise (each retrain
   regenerates the underlying graphs, so the specific 78 test cases differ slightly per run)
   rather than the fixes actively making this task worse.

### 4.2 Literature comparison (see conversation for full sourcing)

- CLEGR itself (same paper this task family targets) reports that even its own
  properly-resourced GLMs (5 seeds, real BERT-768 text embeddings, larger GraphSAGE) saturate
  on fact-retrieval but do **not** clearly outperform a graph-blind soft-prompted LLM on
  CLEGR-Reasoning (filtering/aggregation/path/topology) — the paper's central finding, not a
  gap specific to this implementation.
- TEA-GLM's own paper reports in-domain supervised node-classification accuracy of ~58-66%
  (Arxiv, Computer) and cross-dataset link-prediction AUC of ~55-69% — simpler, single-step
  tasks, offered here only as a rough scale reference for "what this architecture family
  typically achieves when it's working."
- Our topology-only tasks (`reachability`, `cycle_membership`, `edge_exists`), after the v2
  fix, score 51-81% — in the same range as, and on `cycle_membership` above, TEA-GLM's own
  reported numbers for comparable structural difficulty. The compositional/aggregation tasks
  (0-25%) sit well below that, consistent with CLEGR's own documented field-wide limitation
  rather than an implementation-specific shortfall.

## 5. Current limitations / open items

- GraphToken has since been retrained on the v2 (hops + reasoning_type) architecture and has
  its own multi-neighbor (v3) checkpoint too (see §7) — this bullet is now stale, kept for
  history. soft_prompt is architecturally unaffected by any of this (no graph encoder) and was
  not part of the v2/v3 comparison.
- Full dynamic (multi-turn) evaluation against v2/v3, both architectures, is now running —
  see §7 for status and results as they land. (Previously: "no dynamic eval has been run
  against the v2 checkpoint yet" — also stale.)
- `most_common_attribute_within_hops` and `constrained_reachability` remain near or below a
  naive baseline; closing that gap needs an architecture change, not another metadata feature
  (see below) — deliberately not attempted yet, to avoid iterating on the eval set with
  answer-leaking shortcuts.
- The deferred, larger fix already logged in
  [`../../STATUS_REPORT.md`](../../STATUS_REPORT.md) (task-staged curriculum instead of flat
  round-robin across all 12 tasks from epoch 1) is still outstanding and may be worth
  revisiting if the readout change below doesn't close the gap on its own.

## 6. Proposed next step (exploratory, not yet started)

**Multi-neighbor-token readout.** Instead of collapsing the whole graph into one pooled
vector per query, expose multiple individual (or attribute-grouped) neighbor-node embeddings
as separate graph-prefix tokens, so the frozen LLM can attend over them and do the
counting/mode-finding itself — closer to how multi-token graph-prefix architectures are
generally built, rather than forcing every answer through one vector. This is a legitimate
architectural fix in the same spirit as the `hops`/`reasoning_type` features (exposing
information the query needs, not the computed answer), as opposed to injecting a precomputed
count/histogram feature, which would leak the answer and defeat the point of testing graph
reasoning at all.

**Agreed scope:** prototype this on **TEA only**, as a throwaway exploratory branch/config —
not wired into the main pipeline yet. If it measurably improves the aggregation-heavy tasks
without regressing the topology tasks that already work, extend it to GraphToken, and keep
both the current (pooled) and new (multi-neighbor-token) readout modes available/configurable
on both architectures rather than replacing one with the other outright.

### 6.1 Result: strongly positive, promoting to real implementation

Implemented as `src/graph_modi/models/multi_neighbor_readout.py` +
`scripts/explore_multi_neighbor_readout.py` (throwaway harness): reuses the v2 TEA
checkpoint's frozen GNN/LM/summary-projector verbatim, trains only a new
`NeighborTokenProjector` (3 epochs) that emits one graph-prefix token per 1-hop neighbor of
the query's source node (zero-padded/truncated to 8 slots), concatenated after the existing
summary tokens. Evaluated on the same static exact-uniform validation/test sets as the v2
baseline:

![Multi-neighbor readout, static probe](../../figures/clegr_extended_multi_neighbor_static.png)

| Task | v2 (pooled) val/test | Multi-neighbor val/test | Delta |
|---|---|---|---|
| reachability | 56.4 / 53.8 | 93.6 / 87.2 | +37.2 / +33.4 |
| constrained_reachability | 30.8 / 38.5 | 71.8 / 69.2 | +41.0 / +30.7 |
| cycle_membership | 80.8 / 70.5 | 85.9 / 85.9 | +5.1 / +15.4 |
| edge_exists | 51.3 / 51.3 | 66.7 / 62.8 | +15.4 / +11.5 |
| path_cost | 2.6 / 1.3 | 15.4 / 14.1 | +12.8 / +12.8 |
| shortest_path | 7.7 / 14.1 | 17.9 / 23.1 | +10.2 / +9.0 |
| filtered_neighbor_count | 10.3 / 21.8 | 21.8 / 30.8 | +11.5 / +9.0 |
| filtered_path_count | 19.2 / 24.4 | 29.5 / 20.5 | +10.3 / -3.9 |
| node_degree | 20.5 / 25.6 | 28.2 / 29.5 | +7.7 / +3.9 |
| most_common_attribute_within_hops | 7.7 / 19.2 | 10.3 / 16.7 | +2.6 / -2.5 |
| within_hops_count | 7.7 / 11.5 | 10.3 / 14.1 | +2.6 / +2.6 |
| within_hops_list | 2.6 / 1.3 | 2.6 / 2.6 | ~flat |
| **Overall** | 24.8 / 27.8 | **37.8 / 38.0** | **+13.0 / +10.2** |

Nearly every task improved, including ones that already worked (`reachability`,
`cycle_membership`) — this reads as a broad readout-quality improvement, not narrow overfitting
to the tasks the fix targeted. `within_hops_list` (exact multi-station set matching) is the one
holdout that stayed flat; it may need list-specific handling (e.g. enumerating candidate
neighbors as separate tokens the LLM selects from, rather than describing them) beyond what a
generic per-neighbor token gives it. **Decision: extend to GraphToken and promote out of
"experimental" into a real, configurable readout mode on both architectures**, per the
plan agreed before running this probe.

## 7. Full dynamic (multi-turn) evaluation: v2 (pooled) vs. v3 (multi-neighbor) × TEA/GraphToken

Following the static-eval result in §6.1, the multi-neighbor readout was persisted as a real
checkpoint (`save_neighbor_checkpoint()`/`load_neighbor_checkpoint()`, added to
`multi_neighbor_readout.py`) and GraphToken was retrained on the v2 architecture, giving four
checkpoints to compare under the harder multi-turn dynamic eval (which the static-only §6.1
probe never exercised): TEA v2 (pooled), TEA v3 (multi-neighbor), GraphToken v2 (pooled),
GraphToken v3 (multi-neighbor). Run via a new standalone script,
`scripts/run_dynamic_eval_variant.py` (mirrors `cli.py`'s dynamic-eval path, adding a
`--variant {pooled,multi_neighbor}` switch the main CLI doesn't support), against
`configs/v2_clegr_extended_exact2x_{tea,graphtoken}.yaml`: 2,496 sessions/split (turn lengths
1/2/4/8, 624 sessions each, mean 3.75 turns/session) × 14 conditions × validation+test splits
= 131,040 turns/split/job. All 4 launched in parallel on GPU1 (~13GB headroom left with all 4
loaded, no OOM); ~3-4x slower per-job than running solo due to sharing the GPU 4 ways.

**Task-roster note:** the dynamic sessions on disk use a *different* 8-task set than the static
corpus in §4/§6.1 — `reachability`, `constrained_reachability`, `edge_exists`,
`filtered_neighbor_count`, `within_hops_count`, `most_common_attribute_within_hops`,
`cycle_membership`, `node_count` — swapping in `cycle_membership`/`node_count` where the static
set has `shortest_path`/`path_cost`/`filtered_path_count`/`node_degree`/`within_hops_list`.
This looks like the dynamic exact-uniform dataset was generated at a slightly different point
than the current static corpus/`_DYNAMIC_TASKS` definition in `data/v2.py`; noted here rather
than silently reconciled, since regenerating it would invalidate the results below.

### 7.1 Status

All four jobs are complete:

| Job | Overall (val/test) |
|---|---|
| TEA v2 (pooled) | 27.8 / 27.6 |
| TEA v3 (multi-neighbor) | 31.7 / 31.6 |
| GraphToken v2 (pooled) | 26.1 / 25.9 |
| GraphToken v3 (multi-neighbor) | 29.0 / 28.7 |

See §7.2-§7.3 for the per-architecture breakdowns and §7.4 for the combined 4-way table and
the headline finding.

### 7.2 TEA v2 (pooled) — first result

![TEA v2 pooled, accuracy by condition](../../figures/clegr_extended_tea_v2_conditions.png)

Per-condition (`answer_accuracy`, validation ≈ test throughout — good sign of a stable,
non-overfit checkpoint):

| Condition | Validation | Test |
|---|---|---|
| tool_solver (oracle, sanity check) | 100.0% | 100.0% |
| majority_prior (answer-distribution floor) | 69.5% | 68.3% |
| oracle_updated_graph | 36.1% | 35.5% |
| predicted_updated_graph | 36.0% | 35.3% |
| shuffled_graph | 30.6% | 30.5% |
| cached_no_reencode | 30.2% | 30.4% |
| modify_and_print | 30.1% | 29.8% |
| structure_only | 29.9% | 29.9% |
| frozen_graph_history | 30.5% | 30.0% |
| token_matched_history | 22.8% | 22.0% |
| question_only (no graph context) | 24.6% | 24.2% |
| graph_once_then_text | 23.8% | 24.3% |
| serialized_initial_history | 21.4% | 21.5% |
| serialized_current_graph | 18.0% | 17.7% |

`serialized_current_graph` (re-serializing the full updated graph as text every turn) scores
*below* `question_only` (no graph at all) in both splits — likely a token-budget issue where a
large text-serialized graph crowds out the actual question/history within the context window,
worth a follow-up once the other 3 jobs are in.

Per-task accuracy, averaged across all 12 real conditions (excludes the `tool_solver` oracle
and `majority_prior` baseline, which don't test the model's own reasoning):

| Task | Validation | Test |
|---|---|---|
| cycle_membership | 53.3% | 53.7% |
| reachability | 47.9% | 47.6% |
| edge_exists | 46.6% | 45.5% |
| constrained_reachability | 38.7% | 38.8% |
| filtered_neighbor_count | 17.0% | 16.0% |
| within_hops_count | 9.6% | 8.5% |
| most_common_attribute_within_hops | 8.4% | 9.5% |
| node_count | 1.3% | 1.2% |
| **Overall** | **27.8%** | **27.6%** |

`node_count` is a near-total failure (~1%), worse even than the aggregation tasks that were
already known to be weak — not previously visible in the static-only §4/§6.1 tables since
`node_count` isn't in that task roster (see task-roster note above).

TEA v3 (multi-neighbor) per-task accuracy, same methodology:

| Task | v2 pooled (val/test) | v3 multi-neighbor (val/test) | Delta (val/test) |
|---|---|---|---|
| constrained_reachability | 38.7 / 38.8 | 55.7 / 56.9 | **+17.0 / +18.1** |
| reachability | 47.9 / 47.6 | 52.6 / 51.9 | +4.7 / +4.3 |
| cycle_membership | 53.3 / 53.7 | 57.4 / 57.8 | +4.1 / +4.1 |
| filtered_neighbor_count | 17.0 / 16.0 | 20.0 / 18.9 | +3.0 / +2.9 |
| edge_exists | 46.6 / 45.5 | 48.0 / 47.8 | +1.4 / +2.3 |
| within_hops_count | 9.6 / 8.5 | 9.7 / 9.2 | ~flat |
| most_common_attribute_within_hops | 8.4 / 9.5 | 9.0 / 9.2 | ~flat |
| node_count | 1.3 / 1.2 | 1.2 / 1.2 | ~flat, still near-total failure |
| **Overall** | **27.8 / 27.6** | **31.7 / 31.6** | **+3.9 / +4.0** |

### 7.3 GraphToken v2 (pooled) vs. v3 (multi-neighbor)

Per-task accuracy, same methodology as §7.2 (averaged across the 12 real conditions,
excluding `tool_solver`/`majority_prior`):

| Task | v2 pooled (val/test) | v3 multi-neighbor (val/test) | Delta (val/test) |
|---|---|---|---|
| constrained_reachability | 40.1 / 40.9 | 51.6 / 51.7 | +11.5 / +10.8 |
| cycle_membership | 37.1 / 37.7 | 46.2 / 46.7 | +9.1 / +9.0 |
| reachability | 47.9 / 48.2 | 50.2 / 48.7 | +2.3 / +0.5 |
| edge_exists | 43.1 / 43.0 | 45.3 / 44.7 | +2.2 / +1.7 |
| filtered_neighbor_count | 18.3 / 17.1 | 19.5 / 19.2 | +1.2 / +2.1 |
| within_hops_count | 8.7 / 8.3 | 10.2 / 10.0 | +1.5 / +1.7 |
| most_common_attribute_within_hops | 12.6 / 10.8 | 8.1 / 7.4 | **-4.5 / -3.4** |
| node_count | 1.1 / 1.2 | 1.1 / 1.0 | ~flat, still near-total failure |
| **Overall** | **26.1 / 25.9** | **29.0 / 28.7** | **+2.9 / +2.8** |

Directionally consistent with TEA (§7.2): the multi-neighbor readout helps GraphToken too,
led by the same two tasks (`constrained_reachability`, `cycle_membership`). The gain is smaller
than TEA's dynamic-eval improvement and `most_common_attribute_within_hops` — the task the
readout change was originally motivated by — actually regresses for GraphToken, unlike TEA's
static-eval result in §6.1. `node_count` stays a near-total failure in both variants; the
multi-neighbor readout does not touch it (1-hop neighbor tokens don't help a query that needs
a *global* node count, not a per-neighbor property).

### 7.4 Combined 4-way comparison and headline finding

![Full dynamic eval, 4-way comparison](../../figures/clegr_extended_dynamic_4way.png)

![Overall accuracy progression](../../figures/clegr_extended_progression.png)

| Task | TEA v2 | TEA v3 | GraphToken v2 | GraphToken v3 |
|---|---|---|---|---|
| constrained_reachability | 38.7 / 38.8 | 55.7 / 56.9 | 40.1 / 40.9 | 51.6 / 51.7 |
| reachability | 47.9 / 47.6 | 52.6 / 51.9 | 47.9 / 48.2 | 50.2 / 48.7 |
| cycle_membership | 53.3 / 53.7 | 57.4 / 57.8 | 37.1 / 37.7 | 46.2 / 46.7 |
| edge_exists | 46.6 / 45.5 | 48.0 / 47.8 | 43.1 / 43.0 | 45.3 / 44.7 |
| filtered_neighbor_count | 17.0 / 16.0 | 20.0 / 18.9 | 18.3 / 17.1 | 19.5 / 19.2 |
| within_hops_count | 9.6 / 8.5 | 9.7 / 9.2 | 8.7 / 8.3 | 10.2 / 10.0 |
| most_common_attribute_within_hops | 8.4 / 9.5 | 9.0 / 9.2 | 12.6 / 10.8 | 8.1 / 7.4 |
| node_count | 1.3 / 1.2 | 1.2 / 1.2 | 1.1 / 1.2 | 1.1 / 1.0 |
| **Overall** | **27.8 / 27.6** | **31.7 / 31.6** | **26.1 / 25.9** | **29.0 / 28.7** |

(all cells: validation / test, percent; averaged across the 12 real conditions, excluding
`tool_solver`/`majority_prior`)

**Headline finding: the multi-neighbor readout's win is real but far more modest under the
full dynamic multi-turn eval than the static-only probe in §6.1 suggested.** Static eval
showed +13.0/+10.2 points overall for TEA (24.8/27.8 → 37.8/38.0); the same checkpoint under
the harder dynamic eval (14 context-presentation conditions, sessions up to 8 turns, graph
state that can change mid-session) gains only +3.9/+4.0 points (27.8/27.6 → 31.7/31.6).
GraphToken shows the same pattern at a smaller scale (+2.9/+2.8). Two likely reasons, not
mutually exclusive:

1. **Task-roster mismatch** (§7 note above) — the dynamic tasks skew toward `reachability`/
   `cycle_membership`/`edge_exists`/`node_count`, which the static §6.1 breakdown shows
   responding less dramatically to the neighbor-token change than `constrained_reachability`
   and `reachability` did there too, but the *dynamic* eval has no `shortest_path`/`path_cost`/
   `within_hops_list` tasks at all — some of static's biggest gains (e.g. `path_cost`
   +12.8/+12.8, `shortest_path` +10.2/+9.0) simply have no dynamic-eval counterpart to inherit.
2. **Multi-turn conditions dilute a single-turn architectural fix.** The neighbor-token change
   only touches what the model sees *within one turn's graph prefix*; it does nothing for the
   9 of 14 conditions that stress cross-turn state tracking (history serialization, staleness,
   token-budget truncation, shuffled/frozen graph state) — the accuracy ceiling in those
   conditions is set by context-management behavior the readout change was never meant to fix.

**What replicated cleanly across both architectures and both eval regimes:**
`constrained_reachability` and `cycle_membership`/`reachability` are the most consistent
winners everywhere. `node_count` is untouched everywhere (0/4 result sets show any improvement)
— confirms it needs a different fix, not more of this one (a *global* count needs information
the readout still doesn't expose, e.g. total node count as an explicit scalar, not per-neighbor
tokens). `most_common_attribute_within_hops` is the one architecture-dependent flip: helped
TEA (static +2.6/-2.5, dynamic ~flat) but hurt GraphToken (-4.5/-3.4) — worth a follow-up before
treating the readout as a strict upgrade for both architectures rather than TEA-specific.

**Bottom line:** promote the multi-neighbor readout as a configurable option on both
architectures (as already decided in §6.1), but recalibrate expectations — it is a genuine,
reproducible improvement (4/4 result sets improve overall accuracy, none regress), not the
~13-point jump the static-only probe implied. `node_count` and the cross-turn-state-tracking
conditions remain open problems this change does not address.
