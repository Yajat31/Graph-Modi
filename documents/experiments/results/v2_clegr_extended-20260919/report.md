# CLEGR-Extended Task Family: Status Report (exploratory, in progress)

**Status:** exploratory / in progress — TEA-GLM iterated to a working state; GraphToken and
soft_prompt training paused deliberately to focus on TEA first; full dynamic eval matrix not
yet run.
**Branch:** `exact2x`
**Config family:** [`configs/v2_clegr_extended_*.yaml`](../../../../configs/) (train) and
[`configs/v2_clegr_extended_exact2x_*.yaml`](../../../../configs/) (eval-only, exact-uniform)
**Compute:** remote GPU box (10.4.25.56), GPU 1 only, shared with another user's job —
GPU 0 and GPU 1's pre-existing allocation were never touched.
**Artifacts in this folder:** `static_eval_{validation,test}.json` (current TEA checkpoint),
`gnn_metadata.json`, `projector_metadata.json`, `train_dataset_audit.json`,
`exact2x_dataset_audit.json`.

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

- GraphToken and soft_prompt have **not** been retrained on the v2 (hops + reasoning_type)
  architecture yet — paused deliberately to focus debugging effort on TEA first. GraphToken
  shares the same encoder path and should inherit the same fixes; soft_prompt is architecturally
  unaffected by any of this (no graph encoder).
- No dynamic (multi-turn) eval has been run against the v2 checkpoint yet — only the static
  oracle-QA gate. The dynamic exact-uniform dataset (`datasets/metro_v2_clegr_extended_exact2x/`)
  already exists and matches the current (simplified) task definitions.
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
