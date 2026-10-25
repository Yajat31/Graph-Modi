# Status report: the v3 benchmark (compact CLEGR-style subway graphs), TEA vs soft prompt (and a failed GraphToken)

All numbers are generated from the result files in `documents/experiments/results/v3-20260929/` and
`v3x-20260929/` by `scripts/build_report_v3.py`. **Single seed (42) throughout.** Figures are in
`documents/experiments/figures/` (`v3_*.png`, `v3x_variants.png`).

This report supersedes `status_report_fixed_final.md`, which used the earlier benchmark (graphs of 16-48
stations, label skew and a text shortcut).

## 0. Corrections (read first; found by the 2026-10-14 audits)

The first version of this report was wrong in four ways that change what the numbers mean.

1. **"TEA run 2" is not GraphToken.** The column earlier labelled GraphToken has a frozen GNN (training re-froze it) and
   differs from TEA only in the projector's random initialisation. It is a second TEA run, so the TEA / TEA run 2 gap is
   **run-to-run variance**: 40.3% vs 36.3% static test, and on exact2x test
   `oracle_updated_graph` 37.9% vs 35.1%, a paired difference of -2.8 points
   (session-bootstrap 95% CI [-4.3, -1.4]).
   Treat about 3 points as the noise floor for any single-seed comparison here.
2. **No "pretrained" GNN existed.** `TEAGLM` freezes its GNN when built, and the contrastive pretraining optimised that
   frozen GNN, so only the discarded alignment head trained (final loss 4.477 against a chance level of 4.908; the 6-layer
   run ended at 4.906). Every TEA, TEA run 2 and `v3x` result in this report is therefore a **frozen random GNN plus a
   trained projector**. With the optimiser fixed the same objective reaches 1.746. The `v3x_deep` result is not evidence about
   depth: that GNN was random and 6 layers oversmooth more. A properly pretrained TEA is being run as a separate result.
3. **The first end-to-end GraphToken (mean aggregation) collapsed, and the cause was the aggregation (Section 0a).** Static test
   33.0% against a 32.8% question-only prior: it answers "yes" to all 396 `reachability` and all 396 `edge_exists`
   questions, no checkpoint of the 10 saved ever exceeds 34.2% on validation, and its multi-turn `oracle_updated_graph` score
   (34.1%) equals the graph-blind soft prompt's 34.2%. One-epoch pilots that changed warmup, GNN lr, dropout, clipping and bf16 did not
   recover it; changing only the aggregation from mean to sum did (53% vs 33% static validation).
4. **Statistics and baselines.** The per-turn Wilson intervals below ignore that turns of one session are dependent; session-bootstrap
   intervals are about 25% wider and paired tests are in `results/v3-20260929/stats_test_existing_runs.json`. TEA's graph advantage over
   the soft prompt on exact2x test is +0.9 points (CI [-0.8, +2.6]), i.e. not significant.
   The condition called `majority_prior` is the **stale (pre-edit) answer** (59.3%), not a majority baseline (about 29.4%); it is now
   `stale_answer`. The answer parser used to take the first number anywhere in a generation, which under-scores free-form text-only
   baselines; old result rows were scored with it, and re-runs under the leading-token parser are in `outputs/audit/`.

## 0a. Results after the fixes (sum aggregation; single GPU runs, audit of 2026-10-14 to 16)

Mean aggregation divides by the neighbour count, so a node's degree is invisible to the GNN, and with a random
initialisation the mean-aggregated embeddings are almost the same vector for every node (node-to-node variation is 9% of the
embedding norm). **Sum aggregation** keeps degree and gives the encoder usable structure. Everything below uses sum aggregation,
dropout 0, a cosine schedule with warmup, and (for jointly trained GNNs) a GNN learning rate of 1e-4. Accuracy in %; test split
for multi-turn; the question-only prior is 32.8 static and 27.7 multi-turn.

| Model | static val | static test | exact2x test: updated graph | frozen graph | shuffled graph | structure only |
|---|---|---|---|---|---|---|
| soft_prompt (no graph) | 33.7 | 32.8 | - | - | - | - |
| TEA, old (random frozen GNN, mean aggregation) | 36.6 | 36.3 | 35.1 | 32.5 | 30.2 | 32.3 |
| task-only control (zeroed GNN, retrained) | 33.1 | 33.1 | - | - | - | - |
| end-to-end GraphToken, mean aggregation | 32.6 | 33.0 | 34.1 | 31.3 | 31.4 | 31.3 |
| TEA, frozen RANDOM GNN, sum aggregation | 54.3 | 53.3 | 56.3 | 44.3 | 30.9 | 43.7 |
| TEA, frozen PRETRAINED GNN, sum aggregation | 44.4 | 45.2 | 41.8 | 37.0 | 30.2 | 37.1 |
| GraphToken, sum aggregation, seed 42 | 53.3 | 52.7 | 57.3 | 44.9 | 30.6 | 44.5 |
| GraphToken, sum aggregation, seed 43 | 53.7 | 53.0 | 57.0 | 44.7 | 30.9 | 44.6 |
| GraphToken, sum aggregation, seed 44 | 53.4 | 53.1 | 56.8 | 44.2 | 30.5 | 43.7 |
| GraphToken warm start (pretrained + joint), sum aggregation | 67.0 | 65.8 | 63.7 | 46.2 | 30.8 | 45.9 |
| soft prompt, edit history as text / question only (multi-turn) | | | 34.2 / 31.9 | | | |

By task (static test):

| Task | prior | TEA random GNN | TEA pretrained GNN | GraphToken | GraphToken warm start |
|---|---|---|---|---|---|
| node_degree | 12.1 | 98.5 | 27.5 | 100.0 | 98.7 |
| reachability | 50.0 | 53.8 | 84.1 | 51.8 | 94.7 |
| attribute_lookup | 27.3 | 30.1 | 60.1 | 26.5 | 93.4 |
| cycle_membership | 50.0 | 92.9 | 72.5 | 92.2 | 91.4 |
| constrained_reachability | 50.0 | 69.4 | 55.8 | 70.2 | 71.2 |
| edge_exists | 50.0 | 69.4 | 58.8 | 72.5 | 71.0 |
| most_common_attribute_within_hops | 27.0 | 35.4 | 34.8 | 28.5 | 50.0 |
| attribute_check | 50.3 | 47.0 | 54.0 | 48.0 | 48.0 |
| shortest_path | 16.7 | 32.3 | 15.4 | 32.8 | 46.5 |
| filtered_neighbor_count | 16.7 | 36.6 | 21.2 | 37.9 | 40.4 |
| within_hops_count | 10.4 | 20.7 | 12.9 | 19.7 | 18.2 |

What this shows:

1. **A valid GraphToken exists and uses the graph.** Over 3 seeds its static test accuracy is 53.0 +/- 0.2 (sd) points; on multi-turn the
   updated graph beats the frozen graph by about 12 points and a shuffled graph by about 26 (paired session-bootstrap CIs exclude 0;
   `stats_test_final.json`). Model-written edits cost nothing: 56.8% with predicted edits vs
   56.9% with oracle edits (edit accuracy 95.2%, which is mostly in-context copying).
2. **Joint training of the GNN adds nothing over a random GNN with sum aggregation.** A frozen random sum-aggregation GNN
   reaches 53.3% static test, the same as the trained-from-scratch GraphToken (53.0%); on multi-turn updated-graph accuracy the paired difference is
   -1.0 [-2.1, +0.1] points. The gain over the earlier models is the aggregation, not the training of the GNN.
3. **Contrastive pretraining helps only when the GNN is then trained.** A frozen pretrained GNN is *worse* than a frozen random one
   (45.2% vs 53.3% static test): it reads attributes and reachability
   (`attribute_lookup` 60.1%, `reachability` 84.1%) but loses degree
   (`node_degree` 27.5% vs 98.5%), because the text-alignment objective does not reward it. Starting joint training from the pretrained
   GNN gives the best model (65.8% static test, 63.7% multi-turn; `attribute_lookup`
   93.4%, `reachability` 94.7%), about 6 points above GraphToken from scratch on multi-turn
   (paired difference -6.4 [-7.9, -4.9] for scratch minus warm start). Pretraining and joint training are complementary.
4. **Against the graph-blind soft prompt.** The like-for-like control (question only, no history text) scores
   31.9%. Paired differences on multi-turn updated-graph accuracy: GraphToken warm start +31.8 [+29.5, +34.1], GraphToken +25.4 [+23.0, +27.8],
   TEA random sum GNN +24.4 [+22.0, +26.9], TEA pretrained sum GNN +9.9 [+8.1, +11.8], old TEA +3.2 [+1.6, +5.0] (points, 95% CI).
5. **Seed noise.** The three GraphToken seeds differ by at most 0.4 points on multi-turn (paired CIs include 0), so the earlier 3-point
   run-to-run spread belongs to the mean-aggregation setup where the GNN barely carried information. These are still three runs of one
   recipe, on one dataset, and the warm-start and TEA rows are single seeds.
6. **TEA with working pretraining needed its own learning rate.** At the TEA default of 1e-3 the pretrained-GNN projector trains badly
   (answer loss 1.3 vs 0.7 for a random GNN, static test 34.4%) because the pretrained embeddings vary about ten times more across nodes;
   one-epoch pilots gave 42.7% validation at 1e-4 and 35.3% at 3e-4, so 1e-4 was used. This lr was chosen with validation loss/accuracy of a
   1-epoch pilot, not the test split.
7. **Deeper GNNs.** A 6-layer GNN, pretrained properly, loses message passing (neighbour-status probe AUC 0.51-0.62 vs 0.99-1.00 for 3
   layers; `probe_variants.json`), so the earlier `v3x_deep` result says nothing about depth versus pretraining.

Caveats specific to these runs: the TEA rows use projector lr 1e-4 and the GraphToken rows 1e-3 (GNN 1e-4), and the learning rates were not tuned jointly across models (only TEA with a pretrained GNN had a 1e-4 / 3e-4 / 1e-3 pilot); sum aggregation was chosen after seeing pilots on validation data; tasks that depend on degree
(`node_degree` 100%, `cycle_membership` 92%) become easy, so part of the improvement is that the encoder can now count; training data
still has the task-by-cell imbalance and closed-station mismatch listed in Section 8; `checkpoint-best` was recorded but the final
checkpoint is what is evaluated.

## 0b. How much is a node-local shortcut? (audit of 2026-10-16)

Most of the gain in Section 0a can be produced by two node-local facts. `scripts/shortcut_audit.py` fits a gradient-boosted classifier per
task on the static *train* split using only: the degree of the source and target station, their open/closed status, graph size and whether
they share a line (no adjacency, no other attributes, no text). Computed on the 8 tasks without a queried attribute (static test):

| Task | shortcut | old TEA | TEA random sum GNN | TEA pretrained sum GNN | GraphToken seed 42 | GraphToken seed 43 | GraphToken warm start |
|---|---|---|---|---|---|---|---|
| constrained_reachability | 69.4 | 53.8 | 69.4 | 55.8 | 70.2 | 70.7 | 71.2 |
| cycle_membership | 91.4 | 53.5 | 92.9 | 72.5 | 92.2 | 92.4 | 91.4 |
| edge_exists | 70.2 | 52.5 | 69.4 | 58.8 | 72.5 | 72.0 | 71.0 |
| filtered_neighbor_count | 29.5 | 17.7 | 36.6 | 21.2 | 37.9 | 37.4 | 40.4 |
| node_degree | 100.0 | 12.1 | 98.5 | 27.5 | 100.0 | 100.0 | 98.7 |
| reachability | 94.2 | 57.8 | 53.8 | 84.1 | 51.8 | 52.8 | 94.7 |
| shortest_path | 41.4 | 16.9 | 32.3 | 15.4 | 32.8 | 34.6 | 46.5 |
| within_hops_count | 15.4 | 10.9 | 20.7 | 12.9 | 19.7 | 17.7 | 18.2 |
| **mean over these 8 tasks** | **64.0** | 34.4 | 59.2 | 43.5 | 59.6 | 59.7 | 66.5 |
| items where the shortcut is wrong (n=1142) | 0.0 | 23.5 | 25.7 | 22.4 | 25.0 | 24.4 | 22.0 |
| items where the shortcut is right | 100.0 | 40.6 | 78.1 | 55.4 | 79.2 | 79.6 | 91.6 |
| `cycle_membership`, degree >= 2 (n=231; always-yes = 85.7) | - | 86.1 | 87.9 | 70.1 | 86.6 | 87.0 | 85.3 |
| `reachability`, both endpoints open (n=220; always-yes = 90) | - | 25.0 | 90.0 | 79.5 | 68.2 | 39.1 | 90.5 |
| `reachability`, an endpoint closed (n=176) | - | 98.9 | 8.5 | 89.8 | 31.2 | 69.9 | 100.0 |

Multi-turn, applying the same shortcut to the edited graph (a status-and-degree tracker) and to the stale initial graph:

| Model (multi-turn test, 8 non-attribute tasks) | all turns, updated graph | turns where the shortcut fails (n=1401) | turns where it succeeds | frozen graph, all turns |
|---|---|---|---|---|
| shortcut tracker (updated graph / stale graph) | 64.6 | 0.0 | 100.0 | 45.5 |
| TEA random sum GNN | 59.3 | 23.5 | 78.9 | 45.8 |
| GraphToken seed 42 | 60.5 | 24.0 | 80.5 | 46.4 |
| GraphToken warm start | 65.9 | 22.4 | 89.8 | 46.2 |
| soft prompt (question only) | 32.4 | 20.4 | 39.0 | - |

Reading:

1. **The models are at the shortcut ceiling, not above it.** Over these 8 tasks the shortcut scores 64.0%; GraphToken from scratch
   59.6% and the warm start 66.5%. Where it is wrong every model scores 22-26%, so no model shows multi-hop topology
   reasoning here. The warm start's extra points come from reading attributes and status (`attribute_lookup`, and `reachability` where an endpoint is closed).
2. **The "+12 points from re-encoding the edited graph" is what a status tracker gives.** The shortcut gains 19.1 points from
   the updated graph; the models gain about 14 (updated minus frozen). That shows the encoder tracks edits, not that it reasons over paths.
3. **The cycle skill is "degree 1 means no".** On `cycle_membership` items with degree >= 2 every model is at the always-yes rate (the generator forces
   degree-1 terminals and 83% of "no" answers are one).
4. **Reachability is status reading, and seeds differ per task.** Scratch GraphToken and the random-GNN TEA do not read status (chance); the warm start
   reads it exactly but is at the always-yes rate when both endpoints are open, so it detects no disconnection. Two scratch seeds give opposite
   biases on this task (31 / 70 and 68 / 39 on the two slices), so the 0.2-point overall seed spread hides large per-task variation.
5. **Corrections to Section 0a.** (i) Sum aggregation gives degree, not general structure; the mean-to-sum effect is causal (pilots A and D differ in one factor) but
   what it unlocked is degree. (ii) "Joint training adds nothing over a random GNN" compares models that both sit at the shortcut ceiling, and the two differ in lr
   (1e-4 vs 1e-3) and schedule, so it is not a clean test. (iii) The warm start reads `attribute_lookup` at 93% but `attribute_check` on the same facts at 48%; not investigated.
6. **Richer aggregation or readout did not help (1-epoch pilots, static validation, recipe of pilot D = 53.0).** A GIN-style per-neighbour MLP before the sum
   scored 52.6; a 2-layer MLP projector (hidden 2048, lr 3e-4) scored 53.4 (`shortest_path` 32 vs 25 and `filtered_neighbor_count` 37 vs 32 on validation, single
   seed, within what per-task seed spread allows). Neither moves the tasks that need multi-hop structure, which is consistent with the shortcut ceiling above:
   the limit is what the training data rewards, not the aggregator.
7. **Not yet done:** regenerating data balanced on degree and endpoint status (this changes the methodology and invalidates existing results); the audit of the data
   so far covers text leakage only, not structural shortcuts like these.

## 1. Summary

1. **The benchmark now discriminates.** The graph-blind soft prompt scores at the question-only prior
   (static 33.7% validation and 32.8% test, against priors of
   33.1% and 32.8%), so anything clearly above the prior must come from the graph.
2. **A frozen random GNN plus a trained projector beats the graph-blind soft prompt on static QA, but the two TEA runs differ by
   as much as TEA differs from the others**: 40.3% (TEA run 2), 36.3% (TEA, 5 epochs), 37.3% (TEA, 3 epochs) and
   32.8% for soft_prompt (static test). On the multi-turn eval the two TEA runs reach
   35.1% and 37.9% when graph tokens are re-encoded after each edit. No claim that one of these models is better than
   another is supported by a single seed (Section 0).
3. **(Mean-aggregation, random-GNN models only; see Sections 0a-0b for the sum-aggregation ones.) The graph tokens carry node attributes and some neighbourhood information, but not edge structure.** The gains
   are on `attribute_lookup`, `most_common_attribute_within_hops`, `reachability` and `constrained_reachability`.
   `edge_exists`, `cycle_membership`, `node_degree`, `filtered_neighbor_count`, `shortest_path` and `within_hops_count`
   show no consistent gain over their priors for any model (one exception that does not replicate: `node_degree` for the
   3-epoch TEA checkpoint, 16.7% against a 12.1% prior).
4. **Re-encoding the edited graph helps**: updated-graph minus frozen-graph is +4.6 points for TEA run 2 and
   +2.6 for TEA on exact2x test (about 3 points of this is within the run-to-run spread), and a different session's graph is worse than the correct initial graph.
5. **Model-written edits cost nothing for TEA.** A frozen Llama with a few-shot prompt reproduces 95.2%
   of edits exactly, and accuracy with them is 34.9% against 35.0% with
   oracle edits (test). TEA run 2 with model-written edits scores 37.6% on test against 37.9% with oracle edits.
6. **A deeper GNN, a larger projector and 3x more data did not help TEA** (Section 6; invalid as evidence about depth or pretraining, see Section 0).
7. **Caveats that matter** are in Section 8: one seed, unequal epochs, a graph-token position that differs from CLEGR, and
   a soft prompt that is weaker than CLEGR's.

## 2. The benchmark

- **Graphs**: 12-22 stations, exact grid over size (11 values) x density (sparse / medium / dense, defined by mean
  degree) with connectivity and a diameter cap enforced; CLEGR-style node semantics (`disabled_access`, `has_rail`,
  `architecture`, `cleanliness`, `music`, `size`, plus `line` and `status`) and edge attributes (`line_color`,
  `line_stroke`, `has_aircon`, `built`).
- **Tasks** (11 static, 9 in the multi-turn eval): `edge_exists`, `reachability`, `constrained_reachability`,
  `cycle_membership`, `node_degree`, `filtered_neighbor_count`, `shortest_path`, `within_hops_count`,
  `most_common_attribute_within_hops`, plus the Facts tasks `attribute_lookup` and `attribute_check` (static only).
  Dropped as not encodable by a 3-layer GNN without edge weights or name outputs: `path_cost`, `node_count`,
  `within_hops_list`, `filtered_path_count`.
- **Labels are balanced**: yes/no tasks are 50% by construction; the others have flat answer distributions.
  A cross-validated text-only audit on the static sets finds no signal in the prompt text beyond what the question type gives away.

| Task | static test items | majority answer | question-only prior |
|---|---|---|---|
| attribute_check | 396 | 50.3 | 50.3 |
| attribute_lookup | 396 | 6.1 | 27.3 |
| constrained_reachability | 396 | 50.0 | 50.0 |
| cycle_membership | 396 | 50.0 | 50.0 |
| edge_exists | 396 | 50.0 | 50.0 |
| filtered_neighbor_count | 396 | 16.7 | 16.7 |
| most_common_attribute_within_hops | 396 | 5.8 | 27.0 |
| node_degree | 396 | 12.4 | 12.1 |
| reachability | 396 | 50.0 | 50.0 |
| shortest_path | 396 | 16.9 | 16.7 |
| within_hops_count | 396 | 9.8 | 10.4 |

- **Data sizes**: training 23,992 items (2,000 independent graphs x 9 tasks = 18,000 items, plus about 6,000 items from
  1,000 graphs as before/after-one-edit pairs); static eval 2178 validation / 4356 test items; multi-turn eval 1,188 sessions per
  split (1, 2, 4 or 8 turns; 4,455 turns), one cell per (size, density, session length, task).
- **`status` (open/closed) is not in the prompt text**; it is only in the graph, or in the edit history in the multi-turn
  eval. For the Facts tasks and the mode task, the queried attribute is also hidden from the text.

## 3. Models and prompts

| | TEA | TEA run 2 (was "GraphToken") | soft_prompt |
|---|---|---|---|
| Encoder | GraphSAGE 768 / 1024 / 1024, 3 layers, dropout 0.5, mean aggregation, frozen BERT-768 node features; **random, never effectively pretrained** (Section 0) | same GNN as TEA, **frozen** | none |
| Alignment | contrastive pretraining was run for 10 epochs but never updated the GNN (only the discarded head) | uses TEA's | none |
| Trained | one linear projector (126.6M parameters) | one linear projector (different random init) | 10 soft tokens |
| LLM | Llama-3.1-8B-Instruct, frozen | same | same |
| Training | 5 epochs, effective batch 24, lr 1e-3, bf16 | same | same |
| Decoding | greedy, at most 32 new tokens | same | same |

Prompt (all three): `Question: <W(f_i): attributes of the named stations> <question> Answer:` followed by the
10 graph or soft tokens, then the generated answer. The graph tokens come from a linear map of
`[pooled | source node | target node | hop count | task one-hot]` (3,087 numbers); the pooled slot is zeroed whenever the
question names a station, which is every task here. The multi-turn conditions differ only in the graph tokens and
whether the edit history is included as text.

## 4. Static results

| Model | Validation | Test |
|---|---|---|
| majority answer per task | 29.2 | 28.9 |
| question-only prior (best rule from the question fields alone) | 33.1 | 32.8 |
| soft_prompt (no graph) | 33.7 | 32.8 |
| TEA (5 epochs) | 36.6 | 36.3 |
| TEA (3 epochs) | 37.6 | 37.3 |
| TEA run 2 (5 epochs; was labelled GraphToken) | 40.2 | 40.3 |

Per task, test split (accuracy %; the prior is the best rule from the question's own fields):

| Task | prior | soft_prompt | TEA (5 ep) | TEA (3 ep) | TEA run 2 |
|---|---|---|---|---|---|
| reachability | 50.0 | 50.0 | 57.8 | 66.7 | 68.9 |
| constrained_reachability | 50.0 | 48.0 | 53.8 | 60.1 | 61.1 |
| attribute_lookup | 27.3 | 27.3 | 43.2 | 42.4 | 54.5 |
| edge_exists | 50.0 | 48.0 | 52.5 | 51.5 | 54.3 |
| attribute_check | 50.3 | 51.0 | 48.2 | 49.7 | 50.3 |
| cycle_membership | 50.0 | 49.2 | 53.5 | 50.0 | 50.0 |
| most_common_attribute_within_hops | 27.0 | 28.0 | 32.3 | 33.8 | 50.0 |
| shortest_path | 16.7 | 16.7 | 16.9 | 15.7 | 17.2 |
| filtered_neighbor_count | 16.7 | 19.9 | 17.7 | 15.7 | 16.9 |
| node_degree | 12.1 | 14.4 | 12.1 | 16.7 | 11.6 |
| within_hops_count | 10.4 | 8.6 | 10.9 | 8.1 | 8.8 |

Standard errors are about 2.5 points per task (about 396 items) and
0.7 points overall.

- The soft prompt tracks the prior on every task.
- `attribute_lookup` hides the queried attribute from the text, so the gain over the prior (27%) can only come from the
  graph tokens; TEA run 2 reads it back at 54.5% and TEA at
  43.2%.
- `attribute_check`, which asks the same fact as a yes/no question, stays at chance for every model.

## 5. Multi-turn (exact2x) results

Validation:

| Condition | TEA (5 ep) | TEA (3 ep) | TEA run 2 |
|---|---|---|---|
| oracle_updated_graph | 34.6 [33.2-36.0] | - | 38.1 [36.7-39.5] |
| structure_only | 32.7 [31.4-34.1] | - | 33.8 [32.5-35.3] |
| frozen_graph_history | 32.3 [31.0-33.7] | - | 34.1 [32.7-35.5] |
| cached_no_reencode | 32.3 [31.0-33.7] | - | 34.2 [32.8-35.6] |
| shuffled_graph | 29.8 [28.5-31.2] | - | 31.0 [29.7-32.4] |
| graph_once_then_text | 29.5 [28.2-30.9] | - | 29.9 [28.6-31.2] |
| token_matched_history | 28.8 [27.5-30.1] | - | 28.4 [27.1-29.8] |
| question_only | 27.9 [26.6-29.2] | - | 27.6 [26.3-28.9] |
| soft_prompt (reads the edit history as text) | 33.8 [32.5-35.2] | | |
| overall majority baseline | 29.6 | | |

Test:

| Condition | TEA (5 ep) | TEA (3 ep) | TEA run 2 |
|---|---|---|---|
| oracle_updated_graph | 35.1 [33.7-36.5] | 36.6 [35.2-38.0] | 37.9 [36.5-39.3] |
| structure_only | 32.3 [31.0-33.7] | 33.0 [31.7-34.4] | 33.1 [31.7-34.5] |
| frozen_graph_history | 32.5 [31.1-33.8] | 32.9 [31.5-34.3] | 33.3 [31.9-34.7] |
| cached_no_reencode | 32.4 [31.1-33.8] | - | 33.2 [31.8-34.6] |
| shuffled_graph | 30.2 [28.9-31.6] | 31.3 [30.0-32.7] | 30.7 [29.4-32.1] |
| graph_once_then_text | 29.8 [28.5-31.1] | - | 29.6 [28.3-31.0] |
| token_matched_history | 29.0 [27.6-30.3] | - | 28.8 [27.5-30.1] |
| question_only | 27.7 [26.4-29.0] | - | 27.7 [26.4-29.1] |
| soft_prompt (reads the edit history as text) | 34.2 [32.8-35.6] | | |
| overall majority baseline | 29.4 | | |

95% intervals are per-turn and ignore correlation within a session, so they are optimistic. A dash means that condition or split was not run for that model (the 3-epoch checkpoint was evaluated on the test split with four conditions only).

Per task, test split, graph models with re-encoded versus frozen graph tokens:

| Task | TEA updated | TEA frozen | TEA run 2 updated | TEA run 2 frozen | soft_prompt |
|---|---|---|---|---|---|
| constrained_reachability | 55.4 | 55.8 | 65.5 | 57.2 | 51.3 |
| cycle_membership | 53.5 | 49.5 | 50.3 | 50.3 | 57.2 |
| edge_exists | 50.1 | 49.3 | 53.7 | 50.9 | 53.9 |
| filtered_neighbor_count | 14.9 | 12.7 | 13.7 | 11.9 | 16.8 |
| most_common_attribute_within_hops | 33.1 | 31.7 | 42.0 | 41.6 | 26.5 |
| node_degree | 16.8 | 12.5 | 11.9 | 9.5 | 11.7 |
| reachability | 60.6 | 52.9 | 72.5 | 51.9 | 61.2 |
| shortest_path | 18.6 | 17.0 | 19.0 | 17.0 | 18.6 |
| within_hops_count | 12.7 | 10.7 | 12.3 | 9.3 | 10.5 |

Edit tracking, test split (accuracy on turns whose edit changed the answer versus not):

| Model / condition | edit changed the answer | answer unchanged |
|---|---|---|
| TEA updated graph | 34.6 (n=1811) | 35.4 (n=2644) |
| TEA frozen graph | 29.7 (n=1811) | 34.4 (n=2644) |
| TEA no graph tokens | 25.5 (n=1811) | 29.2 (n=2644) |
| TEA run 2 updated graph | 33.2 (n=1811) | 41.1 (n=2644) |
| TEA run 2 frozen graph | 24.4 (n=1811) | 39.4 (n=2644) |
| soft_prompt | 33.0 (n=1811) | 35.0 (n=2644) |

Accuracy does not trend with session length, turn index, graph size or density beyond sampling noise (the
8-turn sessions have only 297 sessions; late-turn points move by about 5 points). See `v3_exact2x_by_*.png`.

### Model-written edits

In this condition the model reads each edit sentence, writes the edit as text, the text is parsed and applied, and the
resulting graph is re-encoded. Each run also evaluates the oracle-edit condition, so the comparison is within one run.

| Model | Split | model-written edits | oracle edits | gap | edits exactly right | graph state exactly right |
|---|---|---|---|---|---|---|
| TEA (5 epochs) | validation | 34.5 | 34.5 | +0.0 | 95.0 | 89.9 |
| TEA (5 epochs) | test | 34.9 | 35.0 | +0.1 | 95.2 | 88.6 |
| TEA (3 epochs) | validation | pending | pending | | | |
| TEA (3 epochs) | test | pending | pending | | | |
| TEA run 2 (5 epochs; was labelled GraphToken) | validation | 38.1 | 38.2 | +0.1 | 95.0 | 89.9 |
| TEA run 2 (5 epochs; was labelled GraphToken) | test | 37.6 | 37.9 | +0.3 | 95.2 | 88.6 |

Edit accuracy by kind (TEA, test):

| Edit kind | turns | reproduced exactly (%) |
|---|---|---|
| NOOP | 1668 | 100.0 |
| SET NODE | 474 | 92.2 |
| DEL EDGE + SET NODE | 438 | 92.2 |
| ADD EDGE | 435 | 89.9 |
| DEL EDGE | 429 | 93.9 |
| ADD EDGE + SET NODE | 424 | 91.7 |
| ADD EDGE + DEL EDGE | 387 | 93.3 |
| ADD EDGE + DEL EDGE + SET NODE | 200 | 93.0 |

An earlier run of this condition was invalid: the edit parser read a station name as one word, and v3 station names
have two words, so every real edit was rejected. The parser now resolves multi-word names by longest match and all
5,541 gold edit turns round-trip through it; the invalid results are kept in `results/v3-20260929/invalid_parser_bug/`.
Oracle, static and every other result were unaffected.

## 6. Does a deeper GNN, a larger projector or more data help TEA?

Same compute budget (about 3,000 optimiser steps); evaluation sets identical to the baseline's. **Read with Section 0:** every row
here used a random frozen GNN (the pretraining never trained it), so the 6-layer row compares depth of a random GNN, not pretraining quality.

| Variant | static val | static test | `reachability` | `attribute_lookup` | exact2x updated | exact2x frozen | exact2x shuffled |
|---|---|---|---|---|---|---|---|
| baseline: 3 layers, linear projector, 24k examples x 3 epochs | 37.6 | 37.3 | 66.7 | 42.4 | 36.6 | 32.9 | 31.3 |
| 6-layer GNN | 34.0 | 33.4 | 51.5 | 30.1 | 33.4 | 31.1 | 31.4 |
| MLP projector 3 x 4096, lr 1e-3 (collapsed to one answer) | 29.2 | 29.1 | 50.0 | 6.6 | 31.4 | 29.2 | 29.3 |
| MLP projector 2 x 2048, lr 2e-4 | 33.4 | 33.2 | 50.0 | 32.3 | 33.6 | 32.1 | 31.9 |
| 3x training data (72k examples), 1 epoch | 33.7 | 34.1 | 50.0 | 36.1 | 33.8 | 32.2 | 31.2 |

None of the three changes beats the baseline. A 6-layer GNN falls to the prior on `reachability` (it answers "yes" to 386 of
396 questions); the first larger projector collapsed to almost constant answers (for example "yes" to all 396 `reachability`
questions), and the corrected one is also at chance on `reachability` and only slightly above the prior on `attribute_lookup`;
3x more data at equal steps is also below the baseline. The 5-epoch TEA run is slightly below the 3-epoch checkpoint of the
same run, so more training or more data is not obviously the bottleneck. These are single runs, so differences of a point or
two are noise.

Why TEA over-predicts "no" on `reachability`: it separates a closed endpoint perfectly (98.9% on those items) but
a closed station one to three hops away also pushes it to "no" (89%, 78% and 67% "no" at 1, 2 and 3 hops), so it gets only
22-33% of those. This is consistent with its frozen station embeddings mixing a station's own status with its neighbours' (not
tested directly). TEA run 2 handles the near-closed cases better (86% accuracy) but misses about
half of the closed endpoints; since it differs from TEA only by projector initialisation, that difference is also run-to-run variation. These probes are on the static test split with the 5-epoch checkpoints.

## 7. Figures

| File | Content |
|---|---|
| `v3_static_by_task.png`, `v3_static_by_density_scale.png` | static accuracy per task, by density and size |
| `v3_exact2x_conditions.png` | every multi-turn condition, validation and test |
| `v3_exact2x_by_task.png`, `v3_exact2x_by_density_size.png` | per task; by density and size |
| `v3_exact2x_by_turn.png`, `v3_exact2x_by_session_length.png`, `v3_exact2x_by_graph_size.png` | turns, sessions, graph size |
| `v3_exact2x_edit_tracking.png` | answer-changing versus unchanged turns |
| `v3_predicted_edits.png` | model-written versus oracle edits |
| `v3x_variants.png` | deeper GNN, larger projector, more data |
| `v3_dataset_composition.png` | label balance and graph sizes |

## 8. Caveats

- **One seed.** CLEGR uses five. Differences of about two points or less between models are within noise.
- **Trainable capacity.** soft_prompt, TEA and TEA run 2 were each trained for 5 epochs. TEA's projector alone has 126.6M trainable
  parameters (a 3,089-to-40,960 linear map); the end-to-end GraphToken adds only the 5.8M-parameter GNN (+4.6%), so capacity does not
  explain its collapse. The 3-epoch TEA checkpoint was chosen to match the variants' compute budget, not selected on test.
- **Not CLEGR's token order.** CLEGR places the graph tokens first (graph, W(f_i), question); here they come after the
  text, just before the answer. The wrapper is `Question: ... Answer:`, not `[INST]`. The projector also receives the hop count
  and a task one-hot, which CLEGR's does not.
- **The soft prompt is weaker than CLEGR's.** CLEGR's soft prompt also sees the full node and edge CSV of the graph as
  text. Here it sees only the named stations' attributes (and the edit history in the multi-turn eval). A CSV-in-prompt
  track was implemented but cancelled before producing results.
- **Graph size and attributes are ours** (12-22 stations, invented categories; CLEGR averages about 26 stations).
- **The old "GraphToken" column is a second TEA run** (frozen GNN) and the first end-to-end GraphToken (mean aggregation) collapsed. The valid
  GraphToken results are the sum-aggregation runs in Section 0a; every table below Section 0a predates them and uses mean aggregation.
- **The history comparison is not symmetric.** The soft prompt reads the edit history as text; `oracle_updated_graph` has none, and no model was
  trained with history in the prompt, so history-text conditions are out of distribution (`frozen_graph_history` 33.3% vs `structure_only` 33.1%
  for TEA run 2: the text is ignored). "Graph beats soft prompt" is not a controlled result; `soft_prompt_no_history` is the like-for-like control.
- **Reachability mostly tests whether an endpoint is closed** (88.9% of "no" answers have a closed endpoint; 91% of "unreachable" answers for
  `shortest_path`), so reachability gains are mainly status reading, not graph structure. Training puts exactly 2 of 11 tasks missing from
  every size-density cell, and 35% of session initial graphs have no closed station, which training never shows.
- **Edit prediction is largely in-context copying.** The few-shot prompt contains the generator's own templates; the 95% figure is the same frozen LLM
  for every model and is not a model comparison.
- **Validation is used for selection in the audit runs.** `checkpoint-best` (validation loss on 1,000 validation examples) and the choice among
  pretraining variants (probe scores on validation graphs) use the validation split, so validation accuracy of those runs is optimistic; only test
  numbers are reported for selected runs.
- **Out-of-distribution graphs were not evaluated.**

## 9. Reproducing

```
python -m graph_modi.cli generate        --config configs/v3_tea.yaml          # datasets/metro_v3
python -m graph_modi.cli pretrain-gnn    --config configs/v3_tea.yaml
python -m graph_modi.cli train-projector --config configs/v3_{tea,graphtoken,soft_prompt}.yaml
python -m graph_modi.cli static-eval     --config configs/v3_<model>.yaml --split {validation,test}
python -m graph_modi.cli evaluate        --config configs/v3_exact2x_<model>.yaml
python -m graph_modi.cli evaluate        --config configs/v3_exact2x_<model>_predfix.yaml   # model-written edits
python scripts/analyze_exact2x.py <evaluation.json> datasets/metro_v3 <analysis.json>
python scripts/make_figures_v3.py && python scripts/build_report_v3.py
```

Variant configs are `archive/configs/v3x_*.yaml`. Code changed for this benchmark: `data/v3.py` (new), `graph/solvers.py`
(W(f_i) masking, Facts tasks), `graph/edits.py` (multi-word station names),
`models/tea_glm.py` and `models/soft_prompt.py` (greedy decoding), `cli.py`.
