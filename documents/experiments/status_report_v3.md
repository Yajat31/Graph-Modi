# Status report: the v3 benchmark (compact CLEGR-style subway graphs), TEA vs GraphToken vs soft prompt

All numbers are generated from the result files in `documents/experiments/results/v3-20260929/` and
`v3x-20260929/` by `scripts/build_report_v3.py`. **Single seed (42) throughout.** Figures are in
`documents/experiments/figures/` (`v3_*.png`, `v3x_variants.png`).

This report supersedes `status_report_fixed_final.md`, which used the earlier benchmark (graphs of 16-48
stations, label skew and a text shortcut).

## 1. Summary

1. **The benchmark now discriminates.** The graph-blind soft prompt scores at the question-only prior
   (static 33.7% validation and 32.8% test, against priors of
   33.1% and 32.8%), so anything clearly above the prior must come from the graph.
2. **GraphToken is the strongest model**: 40.3% static test, against
   36.3% for TEA (5 epochs), 37.3% for TEA (3 epochs) and
   32.8% for soft_prompt. On the multi-turn eval it reaches
   37.9% test when its graph tokens are re-encoded after each edit.
3. **The graph tokens carry node attributes and some neighbourhood information, but not edge structure.** The gains
   are on `attribute_lookup`, `most_common_attribute_within_hops`, `reachability` and `constrained_reachability`.
   `edge_exists`, `cycle_membership`, `node_degree`, `filtered_neighbor_count`, `shortest_path` and `within_hops_count`
   show no consistent gain over their priors for any model (one exception that does not replicate: `node_degree` for the
   3-epoch TEA checkpoint, 16.7% against a 12.1% prior).
4. **Re-encoding the edited graph helps**: updated-graph minus frozen-graph is +4.6 points for GraphToken and
   +2.6 for TEA on exact2x test, and a different session's graph is worse than the correct initial graph.
5. **Model-written edits cost nothing for TEA.** A frozen Llama with a few-shot prompt reproduces 95.2%
   of edits exactly, and accuracy with them is 34.9% against 35.0% with
   oracle edits (test). GraphToken with model-written edits scores 37.6% on test against 37.9% with oracle edits.
6. **A deeper GNN, a larger projector and 3x more data did not help TEA** (Section 6).
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

| | TEA | GraphToken | soft_prompt |
|---|---|---|---|
| Encoder | GraphSAGE 768 / 1024 / 1024, 3 layers, dropout 0.5, mean aggregation, frozen BERT-768 node features | same architecture, **initialised from TEA's pretrained GNN and trained jointly** | none |
| Alignment | contrastive GNN-to-LLM-token pretraining on 5,996 graphs from the eval grid (10 epochs) | uses TEA's | none |
| Trained | one linear projector | GNN + projector | 10 soft tokens |
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
| GraphToken (5 epochs) | 40.2 | 40.3 |

Per task, test split (accuracy %; the prior is the best rule from the question's own fields):

| Task | prior | soft_prompt | TEA (5 ep) | TEA (3 ep) | GraphToken |
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
  graph tokens; GraphToken reads it back at 54.5% and TEA at
  43.2%.
- `attribute_check`, which asks the same fact as a yes/no question, stays at chance for every model.

## 5. Multi-turn (exact2x) results

Validation:

| Condition | TEA (5 ep) | TEA (3 ep) | GraphToken |
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

| Condition | TEA (5 ep) | TEA (3 ep) | GraphToken |
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

| Task | TEA updated | TEA frozen | GraphToken updated | GraphToken frozen | soft_prompt |
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
| GraphToken updated graph | 33.2 (n=1811) | 41.1 (n=2644) |
| GraphToken frozen graph | 24.4 (n=1811) | 39.4 (n=2644) |
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
| GraphToken (5 epochs) | validation | 38.1 | 38.2 | +0.1 | 95.0 | 89.9 |
| GraphToken (5 epochs) | test | 37.6 | 37.9 | +0.3 | 95.2 | 88.6 |

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

Same compute budget (about 3,000 optimiser steps); evaluation sets identical to the baseline's.

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
tested directly). GraphToken, whose GNN is trained jointly, handles the near-closed cases better (86% accuracy) but misses about
half of the closed endpoints. These probes are on the static test split with the 5-epoch checkpoints.

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
- **Different trainable capacity.** soft_prompt, TEA and GraphToken were each trained for 5 epochs, but GraphToken updates the
  GNN as well as the projector, so it has far more trainable parameters than TEA or the soft prompt. The 3-epoch TEA checkpoint was chosen to match the variants'
  compute budget, not selected on test, and is reported next to the 5-epoch run.
- **Not CLEGR's token order.** CLEGR places the graph tokens first (graph, W(f_i), question); here they come after the
  text, just before the answer. The wrapper is `Question: ... Answer:`, not `[INST]`. The projector also receives the hop count
  and a task one-hot, which CLEGR's does not.
- **The soft prompt is weaker than CLEGR's.** CLEGR's soft prompt also sees the full node and edge CSV of the graph as
  text. Here it sees only the named stations' attributes (and the edit history in the multi-turn eval). A CSV-in-prompt
  track was implemented but cancelled before producing results.
- **Graph size and attributes are ours** (12-22 stations, invented categories; CLEGR averages about 26 stations).
- **GraphToken here is a warm start**, initialised from TEA's pretrained GNN, not the from-scratch recipe.
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

Variant configs are `configs/v3x_*.yaml`. Code changed for this benchmark: `data/v3.py` (new), `graph/solvers.py`
(W(f_i) masking, Facts tasks), `graph/edits.py` (multi-word station names),
`models/tea_glm.py` and `models/soft_prompt.py` (greedy decoding), `cli.py`.
