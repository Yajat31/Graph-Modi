# Status report: CLEGR-faithful "fixed final" run (TEA, GraphToken, soft-prompt)

Run date: 2026-09-29. Single seed (42). All numbers below are regenerated from the JSONs in
`documents/experiments/results/fixed_final-20260929/` by `scripts/build_report_tables_fixed_final.py`;
figures come from `scripts/make_figures_fixed_final.py`.

## 1. TL;DR

1. **No model beats a trivial per-task majority-answer predictor.** On the exact2x eval that
   predictor scores 41.4% (validation) / 40.9% (test); on the static eval 44.8% / 44.2% (computed on the eval
   sets themselves, so slightly optimistic as a baseline). The best
   model, soft_prompt, scores 38.1% / 37.3% on exact2x and 35.4% / 34.8% on static. TEA and GraphToken
   are lower still. This is the single most important caveat for everything below.
2. **The graph-blind soft-prompt run beats both graph models on both exact2x splits and both static splits**
   (exact2x test: 37.3% vs TEA 35.3% and GraphToken 33.7%; validation: 38.1% vs 30.4% and 28.5%),
   consistent with CLEGR's headline claim.
   Two confounds: soft_prompt trained 10 epochs vs 3 for the graph models, and its `reachability` score
   (82.6%) is a text shortcut created by W(f_i) (Section 6).
3. **The graph tokens carry little graph-specific information.** Feeding a *different* session's graph
   (`shuffled_graph`) scores the same as the correct one (TEA test 34.7% vs 35.3%), and re-encoding
   the updated graph after each edit changes almost nothing.
4. **Density (sparse/medium/dense) does not give a consistent story.** It flips sign between validation
   and test for the graph models. Section 5 reports it separately as requested, with the
   graph-diameter vs GNN-receptive-field data.
5. **A data bug was found and fixed:** in the main exact2x dynamic set, `path_cost` sessions were
   silently generated as `node_count`. `path_cost` is therefore evaluated in a separate supplementary
   run (Section 7). All three models are below the always-"unreachable" baseline (29.7% test); the graph models
   are ~20 points below it, soft_prompt ~6.
6. **Not done / not controlled:** single seed (CLEGR uses five), unequal epochs, no ablation of the
   contrastive pretraining, no fix for the GNN receptive-field mismatch (deliberately left as in CLEGR).

## 2. What was run

| Item | CLEGR (per `clegr.md`) | This run |
|---|---|---|
| Node features | BERT, 768-d | `bert-base-uncased`, mean-pooled over `"label: attr=val, ..."` text, frozen (768-d) |
| GNN | GraphSAGE 768 / 1024 / 1024, 3 layers, dropout 0.5 | identical, mean aggregation |
| Projector | "linear projector", Proj dim 1024; how 10 tokens are made is **not specified** | one linear layer, input 3,087-d = [pooled 1024 \| source node 1024 \| target node 1024 \| hop radius 1 \| task one-hot 14], output 10 x 4096 (Llama-3.1-8B hidden size) reshaped to 10 graph tokens; ~126M parameters |
| TEA pretraining | contrastive GNN-to-LLM-token alignment (details unspecified) | symmetric InfoNCE between GNN node embeddings and mean-pooled frozen LLM token embeddings of each node's text; 10 epochs, 2,000 graphs, 8 graphs/batch, lr 1e-3, temperature 0.07; final loss 4.37; alignment head discarded, GNN then frozen |
| W(f_i) | node text of the queried node(s) in the prompt | added inside `render_question` (`graph/solvers.py`), so training and eval prompts match; applies to all three models |
| Pooling | always pooled (Eq. 2) | **task-conditional, not CLEGR-faithful:** if the question names a source/target node, the pooled slot is zeroed and the named nodes' own embeddings are used; the mean pool is used only for questions with no named node |
| Extra projector inputs | none | **hop radius (scalar) and a 14-way task-type one-hot are concatenated to the graph vector.** Added earlier to stop answer-format leakage (e.g. a number given to a yes/no question). Not in CLEGR, and it gives the graph models explicit task identity that soft_prompt only gets from the question text |
| Training | AdamW, lr 1e-3, 1 epoch, batch 1, 5 seeds | TEA and GraphToken: 3 epochs, effective batch 24 (4 x 6 accumulation), lr 1e-3, weight decay 1e-4, bf16, 1,998 steps, seed 42. soft_prompt: 10 epochs, batch 24, 6,660 steps |
| Training data | CLEGR corpus | 15,990 static QA examples (built from 2,000 synthetic metro graphs, 12 task families, `counterfactual_static_train` on) |
| LLM | not compared | Llama-3.1-8B-Instruct, frozen |

How the three systems differ here:

- **TEA:** frozen contrastively-pretrained GNN, trains only the projector.
- **GraphToken:** the *same* pretrained GNN weights as a warm start, but GNN and projector are trained
  jointly (`train_graph_token`). This is not the from-scratch GraphToken recipe.
- **soft_prompt:** 10 learned soft tokens, no GNN, no graph tokens. It still sees the W(f_i) text.

Evaluations:

- **Static** (oracle QA on single graphs): 2,250 validation and 4,500 test tuples.
- **Exact2x dynamic**: 2,496 sessions per split (26 node counts x 4 session lengths x 3 densities x
  8 tasks, exactly one cell each), 9,360 turns per split, 14 conditions for the graph models.
  Node counts: 16-24 (small), 25-32 (medium), 40-48 (large). Validation and test splits only.

## 3. Static eval

{{STATIC}}

Per-task majority-class accuracy on the same static sets (test): reachability 100% (every answer is
"yes"), cycle_membership 98.4% ("yes"), edge_exists 84.2% ("no"), constrained_reachability 71.9%
("yes"), most_common_attribute 33.2%, node_degree 34.5%, shortest_path 27.1%, filtered_path_count 32.3%,
filtered_neighbor_count 24.1%, path_cost 14.3%, within_hops_count 10.4%, within_hops_list 0.3%. Overall:
44.2% (test), 44.8% (validation). Validation majorities are similar (reachability 100%, cycle_membership 96.3%, edge_exists 75.9%,
constrained_reachability 74.1%). So the yes/no rows above are all *below* majority except soft_prompt's
`reachability` (100%, equal to the majority rate and consistent with always answering "yes"; not verified).

The per-task swings for the graph models between validation and test (TEA `reachability` 28% -> 86%,
`edge_exists` 71% -> 27%) are far larger than any real difference between the splits and look like
the models' yes/no bias flipping. I did not inspect their predicted-answer distributions.

Figure: `figures/ff_static_test.png` (and `_validation`).

## 4. Exact2x dynamic eval

### Accuracy by condition (test; 95% CI in brackets)

{{COND_TEST}}

Validation:

{{COND_VAL}}

Notes on reading this:

- `tool_solver` (oracle solver) is 100% by construction.
- `majority_prior` is **mislabeled in the code**: it answers with `turn.stale_answer`, the answer to the
  pre-edit graph. 68% means an edit changes the correct answer in only ~32% of turns. It is a
  stale-answer baseline, not a majority-class baseline. The real per-task majority baseline over the 8 tasks
  actually present (including `node_count`) is 41.4% / 40.9%.
- The CIs are per-turn and ignore that turns within a session are correlated, so they are optimistic.
- `structure_only` gives the graph tokens of the initial graph with a text-only prompt; `question_only`
  gives no graph tokens. Their difference is the value of having graph tokens at all:
  **+9.3 (TEA) / +5.9 (GraphToken) points on test, but only +2.5 / +1.4 on validation.**
- **Most of that graph-token gain comes from a few tasks** (`figures/ff_graph_gain_test.png`, test split,
  TEA / GraphToken, in points): `most_common_attribute_within_hops` +36 / +38 (where `question_only` scores
  0%, i.e. unusable answers, so this looks like an answer-format effect), `cycle_membership` +30 / +9,
  `constrained_reachability` +24 / +20, `reachability` +7 / +4. `edge_exists` (-13 / -13) and
  `filtered_neighbor_count` (-7 / -8) get *worse* with graph tokens. Without `most_common_attribute` the
  average TEA gain drops from 9.3 to about 5.5 points.
- `oracle_updated_graph`, `structure_only` and `shuffled_graph` are within about 2.5 points of each
  other on both splits for both models (TEA test: 35.3 / 34.6 / 34.7). Re-encoding the edited graph, or
  using the wrong session's graph, makes little difference.
- The text-serialization conditions (`serialized_*`, which are text-only with no graph tokens) are *below*
  `question_only` (19-22% vs ~25%). A plausible reason is that training prompts were short, but I did not test it.

Figures: `figures/ff_conditions_test.png`, `ff_graph_gain_test.png` (graph tokens vs none, correct vs
wrong graph, per task), plus `_validation` versions.

### Per task (test)

`oracle_updated_graph` for the graph models, the single soft_prompt condition for soft_prompt.
`node_count` is the unintended replacement for `path_cost` (Section 7).

{{TASK_TEST}}

Validation:

{{TASK_VAL}}

Per-task majority-class accuracy on exact2x test: constrained_reachability 64.6%, cycle_membership
75.8%, edge_exists 60.9%, reachability 54.9%, most_common_attribute 36.0%, filtered_neighbor_count
22.5%, within_hops_count 8.4%, node_count 3.8%. Against these, the graph models clear the majority rate on
`reachability` (TEA 60.6, GraphToken 64.9 vs 54.9) and, for GraphToken, on `most_common_attribute_within_hops`
(40.1 vs 36.0). TEA is at majority on `constrained_reachability` (65.0 vs 64.6). Everything else is below it,
including `cycle_membership`, `edge_exists`, `filtered_neighbor_count` and `within_hops_count`.
On `most_common_attribute_within_hops`, `question_only` scores 0% (its answers are not usable there) but
soft_prompt, which also has no graph tokens, reaches the same ~36% as the graph models, so that task shows no
graph-token benefit. `node_count` (a global count) is ~0% for every model, including soft_prompt.

### Edit tracking (test): accuracy on turns where the edit changed the answer

{{ANSWER_CHANGING}}

Re-encoding the updated graph (`oracle_updated_graph`) beats the frozen initial graph on turns whose
answer changed (TEA +3.7 points, GraphToken +5.2), which is the clearest benefit I see from the graph channel.
This is a comparison among graph conditions only: `question_only` scores *higher* on these turns than either
graph condition (30.4 / 31.7), and all graph-model numbers here (25-30%) are below soft_prompt's 37.0%.

## 5. Density and scale breakdown (requested separately)

Accuracy by graph density and scale (`oracle_updated_graph` for graph models):

{{DENSITY}}

The density pattern is **not consistent across splits** for TEA and GraphToken: dense is worst on
validation (25.3% / 19.2%) and best on test (44.3% / 37.9%). I therefore do not treat it as evidence
for or against the receptive-field hypothesis. soft_prompt is flat across density and scale (36-39%), as
expected for a model that does not see the graph. Scale shows no consistent trend either. The full
task x density and density x scale cuts for every condition are in the `*_analysis.json` files
(`figures/ff_task_density_test.png` shows task x density).

Why density matters in principle: a 3-layer GraphSAGE can only aggregate information from 3 hops. Mean
graph diameter (hops, 30 generated graphs per cell, same generator as the eval data):

{{DIAMETER}}

Dense graphs have diameter 2, inside the receptive field. The reason for the rest is in the generator
(`_degree_for_target_density`, Watts-Strogatz degree k): **sparse always uses k=2, i.e. a ring lattice
before rewiring**, so its diameter grows almost linearly with n (up to ~26). **Medium uses k=2 for n <= 22**
(the same ring-like setting as sparse, diameter 10-13), k=4 for n=24-32 and k=6 for n>=40 (diameter 5-6).
Dense uses k=10-28. So at small n the medium and sparse bins are essentially the same kind of graph, and the
"medium" bin is only genuinely denser from n=24 up. Figure: `figures/ff_diameter.png`. This mismatch was left
as is to stay faithful to CLEGR (whose graphs average ~26.5 nodes with the same 3-layer GNN).

## 6. Confounds that change how to read the results

- **Majority baselines.** Section 1, 3 and 4. Label skew (e.g. reachability 100% "yes" in the static
  set) means many per-task scores mostly reflect answer priors.
- **W(f_i) leaks the answer on reachability.** In the exact2x test set, whenever the W(f_i) text shows
  a closed endpoint the gold answer is always "no" (338 of 338 cases; 0 "yes"). The rule "closed
  endpoint -> no, else yes" scores (338+642)/1170 = 83.8%, almost exactly soft_prompt's 82.6%, so its
  `reachability` lead is best explained as a text shortcut, not graph reasoning. For the other yes/no tasks a
  closed endpoint does not decide the answer (e.g. `constrained_reachability`: 78 "yes" vs 37 "no" cases
  with a closed endpoint).
- **Unequal training.** soft_prompt: 10 epochs, 6,660 steps. TEA and GraphToken: 3 epochs, 1,998 steps.
- **Code comment vs CLEGR.** The docstring of `TEAGLM.encode_graphs` says the task-conditional readout
  matches CLEGR's baseline. Per `clegr.md`, CLEGR always pools (Eq. 2), so that comment is inaccurate.
- **One seed.** CLEGR reports five. Validation/test gaps of several points for the graph models
  (e.g. TEA 30.4% vs 35.3%) could be seed noise; I did not measure it.
- **`question_only` is not a "no information" baseline** in the usual sense: on `path_cost` it is a
  constant-answer predictor (Section 7).
- **Contrastive alignment was not ablated.** I did not run a TEA variant without it, so its
  contribution is unknown. Alignment quality was not measured beyond the final loss.

## 7. path_cost supplement (and the generator bug)

**Bug:** `_queries()` in `data/multiturn.py` had no `PATH_COST` branch, so it produced no candidates and
every `path_cost` turn fell back to `node_count`. The main exact2x sets therefore contain `node_count`
sessions where `path_cost` was intended (1,170 turns per split, zero `path_cost`). The static corpus
is unaffected. **Fixed** by adding `PATH_COST` to the pair-query branch; the local test suite passes.
Other sessions in the main set are unchanged (each session has its own seed), so they remain valid.

Supplement: a `path_cost`-only exact2x set (624 sessions, 1,170 turns per split, same grid) generated
with the fixed code and evaluated with the same checkpoints (accuracy %, validation / test):

{{PATHCOST}}

What the predictions look like (TEA test): with graph tokens the model answers 1, 2, 3, 4 or "yes"
and almost never "unreachable" (1 of 1,170), while 29.7% of gold answers are "unreachable" and costs go
up to 40. `question_only` scores ~30% only because it answers "unreachable" for 1,153 of 1,170
questions. GraphToken behaves the same way (6.4% with graph tokens; 1,154 "unreachable" answers under
`question_only`). soft_prompt, with no graph tokens, spreads its answers over numbers and "unreachable"
(211 of 1,170) and scores 23.3% / 23.9%, well above the graph models but still below the constant
"unreachable" answer (31.4% / 29.7%). So the graph tokens carry essentially no path-cost information, and
every model is below the always-"unreachable" and stale-answer baselines.

Figure: `figures/ff_pathcost_test.png`.

## 8. Reproducing

Configs (all under `configs/`): `v2_clegr_extended_{tea,graphtoken,soft_prompt}_fixed_final.yaml` (training and
static eval), `v2_clegr_extended_exact2x_{tea,graphtoken,soft_prompt}_fixed_final.yaml` (exact2x eval),
`v2_clegr_extended_exact2x_pathcost_{...}_fixed_final.yaml` (path_cost supplement).

```
python -m graph_modi.cli pretrain-gnn     --config configs/v2_clegr_extended_tea_fixed_final.yaml     # contrastive, TEA only
python -m graph_modi.cli train-projector  --config configs/v2_clegr_extended_{tea,graphtoken,soft_prompt}_fixed_final.yaml
python -m graph_modi.cli static-eval      --config <training config> --split {validation,test}
python -m graph_modi.cli evaluate         --config configs/v2_clegr_extended_exact2x_<model>_fixed_final.yaml
python scripts/analyze_exact2x.py <evaluation.json> <data_dir> <analysis.json>
```

GraphToken loads the TEA GNN checkpoint (`outputs/v2_clegr_extended_tea_fixed_final/gnn/gnn.pt`); its
GraphSAGE and tensorizer config must match it exactly. Eval batch size 128 for exact2x (512 ran out of
memory on a 48 GB MIG slice). Code touched: `models/bert_tensorizer.py` (new), `training.py`
(contrastive pretraining), `cli.py`, `graph/solvers.py` (W(f_i)), `data/multiturn.py` (path_cost fix).

## 9. Suggested next steps

1. Report against per-task majority baselines and macro-average accuracy, or rebalance the yes/no labels.
2. Remove or mask node `status` from W(f_i) (or ablate W(f_i)) to see how much of soft_prompt's
   advantage is the shortcut.
3. Equalize epochs across the three systems and run 3-5 seeds.
4. Decide whether to fix the sparse-density degeneration (k=2 ring, diameter up to ~26; medium is also k=2 for
   n <= 22) or keep it for CLEGR fidelity.
