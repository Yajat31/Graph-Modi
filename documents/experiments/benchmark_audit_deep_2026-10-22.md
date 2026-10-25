# metro_v3 data audit, second pass (2026-10-22)

Builds on [benchmark_audit_2026-10-21.md](benchmark_audit_2026-10-21.md). Two things are new: the **degree-balanced variant** (`data.degree_balance: true`, `datasets/metro_v3_bal`) is audited, and a set of checks looks at how the data lets a model score without doing the intended work.

Scripts and outputs:
- `scripts/audit_benchmark.py [--degree-balance]` and `scripts/audit_benchmark_deep.py`.
- Plain variant: `results/benchmark-audit-20261021/{results,deep_results,manifest}.json`.
- Balanced variant: `results/benchmark-audit-bal-20261022/…`.
- Both datasets are regenerated at seed 42 in `/tmp`. The CLI, `datasets/` and `log.md` are untouched.

Scope note: the v3 generator is a random recursive tree plus uniform random chords plus 2–3 pendant terminals. It is not Watts-Strogatz, which belongs to the older v1/v2 generator in `multiturn.py`, and the midterm report names no family. Everything below is about v3.

## 1. Critical: the degree-balanced variant is mostly mislabeled on `cycle_membership`

The balanced variant restricts cycle queries to stations of degree ≥ 2 and forces 50/50 answers. Real "no" answers at degree ≥ 2 are very rare, so the 50/50 quota is filled almost entirely by the solver bug (B1 in the first audit: wrong "no" for degree ≥ 3 stations).

| Split | Stored labels wrong | Stored yes-rate | True yes-rate | True "no" items |
|---|---|---|---|---|
| train | 383 / 1154 (33.2%) | 0.50 | 0.83 | 194 |
| validation | 43 / 103 (41.7%) | 0.51 | 0.92 | 8 |
| test | 80 / 202 (39.6%) | 0.51 | 0.91 | 19 |
| session test turns (cycle) | 128 wrong / 495 across the split | | | 34 true "no" turns |

On the balanced test set, **80 of the 99 stored "no" answers are false** (the 19 true "no" answers are the only correct ones). A model can only score above chance by reproducing the bug, which depends on the order of a station's neighbours in the edge list. Whole-dataset stats: static train 23,137 tuples, validation 2,079, test 4,147. Session gold mismatches are 139 (validation) and 128 (test); counterfactual pairs mislabeled are 57 of 2995.

The balanced variant also changes the benchmark in other ways:
- **Fewer eval items.** Test has 202 cycle items (not 396), 394 reachability, and 383 constrained reachability. The two-sided filter drops graphs that cannot supply both answers.
- **The text-only leak audit flags `attribute_check` and `constrained_reachability`** (the plain variant flags nothing).
- **Session cycle answers are not balanced**: P(yes) = 0.67 in test turns.
- **The committed shortcut ceiling does not describe this data.** A lookup table scores 57.1% on the 8 tasks (plain: 63.9%). Degree balancing does reduce the shortcut on `edge_exists`, `constrained_reachability` and `cycle`, which fall to 53–58%. Reachability stays at 93.7% from endpoint status alone, because no status balancing was added.
- Note that `documents/experiments/log.md` says regenerating on the balanced data was "proposed, not run". I found no committed results for `bal_*` configs, so nothing reported is affected. Do not run them until the cycle solver is fixed.

## 2. In the plain variant, cycle is "is the station a leaf?"

With the corrected labels, true "no" answers on test are 165 of 169 at degree 1 and only 4 non-leaf (2.4%). On train it is 816 of 858 at degree 1. In sessions, 207 of 219 true "no" turns are degree ≤ 1. So after the label fix, `degree == 1` predicts the answer about 98% of the time, and a degree-only classifier would score about 99%. The task as sampled does not test cycle reasoning. This matches your point that the graphs are dense and cyclic: nearly every non-leaf station is on a cycle.

## 3. High: the question is chosen to be about the edit, so revision text answers many turns

Queries in sessions are drawn from the pool of answer-changing queries (`pool = changing or candidates`), so the question tends to mention the edited station or edge. On test edit turns:

| Task | Question mentions an edited node (observed vs chance) | P(answer changes) if mentioned / if not | Rule that uses only the revision text: fires on / precision |
|---|---|---|---|
| edge_exists | 68% vs 35% | 0.88 / 0.00 | 60% of edit turns / 1.00 |
| reachability | 71% vs 36% | 0.87 / 0.17 | 46% / 1.00 |
| shortest_path | 76% vs 35% | 0.96 / 0.70 | 20% / 1.00 |
| node_degree | 82% vs 19% | 0.99 / 0.00 | |
| filtered_neighbor_count | 79% vs 18% | 0.98 / 0.00 | |
| constrained_reachability | 56% vs 34% | 0.72 / 0.06 | |
| cycle | 36% vs 19% | 0.80 / 0.06 | |
| within_hops_count | 46% vs 19% | 0.94 / 0.61 | |
| most_common_attribute | 44% vs 19% | 0.90 / 0.40 | |

- For `edge_exists`, `node_degree` and `filtered_neighbor_count`, the answer changes only if the question mentions an edited station or pair. The rules are: "ADD EDGE A B and the question asks about A–B means yes", "DEL EDGE A B means no", and "a closed endpoint means no (or unreachable)". They are exact whenever they fire. The `node_degree` rule also needs the pre-edit degree, so it is not text-only.
- A graph-blind model that reads the revision text can therefore answer a large share of the answer-changing turns. This is consistent with the soft prompt that reads the revision text (34.2%) landing within 1–4 points of the oracle re-encoding runs (35.1% TEA, 37.9% GraphToken).
- The 38% NOOP turns have answer == stale answer by construction.

## 4. High: the write path is a regex task

- All 4,455 test turns (and 4,455 validation turns) are parsed back to the gold edit program exactly by a regex parser of about 25 lines.
- There are only 134 distinct utterance skeletons once station names are masked. Station edits use two templates ("X is closed now." and "X has reopened."), because v3 only uses the `direct` family. The paraphrase families in `multiturn.py` are unused.
- Every edge edit (2,313 of 2,313 in test) is stated as the literal string `ADD EDGE <A> <B> <relation>` or `DEL EDGE …`, so it is a copy task, not language understanding.
- Predicting NOOP everywhere is exact on 37.4% of turns.
- Write-path edit accuracy therefore measures format-following by the frozen Llama, not robustness to paraphrase.

## 5. Medium: train/eval shift

- 35% of test sessions start with 0 closed stations (413 of 1188). Static training always has 1–3 closed stations, so there are no training graphs with all stations open.
- The state keeps drifting after turn 0 (closed fraction 0.08 to 0.21, connectivity 0.96 to 0.83); see B3 in the first audit.
- In sessions, 205 of 249 reachability "no" answers have a closed endpoint, and 201 of 246 cycle "no" answers are degree-1 stations. The same two shortcuts as in static data carry over.

## 6. Low: edge attributes

All 13,008 original edges sampled carry four attributes (`built`, `has_aircon`, `line_color`, `line_stroke`), and `line_color` is "transfer" exactly when the relation is `transfer`. Edges added by ADD carry no attributes, and 27% of them join same-line stations but are still labelled `transfer`. The node-only GNN featurizer does not read edge attributes, so no model is known to exploit this. A text or serialization-based reader could.

## What this changes in the first audit

- B1 now has two consequences: about 7% of cycle labels are wrong in the plain variant, and 33–42% are wrong in the balanced variant. In the balanced variant the cycle task cannot be scored meaningfully.
- Finding C1 of the first audit (shortcut ceiling) stands for the plain data. For the balanced data the ceiling is lower, but reachability remains solvable from the endpoint status.
- The write-path numbers in the reports should be described as copy and template parsing.

## Recommended order of work

1. Fix `_cycle_member`, then regenerate both variants and rerun `audit_benchmark.py` (it should report zero label mismatches).
2. After the fix, re-sample cycle queries so "no" answers are non-leaf stations on bridge chains, otherwise the degree-1 shortcut remains. The generator needs a construction that produces such stations, since only 2–5% of true "no" answers are non-leaf.
3. Add endpoint-status balancing for reachability and shortest_path (the existing degree balancing does not touch it).
4. Choose session queries independently of the edit, or report accuracy separately for "mentioned" and "not mentioned" turns, so revision text does not answer the question alone.
5. Use all three paraphrase families (and add free-text edge edits) so write-path accuracy is not a regex.
6. Include 0-closed graphs in static training, or start sessions with at least one closed station.

## Third pass: semantics, metadata, ties (`scripts/audit_benchmark_extra.py`, run on data from the fixed solver)

Passed: no query asks about a station and itself; no (graph, question) key has two different answers; one train item is repeated and none in validation or test; no station name is a substring of another; session metadata (length, node count, edit count, NOOP tag, density) matches what the sessions contain; the `shortest_path` hop tag follows its answer.

Problems:

1. **Constrained reachability has an unstated convention.** The question says "without passing through any station where X is V", but the solver exempts the two endpoints. On test, 263 of 396 items have an endpoint with the avoided value, and 129 of those (33% of all items) become "no" under the literal reading. Train: 768 of 2140. The text should state the convention, or the sampler should avoid such endpoints.
2. **Tied modes in the stale graph.** In 135 of 495 `most_common_attribute_within_hops` session turns (27%, both splits) the stale graph has a tied mode, so the stale answer is an alphabetical tie-break, and 133–134 of them are counted as "answer changed". The sampler only checks that the mode is unique in the updated graph. The stale-answer baseline and the answer-changing statistic are affected for this task. Together with the 221 tied counterfactual "before" graphs in training, ties are a recurring problem for this task.
3. **`hop_depth` is a task label.** 10 of the 11 tasks always get the same tag (for example reachability is always "mid", cycle and constrained reachability always "global", all the count and lookup tasks "local"). Only `shortest_path` varies (67 local, 132 mid, 197 global on test). Any breakdown by hop depth is mostly a breakdown by task.
