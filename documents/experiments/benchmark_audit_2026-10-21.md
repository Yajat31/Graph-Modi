# metro_v3 benchmark cleanliness audit (2026-10-21)

Scope: is the benchmark itself clean? Labels, graph structure, split hygiene, prompt text, balance, sessions, counterfactual training pairs, structural shortcuts, latent hazards.

Method: `scripts/audit_benchmark.py` regenerates metro_v3 (seed 42, `configs/v3_tea.yaml` parameters) into `/tmp/graphmodi_bench`. It does not use the CLI, so `datasets/` and `log.md` are untouched. Everything is then checked with code that does not call the generator's solvers (independent BFS and bridge-based solvers, independent edit replay). Output: `results/benchmark-audit-20261021/{results.json,manifest.json}`. Run time is about 4 minutes. Result: 46 pass, 10 fail, 8 warn, 44 info.

## Verdict

**The benchmark is not clean. One real label bug and two smaller session defects need fixing before any number is re-reported.** The rest of the benchmark is in good shape: structure, split hygiene, prompt text and answer balance all pass.

## Findings

### B1. Critical: `cycle_membership` labels are wrong for about 7% of items

`_cycle_member` in `src/graph_modi/graph/solvers.py` marks a station as being on a cycle only if the *first* neighbour can reach some other neighbour without passing through the station. If the first neighbour is a pendant or bridge branch and two other neighbours are connected, it wrongly answers "no". A station is on a cycle iff any incident edge is not a bridge; the independent solver uses that rule.

Disagreements (stored label "no", truth "yes"; the reverse never occurs):

| Set | Wrong / cycle items | Share | Share of degree ≥ 3 queries | Stored yes-rate | True yes-rate |
|---|---|---|---|---|---|
| static train | 142 / 2000 | 7.1% | 15.3% | 0.500 | 0.571 |
| static validation | 16 / 198 | 8.1% | 15.0% | 0.500 | 0.581 |
| static test | 29 / 396 | 7.3% | 14.1% | 0.500 | 0.573 |
| session turns, validation | 34 / 495 | 6.9% | | | |
| session turns, test | 27 / 495 | 5.5% | | | |
| counterfactual train pairs | 38 / 2996 pairs | | | | |

Errors occur only for degree ≥ 3 stations (0 errors at degree 1–2). The independent solver agrees with the library's answer on every other task. The one extra flag in static train is a separate issue: 221 of 334 counterfactual "before" graphs for `most_common_attribute_within_hops` have a tied mode, and the stored label breaks the tie alphabetically (train only, not fixed). In sessions, 47 (validation) and 35 (test) stale answers are also wrong, for the same reason.

Consequences:
- The cycle ceiling on the test set is about 92.7% for a perfect reasoner, not 100%.
- The "balanced 50/50" cycle task is balanced on the wrong labels. The true yes-rate is about 57%.
- Any cycle accuracy in the September and October tables measures agreement with a faulty rule on 7% of items.
- The shortcut classifier (degree-only gets 91.7% on cycle) is partly fitting the bug: the bug removes yes-labels from high-degree stations.

Fix: replace the first-neighbour test with a bridge test or a pairwise-neighbour connectivity test, regenerate, then rerun everything that touches cycle.

### B2. Medium: edit programs that do nothing, or cancel themselves

- 7 of 2754 (validation) and 7 of 2787 (test) non-NOOP programs leave the graph unchanged, so they are NOOPs labelled as edits.
- 44 (validation) and 39 (test) programs touch the same node or edge twice.
- 3 (validation) and 1 (test) edit turns return the graph to an earlier state.

These are rare, and each affects well under 2% of turns. However, they inflate the "edit" counts while exercising no write capability.

### B3. Medium: session turns drift away from the static training distribution

- The closed-station fraction rises with turn index, from 0.08 at turn 0 to 0.21 at turn 7. The static train mean is 0.122, and static graphs are generated with 1–3 closed stations.
- Connectivity falls from 0.96 at turn 0 to 0.83 at turn 7, because DEL can disconnect the graph (no connectivity guard in the edit sampler).
- Long sessions are therefore partly out-of-distribution for a static-trained encoder. Session length is confounded with state drift, which matters for the length-trend claim.

### B4. Medium: the structural shortcut is the dominant signal

A lookup table on degree, endpoint status and same-line, fitted on train, scores on the static test set:

| Task | Lookup | Source of signal |
|---|---|---|
| node_degree | 100.0 | degree |
| reachability | 94.4 | closed endpoint implies "no" (176 of 198 "no" answers have one) |
| cycle_membership | 91.4 | degree-1 implies "no" (165 of 198) |
| edge_exists | 68.4 | min degree |
| constrained_reachability | 67.9 | degree |
| shortest_path | 43.4 | |
| other 5 tasks | 5–30 | |

- 8-task mean: **63.9%**. This reproduces the committed gradient-boosting ceiling of 64.0 to within 0.1 point, using only a lookup table. The best trained model is 66.5%.
- Multi-turn (8 tasks, test): stale answer 59.0, shortcut on the updated graph 62.9, shortcut on the stale graph 45.5. The committed gradient-boosting figures were 64.6 and 45.5.
- Status-only reaches 94.4% on reachability, so the test suite cannot show graph reasoning on the 70% gate without an anti-shortcut slice.
- Hard-negative handling is partial. Of reachability "no" answers, 176 of 198 involve a closed endpoint, and P(no | closed endpoint) is exactly 1.0 in both train and test. Both-open "no" is only 10–12%.

This confirms the first audit's finding C1 and its numbers independently.

### B5. Low: latent generator artifacts a model could exploit

- **Node order leaks structure.** In 396 of 396 test graphs the last 2–3 entries of `graph.nodes` are exactly degree-1 terminals. They hold 1008 of 1213 (83%) of all degree-1 stations. Node list position correlates with BFS depth from the first node at +0.62. Node ids are permuted, but list order is not. A GNN fed positions, or an LLM reading serialised order, can exploit this. Shuffle node and edge order at generation time.
- **Added edges are always `transfer`** with no attributes and weight 1.0. About 27% (492 of 1844) join stations on the same line, where an original edge would be a track edge. An edge's relation and attributes therefore reveal whether it was added. A write-path model can learn "ADD means transfer" without reading the revision.
- **NOOP rate is 37–38%**, against the proposal's 15–20% (first audit, row "NOOP ~15–20%"). Repeated identical queries within a session: 268 of 4455 (validation) and 296 of 4455 (test).
- SET only flips `status`, so for 7 of 9 tasks a SET turn never changes the answer. Measured P(answer changes | set_only) = 0 for those 7 tasks, 1.0 for reachability and 0.98 for shortest_path.

### B6. Informational: counterfactual training pairs

- Pairs share question text exactly, answers always differ, and graphs always differ (no failures).
- 38 pairs have a wrong label (B1).
- Reachability: 79 of 298 pairs are status-only edits, and 72 of those flip a queried endpoint. Shortest path: 76 of 474, 40 of which flip an endpoint. Training pairs therefore teach "endpoint closed" heavily for reachability.

## Passed checks

| Area | Result |
|---|---|
| Reproduction | The pretraining-data hash regenerates as `80b1df4735af…`, identical to the original. Operation and answer distributions, the answer-changing fraction (40.2% validation, 40.7% test), and static label counts all match the committed `dataset_audit.json` exactly. **The provenance gap from the first audit (datasets missing, hash unverifiable) is closed for the data.** Weights and per-turn rows are still missing. |
| Labels (other tasks) | 10 of 11 static tasks, and 8 of 9 dynamic tasks, match the independent solver on every item (apart from the tied-mode counterfactual items noted in B1). Independent edit replay matches the library replay on all sessions. No edit anomalies (set-noop, add-existing-edge, del-missing-edge). |
| Structure | 3000 train, 198 validation and 396 test graphs, plus 2376 session starts, with no issues: no duplicate ids or names, parallel edges or self-loops; connected; diameter caps honoured; metadata consistent; 2–3 terminals; attribute domains valid; closed counts in range; density bins separate (2.75–2.86 / 3.73–3.86 / 4.92–5.07); test grid is exactly 12 per cell over 33 cells. |
| Split hygiene | RNG seeds unique. Zero attributed-graph (WL-hash) overlap between train and either eval split, between validation and test, and between static and session sets. Zero topology-only overlap. Zero identical (graph, question) pairs with train. 3 (validation) and 6 (test) identical question strings with train, which is expected from a 240-name pool. Graph ids are reused across static and session sets (names only; different graphs). |
| Prompt text | All 30,526 prompts match the W(f_i) spec. No "status" or "closed" wording. The queried attribute is hidden on Facts tasks. No categorical answer appears in the prompt. The library's text-only leak audit flags nothing. |
| Balance | Yes/no tasks are within 0.5 points of 50/50 in train, validation and test. Train/test answer marginals have TV ≤ 0.053. Constrained reachability is independent of status (gap 0.05, as it should be). |

## Recommended order of work

1. Fix `_cycle_member` (B1), regenerate the data, and rerun the cycle checks. This changes labels, so a retrain or re-eval is required for every cycle number. Until then, report cycle separately or exclude it.
2. Filter or re-sample net-noop and self-cancelling programs, and add a connectivity guard to DEL (B2, B3).
3. Shuffle node and edge order in graph construction, and give added edges track relations on same-line pairs (B5).
4. Add anti-shortcut evaluation slices (reachability with open endpoints and "no" answers; cycle on degree ≥ 3; edge_exists with min degree > 1), so the 70% gate is not reachable with 64% from degree and status alone (B4).

## Reproduce

```bash
PYTHONPATH=src python3 scripts/audit_benchmark.py            # uses /tmp/graphmodi_bench cache if present
```

First run generates the data, which takes about 4 minutes.

## Update 2026-10-22

B1 is fixed in `src/graph_modi/graph/solvers.py` (see the log row of the same date). On regenerated data the cycle labels have zero mismatches. The reproduction numbers above describe the data made with the old solver; reproduce it at commit `5549ac5`.
