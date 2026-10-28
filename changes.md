# What changed since the mid-submission (GraphModi.pdf)

*Written 2026-10-28 against branch `graphtoken-e2e` (merged into `graphmodi-experiments` as
`5f39caf`). The mid-submission report (`GraphModi.pdf`, 2026-10-02) used the September v3 runs.
Numbers come from committed files under `documents/experiments/results/`, except §3.5, §3.7 and
§3.8, which were computed from the per-turn outputs on the GPU server (`/data/lmw/graph-modi`).
All multi-turn numbers are the v3 test split (1,188 sessions, 4,455 turns) unless marked.*

**In one paragraph.** Three bugs in the September pipeline mean the mid-submission compared two
copies of the same untrained model: TEA's GNN was never pretrained, GraphToken's GNN was never
trained, and both used mean aggregation, which cannot count neighbours. Once GraphToken actually
trains with sum aggregation it reaches 57% on the multi-turn grid (3 seeds, ±0.2), and a warm
start from a pretrained GNN reaches 64%, against 35–38% in the report. Re-encoding the edited graph
now beats stale graph tokens by 12–17 points instead of 2.6–4.6. But the gains sit almost entirely
on questions a 4-feature degree/station-status classifier also gets right, so they show the tokens
carry node-local state, not multi-hop reasoning. A new benchmark (metro_v4) was built to remove
those shortcuts; no model has been run on it yet.

---

## 1. Claims in the mid-submission and their status

| # | Claim in GraphModi.pdf | Status now | Evidence |
|---|---|---|---|
| 1 | TEA-style: InfoNCE-pretrained GNN, then frozen (§4) | **Wrong for those runs.** The GNN was frozen *during* pretraining, so TEA used a random GNN. Fixed. | Pretraining loss 1.75 after the fix vs 4.91 chance (`pretrain_pre_base_metadata.json`) |
| 2 | GraphToken-style trains GNN and projector jointly (§4; Table 9 "Done") | **Wrong for those runs** (the appendix says so): GraphToken was a second TEA run. Fixed; joint training now runs. | `training.py` gives the GNN its own optimizer group |
| 3 | Headline: re-encoded TEA 35.1 / GraphToken 37.9 multi-turn; static 36.3 / 40.3 | **Superseded.** Best is now 63.7 multi-turn, 65.8 static (one seed). See §2. | `stats_test_final.json` |
| 4 | "GraphToken vs TEA" differences | **Run-to-run noise.** TEA vs its second run differ by 2.8 [1.4, 4.3] on identical recipes. | `stats_test_existing_runs.json` |
| 5 | Yes/no labels 50/50; leak audit clean (§4, Table 1) | **False for cycle membership.** The solver tried only a station's first neighbour; 29/396 test and 16/198 validation static cycle labels were wrong (all should be "yes"). Fixed in `47a6eab`. | `benchmark_audit_2026-10-21.md` |
| 6 | Effect of the cycle bug on reported numbers | Small: rescoring the September rows moves overall static by ≤0.7 (TEA 36.3 → 36.9, GraphToken 40.3 → 41.0), multi-turn by ≤0.6. The corrected question-only prior is 33.5, not 32.8. | Rescored from committed rows |
| 7 | "Read path is the bottleneck: ~35%, count tasks below 20%" (§7) | **Partly superseded.** With sum aggregation node degree is 98–100%; within-hops count and filtered-neighbour count are still 18–40%. | §3.3 |
| 8 | Correct vs shuffled graph +4.9 (TEA) / +7.2 (GT) is "our clearest evidence the tokens encode the graph" | Now +26 to +33 for working models. But the collapsed GraphToken also gets +2.6, so the September gap was not diagnostic. | Table A |
| 9 | Stale tokens degrade over turns; re-encoding holds (RQ2, Fig. 1) | **Not supported.** No model shows a length trend on either split; the benchmark has as many answer-changing turns in 1-turn as in 8-turn sessions. Accumulation shows only on "unchanged" turns. | §3.5 |
| 10 | Re-encoding beats stale tokens by ~5 on answer-changing turns | Direction holds and is much larger now (+12 to +17 overall), but see §3.4: almost all of it is on shortcut-solvable turns. | Table B, §3.4 |
| 11 | TEA only 0.9 above the soft prompt, not significant | That soft prompt reads the revision history. Against the like-for-like soft prompt without history: TEA +3.2 [1.6, 5.0]; GraphToken (sum) +25.4 [23.0, 27.8]; warm start +31.8 [29.5, 34.1]. | `stats_test_final.json` custom pairs |
| 12 | Model-written edits cost 0.2–0.3 points (RQ1) | **Still holds** for every variant (gap 0.0–0.8). Edit accuracy is 95.2% for all, because the write path is text-only and does not depend on the GNN. Revisions are still templated. | §3.6 |
| 13 | Single seed (Limitations) | GraphToken (sum) has 3 seeds: 57.3 / 57.0 / 56.8, pairwise differences not significant. Warm start and all TEA variants are still single-seed. | Table A |

---

## 2. New results

### Table A — every variant on the v3 multi-turn grid (test)
"Updated" = `oracle_updated_graph` (gold graph re-encoded each turn). "Stale + history" =
`frozen_graph_history`; "Stale" = `structure_only`.

| Variant | Static | Updated | Stale + history | Stale | Shuffled | Graph once | Updated − stale+hist | Updated − shuffled |
|---|---|---|---|---|---|---|---|---|
| TEA, September (random frozen GNN, mean) | 36.3 | 35.1 | 32.5 | 32.3 | 30.2 | 29.8 | 2.6 | 4.9 |
| TEA run 2 (September "GraphToken") | 40.3 | 37.9 | 33.3 | 33.1 | 30.7 | 29.6 | 4.6 | 7.2 |
| GraphToken joint, mean aggregation (collapsed) | 33.0 | 34.1 | 31.3 | 31.3 | 31.4 | 29.3 | 2.7 | 2.6 |
| TEA, random frozen mean GNN (control) | 37.9 | 36.6 | 32.4 | 32.3 | 30.6 | 30.8 | 4.2 | 5.9 |
| TEA, pretrained mean GNN, lr 1e-3 | 34.4 | 32.1 | 30.3 | 30.0 | 28.4 | 30.2 | 1.8 | 3.8 |
| TEA, pretrained **sum** GNN | 45.2 | 41.8 | 37.0 | 37.1 | 30.2 | 32.2 | 4.8 | 11.6 |
| TEA, random frozen **sum** GNN | 53.3 | 56.3 | 44.3 | 43.7 | 30.9 | 33.8 | 11.9 | 25.4 |
| **GraphToken, sum, seed 42 / 43 / 44** | 52.7 / 53.0 / 53.1 | **57.3 / 57.0 / 56.8** | 44.9 / 44.7 / 44.2 | 44.5 / 44.6 / 43.7 | 30.6 / 30.9 / 30.5 | 34.2 / 34.5 / 34.3 | 12.3 / 12.3 / 12.6 | 26.6 / 26.1 / 26.4 |
| **GraphToken warm start** (pretrained sum GNN + joint training) | **65.8** | **63.7** | 46.2 | 45.9 | 30.8 | 34.6 | **17.4** | **32.9** |

Reference points (test): soft prompt with history 34.2 [32.3, 36.0], without history 31.9
[30.1, 33.6]; question only 27.7; token-matched history 30.4; per-task majority 29.2; stale answer
59.3 (59% of turns keep their pre-edit answer). Multi-turn session-clustered 95% intervals:
GraphToken sum 57.3 [54.9, 59.6], warm start 63.7 [61.3, 65.9], TEA random sum 56.3 [54.0, 58.5],
TEA pretrained sum 41.8 [39.9, 43.8], TEA September 35.1 [33.4, 36.8].

### Table B — re-encoding gain, paired (test, session-clustered 95% CI)

| Model | Updated − stale+history | Updated − stale | Updated − shuffled |
|---|---|---|---|
| TEA, September | +2.6 [1.8, 3.4] | +2.7 [2.0, 3.5] | +4.9 [3.7, 6.1] |
| TEA, pretrained sum | +4.8 [3.6, 6.0] | +4.7 [3.6, 5.9] | +11.6 [9.6, 13.6] |
| TEA, random frozen sum | +11.9 [10.3, 13.6] | +12.5 [11.0, 14.1] | +25.4 [23.0, 27.8] |
| GraphToken sum (3 seeds) | +12.3 to +12.6 | +12.4 to +13.1 | +26.1 to +26.6 |
| GraphToken warm start | +17.4 [15.7, 19.2] | +17.8 [16.0, 19.5] | +32.9 [30.3, 35.3] |

Across models (updated graph): warm start − GraphToken sum = +6.4 [4.9, 7.9]; GraphToken sum −
TEA random sum = +1.0 [−0.1, 2.1] (not significant); TEA random sum − TEA pretrained sum = +14.5
[12.1, 16.9]; GraphToken seeds differ by 0.2–0.4, all n.s.

---

## 3. Trends: how the benchmark behaves in multi-turn

### 3.1 The re-encoding gain grows with how well the model reads the graph
Across all eleven variants, the gap between re-encoded and stale tokens tracks static accuracy:
models at 33–40% static gain 1.8–4.6 points, models at 53% gain ~12, and the warm start at 66% gains 17.
Two floors stay fixed while the updated condition climbs: the **shuffled graph stays at 28–31% for
every model**, and **graph-once-then-text stays at 29–35%**. So the multi-turn grid rewards reading
the *current* graph, and a model that reads the graph badly cannot show a re-encoding benefit at
all. This is why the September gap (2.6) was so small.

### 3.2 Revision text in the prompt never helps the graph models
For every variant, "stale tokens + revision history" equals "stale tokens, no history" within
0.6 points (Table A, columns 4 vs 5). The models were never trained with history in the prompt,
so they ignore it. The only system that uses history is the soft prompt (+2.3 over its no-history
version). Re-encoding is the only route by which edits reach the graph models.

### 3.3 What each design change unlocks (static test, per task)

| Task | TEA Sept | GT collapsed | TEA pretrained sum | TEA random sum | GT sum s42 | GT warm | Soft |
|---|---|---|---|---|---|---|---|
| edge exists | 52.5 | 50.0 | 58.8 | 69.4 | 72.5 | 71.0 | 48.0 |
| reachability | 57.8 | 50.0 | **84.1** | 53.8 | 51.8 | **94.7** | 50.0 |
| constrained reachability | 53.8 | 50.0 | 55.8 | 69.4 | 70.2 | 71.2 | 48.0 |
| cycle membership | 53.5 | 49.7 | 72.5 | 92.9 | 92.2 | 91.4 | 49.2 |
| node degree | 12.1 | 9.8 | 27.5 | **98.5** | **100.0** | **98.7** | 14.4 |
| filtered neighbour count | 17.7 | 22.0 | 21.2 | 36.6 | 37.9 | 40.4 | 19.9 |
| shortest path | 16.9 | 16.2 | 15.4 | 32.3 | 32.8 | 46.5 | 16.7 |
| within-hops count | 10.9 | 9.6 | 12.9 | 20.7 | 19.7 | 18.2 | 8.6 |
| most common attribute | 32.3 | 28.0 | 34.8 | 35.4 | 28.5 | 50.0 | 28.0 |
| attribute lookup | 43.2 | 28.8 | **60.1** | 30.1 | 26.5 | **93.4** | 27.3 |
| attribute check | 48.2 | 49.2 | 54.0 | 47.0 | 48.0 | 48.0 | 51.0 |

The pattern is clean:
- **Sum aggregation unlocks structure counts** (node degree 12 → 98–100, cycle 54 → 92, edge
  existence and constrained reachability ~70), whether the GNN is random or trained.
- **Contrastive pretraining unlocks node content** — station status (reachability 84) and
  attributes (attribute lookup 60) — which a random GNN cannot read (54, 30).
- **The warm start combines both** (reachability 94.7, attribute lookup 93.4, degree 98.7) and is
  the only October variant that moves shortest path (46.5) and most-common-attribute (50.0).
- Within-hops count, filtered-neighbour count and attribute check stay near their priors for every
  model: multi-hop counting and fact verification are still unsolved.
- Pretraining alone, frozen, *hurts* overall (45.2 vs 53.3 static, −14.5 multi-turn) because it
  loses the degree signal; it only pays off when joint training follows.

### 3.4 The gains live on shortcut-solvable turns
A gradient-boosting classifier using only node degree, endpoint status, graph size and same-line
scores 64.0% static (8 tasks) and, on the multi-turn grid, 64.6% with the updated graph vs 45.5%
with the stale graph — a +19.1 "re-encoding gain" with no graph reasoning at all.

| Multi-turn, 8 tasks (test) | All (n=3,960) | Shortcut-right turns (n=2,559) | Shortcut-wrong turns (n=1,401) |
|---|---|---|---|
| Shortcut classifier, updated / stale | 64.6 / 45.5 | — | — |
| GraphToken sum: updated / stale+history | 60.5 / 46.4 | 80.5 / 60.8 | 24.0 / 20.1 |
| GraphToken warm: updated / stale+history | 65.9 / 46.2 | 89.8 / 61.6 | 22.4 / 18.0 |
| TEA random sum: updated / stale+history | 59.3 / 45.8 | 78.9 / 59.8 | 23.5 / 20.1 |
| Soft prompt (history / no history) | 35.2 / 32.4 | 42.6 / 39.0 | 21.6 / 20.4 |
| Shuffled graph (any model) | 31.3–31.5 | 37.1 | 20.6–21.3 |

On the turns the shortcut gets wrong, every model scores 18–24% and the re-encoding gain shrinks
to +3.4 to +4.4. The static picture is the same: on the 1,142 static items the shortcut gets wrong,
all models (including September TEA) score 22–26%. Conclusion: the working models use the graph
tokens for node-local state (degree, status, attributes) and re-encoding delivers the *updated*
local state; the v3 benchmark cannot show multi-hop reasoning or real state tracking beyond that.

### 3.5 Session length, turn index and answer changes (October models, from the GPU server)
The October per-turn rows are not in the repo; they are on the GPU server
(`/data/lmw/graph-modi/outputs/audit/*/evaluation.json`). The breakdowns below were computed from
them and reproduce the committed overall numbers exactly (e.g. GraphToken sum 57.26, warm start
63.66). Cells: 297 / 594 / 1,188 / 2,376 test turns for 1 / 2 / 4 / 8-turn sessions.

**Test accuracy by session length (updated / stale + history; gain):**

| Model | 1 turn | 2 turns | 4 turns | 8 turns |
|---|---|---|---|---|
| TEA September | 35.7 / 34.7 (+1.0) | 38.0 / 34.8 (+3.2) | 34.8 / 31.9 (+2.9) | 34.6 / 31.7 (+2.9) |
| TEA pretrained sum | 37.7 / 33.7 (+4.0) | 43.9 / 37.9 (+6.1) | 41.8 / 37.6 (+4.1) | 41.8 / 36.8 (+4.9) |
| TEA random sum | 50.8 / 39.7 (+11.1) | 59.8 / 46.3 (+13.5) | 53.9 / 44.4 (+9.5) | 57.3 / 44.4 (+12.9) |
| GraphToken sum s42 | 57.2 / 43.4 (+13.8) | 58.6 / 46.1 (+12.5) | 55.8 / 46.1 (+9.7) | 57.7 / 44.2 (+13.5) |
| GraphToken warm start | 63.0 / 42.4 (+20.5) | 68.4 / 49.2 (+19.2) | 63.8 / 47.1 (+16.8) | 62.5 / 45.6 (+16.9) |
| Soft prompt (history / no history) | 32.3 / 29.0 | 33.0 / 34.8 | 34.4 / 31.8 | 34.6 / 31.5 |
| Stale answer (no model) | 54.9 | 57.2 | 59.9 | 60.1 |

Findings:
- **Accuracy and the re-encoding gain are flat across session length** for every model, on both
  splits (validation GraphToken sum: +13.8 / +11.6 / +11.4 / +12.4). Per-turn-index curves are
  also flat (GraphToken sum gain +11.3 to +13.5 at turns 0–6). The mid-submission's "stale tokens
  degrade as edits accumulate" does not appear for any model.
- **Why it's flat: the benchmark doesn't get harder with length.** The share of answer-changing
  turns is 45 / 43 / 40 / 40% for 1 / 2 / 4 / 8-turn sessions, so long sessions contain no more
  changed questions than short ones (the stale answer actually scores *higher* in long sessions).
- **Answer-changing turns carry the whole effect** (test, updated / stale + history):

  | Model | Changed turns (41%) | Unchanged turns (59%) |
  |---|---|---|
  | TEA September | 34.6 / 29.7 (+5.0) | 35.5 / 34.2 (+1.3) |
  | TEA pretrained sum | 34.6 / 25.8 (+8.8) | 46.7 / 44.6 (+2.0) |
  | TEA random sum | 48.9 / 21.9 (+27.0) | 61.3 / 59.7 (+1.6) |
  | GraphToken sum (3 seeds) | 49.0–53.4 / 21.8–26.4 (+26.9 to +27.2) | 59.5–62.2 / 57.3–59.6 (+2.2 to +2.6) |
  | GraphToken warm start | 60.1 / 22.1 (+38.0) | 66.1 / 62.8 (+3.3) |
  | Soft prompt (history / no history) | 33.0 / 28.5 | 35.0 / 34.1 |

  On changed turns stale tokens score 22–26%, *below* question-only (25.5): stale tokens actively
  give the old answer. Even re-encoded, models are 6–13 points worse on changed than on unchanged
  turns, so the updated state is read imperfectly.
- **The only accumulation effect is on "unchanged" turns.** "Unchanged" means the answer matches
  the previous turn's graph, but the stale conditions encode the *initial* graph. On those turns the
  re-encoding gain grows with session length: GraphToken sum −0.3 / +0.3 / +3.6 (validation, 2 / 4 /
  8 turns), warm start 0.0 / +0.6 / +2.3 / +5.4 (validation, 1–8 turns); same direction on test.
  This is the clean signal that the initial graph drifts out of date as edits pile up.
- **Per task, the changed-turn gain comes from a few tasks** (GraphToken sum, test, updated / stale +
  history on changed turns): node degree 100.0 / 8.6, cycle 91.6 / 17.8, edge existence 71.4 / 28.6,
  constrained reachability 76.3 / 42.2. Little or no gain on reachability (44.5 / 43.0; the random
  GNN can't read station status), within-hops count (19.8 / 15.2) and most-common attribute
  (28.6 / 33.3, negative). The warm start adds reachability (94.0 / 16.5) and shortest path
  (48.3 / 18.9).
- **Density:** denser graphs score 2–4 points higher for the sum-aggregation models; the gain does
  not change with density.
- **Soft prompt:** with history it rises slightly with length (32.3 → 34.6); without history it moves
  between 29.0 and 34.8 with no trend, so history text is worth ~3 points to it in 8-turn sessions.

The September runs show the same picture: no length trend that replicates on both splits, and the
changed/unchanged split as the clear signal (TEA: stale tokens lose 4.7 on changed turns, re-encoded
lose 0.8).

### 3.6 Write path is not the bottleneck on this benchmark

| Model | Model-written edits | Gold edits | Edit exec-equivalent | Graph state exact |
|---|---|---|---|---|
| TEA September | 34.9 | 35.0 | 95.2 | 88.6 |
| Sept "GraphToken" | 37.6 | 37.9 | 95.2 | 88.6 |
| GraphToken collapsed | 34.0 | 34.0 | 95.2 | 88.6 |
| GraphToken sum | 56.8 | 56.9 | 95.2 | 88.6 |
| GraphToken warm start | 62.9 | 63.7 | 95.2 | 88.6 |

The cost of model-written edits grows slightly with model quality (0.0 → 0.8) because a better
reader is hurt more by a wrong graph, but stays under one point, and it does not grow with session
length (warm start: −2.0 / −0.3 / −0.9 / −0.8 for 1 / 2 / 4 / 8 turns), so edit errors don't compound. Revisions are templated (edge
edits are close to the command itself), so this does not measure free-form understanding.

### 3.7 Degree-balanced v3 (server only, never reported)
On 10-16 the team regenerated v3 balanced on station degree (`datasets/metro_v3_bal`) and retrained
GraphToken sum and the warm start on it. **Caveat:** this data predates the cycle-solver fix, and the
repo's own audit found 33–42% of its cycle labels wrong, so cycle numbers below are unreliable.

| Test | Static | Updated | Stale + history | Gain (all) | Gain (changed turns) |
|---|---|---|---|---|---|
| GraphToken sum, trained on balanced data | 51.2 | 54.9 | 41.7 | +13.2 | +28.9 |
| GraphToken warm start, trained on balanced data | 62.0 | 59.4 | 42.6 | +16.8 | +35.6 |

Balancing on degree costs 2–4 points but leaves the re-encoding gain intact, and length trends stay
flat. It also weakens the shortcut: on balanced data the degree/status classifier scores 56.5% (vs
64.0%), and models trained on the original data score 31–34% on the items it gets wrong (vs 22–26%
before). So degree balancing is a usable lever for v4, but node degree stays at 98–100% for every
sum model and reachability for the warm start at 91–94%: both are still single-node reads.

### 3.8 Other results that were only on the server
- **Zero-GNN control** (graph tokens from a zeroed GNN, old checkpoints): 24.7–28.3 static, confirming
  the log's "24.7–25.4" claim for TEA and the September GraphToken (the collapsed run gives 28.3).
- **Encoder autocast** (`amp_*`): static TEA 36.6 vs 36.3, September GraphToken 40.3 vs 40.3,
  collapsed GraphToken 33.4 vs 33.0 — under 1 point, consistent with the log.
- **Never evaluated:** `pilot_frozen` and `pilot_projlr` have empty projector folders;
  `tea_fixed_s44` stopped at checkpoint 500; `tea_fixed_s43` doesn't exist. **No GraphModi run on
  the server is newer than 10-16 20:54, and nothing exists for metro_v4.**

---

## 4. What else changed

- **Benchmark metro_v4** (`src/graph_modi/data/v4.py`, 2026-10-25) replaces v3 for new runs: ring
  core with pendant chains, cycle queries only at degree 2, reachability only between open
  stations, quota-balanced sessions, three paraphrase families with `notice` held out of the
  few-shot prompt, NOOP ~17%. Its gates pass (43/0) and an independent audit passes (75/0); both
  regenerate bit-identically. **No v4 model results exist yet.**
- **Code fixes** (mostly in `5549ac5`): GraphToken trains its GNN; contrastive pretraining updates
  the GNN; sum / max / GIN-style aggregation; learning-rate schedules with warmup; validation loss
  and `checkpoint-best`; resume fixes; leading-token answer parsing; real `majority_answer` and
  `stale_answer` baselines; `soft_prompt_no_history`; session-clustered bootstrap CIs; added edges
  get deterministic attributes; cycle solver fixed.
- **Repo**: superseded audit configs and v3x variants moved to `archive/`; README rewritten for v4.

---

## 5. What is still to be fixed

### Before any v4 GPU run (blocking)
1. **v4 configs repeat the failed recipes.** `configs/v4_graphtoken.yaml` uses mean aggregation,
   dropout 0.5, one lr for GNN and projector, no schedule: the setup that collapsed to 33%. Use the
   working recipe (sum aggregation, dropout 0, GNN lr 1e-4, cosine with 200 warmup steps,
   `eval_every_steps: 500`). `configs/v4_tea.yaml` also uses mean aggregation and projector lr 1e-3
   (October found pretrained TEA needs 1e-4).
2. **Seeds and checkpoint selection:** all v4 configs are seed 42 only and never produce
   `checkpoint-best`; the run checklist requires 3 seeds and best + last.
3. **Missing controls in v4 configs:** `majority_answer` and `soft_prompt_no_history`.
4. **Data plumbing:** the CLI writes sessions as `validation.jsonl`/`test.jsonl` but the manifest
   and README use `sessions_*.jsonl`, so following the README makes training regenerate data
   silently; `v4_data.yaml` needs torch just to generate; `generate` run from the repo root
   appends to the tracked `log.md`.

### Benchmark (v4 introduced new shortcuts the gates don't measure)
5. Cycle: a "has a degree-1 neighbour" rule scores 91% (the `hop_depth` tag encodes the answer).
6. Reachability: endpoint degree alone 72%, a small local lookup 84%.
7. Only 14.6% of v4 session turns change their answer, so the stale answer scores 85%; for
   edge existence, constrained reachability and most-common-attribute under 4% of turns change.
8. Counterfactual "before" items reintroduce v3 shortcuts (closed endpoints, degree-1 cycle "no").
9. The write path is still 100% regex-parsable; revisions are templated.

### Code
10. Training prompts are the bare question; inference wraps `Question: …\nAnswer:`; no training
    example contains revision history.
11. `max_sequence_length` 512 with left truncation; the edit-parsing prompt is close to the limit
    and loses its instructions first.
12. `frozen_graph_history` and `cached_no_reencode` still receive identical inputs.
13. Added-edge attributes depend on endpoint order, so a correct reversed ADD fails the exact-state
    check; this executor change also alters regenerated v3 data.
14. Resume does not restore RNG or best-validation state; `final-study` seed loop is a no-op;
    soft-prompt training has no schedule or validation; evaluation defaults to `checkpoint-final`;
    training and evaluation use different encoder precision by default; several options are not
    stored in checkpoint metadata.
15. All 9 torch tests (GNN training, pretraining, resume, warm start) are skipped without torch, so
    the two key fixes are verified only by code reading; lint rules and type checks were relaxed
    to pass.

### Experiments and evidence
16. Warm start and every TEA variant are single-seed; the "joint training adds nothing over a random
    GNN" comparison differs in lr, schedule and warmup, not just training mode.
17. Re-score the October runs with corrected cycle labels (effect ≤0.7, but it changes the
    "balanced labels" claim). Copy their per-turn rows (`/data/lmw/graph-modi/outputs/audit/`,
    444 MB) or at least the §3.5 breakdowns into the repo so the trends are reproducible.
18. Still unbacked by committed files: old pretraining loss 4.477, MLP pilots 52.6/53.4 (zero-GNN
    and autocast are now confirmed from the server, §3.8). `pilot_frozen`, `pilot_projlr` and
    `tea_fixed_s43/s44` were never completed; the degree-balanced runs (§3.7) were never logged
    and must be re-run after the cycle fix if used.
23. The server checkout `/data/lmw/graph-modi` is on an old `exact2x` commit with uncommitted local
    edits (applied from `src.patch` etc.), not on `graphtoken-e2e`; sync it before any v4 run.
19. Still open from the mid-submission plan: serialized-graph input, contradiction probe, free-form
    revisions, GLM architecture ablations beyond aggregation.

### Documents
20. `status_report_v3.md` still says labels are balanced 50/50 and never mentions the cycle bug;
    its §0a says all runs used cosine warmup (false for the TEA sum runs).
21. The presentation (`documents/presentation/`) is pre-audit (says TEA uses trained alignment and
    calls TEA run 2 "GraphToken").
22. README's model table says v4 uses mean aggregation and lr 1e-3; `log.md` rows are out of date
    order, one link is broken, and two automatic smoke-run lines are committed in it.
