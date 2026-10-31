# GraphModi v4.1: report guide

This guide holds every final result of the v4.1 experiments, with an explanation of each, so that
the LaTeX report (`documents/mid-report/main.tex`) can be written from it alone. It covers the
setup, all result tables, what each result shows, the caveats, and a checklist of numbers in the
current `main.tex` draft that must change.

**Where the numbers come from.** Every table below is generated from the analysis files in
`documents/experiments/results/v4_1-analysis/` (`report_tables.json`, `analysis.json`,
`projector_lr_sweep.json`, `gnn_probe.json`, `shortcut_ceilings.json`). Those files are produced by
the scripts in `scripts/` from the per-turn outputs in
`documents/experiments/results/v4_1-20261030/runs.tar.gz` (see [Reproducing](#9-reproducing)).
Do not type numbers into the report by hand from anywhere else.

**Conventions.** Accuracy is exact match in %, answers read from the first non-empty line of the
generation. "GT" is GraphToken-style, "TEA" is TEA-style, "Warm" is the warm start, "Random" is the
frozen random GNN. GT and TEA are the mean ± s.d. over seeds 42, 43 and 44; Warm, Random, the soft
prompt and the text baselines are one run (seed 42). Multi-turn results are on the 4,455 test turns.
"Oracle" is `oracle_updated_graph`, "predicted" is `predicted_updated_graph`, "stale + history" is
`frozen_graph_history`.

---

## 1. Headline findings

1. **Re-encoding the updated graph works.** GT answers 54.3 ± 0.2% of test turns from re-encoded
   tokens, against 41.2% for stale tokens plus the revision text, 32.5% for another session's graph,
   34.7% for a trained soft prompt that reads the revisions, and 29.9% for zero-shot Llama with the
   initial graph and all revisions in context. Paired on seed 42: +13.6 points over stale + history
   (CI 11.7–15.5) and +22.0 over the shuffled graph (19.5–24.5).
2. **The gain sits on the turns where the edit changes the answer.** On answer-changing turns GT
   scores 51.8% re-encoded against 24.2% with stale tokens plus history (a 27.6-point gap). On
   unchanged turns the gap is 4.8 points (55.7 vs 50.9). On answer-changing turns whose question names
   no edited station, where no text lookup works (2.5%), re-encoding still gives 38.3% against 26.8%.
3. **The write path is not the bottleneck.** Frozen Llama writes an execution-equivalent edit
   program on 96.6% of turns. Using its own edits instead of gold edits costs at most 0.1 points
   overall for every reader, and the paired CI includes zero in every run.
4. **A frozen random GNN reads as well as a jointly trained one.** The frozen random GNN scores
   55.2% (static 50.8%) against 54.3% (static 50.3%) for GT. Paired with GT seed 42 the difference is
   −1.1 points (CI −2.3 to 0.0). Linear probes show why: a random sum-aggregation GraphSAGE already
   makes degree (100%), station status (96.6%) and two-hop counts (92.3%) linearly decodable, so a
   linear projector does not need a trained GNN.
5. **TEA-GLM pretraining discards what updates change.** TEA scores 42.1 ± 0.4%, 13.2 points below
   the frozen random GNN with the same frozen-GNN recipe (seed 42, CI 11.0–15.4). Its pretrained node
   embeddings probe at 88.1% for status, which is the majority class, so status is gone. TEA still
   beats every graph-blind baseline (+8.5 to +9.3 over the soft prompt without history) and gains
   3.7–4.5 points from updates (oracle vs stale + history, every seed).
6. **TEA-GLM pretraining keeps node-text content.** On attribute questions TEA beats GT: attribute
   lookup 46.5% vs 29.5% (static), most-common attribute within two hops 36.3% vs 28.5% (static) and
   36.0% vs 28.6% (multi-turn). The warm start, which starts from the same pretrained GNN, shows the
   same pattern. TEA loses on structure: node degree 49.6% vs 99.9%, cycle membership 49.2% (chance)
   vs 83.4% (multi-turn).
7. **All readers read locally.** Every reader stays at or below the learned local-feature ceiling on
   the structural tasks. On the 1,052 turns that both local classifiers get wrong, every graph reader
   scores 19–24%, no better than the question alone (24.6%).
8. **Re-encoded accuracy does not degrade over a session.** GT oracle stays between 53.6% and 55.1%
   for session lengths 1, 2, 4 and 8. Stale tokens plus history drop from 47.0% in one-turn sessions
   to 40.3–41.7% in longer ones.
9. **The projector learning rate mattered for the frozen-GNN readers.** At 1e-3 their validation
   loss was far worse than at 1e-4 (TEA 0.874 vs 0.572); all four readers chose 1e-4 on validation
   loss.

---

## 2. Problem and research questions

A graph language model (GLM) answers questions from soft graph tokens: a GNN encodes the graph, a
projector maps the encoding to a few LLM input embeddings, and a frozen LLM reads them. Existing
GLMs assume the graph is fixed. In a dialogue, users revise the world ("Station X is now closed",
"a new track links A and B"), and tokens computed before a revision describe a world that no longer
exists.

GraphModi keeps the state outside the model. Each turn, the revision text is turned into a typed
edit program, the program is applied to an explicit graph, the graph is re-encoded, and the LLM
answers from the new tokens and the question only. It never sees the revision text in the QA call.
This separates **writing** (did the edit land?) from **reading** (can the model answer from the new
tokens?).

- **RQ1.** Can a frozen LLM emit executable edit programs from a few-shot prompt, and what do its
  errors cost downstream?
- **RQ2.** Does answering from re-encoded tokens beat stale tokens and text-history controls as turns
  accumulate, especially on answer-changing turns?
- **RQ3.** Do graph tokens carry information beyond graph-blind baselines, and beyond what local
  features of the queried stations give?

Formal setup: an undirected attributed metro graph G = (V, E, a, ρ). Node attributes are line,
accessibility, architecture, cleanliness, music, size and status ∈ {open, closed}; edge relations
are track or transfer. A session is (G₀, (r_t, q_t) for t = 1..T): a revision followed by a question
each turn. Gold programs give G_t = Apply(G_{t−1}, P_t*). The gold answer is Solve(G_t, q_t) and the
stale answer is Solve(G_{t−1}, q_t). A turn is answer-changing when they differ.

**Edit language.** At most three operations per program, separated by `;` and ended by `END`:
`SET NODE` (station status), `ADD EDGE`, `DEL EDGE`, `NOOP`.

---

## 3. Benchmark (metro_v4_1)

**Graphs.** A ring core with pendant chains, 12–22 stations. Mean degree is 2.8 / 3.8 / 5.0 and
the diameter is capped at 7 / 6 / 5 for sparse / medium / dense. Nodes carry CLEGR-like attributes
and fictional two-word station names.

**Static QA.** Eleven task families with per-task answer quotas, so yes/no tasks are balanced. The
evaluation sets have 2,178 validation and 4,356 test items. The tasks:

| Task | Question | In sessions |
|---|---|---|
| `edge_exists` | Is there a direct edge between A and B? | yes |
| `reachability` | Can B be reached from A through open stations only? | yes |
| `constrained_reachability` | Can B be reached from A avoiding a station class (no disabled access, no rail, or dirty)? | yes |
| `cycle_membership` | Is A on any cycle? Asked only at degree-2 stations | yes |
| `node_degree` | How many stations are directly connected to A? | yes |
| `filtered_neighbor_count` | How many direct neighbours of A have a given attribute value? | yes |
| `shortest_path` | How many stations (with both endpoints) are on the shortest open path from A to B, or `unreachable`? | yes |
| `within_hops_count` | How many other stations within two hops of A have a given attribute value? | yes |
| `most_common_attribute_within_hops` | Most common value of an attribute among stations within two hops of A (unique mode only) | yes |
| `attribute_lookup` | Value of one of A's attributes (open categorical) | static only |
| `attribute_check` | Does A have a given attribute value? | static only |

As in CLEGR, the text of the queried station(s) is added to the question with status removed, so
status must come from the graph. For the attribute tasks the queried attribute is also removed.

**Multi-turn sessions.** The grid crosses 11 node counts × 4 session lengths {1, 2, 4, 8} × 3
densities × 9 tasks, giving 1,188 sessions and 4,455 turns per split. Revisions are written from
templates in three paraphrase families (`direct`, `service`, `notice`). The `notice` family is held
out of the few-shot edit prompt.

**Fixes made in generator v4.1** (Yajat's freeze commit `c4a6275`; full audit in
`documents/experiments/benchmark_v4_1_2026-10-30.md`):

| Problem in v4.0 | Fix in v4.1 |
|---|---|
| Only 14.6% of turns changed the answer (stale answer scored ~85%) | Per-split quota prefers answer-changing queries (`TARGET_CHANGED_SHARE = 0.45`), giving 36.4% on test |
| "A neighbour of a leaf is not on a cycle" solved ~95% of cycle items | Pendant chains of length ≥ 2 end in a triangle with probability 0.6 (`LOOP_CHAIN_PROB`); the rule now scores 79% |
| Cycle items' hop tag leaked the answer | Cycle items always use the `global` tag |
| Counterfactual "before" items could break the test filters (closed endpoint, non-degree-2 cycle station) | `_v4_eligible` is applied to the before graph too (0 violations) |

Every stored label and edit was re-derived by independent solvers (0 mismatches), and the
question-text leak audit is clean (`results/benchmark-audit-v4_1-20261030/`,
`results/benchmark-final-audit-v4_1-20261030/`). No session contains a net-no-op, cancelled or
disconnecting edit.

#### Benchmark statistics (test split)

| Quantity | Value |
|---|---|
| Sessions / turns | 1188 / 4455 |
| Answer-changing turns | 36.4% |
| NOOP turns | 16.7% |
| Turns whose question names an edited station | 50.2% |
| Slice size: `all` | 4455 |
| Slice size: `answer_changing` | 1622 |
| Slice size: `answer_unchanged` | 2833 |
| Slice size: `changed_and_no_mention` | 359 |
| Slice size: `family_direct` | 1485 |
| Slice size: `family_notice` | 1485 |
| Slice size: `family_service` | 1485 |
| Slice size: `local_right` | 2908 |
| Slice size: `local_wrong` | 1052 |
| Slice size: `mention` | 2235 |
| Slice size: `no_mention` | 2220 |
| Slice size: `shortcut_right` | 2450 |
| Slice size: `shortcut_wrong` | 1510 |

Edit-operation mix of test turns (gold programs):

| Operations | Turns |
|---|---|
| noop | 745 |
| ADD EDGE | 608 |
| ADD EDGE+DEL EDGE | 591 |
| ADD EDGE+SET NODE | 577 |
| DEL EDGE | 571 |
| DEL EDGE+SET NODE | 540 |
| SET NODE | 539 |
| ADD EDGE+DEL EDGE+SET NODE | 284 |


**Important confound: paraphrase family is fixed by task.** Each task uses exactly one family:
`direct` for cycle membership, edge existence and shortest path; `service` for node degree,
reachability and within-hops count; `notice` for constrained reachability, filtered neighbour count
and most-common attribute (495 turns each). Any accuracy-by-family table is therefore a task-mix
table. Do not present read accuracy by family as an effect of wording. The write-path result on
`notice` (91.5%) is also partly a task effect.

For completeness, read accuracy by family (this is the task mix, not a wording effect):

#### By revision paraphrase family (%) — `notice` is held out of the few-shot edit prompt

| Reader · condition | direct | service | notice |
|---|---|---|---|
| GT · `oracle_updated_graph` | 58.4 ± 1.2 | 63.2 ± 1.1 | 41.3 ± 0.8 |
| GT · `predicted_updated_graph` | 58.3 ± 1.2 | 63.1 ± 1.1 | 41.1 ± 0.7 |
| GT · `frozen_graph_history` | 46.6 ± 0.7 | 39.6 ± 1.1 | 37.5 ± 0.6 |
| GT · `structure_only` | 45.9 ± 0.6 | 40.2 ± 0.7 | 37.7 ± 0.4 |
| GT · `shuffled_graph` | 38.9 ± 0.1 | 26.5 ± 0.9 | 32.0 ± 0.5 |
| GT · `question_only` | 35.0 ± 0.0 | 20.9 ± 0.0 | 23.6 ± 0.0 |
| TEA · `oracle_updated_graph` | 42.6 ± 0.7 | 42.8 ± 1.2 | 41.1 ± 0.4 |
| TEA · `predicted_updated_graph` | 42.6 ± 0.7 | 42.7 ± 1.2 | 40.9 ± 0.7 |
| TEA · `frozen_graph_history` | 39.8 ± 0.6 | 35.8 ± 0.2 | 39.0 ± 0.8 |
| TEA · `structure_only` | 39.4 ± 0.8 | 35.1 ± 0.5 | 39.2 ± 0.9 |
| TEA · `shuffled_graph` | 37.4 ± 0.2 | 25.8 ± 1.3 | 32.4 ± 0.5 |
| TEA · `question_only` | 35.0 ± 0.0 | 20.9 ± 0.0 | 23.6 ± 0.0 |
| Warm · `oracle_updated_graph` | 41.2 | 51.7 | 40.5 |
| Warm · `predicted_updated_graph` | 41.3 | 51.6 | 40.9 |
| Warm · `frozen_graph_history` | 38.9 | 38.0 | 39.3 |
| Warm · `structure_only` | 37.6 | 38.4 | 38.0 |
| Warm · `shuffled_graph` | 38.3 | 23.4 | 30.7 |
| Warm · `question_only` | 35.0 | 20.9 | 23.6 |
| Random · `oracle_updated_graph` | 60.9 | 62.1 | 42.7 |
| Random · `predicted_updated_graph` | 60.9 | 62.0 | 42.4 |
| Random · `frozen_graph_history` | 46.5 | 39.9 | 38.7 |
| Random · `structure_only` | 46.3 | 39.9 | 38.7 |
| Random · `shuffled_graph` | 37.9 | 26.2 | 31.4 |
| Random · `question_only` | 35.0 | 20.9 | 23.6 |
| shared · `soft_prompt` | 42.8 | 29.2 | 32.2 |
| shared · `soft_prompt_no_history` | 39.5 | 28.7 | 31.7 |
| shared · `serialized_current_graph` | 37.6 | 22.1 | 25.3 |
| shared · `serialized_initial_history` | 42.8 | 22.7 | 24.3 |
| shared · `token_matched_history` | 38.5 | 21.9 | 23.5 |
| shared · `majority_answer` | 41.1 | 29.2 | 25.1 |
| shared · `stale_answer` | 63.2 | 60.9 | 66.7 |


### Local-shortcut ceilings and text lookups

The benchmark is synthetic, so we measure how far local features of the queried stations go.
`scripts/analyze_v4.py` fits one gradient-boosted classifier per task on static train:

- **One-hop features:** degree, status, number of leaf and closed neighbours, smallest neighbour
  degree, hop radius, graph size, closed count.
- **Two-hop features** add stations, leaves and closed stations at two hops, triangles, max and
  summed neighbour degree, and common neighbours. There are no distances.

A 3-layer GNN can see all of these features. Accuracy near these ceilings therefore shows local
reading, not multi-hop reasoning.

The **local-wrong slice** is the 1,052 test turns that both classifiers get wrong. The text lookups
are fit on validation sessions with no graph:

- `text_only` uses only what the revision text says about the queried stations.
- `stale_plus_text` uses the pre-edit answer plus that text.

They score high because 50% of turns ask about a station that was just edited. The
**changed-and-unmentioned slice** (359 answer-changing turns whose question names no edited station)
removes that route.

#### Local-shortcut ceilings and text lookups

| Reference | Accuracy (%) |
|---|---|
| One-hop classifier, static test (non-attribute tasks) | 63.4 |
| One-hop classifier, updated session graphs | 61.9 |
| Two-hop classifier, static test (non-attribute tasks) | 70.1 |
| Two-hop classifier, updated session graphs | 67.0 |
| Text lookup `stale_plus_text` (all / changed / changed & unmentioned) | 78.7 / 54.3 / 2.5 |
| Text lookup `text_only` (all / changed / changed & unmentioned) | 36.6 / 41.4 / 28.4 |

Learned local ceiling per task (better of one- and two-hop classifiers):

| Task | Static test | Updated session graphs |
|---|---|---|
| `constrained_reachability` | 65.4 | 64.4 |
| `cycle_membership` | 95.7 | 88.1 |
| `edge_exists` | 95.2 | 91.5 |
| `filtered_neighbor_count` | 29.0 | 33.7 |
| `node_degree` | 100.0 | 100.0 |
| `reachability` | 92.4 | 91.3 |
| `shortest_path` | 62.1 | 55.6 |
| `within_hops_count` | 20.7 | 16.8 |


---

## 4. Method

**Write path.** A few-shot prompt makes the frozen LLM emit an edit program with greedy decoding.
The prompt shows the output format, a NOOP example, and multi-operation outputs separated by `;` and
ended by `END`. Station labels, including multi-word names, are mapped to node IDs, and operations
are applied as given. Invalid operations are skipped, so an edit error persists into later turns.
Edit accuracy is execution equivalence: the predicted program yields the same graph as the gold one.

**Read path (shared by every graph reader).**

- **Node features:** frozen `bert-base-uncased`, 768-d, mean-pooled over `label: attr=val, …`,
  status included.
- **GNN:** 3-layer GraphSAGE with sum aggregation, LayerNorm and no dropout. Widths are
  768 → 1024 → 1024 → 4096; the output width equals the Llama hidden size, which TEA-GLM's
  feature-wise loss requires.
- **Task-conditional readout:** the source and target node embeddings when the question names
  them, otherwise the mean pool. This is concatenated with a hop-radius scalar and a 16-way task
  one-hot, giving 12,305 dimensions.
- **Projector:** one linear layer to K = 10 tokens of width 4096 (504M parameters).
- **LLM:** frozen Llama-3.1-8B-Instruct in bf16.
- **Prompt:** `Question: {q}\nAnswer:` followed by the graph tokens.

**Readers.**

| Reader | GNN | What trains | Seeds |
|---|---|---|---|
| GraphToken-style | random init | GNN (lr 1e-4) and projector, jointly | 42, 43, 44 |
| TEA-style | TEA-GLM-pretrained, then frozen | projector only | 42, 43, 44 |
| Warm start | TEA-GLM-pretrained | GNN (lr 1e-4) and projector, jointly | 42 |
| Frozen random GNN | seeded random init, frozen | projector only | 42 |
| Soft prompt | none | 10 × 4096 prompt vectors | 42 |
| Zero-shot text baselines | none | nothing | – |

GraphToken trains everything at once, end to end through the frozen LLM's loss. TEA-style trains
in two stages: GNN pretraining, then the projector alone.

The **warm start** and the **random GNN** are ablations that separate the two choices. The warm
start differs from TEA only in whether the GNN trains, and from GT only in the GNN's starting
weights. The random GNN differs from TEA only in the GNN's weights.

**TEA-GLM GNN pretraining** follows the authors' released code (`GNNPreTrain/train_valid_paper.py`):

- **Views:** two views of each batch of 32 graphs. Edges are dropped with probability 0.3 / 0.4 and
  feature dimensions masked with probability 0 / 0.1.
- **Instance-wise loss:** GRACE, through a two-layer projection head.
- **Feature-wise loss:** contrastive, on the node embeddings projected onto the top 1,000 principal
  components of Llama's token-embedding table.
- **Settings:** τ = 0.4, Adam lr 2e-3, dropout 0.5, up to 70 epochs with patience 10.
- **Output:** one pretrained GNN, `outputs/v4_pretrain_sum/gnn/gnn.pt`, shared by all TEA seeds and
  the warm start.

#### TEA-GLM GNN pretraining

| Quantity | Value |
|---|---|
| Epochs run / best epoch | 70 / 68 |
| Train loss epoch 1 → best | 12.80 → 10.11 |
|   instance (GRACE) term, epoch 1 → last | 6.46 → 4.76 |
|   feature-wise term, epoch 1 → last | 6.34 → 5.35 |
| Held-out loss (instance + feature) | 9.90 (4.56 + 5.34) |


**Training recipe (every reader).**

- **Data:** one epoch on 24,000 static QA pairs: 18,000 static items plus 6,000 before/after pairs
  around a random edit.
- **Optimiser:** AdamW, weight decay 1e-4, effective batch 24 (8 × 3 accumulation), bf16.
- **Schedule:** cosine with 200 warmup steps and a minimum lr ratio of 0.05.
- **Checkpoint:** validation loss every 100 steps on 1,000 validation items; the lowest-loss
  checkpoint is evaluated.
- **No session training:** no model sees multi-turn sessions. All of them must generalize from
  static items to graph states created at evaluation time.
- **Soft prompt:** lr 1e-3, no schedule, final checkpoint.

**Projector learning-rate selection.** For each graph reader, seed 42 was trained at 1e-3, 3e-4 and
1e-4, and the lr with the lowest best validation loss was kept. Test data was never used. All four
chose 1e-4 (`scripts/sweep_projector_lr.py`, `projector_lr_sweep.json`). At 1e-3, the readers whose
projector reads a frozen or pretrained GNN spiked at the peak lr and had not converged after one
epoch.

#### Projector learning-rate selection (seed 42, best validation loss; lower is better)

| Reader | 1e-3 | 3e-4 | 1e-4 | Chosen |
|---|---|---|---|---|
| GraphToken-style | 0.4862 | 0.4809 | 0.4764 | 0.0001 |
| TEA-style | 0.8738 | 0.6049 | 0.572 | 0.0001 |
| Warm start | 0.7612 | 0.5344 | 0.5234 | 0.0001 |
| Frozen random GNN | 0.7812 | 0.5046 | 0.4852 | 0.0001 |


**Evaluation conditions.** Each test session is evaluated under paired conditions with the same
session IDs:

| Condition | Graph tokens | Text in the prompt | Purpose |
|---|---|---|---|
| `oracle_updated_graph` | gold G_t, re-encoded every turn | question | reading from correct current tokens |
| `predicted_updated_graph` | graph built from the LLM's own edits, re-encoded | question | full GraphModi pipeline; gap to oracle = cost of edit errors |
| `structure_only` | initial G₀, never updated | question | what stale tokens give |
| `frozen_graph_history` | initial G₀ | revision history + question | can text make up for stale tokens? |
| `shuffled_graph` | another session's G₀ | question | do tokens encode *this* graph? |
| `question_only` | none | question | floor |
| `soft_prompt` | trained soft prompt, no GNN | revision history + question | graph-blind trained baseline |
| `soft_prompt_no_history` | trained soft prompt | question | like-for-like with graph conditions |
| `serialized_current_graph` | none (zero-shot Llama) | gold G_t as text + question | can Llama read the graph as text? |
| `serialized_initial_history` | none (zero-shot) | G₀ as text + revision history + question | pure in-context updating |
| `token_matched_history` | none (zero-shot) | last 64 words of history + question | text budget control |
| `tool_solver` | symbolic solver on gold G_t | – | label check (100% by construction) |
| stale answer | no model | – | answer before the edit; reference for ignoring edits |
| per-task majority | no model | – | each task's most common gold answer |

**Statistics.**

- **CIs:** 95% session-clustered bootstrap intervals.
- **Differences:** paired by turn and clustered by session.
- **Seed spread:** "± s.d." in group rows is the spread over seeds, not a CI.
- **Rescoring:** every row is rescored from the stored raw generation with the current scorer.

---

## 5. Results

### 5.1 Static QA (single graph, no revisions)

#### Static QA, overall exact match (%)

| Reader | Validation | Test |
|---|---|---|
| GraphToken-style (3 seeds) | 51.1 ± 0.6 | 50.3 ± 0.1 |
| TEA-style (3 seeds) | 43.2 ± 1.0 | 42.7 ± 0.1 |
| Warm start (s42) | 47.1 | 46.8 |
| Frozen random GNN (s42) | 51.5 | 50.8 |
| Soft prompt (s42) | 33.7 | 32.6 |
| Per-task majority | – | 29.6 |


**What it shows.**

- **Graph tokens help on a single graph.** Every graph reader beats the soft prompt (32.6%) and the
  per-task majority (29.6%).
- **GT and the frozen random GNN are level** (50.3% and 50.8%).
- **The pretrained GNN costs accuracy.** The warm start (46.8%) and TEA (42.7%) fall behind. The
  validation ranking is the same, so this is not test noise.

#### Static QA by task family, test (%)

| Task | GT | TEA | Warm | Random | Soft | Majority | Local ceiling |
|---|---|---|---|---|---|---|---|
| `attribute_check` | 49.4 ± 1.4 | 52.9 ± 1.0 | 53.8 | 48.7 | 50.5 | 50.5 | – |
| `attribute_lookup` | 29.5 ± 1.8 | 46.5 ± 1.7 | 46.2 | 28.0 | 25.8 | 5.8 | – |
| `constrained_reachability` | 56.6 ± 1.2 | 51.7 ± 1.2 | 55.8 | 57.1 | 50.0 | 50.0 | 65.4 |
| `cycle_membership` | 79.0 ± 0.4 | 52.0 ± 0.5 | 48.0 | 81.8 | 51.8 | 50.5 | 95.7 |
| `edge_exists` | 61.2 ± 1.5 | 56.4 ± 0.4 | 59.6 | 65.9 | 45.7 | 55.3 | 95.2 |
| `filtered_neighbor_count` | 32.4 ± 3.3 | 27.6 ± 1.9 | 30.8 | 31.3 | 18.2 | 15.7 | 29.0 |
| `most_common_attribute_within_hops` | 28.5 ± 0.5 | 36.3 ± 0.9 | 37.9 | 30.1 | 26.5 | 6.3 | – |
| `node_degree` | 99.1 ± 1.4 | 44.2 ± 1.3 | 75.0 | 94.9 | 11.6 | 11.4 | 100.0 |
| `reachability` | 65.7 ± 0.8 | 61.2 ± 0.8 | 62.6 | 67.4 | 54.0 | 54.0 | 92.4 |
| `shortest_path` | 33.6 ± 0.7 | 23.0 ± 2.0 | 26.5 | 34.3 | 14.1 | 14.4 | 62.1 |
| `within_hops_count` | 18.0 ± 0.3 | 17.8 ± 1.5 | 18.2 | 19.2 | 10.9 | 12.1 | 20.7 |


**What it shows.**

- **GT and random win on structure.** Node degree is 99.1 / 94.9%, cycle membership 79.0 / 81.8%,
  edge existence 61.2 / 65.9%, shortest path 33.6 / 34.3%. These are the tasks a local classifier
  solves, and both readers stay below those ceilings on every structural task.
- **TEA and the warm start win on attributes.** Attribute lookup is 46.5 / 46.2% against
  29.5 / 28.0% for GT / random, and most-common attribute is 36.3 / 37.9% against 28.5 / 30.1%.
  GT's attribute accuracy is close to the soft prompt's (25.8%, 26.5%).
- **TEA collapses on structure.** Node degree is 44.2% and cycle membership 52.0% (chance is
  50.5%).
- **Counting tasks are hard for everyone.** Filtered count is 28–32% and within-hops count 18–19%,
  but those learned ceilings are also low (29.0, 20.7).

### 5.2 Multi-turn: all conditions

#### Multi-turn, all 4,455 test turns (%); seed groups are mean ± s.d. over seeds

| Condition | GT | TEA | Warm | Random |
|---|---|---|---|---|
| `oracle_updated_graph` | 54.3 ± 0.2 | 42.1 ± 0.4 | 44.5 | 55.2 |
| `predicted_updated_graph` | 54.2 ± 0.3 | 42.1 ± 0.3 | 44.6 | 55.1 |
| `frozen_graph_history` | 41.2 ± 0.6 | 38.2 ± 0.1 | 38.8 | 41.7 |
| `structure_only` | 41.2 ± 0.5 | 37.9 ± 0.1 | 38.0 | 41.7 |
| `shuffled_graph` | 32.5 ± 0.4 | 31.9 ± 0.5 | 30.8 | 31.8 |
| `question_only` | 26.5 ± 0.0 | 26.5 ± 0.0 | 26.5 | 26.5 |

Reader-independent conditions (one run each):

| Condition | Accuracy |
|---|---|
| `soft_prompt` | 34.7 |
| `soft_prompt_no_history` | 33.3 |
| `serialized_current_graph` | 28.4 |
| `serialized_initial_history` | 29.9 |
| `token_matched_history` | 27.9 |
| `majority_answer` | 31.8 |
| `stale_answer` | 63.6 |
| `tool_solver` | 100.0 |


**What it shows (RQ2, RQ3).**

- **Re-encoding beats every alternative for every reader:** stale tokens, stale tokens plus
  history, the shuffled graph and question only. This holds per seed with clustered CIs (table
  below).
- **Text does not repair stale tokens.** `frozen_graph_history` equals `structure_only` for every
  reader (GT 41.2 vs 41.2): adding the revision text to stale tokens does not help.
- **Graph-blind channels that see the full history do not catch up.** The soft prompt with history
  scores 34.7%. Zero-shot Llama reading the current graph as text scores 28.4%, and with the initial
  graph plus every revision 29.9%, both below the per-task majority (31.8%).
- **The stale answer (63.6%) is a reference, not a competitor.** It is high because 63.6% of turns
  keep their pre-edit answer by construction, and it is 0% on answer-changing turns.
- **The seed spread is small:** 0.2 for GT and 0.4 for TEA on oracle.

#### Per-run accuracy with 95% session-clustered bootstrap CI (`oracle_updated_graph`, `predicted_updated_graph`)

| Run | oracle | predicted | stale + history | shuffled |
|---|---|---|---|---|
| v4_graphtoken_s42 | 54.1 [51.8, 56.3] | 54.0 [51.7, 56.3] | 40.5 [38.7, 42.4] | 32.1 [30.3, 33.9] |
| v4_graphtoken_s43 | 54.5 [52.2, 56.8] | 54.5 [52.2, 56.8] | 41.7 [39.9, 43.5] | 32.8 [30.9, 34.6] |
| v4_graphtoken_s44 | 54.3 [51.9, 56.4] | 54.0 [51.7, 56.2] | 41.4 [39.6, 43.2] | 32.5 [30.7, 34.4] |
| v4_tea_s42 | 42.0 [40.2, 43.9] | 41.8 [40.1, 43.7] | 38.3 [36.6, 40.1] | 31.4 [29.6, 33.3] |
| v4_tea_s43 | 41.8 [40.0, 43.7] | 41.9 [40.1, 43.8] | 38.1 [36.3, 39.9] | 32.4 [30.5, 34.3] |
| v4_tea_s44 | 42.6 [40.8, 44.6] | 42.5 [40.6, 44.4] | 38.1 [36.4, 39.9] | 31.7 [30.0, 33.4] |
| v4_graphtoken_warm_s42 | 44.5 [42.6, 46.5] | 44.6 [42.7, 46.6] | 38.8 [37.1, 40.5] | 30.8 [29.0, 32.6] |
| v4_tea_random_sum | 55.2 [52.9, 57.5] | 55.1 [52.7, 57.4] | 41.7 [39.8, 43.5] | 31.8 [30.0, 33.7] |

#### Paired differences within a run (oracle minus X, percentage points, turn-paired, session-clustered CI)

| Run | Difference | Δ | 95% CI | p |
|---|---|---|---|---|
| v4_graphtoken_s42 | oracle − frozen_graph_history | +13.6 | [11.7, 15.5] | 0 |
| v4_graphtoken_s42 | oracle − predicted_updated_graph | +0.1 | [-0.2, 0.4] | 0.539 |
| v4_graphtoken_s42 | oracle − shuffled_graph | +22.0 | [19.5, 24.5] | 0 |
| v4_graphtoken_s42 | oracle − structure_only | +13.4 | [11.6, 15.4] | 0 |
| v4_graphtoken_s43 | oracle − frozen_graph_history | +12.8 | [10.9, 14.7] | 0 |
| v4_graphtoken_s43 | oracle − predicted_updated_graph | -0.0 | [-0.3, 0.3] | 0.949 |
| v4_graphtoken_s43 | oracle − shuffled_graph | +21.7 | [19.3, 24.2] | 0 |
| v4_graphtoken_s43 | oracle − structure_only | +12.8 | [10.9, 14.6] | 0 |
| v4_graphtoken_s44 | oracle − frozen_graph_history | +12.8 | [10.9, 14.7] | 0 |
| v4_graphtoken_s44 | oracle − predicted_updated_graph | +0.2 | [-0.1, 0.6] | 0.162 |
| v4_graphtoken_s44 | oracle − shuffled_graph | +21.7 | [19.1, 24.1] | 0 |
| v4_graphtoken_s44 | oracle − structure_only | +13.0 | [11.0, 14.8] | 0 |
| v4_tea_s42 | oracle − frozen_graph_history | +3.7 | [2.4, 4.9] | 0 |
| v4_tea_s42 | oracle − predicted_updated_graph | +0.2 | [-0.1, 0.4] | 0.263 |
| v4_tea_s42 | oracle − shuffled_graph | +10.6 | [8.4, 12.7] | 0 |
| v4_tea_s42 | oracle − structure_only | +4.0 | [2.9, 5.1] | 0 |
| v4_tea_s43 | oracle − frozen_graph_history | +3.7 | [2.5, 4.8] | 0 |
| v4_tea_s43 | oracle − predicted_updated_graph | -0.1 | [-0.3, 0.1] | 0.367 |
| v4_tea_s43 | oracle − shuffled_graph | +9.4 | [7.3, 11.5] | 0 |
| v4_tea_s43 | oracle − structure_only | +3.8 | [2.8, 4.9] | 0 |
| v4_tea_s44 | oracle − frozen_graph_history | +4.5 | [3.3, 5.8] | 0 |
| v4_tea_s44 | oracle − predicted_updated_graph | +0.2 | [-0.0, 0.4] | 0.171 |
| v4_tea_s44 | oracle − shuffled_graph | +10.9 | [8.8, 13.0] | 0 |
| v4_tea_s44 | oracle − structure_only | +4.9 | [3.8, 6.0] | 0 |
| v4_graphtoken_warm_s42 | oracle − frozen_graph_history | +5.7 | [4.3, 7.2] | 0 |
| v4_graphtoken_warm_s42 | oracle − predicted_updated_graph | -0.1 | [-0.4, 0.1] | 0.407 |
| v4_graphtoken_warm_s42 | oracle − shuffled_graph | +13.7 | [11.3, 16.0] | 0 |
| v4_graphtoken_warm_s42 | oracle − structure_only | +6.5 | [5.1, 7.8] | 0 |
| v4_tea_random_sum | oracle − frozen_graph_history | +13.5 | [11.6, 15.5] | 0 |
| v4_tea_random_sum | oracle − predicted_updated_graph | +0.1 | [-0.2, 0.5] | 0.489 |
| v4_tea_random_sum | oracle − shuffled_graph | +23.4 | [21.0, 25.8] | 0 |
| v4_tea_random_sum | oracle − structure_only | +13.6 | [11.7, 15.4] | 0 |


**What it shows.**

- **Within GT, oracle beats** stale + history by 12.8–13.6 points, structure-only by 12.8–13.4 and
  the shuffled graph by 21.7–22.0, on every seed.
- **Within TEA, updates help less but significantly:** +3.7 to +4.5 over stale + history and
  +9.4 to +10.9 over the shuffled graph.
- **Oracle minus predicted is between −0.1 and +0.2 in every run**, and every CI includes 0.

#### Paired differences across runs (percentage points)

| Comparison [slice] | Δ | 95% CI |
|---|---|---|
| v4_graphtoken_s42:oracle - v4_soft_prompt:soft_prompt [all] | +19.4 | [16.8, 22.0] |
| v4_graphtoken_s42:oracle - v4_soft_prompt:soft_prompt [answer_changing] | +19.7 | [16.1, 23.2] |
| v4_graphtoken_s42:oracle - v4_soft_prompt:soft_prompt [local_wrong] | +4.0 | [0.2, 7.7] |
| v4_graphtoken_s42:oracle - v4_soft_prompt:soft_prompt_no_history [all] | +20.8 | [18.2, 23.3] |
| v4_graphtoken_s42:oracle - v4_soft_prompt:soft_prompt_no_history [answer_changing] | +22.6 | [19.4, 26.0] |
| v4_graphtoken_s42:oracle - v4_soft_prompt:soft_prompt_no_history [local_wrong] | +4.5 | [0.9, 7.9] |
| v4_graphtoken_s42:oracle - v4_tea_random_sum:oracle [all] | -1.1 | [-2.3, 0.0] |
| v4_graphtoken_s42:oracle - v4_tea_random_sum:oracle [answer_changing] | -0.9 | [-2.6, 0.7] |
| v4_graphtoken_s42:oracle - v4_tea_random_sum:oracle [local_wrong] | +3.4 | [1.2, 5.7] |
| v4_graphtoken_s42:oracle - v4_tea_s42:oracle [all] | +12.1 | [9.9, 14.3] |
| v4_graphtoken_s42:oracle - v4_tea_s42:oracle [answer_changing] | +14.4 | [11.3, 17.5] |
| v4_graphtoken_s42:oracle - v4_tea_s42:oracle [local_wrong] | -2.6 | [-5.8, 0.5] |
| v4_graphtoken_s42:oracle - v4_tea_s43:oracle [all] | +12.3 | [10.0, 14.4] |
| v4_graphtoken_s42:oracle - v4_tea_s43:oracle [answer_changing] | +14.6 | [11.4, 17.8] |
| v4_graphtoken_s42:oracle - v4_tea_s43:oracle [local_wrong] | +1.0 | [-2.4, 4.5] |
| v4_graphtoken_s42:oracle - v4_tea_s44:oracle [all] | +11.5 | [9.3, 13.6] |
| v4_graphtoken_s42:oracle - v4_tea_s44:oracle [answer_changing] | +13.3 | [10.3, 16.3] |
| v4_graphtoken_s42:oracle - v4_tea_s44:oracle [local_wrong] | -1.4 | [-4.8, 1.8] |
| v4_graphtoken_s43:oracle - v4_soft_prompt:soft_prompt [all] | +19.8 | [17.3, 22.3] |
| v4_graphtoken_s43:oracle - v4_soft_prompt:soft_prompt [answer_changing] | +19.7 | [16.5, 23.4] |
| v4_graphtoken_s43:oracle - v4_soft_prompt:soft_prompt [local_wrong] | +3.3 | [-0.1, 6.7] |
| v4_graphtoken_s43:oracle - v4_soft_prompt:soft_prompt_no_history [all] | +21.2 | [18.7, 23.6] |
| v4_graphtoken_s43:oracle - v4_soft_prompt:soft_prompt_no_history [answer_changing] | +22.7 | [19.5, 26.2] |
| v4_graphtoken_s43:oracle - v4_soft_prompt:soft_prompt_no_history [local_wrong] | +3.8 | [0.4, 7.2] |
| v4_graphtoken_s43:oracle - v4_tea_s42:oracle [all] | +12.5 | [10.2, 14.6] |
| v4_graphtoken_s43:oracle - v4_tea_s42:oracle [answer_changing] | +14.5 | [11.5, 17.5] |
| v4_graphtoken_s43:oracle - v4_tea_s42:oracle [local_wrong] | -3.2 | [-6.6, 0.3] |
| v4_graphtoken_s43:oracle - v4_tea_s43:oracle [all] | +12.7 | [10.5, 14.8] |
| v4_graphtoken_s43:oracle - v4_tea_s43:oracle [answer_changing] | +14.7 | [11.7, 17.6] |
| v4_graphtoken_s43:oracle - v4_tea_s43:oracle [local_wrong] | +0.4 | [-2.9, 3.5] |
| v4_graphtoken_s43:oracle - v4_tea_s44:oracle [all] | +11.9 | [9.6, 14.0] |
| v4_graphtoken_s43:oracle - v4_tea_s44:oracle [answer_changing] | +13.4 | [10.5, 16.3] |
| v4_graphtoken_s43:oracle - v4_tea_s44:oracle [local_wrong] | -2.1 | [-5.5, 1.3] |
| v4_graphtoken_s44:oracle - v4_soft_prompt:soft_prompt [all] | +19.5 | [17.0, 22.0] |
| v4_graphtoken_s44:oracle - v4_soft_prompt:soft_prompt [answer_changing] | +19.4 | [16.0, 22.9] |
| v4_graphtoken_s44:oracle - v4_soft_prompt:soft_prompt [local_wrong] | +3.5 | [0.1, 7.0] |
| v4_graphtoken_s44:oracle - v4_soft_prompt:soft_prompt_no_history [all] | +20.9 | [18.4, 23.4] |
| v4_graphtoken_s44:oracle - v4_soft_prompt:soft_prompt_no_history [answer_changing] | +22.3 | [19.1, 25.7] |
| v4_graphtoken_s44:oracle - v4_soft_prompt:soft_prompt_no_history [local_wrong] | +4.0 | [0.7, 7.4] |
| v4_graphtoken_s44:oracle - v4_tea_s42:oracle [all] | +12.3 | [10.0, 14.3] |
| v4_graphtoken_s44:oracle - v4_tea_s42:oracle [answer_changing] | +14.1 | [11.1, 17.0] |
| v4_graphtoken_s44:oracle - v4_tea_s42:oracle [local_wrong] | -3.0 | [-6.2, 0.0] |
| v4_graphtoken_s44:oracle - v4_tea_s43:oracle [all] | +12.4 | [10.2, 14.5] |
| v4_graphtoken_s44:oracle - v4_tea_s43:oracle [answer_changing] | +14.3 | [11.3, 17.2] |
| v4_graphtoken_s44:oracle - v4_tea_s43:oracle [local_wrong] | +0.6 | [-2.5, 3.5] |
| v4_graphtoken_s44:oracle - v4_tea_s44:oracle [all] | +11.6 | [9.4, 13.7] |
| v4_graphtoken_s44:oracle - v4_tea_s44:oracle [answer_changing] | +13.0 | [10.0, 15.8] |
| v4_graphtoken_s44:oracle - v4_tea_s44:oracle [local_wrong] | -1.9 | [-5.0, 1.2] |
| v4_graphtoken_warm_s42:oracle - v4_graphtoken_s42:oracle [all] | -9.6 | [-11.5, -7.7] |
| v4_graphtoken_warm_s42:oracle - v4_graphtoken_s42:oracle [answer_changing] | -11.8 | [-14.5, -9.1] |
| v4_graphtoken_warm_s42:oracle - v4_graphtoken_s42:oracle [local_wrong] | -1.0 | [-4.1, 2.0] |
| v4_tea_s42:oracle - v4_soft_prompt:soft_prompt_no_history [all] | +8.7 | [6.3, 11.1] |
| v4_tea_s42:oracle - v4_soft_prompt:soft_prompt_no_history [answer_changing] | +8.2 | [5.0, 11.4] |
| v4_tea_s42:oracle - v4_soft_prompt:soft_prompt_no_history [local_wrong] | +7.0 | [3.3, 10.6] |
| v4_tea_s42:oracle - v4_tea_random_sum:oracle [all] | -13.2 | [-15.4, -11.0] |
| v4_tea_s42:oracle - v4_tea_random_sum:oracle [answer_changing] | -15.4 | [-18.2, -12.4] |
| v4_tea_s42:oracle - v4_tea_random_sum:oracle [local_wrong] | +6.0 | [3.2, 9.0] |
| v4_tea_s43:oracle - v4_soft_prompt:soft_prompt_no_history [all] | +8.5 | [6.5, 10.6] |
| v4_tea_s43:oracle - v4_soft_prompt:soft_prompt_no_history [answer_changing] | +8.0 | [5.1, 10.9] |
| v4_tea_s43:oracle - v4_soft_prompt:soft_prompt_no_history [local_wrong] | +3.4 | [0.1, 6.6] |
| v4_tea_s44:oracle - v4_soft_prompt:soft_prompt_no_history [all] | +9.3 | [7.2, 11.5] |
| v4_tea_s44:oracle - v4_soft_prompt:soft_prompt_no_history [answer_changing] | +9.3 | [6.4, 12.3] |
| v4_tea_s44:oracle - v4_soft_prompt:soft_prompt_no_history [local_wrong] | +5.9 | [2.5, 9.4] |


**What it shows.**

- **GT vs soft prompt:** +19.4 to +19.8 with history, +20.8 to +21.2 without (all seeds).
- **GT vs TEA:** +11.5 to +12.7, and +13.0 to +14.7 on answer-changing turns.
- **GT vs random GNN:** −1.1 (CI −2.3 to 0.0). The random GNN is not worse, and is possibly 1 point
  better, overall and on answer-changing turns (−0.9, CI −2.6 to 0.7). GT is 3.4 points better on
  local-wrong turns (1.2 to 5.7), but both are below question-only there.
- **Warm vs GT:** −9.6 (−11.5 to −7.7). Starting from the TEA-GLM GNN hurts even when the GNN
  trains.
- **TEA vs random GNN:** −13.2 (−15.4 to −11.0) overall, but +6.0 (3.2 to 9.0) on local-wrong
  turns. There the random GNN is the lowest reader (19.2%, below question-only at 24.6%), so this
  says more about the random GNN's errors on those turns than about TEA reading beyond local
  features.
- **TEA vs soft prompt without history:** +8.5 to +9.3.

### 5.3 Answer-changing turns

#### Answer-changing vs unchanged turns (%)

| Reader · condition | Changed (1,622) | Unchanged (2,833) | Changed & unmentioned (359) |
|---|---|---|---|
| GT · `oracle_updated_graph` | 51.8 ± 0.2 | 55.7 ± 0.3 | 38.3 ± 1.4 |
| GT · `predicted_updated_graph` | 51.2 ± 0.4 | 55.9 ± 0.5 | 38.2 ± 1.5 |
| GT · `frozen_graph_history` | 24.2 ± 1.0 | 50.9 ± 0.5 | 26.8 ± 1.1 |
| GT · `structure_only` | 23.0 ± 1.1 | 51.7 ± 0.2 | 27.2 ± 1.1 |
| GT · `shuffled_graph` | 29.4 ± 0.6 | 34.2 ± 0.5 | 27.8 ± 1.3 |
| GT · `question_only` | 22.4 ± 0.0 | 28.8 ± 0.0 | 24.2 ± 0.0 |
| TEA · `oracle_updated_graph` | 37.8 ± 0.7 | 44.6 ± 0.3 | 26.3 ± 1.5 |
| TEA · `predicted_updated_graph` | 37.6 ± 0.6 | 44.6 ± 0.2 | 26.5 ± 1.7 |
| TEA · `frozen_graph_history` | 29.1 ± 0.5 | 43.4 ± 0.4 | 27.5 ± 0.9 |
| TEA · `structure_only` | 27.9 ± 0.4 | 43.6 ± 0.1 | 26.6 ± 2.7 |
| TEA · `shuffled_graph` | 28.5 ± 0.7 | 33.8 ± 0.4 | 24.5 ± 0.3 |
| TEA · `question_only` | 22.4 ± 0.0 | 28.8 ± 0.0 | 24.2 ± 0.0 |
| Warm · `oracle_updated_graph` | 40.1 | 47.0 | 25.6 |
| Warm · `predicted_updated_graph` | 40.0 | 47.2 | 26.2 |
| Warm · `frozen_graph_history` | 27.0 | 45.5 | 29.5 |
| Warm · `structure_only` | 26.1 | 44.8 | 27.3 |
| Warm · `shuffled_graph` | 27.6 | 32.7 | 25.9 |
| Warm · `question_only` | 22.4 | 28.8 | 24.2 |
| Random · `oracle_updated_graph` | 52.8 | 56.6 | 40.9 |
| Random · `predicted_updated_graph` | 52.1 | 56.8 | 40.7 |
| Random · `frozen_graph_history` | 24.8 | 51.4 | 24.8 |
| Random · `structure_only` | 22.8 | 52.5 | 26.7 |
| Random · `shuffled_graph` | 28.0 | 34.0 | 27.3 |
| Random · `question_only` | 22.4 | 28.8 | 24.2 |
| shared · `soft_prompt` | 32.2 | 36.1 | 24.8 |
| shared · `soft_prompt_no_history` | 29.3 | 35.6 | 22.0 |
| shared · `serialized_current_graph` | 25.4 | 30.0 | 26.7 |
| shared · `serialized_initial_history` | 27.8 | 31.2 | 25.3 |
| shared · `token_matched_history` | 25.5 | 29.4 | 24.8 |
| shared · `majority_answer` | 27.3 | 34.4 | 25.3 |
| shared · `stale_answer` | 0.0 | 100.0 | 0.0 |


**What it shows.** This is the central RQ2 result.

- **Stale tokens fall below the majority where the answer changed.** For GT they drop to 24.2%
  (23.0% without history), below the per-task majority on these turns (27.3%). Re-encoded tokens
  keep 51.8%: a 27.6-point gap where the edit matters, against 4.8 points where it does not.
- **The gap survives without a text route.** On the 359 changed turns whose question names no
  edited station, re-encoding scores 38.3% against 26.8% for stale + history and 24.8% for the soft
  prompt; the `stale_plus_text` lookup scores 2.5% here.
- **The random GNN behaves like GT:** 52.8 / 56.6 / 40.9.
- **TEA's update gain on changed turns is 8.7 points** (37.8 vs 29.1). Its oracle on changed and
  unmentioned turns (26.3%) is no better than stale + history (27.5%), so TEA's tokens barely track
  the status and edge changes.

### 5.4 Session length and turn index

#### By session length (%)

| Reader · condition | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| GT · `oracle_updated_graph` | 54.1 ± 2.9 | 53.6 ± 0.1 | 55.1 ± 0.7 | 54.1 ± 0.4 |
| GT · `predicted_updated_graph` | 53.5 ± 3.2 | 53.5 ± 0.3 | 55.4 ± 0.7 | 53.8 ± 0.6 |
| GT · `frozen_graph_history` | 47.0 ± 1.7 | 40.7 ± 0.4 | 41.7 ± 1.1 | 40.3 ± 0.5 |
| GT · `structure_only` | 45.0 ± 2.4 | 39.6 ± 0.7 | 42.2 ± 0.3 | 40.7 ± 0.5 |
| GT · `shuffled_graph` | 28.7 ± 1.2 | 31.7 ± 1.9 | 33.1 ± 0.6 | 32.8 ± 0.4 |
| GT · `question_only` | 26.6 ± 0.0 | 29.5 ± 0.0 | 26.0 ± 0.0 | 26.0 ± 0.0 |
| TEA · `oracle_updated_graph` | 45.7 ± 1.0 | 40.8 ± 0.8 | 42.0 ± 1.4 | 42.1 ± 0.3 |
| TEA · `predicted_updated_graph` | 45.6 ± 0.8 | 40.6 ± 0.7 | 42.1 ± 1.4 | 42.0 ± 0.5 |
| TEA · `frozen_graph_history` | 43.7 ± 0.8 | 35.6 ± 1.4 | 38.1 ± 0.5 | 38.2 ± 0.7 |
| TEA · `structure_only` | 42.2 ± 0.7 | 36.2 ± 0.6 | 38.6 ± 0.8 | 37.5 ± 0.8 |
| TEA · `shuffled_graph` | 29.9 ± 2.5 | 30.1 ± 1.2 | 32.6 ± 1.8 | 32.2 ± 0.9 |
| TEA · `question_only` | 26.6 ± 0.0 | 29.5 ± 0.0 | 26.0 ± 0.0 | 26.0 ± 0.0 |
| Warm · `oracle_updated_graph` | 46.8 | 44.3 | 45.8 | 43.6 |
| Warm · `predicted_updated_graph` | 46.5 | 44.1 | 45.9 | 43.9 |
| Warm · `frozen_graph_history` | 42.1 | 38.7 | 38.8 | 38.3 |
| Warm · `structure_only` | 42.4 | 37.9 | 39.0 | 37.0 |
| Warm · `shuffled_graph` | 28.6 | 32.0 | 32.2 | 30.1 |
| Warm · `question_only` | 26.6 | 29.5 | 26.0 | 26.0 |
| Random · `oracle_updated_graph` | 52.2 | 55.9 | 55.5 | 55.3 |
| Random · `predicted_updated_graph` | 52.2 | 55.9 | 55.5 | 55.1 |
| Random · `frozen_graph_history` | 45.1 | 39.6 | 42.3 | 41.5 |
| Random · `structure_only` | 45.8 | 41.4 | 42.3 | 40.9 |
| Random · `shuffled_graph` | 28.3 | 30.3 | 32.2 | 32.4 |
| Random · `question_only` | 26.6 | 29.5 | 26.0 | 26.0 |
| shared · `soft_prompt` | 30.6 | 31.8 | 35.4 | 35.6 |
| shared · `soft_prompt_no_history` | 33.0 | 32.2 | 33.2 | 33.7 |
| shared · `serialized_current_graph` | 27.6 | 30.6 | 27.9 | 28.1 |
| shared · `serialized_initial_history` | 27.3 | 30.1 | 29.2 | 30.6 |
| shared · `token_matched_history` | 27.9 | 30.1 | 27.1 | 27.8 |
| shared · `majority_answer` | 27.9 | 31.1 | 32.2 | 32.2 |
| shared · `stale_answer` | 64.0 | 61.1 | 63.3 | 64.3 |

#### By turn index (%) — turns 5–8 exist only in 8-turn sessions

| Reader · condition | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| GT · `oracle_updated_graph` | 53.9 ± 0.3 | 54.3 ± 0.8 | 55.4 ± 0.9 | 55.7 ± 0.9 | 56.2 ± 0.9 | 52.4 ± 1.0 | 53.3 ± 0.7 | 51.6 ± 1.4 |
| GT · `predicted_updated_graph` | 53.5 ± 0.4 | 54.3 ± 0.8 | 55.3 ± 0.9 | 56.2 ± 0.7 | 55.8 ± 0.8 | 52.2 ± 0.9 | 53.2 ± 0.6 | 51.5 ± 1.2 |
| GT · `frozen_graph_history` | 44.4 ± 0.7 | 39.8 ± 0.3 | 41.4 ± 1.9 | 41.4 ± 0.9 | 39.2 ± 0.5 | 41.3 ± 2.2 | 37.0 ± 0.9 | 38.4 ± 2.9 |
| GT · `structure_only` | 42.8 ± 1.3 | 40.3 ± 0.9 | 42.0 ± 1.0 | 42.9 ± 0.9 | 38.8 ± 0.2 | 42.6 ± 1.2 | 36.7 ± 1.2 | 38.5 ± 2.6 |
| GT · `shuffled_graph` | 32.3 ± 0.9 | 31.3 ± 2.0 | 33.5 ± 0.3 | 32.1 ± 1.4 | 30.0 ± 0.0 | 36.9 ± 0.4 | 32.4 ± 0.8 | 33.2 ± 2.0 |
| GT · `question_only` | 26.0 ± 0.0 | 29.0 ± 0.0 | 26.8 ± 0.0 | 26.8 ± 0.0 | 24.9 ± 0.0 | 25.3 ± 0.0 | 27.6 ± 0.0 | 21.5 ± 0.0 |
| TEA · `oracle_updated_graph` | 43.0 ± 0.1 | 41.1 ± 0.9 | 40.3 ± 1.5 | 43.3 ± 1.0 | 45.9 ± 1.0 | 42.3 ± 1.1 | 43.8 ± 2.0 | 37.5 ± 2.5 |
| TEA · `predicted_updated_graph` | 42.8 ± 0.1 | 41.2 ± 0.7 | 40.1 ± 1.6 | 43.7 ± 1.0 | 45.7 ± 1.1 | 42.4 ± 0.9 | 43.5 ± 2.0 | 37.3 ± 2.0 |
| TEA · `frozen_graph_history` | 38.9 ± 0.6 | 37.7 ± 0.6 | 36.6 ± 1.6 | 40.2 ± 0.4 | 38.5 ± 1.2 | 39.3 ± 1.2 | 39.1 ± 0.7 | 34.1 ± 0.8 |
| TEA · `structure_only` | 38.6 ± 0.2 | 37.3 ± 0.6 | 35.6 ± 1.9 | 41.2 ± 0.8 | 38.2 ± 0.5 | 38.0 ± 1.5 | 39.3 ± 1.2 | 33.0 ± 1.5 |
| TEA · `shuffled_graph` | 31.4 ± 0.9 | 31.3 ± 0.6 | 33.2 ± 1.4 | 30.4 ± 1.7 | 29.6 ± 2.1 | 36.4 ± 2.0 | 33.4 ± 1.9 | 31.6 ± 0.3 |
| TEA · `question_only` | 26.0 ± 0.0 | 29.0 ± 0.0 | 26.8 ± 0.0 | 26.8 ± 0.0 | 24.9 ± 0.0 | 25.3 ± 0.0 | 27.6 ± 0.0 | 21.5 ± 0.0 |
| Warm · `oracle_updated_graph` | 44.5 | 44.6 | 44.8 | 45.6 | 46.5 | 45.1 | 42.1 | 41.1 |
| Warm · `predicted_updated_graph` | 44.1 | 44.8 | 44.8 | 46.1 | 47.1 | 45.8 | 42.8 | 40.7 |
| Warm · `frozen_graph_history` | 39.5 | 37.8 | 38.9 | 40.9 | 39.7 | 37.0 | 37.0 | 36.7 |
| Warm · `structure_only` | 38.4 | 38.5 | 37.7 | 41.4 | 37.7 | 37.0 | 35.4 | 32.7 |
| Warm · `shuffled_graph` | 29.5 | 31.5 | 31.1 | 29.6 | 29.3 | 34.7 | 32.7 | 31.3 |
| Warm · `question_only` | 26.0 | 29.0 | 26.8 | 26.8 | 24.9 | 25.3 | 27.6 | 21.5 |
| Random · `oracle_updated_graph` | 54.0 | 54.7 | 57.2 | 57.1 | 56.6 | 53.2 | 55.6 | 54.2 |
| Random · `predicted_updated_graph` | 53.8 | 54.9 | 56.9 | 57.4 | 55.2 | 53.2 | 55.9 | 53.5 |
| Random · `frozen_graph_history` | 42.6 | 41.1 | 42.3 | 42.1 | 40.7 | 41.4 | 39.1 | 41.8 |
| Random · `structure_only` | 43.1 | 41.5 | 43.4 | 42.8 | 37.7 | 43.1 | 35.7 | 39.1 |
| Random · `shuffled_graph` | 30.6 | 31.9 | 32.8 | 30.1 | 29.3 | 38.4 | 32.3 | 33.7 |
| Random · `question_only` | 26.0 | 29.0 | 26.8 | 26.8 | 24.9 | 25.3 | 27.6 | 21.5 |
| shared · `soft_prompt` | 34.1 | 31.8 | 36.7 | 35.0 | 34.0 | 38.0 | 36.0 | 37.7 |
| shared · `soft_prompt_no_history` | 31.7 | 31.8 | 33.7 | 34.3 | 36.7 | 37.0 | 33.0 | 34.7 |
| shared · `serialized_current_graph` | 28.6 | 30.0 | 27.1 | 29.1 | 24.9 | 29.6 | 27.6 | 26.3 |
| shared · `serialized_initial_history` | 30.7 | 28.5 | 31.3 | 29.3 | 30.3 | 31.6 | 28.6 | 29.0 |
| shared · `token_matched_history` | 27.5 | 29.5 | 28.6 | 27.9 | 26.6 | 27.3 | 27.3 | 26.3 |
| shared · `majority_answer` | 31.6 | 31.5 | 31.0 | 31.5 | 27.9 | 37.0 | 32.7 | 33.3 |
| shared · `stale_answer` | 63.4 | 63.6 | 66.2 | 63.6 | 57.9 | 65.7 | 59.9 | 66.3 |


**What it shows.**

- **Re-encoded tokens hold their accuracy at every session length** (GT 53.6–55.1%, random
  52.2–55.9%).
- **Stale tokens lose about 5–7 points once a session has more than one turn** (GT stale + history
  47.0% → 40.3–41.7%). They do not keep declining with length because the per-split quota fixes the
  answer-changing share at 36–39% for every length, so a further edit is as likely to restore an
  answer as to change it.
- **The soft prompt rises slightly with length** (30.6% → 35.6%) because the accumulated revision
  text answers more questions directly. Zero-shot Llama with the initial graph and history stays
  near 30%.
- **Turns 5–8 exist only in 8-turn sessions** (297 sessions), so the late-turn columns are noisier.

### 5.5 Density

#### By density (%)

| Reader · condition | sparse | medium | dense |
|---|---|---|---|
| GT · `oracle_updated_graph` | 53.0 ± 0.4 | 54.5 ± 0.7 | 55.4 ± 0.8 |
| GT · `predicted_updated_graph` | 52.8 ± 0.5 | 54.4 ± 0.9 | 55.4 ± 0.8 |
| GT · `frozen_graph_history` | 39.6 ± 0.8 | 41.7 ± 0.5 | 42.3 ± 0.9 |
| GT · `structure_only` | 40.0 ± 0.5 | 41.2 ± 0.6 | 42.5 ± 1.6 |
| GT · `shuffled_graph` | 32.3 ± 0.3 | 33.6 ± 0.8 | 31.4 ± 1.2 |
| GT · `question_only` | 29.2 ± 0.0 | 26.3 ± 0.0 | 23.9 ± 0.0 |
| TEA · `oracle_updated_graph` | 42.7 ± 0.8 | 42.6 ± 2.0 | 41.1 ± 0.4 |
| TEA · `predicted_updated_graph` | 42.5 ± 0.6 | 42.5 ± 2.0 | 41.2 ± 0.5 |
| TEA · `frozen_graph_history` | 38.5 ± 0.7 | 38.4 ± 1.0 | 37.7 ± 1.4 |
| TEA · `structure_only` | 38.2 ± 1.1 | 37.7 ± 1.6 | 37.8 ± 1.0 |
| TEA · `shuffled_graph` | 32.5 ± 1.0 | 31.8 ± 0.4 | 31.3 ± 1.0 |
| TEA · `question_only` | 29.2 ± 0.0 | 26.3 ± 0.0 | 23.9 ± 0.0 |
| Warm · `oracle_updated_graph` | 45.3 | 44.4 | 43.8 |
| Warm · `predicted_updated_graph` | 45.1 | 44.8 | 43.9 |
| Warm · `frozen_graph_history` | 38.0 | 38.5 | 39.9 |
| Warm · `structure_only` | 37.4 | 37.6 | 39.0 |
| Warm · `shuffled_graph` | 31.8 | 30.2 | 30.5 |
| Warm · `question_only` | 29.2 | 26.3 | 23.9 |
| Random · `oracle_updated_graph` | 53.4 | 56.5 | 55.8 |
| Random · `predicted_updated_graph` | 53.3 | 56.6 | 55.4 |
| Random · `frozen_graph_history` | 39.7 | 42.7 | 42.6 |
| Random · `structure_only` | 39.6 | 42.2 | 43.2 |
| Random · `shuffled_graph` | 30.2 | 34.0 | 31.2 |
| Random · `question_only` | 29.2 | 26.3 | 23.9 |
| shared · `soft_prompt` | 34.3 | 33.9 | 36.0 |
| shared · `soft_prompt_no_history` | 31.9 | 33.5 | 34.5 |
| shared · `serialized_current_graph` | 31.1 | 27.5 | 26.4 |
| shared · `serialized_initial_history` | 31.4 | 30.9 | 27.5 |
| shared · `token_matched_history` | 30.0 | 27.2 | 26.6 |
| shared · `majority_answer` | 31.7 | 31.0 | 32.7 |
| shared · `stale_answer` | 62.0 | 63.2 | 65.6 |


**What it shows.** GT oracle varies by 2.4 points across densities (53.0–55.4%) and stays 21–24
points ahead of the shuffled graph at every density (20.7, 20.9, 24.0).

### 5.6 Task

#### By task, multi-turn test (%), n = 495 per task

| Task | GT oracle | TEA oracle | Warm oracle | Random oracle | GT stale+hist | GT shuffled | Soft | Majority | Stale answer | Local ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| `constrained_reachability` | 61.0 ± 0.4 | 56.6 ± 0.3 | 55.4 | 61.8 | 52.9 ± 1.1 | 47.3 ± 0.7 | 50.7 | 50.1 | 74.1 | 64.4 |
| `cycle_membership` | 83.4 ± 1.3 | 49.2 ± 2.6 | 43.4 | 87.3 | 62.5 ± 0.6 | 54.5 ± 0.7 | 58.0 | 57.8 | 76.0 | 88.1 |
| `edge_exists` | 66.3 ± 1.9 | 58.1 ± 1.5 | 58.0 | 70.5 | 53.2 ± 1.3 | 46.9 ± 0.6 | 55.4 | 50.3 | 59.2 | 91.5 |
| `filtered_neighbor_count` | 34.3 ± 1.3 | 30.6 ± 1.6 | 31.5 | 33.1 | 30.0 ± 0.5 | 20.1 ± 0.7 | 18.2 | 19.0 | 54.9 | 33.7 |
| `most_common_attribute_within_hops` | 28.6 ± 2.6 | 36.0 ± 0.5 | 34.7 | 33.1 | 29.6 ± 1.6 | 28.6 ± 0.8 | 27.7 | 6.3 | 70.9 | – |
| `node_degree` | 99.9 ± 0.1 | 49.6 ± 1.2 | 72.7 | 97.6 | 40.8 ± 0.2 | 11.1 ± 0.0 | 11.7 | 13.9 | 54.9 | 100.0 |
| `reachability` | 68.4 ± 0.6 | 60.2 ± 1.7 | 63.2 | 70.3 | 61.0 ± 3.0 | 53.7 ± 1.5 | 59.2 | 59.6 | 72.7 | 91.3 |
| `shortest_path` | 25.5 ± 1.9 | 20.5 ± 2.1 | 22.2 | 24.8 | 24.0 ± 1.9 | 15.1 ± 1.4 | 15.2 | 15.4 | 54.5 | 55.6 |
| `within_hops_count` | 21.3 ± 2.8 | 18.6 ± 1.0 | 19.2 | 18.4 | 17.0 ± 0.2 | 14.7 ± 1.6 | 16.6 | 13.9 | 54.9 | 16.8 |

Full by-task table for every condition:

| Reader · condition | constrained_reachability | cycle_membership | edge_exists | filtered_neighbor_count | most_common_attribute_within_hops | node_degree | reachability | shortest_path | within_hops_count |
|---|---|---|---|---|---|---|---|---|---|
| GT · `oracle_updated_graph` | 61.0 ± 0.4 | 83.4 ± 1.3 | 66.3 ± 1.9 | 34.3 ± 1.3 | 28.6 ± 2.6 | 99.9 ± 0.1 | 68.4 ± 0.6 | 25.5 ± 1.9 | 21.3 ± 2.8 |
| GT · `predicted_updated_graph` | 59.9 ± 0.3 | 83.2 ± 1.3 | 66.4 ± 2.0 | 34.6 ± 2.6 | 28.8 ± 2.7 | 99.7 ± 0.1 | 68.3 ± 0.7 | 25.5 ± 1.9 | 21.3 ± 2.7 |
| GT · `frozen_graph_history` | 52.9 ± 1.1 | 62.5 ± 0.6 | 53.2 ± 1.3 | 30.0 ± 0.5 | 29.6 ± 1.6 | 40.8 ± 0.2 | 61.0 ± 3.0 | 24.0 ± 1.9 | 17.0 ± 0.2 |
| GT · `structure_only` | 52.3 ± 1.3 | 62.2 ± 0.7 | 50.9 ± 0.0 | 31.2 ± 1.0 | 29.5 ± 0.9 | 41.6 ± 0.2 | 61.1 ± 2.0 | 24.5 ± 1.3 | 17.8 ± 0.6 |
| GT · `shuffled_graph` | 47.3 ± 0.7 | 54.5 ± 0.7 | 46.9 ± 0.6 | 20.1 ± 0.7 | 28.6 ± 0.8 | 11.1 ± 0.0 | 53.7 ± 1.5 | 15.1 ± 1.4 | 14.7 ± 1.6 |
| GT · `question_only` | 49.9 ± 0.0 | 41.6 ± 0.0 | 49.7 ± 0.0 | 19.8 ± 0.0 | 1.0 ± 0.0 | 11.3 ± 0.0 | 40.6 ± 0.0 | 13.7 ± 0.0 | 10.7 ± 0.0 |
| TEA · `oracle_updated_graph` | 56.6 ± 0.3 | 49.2 ± 2.6 | 58.1 ± 1.5 | 30.6 ± 1.6 | 36.0 ± 0.5 | 49.6 ± 1.2 | 60.2 ± 1.7 | 20.5 ± 2.1 | 18.6 ± 1.0 |
| TEA · `predicted_updated_graph` | 56.0 ± 0.6 | 49.2 ± 2.6 | 58.1 ± 1.5 | 30.8 ± 2.2 | 36.0 ± 0.7 | 49.4 ± 1.2 | 60.2 ± 1.7 | 20.5 ± 2.1 | 18.6 ± 1.0 |
| TEA · `frozen_graph_history` | 51.7 ± 1.4 | 45.9 ± 2.5 | 52.7 ± 0.6 | 27.9 ± 1.0 | 37.4 ± 0.5 | 32.5 ± 0.8 | 57.2 ± 0.3 | 20.9 ± 2.4 | 17.6 ± 0.7 |
| TEA · `structure_only` | 52.2 ± 1.0 | 47.7 ± 2.2 | 49.1 ± 2.2 | 28.8 ± 1.1 | 36.8 ± 0.7 | 32.2 ± 1.2 | 56.4 ± 0.9 | 21.5 ± 2.7 | 16.6 ± 1.1 |
| TEA · `shuffled_graph` | 48.9 ± 1.8 | 51.5 ± 0.9 | 46.3 ± 1.3 | 20.3 ± 1.0 | 27.9 ± 0.5 | 13.8 ± 0.9 | 50.0 ± 4.2 | 14.3 ± 0.6 | 13.7 ± 0.6 |
| TEA · `question_only` | 49.9 ± 0.0 | 41.6 ± 0.0 | 49.7 ± 0.0 | 19.8 ± 0.0 | 1.0 ± 0.0 | 11.3 ± 0.0 | 40.6 ± 0.0 | 13.7 ± 0.0 | 10.7 ± 0.0 |
| Warm · `oracle_updated_graph` | 55.4 | 43.4 | 58.0 | 31.5 | 34.7 | 72.7 | 63.2 | 22.2 | 19.2 |
| Warm · `predicted_updated_graph` | 54.9 | 43.4 | 58.0 | 32.7 | 34.9 | 72.5 | 63.2 | 22.4 | 19.2 |
| Warm · `frozen_graph_history` | 50.9 | 44.4 | 51.9 | 31.3 | 35.8 | 39.2 | 57.8 | 20.4 | 17.2 |
| Warm · `structure_only` | 49.5 | 43.4 | 47.7 | 29.3 | 35.2 | 38.4 | 57.8 | 21.8 | 19.0 |
| Warm · `shuffled_graph` | 47.3 | 53.3 | 48.1 | 18.4 | 26.5 | 12.1 | 43.6 | 13.5 | 14.5 |
| Warm · `question_only` | 49.9 | 41.6 | 49.7 | 19.8 | 1.0 | 11.3 | 40.6 | 13.7 | 10.7 |
| Random · `oracle_updated_graph` | 61.8 | 87.3 | 70.5 | 33.1 | 33.1 | 97.6 | 70.3 | 24.8 | 18.4 |
| Random · `predicted_updated_graph` | 60.8 | 87.3 | 70.5 | 33.3 | 32.9 | 97.4 | 70.3 | 24.8 | 18.4 |
| Random · `frozen_graph_history` | 53.7 | 63.2 | 54.5 | 31.5 | 30.7 | 41.2 | 62.2 | 21.6 | 16.4 |
| Random · `structure_only` | 53.9 | 63.0 | 52.3 | 29.9 | 32.3 | 41.4 | 61.0 | 23.6 | 17.4 |
| Random · `shuffled_graph` | 46.1 | 56.4 | 45.1 | 20.0 | 28.1 | 11.1 | 54.5 | 12.3 | 12.9 |
| Random · `question_only` | 49.9 | 41.6 | 49.7 | 19.8 | 1.0 | 11.3 | 40.6 | 13.7 | 10.7 |
| shared · `soft_prompt` | 50.7 | 58.0 | 55.4 | 18.2 | 27.7 | 11.7 | 59.2 | 15.2 | 16.6 |
| shared · `soft_prompt_no_history` | 49.9 | 54.9 | 50.7 | 18.2 | 27.1 | 11.5 | 59.6 | 12.9 | 14.9 |
| shared · `serialized_current_graph` | 50.7 | 47.1 | 50.3 | 16.6 | 8.7 | 14.7 | 39.2 | 15.6 | 12.3 |
| shared · `serialized_initial_history` | 50.9 | 56.0 | 55.8 | 19.8 | 2.2 | 13.7 | 42.6 | 16.8 | 11.7 |
| shared · `token_matched_history` | 50.1 | 46.7 | 55.2 | 19.4 | 1.0 | 12.9 | 40.4 | 13.5 | 12.3 |
| shared · `majority_answer` | 50.1 | 57.8 | 50.3 | 19.0 | 6.3 | 13.9 | 59.6 | 15.4 | 13.9 |
| shared · `stale_answer` | 74.1 | 76.0 | 59.2 | 54.9 | 70.9 | 54.9 | 72.7 | 54.5 | 54.9 |


**What it shows.**

- **Re-encoding helps most where the answer is a local structural fact that edits change.** For GT:
  node degree 99.9% vs 40.8% stale, cycle membership 83.4% vs 62.5%, edge existence 66.3% vs 53.2%.
- **GT reaches the local ceiling only on node degree and the two counting tasks.** It is far below
  on shortest path (25.5 vs 55.6), reachability (68.4 vs 91.3) and edge existence (66.3 vs 91.5).
  It reaches or slightly exceeds the low counting ceilings (filtered count 34.3 vs 33.7, within-hops
  count 21.3 vs 16.8). On most-common attribute it equals the shuffled graph (28.6 vs 28.6).
- **The random GNN matches or beats GT on the structural tasks:** cycle 87.3, edge 70.5,
  reachability 70.3.
- **TEA is at chance on cycle membership** (49.2%) and at 49.6% on node degree. Its best task
  relative to the others is most-common attribute (36.0%).
- **The stale-answer column is a reference only.** It is high by construction (100% on unchanged
  turns, 0% on changed ones) and is what a system that ignores edits would score.


### 5.7 Diagnostic slices

#### Diagnostic slices per run (%)

`answer_changing` = gold ≠ stale answer (1,622 turns); `changed_and_no_mention` = answer-changing and the question names no edited station (359); `local_wrong` = both learned local classifiers wrong (1,052).

| Run | Condition | All | Answer-changing | Changed & unmentioned | Local-wrong |
|---|---|---|---|---|---|
| v4_graphtoken_s42 | `frozen_graph_history` | 40.5 | 23.1 | 27.3 | 23.0 |
| v4_graphtoken_s42 | `majority_answer` | 31.8 | 27.3 | 25.3 | 24.6 |
| v4_graphtoken_s42 | `oracle_updated_graph` | 54.1 | 51.9 | 39.8 | 22.6 |
| v4_graphtoken_s42 | `predicted_updated_graph` | 54.0 | 51.5 | 39.8 | 23.2 |
| v4_graphtoken_s42 | `question_only` | 26.5 | 22.4 | 24.2 | 24.6 |
| v4_graphtoken_s42 | `shuffled_graph` | 32.1 | 29.2 | 27.9 | 21.2 |
| v4_graphtoken_s42 | `stale_answer` | 63.6 | 0.0 | 0.0 | 57.3 |
| v4_graphtoken_s42 | `structure_only` | 40.7 | 21.8 | 27.9 | 23.4 |
| v4_graphtoken_s43 | `frozen_graph_history` | 41.7 | 24.7 | 25.6 | 21.4 |
| v4_graphtoken_s43 | `majority_answer` | 31.8 | 27.3 | 25.3 | 24.6 |
| v4_graphtoken_s43 | `oracle_updated_graph` | 54.5 | 52.0 | 37.6 | 22.0 |
| v4_graphtoken_s43 | `predicted_updated_graph` | 54.5 | 51.3 | 37.6 | 22.6 |
| v4_graphtoken_s43 | `question_only` | 26.5 | 22.4 | 24.2 | 24.6 |
| v4_graphtoken_s43 | `shuffled_graph` | 32.8 | 30.1 | 29.0 | 21.3 |
| v4_graphtoken_s43 | `stale_answer` | 63.6 | 0.0 | 0.0 | 57.3 |
| v4_graphtoken_s43 | `structure_only` | 41.7 | 23.9 | 25.9 | 22.1 |
| v4_graphtoken_s44 | `frozen_graph_history` | 41.4 | 24.9 | 27.6 | 23.0 |
| v4_graphtoken_s44 | `majority_answer` | 31.8 | 27.3 | 25.3 | 24.6 |
| v4_graphtoken_s44 | `oracle_updated_graph` | 54.3 | 51.6 | 37.3 | 22.1 |
| v4_graphtoken_s44 | `predicted_updated_graph` | 54.0 | 50.7 | 37.0 | 22.1 |
| v4_graphtoken_s44 | `question_only` | 26.5 | 22.4 | 24.2 | 24.6 |
| v4_graphtoken_s44 | `shuffled_graph` | 32.5 | 28.9 | 26.5 | 21.2 |
| v4_graphtoken_s44 | `stale_answer` | 63.6 | 0.0 | 0.0 | 57.3 |
| v4_graphtoken_s44 | `structure_only` | 41.3 | 23.2 | 27.9 | 24.1 |
| v4_tea_s42 | `frozen_graph_history` | 38.3 | 28.7 | 27.9 | 23.2 |
| v4_tea_s42 | `majority_answer` | 31.8 | 27.3 | 25.3 | 24.6 |
| v4_tea_s42 | `oracle_updated_graph` | 42.0 | 37.5 | 27.9 | 25.2 |
| v4_tea_s42 | `predicted_updated_graph` | 41.8 | 37.3 | 28.1 | 25.3 |
| v4_tea_s42 | `question_only` | 26.5 | 22.4 | 24.2 | 24.6 |
| v4_tea_s42 | `shuffled_graph` | 31.4 | 28.1 | 24.2 | 22.1 |
| v4_tea_s42 | `stale_answer` | 63.6 | 0.0 | 0.0 | 57.3 |
| v4_tea_s42 | `structure_only` | 38.0 | 28.2 | 29.8 | 24.7 |
| v4_tea_s43 | `frozen_graph_history` | 38.1 | 29.6 | 28.1 | 21.3 |
| v4_tea_s43 | `majority_answer` | 31.8 | 27.3 | 25.3 | 24.6 |
| v4_tea_s43 | `oracle_updated_graph` | 41.8 | 37.3 | 24.8 | 21.6 |
| v4_tea_s43 | `predicted_updated_graph` | 41.9 | 37.2 | 24.8 | 22.1 |
| v4_tea_s43 | `question_only` | 26.5 | 22.4 | 24.2 | 24.6 |
| v4_tea_s43 | `shuffled_graph` | 32.4 | 29.3 | 24.5 | 20.2 |
| v4_tea_s43 | `stale_answer` | 63.6 | 0.0 | 0.0 | 57.3 |
| v4_tea_s43 | `structure_only` | 38.0 | 28.1 | 25.3 | 21.8 |
| v4_tea_s44 | `frozen_graph_history` | 38.1 | 28.9 | 26.5 | 22.5 |
| v4_tea_s44 | `majority_answer` | 31.8 | 27.3 | 25.3 | 24.6 |
| v4_tea_s44 | `oracle_updated_graph` | 42.6 | 38.6 | 26.2 | 24.0 |
| v4_tea_s44 | `predicted_updated_graph` | 42.5 | 38.3 | 26.5 | 24.1 |
| v4_tea_s44 | `question_only` | 26.5 | 22.4 | 24.2 | 24.6 |
| v4_tea_s44 | `shuffled_graph` | 31.7 | 28.1 | 24.8 | 21.5 |
| v4_tea_s44 | `stale_answer` | 63.6 | 0.0 | 0.0 | 57.3 |
| v4_tea_s44 | `structure_only` | 37.7 | 27.4 | 24.8 | 23.1 |
| v4_graphtoken_warm_s42 | `frozen_graph_history` | 38.8 | 27.0 | 29.5 | 24.0 |
| v4_graphtoken_warm_s42 | `majority_answer` | 31.8 | 27.3 | 25.3 | 24.6 |
| v4_graphtoken_warm_s42 | `oracle_updated_graph` | 44.5 | 40.1 | 25.6 | 21.6 |
| v4_graphtoken_warm_s42 | `predicted_updated_graph` | 44.6 | 40.0 | 26.2 | 22.3 |
| v4_graphtoken_warm_s42 | `question_only` | 26.5 | 22.4 | 24.2 | 24.6 |
| v4_graphtoken_warm_s42 | `shuffled_graph` | 30.8 | 27.6 | 25.9 | 21.9 |
| v4_graphtoken_warm_s42 | `stale_answer` | 63.6 | 0.0 | 0.0 | 57.3 |
| v4_graphtoken_warm_s42 | `structure_only` | 38.0 | 26.1 | 27.3 | 23.0 |
| v4_tea_random_sum | `frozen_graph_history` | 41.7 | 24.8 | 24.8 | 21.3 |
| v4_tea_random_sum | `majority_answer` | 31.8 | 27.3 | 25.3 | 24.6 |
| v4_tea_random_sum | `oracle_updated_graph` | 55.2 | 52.8 | 40.9 | 19.2 |
| v4_tea_random_sum | `predicted_updated_graph` | 55.1 | 52.1 | 40.7 | 20.2 |
| v4_tea_random_sum | `question_only` | 26.5 | 22.4 | 24.2 | 24.6 |
| v4_tea_random_sum | `shuffled_graph` | 31.8 | 28.0 | 27.3 | 18.6 |
| v4_tea_random_sum | `stale_answer` | 63.6 | 0.0 | 0.0 | 57.3 |
| v4_tea_random_sum | `structure_only` | 41.7 | 22.8 | 26.7 | 21.6 |
| v4_soft_prompt | `soft_prompt` | 34.7 | 32.2 | 24.8 | 18.6 |
| v4_soft_prompt | `soft_prompt_no_history` | 33.3 | 29.3 | 22.0 | 18.2 |
| v4_text_baselines | `serialized_current_graph` | 28.4 | 25.4 | 26.7 | 24.0 |
| v4_text_baselines | `serialized_initial_history` | 29.9 | 27.8 | 25.3 | 25.4 |
| v4_text_baselines | `token_matched_history` | 27.9 | 25.5 | 24.8 | 24.1 |


**What it shows.**

- **No graph reader beats the question alone on local-wrong turns.** Every run is at 19–25% under
  every condition, and every reader's oracle mean is below `question_only` (24.6%): GT 22.2 ± 0.3,
  TEA 23.6 ± 1.8, warm start 21.6, random 19.2.
- **So no reader shows reasoning beyond a two-hop neighbourhood.** The readers use the graph to
  track local facts such as degree, status and nearby closures, which is what most updates change.

#### Seed means ± s.d. by slice (3-seed groups)

| Group · condition [slice] | Mean ± s.d. |
|---|---|
| v4_graphtoken_s* frozen_graph_history [all] | 41.2 ± 0.6 |
| v4_graphtoken_s* frozen_graph_history [answer_changing] | 24.2 ± 1.0 |
| v4_graphtoken_s* frozen_graph_history [changed_and_no_mention] | 26.8 ± 1.1 |
| v4_graphtoken_s* frozen_graph_history [local_wrong] | 22.5 ± 0.9 |
| v4_graphtoken_s* frozen_graph_history [no_mention] | 47.6 ± 0.3 |
| v4_graphtoken_s* frozen_graph_history [shortcut_wrong] | 25.5 ± 0.6 |
| v4_graphtoken_s* oracle_updated_graph [all] | 54.3 ± 0.2 |
| v4_graphtoken_s* oracle_updated_graph [answer_changing] | 51.8 ± 0.2 |
| v4_graphtoken_s* oracle_updated_graph [changed_and_no_mention] | 38.3 ± 1.4 |
| v4_graphtoken_s* oracle_updated_graph [local_wrong] | 22.2 ± 0.3 |
| v4_graphtoken_s* oracle_updated_graph [no_mention] | 53.6 ± 0.2 |
| v4_graphtoken_s* oracle_updated_graph [shortcut_wrong] | 26.9 ± 0.4 |
| v4_graphtoken_s* predicted_updated_graph [all] | 54.2 ± 0.3 |
| v4_graphtoken_s* predicted_updated_graph [answer_changing] | 51.2 ± 0.4 |
| v4_graphtoken_s* predicted_updated_graph [changed_and_no_mention] | 38.2 ± 1.5 |
| v4_graphtoken_s* predicted_updated_graph [local_wrong] | 22.7 ± 0.5 |
| v4_graphtoken_s* predicted_updated_graph [no_mention] | 53.7 ± 0.3 |
| v4_graphtoken_s* predicted_updated_graph [shortcut_wrong] | 27.0 ± 0.6 |
| v4_graphtoken_s* question_only [all] | 26.5 ± 0.0 |
| v4_graphtoken_s* question_only [answer_changing] | 22.4 ± 0.0 |
| v4_graphtoken_s* question_only [changed_and_no_mention] | 24.2 ± 0.0 |
| v4_graphtoken_s* question_only [local_wrong] | 24.6 ± 0.0 |
| v4_graphtoken_s* question_only [no_mention] | 28.4 ± 0.0 |
| v4_graphtoken_s* question_only [shortcut_wrong] | 27.2 ± 0.0 |
| v4_graphtoken_s* shuffled_graph [all] | 32.5 ± 0.4 |
| v4_graphtoken_s* shuffled_graph [answer_changing] | 29.4 ± 0.6 |
| v4_graphtoken_s* shuffled_graph [changed_and_no_mention] | 27.8 ± 1.3 |
| v4_graphtoken_s* shuffled_graph [local_wrong] | 21.2 ± 0.1 |
| v4_graphtoken_s* shuffled_graph [no_mention] | 32.7 ± 0.4 |
| v4_graphtoken_s* shuffled_graph [shortcut_wrong] | 25.0 ± 0.3 |
| v4_graphtoken_s* structure_only [all] | 41.2 ± 0.5 |
| v4_graphtoken_s* structure_only [answer_changing] | 23.0 ± 1.1 |
| v4_graphtoken_s* structure_only [changed_and_no_mention] | 27.2 ± 1.1 |
| v4_graphtoken_s* structure_only [local_wrong] | 23.2 ± 1.1 |
| v4_graphtoken_s* structure_only [no_mention] | 48.1 ± 0.0 |
| v4_graphtoken_s* structure_only [shortcut_wrong] | 26.5 ± 0.8 |
| v4_tea_s* frozen_graph_history [all] | 38.2 ± 0.1 |
| v4_tea_s* frozen_graph_history [answer_changing] | 29.1 ± 0.5 |
| v4_tea_s* frozen_graph_history [changed_and_no_mention] | 27.5 ± 0.9 |
| v4_tea_s* frozen_graph_history [local_wrong] | 22.3 ± 1.0 |
| v4_tea_s* frozen_graph_history [no_mention] | 39.8 ± 0.4 |
| v4_tea_s* frozen_graph_history [shortcut_wrong] | 26.2 ± 0.4 |
| v4_tea_s* oracle_updated_graph [all] | 42.1 ± 0.4 |
| v4_tea_s* oracle_updated_graph [answer_changing] | 37.8 ± 0.7 |
| v4_tea_s* oracle_updated_graph [changed_and_no_mention] | 26.3 ± 1.5 |
| v4_tea_s* oracle_updated_graph [local_wrong] | 23.6 ± 1.8 |
| v4_tea_s* oracle_updated_graph [no_mention] | 40.8 ± 0.6 |
| v4_tea_s* oracle_updated_graph [shortcut_wrong] | 27.7 ± 1.3 |
| v4_tea_s* predicted_updated_graph [all] | 42.1 ± 0.3 |
| v4_tea_s* predicted_updated_graph [answer_changing] | 37.6 ± 0.6 |
| v4_tea_s* predicted_updated_graph [changed_and_no_mention] | 26.5 ± 1.7 |
| v4_tea_s* predicted_updated_graph [local_wrong] | 23.8 ± 1.6 |
| v4_tea_s* predicted_updated_graph [no_mention] | 40.8 ± 0.6 |
| v4_tea_s* predicted_updated_graph [shortcut_wrong] | 27.9 ± 1.1 |
| v4_tea_s* question_only [all] | 26.5 ± 0.0 |
| v4_tea_s* question_only [answer_changing] | 22.4 ± 0.0 |
| v4_tea_s* question_only [changed_and_no_mention] | 24.2 ± 0.0 |
| v4_tea_s* question_only [local_wrong] | 24.6 ± 0.0 |
| v4_tea_s* question_only [no_mention] | 28.4 ± 0.0 |
| v4_tea_s* question_only [shortcut_wrong] | 27.2 ± 0.0 |
| v4_tea_s* shuffled_graph [all] | 31.9 ± 0.5 |
| v4_tea_s* shuffled_graph [answer_changing] | 28.5 ± 0.7 |
| v4_tea_s* shuffled_graph [changed_and_no_mention] | 24.5 ± 0.3 |
| v4_tea_s* shuffled_graph [local_wrong] | 21.3 ± 0.9 |
| v4_tea_s* shuffled_graph [no_mention] | 32.0 ± 0.6 |
| v4_tea_s* shuffled_graph [shortcut_wrong] | 25.7 ± 0.3 |
| v4_tea_s* structure_only [all] | 37.9 ± 0.1 |
| v4_tea_s* structure_only [answer_changing] | 27.9 ± 0.4 |
| v4_tea_s* structure_only [changed_and_no_mention] | 26.6 ± 2.7 |
| v4_tea_s* structure_only [local_wrong] | 23.2 ± 1.5 |
| v4_tea_s* structure_only [no_mention] | 40.1 ± 0.7 |
| v4_tea_s* structure_only [shortcut_wrong] | 26.9 ± 1.3 |


### 5.8 Write path (RQ1)

#### Write path (frozen Llama, few-shot edit prompt; identical for every reader)

| Quantity | Value |
|---|---|
| Execution-equivalent edit accuracy | 96.6% |
| Graph state exactly right (errors persist) | 91.1% |
| Same predicted edits in every run | True |

| Gold operations | Edit accuracy (%) | Turns |
|---|---|---|
| ADD EDGE | 79.9 | 608 |
| ADD EDGE+DEL EDGE | 99.2 | 591 |
| ADD EDGE+DEL EDGE+SET NODE | 99.6 | 284 |
| ADD EDGE+SET NODE | 99.0 | 577 |
| DEL EDGE | 99.5 | 571 |
| DEL EDGE+SET NODE | 99.6 | 540 |
| SET NODE | 97.6 | 539 |
| noop | 100.0 | 745 |

| Paraphrase family | Edit accuracy (%) |
|---|---|
| direct | 99.3 |
| notice | 91.5 |
| service | 98.9 |


**What it shows.**

- **Edits are almost always correct.** The frozen LLM writes an execution-equivalent program on
  96.6% of test turns, and the graph built from its own edits is exactly right on 91.1% of turns
  (an error persists for the rest of the session).
- **Most operations are near-perfect.** NOOP is always correct and status and deletion edits are
  above 97%. The weakest case is a lone `ADD EDGE` at 79.9%.
- **The held-out family is harder:** 91.5% on `notice` against 98.9–99.3% for the other two,
  confounded with task as noted in §3.
- **Edit errors cost at most 0.1 points overall for every reader.** Oracle and predicted agree
  within 1.2 points in every breakdown: length, change status, density and task. The largest gaps
  are 1.1 (GT, constrained reachability) and 1.2 (warm start, filtered count).
- **The predicted edits are identical in every run**, because the edit model is the same frozen LLM.

### 5.9 GNN probe

#### GNN probe: linear decodability from frozen node embeddings (test-split graphs, accuracy %)

| Embeddings | Degree | Status | Within 2 hops | Mean within-graph cosine | Effective rank | Dim |
|---|---|---|---|---|---|---|
| BERT input features (no GNN) | 32.2 | 100.0 | 57.9 | 0.97 | 13.0 | 768 |
| Frozen random GNN | 100.0 | 96.6 | 92.3 | 0.92 | 1.9 | 4096 |
| TEA-GLM-pretrained GNN | 96.4 | 88.1 | 79.2 | 0.24 | 16.2 | 4096 |
| GraphToken-style GNN (s42) | 100.0 | 95.8 | 96.9 | 0.97 | 1.9 | 4096 |
| Warm-start GNN (s42) | 98.3 | 88.1 | 82.1 | 0.37 | 12.9 | 4096 |
| Majority class | 36.5 | 88.2 | 59.0 |  |  |  |

Probe graphs: 600 train / 396 test.


How the probe works (`scripts/probe_gnn_v4.py`):

- **Fit:** node embeddings of static graphs are computed once per encoder. A multinomial logistic
  regression (full-batch L-BFGS on standardized features) is fit on train-split graphs and scored
  on test-split graphs.
- **Targets:** node degree, station status, and the number of stations within two hops.
- **Collapse measures:** "cosine" is the mean cosine similarity between nodes of the same graph.
  "Effective rank" is the participation ratio of the covariance spectrum.

**What it shows.**

- **BERT features alone:** status is 100% decodable (it is in the text) but degree is not (32.2%).
- **A frozen random GNN already decodes everything the tasks need:** degree 100%, status 96.6%,
  two-hop count 92.3%. Sum aggregation adds degree (the sum's size grows with neighbour count) and
  neighbourhood counts, and keeps the node's own features. Joint training (GT GNN: 100 / 95.8 / 96.9)
  adds almost nothing, which explains finding 4.
- **The TEA-GLM-pretrained GNN loses status:** 88.1% is exactly the majority class (88.2%). It also
  loses some two-hop information (79.2%).
- **The loss is not collapse.** The TEA-GLM embeddings are the *least* collapsed (cosine 0.24,
  effective rank 16.2), while GT and random have effective rank ~1.9 with cosine 0.92–0.97 and still
  decode everything. Collapse statistics do not measure what a linear probe or projector can
  recover.
- **The warm start does not recover status.** It keeps the pretrained GNN's status loss (88.1%) and
  most of its two-hop loss (82.1%) after one epoch of joint training at lr 1e-4, but recovers degree
  (98.3%). That matches its multi-turn profile: node degree 72.7%, cycle 43.4%.

---

## 6. Explanations

**Why re-encoding beats text history.** The QA call never sees the revisions in the oracle and
predicted conditions, so their gain must come from the updated tokens. Adding the revision text to
stale tokens does not help (`frozen_graph_history` = `structure_only`). Frozen Llama cannot apply a
revision to a graph it holds only as soft tokens. Even when it holds the whole graph as text, it
cannot answer these questions zero-shot (28–30%).

**Why a frozen random GNN matches GraphToken.** The projector is one linear layer into 10 × 4096
input embeddings, and a random sum-aggregation GraphSAGE already exposes degree, status and
neighbourhood counts linearly (probe). The bottleneck is the frozen LLM's reading of 10 soft tokens,
not the encoder. Training the GNN end to end through the LLM loss for one epoch does not add a
representation the projector can use. GT is slightly better on local-wrong turns (+3.4), but neither
beats question-only there. So the GraphToken-style advantage over TEA comes from *not* pretraining
the GNN contrastively, not from training it jointly.

**Why TEA-style lags.** TEA-GLM's instance loss asks the GNN to give the same embedding to two
corrupted views of a graph: one with up to 40% of edges dropped, one with features masked. A GNN
that succeeds becomes invariant to exactly the local details our questions test: whether a specific
edge exists, a node's exact degree, a station's open/closed status. The probe confirms status is
gone (88.1% = majority), and TEA is at chance on cycle membership. TEA-GLM was designed for
zero-shot transfer of semantic categories (e.g. paper topics), where that invariance is desirable.
Our updates change exactly the details it discards.

**Why TEA wins on attribute questions.** TEA-GLM's feature-wise loss projects node embeddings onto
the principal components of Llama's token-embedding table. This seems to keep the content of the
node text (line, architecture, music and so on) in a form the projector can map to word-like
tokens. GT and the random GNN are trained or initialised without that pressure. Their sum
aggregation mixes neighbours into the node's own features, which suits counting and degree but
blurs categorical attribute values. The warm start shows the same attribute advantage (46.2%), so
it comes from the pretraining, not from freezing. This explanation is a hypothesis consistent with
the probe and the per-task results; we did not run a dedicated attribute probe.

**Why everything is local.** A 3-layer GNN with a linear projector into a frozen LLM can expose
features of the queried stations and their neighbourhood. Nothing in it computes paths or global
connectivity, and the learned local ceilings show how far such features go. Every reader sits at or
below those ceilings, and on turns no local rule solves, every reader is at question-only level.
The benchmark rewards tracking local facts that updates change. It does not show that soft graph
tokens support multi-hop reasoning.

**Why the write path is cheap.** Revisions are templated, and the three families map cleanly to
the four operators. The few-shot prompt covers the format, and the executor resolves multi-word
station names. The main error is a lone `ADD EDGE` (79.9%), and errors only matter when they change
the graph on a later question's path. Free-form revisions would be harder; this is a limitation.

---

## 7. Caveats and limitations

- The benchmark is synthetic and its revisions follow templates, so edit accuracy likely overstates
  performance on free-form dialogue.
- **Paraphrase family is confounded with task** (§3). Never report read accuracy by family as a
  wording effect.
- GT and TEA use three seeds. The warm start, random GNN, soft prompt and text baselines use one, so
  the GT-vs-random comparison (−1.1, CI −2.3 to 0.0) rests on one random-GNN seed. Call it a tie,
  not a win for the random GNN.
- Every reader trains for one epoch on static QA. Longer training could change the GT-vs-random
  ordering.
- The projector lr was chosen on validation loss after a first round at 1e-3, in which the frozen-
  and pretrained-GNN readers had not converged. The soft prompt's lr (1e-3) was not swept.
- The learned local classifiers reach 67–70% on the structural tasks, so absolute accuracy on this
  benchmark mostly measures local reading.
- Our readers are "TEA-style" and "GraphToken-style", not reproductions: readouts, tasks and
  training data differ from the original papers.
- The attribute explanation for TEA (§6) is a hypothesis without a dedicated probe.
- Open items from the proposal: a contradiction probe (a session that evolves into a logically
  inconsistent state the GLM should flag), and free-form revisions.

---

## 8. Writing the LaTeX report

**Suggested structure and what goes where.**

| Report section | Content | Source in this guide |
|---|---|---|
| Abstract | Findings 1, 2, 3, 4, 5, 7 with their numbers | §1 |
| Introduction / RQs / contributions | §2 | §2 |
| Problem formulation, methodology | §2, §4 | §2, §4 |
| Benchmark + shortcut ceilings | §3, Table "Benchmark statistics", shortcut table | §3 |
| Static results | Table 5.1 overall; per-task in the appendix figure | §5.1 |
| Multi-turn conditions | Table 5.2 + paired differences | §5.2 |
| Degradation over turns | Figures fig1/fig2 + session-length table | §5.4 |
| Answer-changing turns | Table 5.3 + fig3 | §5.3 |
| Density, task, local shortcuts | Tables 5.5, 5.6, local-wrong slice | §5.5–5.7 |
| Write path | §5.8 | §5.8 |
| What the readers learn (probe) | Probe table + §6 TEA / random explanations | §5.9, §6 |
| Inferences | §6 | §6 |
| Limitations | §7 | §7 |
| Appendix | combined figures, all-conditions figures, per-task static figures, hyperparameters, task definitions | §3, §4 |

The **contribution (iv)** sentence in the current draft must change to the new finding. Suggested
wording: "A diagnosis of what the readers learn: a frozen random GNN matches a jointly trained one,
both read local features only, and TEA-GLM contrastive pretraining discards the node status that
updates change while keeping node-text content." The section title "Why TEA-style lags" can become
"What the readers learn".

**Figures** (in `documents/mid-report/figures/`, regenerated from the final runs by
`scripts/make_figures_v4.py`). There is one set per reader prefix (`graphtoken`, `tea`, `warm`,
`random`):

| File | Shows |
|---|---|
| `<reader>_fig1_accuracy_by_session_length.png` | main conditions vs session length (seed range shaded) |
| `<reader>_fig2_accuracy_by_turn_index.png` | main conditions vs turn index |
| `<reader>_fig3_edit_tracking.png` | answer-changing vs unchanged turns |
| `<reader>_figA1/A2/A3_all_conditions_*.png` | the same three with every condition |
| `<reader>_figA4_static_accuracy_by_task.png` | static per-task accuracy vs soft prompt and local ceiling |
| `v4_combined_conditions.png` | all four readers side by side, every condition |
| `v4_combined_by_session_length.png` | oracle by session length, all readers + soft prompt |
| `v4_combined_edit_tracking.png` | oracle on changed / unchanged turns, all readers |

The current draft uses the `graphtoken` and `tea` sets in the main text and the combined figures in
the appendix. Since the random GNN is now a headline result, consider adding its fig1 or fig3, or
pointing to the combined figures from the main text.

**Status of `main.tex`.** As of 31 October, `main.tex` has been updated with every value in this
guide: the `@@path@@` placeholder tokens were filled from `report_tables.json`, and the prose,
the TEA / warm-start / random-GNN paragraphs, the probe section (now "What the readers learn"),
the inferences and the limitations were rewritten. The table below records which hand-typed
numbers changed relative to the earlier draft, for anyone checking an older copy:

| Draft line (approx.) | Current text | Final value |
|---|---|---|
| Abstract | GT 53.7; stale + history 41.3; shuffled 32.4 | 54.3; 41.2; 32.5 |
| Abstract | edits "cost 0.2 points" | 0.1 points |
| Abstract | changed turns 51.2 vs 25.2 | 51.8 vs 24.2 |
| Abstract | "every model is near 21%" on turns no local rule solves | 19–24%, below question-only (24.6%) |
| Abstract, contributions, §Why TEA lags | TEA lags, nothing about the random GNN | add finding 4 (random GNN = GT) and finding 6 (TEA keeps attributes) |
| Static table caption | GT test s.d. 0.6 | 0.1 |
| Static text | GT 49.7; degree 97.5, cycle 79.0, reach 65.2, edge 61.7, shortest path 30.0 | 50.3; 99.1, 79.0, 65.7, 61.2, 33.6 |
| Conditions caption | GT oracle s.d. 0.8 | 0.2 |
| Conditions text | 53.7 (CI 51.5–56.2), predicted 53.5 | 54.3 (s42 CI 51.8–56.3), predicted 54.2 |
| Conditions text | vs shuffled 21.8 (19.4–24.2), stale+hist 12.9 (11.1–14.7), stale 13.0 (11.2–14.9) | 22.0 (19.5–24.5), 13.6 (11.7–15.5), 13.4 (11.6–15.4) |
| Conditions text | vs soft prompt 19.1 (16.5–21.7), no history 20.5 (18.0–23.0) | 19.4 (16.8–22.0), 20.8 (18.2–23.3) |
| Degradation text | re-encoded 52.5–54.5; stale 45.1 → 38.5–41.6, loses 3.5–6.6 | 53.6–55.1; stale + history 47.0 → 40.3–41.7, loses 5.3–6.7 |
| Changing text | unchanged 50.5 vs 55.1; changed 25.2 vs 51.2 (26-pt gap vs 4.6) | 50.9 vs 55.7; 24.2 vs 51.8 (27.6 vs 4.8) |
| Changing text | unmentioned 35.8 vs 26.1 vs 24.8 | 38.3 vs 26.8 vs 24.8 |
| Breakdown text | density "about 2 points", shuffled gap 20–23 | 2.4 points; 21–24 |
| Breakdown text | degree 99.1 vs 41.4, cycle 84.2 vs 62.8, edge 65.9 vs 53.7 | 99.9 vs 40.8, 83.4 vs 62.5, 66.3 vs 53.2 |
| Breakdown text | shortest 23.8, reach 69.0; counting 34.0 / 18.9; most-common "within a point of shuffled" | 25.5, 68.4; 34.3 / 21.3; equal to shuffled (28.6) |
| Local-wrong text | 20.9 (s.d. 0.9), shuffled 20.1, question 24.6 | 22.2 (0.3), 21.2, 24.6 |
| Write path text | 0.2 points (53.5 vs 53.7; CI −0.1 to 0.5); "within 1.3 points" | 0.1 (54.2 vs 54.3; s42 CI −0.2 to 0.4); within 1.2 points |
| Inferences | shuffled 21.8; zero-shot "25 points"; 25.2 / 51.2; 35.8 / 26.1; 41.3; 0.2; 20.9 vs 20.1 | 22.0; 26 points; 24.2 / 51.8; 38.3 / 26.8; 41.2; 0.1; 22.2 vs 21.2 |
| Appendix static | degree 97.5, cycle 79.0, reach 65.2, edge 61.7; attr lookup 29.0 (soft 25.8), most common 27.9 (26.5), check 49.9 | 99.1, 79.0, 65.7, 61.2; 29.5 (25.8), 28.5 (26.5), 49.4 |

The former `%TEA-*` placeholder comments have been replaced by those paragraphs.

### Hyperparameters (appendix table)

| Item | Value |
|---|---|
| LLM | Llama-3.1-8B-Instruct, bf16, frozen |
| Node features | `bert-base-uncased`, 768-d, frozen |
| GNN | GraphSAGE 768/1024/1024/4096, sum, LayerNorm, dropout 0 |
| Readout | pooled / source / target, hop scalar, task one-hot (12,305-d) |
| Projector | 12,305 → 10 × 4096, linear |
| TEA-GLM pretraining | Adam 2e-3, τ = 0.4, dropout 0.5, edge drop 0.3/0.4, feature mask 0/0.1, 1,000 PCs, 32 graphs/batch, ≤ 70 epochs, patience 10 |
| Training | AdamW, wd 1e-4, effective batch 24, 1 epoch, cosine, 200 warmup, min lr ratio 0.05 |
| Projector lr | 1e-4 for all four graph readers (chosen from {1e-3, 3e-4, 1e-4} on validation loss) |
| GNN lr (joint) | 1e-4 (GT, warm start) |
| Soft prompt | 10 × 4096, lr 1e-3, final checkpoint |
| Checkpoint | lowest validation loss, every 100 steps on 1,000 items |
| Evaluation | greedy, max 32 new tokens, first non-empty line scored |

---

## 9. Reproducing

#### Compute (one RTX PRO 6000 96 GB on Jarvislabs; wall time per ladder rung of the final runs)

| Rung | Minutes | Finished (UTC) |
|---|---|---|
| setup | 2 | 2026-10-30 11:56:48 |
| soft | 9 | 2026-10-30 12:06:52 |
| gt42 | 26 | 2026-10-31 04:33:21 |
| gt43 | 26 | 2026-10-31 04:59:26 |
| gt44 | 26 | 2026-10-31 05:25:47 |
| random | 25 | 2026-10-30 21:15:19 |
| tea42 | 26 | 2026-10-30 19:33:50 |
| warm | 26 | 2026-10-30 19:59:30 |
| tea43 | 25 | 2026-10-30 20:24:46 |
| tea44 | 25 | 2026-10-30 20:50:03 |
| text | 9 | 2026-10-30 16:42:34 |
| sweep | 21 | 2026-10-30 19:08:16 |


The data rungs (`setup`), soft prompt and text baselines ran on 30 October and do not depend on
the graph reader's learning rate. Every graph-reader number comes from the 1e-4 runs of 30–31
October.

**Artifacts.**

- **Per-turn outputs, logs, validation histories and sweep runs:**
  `documents/experiments/results/v4_1-20261030/runs.tar.gz`, with wall times in `timings.tsv`.
- **Checkpoints (projector and GNN weights) and the same artifacts:** the private Hugging Face
  bucket `prasoon50/graphmodi-v4_1`, under `outputs/` and `artifacts/`.
- **Configs:** `configs/v4_*.yaml` (sweep configs in `configs/sweep/`), generated by
  `scripts/make_v4_configs.py`.
- **GPU ladder:** `scripts/run_v4_ladder.sh`. Its rungs are setup, soft, gt42, tea42, warm, gt43,
  gt44, tea43, tea44, random, text, sweep.

**Regenerating every number and figure (CPU only, about 10 minutes):**

```bash
mkdir -p /tmp/v4_1 && tar xzf documents/experiments/results/v4_1-20261030/runs.tar.gz -C /tmp/v4_1
PYTHONPATH=src .venv/bin/python scripts/analyze_v4.py --runs /tmp/v4_1 \
    --out documents/experiments/results/v4_1-analysis/analysis.json
PYTHONPATH=src .venv/bin/python scripts/report_tables_v4.py --runs /tmp/v4_1 \
    --out documents/experiments/results/v4_1-analysis/report_tables.json
PYTHONPATH=src .venv/bin/python scripts/make_figures_v4.py --runs /tmp/v4_1 \
    --out documents/mid-report/figures
python3 scripts/sweep_projector_lr.py --runs /tmp/v4_1 \
    --out documents/experiments/results/v4_1-analysis/projector_lr_sweep.json
```

The GNN probe needs the GNN checkpoints (from the bucket) and a GPU:
`PYTHONPATH=src python scripts/probe_gnn_v4.py --outputs outputs --out documents/experiments/results/v4_1-analysis/gnn_probe.json`.
