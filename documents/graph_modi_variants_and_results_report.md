# Graph-Modi variants, dataset examples, and bracketed results

**Repository state reviewed:** `dynamic` branch at `00c0a49`  
**Report date:** 2026-08-24  
**Primary result source:** [`experiments/results/v2_gate_variant_cf-20260823/report.md`](experiments/results/v2_gate_variant_cf-20260823/report.md)

## 1. Executive summary

Graph-Modi tests whether a graph-language model should explicitly update and re-encode its graph after a natural-language correction. Its operational loop is:

```text
natural-language revision
        ↓
predict SET / ADD / DEL / NOOP program
        ↓
validate and apply edits to the current graph
        ↓
re-encode the updated graph
        ↓
answer from updated graph + question
```

The strongest current result is the **v2 task-balanced, counterfactual-trained, multi-operation variant**. On 3,165 turns pooled across validation, test, and OOD:

| System | TEA-style | GraphToken-style |
|---|---:|---:|
| Question only | 46.8% | 46.8% |
| Strongest non-updated control | 51.5% | 51.3% |
| Oracle edit + re-encode | **70.1%** | **71.2%** |
| Predicted edit + re-encode (GraphModi) | **69.0%** | **69.4%** |
| Edit-program accuracy | **96.4%** | **96.4%** |

This confirms the update mechanism for the narrowed `edge_exists`, `reachability`, and `cycle_membership` task suite. It does **not** yet establish the claim for the full numeric/path task mix, which failed the static capability gate.

## 2. What the LLM actually receives

The primary TEA-style and GraphToken-style paths do not serialize the graph into the prompt. The LLM receives:

```text
[tokenized question] + [10 learned graph-prefix embeddings]
```

The prefix embeddings are produced from the current graph by a GraphSAGE encoder and projector. The examples in Section 6 show the literal text prompt and a complete human-readable graph serialization. That serialization documents what the prefix represents; it is not falsely presented as literal prompt text.

Edit prediction is a separate text-only call. It receives a few-shot instruction prompt plus the revision and emits an ordered program such as:

```text
SET NODE Elm status closed ; ADD EDGE Elm Cedar transfer ; END
```

## 3. Model and training variants

### 3.1 Symbolic mock/oracle

- **How it works:** Parses canonical revisions deterministically and answers with the executable solver.
- **Purpose:** Unit tests, smoke runs, and an upper-bound sanity check.
- **Result:** 100% when given the correct current graph. This is not a learned-model result.

### 3.2 v1 TEA-style Llama-3.1-8B

- **Graph:** Watts–Strogatz metro, 15–30 nodes, degree 4.
- **Features:** Deterministic hashed node IDs, labels, and attributes.
- **Encoder:** Eight-layer, sum-aggregation GraphSAGE, hidden size 4,096.
- **Readout:** Concatenated global mean, source-node, and target-node representations.
- **Projector:** One linear layer producing eight graph-prefix tokens.
- **LLM:** Frozen Llama-3.1-8B-Instruct in bf16.
- **Training:** Supervised solver-answer classification for the GNN, followed by answer-token cross-entropy for the projector.

Reference four-turn result, 1,400 turns per condition:

| Condition | Accuracy |
|---|---:|
| Question only | 17.2% |
| Graph once then text | 17.5% |
| Shuffled graph | 28.4% |
| Frozen graph + history | 31.0% |
| Cached/no re-encode | 31.6% |
| Predicted update + re-encode | **44.4%** |
| Oracle update + re-encode | **44.7%** |
| Tool solver | 100.0% |

Predicted and oracle updating were statistically indistinguishable: −0.4 percentage points, 95% CI [−3.1, +2.4].

### 3.3 v2 full-task TEA-style

- **Tasks intended:** edge existence, node degree, reachability, cycle membership, filtered counts, shortest path, and path cost.
- **Static corpus:** Independent `(graph, question, answer)` tuples.
- **Dynamic corpus:** Watts–Strogatz/SBM ID graphs and Erdős–Rényi/large-graph OOD cases, 0–3 edits per turn.
- **Result:** Failed the static oracle-QA gate. `shortest_path`, `path_cost`, `filtered_neighbor_count`, and `filtered_path_count` were below the 50% per-task floor.
- **Interpretation:** Dynamic-update conclusions from this full task mix are not accepted because the model could not reliably read even an oracle current graph.

### 3.4 v2 full-task GraphToken-style

- **Intended difference:** Jointly train the GNN and projector through answer-token likelihood rather than freezing a separately pretrained GNN.
- **Result:** Also failed the full-task static capability gate.
- **Implementation caveat:** In the current code, the GraphToken wrapper eventually enters `train_projector()`, which freezes the GNN. Therefore this is a labeled **GraphToken-style** variant, not a faithful end-to-end GraphToken reproduction.

### 3.5 Task-balance-only TEA and GraphToken variants

- **Change:** Restrict static and dynamic tasks to `edge_exists`, `reachability`, and `cycle_membership`.
- **Static gate:** TEA 91.9%; GraphToken 90.6%.
- **Dynamic result:** Oracle updated graph remained around 46%, close to question-only around 44%. Edit prediction was approximately 46–48%.
- **Diagnosis:** High IID static accuracy did not teach the projector to distinguish before/after states of the same entities.

### 3.6 Counterfactual-training variants

- **Change:** Replace independent static training examples with paired graph states: same entities and question, one relevant graph edit, answer forced to change.
- **Additional edit fix:** Add an explicit NOOP demonstration to the edit-generation prompt.
- **Result before multi-op parsing:** Oracle/GraphModi rose to roughly 69–70%, question-only stayed at 45–47%, and edit prediction rose to about 79%.
- **Conclusion:** Counterfactual state pairs were the main improvement in graph-state sensitivity.

### 3.7 Counterfactual + multi-operation variants (current best)

- **Change:** Predict all edits separated by `;` and terminated by `END`, instead of only parsing the first operation.
- **Static gate:** TEA 77.4%; GraphToken 77.1%.
- **Dynamic result:** TEA GraphModi 69.0%; GraphToken GraphModi 69.4%.
- **Edit accuracy:** 96.4% for both.
- **Oracle gap:** 1.1 points for TEA and 1.8 points for GraphToken.
- **Scope:** 3,165 turns per condition; three binary/near-binary tasks only.

### 3.8 Learned soft-prompt variant

- **How it works:** A shared learned `10 × d_model` prefix with no GNN and no topology-dependent representation.
- **Purpose:** Test whether a fixed learned prompt plus text/history can replace the graph channel.
- **Result status:** Implemented and smoke-tested, but no durable full-scale real-model result is checked into the repository. Mock-backend scores must not be treated as evidence for this variant.

### 3.9 Configuration/run inventory

| Configuration family | Role | Durable result status |
|---|---|---|
| `smoke.yaml`, `v2_smoke.yaml` | Download-free mock end-to-end checks | Pass; engineering validation only |
| `qwen8b_projector.yaml` | Qwen3-8B v1 training proposal | Dataset generated/audited; no checked-in real neural result |
| `llama3_1_8b_projector.yaml` | Main v1 Llama training | Completed; reference v1 checkpoint/result family |
| `llama3_1_8b_eval_subset.yaml` | Variable-turn v1 evaluation subset | Completed |
| `llama3_1_8b_4step_eval.yaml` | Fixed four-turn accumulation test | Completed; strongest clean v1 comparison |
| `v2_full_tea.yaml` | Full v2 task mix, TEA-style | Ran; static gate failed |
| `v2_full_graphtoken.yaml` | Full v2 task mix, GraphToken-style | Ran; static gate failed |
| `v2_full_soft_prompt.yaml` | Learned graph-independent prefix | Implemented; no durable full-scale result |
| `v2_gate_variant_*.yaml` | Three-task diagnostic without counterfactual pairs | Static gate passed; dynamic state tracking weak |
| `v2_gate_variant_cf_*.yaml` | Counterfactual TEA/GraphToken comparison | Completed; current best result |

## 4. Evaluation-condition variants

The following table describes every condition currently supported by the evaluator. Accuracy is the turn-weighted result from the current best v2 run, pooled over validation, test, and OOD.

| Condition | Information supplied | TEA | GraphToken |
|---|---|---:|---:|
| `question_only` | Question, no graph state | 46.8% | 46.8% |
| `majority_prior` | Uses the stored stale answer as a privileged baseline | 37.6% | 37.6% |
| `shuffled_graph` | Graph tokens from the wrong session | 49.0% | 50.5% |
| `structure_only` | Intended topology-only control | 49.3% | 48.1% |
| `serialized_current_graph` | Current graph rendered as text | 40.3% | 39.2% |
| `serialized_initial_history` | Initial graph text plus accumulated revisions | 47.2% | 47.1% |
| `token_matched_history` | Truncated history-only token-budget control | 51.5% | 51.3% |
| `graph_once_then_text` | Graph at first turn, text-only later | 48.6% | 48.9% |
| `frozen_graph_history` | Initial graph tokens plus revision history | 49.2% | 42.5% |
| `cached_no_reencode` | State updated internally, but graph tokens stay stale | 50.3% | 39.6% |
| `modify_and_print` | Textual graph state materialized after updates | 50.6% | 40.4% |
| `oracle_updated_graph` | Gold edit, apply, re-encode, answer | **70.1%** | **71.2%** |
| `predicted_updated_graph` | LLM edit, apply, re-encode, answer | **69.0%** | **69.4%** |
| `tool_solver` | Deterministic symbolic answer | 100.0% | 100.0% |
| `soft_prompt` | Shared learned prefix plus text/history | No full result | No full result |

Some baseline names overstate current isolation. `structure_only` still uses tensorized labels/attributes, and `modify_and_print` may also receive graph-prefix tokens in the TEA backend. These should be corrected before using them for a strict modality claim.

### Visual comparison of the principal conditions

Each block is approximately two accuracy points.

```text
TEA-style
question only          46.8% |███████████████████████
best non-updated       51.5% |██████████████████████████
GraphModi              69.0% |██████████████████████████████████
oracle re-encode       70.1% |███████████████████████████████████

GraphToken-style
question only          46.8% |███████████████████████
best non-updated       51.3% |██████████████████████████
GraphModi              69.4% |███████████████████████████████████
oracle re-encode       71.2% |████████████████████████████████████
```

## 5. Results by dataset bracket

All tables below come from the final counterfactual + multi-operation experiment. `GT` means GraphToken-style. Counts are turns, not sessions.

### 5.1 Graph-size bracket

| Size bracket | Definition | n | Question only TEA/GT | Oracle TEA/GT | GraphModi TEA/GT |
|---|---|---:|---:|---:|---:|
| Small | 16–24 nodes | 582 | 45.5 / 48.5 | **84.9 / 84.5** | **82.6 / 83.2** |
| Medium | 25–32 nodes | 583 | 45.1 / 50.3 | **65.5 / 67.6** | **62.1 / 64.0** |
| Scale OOD | 40–48 nodes | 2,000 | 47.6 / 45.3 | **67.2 / 68.3** | **67.0 / 67.0** |

```text
GraphModi accuracy by graph size
small      TEA 82.6% |█████████████████████████████████████████
           GT  83.2% |██████████████████████████████████████████
medium     TEA 62.1% |███████████████████████████████
           GT  64.0% |████████████████████████████████
scale OOD  TEA 67.0% |██████████████████████████████████
           GT  67.0% |██████████████████████████████████
```

Small graphs are substantially easier. Medium graphs unexpectedly score below scale-OOD; the OOD bracket is confounded with topology and eight-turn sessions, so this is not a clean size-only effect.

### 5.2 Density bracket

| Density | n | Question only TEA/GT | Oracle TEA/GT | GraphModi TEA/GT |
|---|---:|---:|---:|---:|
| Sparse | 0 | — | — | — |
| Medium | 2,397 | 47.0 / 46.5 | **67.4 / 68.6** | **66.8 / 67.2** |
| Dense | 768 | 46.0 / 47.5 | **78.6 / 79.2** | **75.9 / 76.4** |

```text
GraphModi accuracy by density
medium  TEA 66.8% |█████████████████████████████████
        GT  67.2% |██████████████████████████████████
dense   TEA 75.9% |██████████████████████████████████████
        GT  76.4% |██████████████████████████████████████
```

Dense graphs score higher, but cycle-membership questions are over-represented in that bracket. No sparse examples were produced, revealing a generator/threshold gap.

### 5.3 Session-length bracket

| Turns/session | n | Question only TEA/GT | Oracle TEA/GT | GraphModi TEA/GT |
|---:|---:|---:|---:|---:|
| 1 | 167 | 45.5 / 49.1 | **80.8 / 85.0** | **77.8 / 84.4** |
| 2 | 334 | 43.4 / 51.8 | **73.1 / 75.7** | **70.7 / 69.5** |
| 4 | 664 | 46.2 / 48.2 | **74.8 / 73.9** | **71.8 / 72.9** |
| 8, OOD | 2,000 | 47.6 / 45.3 | **67.2 / 68.3** | **67.0 / 67.0** |

```text
GraphModi accuracy by session length
1 turn   TEA 77.8% |███████████████████████████████████████
         GT  84.4% |██████████████████████████████████████████
2 turns  TEA 70.7% |███████████████████████████████████
         GT  69.5% |███████████████████████████████████
4 turns  TEA 71.8% |████████████████████████████████████
         GT  72.9% |████████████████████████████████████
8 turns  TEA 67.0% |██████████████████████████████████
         GT  67.0% |██████████████████████████████████
```

Longer sessions are harder, but every eight-turn session is also scale-OOD; length and graph size cannot be separated in this dataset.

### 5.4 Reasoning-task bracket

| Task | n | Stale-answer baseline | Question only TEA/GT | Oracle TEA/GT | GraphModi TEA/GT |
|---|---:|---:|---:|---:|---:|
| Edge existence | 1,549 | 25.5% | 49.8 / 49.3 | **63.9 / 64.4** | **61.7 / 61.4** |
| Reachability | 1,348 | 40.2% | 44.5 / 45.8 | **74.3 / 75.2** | **73.4 / 75.0** |
| Cycle membership | 268 | 94.0% | 40.3 / 36.6 | 85.4 / 90.3 | 88.8 / 87.7 |

```text
GraphModi accuracy by reasoning task
edge exists   TEA 61.7% |███████████████████████████████
              GT  61.4% |███████████████████████████████
reachability  TEA 73.4% |█████████████████████████████████████
              GT  75.0% |██████████████████████████████████████
cycle member  TEA 88.8% |████████████████████████████████████████████
              GT  87.7% |████████████████████████████████████████████
```

Cycle membership has a 94% stale-answer baseline, so its high raw accuracy is not strong evidence. Edge existence and reachability provide the clearest state-update signal.

### 5.5 Edit-count bracket

| Operations in turn | n | Edit accuracy TEA/GT | Final-answer accuracy TEA/GT |
|---:|---:|---:|---:|
| 1, including NOOP | 1,830 | approximately 100 / 100 | 68.5 / 69.9 |
| 2 | 662 | **94.7 / 95.2** | 68.9 / 68.3 |
| 3 | 673 | **88.7 / 88.0** | 70.3 / 69.1 |

```text
Edit-program accuracy
1 op   TEA ~100% |██████████████████████████████████████████████████
       GT  ~100% |██████████████████████████████████████████████████
2 ops  TEA 94.7% |███████████████████████████████████████████████
       GT  95.2% |████████████████████████████████████████████████
3 ops  TEA 88.7% |████████████████████████████████████████████
       GT  88.0% |████████████████████████████████████████████
```

Edit parsing degrades with program length, but final QA accuracy remains approximately stable because many wrong edits are irrelevant to the query.

### 5.6 Turn-index bracket

| Turn index | TEA oracle | TEA GraphModi | GT oracle | GT GraphModi |
|---:|---:|---:|---:|---:|
| 0 | 76.7 | 75.7 | 79.2 | 76.0 |
| 1 | 63.3 | 60.0 | 64.5 | 58.5 |
| 2 | 75.7 | 76.2 | 78.1 | 75.0 |
| 3 | 72.1 | 70.9 | 70.9 | 72.1 |
| 4 | 62.8 | 57.6 | 57.2 | 63.6 |
| 5 | 73.6 | 72.8 | 74.0 | 74.8 |
| 6 | 67.2 | 68.0 | 69.6 | 71.2 |
| 7 | 60.8 | 62.8 | 64.4 | 60.0 |

Performance oscillates rather than declining smoothly. This likely reflects task/edit cycling in generation, not pure memory fatigue.

## 6. Five complete dataset examples

These examples are deterministically regenerated from `generate_dataset_v2(counts={"test": 9}, seed=42)`. They are representative v2 test items, not hand-written examples. The displayed graph is the **post-edit current graph** used to create graph-prefix embeddings for QA.

The compact node notation is `node=label(accessible,line,status)`. Examples 4 and 5 illustrate tasks from the broader default v2 generator; filtered-neighbor counting was excluded from the narrowed three-task run that produced the best final result.

### Example 1: explicit NOOP + shortest path

- **Source:** `test-v2-000000`, turn 0
- **Bracket:** Watts–Strogatz; 21 nodes; 42 edges; small; dense
- **Revision:** `NOOP: no graph update this turn.`
- **Gold edit program:** `NOOP explicit ; END`
- **Literal QA prompt:** `Question: How many stations, including endpoints, are on the shortest open path from TestStation000000_14 to TestStation000000_19? Answer with a number or unreachable.\nAnswer:`
- **Gold answer:** **2**
- **Stale/pre-edit answer:** **2**

<details><summary>Complete current graph</summary>

```text
Nodes:
n0=TestStation000000_00(accessible=false,line=yellow,status=open); n1=TestStation000000_01(true,blue,open); n2=TestStation000000_02(false,red,open); n3=TestStation000000_03(false,yellow,open); n4=TestStation000000_04(true,red,open); n5=TestStation000000_05(false,yellow,open); n6=TestStation000000_06(false,green,open); n7=TestStation000000_07(true,green,open); n8=TestStation000000_08(false,red,open); n9=TestStation000000_09(false,yellow,open); n10=TestStation000000_10(true,green,open); n11=TestStation000000_11(false,blue,open); n12=TestStation000000_12(true,green,open); n13=TestStation000000_13(true,red,open); n14=TestStation000000_14(true,blue,open); n15=TestStation000000_15(true,yellow,open); n16=TestStation000000_16(false,blue,open); n17=TestStation000000_17(true,green,open); n18=TestStation000000_18(true,red,open); n19=TestStation000000_19(true,blue,open); n20=TestStation000000_20(true,red,open)

Edges (all weight 1):
n4--n8(track); n4--n20(track); n4--n15(transfer); n4--n0(track); n8--n20(track); n8--n14(transfer); n8--n0(track); n20--n13(track); n20--n7(transfer); n13--n18(track); n13--n2(track); n13--n1(transfer); n18--n2(track); n18--n19(track); n2--n19(track); n2--n0(transfer); n19--n16(track); n19--n14(track); n16--n14(track); n16--n1(track); n14--n1(track); n14--n3(transfer); n1--n11(track); n11--n10(track); n11--n7(track); n10--n7(track); n10--n5(transfer); n7--n12(track); n7--n17(track); n12--n6(track); n12--n0(transfer); n17--n6(track); n17--n3(track); n6--n3(track); n6--n15(track); n3--n15(track); n3--n5(track); n15--n5(track); n15--n9(track); n5--n9(track); n5--n0(track); n9--n0(track)
```

</details>

### Example 2: edge addition changes shortest path

- **Source:** `test-v2-000002`, turn 1
- **Bracket:** Watts–Strogatz; 21 nodes; 43 edges; small; dense
- **Revision:** `ADD EDGE TestStation000002_00 TestStation000002_08 transfer`
- **Gold edit program:** `ADD EDGE n0 n8 transfer ; END`
- **Literal QA prompt:** `Question: How many stations, including endpoints, are on the shortest open path from TestStation000002_07 to TestStation000002_08? Answer with a number or unreachable.\nAnswer:`
- **Gold answer:** **3**
- **Stale/pre-edit answer:** **4**

<details><summary>Complete current graph</summary>

```text
Nodes:
n0=TestStation000002_00(true,yellow,open); n1=TestStation000002_01(false,green,open); n2=TestStation000002_02(true,green,open); n3=TestStation000002_03(false,red,open); n4=TestStation000002_04(false,green,open); n5=TestStation000002_05(true,green,open); n6=TestStation000002_06(false,yellow,open); n7=TestStation000002_07(true,yellow,open); n8=TestStation000002_08(true,blue,open); n9=TestStation000002_09(false,red,open); n10=TestStation000002_10(true,blue,open); n11=TestStation000002_11(false,red,open); n12=TestStation000002_12(false,red,open); n13=TestStation000002_13(true,yellow,open); n14=TestStation000002_14(false,green,open); n15=TestStation000002_15(true,red,open); n16=TestStation000002_16(true,blue,open); n17=TestStation000002_17(false,blue,open); n18=TestStation000002_18(true,red,open); n19=TestStation000002_19(false,blue,open); n20=TestStation000002_20(false,yellow,open)

Edges (all weight 1):
n11--n3(track); n11--n18(track); n11--n7(track); n11--n6(track); n3--n18(track); n3--n15(track); n3--n6(track); n18--n15(track); n18--n12(track); n15--n12(track); n15--n9(track); n12--n9(track); n12--n17(transfer); n9--n19(track); n9--n10(track); n19--n17(track); n19--n6(transfer); n10--n17(track); n10--n16(track); n17--n8(track); n17--n0(transfer); n16--n8(track); n16--n1(track); n8--n1(track); n8--n4(track); n1--n4(track); n1--n14(track); n4--n14(track); n4--n2(track); n14--n2(track); n14--n5(track); n2--n5(track); n2--n20(track); n5--n20(track); n5--n0(track); n20--n0(track); n20--n13(track); n0--n13(track); n0--n7(track); n13--n7(track); n13--n6(track); n7--n6(track); n0--n8(transfer)
```

</details>

### Example 3: explicit NOOP + reachability

- **Source:** `test-v2-000008`, turn 1
- **Bracket:** Watts–Strogatz; 23 nodes; 47 edges; small; dense
- **Revision:** `NOOP: no graph update this turn.`
- **Gold edit program:** `NOOP explicit ; END`
- **Literal QA prompt:** `Question: Can TestStation000008_18 be reached from TestStation000008_16 using open stations? Answer yes or no.\nAnswer:`
- **Gold answer:** **yes**
- **Stale/pre-edit answer:** **yes**

<details><summary>Complete current graph</summary>

```text
Nodes:
n0=TestStation000008_00(true,green,open); n1=TestStation000008_01(false,green,open); n2=TestStation000008_02(true,blue,open); n3=TestStation000008_03(false,yellow,open); n4=TestStation000008_04(false,yellow,open); n5=TestStation000008_05(false,green,open); n6=TestStation000008_06(false,blue,open); n7=TestStation000008_07(true,green,open); n8=TestStation000008_08(false,blue,open); n9=TestStation000008_09(false,red,open); n10=TestStation000008_10(false,red,open); n11=TestStation000008_11(false,blue,open); n12=TestStation000008_12(false,red,open); n13=TestStation000008_13(true,red,open); n14=TestStation000008_14(true,yellow,open); n15=TestStation000008_15(false,blue,open); n16=TestStation000008_16(false,red,open); n17=TestStation000008_17(true,green,open); n18=TestStation000008_18(true,blue,open); n19=TestStation000008_19(true,green,open); n20=TestStation000008_20(false,red,open); n21=TestStation000008_21(true,yellow,open); n22=TestStation000008_22(false,yellow,open)

Edges (all weight 1):
n20--n13(track); n20--n10(track); n20--n0(transfer); n20--n4(track); n20--n21(track); n13--n12(track); n13--n11(transfer); n13--n21(track); n10--n12(track); n10--n16(track); n12--n16(track); n12--n9(track); n16--n9(track); n16--n6(track); n9--n6(track); n9--n2(track); n6--n8(track); n6--n5(transfer); n2--n8(track); n2--n19(transfer); n8--n11(track); n8--n15(track); n11--n15(track); n11--n18(track); n15--n18(track); n15--n19(track); n18--n19(track); n18--n1(track); n19--n1(track); n19--n17(track); n1--n17(track); n1--n0(track); n1--n14(transfer); n17--n0(track); n17--n5(track); n0--n5(track); n5--n7(track); n5--n14(track); n7--n14(track); n7--n22(track); n14--n22(track); n14--n3(transfer); n22--n3(track); n22--n4(track); n3--n4(track); n4--n21(track); n4--n16(transfer)
```

</details>

### Example 4: two-operation filtered-neighbor query

- **Source:** `test-v2-000008`, turn 2
- **Bracket:** Watts–Strogatz; 23 nodes; 46 edges; small; dense
- **Revision:** `TestStation000008_17 is closed now. ; DEL EDGE TestStation000008_15 TestStation000008_19 track`
- **Gold edit program:** `SET NODE n17 status closed ; DEL EDGE n15 n19 track ; END`
- **Literal QA prompt:** `Question: How many stations directly connected to TestStation000008_15 have status equal to open? Answer with a number.\nAnswer:`
- **Gold answer:** **3**
- **Stale/pre-edit answer:** **4**

<details><summary>Complete current graph</summary>

```text
Nodes:
n0=TestStation000008_00(true,green,open); n1=TestStation000008_01(false,green,open); n2=TestStation000008_02(true,blue,open); n3=TestStation000008_03(false,yellow,open); n4=TestStation000008_04(false,yellow,open); n5=TestStation000008_05(false,green,open); n6=TestStation000008_06(false,blue,open); n7=TestStation000008_07(true,green,open); n8=TestStation000008_08(false,blue,open); n9=TestStation000008_09(false,red,open); n10=TestStation000008_10(false,red,open); n11=TestStation000008_11(false,blue,open); n12=TestStation000008_12(false,red,open); n13=TestStation000008_13(true,red,open); n14=TestStation000008_14(true,yellow,open); n15=TestStation000008_15(false,blue,open); n16=TestStation000008_16(false,red,open); n17=TestStation000008_17(true,green,closed); n18=TestStation000008_18(true,blue,open); n19=TestStation000008_19(true,green,open); n20=TestStation000008_20(false,red,open); n21=TestStation000008_21(true,yellow,open); n22=TestStation000008_22(false,yellow,open)

Edges are the Example 3 edge set with n15--n19(track) removed. The complete resulting edge set is:
n20--n13(track); n20--n10(track); n20--n0(transfer); n20--n4(track); n20--n21(track); n13--n12(track); n13--n11(transfer); n13--n21(track); n10--n12(track); n10--n16(track); n12--n16(track); n12--n9(track); n16--n9(track); n16--n6(track); n9--n6(track); n9--n2(track); n6--n8(track); n6--n5(transfer); n2--n8(track); n2--n19(transfer); n8--n11(track); n8--n15(track); n11--n15(track); n11--n18(track); n15--n18(track); n18--n19(track); n18--n1(track); n19--n1(track); n19--n17(track); n1--n17(track); n1--n0(track); n1--n14(transfer); n17--n0(track); n17--n5(track); n0--n5(track); n5--n7(track); n5--n14(track); n7--n14(track); n7--n22(track); n14--n22(track); n14--n3(transfer); n22--n3(track); n22--n4(track); n3--n4(track); n4--n21(track); n4--n16(transfer)
```

</details>

### Example 5: three-operation edge-existence query

- **Source:** `test-v2-000008`, turn 3
- **Bracket:** Watts–Strogatz; 23 nodes; 45 edges; small; dense
- **Revision:** `TestStation000008_15 is closed now. ; DEL EDGE TestStation000008_08 TestStation000008_15 track ; TestStation000008_06 is closed now.`
- **Gold edit program:** `SET NODE n15 status closed ; DEL EDGE n8 n15 track ; SET NODE n6 status closed ; END`
- **Literal QA prompt:** `Question: Is there a direct edge between TestStation000008_08 and TestStation000008_15? Answer yes or no.\nAnswer:`
- **Gold answer:** **no**
- **Stale/pre-edit answer:** **yes**

<details><summary>Complete current graph</summary>

```text
Nodes:
n0=TestStation000008_00(true,green,open); n1=TestStation000008_01(false,green,open); n2=TestStation000008_02(true,blue,open); n3=TestStation000008_03(false,yellow,open); n4=TestStation000008_04(false,yellow,open); n5=TestStation000008_05(false,green,open); n6=TestStation000008_06(false,blue,closed); n7=TestStation000008_07(true,green,open); n8=TestStation000008_08(false,blue,open); n9=TestStation000008_09(false,red,open); n10=TestStation000008_10(false,red,open); n11=TestStation000008_11(false,blue,open); n12=TestStation000008_12(false,red,open); n13=TestStation000008_13(true,red,open); n14=TestStation000008_14(true,yellow,open); n15=TestStation000008_15(false,blue,closed); n16=TestStation000008_16(false,red,open); n17=TestStation000008_17(true,green,closed); n18=TestStation000008_18(true,blue,open); n19=TestStation000008_19(true,green,open); n20=TestStation000008_20(false,red,open); n21=TestStation000008_21(true,yellow,open); n22=TestStation000008_22(false,yellow,open)

Edges are the Example 4 edge set with n8--n15(track) removed. The complete resulting edge set is:
n20--n13(track); n20--n10(track); n20--n0(transfer); n20--n4(track); n20--n21(track); n13--n12(track); n13--n11(transfer); n13--n21(track); n10--n12(track); n10--n16(track); n12--n16(track); n12--n9(track); n16--n9(track); n16--n6(track); n9--n6(track); n9--n2(track); n6--n8(track); n6--n5(transfer); n2--n8(track); n2--n19(transfer); n8--n11(track); n11--n15(track); n11--n18(track); n15--n18(track); n18--n19(track); n18--n1(track); n19--n1(track); n19--n17(track); n1--n17(track); n1--n0(track); n1--n14(transfer); n17--n0(track); n17--n5(track); n0--n5(track); n5--n7(track); n5--n14(track); n7--n14(track); n7--n22(track); n14--n22(track); n14--n3(transfer); n22--n3(track); n22--n4(track); n3--n4(track); n4--n21(track); n4--n16(transfer)
```

</details>

## 7. Interpretation and limitations

1. **The update loop works in the narrowed setting.** GraphModi improves by roughly 18–29 points over stale/text controls and stays within 1–2 points of the oracle edit path.
2. **Counterfactual training matters more than architectural labeling.** TEA-style and GraphToken-style results are very close; teaching before/after state discrimination caused the large gain.
3. **The full benchmark remains unsolved.** Numeric, filtered-count, and path-cost tasks do not yet clear the static reading gate.
4. **Size and length brackets are confounded.** All eight-turn sessions are also 40–48-node OOD sessions.
5. **The task distribution is uneven.** Cycle membership has a 94% stale-answer baseline, and no sparse-density examples appear.
6. **Several baselines are not fully isolated.** Strict multimodal claims should wait until structure-only and Modify-and-Print channels are corrected.
7. **Architecture names are qualified.** Current node features are hashed rather than frozen BERT embeddings, TEA pretraining is supervised rather than contrastive PCA alignment, and GraphToken training is not fully end-to-end.

## 8. Reproducibility references

- Benchmark and intended experiment plan: [`../README.md`](../README.md)
- Original proposal: [`interim_proposal.pdf`](interim_proposal.pdf)
- v1 distribution decision: [`decisions/0001-graph-distribution.md`](decisions/0001-graph-distribution.md)
- Append-only experiment history: [`experiments/log.md`](experiments/log.md)
- v1 scaled Llama result: [`experiments/results/llama3_1_8b_projector-20260822-batched_scale/prelim.md`](experiments/results/llama3_1_8b_projector-20260822-batched_scale/prelim.md)
- Current best v2 result: [`experiments/results/v2_gate_variant_cf-20260823/report.md`](experiments/results/v2_gate_variant_cf-20260823/report.md)
- Current best run logs: [`experiments/gate_variant_cf_run_logs/`](experiments/gate_variant_cf_run_logs/)
- v1 released dataset: [`../datasets/watts_strogatz_metro_v1/`](../datasets/watts_strogatz_metro_v1/)
