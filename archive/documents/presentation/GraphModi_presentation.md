# GraphModi: Updating Graph Language Models in Multi-turn Dialogue

Team Hansraj (Prasoon Dev, Yajat Lakhanpal, Hrishiraj Mitra, Arihant Rastogi), IIIT Hyderabad.
Short presentation for evaluation. Source: `ANLP.pdf` (mid-project report) plus the repo's later results
(`documents/experiments/status_report_v3.md`). Slides: `GraphModi_presentation.pptx` (same content, one section per slide).

---

## Motivation: every graph model keeps the graph static

Prior work answers questions about a graph that is handed over once and never changes.

- **Graphs as text.** Talk-like-a-graph (Fatemi et al., 2024) and Wang et al. (2023) serialise a fixed graph into the prompt. Accuracy depends on the encoding and is brittle to spurious correlations, but the graph itself never changes.
- **Soft graph tokens.** GraphToken, TEA-GLM, G-Retriever and Graph Language Models pass a GNN encoding of one graph to a frozen or lightly tuned LLM. Each query gets a graph supplied once, and the tokens are computed once.
- **Evaluations.** CLEGR and soft-prompt studies ask whether the model really uses the graph, yet still on a fixed graph. Dynamic-graph QA (DyGraphQA) and knowledge-editing work (MQuAKE) describe changes in text or in model weights, not in graph tokens.

No benchmark exists for this setting: none defines how a natural-language revision should change the graph a GLM encodes, or scores answers turn by turn as the graph evolves. **Our main contribution is that multi-turn benchmark**, with GraphModi (an explicit edit-and-re-encode loop) as the first system evaluated on it.

---

## 1. Problem

Graph language models (GLMs: GNN -> soft tokens -> frozen LLM) assume the graph is fixed. In dialogue, users revise the world
("station X is closed now"). Tokens computed before the revision are **stale** and give the old answer.

The system must (a) **write**: turn the revision into a change to the graph, and (b) **read**: answer from tokens recomputed after
the change, without the LLM ever seeing the revision text.

## 2. Research questions and contributions

- **RQ1** Can a frozen LLM emit executable edit programs from a few-shot prompt, and what do its errors cost?
- **RQ2** Does answering from re-encoded tokens beat stale G0 tokens and text-history controls as turns accumulate?
- **RQ3** Do graph tokens carry information beyond graph-blind baselines?

Contributions: (i) **a multi-turn benchmark for graph language models** (the main contribution; see section 4); (ii) a formulation with an edit language + executor that separates write accuracy from read accuracy;
(iii) paired conditions across the grid (size x density x session length x task) that isolate stale tokens, text history and graph-blind controls; (iv) GraphModi as a first system on the benchmark, and evidence on when stale tokens hurt.

## 3. Method (pipeline)

```
revision r_t -> few-shot edit prompt (frozen Llama) -> edit program  SET NODE / ADD EDGE / DEL EDGE / NOOP ; ... END
G_{t-1} -> executor -> G_t -> BERT-768 node features -> GraphSAGE (3 layers) -> task-conditional readout
        -> linear projector -> 10 graph tokens -> frozen Llama-3.1-8B-Instruct + question -> answer
```

- **Write path** (`models/tea_glm.py`, `graph/edits.py`, `graph/executor.py`): greedy decoding, at most 3 ops; invalid ops are skipped, so an edit error persists into later turns.
- **Read path**: GraphSAGE 768/1024/1024, mean aggregation. If the question names a source/target station the readout uses those nodes, else the mean pool. Projector input = pooled + source + target + hop scalar + task one-hot (3087-d) -> 10 x 4096. As in CLEGR, the queried node text W(f_i) is added to the question.
- **Training**: no multi-turn training. Static QA + before/after counterfactual pairs; 5 epochs, AdamW lr 1e-3, batch 24, seed 42. Llama stays frozen.

## 4. Our benchmark: multi-turn (v3, `data/v3.py`, seed 42)

- Synthetic metro graphs: 12-22 stations, mean degree 2.8 / 3.8 / 5.0, diameter capped 7 / 6 / 5, CLEGR-style attributes.
- Station `status` is in the graph but withheld from W(f_i), so it cannot be read off the question.
- Yes/no answers balanced 50/50 per stratum; leak audit clean.
- Static QA: 11 tasks (val 2,178 / test 4,356). Multi-turn grid: 11 sizes x lengths {1,2,4,8} x 3 densities x 9 tasks = **1,188 sessions, 4,455 turns per split**.
- Test: **41% of turns change the answer**, 37% are NOOP.

## 5. Models and conditions

Three models share the frozen Llama-3.1-8B decoder. The **TEA-style** model uses a GraphSAGE encoder with contrastive alignment and trains a linear projector. The **GraphToken-style** model starts from the same GNN and trains the projector. The **soft prompt** has no GNN: it learns 10 soft tokens and reads the revision history as text.

Paired conditions on identical sessions: `oracle_updated_graph` (re-encode gold G_t), `predicted_updated_graph` (own edits),
`structure_only` / `frozen_graph_history` (stale G0), `graph_once_then_text`, `shuffled_graph`, `question_only`,
`token_matched_history`, `soft_prompt`, plus references: stale-answer (59.3%), per-task majority (29.4%), symbolic `tool_solver` (100%).

## 6. Results

**Static QA, test (%)**: TEA 36.3, GraphToken-style 40.3, soft prompt 32.8, majority 28.9. Graph-dependent tasks show the gap
(reachability 68.9 vs 50.0; attribute lookup 54.5 vs 27.3). Count tasks stay under 20% for every model.

**Multi-turn, TEA-style, test accuracy.** The full GraphModi pipeline, with its own predicted edits, scores 34.9%; re-encoding the gold graph every turn (oracle) scores 35.1%. The soft prompt, which reads the revision history, scores 34.2%. Stale tokens score 32.5% with history and 32.3% without. A shuffled graph scores 30.2%, graph-once-then-text 29.8%, token-matched history 29.0% and question only 27.7%.

- **Tokens encode the specific graph**: correct vs shuffled gap +4.9 test / +4.8 val; most-common-attribute 33.1 vs 19.4 shuffled vs 0.0 no graph.
- **Staleness hurts where the edit matters**: on answer-changing turns GraphModi scores 34.2 and the oracle graph 34.6, against 29.7 for stale tokens (about +4.5 to +4.9); on unchanged turns they agree within 1 point. Re-encoded accuracy is flat between changed and unchanged turns.
- **Session length (test)**: stale tokens + history fall 35.4 -> 31.6 (1 to 8 turns); GraphModi stays about 34.5-37 (36.0, 36.9, 34.5, 34.5 at lengths 1, 2, 4, 8). This trend does not replicate on validation.
- **Write path (RQ1)**: model-written edits give the same graph as gold on 88.6% of turns (95.2% execution-equivalent); accuracy with predicted edits 34.9 (TEA) / 37.6 (GraphToken-style) vs 35.0 / 37.9 with oracle edits, so edit errors cost almost nothing here.

## 7. Scope of the evidence

1. **One seed.** An identical recipe varies by about 3-4 points between runs, so small gaps between models or conditions are not interpretable.
2. **TEA vs soft prompt is not significant**: +0.9 points, CI [-0.8, +2.6]. The soft prompt reads the revision text and matches or beats re-encoding on reachability, cycle membership and edge existence; re-encoding wins on attribute and structure tasks. We do not claim graph tokens beat a text channel.
3. **Templated revisions**, so edit accuracy likely overstates free-form performance.

## 8. Takeaways

- Stale graph tokens are measurably worse exactly on turns where an edit changes the answer; re-encoding closes that gap and the write path is not the bottleneck.
- The read path is the bottleneck (about 35% absolute; counts below 20%).
- The tokens encode the specific graph: the correct graph beats a shuffled one by about 5 points.
- We do not claim that graph tokens beat a text channel that sees the history (+0.9 points, not significant), or any model-vs-model ranking from a single seed.

## Appendix: reproduce and layout

```bash
uv run graph-modi generate --config configs/v3_tea.yaml
uv run graph-modi pretrain-gnn --config configs/v3_tea.yaml
uv run graph-modi train-projector --config configs/v3_tea.yaml
uv run graph-modi static-eval --config configs/v3_tea.yaml --split test
uv run graph-modi evaluate --config configs/v3_exact2x_tea.yaml
```

`src/graph_modi/`: `data/` (v3 benchmark), `graph/` (edit parser, executor, solvers), `models/` (tea_glm, graph_token, soft_prompt, bert_tensorizer),
`evaluation/` (runner, static_eval, metrics), `pipeline/` (edit/apply/re-encode loop), `training.py`, `cli.py`. Tests: `make test`; CPU smoke: `make smoke`.
