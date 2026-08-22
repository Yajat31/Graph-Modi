# Dynamic GLM Benchmark and Training Plan

Extend Graph-Modi into a paper-grounded dynamic graph-language benchmark, then compare TEA-GLM and GraphToken under the same 8B LLM, data, graph encoder, token budget, and multi-turn evaluation. The design uses CLEGR-style modality controls, GraphQA-style topology variation, DyGraphQA-style update stress tests, and a staged experiment matrix sized for two 48 GB NVIDIA L40 GPUs.

## 0. Git branch and engineering requirements

### Branch workflow
- Do all preliminary implementation on a dedicated git branch named **`test`**, created from the current working branch at execution start.
- Keep `main` / existing experiment branches unchanged until `test` passes smoke tests and static oracle-QA gates.
- Every experiment log in [`documents/experiments/`](documents/experiments/) must record the `test` branch commit SHA.

### Batching (mandatory everywhere)
- **No sequential per-example GPU loops** in new training, evaluation, or inference paths when a batched equivalent exists.
- Reuse and extend existing batched primitives in [`training.py`](src/graph_modi/training.py) (`_collate_batch`, `_batch_query_vectors`) and [`evaluation/runner.py`](src/graph_modi/evaluation/runner.py) (`answer_batch`, `predict_edit_batch`, round-based chunking).
- New code must batch:
  - GNN pretraining and projector training (graph collation + padded LLM forward where shapes differ).
  - Static oracle-QA evaluation across tasks.
  - Dynamic evaluation across sessions/conditions/turns (configurable `batch_size`, default 16).
  - Graph encoding for multi-turn sessions (batch `encode()` where graphs share structure size bins, or micro-batch by node count).
- Expose `batch_size` in YAML configs ([`config.py`](src/graph_modi/config.py), [`cli.py`](src/graph_modi/cli.py)) for every long-running command.
- Dataset generation may stay CPU-sequential but should batch solver calls where possible; GPU paths must never fall back to one-item `.generate()` loops.

### Monitorable progress prints (for coding-agent tracking)
- Every long loop (dataset generation, GNN pretrain, projector train, static QA eval, dynamic eval) must emit **structured, flush=True stdout** that a coding agent can grep and use for ETA estimation.
- Follow the existing tag convention already used in [`training.py`](src/graph_modi/training.py) and [`evaluation/runner.py`](src/graph_modi/evaluation/runner.py):

```text
[tag] phase=<name> step=<i>/<total> elapsed=<secs>s rate=<items/s> eta=<secs>s extra=<key=value ...>
```

- Required tags and minimum fields:

| Tag | When | Minimum fields |
|-----|------|----------------|
| `[generate]` | Dataset/session generation | `graphs`, `sessions`, `graph_i/total`, `elapsed`, `rate` |
| `[pretrain-gnn]` | GNN alignment + auxiliary heads | `epoch`, `batch`, `loss`, `acc`, `elapsed`, `eta` |
| `[train-projector]` | Projector / GraphToken training | `epoch`, `batch`, `loss`, `elapsed`, `eta`, `stage` |
| `[static-eval]` | Held-out oracle QA gate | `task`, `batch`, `correct/total`, `elapsed`, `eta` |
| `[evaluate]` | Dynamic multi-turn eval | `condition`, `round`, `turns_done/total`, `batch_size`, `elapsed`, `eta` |
| `[encode-batch]` | Batched graph encoding | `batch`, `nodes`, `elapsed` |

- Print a **start banner** once per phase (total work units, batch size, device, seed) and an **end banner** (wall time, throughput, summary metric).
- Log every N steps where `N = max(1, total_steps // 10)` so agents get ~10 progress updates per phase without log spam.
- All progress printing goes through a small shared helper (e.g. `graph_modi/utils/progress.py`) rather than ad-hoc formats, so tags stay consistent across TEA-GLM, GraphToken, and baselines.
- CLI commands accept `--progress` (default `True` for train/eval; `False` only in unit tests).

## 1. Lock the scientific question and reproducibility baseline
- Treat the primary claim as: **after a natural-language update, explicitly applying the edit to the graph and re-encoding the resulting state improves graph-grounded QA over retaining stale graph tokens or text-only state**.
- Freeze the existing v1 results, dataset manifest, model hashes, seeds, and batched-inference policy before changing the task. Record all runs against a git SHA in [`documents/experiments/`](documents/experiments/).
- Use one primary frozen LLM for fair comparison: **Llama-3.1-8B-Instruct loaded and executed in bf16 throughout**, with the same tokenizer and answer prompts for both architectures. Do not quantize or use 4-bit loading. Use GraphSAGE as the primary common graph encoder and **10 graph-prefix tokens**, matching CLEGR’s controlled comparison. Label implementations “TEA-style” or “GraphToken-style” wherever they depart from the papers rather than claiming exact reproduction.
- Add a static **oracle-input capability gate** before dynamic experiments: given the correct current graph and question, each GLM must reach a predeclared threshold (initial target: at least 70% overall exact answer accuracy, no core task below 50%, and at least +10 points over the stronger unimodal baseline). If this fails, improve static graph reading before interpreting dynamic-update results.

Paper basis: [TEA-GLM, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0b77d3a82b59e9d9899370b378087faf-Abstract-Conference.html), [GraphToken](https://arxiv.org/abs/2402.05862), and [CLEGR, Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1624/).

## 2. Build a bounded preliminary dataset
Extend [`src/graph_modi/data/multiturn.py`](src/graph_modi/data/multiturn.py), [`schema.py`](src/graph_modi/schema.py), [`graph/solvers.py`](src/graph_modi/graph/solvers.py), and [`data/validation.py`](src/graph_modi/data/validation.py).

- Preserve fictional metro entities and executable gold solvers from CLEGR, preventing pretrained factual recall.
- Limit the preliminary study to two metro-plausible topology families: **Watts–Strogatz** and **stochastic block model**. Stratify both by scale and density. Keep Erdős–Rényi as one optional topology-OOD test only; omit Barabási–Albert, path, star, and complete families for now. This still reflects GraphQA’s finding that topology and density affect performance without turning the pilot into a broad graph-family study.
- Stratify complexity explicitly:
  - Scale: 16–24 and 25–32 nodes in-distribution; 40–48 nodes scale-OOD.
  - Density: sparse, medium, and dense bins based on normalized edge density/average degree.
  - Dialogue length: 1, 2, and 4 turns in-distribution; **8 turns is the only turn-count OOD condition and the absolute maximum**.
  - Reasoning depth: local/1-hop, 2–3 hop, and 4+ hop or global topology.
  - Update load: **0, 1, 2, or 3 ordered edit operations before each query**.
- Make `NOOP` a first-class operation rather than an invalid parse:
  - Sample 15–20% zero-operation turns with an explicit canonical utterance and gold `NOOP`.
  - Also retain turns where valid edits are irrelevant to the current query, so the model cannot assume every turn changes the answer.
  - Represent multi-edit outputs as an ordered list of atomic `SET`, `ADD`, `DEL`, and `NOOP` commands terminated by `END`; validate and apply them sequentially.
- Use a small executable task set:
  - GraphQA capability warm-up: edge existence, node degree, reachability, and cycle membership.
  - CLEGR-style joint tasks: attribute-filtered neighbors, filtered aggregation, constrained reachability, shortest path avoiding a semantic class, cycle/topology queries with semantic filters.
  - Weighted tasks: shortest-path **cost**, cheapest valid route, and cost comparison.
  - Prefer open numeric, route/list, and entity-set outputs; yes/no tasks remain capability controls rather than dominating dynamic evaluation.
  - Dynamic sessions include answer-changing edits, irrelevant edits, and explicit NOOP turns, preventing the query from reducing to “which one of a few edits happened?”
  - Keep true link prediction separate from edge existence: mask positive edges, create type/degree-matched negatives, and evaluate candidate ranking with MRR/Hits@K. Treat it as an auxiliary capability suite, not as deterministic dynamic QA.
- Store for every turn: `G0`, gold edit program, `G_before`, `G_after`, stale and current answers, executable query program, minimal support nodes/edges, task/complexity labels, and fingerprints.
- Preliminary scale:
  - Static graph pretraining corpus: about 2,000 train, 250 validation, and 500 held-out graphs, yielding roughly 20k–30k static QA tuples.
  - Dynamic benchmark: about 500 paired in-distribution sessions at 1/2/4 turns and 250 8-turn OOD sessions. Increase only after the capability and leakage gates pass.

Paper basis: [Talk Like a Graph / GraphQA, ICLR 2024](https://openreview.net/forum?id=IuXR1CCrSi), CLEGR, and [DyGraphQA](https://openreview.net/pdf/6d08f3bac18b8ebb2b748ff6e898ae3008bf9953.pdf).

## 3. Make single-modality solutions impossible by construction
- Partition information deliberately:
  - The **structure channel** receives node IDs, topology, and numeric edge weights but not the natural-language meaning of attributes.
  - The **language channel** receives node/edge descriptions and the question/update utterance but not a serialized adjacency list in the primary soft-prompt baseline.
  - The full GLM receives both through the graph encoder plus question/context.
- Produce paired counterfactuals for every retained template:
  - Topology counterfactual: preserve all text/attributes, alter one relevant edge, and force the answer to flip.
  - Semantic counterfactual: preserve topology, alter one relevant attribute/relation meaning, and force the answer to flip.
- Balance labels against graph size, density, degree, template, lexical items, and edit type. Reject examples where the utterance/question leaks the answer or where either counterfactual does not flip it.
- Run text-only and structure-only admission tests. Report **multimodal gain** as `full GLM − max(text-only, structure-only)`; do not claim multimodal integration from raw accuracy alone.
- Keep a secondary “full graph serialized as text” baseline because CLEGR uses this harder language-only setting, but distinguish it from the information-partitioned soft-prompt control.

## 4. Keep edit language simple; vary only state complexity
- Use one canonical language template per atomic operation and no paraphrase, alias, coreference, or held-out wording experiment in the preliminary study.
- Obtain edit difficulty only from graph scale/density, 0–3 ordered operations per query, cumulative turn count, and interactions between operations.
- Include explicit `NOOP`, idempotent state updates, irrelevant-but-valid edits, and cancelling operation pairs. Score an ordered edit sequence by execution equivalence against the final graph state, while separately reporting exact command-sequence match.
- Preserve two-stage inference: the same LLM emits a constrained edit; deterministic code validates/applies it; then QA receives only `G_t + Q_t`, not the revision text. Measure syntax validity, execution-equivalent edit accuracy, exact state accuracy, and downstream answer accuracy separately.
- Keep edit generation prompt-based and bf16. Do not add an edit-specific LoRA or a second editing model in the preliminary experiment.

## 5. Pretrain strong static GLMs before testing updates
Refactor [`models/tea_glm.py`](src/graph_modi/models/tea_glm.py), add [`models/graph_token.py`](src/graph_modi/models/graph_token.py) and [`models/soft_prompt.py`](src/graph_modi/models/soft_prompt.py), and generalize [`training.py`](src/graph_modi/training.py), [`models/base.py`](src/graph_modi/models/base.py), [`config.py`](src/graph_modi/config.py), and [`cli.py`](src/graph_modi/cli.py).

- **Do not pretrain on dynamic tuples, dialogue histories, or edit sequences.** Pretraining uses independent static `(G, Q, A)` samples. Dynamic sessions are reserved for downstream inference/evaluation. The static corpus must nevertheless match the downstream graph families, scale/density range, attributes, edge-weight distribution, query scopes, and reasoning depths.
- Shared controls for both GLMs:
  - Frozen BERT-style node/edge text embeddings (768-dimensional, as in CLEGR), structural/Laplacian positional features as used by GraphToken, and the same residual GraphSAGE backbone.
  - Use task-appropriate readouts: anchor-node embeddings for node tasks, endpoint plus neighborhood embeddings for edge/link/path tasks, and attention/global pooling for graph-level tasks. Do not represent every problem with one repeated mean-pooled vector.
  - Same bf16 Llama-3.1-8B LLM, graph-token count, static training corpus, input/output templates, token budget, seeds, and validation stopping rule.
  - Static curriculum stages: atomic GraphQA tasks first, then CLEGR-style filtering plus aggregation, then constrained path and weighted-cost tasks. Balance batches by task and complexity so shortest path does not dominate.
- TEA-GLM path:
  1. Encode node/edge text with the frozen BERT encoder.
  2. Pretrain the GNN on the static graph corpus with TEA’s feature-wise contrastive alignment to PCA components of the frozen LLM token-embedding matrix.
  3. Because the downstream tasks are more complex than TEA’s original node-classification/link-prediction setting, jointly add solver-supervised auxiliary heads for the same static task families: node/edge properties, reachability, filtered aggregation, constrained distance, and path cost. This is a deliberate strong **task-pretrained TEA-style** variant; retain alignment-only TEA as a diagnostic ablation.
  4. Freeze the pretrained GNN and frozen bf16 LLM. Train only the linear projector/readout to produce 10 graph tokens by answer-token cross-entropy on the static QA curriculum.
- GraphToken path:
  1. GraphToken has no separate TEA-style alignment phase: initialize the shared BERT features, positional features, GNN, readout, and projector.
  2. Keep the bf16 LLM frozen and train the GNN plus readout/projector end-to-end through answer-token likelihood on the **same static QA curriculum**, following GraphToken’s central method.
  3. Validate each curriculum stage before adding the next; retain the best checkpoint by balanced per-task oracle-QA accuracy rather than aggregate loss alone.
- The current supervised query-head pretraining remains a separate baseline, not the primary TEA implementation.
- Add a learned 10-vector soft-prompt baseline: one shared `10 × d_model` parameter matrix trained over all training examples, with no GNN or projector.
- Before dynamic evaluation, run both GLMs on held-out static graphs across every task, scale, density, and reasoning-depth bin. If either architecture fails the oracle-input gate, tune static pretraining/readout capacity; do not compensate with dynamic training.

## 6. Run the full baseline matrix on paired sessions
Extend [`evaluation/runner.py`](src/graph_modi/evaluation/runner.py) and [`evaluation/metrics.py`](src/graph_modi/evaluation/metrics.py).

Required conditions:
- Question only and majority/random priors.
- Graph once at turn 0, then update history as text (`graph_once_then_text`).
- Stale `G0` graph tokens at every turn plus accumulated updates (`frozen_graph_history`).
- Soft-prompted LLM with semantic text/history but no topology channel.
- Structure-only GraphSAGE/GAT control with semantic attributes removed.
- Full graph serialized as text, for a CLEGR-style language-only control.
- Modify-and-Print text baseline from DyGraphQA: explicitly materialize a textual graph state after every edit.
- Oracle `G_t + Q_t` with gold edits and re-encoding.
- GraphModi: predicted edit, deterministic apply, re-encode `G_t`, then answer `Q_t`.
- Symbolic solver ceiling.

Run every condition with both TEA-GLM and GraphToken checkpoints where applicable, on exactly the same session IDs.

## 7. Evaluation, gates, and reporting
- Core dynamic metrics: exact answer accuracy, stale-answer rate, per-turn and whole-dialogue success, edit syntax validity, execution-equivalent edit accuracy, exact graph-state accuracy, counterfactual flip consistency, and latency/memory.
- Task metrics: set F1 for node lists, exact/MAE for costs, and filtered MRR/Hits@1/3/10 for masked-edge link prediction.
- Break down all results by task, graph family, size, density, hop depth, operation count, operation type, NOOP/irrelevant/answer-changing status, and turn index; provide paired bootstrap confidence intervals. Use one seed for the preliminary matrix and three seeds only for the final retained configurations if compute/time permits.
- Predeclare interpretation gates:
  1. Oracle-current-graph QA must pass the static capability threshold.
  2. Both modality counterfactual tests must pass and unimodal baselines must remain substantially below the full model.
  3. GraphModi must beat stale-graph and graph-once baselines on paired sessions.
  4. Report oracle–GraphModi gap as the cost of edit prediction; do not attribute that gap to graph reasoning.
- Add tests in [`tests/test_multiturn.py`](tests/test_multiturn.py), [`test_distribution.py`](tests/test_distribution.py), and [`test_evaluation.py`](tests/test_evaluation.py) for solver correctness, split leakage, counterfactual guarantees, task balance, state transitions, and baseline channel isolation.

## 8. Two-L40 execution plan
- **Step 0:** `git checkout -b test` (or `git switch -c test`) before any code changes.
- Keep the complete model path at 8B/bf16, including loading, forward passes, generation, and checkpoint reload. One 48 GB L40 can hold the frozen 8B model plus GNN/projector; run TEA and GraphToken independently, one per GPU, rather than introducing DDP complexity early.
- Phase A: generate the bounded static corpus and run smoke tests on every task/readout. Watch `[generate]` progress lines for throughput and ETA.
- Phase B: pretrain TEA on GPU 0 and GraphToken on GPU 1 in parallel using approximately 20k–30k **static** QA tuples. Validate oracle QA by task and complexity after every curriculum stage.
- Phase C: screen ablations with one seed; retain only architecture/training choices that improve validation multimodal gain and oracle QA.
- Phase D: evaluate the paired 1/2/4-turn sessions and the 8-turn OOD set. Rerun three seeds only for the final retained configurations if the preliminary result is promising.
- Avoid 14B+ models, G-Retriever, quantized loading, LoRA, and broad encoder sweeps in the preliminary experiment.
- Log effective tokens, samples/sec, peak VRAM, wall-clock time, checkpoint/data hashes, and exact CUDA/PyTorch versions for every run.
