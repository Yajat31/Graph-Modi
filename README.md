# GraphModi

GraphModi is a controlled experiment for one question: does updating and
re-encoding a graph help an instruction-tuned language model answer multi-turn
questions more reliably than keeping the initial graph and a short conversation
history?

The first experiment intentionally uses in-distribution graphs and short
histories. It does not make the text baseline fail by overflowing its context
window, and it does not hide graph facts from that baseline. The aim is to test
whether the updated graph representation itself is useful.

## Method

Each session starts with an attributed metro graph `G_0`. A turn contains a
natural-language revision, a typed edit, and a question whose answer changes
because of that edit:

```text
revision -> parse edit -> validate/apply -> G_t -> encode -> graph tokens -> Qwen -> answer
```

Edits are cumulative: turn three applies to `G_2`, never to the original graph.
The symbolic solver verifies every answer before an example is written.
Questions cover shortest paths, reachability, filtered neighbor/path counts,
cycle membership, and edge relationships.

The model follows the core [TEA-GLM](https://github.com/W-rudder/TEA-GLM)
design: a pretrained GraphSAGE encoder and the language model are frozen while a
projector maps pooled graph representations into a fixed number of language
model embedding tokens. The data interface follows the graph and compositional
reasoning principles of
[CLEGR](https://github.com/rethinking-graph-language-evals/CLEGR). GraphModi
adds cumulative graph mutations and multi-turn controls; it does not vendor or
claim to reproduce either project.

## Evaluation conditions

Every condition uses the same sessions and decoder:

- `question_only`: shortcut check without graph information.
- `frozen_graph_history`: initial graph tokens plus all short revisions.
- `shuffled_graph`: graph tokens from another session.
- `oracle_updated_graph`: gold edit, re-encoded current graph, no history.
- `cached_no_reencode`: apply the edit but retain old graph tokens.
- `predicted_updated_graph`: model-produced edit followed by re-encoding.
- `serialized_initial_history`: complete initial graph text plus revisions.
- `token_matched_history`: a fixed-budget history control.
- `serialized_current_graph`: current graph text, which diagnoses projector
  failures separately from language-model reasoning failures.
- `tool_solver`: typed edit plus external solver, reported as a practical
  ceiling rather than as graph-token reasoning.

Metrics include answer and final-turn accuracy, execution-equivalent edit
accuracy, exact graph-state accuracy, stale-answer rate, latency, and input
token counts.

## Installation

Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/Yajat31/Graph-Modi.git
cd Graph-Modi
./scripts/setup.sh
```

The default setup installs only generation, evaluation, and development
dependencies. Install PyTorch, Transformers, and Accelerate for real training:

```bash
./scripts/setup.sh --tea
```

PyTorch/CUDA selection is deliberately not pinned to one machine. On a GPU
server, install the wheel recommended for the server's CUDA driver before
running `uv sync --extra tea`, if necessary.

## Quick CPU smoke test

The smoke configuration is download-free and uses a symbolic backend:

```bash
make smoke
make test
make lint
```

Its generated files are written under `outputs/smoke/`. The mock checkpoint is
only an integration test and must not be reported as a learned projector.

## Generate data

```bash
uv run graph-modi generate --config configs/qwen8b_projector.yaml
```

This creates graph-instance, entity-name, seed, session, and
paraphrase-family-disjoint JSONL files. Train, validation, and test graphs come
from the same generator distribution for the initial feasibility experiment.
The generated audit rejects non-cumulative sessions, invalid edits, leaked
answers, incorrect oracle labels, and unchanged-answer examples.

To adapt official CLEGR files, use `graph_modi.data.clegr.load_clegr_pt`.
It accepts common `x` and `edge_index` records and an optional JSON ID mapper.

## Train

First pretrain or load the graph encoder, then train only the projector:

```bash
uv run graph-modi pretrain-gnn --config configs/qwen8b_projector.yaml
uv run graph-modi train-projector --config configs/qwen8b_projector.yaml
```

The server template runs training and evaluation:

```bash
sbatch scripts/slurm_projector.sh configs/qwen8b_projector.yaml
```

`configs/qwen8b_projector.yaml` uses `Qwen/Qwen3-8B`, a 4-layer GraphSAGE
encoder, and eight projected graph tokens. Change these values without changing
code. Training uses full answer-sequence loss through `inputs_embeds`,
Accelerate, gradient accumulation, mixed precision, deterministic seeds, and
checkpoint/resume. The language model and graph encoder remain frozen during
projector training.

Checkpoints include model names, tokenizer identity, tensor dimensions, data
split fingerprints, and training configuration. Incompatible external TEA-GLM
checkpoints fail with a shape/backbone error instead of loading partially.

## Evaluate or inspect one session

```bash
uv run graph-modi evaluate --config configs/qwen8b_projector.yaml
uv run graph-modi run-session \
  --config configs/qwen8b_projector.yaml \
  --session outputs/qwen8b/data/test.jsonl \
  --index 0
```

The evaluation summary and per-turn rows are JSON. Retain per-turn rows: a high
aggregate score can otherwise conceal stale predictions or an update bug.

## Go/no-go interpretation

Proceed to structural out-of-distribution experiments only when:

1. the symbolic oracle and exact graph-state checks are 100%;
2. the pretrained GNN solves the in-distribution graph task substantially above
   its majority baseline;
3. the trained projector clearly improves over an untrained or shuffled
   projector;
4. `oracle_updated_graph` improves over `cached_no_reencode` with a low stale
   rate; and
5. the comparison with `serialized_initial_history` is based on paired examples
   and confidence intervals, not a few curated conversations.

If serialized current graphs work but graph tokens fail, the bottleneck is the
projector. If short history performs as well as updated graph tokens, GraphModi
has not established an accuracy advantage; later experiments may still test
state compression or much longer sessions, but that is a different claim.

## Resource guidance

Generation, solver checks, and the mock smoke path run on CPU. Projector
training with an 8B frozen decoder generally needs a GPU server. A 24 GB GPU may
work with batch size 1, bf16, gradient accumulation, and memory-efficient model
loading; 40–48 GB provides safer headroom. CPU-only 8B training is technically
possible but not a practical iteration path. Start with the smoke config and a
small language model before scheduling the full run.

## Repository layout

```text
src/graph_modi/
  data/          synthetic sessions, validation, CLEGR adapters
  graph/         edit parser, immutable executor, solvers, serialization
  models/        mock and TEA-GLM-compatible backends
  pipeline/      cumulative edit/apply/re-encode loop
  evaluation/    paired conditions and metrics
configs/         smoke, Qwen projector, and evaluation configurations
tests/           unit and download-free integration tests
scripts/         setup and SLURM launchers
```

## Current limitations

- The synthetic generator is a controlled feasibility benchmark, not natural
  dialogue.
- Initial results are in-distribution by design.
- Tool calling with an external solver changes the task and is therefore
  reported separately.
- Official TEA-GLM checkpoints are loadable only when backbone metadata and
  tensor shapes match this implementation.
- Free-form edit prediction needs model-specific prompting and should always be
  scored by execution equivalence as well as exact text.
