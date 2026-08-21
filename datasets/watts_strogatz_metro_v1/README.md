# watts_strogatz_metro_v1

In-distribution multi-turn metro sessions for experiment 1.

| Field | Value |
|---|---|
| Distribution | `watts_strogatz_metro_v1` (`k=4`, `p_rewire=0.15`, `line_count=4`) |
| Decision | [`documents/decisions/0001-graph-distribution.md`](../../documents/decisions/0001-graph-distribution.md) |
| Generator config | [`configs/qwen8b_projector.yaml`](../../configs/qwen8b_projector.yaml) |
| Seed | `42` |
| Node count | 15–30 |
| Turns per session | cycled `[1, 2, 3, 5]` |
| Splits | train 10 000 / validation 1 000 / test 2 000 sessions |
| Audit | all splits `valid: true` (see [`audit.json`](audit.json)) |

## Files

| File | Role |
|---|---|
| `train.jsonl` | projector / GNN training sessions |
| `validation.jsonl` | mid-run checks |
| `test.jsonl` | held-out evaluation |
| `audit.json` | solver re-check, operation/reasoning/answer histograms |

Each JSONL line is one `Session` (`graph_modi.schema.Session`).

## Use it

Point the experiment config at this folder (already done for the Qwen
projector and multiturn-eval configs):

```yaml
data_dir: datasets/watts_strogatz_metro_v1
output_dir: outputs/qwen8b
```

Then train and evaluate without regenerating:

```bash
uv run graph-modi pretrain-gnn --config configs/qwen8b_projector.yaml
uv run graph-modi train-projector --config configs/qwen8b_projector.yaml
uv run graph-modi evaluate --config configs/qwen8b_projector.yaml
```

Inspect one session:

```bash
uv run graph-modi run-session \
  --config configs/qwen8b_projector.yaml \
  --session datasets/watts_strogatz_metro_v1/test.jsonl \
  --index 0
```

## Regenerate (only if the locked generator changes)

```bash
uv run graph-modi generate --config configs/qwen8b_projector.yaml
```

That writes back into `data_dir` from the config. Re-check `audit.json`
before anyone trains on the new files.
