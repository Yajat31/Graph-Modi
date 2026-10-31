# Asset links

## Code (GitHub)

- Repository: <https://github.com/Yajat31/Graph-Modi>
- **Final code and results: default branch `graphmodi-experiments`**:
  <https://github.com/Yajat31/Graph-Modi/tree/graphmodi-experiments>
- The same commits are on the working branch `v4_1-launch-fixes`:
  <https://github.com/Yajat31/Graph-Modi/tree/v4_1-launch-fixes>

The branch holds the code, configs, analysis, figures, the final report
(`documents/mid-report/main.tex`) and the results guide (`documents/report-guide.md`).

```bash
git clone -b graphmodi-experiments git@github.com:Yajat31/Graph-Modi.git
```

## Checkpoints and run artifacts (Hugging Face bucket)

- Bucket: <https://huggingface.co/buckets/prasoon50/graphmodi-v4_1> (private; ask Prasoon for
  access)
- 431 files, 33.6 GB

| Path in the bucket | Contents |
|---|---|
| `outputs/v4_graphtoken_s{42,43,44}/projector/checkpoint-{best,final}/` | GraphToken-style: `projector.pt` (2.0 GB) and the jointly trained `gnn.pt` (48 MB) |
| `outputs/v4_tea_s{42,43,44}/projector/checkpoint-{best,final}/` | TEA-style: `projector.pt` and the frozen TEA-GLM GNN `gnn.pt` |
| `outputs/v4_graphtoken_warm_s42/projector/checkpoint-{best,final}/` | Warm start: `projector.pt` and `gnn.pt` |
| `outputs/v4_tea_random_sum/projector/checkpoint-{best,final}/` | Frozen random GNN: `projector.pt` and `gnn.pt` |
| `outputs/v4_pretrain_sum/gnn/gnn.pt` | TEA-GLM-pretrained GNN (used by TEA-style and the warm start) |
| `outputs/v4_soft_prompt/soft_prompt/checkpoint-final/soft_prompt.pt` | Soft prompt |
| `outputs/<run>/projector/val_history.json`, `training_result.json` | Validation-loss curves and training summaries |
| `outputs/sweep/<run>_lr{3e-4,1e-4}/` | Projector learning-rate sweep (validation histories only; weights not kept) |
| `artifacts/` | Per-turn evaluation outputs, static evaluations, logs, timings, lr sweep and GNN probe results |
| `reruns/text_baselines_len4096/` | Zero-shot text baselines rerun with a 4,096-token budget. These supersede `artifacts/v4_text_baselines/` and `outputs/v4_text_baselines/`, whose prompts were truncated to 502 tokens |

`documents/experiments/results/v4_1-20261030/runs.tar.gz` holds `artifacts/` with the rerun text
baselines in place of the truncated ones.

The report evaluates `checkpoint-best`, the checkpoint with the lowest validation loss. Every graph
reader was trained with projector lr 1e-4. The base LLM (`meta-llama/Llama-3.1-8B-Instruct`) and
`bert-base-uncased` are not in the bucket; download them from their Hugging Face model pages.

```bash
hf auth login   # an account with access to the bucket
# everything:
hf buckets sync hf://buckets/prasoon50/graphmodi-v4_1/outputs outputs
# one run, best checkpoint only:
hf buckets sync hf://buckets/prasoon50/graphmodi-v4_1/outputs/v4_graphtoken_s42 \
    outputs/v4_graphtoken_s42 --include "projector/checkpoint-best/*"
```

The configs in `configs/v4_*.yaml` expect the checkpoints under `outputs/<run>/` in the
repository root.
