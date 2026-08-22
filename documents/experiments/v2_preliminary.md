# Dynamic GLM preliminary study (branch `test`)

Implementation completed on branch `test`. Record SHA for every run.

## Smoke run (CPU mock backend)

Config: `configs/v2_smoke.yaml`

```bash
graph-modi generate --config configs/v2_smoke.yaml
graph-modi freeze-v1 --config configs/v2_smoke.yaml
graph-modi pretrain-gnn --config configs/v2_smoke.yaml
graph-modi train-projector --config configs/v2_smoke.yaml
graph-modi static-eval --config configs/v2_smoke.yaml
graph-modi evaluate --config configs/v2_smoke.yaml
graph-modi pilot-gates --config configs/v2_smoke.yaml
graph-modi final-study --config configs/v2_smoke.yaml
```

Outputs: `outputs/v2_smoke/`

## Key modules

| Component | Path |
|-----------|------|
| Progress helper | `src/graph_modi/utils/progress.py` |
| Dataset v2 | `src/graph_modi/data/v2.py` |
| v1 freeze manifest | `src/graph_modi/data/freeze_v1.py` |
| GraphToken path | `src/graph_modi/models/graph_token.py` |
| Soft-prompt baseline | `src/graph_modi/models/soft_prompt.py` |
| Static oracle gate | `src/graph_modi/evaluation/static_eval.py` |

## Full-scale config

Scale session/static counts in `configs/v2_smoke.yaml` toward plan targets (500 ID + 250 OOD sessions; ~2k static graphs) before GPU training on L40s.
