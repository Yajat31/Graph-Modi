# Shared datasets

Versioned session JSONL that the team trains and evaluates on. Unlike
`outputs/` (gitignored, local checkpoints and scratch), everything under
`datasets/` is meant to be checked into git and reused across machines.

```text
datasets/
  README.md
  <dataset-id>/
    README.md          provenance: config, seed, decision, regenerate command
    audit.json         validity and label distributions
    train.jsonl
    validation.jsonl
    test.jsonl
```

## Current datasets

No dataset files are committed. The v3 benchmark is generated deterministically (seed 42) into
`datasets/metro_v3/` by its training config; the audit of the committed runs is in
[`documents/experiments/results/v3-20260929/dataset_audit.json`](../documents/experiments/results/v3-20260929/dataset_audit.json).

| Dataset ID | Config | Status |
|---|---|---|
| `metro_v3` | [`configs/v3_tea.yaml`](../configs/v3_tea.yaml) | v3 static + multi-turn benchmark (`graph-modi generate --config configs/v3_tea.yaml`) |
| `metro_v3_big` | [`configs/v3x_data.yaml`](../configs/v3x_data.yaml) | 3x training data for the `v3x_data` variant |

## Layout rules

- **`datasets/`** — released session files teammates load. Tracked in git.
- **`outputs/`** — local GNN/projector checkpoints, evaluation dumps, smoke
  scratch. Gitignored. Never put the team dataset only here.
- Configs that train or evaluate against a shared dataset set both:
  - `data_dir: datasets/<dataset-id>`
  - `output_dir: outputs/<run-name>` for checkpoints and eval JSON

Regenerate a dataset only when the generator or locked parameters change,
then bump or replace the dataset README and re-audit. Do not silently
overwrite a shared folder mid-experiment.
