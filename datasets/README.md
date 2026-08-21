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

| Dataset ID | Config | Status |
|---|---|---|
| [`watts_strogatz_metro_v1`](watts_strogatz_metro_v1/) | [`configs/qwen8b_projector.yaml`](../configs/qwen8b_projector.yaml) | experiment-1 sessions |

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
