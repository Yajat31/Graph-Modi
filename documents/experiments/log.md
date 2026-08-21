# Experiment log

Append-only. Add a new row for every run whose result was actually looked
at (not throwaway local debugging). See
[`README.md`](README.md) for the run-id convention and what to copy into
`results/<run-id>/`. Do not edit or delete past rows; if a run was wrong or
superseded, add a new row and say so in Notes rather than rewriting
history.

| Date | Run ID | Config | Git commit | Purpose / hypothesis | Key result | Decision / next step | Notes |
|---|---|---|---|---|---|---|---|
| 2026-08-21 | qwen8b_projector-20260821-data | configs/qwen8b_projector.yaml | uncommitted (WS lock + data_dir layout) | Generate locked Watts–Strogatz experiment-1 sessions for the team | audit valid on all splits; train 10k / val 1k / test 2k; majority ≈ 0.23–0.24; files in `datasets/watts_strogatz_metro_v1/` | [0001-graph-distribution](../decisions/0001-graph-distribution.md); next: GPU `pretrain-gnn` / `train-projector` | Summary: [results/qwen8b_projector-20260821-data/audit.json](results/qwen8b_projector-20260821-data/audit.json). Do not regenerate unless the locked generator changes. |

## Column guide

- **Config** — path under [`configs/`](../../configs/), e.g.
  `configs/smoke.yaml`.
- **Git commit** — short hash the run was executed against. If uncommitted,
  say `uncommitted` and note why in Notes.
- **Purpose / hypothesis** — one line: what question this run answers (e.g.
  "does the new Watts-Strogatz generator keep majority-answer accuracy
  below 60% at full scale?").
- **Key result** — the one or two numbers that mattered (e.g. from
  `evaluation.json`'s `summary`, or `audit.json`'s
  `majority_answer_accuracy` / `valid`). Link to
  `results/<run-id>/` for the full JSON.
- **Decision / next step** — did this confirm/break a decision record, or
  trigger a follow-up run? Link the relevant file under
  [`documents/decisions/`](../decisions/).
