# Experiment log and results

Two different places hold run artifacts:

- [`datasets/`](../../datasets/) — shared session JSONL the whole team trains
  and evaluates on. Tracked in git.
- `outputs/` — local checkpoints, smoke scratch, and full evaluation dumps.
  Gitignored; nothing there survives a fresh clone.

This folder is the durable run record: an append-only log of every run
whose result was actually used, plus tiny summary JSON so someone can check
the outcome without re-running or needing `outputs/` to still exist.

Do not start generating or training runs "just to try something" without a
line in [`log.md`](log.md). The point is that six months from now, or after
someone's laptop is wiped, `documents/experiments/log.md` plus
`documents/experiments/results/<run-id>/` (and the matching
`datasets/<id>/` for data) should be enough to know what was run.

## Before running anything

1. Confirm the config you're about to run corresponds to a locked choice —
   check [`documents/decisions/`](../decisions/) for the distribution,
   scale, and model settings you're relying on. If nothing is locked yet,
   either lock it first (write a decision record) or mark the run in the
   log as exploratory/pre-decision.
2. Make sure the config lives in [`configs/`](../../configs/) under version
   control — do not run an experiment from an untracked, ad hoc config.
3. For training/eval, prefer an existing `data_dir` under `datasets/` over
   regenerating. Only regenerate when the locked generator changes.
4. Commit or at least note the exact git commit you're running from. A run
   on uncommitted code is not reproducible; say so in the log if it
   happens anyway.

## Run ID convention

```text
<config-stem>-<YYYYMMDD>[-<short-suffix>]
```

Examples: `smoke-20260821`, `qwen8b_projector-20260821-data`. The config
stem should match the run name used in `output_dir`
(`outputs/<config-stem>/...`).

## After running

1. Append one row to [`log.md`](log.md).
2. If the run released or updated a shared dataset, put the JSONL under
   [`datasets/<id>/`](../../datasets/) and keep only `audit.json` (plus any
   tiny summaries) under `results/<run-id>/`.
3. For training/eval runs, copy only small summary artifacts into
   `documents/experiments/results/<run-id>/` — never checkpoints,
   `.pt`/`.safetensors` files, or full session JSONL here:
   - `datasets/<id>/audit.json` or `outputs/<name>/data/audit.json`
   - `outputs/<name>/evaluation.json`
   - GNN/projector `metadata.json` files only
4. If the run's result changes or confirms a decision, link it from that
   decision record's **Related** section, and link the decision back from
   the log row.

## Layout

```text
documents/experiments/
  README.md
  log.md
  results/
    <run-id>/
      audit.json
      evaluation.json
      ...
```
