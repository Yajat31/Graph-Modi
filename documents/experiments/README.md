# Experiment log and results

Two different places hold run artifacts:

- [`datasets/`](../../datasets/) — shared session JSONL (gitignored) plus manifests.
  Regenerate with the configs; verify sha256 against the manifest.
- `outputs/` — local checkpoints, smoke scratch, and full evaluation dumps.
  Gitignored; nothing there survives a fresh clone.

This folder is the durable run record: an append-only log of every run whose
result was used, plus tiny summary JSON so someone can check the outcome
without re-running or needing `outputs/` to still exist.

## Before running anything

1. Confirm the config matches a locked choice in [`documents/decisions/`](../decisions/),
   or mark the run exploratory in [`log.md`](log.md).
2. Keep the config under [`configs/`](../../configs/) (or [`archive/configs/`](../../archive/configs/)
   for superseded recipes).
3. Prefer an existing `data_dir` under `datasets/` over regenerating. Only
   regenerate when the locked generator changes; then update the manifest.
4. Record the exact git commit in the log.

## Current vs historical

- **metro_v4** — current benchmark. Gates: `scripts/audit_v4_gates.py`. Final
  audit: `scripts/audit_v4_final.py`. See [benchmark_v4_2026-10-25.md](benchmark_v4_2026-10-25.md)
  and [benchmark_v4_final_audit_2026-10-25.md](benchmark_v4_final_audit_2026-10-25.md).
- **metro_v3** — frozen. Status report: [status_report_v3.md](status_report_v3.md).
  Results under `results/v3-20260929/` and `results/v3x-20260929/`.

## Audit documents (date order)

| Date | Doc |
|---|---|
| 2026-10-21 | [experimental_audit_2026-10-21.md](experimental_audit_2026-10-21.md) |
| 2026-10-21 | [experimental_audit_addendum_2026-10-21.md](experimental_audit_addendum_2026-10-21.md) |
| 2026-10-21 | [benchmark_audit_2026-10-21.md](benchmark_audit_2026-10-21.md) |
| 2026-10-22 | [benchmark_audit_deep_2026-10-22.md](benchmark_audit_deep_2026-10-22.md) |
| 2026-10-25 | [benchmark_v4_2026-10-25.md](benchmark_v4_2026-10-25.md) |
| 2026-10-25 | [benchmark_v4_final_audit_2026-10-25.md](benchmark_v4_final_audit_2026-10-25.md) |
| 2026-10-25 | [v4_run_checklist.md](v4_run_checklist.md) |

## After running

1. Append one row to [`log.md`](log.md).
2. If the run released or updated a shared dataset, regenerate under
   `datasets/<id>/`, update `manifest.json`, and keep only small summaries under
   `results/<run-id>/`.
3. Never put checkpoints or full session JSONL under `documents/`.

## Layout

```text
documents/experiments/
  README.md
  log.md
  results/
    <run-id>/
      audit.json / results.json / *_summary.json
```
