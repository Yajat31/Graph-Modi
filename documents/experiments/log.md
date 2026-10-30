# Experiment log

Add a row for every run whose result was used. See [`README.md`](README.md) for what to copy into
`results/<run-id>/`.

| Date | Run ID | Config / script | Git commit | Purpose | Key result | Notes |
|---|---|---|---|---|---|---|
| 2026-10-30 | benchmark-audit-v4_1-20261030 | `scripts/audit_v4_gates.py`, `configs/v4_data.yaml` | c4a6275 | Generate metro_v4 (generator v4.1) and run the acceptance gates | 45 pass / 0 fail / 2 warn (mention rate). Answer-changing share 0.356 val / 0.364 test; stale-answer baseline 63.6%; cycle leaf-neighbour rule 79.0%; 7-task lookup ceiling 41.0%; NOOP 0.175 / 0.167 | Data in `datasets/metro_v4_1/` (gitignored; hashes in its `manifest.json`). Note: [benchmark_v4_1_2026-10-30.md](benchmark_v4_1_2026-10-30.md) |
| 2026-10-30 | benchmark-final-audit-v4_1-20261030 | `scripts/audit_v4_final.py` | c4a6275 | Independent correctness audit (determinism, labels, edits, sessions, splits, plumbing, evaluation) | 75 pass / 0 fail; regeneration reproduces every manifest hash | |
| 2026-10-30 | v4_1-analysis (ceilings) | `scripts/analyze_v4.py --shortcut-only` | 8d8841d | Learned local ceilings and text baselines, fixed before any model result | Non-attribute tasks: one-hop 63.4% static / 61.9% sessions, two-hop 70.1% / 67.0%. Text-only 36.6%, stale + text 78.7% (test sessions). Local-wrong slice 1,052 turns; `changed_and_no_mention` 359 | `results/v4_1-analysis/shortcut_ceilings.json` |

## Column guide

- **Config / script** — path under [`configs/`](../../configs/) or `scripts/`.
- **Git commit** — short hash the run was executed against.
- **Key result** — the numbers that mattered; full JSON under `results/<run-id>/`.
