# Experiment log

Add a row for every run whose result was used. See [`README.md`](README.md) for what to copy into
`results/<run-id>/`.

| Date | Run ID | Config / script | Git commit | Purpose | Key result | Notes |
|---|---|---|---|---|---|---|
| 2026-10-30 | benchmark-audit-v4_1-20261030 | `scripts/audit_v4_gates.py`, `configs/v4_data.yaml` | c4a6275 | Generate metro_v4 (generator v4.1) and run the acceptance gates | 45 pass / 0 fail / 2 warn (mention rate). Answer-changing share 0.356 val / 0.364 test; stale-answer baseline 63.6%; cycle leaf-neighbour rule 79.0%; 7-task lookup ceiling 41.0%; NOOP 0.175 / 0.167 | Data in `datasets/metro_v4_1/` (gitignored; hashes in its `manifest.json`). Note: [benchmark_v4_1_2026-10-30.md](benchmark_v4_1_2026-10-30.md) |
| 2026-10-30 | benchmark-final-audit-v4_1-20261030 | `scripts/audit_v4_final.py` | c4a6275 | Independent correctness audit (determinism, labels, edits, sessions, splits, plumbing, evaluation) | 75 pass / 0 fail; regeneration reproduces every manifest hash | |
| 2026-10-30 | v4_1-analysis (ceilings) | `scripts/analyze_v4.py --shortcut-only` | 8d8841d | Learned local ceilings and text baselines, fixed before any model result | Non-attribute tasks: one-hop 63.4% static / 61.9% sessions, two-hop 70.1% / 67.0%. Text-only 36.6%, stale + text 78.7% (test sessions). Local-wrong slice 1,052 turns; `changed_and_no_mention` 359 | `results/v4_1-analysis/shortcut_ceilings.json` |
| 2026-10-30 | v4_1-20261030 (GPU ladder) | `scripts/run_v4_ladder.sh`; `configs/v4_{soft_prompt,graphtoken_s4[234],pretrain_sum,tea_s4[234],graphtoken_warm_s42,tea_random_sum,text_baselines}.yaml` | 20385b5 (graph runs), a28fa32 (soft prompt), eaa064b (gt44 onwards; checkpoint pruning only) | Final runs on one RTX PRO 6000: GraphToken and TEA-style (TEA-GLM-pretrained GNN, frozen) x 3 seeds, warm start, frozen random GNN, soft prompt, zero-shot serialized-graph baselines | Static test: GraphToken 49.7 (3 seeds), TEA-style 32.0, warm start 35.5, random GNN 39.6, soft prompt 32.6. Test sessions, updated graph (oracle / predicted edits): GraphToken 53.6 / 53.5 (sd 0.8), TEA-style 33.2 / 33.1, warm 36.7, random 44.3; shuffled graph 32.4 (GraphToken); soft prompt 34.7; zero-shot serialized graph 28.4; stale answer 63.6 | Result JSONs with per-turn rows: `results/v4_1-20261030/runs.tar.gz`; analysis: `results/v4_1-analysis/analysis.json`. Checkpoints in the private HF bucket `prasoon50/graphmodi-v4_1` |
| 2026-10-30 | v4_1-analysis (GNN probe) | `scripts/probe_gnn_v4.py` | — | Why TEA-style is weak: linear probes on frozen node embeddings | Status (open/closed) is decodable from BERT input (100%), the random GNN (96.6) and GraphToken's GNN (96.4) but not from the TEA-GLM-pretrained GNN (88.1 = majority) or the warm start (88.1); within-2-hop count 79.2 vs 92-96. Not collapsed (within-graph cosine 0.24, effective rank 16) | `results/v4_1-analysis/gnn_probe.json` |

## Column guide

- **Config / script** — path under [`configs/`](../../configs/) or `scripts/`.
- **Git commit** — short hash the run was executed against.
- **Key result** — the numbers that mattered; full JSON under `results/<run-id>/`.
