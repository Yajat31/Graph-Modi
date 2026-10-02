# Experiment log

Append-only. Add a new row for every run whose result was actually looked
at (not throwaway local debugging). See
[`README.md`](README.md) for the run-id convention and what to copy into
`results/<run-id>/`. Do not edit or delete past rows; if a run was wrong or
superseded, add a new row and say so in Notes rather than rewriting
history.

**Pruned 2026-10-02 (branch `prune-v3`).** Only the v3 runs are kept: BERT-768 node features and
yes/no labels balanced 50/50. Rows and auto-appended entries for the earlier runs (v1
Watts-Strogatz/Llama prelims, v2 gate variants, factorial/exact2x, CLEGR-extended, fixed_final) and
their result folders were removed with them; they remain in git history on the `exact2x` branch.

| Date | Run ID | Config | Git commit | Purpose / hypothesis | Key result | Decision / next step | Notes |
|---|---|---|---|---|---|---|---|
| 2026-09-29 | v3-20260929 | configs/v3_{tea,graphtoken,soft_prompt}.yaml (micro-batch variants v3_tea_b2, v3_graphtoken_b1/b2); configs/v3_exact2x_{tea,graphtoken,soft_prompt}.yaml | e883535 (GraphToken exact2x JSON in dc25a5f) | v3 benchmark: 12-22 stations, BERT-768 node features, labels balanced 50/50 per yes/no task, contrastive TEA pretraining, task-conditional readout, greedy decoding. Static QA plus the exact2x multi-turn grid (1,188 sessions / 4,455 turns per split) | Static test: GraphToken 40.3, TEA 36.3, soft prompt 32.8, question-only prior 32.8. Exact2x test: oracle_updated_graph TEA 35.1 / GraphToken 37.9; frozen_graph_history 32.5 / 33.3; structure_only 32.3 / 33.1; shuffled_graph 30.2 / 30.7; question_only 27.7; soft prompt 34.2 | Write-up: [status_report_v3.md](status_report_v3.md). Report figures: `figures/report/{tea,graphtoken}/` (`scripts/make_report_figures.py`) | Single seed. GraphToken's GNN is never trained (`train_projector` re-freezes it), so GraphToken is a second TEA run and TEA-GraphToken gaps are run-to-run variance. Reachability mostly tracks the endpoint's hidden status. The b1/b2 variants share output dirs with the main configs |
| 2026-09-29 | v3x-20260929 | configs/v3x_{base3,deep,proj,proj2,data}{,_exact2x}.yaml | e883535 | Does a deeper GNN, a larger projector or more data help TEA? base3 = 3-epoch checkpoint of v3_tea; deep = 6-layer GNN; proj = MLP 3x4096; proj2 = MLP 2x2048, lr 2e-4; data = 3x training data, 1 epoch | Static test: base3 37.3, deep 33.4, proj 29.1, proj2 33.2, data 34.1. Exact2x test oracle_updated_graph: 36.6 / 33.4 / 31.4 / 33.6 / 33.8 | None beats base3. `v3x_base3_predfix` not yet run | Single seed, constant lr, no checkpoint selection; proj and data collapse to constant yes/no answers, so this shows optimisation instability rather than a capacity or data effect |
| 2026-10-02 | v3-20260929 (predfix) | configs/v3_exact2x_{tea,graphtoken}_predfix.yaml | dc25a5f, fd20726 | Model-written edit programs (parsed, applied, re-encoded) after fixing the parser for multi-word station names | Edits execution-equivalent 95.2%, exact graph state 88.6% (test). predicted vs oracle, test: TEA 34.9 vs 35.0, GraphToken 37.6 vs 37.9 | RQ1 answered for templated revisions | The pre-fix runs (`v3_exact2x_*_predicted`, invalid parser bug) were removed in pruning. Edge-edit utterances are the edit command itself, so edit accuracy overstates free-form performance |

## Column guide

- **Config** — path under [`configs/`](../../configs/), e.g.
  `configs/v3_tea.yaml`.
- **Git commit** — short hash the run was executed against. If uncommitted,
  say `uncommitted` and note why in Notes.
- **Purpose / hypothesis** — one line: what question this run answers.
- **Key result** — the one or two numbers that mattered (e.g. from
  `evaluation.json`'s `summary` or a `*_analysis.json`). Link to
  `results/<run-id>/` for the full JSON.
- **Decision / next step** — did this confirm/break a decision record, or
  trigger a follow-up run? Link the relevant file under
  [`documents/decisions/`](../decisions/).
