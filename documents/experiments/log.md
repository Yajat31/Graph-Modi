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
| 2026-08-21 | llama3_1_8b_projector-20260821-eval_subset | configs/llama3_1_8b_projector.yaml, configs/llama3_1_8b_eval_subset.yaml | uncommitted (projector arch fix, GNN query-conditioning/sum-aggregation, GPU + dtype fixes for evaluate) | First real (non-mock) pretrain-gnn/train-projector/evaluate run on Llama-3.1-8B-Instruct; compute-constrained preliminary check of whether re-encoding an updated graph beats a stale graph + text history | GNN pretrain 52.5% train acc (majority ≈ 33.7%); projector final loss 0.57; on a 150-session subset, paired bootstrap shows `oracle_updated_graph` beats `frozen_graph_history` (+12–16pp), `cached_no_reencode` (+17–22pp), `serialized_initial_history` (+20–22pp), and `shuffled_graph` (+12–16pp), all 95% CI excludes 0, on both validation and test splits | Hypothesis supported at subset scale; next: commit code fixes, decide on fixing self-directed edit prediction (`predicted_updated_graph` currently 0% execution-equivalent), full 3k-session run only needed for reportable effect-size, not to answer yes/no | Full write-up: [results/llama3_1_8b_projector-20260821-eval_subset/prelim.md](results/llama3_1_8b_projector-20260821-eval_subset/prelim.md) |
| 2026-08-22 | llama3_1_8b_projector-20260822-edit_fix | configs/llama3_1_8b_eval_subset.yaml, configs/llama3_1_8b_4step_eval.yaml | uncommitted (predict_edit prompt/routing/truncation fix; new `graph_once_then_text` condition + `ModelInput.turn_index`) | Same checkpoint as 2026-08-21, no retraining. Fix self-directed edit prediction (was 0% execution-equivalent) and add a fixed-4-turn dataset to test whether stale-graph baselines degrade specifically as edits accumulate | `predicted_updated_graph` edit_exec_eq 0.000→1.000 on every split tested (410+ turns); paired bootstrap shows it's statistically indistinguishable from `oracle_updated_graph` (largest sample n=600, diff −0.2pp) while still significantly beating all stale baselines; on a new 150-session/exactly-4-turns set, `graph_once_then_text` (graph shown once, then pure text) ≈ `question_only` (not significant, diff +2.5pp) — a single graph exposure doesn't persist; turn-by-turn breakdown shows `frozen_graph_history`/`cached_no_reencode` collapse to ~0.21 by turn 3 (≈ question-only floor) while `oracle`/`predicted_updated_graph` hold ~0.48 | Realistic (non-oracle) framework confirmed working and beats every stale baseline, including a repeated-history control; core hypothesis now supported by 3 independent runs across 2 days; next: commit code, sanity-check the turn-1 accuracy spike shared across conditions before reporting it | Full write-up: [results/llama3_1_8b_projector-20260822-edit_fix/prelim.md](results/llama3_1_8b_projector-20260822-edit_fix/prelim.md) |

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
