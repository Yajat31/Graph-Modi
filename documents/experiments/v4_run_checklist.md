# metro_v4 (generator v4.1) GPU run checklist

Frozen plan for the paper runs: one Blackwell-class GPU, about 12–15 GPU-hours. Nothing is trained on
the audit machine. Benchmark note: [benchmark_v4_1_2026-10-30.md](benchmark_v4_1_2026-10-30.md).

Framing: this is a **benchmark paper**. The reader (GraphSAGE + linear prefix projector into frozen
Llama-3.1-8B) is frozen as-is; its degree-heuristic ceiling is a reported finding, not something to
tune away during these runs.

## Before launch

1. Commit everything and push; on the server check out that commit. Every result JSON records
   `git_sha` and `git_dirty`, so do not edit tracked files on the server.
2. Data: copy `datasets/metro_v4_1/*.jsonl` to the server, or let the script regenerate them (CPU,
   deterministic, ~10 min). The script refuses to run on any sha256 mismatch against
   `datasets/metro_v4_1/manifest.json`.
3. Llama: pass `LM_PATH=/path/to/Meta-Llama-3.1-8B-Instruct`. If the configs' `model.lm_name`
   (`/data/lmw/models/...`) does not exist the script symlinks it; configs stay untouched.
4. Launch inside `tmux`:
   ```bash
   LM_PATH=/path/to/Meta-Llama-3.1-8B-Instruct ART=/persistent/graphmodi_v4_1 scripts/run_v4_ladder.sh
   ```
   `setup` runs `uv sync --extra dev --extra tea`, `nvidia-smi`, the hash check and the full pytest.
   It fails if any test is skipped (the 6 trainer/ablation tests need `accelerate` and only run on
   the GPU box). Finished rungs leave `<rung>.done` in `$ART`, so a restart resumes.

## Ladder (budget order)

| Rung | Config(s) | What it answers | Rough time |
|---|---|---|---|
| `setup` | — | env, data hashes, tests | 15 min |
| `soft` | `v4_soft_prompt` | graph-blind trained control (with and without history) | 1 h |
| `gt42` | `v4_graphtoken_s42` | main reader, sum aggregation, working audit recipe | 1.5 h |
| `warm` | `v4_pretrain_sum` → `v4_graphtoken_warm_s42` | does contrastive pretraining help the joint reader? | 3 h |
| `gt43`, `gt44` | `v4_graphtoken_s{43,44}` | 3 seeds for the main reader | 3 h |
| `random` | `v4_tea_random_sum` | frozen random GNN, projector only: how much does GNN training add? | 1.5 h |

Real wall times go to `$ART/timings.tsv`. If time runs short, drop `random` first, then `gt44`.

Every graph rung runs `static-eval` on validation and test for **both** `checkpoint-best` (lowest
validation loss) and `checkpoint-final`, then multi-turn `evaluate` on test sessions with
`checkpoint-best`. Soft prompt has no validation pass, so it uses `checkpoint-final`.

Recipe (all graph runs, from the audit run that trained, `archive/configs/audit/gt_sum.yaml`):
sum aggregation, GNN dropout 0, projector lr 1e-3, joint GNN lr 1e-4, cosine schedule with 200
warmup steps, validation every 500 steps on 1000 examples, chat-template-wrapped train prompts
(`wrap_prompts: true`), encoder autocast matched to training. GraphToken and random-frozen use the
same projector lr, so the comparison changes only whether the GNN trains.

## Go / no-go after `gt42`

Check `outputs/v4_graphtoken_s42/projector/val_history.json`, `static_eval_test_best.json` and
`evaluation_best.json`:

- **Stop and debug** if validation loss never improves on its first value, static test accuracy is
  at or below the soft prompt, or `oracle_updated_graph` is within noise of `shuffled_graph` in the
  session evaluation (the prefix is being ignored).
- **Continue** otherwise, including when the reader sits near the shortcut ceiling: that is the
  expected result, and the seed and control rungs are what make it reportable.

## Required reporting

- **3 seeds** (42, 43, 44) for GraphToken: mean and spread; single-seed rungs are labelled as such.
- Best and final checkpoint both reported; the headline uses best (selected on validation only).
- Session-clustered bootstrap CIs; paired, session-clustered differences between models
  (`scripts/analyze_v4.py`).
- Conditions: `oracle_updated_graph`, `predicted_updated_graph`, `structure_only`,
  `frozen_graph_history`, `shuffled_graph`, `question_only`, `stale_answer`, `majority_answer`,
  `tool_solver`, soft prompt (with and without history). `cached_no_reencode` is gone: it was the
  same input as `frozen_graph_history` and is now an alias for it.

## Pre-registered slices

Fixed before any v4.1 result exists; `scripts/analyze_v4.py` computes all of them.

- **Answer-changing turns:** `gold_answer != stale_answer` (36% of test turns).
- **Text-unresolvable turns:** `mentions_edit == false`. Required next to the headline because the
  answer-changing quota raised the mention rate (0.60 vs 0.26 chance, a gate warning).
- **Shortcut-wrong turns:** turns where the learned one-hop shortcut (`analyze_v4.py`; 61.9% on
  non-attribute test turns, reachability 83%, cycle 82%) answers wrong: 1,510 turns. Accuracy here is
  the cleanest evidence of graph use beyond a local degree/status reader.
- **Write path by paraphrase family**, especially the held-out `notice` family.
- **Per-task tables with ceilings:** each task next to its shortcut ceiling, its majority answer, and
  the tool solver. Cycle is degree-2 only, and its leaf-neighbour rule reaches 79%.

## After the runs (CPU)

```bash
PYTHONPATH=src python3 scripts/analyze_v4.py --runs "$ART" \
    --out documents/experiments/results/v4_1-analysis/analysis.json
```

`$ART/<run>/` mirrors `outputs/<run>/` (result JSON with per-turn rows, metadata, val history), so
the analysis runs from the copied artifacts alone. Commit `$ART` minus checkpoints under
`documents/experiments/results/v4_1-<date>/`.

## Optional: v3 per-turn rows (old server only)

The v3 warm-start validation number is disputed (67.0 in `static_gt_warm_full_validation.json`,
62.6 / 63.1 in the checkpoint sweep). Resolving it needs the per-turn rows, which exist only on the
old server. If it is reachable, and without running anything there:

```bash
rsync -av --include='*/' --include='*.json' --include='*.log' --exclude='*' \
    lmw@<old-server>:/data/lmw/graph-modi/outputs/audit/ documents/experiments/results/v3-audit-rows/
```

Then rescore `static_gt_warm_full_validation.json` and the sweep rows against the same tuple IDs;
a different tuple count or checkpoint path explains the gap. Until then, v3 numbers are a diagnostic
story only, and the warm-start gap is reported as disputed.

## Do not

- Compare v4.1 numbers with v4.0, v3 or September tables as if they were the same benchmark.
- Claim graph reasoning if a model is within a few points of the soft prompt or of the learned
  one-hop shortcut (63.4% static / 61.9% sessions on non-attribute tasks; the lookup-rule 7-task
  ceiling is 41.0%), or if its shortcut-wrong accuracy is at chance. Beating the lookup rules only
  shows degree/status reading.
- Tune hyperparameters on test, or change the configs after `gt42` has run (that breaks the seeds).
