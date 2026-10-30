# metro_v4 (generator v4.1) GPU run checklist

Frozen plan for the paper runs: one GPU with at least 40 GB (A100 40/80 GB, L40S, A6000 / 6000 Ada,
H100), about 11–14 GPU-hours. Nothing is trained on the audit machine. Benchmark note: [benchmark_v4_1_2026-10-30.md](benchmark_v4_1_2026-10-30.md).

Framing: this is a **benchmark paper**. The reader (GraphSAGE + linear prefix projector into frozen
Llama-3.1-8B) is frozen as-is; its degree-heuristic ceiling is a reported finding, not something to
tune away during these runs.

## Before launch

0. **GPU machine.** At least 40 GB of GPU memory (training peaks around 24–27 GB; gradient
   checkpointing is a no-op because the frozen LLM stays in eval mode) and about 100 GB of
   persistent disk (checkpoints ~30 GB, Llama 16 GB, environment ~8 GB). The locked torch 2.13
   wheel is built for CUDA 13 and needs **NVIDIA driver ≥ 580**; check `nvidia-smi` first. On an
   older driver, install a torch wheel built for that driver's CUDA into `.venv` and launch with
   `GM=".venv/bin/graph-modi" PY=".venv/bin/python"` so `uv run` does not reinstall the locked
   wheel. Install `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`) and `tmux`; the
   machine needs network access for the BERT download (`bert-base-uncased`).
1. Commit everything and push; on the server `git clone` the repo (not a tarball: provenance
   reads git) and check out that commit. Every result JSON records
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
   Every launch first runs a **preflight**, even after `setup` is done: `uv sync --locked` with the
   training extras, `nvidia-smi`, a check that torch can actually use the GPU (the CLI also refuses
   to build a model on CPU unless `GRAPHMODI_ALLOW_CPU=1`), the Llama path, the data hashes, and the
   BERT download. The `setup` rung then runs the full pytest and fails if any test is skipped (the
   9 torch tests only run on the GPU box) or no summary line is printed. Finished rungs leave
   `<rung>.done` in `$ART`, so a restart resumes; inside a rung, training is skipped when its
   `checkpoint-final` already exists, so a crash during evaluation does not cost a retrain.

## Ladder (budget order)

| Rung | Config(s) | What it answers | Rough time |
|---|---|---|---|
| `setup` | — | env, data hashes, tests | 15 min |
| `soft` | `v4_soft_prompt` | graph-blind trained control (with and without history) | 1 h |
| `gt42` | `v4_graphtoken_s42` | main reader, sum aggregation, working audit recipe | 1.5 h |
| `warm` | `v4_pretrain_sum` → `v4_graphtoken_warm_s42` (1 epoch, like s42) | does contrastive pretraining help the joint reader? | 2.5 h |
| `gt43`, `gt44` | `v4_graphtoken_s{43,44}` | 3 seeds for the main reader | 3 h |
| `random` | `v4_tea_random_sum` | frozen random GNN, projector only: how much does GNN training add? | 1.5 h |

Real wall times go to `$ART/timings.tsv`. If time runs short, drop `random` first, then `gt44`.

Every graph rung runs `static-eval` on validation and test for **both** `checkpoint-best` (lowest
validation loss) and `checkpoint-final`, then multi-turn `evaluate` on test sessions with
`checkpoint-best`. Soft prompt has no validation pass, so it uses `checkpoint-final`.

Recipe (all graph runs, from the audit run that trained, `archive/configs/audit/gt_sum.yaml`):
sum aggregation, GNN dropout 0, projector lr 1e-3, joint GNN lr 1e-4, cosine schedule with 200
warmup steps, validation every 100 steps on 1000 examples, train prompts wrapped in the same
`Question: {q}\nAnswer:` template the evaluator sends (`wrap_prompts: true`; this is not the Llama
chat template), encoder autocast matched to training. GraphToken and random-frozen use the
same projector lr, so the comparison changes only whether the GNN trains. The warm start trains
for 1 epoch like s42, so warm vs s42 changes only the GNN initialisation (it was 2 epochs before
2026-10-30, which confounded pretraining with training length). At evaluation BERT node features
are computed in fp32 while GNN + projector use the training precision.

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
- **Amendment (2026-10-30, before any v4.1 result).** An independent check found the one-hop
  shortcut is not the local ceiling: a two-hop local classifier (no distances) reaches 70.1% static /
  67.0% sessions on non-attribute tasks (reachability 92, cycle 96, edge 95 static), and text-only
  lookups fit on validation sessions reach 36.6% (revision text only) and 78.7% (stale answer +
  revision text) on test sessions. `analyze_v4.py` now reports all of them, and:
  - **Headline update slice:** answer-changing turns whose question does **not** mention the edit
    (`changed_and_no_mention`, 359 test turns). The plain `mentions_edit == false` slice has only
    16% answer-changing turns and the stale answer scores 84% on it, so report it next to stale
    and do not use it for update claims on edge existence, node degree or filtered count (their
    answers cannot change without a mention).
  - **Local-wrong turns:** wrong under both the one-hop and the two-hop classifier (1,052 test
    turns). This replaces shortcut-wrong as the evidence for graph use beyond local reading; report
    it per task next to the slice's majority answer (chance is not 50%).
  - **Text baselines** (`text_only`, `stale_plus_text`) go next to the soft prompt and the frozen
    conditions in every table.
  - Do not slice reachability or shortest path by `hop_depth`: that metadata tag encodes the answer.
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
  local ceiling (two-hop 70.1% static / 67.0% sessions, one-hop 63.4% / 61.9% on non-attribute
  tasks; the lookup-rule 7-task ceiling is 41.0%), or if its local-wrong accuracy is at the slice's
  majority-answer rate. Beating the lookup rules only
  shows degree/status reading.
- Tune hyperparameters on test, or change the configs after `gt42` has run (that breaks the seeds).
