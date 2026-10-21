# Experimental Validity Audit — 2026-10-21

**Scope.** Artifact-first audit of experimental validity and reproducibility for GraphModi’s v3 / audit runs. No GPU training or full model reruns.  
**Protocol standard.** Locked plan and static gate in `documents/final_proposal.pdf`; reconciled against `documents/midterm_report/main.tex`, `README.md`, and `documents/experiments/status_report_v3.md`.  
**Auditor method.** Claim→evidence matrix over committed configs, code, result JSON, and lightweight CPU checks. HEAD at audit time: `5549ac5`.  

---

## 0. Executive verdict

| Question | Verdict |
|---|---|
| Can headline **numbers** in the README / midterm / status report be checked against retained files? | **Mostly yes** for September v3/v3x static rows and multi-turn aggregates; **partially** for October audit aggregates. |
| Do those numbers support the **scientific claims** as originally framed (TEA vs GraphToken; pretrained GNN; static gate cleared; multi-hop re-encoding)? | **No.** Midterm method labels and several dynamic claims are **unsupported** or only **qualified**. The status report’s corrections are closer to the evidence. |
| Is the study **reproducible** from a fresh clone? | **No.** `datasets/metro_v3` is absent and gitignored; weights and per-turn `evaluation.json` rows are absent; October audit runs are logged as `uncommitted`. |

**Bottom line.** Treat `status_report_v3.md` Sections 0–0b as the current interpretation of the numbers, not the midterm abstract or Tables that still present “TEA vs GraphToken” and ~35% dynamic gains as method comparisons. Absolute accuracies in committed JSON match the write-ups; the **causal / methodological story** does not clear the proposal’s own gates.

---

## 1. Scope and limitations

**In scope**

- Claim–evidence consistency across proposal, midterm, README, status report, and `log.md`.
- Provenance of data, configs, checkpoints, git state, and environment.
- Recomputation of retained static per-item rows and consistency of multi-turn / audit aggregates.
- Lightweight CPU checks of generation invariants, metrics, and baseline naming (pytest unavailable in this environment; see §5).

**Out of scope (by design)**

- GPU retraining, multi-seed reruns, degree/status-balanced data regeneration.
- Security / dependency CVE review.
- Re-scoring of free-form generations without retained `raw_answer` rows (multi-turn).

**Limitations of this audit**

- Paired session-bootstrap intervals in `stats_*.json` cannot be recomputed without per-turn rows; they are treated as **internally consistent stored results**, not independently re-derived.
- October audit static files have **no per-item rows**; only aggregates were verified against the status report.
- `uv run pytest` could not be completed here (dependency sync hung; no `pytest` in the active interpreter). Equivalent checks were run via `PYTHONPATH=src` scripts (§5).

---

## 2. Claim-to-evidence matrix

Legend: **S** supported · **Q** qualified · **U** unsupported · **X** unverifiable from retained artifacts.

### 2.1 Protocol and gates

| Claim | Source | Evidence | Class |
|---|---|---|---|
| Static reader must clear ≥70% overall and no core task &lt;50% before dynamic claims | Final proposal §7; `static_eval.GATE_OVERALL_THRESHOLD = 0.70` | Static test TEA 36.3%, “GraphToken” 40.3%, soft 32.8%; count tasks remain &lt;20% in midterm | **U** (gate failed; midterm still answers RQ2) |
| Graph family is locked Watts–Strogatz metro (`documents/decisions/0001-…`, status accepted, not superseded) | Decision 0001 | Actual benchmark is `metro_v3` (12–22 nodes, three densities). No decision record for v3 | **U** (protocol drift) |
| OOD / larger graphs evaluated | Final proposal §5 | Status report: no OOD eval | **U** / **X** |

### 2.2 Methods as labeled

| Claim | Source | Evidence | Class |
|---|---|---|---|
| TEA contrastive-pretrains the GraphSAGE encoder | Midterm; README Models table | Historical bug: GNN frozen during pretrain; old loss ~4.48 vs chance 4.91. Fix now in `training.py` (~503–507). Pre-fix TEA/v3x = random frozen GNN + projector | **U** for Sept results; fix **S** in current code |
| GraphToken jointly trains GNN + projector | Midterm tables; README (partially corrected in Limitations) | Sept “GraphToken” is second TEA run (frozen GNN). True e2e mean-agg collapsed (33.0%). Working GT is audit `gt_sum*` | **U** for midterm column; **Q** for audit sum-agg |
| Soft prompt is a graph-blind control | README / status | Soft static test = 32.8% = question-only prior; recomputed from rows | **S** |
| Training-free framework | Final proposal abstract | Projector (and sometimes GNN) are trained; only outer edit loop is training-free | **Q** (wording overclaim) |

### 2.3 Dataset and labels

| Claim | Source | Evidence | Class |
|---|---|---|---|
| v3 grid: 1188 sessions / 4455 turns; yes/no 50/50; seed 42 | README; `dataset_audit.json` | Audit JSON + regeneration: 1188/4455; lengths {1,2,4,8}×297; deterministic at seed 42 | **S** (shape); **X** (bit-identical to original without `data_sha256` check on JSONL) |
| Text leak audit clean | Midterm / presentation | `dataset_audit.json` `text_leak_audit`: all `leak_flag: false` | **S** for **text** leak only |
| No structural shortcut | Implied by early write-ups | `shortcut_audit.json`: degree+status(+size,same-line) mean 64.0% on 8 tasks; models at ceiling | **U** (shortcut dominates) |
| NOOP ~15–20% | Final proposal | Test NOOP 1668/4455 ≈ **37.4%**; 100% edit-correct on NOOP | **U** vs protocol; inflates write metrics |

### 2.4 Results and RQs

| Claim | Source | Evidence | Class |
|---|---|---|---|
| Static: TEA 36.3 / GT 40.3 / soft 32.8 | README; midterm | Recomputed from `static_*_test.json` rows — exact match | **S** (numbers); **U** (GT method label) |
| Multi-turn oracle TEA 35.1 / GT 37.9; pred 34.9 / 37.6 | README; midterm | Aggregates in `exact2x_*_analysis.json` match | **S** (numbers); **Q** (interpretation) |
| Edit exec-equiv 95.2%; state exact 88.6% | README; midterm | Predfix summaries match; NOOP 37% perfect; edge utterances = edit commands | **Q** (overstates free-form write path) |
| RQ2: re-encoding beats stale+history | Midterm inferences | TEA oracle−frozen ≈ +2.6 pts, paired CI excludes 0 (`stats_test_existing_runs.json`), but below ~3 pt run-noise floor vs second TEA (−2.8 [−4.3,−1.4]); soft-prompt gap +0.9 [−0.8,+2.6] n.s. | **Q** / weak |
| TEA graph tokens beat soft prompt | Midterm (hedged) / status | Session bootstrap: not significant on mean-agg TEA | **U** as a firm claim |
| Sum-agg GraphToken ~53% static; warm start 65.8%; +12 re-encode | Status §0a | Aggregate JSON matches; 3 seeds for `gt_sum`; warm/TEA sum single-seed; runs `uncommitted` | **Q** (numbers consistent; provenance weak; shortcut §0b) |
| Gains are multi-hop graph reasoning | Status §0a before 0b; midterm | Shortcut audit: where shortcut wrong, all models ~22–26% | **U** |
| v3x depth / projector / data ablations inform architecture | Midterm timeline / log | All used broken pretrain; depth oversmooths; proj/data collapsed | **U** |

---

## 3. Severity-ranked findings

### Critical

**C1. Static gate failed; dynamic claims were published anyway.**  
Proposal requires ≥70% static accuracy before interpreting multi-turn gains. Observed static test ≈36–40%. Midterm still answers RQ2 and frames re-encoding as motivated.

**C2. Method labels do not match executed training (Sept).**  
- “Pretrained TEA” = frozen **random** GNN + projector.  
- “GraphToken” column = second TEA projector init.  
Status report §0 corrects this; midterm abstract/tables and parts of README Models table still mislead if read alone.

**C3. Structural shortcut ceiling, not multi-hop reasoning (Oct).**  
Degree + endpoint status (+ size, same-line) matches or exceeds sum-aggregation models on 8 tasks (shortcut mean 64.0% vs GT 59.6% / warm 66.5%). On shortcut-wrong items every model scores ~22–26%. The “+12 from re-encoding” matches a status/degree tracker. Decision 0001 was written to kill “closed ⇒ unreachable”; v3 reintroduces endpoint-status dominance (status in BERT features, withheld only from W(f_i) text).

**C4. Study not reconstructible from the repo.**  
- `datasets/` gitignored and empty (no `metro_v3`).  
- Checkpoints / `evaluation.json` per-turn rows not retained.  
- Oct 13–16 audit rows in `log.md` marked `uncommitted`; no `git_sha` in result JSON.  
- Protocol docs say datasets and `evaluation.json` should be committed (`documents/experiments/README.md`); practice diverged.

### Major

**M1. Single seed and last-checkpoint selection.**  
Main v3 numbers are seed 42, `checkpoint-final`. Documented run variance ~3–4 points under mean aggregation. `final-study` CLI does not retrain per seed. No `checkpoint-best` for main configs (`eval_every_steps` default 0).

**M2. Train/eval prompt and history mismatch.**  
Training uses bare `render_question`; inference wraps `Question: …\nAnswer:`. No training example contains revision history, so `frozen_graph_history` ≈ `structure_only` (0.2 pts) is not evidence that “history does not help” in a trained sense—it is OOD text.

**M3. Statistical unit mismatch in published figures.**  
Midterm / `summarize_v3_exact2x.py` use turn-level Wilson / binomial SE. Session bootstrap CIs are ~25% wider. Prefer `scripts/stats_v3.py` / stored `stats_*.json` for claims.

**M4. Write-path metrics inflated by template design.**  
37% NOOP (100% correct); edge revisions are the edit command itself; few-shot uses single-word names vs two-word v3 names. 95.2% exec-equivalent is not free-form edit competence.

**M5. Shared `output_dir` overwrite risk.**  
`v3_tea` / `v3_tea_b2` and GraphToken b1/b2 share directories; later runs can replace `checkpoint-final` without a new run id.

**M6. Invalid GraphToken and broken pretrain contaminate v3x conclusions.**  
Depth, MLP projector, and 3× data variants are not informative about capacity given random GNN / collapse.

### Moderate

**N1. `shuffled_graph` is a one-step rotation of file order**, not an independent random shuffle (`runner.py`).  
**N2. `majority_prior` name means stale answer (59.3%)**, not per-task majority (~29%). Legacy alias exists; easy to misread old tables.  
**N3. Predicted vs oracle table rows sometimes mix runs** (midterm captions); within-run gaps live in status report (34.9 vs 35.0).  
**N4. Environment unbound:** machine-local Llama path; `uv.lock` predates experiments; `use_deterministic_algorithms(..., warn_only=True)`.  
**N5. `.gitignore` contradicts comments** (blanket `datasets/` ignore vs “do not ignore datasets”). User’s local `.gitignore` modification preserved; not changed by this audit.  
**N6. Auto experiment log hardcodes `"branch": "test"`** and SHA lines were pruned from committed `log.md`.

### Minor

**n1.** Soft prompt static equals prior — good negative control (**supported**).  
**n2.** Tool solver 100% — sanity ceiling present in aggregates.  
**n3.** Leading-token answer parser fixed and tested in code; old free-form baselines may still be under-scored if not re-eval’d.  
**n4.** `documents/README.md` still points primarily at interim proposal.

---

## 4. Reproducibility assessment

| Artifact | Present? | Role |
|---|---|---|
| Configs `configs/v3_*.yaml`, `configs/audit/*.yaml` | Yes | Recipe |
| `documents/experiments/results/v3-20260929/` static **with rows** | Yes | Recompute static accuracy |
| Multi-turn `exact2x_*_analysis.json` aggregates | Yes | Headline check only |
| `stats_test_existing_runs.json`, `stats_test_final.json` | Yes | Stored paired tests (not re-derivable) |
| `dataset_audit.json` | Yes | Shape / label / text-leak; **no seed/hash/git** |
| Pretrain `metadata.json` (audit) with `data_sha256`, losses | Yes | Shows fixed pretrain loss 1.746 vs chance 4.908 |
| `datasets/metro_v3/*.jsonl` | **No** | Blocks hash verify / join to sessions |
| Projector/GNN weights | **No** | Blocks exact re-eval |
| Per-turn `evaluation.json` | **No** | Blocks re-bootstrap / re-parse |
| Git pin for Oct audit | **No** (`uncommitted`) | Blocks code identity |
| `uv.lock` + CUDA / GPU id in results | Partial / **No** | Environment drift |

**What a fresh clone can do today**

1. Regenerate *some* `metro_v3` at seed 42 from current `data/v3.py` (grid shape verified).  
2. Recompute September static accuracies from committed rows.  
3. Re-read aggregates and status report.

**What it cannot do**

1. Prove regenerated data matches `data_sha256` `80b1df47…` without the original JSONL.  
2. Reproduce September TEA numbers under **current** code (pretrain unfreeze changes the recipe).  
3. Independently recompute October paired CIs or shortcut features.

---

## 5. Lightweight verification performed

### 5.1 Metric recomputation

| Check | Result |
|---|---|
| All `v3-20260929` / `v3x-20260929` `static_*_test.json` overall vs mean(`rows.correct`) | **Exact match** (9 files) |
| TEA oracle / predicted / edit / state headlines vs README | **Match** (35.1 / 34.9 / 95.2 / 88.6) |
| Audit static aggregates vs status §0a (gt_sum 52.7, warm 65.8, tea_sum_random 53.3, tea_sum 45.2) | **Match** within rounding |
| Soft prompt static = 32.828% | **Match** prior |

### 5.2 CPU invariant checks (`PYTHONPATH=src`, no pytest)

| Check | Result |
|---|---|
| `GATE_OVERALL_THRESHOLD == 0.70` | Pass |
| `majority_prior` → `stale_answer` | Pass |
| `WFI_EXCLUDE == "status"`; `status=` absent from `render_question` | Pass |
| Leading-token normalize (`No, there are 2…` → `no`) | Pass |
| Multi-word edit parse | Pass |
| `generate_sessions_v3` → 1188 sessions, 4455 turns, lengths 297×{1,2,4,8} | Pass |
| Deterministic regeneration (seed 42) | Pass |
| NOOP-by-fingerprint ≈ 38.3% on regenerated val | Pass (near reported 37%) |
| Session bootstrap CI wider than Wilson on correlated toy data | Pass |
| `set_degree_balance(True)` changes static tuple ids | Pass |

**Blocked:** full `pytest` (`ModuleNotFoundError: pytest`; `uv run` sync did not finish). Torch GNN update test (`test_contrastive_pretraining_actually_updates_the_gnn`) not executed here; fix is present in source at `training.py` 503–507.

---

## 6. What can be claimed now vs not

### Supported now (with retained artifacts)

- Soft prompt matches the question-only prior on static test (graph-blind control works).  
- September static and multi-turn **headline accuracies** match committed JSON.  
- Under mean aggregation + frozen random GNN, absolute accuracy is ~35% multi-turn; write path on **templated** edits is high (95% exec-equiv) with little QA drop.  
- TEA vs second TEA run differs by ~2.8 pts (stored paired test)—single-seed gaps of that size are not method differences.  
- TEA vs soft prompt on exact2x is **not** significant by stored session bootstrap.  
- Text-only leak flags are clean on the audited static sets.  
- Current code intends to unfreeze the GNN during contrastive pretraining.

### Qualified (true only with caveats)

- Small positive oracle−frozen gap for mean-agg TEA (~2.6 pts) — real in paired test, but within run-noise vs the second TEA run; does not clear the static gate.  
- Sum-aggregation GraphToken ~53% and warm start ~66% — consistent aggregates, three seeds for scratch GT, but uncommitted code, no weights, and **at shortcut ceiling**.  
- Edit prediction “costs nothing” — only for templated revisions with high NOOP rate.

### Unsupported

- Midterm TEA vs GraphToken as two methods.  
- Midterm/README implication that InfoNCE trained the Sept GNN.  
- Clearing of the 70% static gate / readiness for strong dynamic claims.  
- Multi-hop topological reasoning as the explanation for sum-agg gains.  
- v3x depth/projector/data conclusions about architecture.  
- Adherence to decision 0001’s locked distribution.

### Unverifiable without new artifacts or runs

- Bit-identical dataset to September (`data_sha256`).  
- Re-derivation of paired bootstrap tables.  
- Exact October checkpoint identity and dirty-tree state.  
- Effect of degree/status-balanced regeneration (`bal_*` not run).

---

## 7. Prioritized remediation

1. **Correct public claims** in midterm/README/presentation: retitle Sept “GraphToken” as TEA run 2; state frozen-random-GNN; point to status §0–0b; do not claim gate pass.  
2. **Restore provenance:** stop ignoring session JSONL (or vendor a hashed tarball); commit `evaluation.json` summaries **with rows** or a compact parquet of `(session_id, turn, condition, raw_answer, correct)`; write `git_sha`, `dirty`, `data_sha256`, `config_path` into every result JSON.  
3. **Supersede or amend decision 0001** with a `metro_v3` decision record (and optional `bal` variant).  
4. **Before any new scientific claim:** regenerate with degree/status balance; report session-bootstrap + multi-seed; select checkpoints on validation; match train/eval prompts; include a shortcut baseline in every table.  
5. **Separate output dirs** per run id; never share `checkpoint-final` across recipes.  
6. **Pin environment** in the run record: `uv.lock` hash, torch/CUDA, GPU name, LM path + revision.

---

## 8. Minimal rerun package (not executed)

Decisive controls only; retain artifacts listed under each.

| # | Run | Config / script | Why decisive | Retain |
|---|---|---|---|---|
| R1 | Regenerate `metro_v3` @ seed 42 on a **pinned commit** | `configs/v3_tea.yaml` `generate` | Hash vs `80b1df47…` / freeze manifest | `*.jsonl`, `audit.json`, `data_sha256` |
| R2 | Degree+status balanced data | `configs/audit/bal_*.yaml` | Break shortcut ceiling | Full JSONL + shortcut_audit redo |
| R3 | TEA fixed pretrain ×3 seeds (sum agg, lr from pilots) | `tea_fixed_s42/43/44` + sum twin | Separates pretrain from aggregation | `gnn.pt`, projector ckpt, static rows, eval rows |
| R4 | GraphToken sum ×3 seeds | `gt_sum`, `gt_sum_s43`, `gt_sum_s44` | Already partially done; needs commit pin + rows | Same |
| R5 | Warm start ×3 seeds | `gt_warm_full` | Currently single-seed | Same |
| R6 | Shortcut + zero-GNN + matched autocast | `shortcut_audit.py`, `zerognn_*`, `amp_*` | Ceiling and numerics | JSON + per-item static |
| R7 | Prompt-matched retrain (train with `Question:/Answer:`) | new config | Isolates history OOD issue | Eval rows for history conditions |

**Do not claim** multi-hop updating until R2 models exceed the shortcut on shortcut-wrong items with session-bootstrap CIs excluding 0, and static gate policy is either met or explicitly waived in a decision record.

---

## 9. Artifact index used

- `README.md`, `documents/midterm_report/main.tex`, `documents/final_proposal.pdf`, `documents/decisions/0001-graph-distribution.md`  
- `documents/experiments/{log.md,status_report_v3.md,README.md}`  
- `documents/experiments/results/{v3-20260929,v3x-20260929,audit-20261016}/`  
- `src/graph_modi/{cli.py,training.py,evaluation/runner.py,evaluation/static_eval.py,evaluation/metrics.py,data/v3.py,models/tea_glm.py}`  
- `configs/v3_*.yaml`, `configs/audit/` (via `scripts/make_audit_configs.py`)  
- `.gitignore`, `pyproject.toml` / `uv.lock` (presence only)

---

## 10. Sign-off

| Item | Status |
|---|---|
| Artifact-first (no GPU reruns) | Done |
| Claim–evidence matrix | Done |
| Severity-ranked issues | Done |
| Reproducibility gaps | Done |
| Minimal rerun package | Specified, not executed |
| User `.gitignore` change | Untouched |

*End of audit.*
