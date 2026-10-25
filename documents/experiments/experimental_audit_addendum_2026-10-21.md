# Audit addendum: second pass on `results/audit-20261016`

Builds on [`experimental_audit_2026-10-21.md`](experimental_audit_2026-10-21.md). That audit established that the October results exist and match the write-ups. This addendum asks a narrower question: **do the October results support the interpretation the status report gives them?** Everything here is recomputed from committed JSON by [`archive/scripts/audit_second_pass.py`](../../archive/scripts/audit_second_pass.py); no GPU, no new runs.

## Summary

| # | Finding | Effect on the first audit |
|---|---|---|
| 1 | On items where the shortcut is wrong, the GraphToken models are **below chance** on binary tasks. They are following the shortcut, not just failing to beat it. | Strengthens C3 |
| 2 | About 90% of the re-encoding gain comes from turns where the shortcut is right. On shortcut-wrong turns the gain is +3 to +4 points. | Refines C3 |
| 3 | Only the warm start reaches the shortcut tracker, and not distinguishably (+1.3 multi-turn, +2.6 static, against unpaired half-widths of about 1.5). The scratch and random-GNN models are 4 to 5 points below it. | Refines C3 |
| 4 | The `gt_warm` checkpoint sweep scores 62.6% (final) and 63.1% (best) on validation. `static_gt_warm_full_validation.json` reports 67.0%. **Unexplained 4.4-point gap.** | New, open |
| 5 | Seed spread in aggregate accuracy (0.4 points) hides large per-slice swings. Every pretraining and warm-start claim rests on one pretraining seed and one dataset draw. | Refines M1 |
| 6 | Multiplicity is not a problem for the headline conclusions: 114 paired tests, only two have p in [0.01, 0.05). `p = 0.0` is a resolution floor. | Closes a worry |
| 7 | Pretraining-variant losses are not comparable across objectives. | Minor, new |

## 1. Models follow the shortcut on shortcut-wrong items

The shortcut (`scripts/shortcut_audit.py`) is a gradient-boosted classifier on degree, endpoint status, size and same-line. On the 1,142 static test items where it is wrong, the first audit and the status report both note every model scores 22-26%. That figure averages binary and multi-class tasks, so it hides something. On a binary task, a model that is wrong on every shortcut-wrong item is agreeing with the shortcut. Agreement with the shortcut is `1 - accuracy` there, and chance is 0.50.

| Model | constrained_reach (n=121) | cycle_membership (n=34) | edge_exists (n=118) | reachability (n=23) |
|---|---|---|---|---|
| gt_sum (seed 42) | 0.82 | 0.50 | 0.76 | 0.30 |
| gt_sum_s43 | 0.88 | 0.74 | 0.75 | 0.17 |
| gt_warm | 0.83 | 0.88 | 0.82 | 0.74 |
| tea_sum_random | 0.69 | 0.71 | 0.74 | 0.61 |
| tea_sum (pretrained, frozen) | 0.64 | 0.47 | 0.55 | 0.65 |
| old_tea (random mean-agg GNN) | 0.54 | 0.94 | 0.42 | 0.17 |

Reading:

- On `constrained_reachability` and `edge_exists`, every model that uses sum aggregation agrees with the shortcut 69-88% of the time (n=121 and 118, so about +/-9 points). Those are the two tasks where a real graph reader would have to look at an edge or a path.
- `old_tea` sits near 0.5 on those same two tasks. It is not using the shortcut; it is also not reading the graph. The sum-aggregation models moved from "ignores the graph" to "uses degree and status", which is what the status report concludes in Section 0b. This table adds the sign: they are **systematically wrong** where the shortcut fails.
- `cycle_membership` and `reachability` slices are n=34 and n=23. Treat them as directional only. `reachability` flips between seeds (0.30 vs 0.17 for two scratch seeds, 0.74 for warm start) and is too small to separate from noise.

## 2. Where the re-encoding gain lives

The status report's "+12 from re-encoding" is the updated-graph minus frozen-graph accuracy. `shortcut_audit.json` stores accuracy on shortcut-right ("easy", n=2,559) and shortcut-wrong ("hard", n=1,401) turns separately.

| Model | gain, all turns | gain, easy turns | gain, hard turns |
|---|---|---|---|
| gt_sum | 14.1 | 19.7 | 3.9 |
| gt_warm | 19.8 | 28.2 | 4.4 |
| tea_sum_random | 13.6 | 19.1 | 3.4 |
| shortcut tracker | 19.1 | n/a | n/a |

Hard turns are 35% of the 8-task multi-turn set. They contribute about 1.2 to 1.4 of the 13.6-14.1 points for the scratch and random-GNN models (1.6 of 19.8 for the warm start), so roughly 90% of the gain is on easy turns. The +3 to +4 on hard turns is positive and consistent across the three models, which is mild evidence that something beyond the shortcut is read. It cannot be tested properly from the stored numbers: there is no per-turn data to pair on, and the unpaired half-width on 1,401 turns is about 2.3 points for each accuracy.

## 3. The warm start is the only model that reaches the shortcut

| 8 non-attribute tasks | Model | Shortcut | Difference | Unpaired +/- |
|---|---|---|---|---|
| Multi-turn (n=3,960) | gt_sum 60.5 | 64.6 | -4.1 | 1.5 |
| | gt_warm 65.9 | 64.6 | +1.3 | 1.5 |
| | tea_sum_random 59.3 | 64.6 | -5.3 | 1.5 |
| Static (n=3,168) | gt_sum 59.6 | 64.0 | -4.3 | 1.7 |
| | gt_warm 66.5 | 64.0 | +2.6 | 1.6 |
| | tea_sum_random 59.2 | 64.0 | -4.7 | 1.7 |

The status report says the models are "at the shortcut ceiling". The numbers say more precisely: the scratch and random-GNN models are **below** it by 4-5 points, and only the warm start is level with it. The warm start's static margin (+2.6) is slightly outside the unpaired interval. A paired test would be tighter but needs per-item rows that are not committed. The honest statement is "warm start matches or slightly exceeds a four-feature classifier", which is not evidence of multi-hop reasoning either way.

## 4. Open discrepancy: warm-start sweep vs reported validation

| Source | gt_warm validation |
|---|---|
| `static_gt_warm_full_validation.json` (`overall_accuracy`) | 67.0 |
| `sweep_gt_warm.log`, `checkpoint-final` | 62.6 |
| `sweep_gt_warm.log`, `checkpoint-best` | 63.1 |

For the other sweep with a committed JSON, `sweep_e2e_validation.json`, `tuples` is 2,178 (the full validation set). For `gt_warm` the committed artifact is only the log, and the log does not record the `--limit` argument or the loaded checkpoint path. `command_sweep_checkpoints` subsamples when `--limit` is given, so a subsample is one candidate explanation. A loading difference between `static-eval` and the sweep's `_load_tea_checkpoint` path is another. The warm start is also the only model that loads a separate `pre_sum` GNN, which makes a loading difference worth checking first. There is no clean control for this: the other sweep logs (for example pilot D, final 0.530) come from one-epoch pilots, not from the runs behind the static validation files, so they cannot be compared with those files.

This matters because the warm start is the headline model (65.8% test) and the only one level with the shortcut. Until the gap is explained it is not clear which number is the warm start's validation accuracy. This is the cheapest open item to resolve (see Section 8).

## 5. Seed and pretraining evidence is thinner than the tables imply

- **Aggregate spread hides slice spread.** Three scratch GraphTokens agree to 0.4 points on multi-turn (57.3 / 57.0 / 56.8) and 0.4 on static (52.7 / 53.0 / 53.1 across the three static files). On the reachability slices the two seeds with slice data disagree completely: with both endpoints open 68.2 vs 39.1; with an endpoint closed 31.2 vs 69.9. Aggregate seed stability is not evidence that two seeds learned the same function.
- **One dataset draw.** All ten pretraining metadata files share `data_sha256` `80b1df47...` and `seed: 42`, and all training runs read `datasets/metro_v3` generated at seed 42. Seeds 43 and 44 change initialisation and shuffle only. The seed spread therefore excludes dataset-sampling variance.
- **One pretraining seed behind "pretraining helps".** The warm start loads `outputs/audit/pre_sum/gnn/gnn.pt` (seed 42) and is a single joint-training seed. Its +6.4 points over scratch (paired CI [-7.9, -4.9]) is far larger than the 0.4-point seed spread, so a real effect is plausible. The claim "contrastive pretraining then joint training is complementary" is still n=1 for pretraining and n=1 for joint training.

## 6. Multiplicity

`stats_test_final.json` contains 114 paired tests (21 cross-model pairs and the within-model contrasts, across conditions). 73 have p < 0.05. Only two fall in [0.01, 0.05):

- `gt_sum - gt_warm`, `structure_only`: p=0.034, -1.4 points
- `gt_sum_s43 - gt_warm`, `frozen_graph_history`: p=0.029, -1.5 points

The status report's conclusions rest on the large contrasts (scratch vs warm start on the updated graph, sum vs mean aggregation, graph vs soft prompt), all with p below the Bonferroni level of 0.00044. The two borderline results are not cited as findings. Separately, `p_value: 0.0` appears throughout; with 5,000 resamples it should be reported as p < 0.0002.

## 7. Pretraining variants are not comparable by loss

Across `pretrain_pre_*_metadata.json`, final losses range from 1.58 (`pre_long`) to 3.69 (`pre_status`). These are not on one scale: `pre_nodrop_t02` uses temperature 0.2 (loss 3.61), `pre_status` uses a status-weighted target (3.69), and `pre_attrs` uses an attributes-only target (2.06). Only the same-objective, same-temperature family (`pre_base` 1.75, `pre_nodrop` 1.59, `pre_nodrop_lr3` 1.59, `pre_long` 1.58, `pre_sum` 1.67) can be compared by loss, and there the differences are small. In-batch retrieval is 1.0 for most variants because raw BERT already scores 1.0 (status report), so it does not discriminate. `pre_deep_base` and `pre_deep_nodrop` are the only variants that fit poorly (top-1 0.82 and 0.68). `select_pretraining.py` is therefore right to rank by flip sensitivity rather than loss, and that choice should be cited as a validation-graph heuristic, not as a QA-validated one.

## 8. Changes to the first audit's classifications

| Claim | First audit | Now |
|---|---|---|
| Gains are multi-hop graph reasoning | Unsupported | **Unsupported, with direct evidence against**: below-chance on shortcut-wrong binary items for sum-aggregation models |
| Models are at the shortcut ceiling | (reported as stated) | Qualified: scratch and random-GNN models are 4-5 points below it; only the warm start is level |
| Pretraining + joint training is complementary | Qualified | Qualified, narrower: n=1 pretraining seed, n=1 joint seed, and an unexplained validation gap (Section 4) |
| Seed noise is ~0.4 points | Supported | Qualified: true in aggregate and for one dataset; per-slice behaviour differs by seed |
| Paired statistics are sound | Stored, not re-derivable | Supported for headline contrasts; multiplicity checked (Section 6) |

## 9. Decisive follow-ups

Ordered by cost.

1. **Resolve the warm-start gap (minutes of GPU).** Re-run `static-eval` on `gt_warm_full` validation at `checkpoint-final` and `checkpoint-best`; re-run the sweep without `--limit`. Record the exact command, the checkpoint path and the tuple count in the log row. If the sweep and `static-eval` still disagree, the discrepancy is a checkpoint-loading bug.
2. **Commit per-item rows for the audit runs.** Static `rows` and multi-turn `evaluation.json` rows for `gt_sum*`, `gt_warm`, `tea_sum*`. With them, Sections 2 and 3 become paired tests with confidence intervals, and the hard-turn gain can be tested directly.
3. **Pre-register the primary metric for any new run.** Accuracy on shortcut-wrong items (hard slice), per task, with a session-cluster bootstrap, compared with each task's chance level. Report agreement-with-shortcut next to it.
4. **Degree and status balanced data (`archive/configs/audit/bal_*.yaml`).** Already specified, not run. It removes the shortcut by construction; the hard-slice metric above then becomes the whole test set. Superseded by metro_v4.
5. **Replicate warm start and its pretraining over seeds 42-44, and over two dataset seeds.** This closes Section 5.

## 10. Reproducing the numbers

```bash
python archive/scripts/audit_second_pass.py
```

reads `documents/experiments/results/audit-20261016/{shortcut_audit,stats_test_final}.json` and prints the tables in Sections 1-3 and 5-6. Sections 4 and 7 were read from `sweep_gt_warm.log`, `static_gt_warm_full_validation.json` and the `pretrain_*_metadata.json` files directly.
