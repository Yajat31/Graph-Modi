# metro_v4 final correctness audit (2026-10-25)

Independent re-check of every stage of the metro_v4 benchmark before merging to the default branch. Labels and edit effects were recomputed with solvers written inside `scripts/audit_v4_final.py` (plain BFS, neighbour-connectivity cycle test, shortest paths) — not by calling `graph_modi.graph.solvers` or `graph_modi.graph.executor` against themselves.

**Verdict: PASS.** `75 pass, 0 fail, 0 warn` (10 info). Results: [`results/benchmark-final-audit-v4-20261025/`](results/benchmark-final-audit-v4-20261025/).

## What was checked

1. **Determinism.** Manifest SHA-256 matches `datasets/metro_v4/`. Regenerating validation static + sessions twice yields identical files. `make_metro_graph_v4` is seed-stable.
2. **Graph generation.** 5376 unique graphs: connected, size/diameter in spec, closed count 1–3, unique names/ids, no self-loops or duplicate edges, attribute domains hold.
3. **Questions.** Every task renders; source/target labels round-trip; `status=` never appears in W(f_i) text; Facts attributes withheld; constrained reachability states the endpoint exemption.
4. **Labels (independent solver).** 0 mismatches on 24k/2.2k/4.4k static and 4455+4455 session turns; 0 mode ties; full 396×11 test grid.
5. **Edits.** Canonical program text parses back; independent apply matches `apply_program_v4`; utterances name the touched stations; `notice` held out of few-shot; invalid DEL skipped.
6. **Sessions.** Gold edits from G_0 replay every G_t; `answer_changed` / stale fields correct; turn indices contiguous; NOOP 0.169/0.168; no net-noop, cancel, disconnect; 1188 sessions, 495 turns/task.
7. **Shortcuts.** Cycle degree-only 50.5%; reach status-only 52.5%; 7-task ceiling 44.1%. Majority prior ~31%; stale answer ~85%.
8. **Splits.** Zero WL-hash overlap across train/val/test and sessions. Station names are drawn from a shared pool (reported, not a fail).
9. **Model plumbing.** `source_id` / `target_id` / hops / task metadata match the query for every v4 task. BERT includes `status` **by design** (graph path carries open/closed); W(f_i) excludes it. Training uses bare questions; eval wraps via `runner._prompt`.
10. **Evaluation.** `normalize_answer` / `answers_match` cover v4 golds; `tool_solver` = 100% and stale matches analytic on a live `evaluate_sessions` run; clustered CIs group by session; `predicted_updated_graph` is in the v4 graph-model configs.
11. **Docs.** Numbers in `benchmark_v4_2026-10-25.md` match the gate `results.json`.

## Bug found and fixed during this audit

ADD edges in v4 used to draw `has_aircon` / `built` from the session RNG. `materialize_states` (and therefore `evaluate_sessions`) replays edits with `apply_edit`, which left those attributes empty, so **every v4 session containing an ADD failed fingerprint checks**.

Fix: `executor.metro_edge_attributes` derives attributes deterministically from endpoint ids; `apply_edit` fills them on ADD; `apply_program_v4` uses the same path. Dataset regenerated. New regression: `test_materialize_states_replays_add_edge_fingerprints`.

A thin cycle text-leak (55% vs 50%) after the RNG-stream change was cleared by balancing cycle yes/no inside prompt-visible source attribute cells (`cleanliness`, `has_rail`).

## Conventions pinned

| Task | Convention |
|---|---|
| Reachability / shortest path | Open endpoints and open intermediates; unweighted BFS; shortest-path answer is path length including endpoints |
| Constrained reachability | Endpoints exempt from the avoided class; open/closed not filtered |
| Within-hops count | Excludes the start node |
| Most-common attribute within hops | Includes the start node; unique mode required |
| Cycle membership | Two neighbours remain connected after the node is removed |
| BERT `status` | Present in node features (graph path); withheld from W(f_i) question text |

## How to re-run

```bash
PYTHONPATH=src python3 scripts/audit_v4_gates.py \
  --work /tmp/graphmodi_v4 \
  --out documents/experiments/results/benchmark-audit-v4-20261025

PYTHONPATH=src python3 scripts/audit_v4_final.py \
  --data datasets/metro_v4 \
  --out documents/experiments/results/benchmark-final-audit-v4-20261025
```
