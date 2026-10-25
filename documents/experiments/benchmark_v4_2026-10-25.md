# metro_v4 benchmark note (2026-10-25)

Replaces metro_v3 for new runs. Fixes the data problems found in the October audits. **Not comparable with September/October v3 numbers.**

## Generator (`src/graph_modi/data/v4.py`)

- Ring core with hub-restricted chords + pendant chains of length 1–4 (shorter on dense/large graphs so diameter caps hold).
- Cycle queries only at degree 2 (both answers exist; degree-only lookup ≈ majority).
- Reachability / shortest_path only with open endpoints (status-only lookup ≈ 52%).
- Constrained reachability question states that start and end stations are exempt.
- Unique mode required on both before and after for `most_common_attribute_within_hops`.
- Hop tags vary with distance / hop radius in {1,2,3}.
- Node and edge order shuffled; closed count always 1–3.
- ADD edge attributes are **deterministic** (`executor.metro_edge_attributes`), so `materialize_states` / `evaluate_sessions` replay the stored fingerprints.

## Sessions and write path

- Realized NOOP ≈ 17% (force_noop only; non-NOOP programs always change the graph).
- Rejects net-noop, self-cancelling, and reverting programs; DEL only on non-bridges; closed count capped at 3; graphs stay connected.
- ADD edges get track/transfer + attributes like original edges.
- Query sampler balances mention-of-edit × answer; `mentions_edit` in turn complexity.
- Paraphrase families direct / service / notice; notice held out of few-shot examples in `tea_glm.py`.

## Gate results (`scripts/audit_v4_gates.py`)

SUMMARY: **43 pass, 0 fail, 0 warn.**

| Gate | Result |
|---|---|
| Labels (static + sessions gold/stale) | 0 mismatches, 0 ties |
| Test grid | 396 items × 11 tasks |
| Cycle degree-only ceiling | 50.5% |
| Reachability status-only ceiling | 52.5% |
| 7-task shortcut ceiling (ex node_degree) | 44.1% (8-task 51.1% with node_degree at 100%) |
| Session NOOP | 0.169 val / 0.168 test |
| Net-noop / cancel / disconnect | 0 |
| Text leak audit | clean |
| Split overlap | 0 |

Manifest hashes: `documents/experiments/results/benchmark-audit-v4-20261025/manifest.json` and `datasets/metro_v4/manifest.json`.

## Configs

`configs/v4_data.yaml`, `v4_tea.yaml`, `v4_graphtoken.yaml`, `v4_soft_prompt.yaml` — all set `distribution: metro_v4`, `data_dir: datasets/metro_v4`. Graph-model configs include `predicted_updated_graph`.

## Run checklist

See [v4_run_checklist.md](v4_run_checklist.md).
