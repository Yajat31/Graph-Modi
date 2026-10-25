# metro_v4 GPU run checklist

Weights, Llama, and `outputs/` are not on the audit machine. Run training on the GPU server.

## Before launch

1. Sync this branch; confirm `src/graph_modi/data/v4.py` and the cycle-solver fix are present.
2. Generate or copy data:
   ```bash
   PYTHONPATH=src python3 -m graph_modi.cli generate --config configs/v4_data.yaml
   # or copy datasets/metro_v4/ and verify sha256 against manifest.json
   ```
3. Confirm Llama path in the configs (`model.lm_name`) exists on the server.
4. Use these configs (or copies with per-seed `output_dir` / `training.seed`):
   - `configs/v4_tea.yaml` — contrastive GNN pretrain, then projector
   - `configs/v4_graphtoken.yaml` — joint GNN + projector (must actually train the GNN)
   - `configs/v4_soft_prompt.yaml` — graph-blind control

## Required run settings (from the experimental audit)

- **3 seeds** (e.g. 42, 43, 44) for GraphToken and TEA; report mean ± spread.
- **Contrastive pretraining must update the GNN** (the September TEA bug froze it). Check that `pretrain_graph_encoder_contrastive` leaves a non-random checkpoint and loss drops below ~2.
- **GraphToken must train the GNN** end-to-end (not a frozen TEA copy).
- **Checkpoint selection:** report best validation checkpoint and last checkpoint.
- **Confidence intervals:** session-clustered (cluster bootstrap), not turn-iid.
- Keep evaluation conditions: `oracle_updated_graph`, `frozen_graph_history`, `shuffled_graph`, `question_only`, `cached_no_reencode`, soft prompt.

## Report slices (new for v4)

Alongside overall accuracy, report:

- **Text-unresolvable turns:** `mentions_edit == false` (and, for edge_exists / node_degree / filtered_neighbor_count, turns where the revision text alone cannot decide the answer).
- **Answer-changing turns:** `gold_answer != stale_answer`.
- **Write-path by paraphrase family,** especially the held-out `notice` family.
- Per-task tables; do not average cycle into a headline without noting it is degree-2 only.

## Do not

- Compare v4 numbers to v3 / September tables as if they were the same benchmark.
- Run `bal_*` configs on v3 data.
- Claim graph reasoning if the model is within a few points of the 7-task shortcut ceiling (~44%) or the soft prompt.
