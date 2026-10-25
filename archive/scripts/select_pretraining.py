"""Pick the best GNN-pretraining variant from probe_gnn.py output and write the follow-up configs.

The ridge retrieval probe is uninformative (raw BERT features score 1.000), so variants are
ranked by status sensitivity (how far flipping a node's status moves its embedding, relative to the
distance between two nodes), which separates pretrained from random GNNs (about 0.14 vs 0.03).
Probe AUCs are saturated and are only used to check message passing is intact. A variant must beat
the default by 20% or the default recipe is kept.
Selection uses the validation graphs; report test numbers only for selected runs. Run from the repo root after probe_variants.json exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

AUDIT = Path("configs/audit")
probe = json.loads(Path("outputs/audit/probe_variants.json").read_text())


def score(entry: dict) -> float:
    """Status sensitivity: how far flipping a node's status moves its own embedding, relative to the
    distance between two different nodes. (Status / neighbour-status probe AUCs are saturated at
    0.99-1.00 for every message-passing GNN, random ones included, so they cannot rank variants.)"""
    return entry["flip_own_shift_over_between"]


MIN_NEIGHBOUR_AUC = 0.95  # keep message passing: raw features score 0.51 here
NOISE = 0.20              # a variant must beat the default by 20% to displace it
random_band = [score(v) for k, v in probe.items() if isinstance(v, dict) and k.startswith("random")]
BAND = max(random_band) if random_band else 0.0


def best(prefix: str, default: str, exclude: tuple[str, ...] = ()) -> str:
    candidates = {k: v for k, v in probe.items() if k.startswith(prefix) and isinstance(v, dict)
                  and not k.startswith(exclude)}
    ranked = sorted(candidates, key=lambda k: score(candidates[k]), reverse=True)
    for name in ranked:
        entry = candidates[name]
        print(f"  {name:18s} flip_ratio={score(entry):.3f} neighbour_auc={entry['neighbour_status_probe_auc']:.3f}")
    eligible = [k for k in ranked if candidates[k]["neighbour_status_probe_auc"] >= MIN_NEIGHBOUR_AUC
                and score(candidates[k]) > 2 * BAND]
    if not eligible:
        print(f"  no variant clears 2x the random-GNN spread ({BAND:.3f}) with message passing intact: keeping {default}")
        return default
    winner = eligible[0]
    if default in candidates and score(candidates[winner]) < (1 + NOISE) * score(candidates[default]):
        print(f"  {winner} is within {NOISE:.0%} of the default {default}: keeping the default")
        return default
    return winner


print(f"random-GNN flip ratios: {[round(x, 3) for x in random_band]} (max {BAND:.3f})")
print("3-layer variants:")
shallow = best("pre_", "pre_base", exclude=("pre_deep",))
print("6-layer variants:")
deep = best("pre_deep", "pre_deep_base")
print(f"selected: {shallow} (3-layer), {deep} (6-layer)")

def load(name: str) -> dict:
    return yaml.safe_load((AUDIT / f"{name}.yaml").read_text())


def write(name: str, cfg: dict, header: str) -> None:
    (AUDIT / f"{name}.yaml").write_text(f"# {header}\n" + yaml.safe_dump(cfg, sort_keys=False))


# the 6-layer GNN redone with working pretraining (v3x_deep's training recipe)
cfg = load(deep)
cfg["output_dir"] = "outputs/audit/deep_best"
cfg["model"]["gnn_checkpoint"] = "outputs/audit/deep_best/gnn/gnn.pt"
cfg["training"].update({"epochs": 3, "batch_size": 2, "gradient_accumulation_steps": 12})
write("deep_best", cfg, f"6-layer TEA with working pretraining ({deep}); v3x_deep training recipe")

Path("outputs/audit/selected_pretraining.json").write_text(
    json.dumps({"shallow": shallow, "deep": deep}, indent=2) + "\n")
