"""Pick the projector learning rate of each graph-reader family on validation loss.

  python3 scripts/sweep_projector_lr.py --runs outputs \
      --out documents/experiments/results/v4_1-analysis/projector_lr_sweep.json

For each family (GraphToken-style, TEA-style, frozen random GNN, warm start) the candidates are the 1e-3 run
(<run>/projector/val_history.json) and the sweep runs (sweep/<run>_lr<x>/projector/...). The chosen
lr has the lowest best validation loss, the same quantity that selects checkpoint-best. Test data
is never read. Copy the chosen values into PROJECTOR_LR in scripts/make_v4_configs.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FAMILIES = {
    "graphtoken": "v4_graphtoken_s42",
    "tea": "v4_tea_s42",
    "random": "v4_tea_random_sum",
    "warm": "v4_graphtoken_warm_s42",
}
SWEEP = {"3e-4": 3e-4, "1e-4": 1e-4}


def best_val(path: Path) -> tuple[float, list[float]] | None:
    if not path.exists():
        return None
    history = [entry["val_loss"] for entry in json.loads(path.read_text())]
    return min(history), history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="outputs")
    parser.add_argument(
        "--out", default="documents/experiments/results/v4_1-analysis/projector_lr_sweep.json"
    )
    args = parser.parse_args()
    root = Path(args.runs)
    report: dict = {}
    for family, run in FAMILIES.items():
        candidates = {1e-3: root / run / "projector" / "val_history.json"}
        for tag, lr in SWEEP.items():
            candidates[lr] = root / "sweep" / f"{run}_lr{tag}" / "projector" / "val_history.json"
        results = {lr: best_val(path) for lr, path in candidates.items()}
        found = {lr: r for lr, r in results.items() if r is not None}
        chosen = min(found, key=lambda lr: found[lr][0]) if found else None
        report[family] = {
            "run": run,
            "best_val_loss": {f"{lr:g}": round(r[0], 4) for lr, r in found.items()},
            "val_history": {f"{lr:g}": [round(v, 4) for v in r[1]] for lr, r in found.items()},
            "missing": [f"{lr:g}" for lr, r in results.items() if r is None],
            "chosen_lr": chosen,
        }
        print(f"{family:7s} {report[family]['best_val_loss']} -> {chosen}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
