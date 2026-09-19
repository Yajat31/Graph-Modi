#!/usr/bin/env python3
"""Exploratory probe: does a per-neighbor-token graph prefix help the
aggregation-heavy tasks that stayed near floor after the hops/reasoning_type
fixes (filtered_neighbor_count, most_common_attribute_within_hops,
within_hops_count, within_hops_list)?

Reuses an already-trained TEA checkpoint's frozen GNN/tensorizer/LM/summary
projector verbatim (no retraining of those); trains only the new
NeighborTokenProjector from src/graph_modi/models/multi_neighbor_readout.py
for a small number of epochs, then runs the same static-eval used for the
main pipeline so results are directly comparable to
documents/experiments/results/v2_clegr_extended-20260919/report.md.

Not wired into cli.py / the main pipeline -- deliberately a throwaway script.
Usage (on the remote box, matching the main pipeline's env):
  CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/lmw/hf_cache/huggingface \
    .venv/bin/python scripts/explore_multi_neighbor_readout.py \
    --config configs/v2_clegr_extended_exact2x_tea.yaml \
    --train-config configs/v2_clegr_extended_tea.yaml \
    --epochs 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graph_modi.cli import _static_examples, _static_path, _tea_model  # noqa: E402
from graph_modi.config import load_config  # noqa: E402
from graph_modi.data.v2 import load_static_tuples  # noqa: E402
from graph_modi.evaluation.static_eval import evaluate_static_oracle  # noqa: E402
from graph_modi.models.multi_neighbor_readout import (  # noqa: E402
    MultiNeighborConfig,
    MultiNeighborTEAGLM,
    NeighborTokenProjector,
)
from graph_modi.models.tea_glm import TEAGLMBackend, load_external_component  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="eval-only exact2x config (checkpoint source)"
    )
    parser.add_argument(
        "--train-config", required=True, help="train config (static training data source)"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--neighbor-slots", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-train-examples", type=int, default=0, help="0 = use all")
    parser.add_argument(
        "--eval-tasks", nargs="*", default=None, help="restrict eval print-out to these tasks"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    eval_config = load_config(Path(args.config))
    train_config = load_config(Path(args.train_config))

    print("[explore] loading base TEA model + checkpoint...", flush=True)
    base_model = _tea_model(eval_config)
    checkpoint_dir = Path(
        eval_config.section("model").get(
            "checkpoint", eval_config.output_dir / "projector" / "checkpoint-final"
        )
    )
    metadata_path = checkpoint_dir / "metadata.json"
    load_external_component(
        base_model,
        component="gnn",
        weights_path=checkpoint_dir / "gnn.pt",
        metadata_path=metadata_path,
    )
    load_external_component(
        base_model,
        component="projector",
        weights_path=checkpoint_dir / "projector.pt",
        metadata_path=metadata_path,
    )

    lm_hidden_dim = int(base_model.language_model.get_input_embeddings().embedding_dim)
    neighbor_config = MultiNeighborConfig(
        graph_dim=base_model.gnn.config.output_dim,
        lm_hidden_dim=lm_hidden_dim,
        neighbor_slots=args.neighbor_slots,
    )
    neighbor_projector = NeighborTokenProjector(neighbor_config).to(base_model.device)
    model = MultiNeighborTEAGLM(
        base=base_model,
        neighbor_projector=neighbor_projector,
        neighbor_slots=args.neighbor_slots,
    )

    print("[explore] loading static training data...", flush=True)
    train_tuples = load_static_tuples(_static_path(train_config, "train"))
    rng = random.Random(args.seed)
    rng.shuffle(train_tuples)
    if args.max_train_examples:
        train_tuples = train_tuples[: args.max_train_examples]
    train_examples = _static_examples(train_tuples)
    print(f"[explore] {len(train_examples)} training examples", flush=True)

    optimizer = torch.optim.AdamW(neighbor_projector.parameters(), lr=args.learning_rate)
    batch_size = args.batch_size
    steps_per_epoch = (len(train_examples) + batch_size - 1) // batch_size
    for epoch in range(args.epochs):
        rng.shuffle(train_examples)
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            batch = train_examples[step * batch_size : (step + 1) * batch_size]
            if not batch:
                continue
            output = model(
                graphs=[example.graph for example in batch],
                prompts=[example.prompt for example in batch],
                answers=[example.answer for example in batch],
                source_ids=[example.metadata.get("source_id") for example in batch],
                target_ids=[example.metadata.get("target_id") for example in batch],
                hops=[example.metadata.get("hops") for example in batch],
                reasoning_types=[example.metadata.get("reasoning_type") for example in batch],
            )
            loss = output.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(neighbor_projector.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            if step % max(1, steps_per_epoch // 10) == 0:
                print(
                    f"[explore] epoch {epoch + 1}/{args.epochs} step {step}/{steps_per_epoch} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )
        mean_loss = epoch_loss / max(1, steps_per_epoch)
        print(
            f"[explore] epoch {epoch + 1}/{args.epochs} done mean_loss={mean_loss:.4f}",
            flush=True,
        )

    print("[explore] evaluating on static exact-uniform validation/test...", flush=True)
    max_new_tokens = int(eval_config.section("evaluation").get("max_new_tokens", 96))
    backend = TEAGLMBackend(model, max_new_tokens=max_new_tokens)
    for split in ("validation", "test"):
        tuples = load_static_tuples(_static_path(eval_config, split))
        result = evaluate_static_oracle(tuples, backend, batch_size=16, progress=False)
        print(f"\n=== {split} ===")
        print("overall_accuracy:", round(result["overall_accuracy"], 4))
        print("passed:", result["passed"])
        tasks = args.eval_tasks or sorted(result["task_accuracy"])
        for task in tasks:
            if task in result["task_accuracy"]:
                print(f"  {task}: {round(result['task_accuracy'][task], 4)}")
        out_path = Path(eval_config.output_dir) / f"multi_neighbor_static_eval_{split}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"[explore] wrote {out_path}")


if __name__ == "__main__":
    main()
