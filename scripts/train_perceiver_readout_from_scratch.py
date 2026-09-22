#!/usr/bin/env python3
"""Two-phase from-scratch training for the Perceiver/Q-Former-style
cross-attention resampler readout (src/graph_modi/models/perceiver_readout.py):
pretrain a fresh GNN, freeze it, then train the resampler alone against the
LM loss on the static training corpus.

Mirrors scripts/train_aspect_readout_from_scratch.py's structure exactly
(same two phases, same GPU-memory-cap discipline) -- see that script's and
perceiver_readout.py's docstrings for why this one also trains its own GNN
from scratch (run in parallel with the aspect-readout job, since GNN
pretrain alone is cheap and GPU 1 had spare capacity under the 50% cap).

Not wired into cli.py / the main pipeline -- throwaway exploratory script.

Usage (on the remote box, matching the main pipeline's env):
  CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/lmw/hf_cache/huggingface \
    .venv/bin/python scripts/train_perceiver_readout_from_scratch.py \
    --config configs/v2_clegr_extended_perceiver_readout.yaml \
    --gpu-memory-fraction 0.5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graph_modi.cli import (  # noqa: E402
    _load_local_gnn,
    _neural_components,
    _static_examples,
    _static_path,
)
from graph_modi.config import load_config  # noqa: E402
from graph_modi.data.v2 import load_static_tuples  # noqa: E402
from graph_modi.evaluation.static_eval import evaluate_static_oracle  # noqa: E402
from graph_modi.models.perceiver_readout import (  # noqa: E402
    PerceiverTEAGLM,
    save_perceiver_checkpoint,
)
from graph_modi.models.tea_glm import TEAGLMBackend  # noqa: E402
from graph_modi.training import pretrain_graph_encoder  # noqa: E402


def _apply_gpu_memory_cap(fraction: float) -> None:
    import torch

    if fraction >= 1.0 or not torch.cuda.is_available():
        return
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(f"[perceiver-readout] capped this process to {fraction:.0%} of GPU memory", flush=True)


def _run_gnn_pretrain(config, force: bool) -> None:
    gnn_dir = config.output_dir / "gnn"
    if (gnn_dir / "gnn.pt").exists() and not force:
        print(
            f"[perceiver-readout] GNN checkpoint already exists at {gnn_dir}, skipping pretrain "
            "(pass --force-gnn-pretrain to redo)",
            flush=True,
        )
        return
    _, gnn, tensorizer = _neural_components(config)
    examples = _static_examples(load_static_tuples(_static_path(config, "train")))
    training = config.section("training")
    print(
        f"[perceiver-readout] pretraining GNN from scratch: {len(examples)} examples, "
        f"epochs={training.get('gnn_epochs')}",
        flush=True,
    )
    result = pretrain_graph_encoder(
        gnn,
        tensorizer,
        examples,
        output_dir=gnn_dir,
        epochs=int(training.get("gnn_epochs", 5)),
        batch_size=int(training.get("gnn_batch_size", 64)),
        learning_rate=float(training.get("gnn_learning_rate", 0.002)),
        seed=int(training.get("seed", config.seed)),
    )
    print(f"[perceiver-readout] GNN pretrain done: {result}", flush=True)


def _build_fresh_model(config) -> PerceiverTEAGLM:
    import torch

    model_config = config.section("model")
    perceiver_config = config.section("perceiver_readout")
    from graph_modi.models.tea_glm import GraphSAGEConfig, NodeTensorizerConfig

    tensor_config = NodeTensorizerConfig(feature_dim=int(model_config.get("node_feature_size", 256)))
    graph_config = GraphSAGEConfig(
        input_dim=tensor_config.feature_dim,
        hidden_dim=int(model_config.get("graph_hidden_size", 256)),
        output_dim=int(model_config.get("graph_hidden_size", 256)),
        num_layers=int(model_config.get("graph_layers", 4)),
        aggregation=str(model_config.get("graph_aggregation", "mean")),
    )
    dtype_name = str(model_config.get("torch_dtype", "float32"))
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise ValueError(f"Unknown torch dtype: {dtype_name}")
    model = PerceiverTEAGLM.from_scratch(
        str(model_config["lm_name"]),
        gnn_config=graph_config,
        tensorizer_config=tensor_config,
        num_queries=int(perceiver_config.get("num_queries", 10)),
        num_heads=int(perceiver_config.get("num_heads", 4)),
        num_layers=int(perceiver_config.get("num_layers", 2)),
        ff_hidden_dim=int(perceiver_config.get("ff_hidden_dim", 512)),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        torch_dtype=dtype,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def _train_resampler(model: PerceiverTEAGLM, config, epochs: int, batch_size: int, learning_rate: float) -> None:
    import torch

    train_tuples = load_static_tuples(_static_path(config, "train"))
    rng = random.Random(int(config.seed))
    rng.shuffle(train_tuples)
    train_examples = _static_examples(train_tuples)
    print(f"[perceiver-readout] {len(train_examples)} training examples", flush=True)

    optimizer = torch.optim.AdamW(model.resampler.parameters(), lr=learning_rate)
    steps_per_epoch = (len(train_examples) + batch_size - 1) // batch_size
    for epoch in range(epochs):
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
            torch.nn.utils.clip_grad_norm_(model.resampler.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            if step % max(1, steps_per_epoch // 10) == 0:
                print(
                    f"[perceiver-readout] epoch {epoch + 1}/{epochs} step {step}/{steps_per_epoch} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )
        mean_loss = epoch_loss / max(1, steps_per_epoch)
        print(f"[perceiver-readout] epoch {epoch + 1}/{epochs} done mean_loss={mean_loss:.4f}", flush=True)


def _save_checkpoint(model: PerceiverTEAGLM, config) -> None:
    import torch
    from dataclasses import asdict

    checkpoint_dir = Path(config.output_dir) / "perceiver_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.gnn.state_dict(), checkpoint_dir / "gnn.pt")
    metadata = {
        "format": "graph-modi-gnn-v1",
        "gnn_config": asdict(model.gnn.config),
        "tensorizer_config": asdict(model.tensorizer.config),
    }
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_perceiver_checkpoint(model.resampler, checkpoint_dir / "resampler")
    print(f"[perceiver-readout] saved checkpoint to {checkpoint_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.5)
    parser.add_argument("--force-gnn-pretrain", action="store_true")
    parser.add_argument("--resampler-epochs", type=int, default=None)
    parser.add_argument("--resampler-batch-size", type=int, default=8)
    parser.add_argument("--resampler-learning-rate", type=float, default=5e-4)
    args = parser.parse_args()

    _apply_gpu_memory_cap(args.gpu_memory_fraction)

    config = load_config(Path(args.config))

    print("[perceiver-readout] phase 1: GNN pretrain from scratch", flush=True)
    _run_gnn_pretrain(config, force=args.force_gnn_pretrain)

    print("[perceiver-readout] phase 2: building fresh model (frozen LM only)", flush=True)
    model = _build_fresh_model(config)
    _load_local_gnn(model, config)  # loads phase-1 GNN weights, freezes GNN

    resampler_epochs = args.resampler_epochs or int(config.section("training").get("epochs", 10))
    print("[perceiver-readout] phase 2: training resampler", flush=True)
    _train_resampler(
        model,
        config,
        epochs=resampler_epochs,
        batch_size=args.resampler_batch_size,
        learning_rate=args.resampler_learning_rate,
    )

    _save_checkpoint(model, config)

    print("[perceiver-readout] evaluating on static exact-uniform validation/test...", flush=True)
    max_new_tokens = int(config.section("evaluation").get("max_new_tokens", 96))
    backend = TEAGLMBackend(model, max_new_tokens=max_new_tokens)
    for split in ("validation", "test"):
        tuples = load_static_tuples(_static_path(config, split))
        result = evaluate_static_oracle(tuples, backend, batch_size=16, progress=False)
        print(f"\n=== {split} ===")
        print("overall_accuracy:", round(result["overall_accuracy"], 4))
        print("passed:", result["passed"])
        for task in sorted(result["task_accuracy"]):
            print(f"  {task}: {round(result['task_accuracy'][task], 4)}")
        out_path = Path(config.output_dir) / f"perceiver_static_eval_{split}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"[perceiver-readout] wrote {out_path}")


if __name__ == "__main__":
    main()
