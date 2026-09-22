#!/usr/bin/env python3
"""Two-phase from-scratch training for the MHLA-style multi-aspect readout
(src/graph_modi/models/aspect_readout.py): pretrain a fresh GNN, freeze it,
then jointly train the (also fresh) summary projector + AspectHeads +
TaskConditionedRouter against the LM loss on the static training corpus.

Decided from-scratch rather than reusing v2's already-trained frozen GNN
(as the earlier multi-neighbor-readout probe did): a GNN already optimized
for v2's single-pooled-vector readout has no reason to produce embeddings
whose linear combinations separate into distinct "aspects". If this
from-scratch attempt doesn't work either, the next fallback is changing the
GNN's own architecture (e.g. separate per-aspect message-passing streams),
not attempted here.

Not wired into cli.py / the main pipeline -- a throwaway exploratory script,
same status as explore_multi_neighbor_readout.py.

Usage (on the remote box, matching the main pipeline's env):
  CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/lmw/hf_cache/huggingface \
    .venv/bin/python scripts/train_aspect_readout_from_scratch.py \
    --config configs/v2_clegr_extended_aspect_readout.yaml \
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
from graph_modi.models.aspect_readout import (  # noqa: E402
    AspectTEAGLM,
    save_aspect_checkpoint,
)
from graph_modi.models.tea_glm import TEAGLMBackend  # noqa: E402
from graph_modi.training import pretrain_graph_encoder  # noqa: E402


def _apply_gpu_memory_cap(fraction: float) -> None:
    """Cap this process to at most `fraction` of the visible GPU's memory,
    so we never crowd out other users' jobs sharing GPU 1. Must run before
    any CUDA allocation."""
    import torch

    if fraction >= 1.0 or not torch.cuda.is_available():
        return
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(f"[aspect-readout] capped this process to {fraction:.0%} of GPU memory", flush=True)


def _run_gnn_pretrain(config, force: bool) -> None:
    gnn_dir = config.output_dir / "gnn"
    if (gnn_dir / "gnn.pt").exists() and not force:
        print(f"[aspect-readout] GNN checkpoint already exists at {gnn_dir}, skipping pretrain "
              "(pass --force-gnn-pretrain to redo)", flush=True)
        return
    _, gnn, tensorizer = _neural_components(config)
    examples = _static_examples(load_static_tuples(_static_path(config, "train")))
    training = config.section("training")
    print(f"[aspect-readout] pretraining GNN from scratch: {len(examples)} examples, "
          f"epochs={training.get('gnn_epochs')}", flush=True)
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
    print(f"[aspect-readout] GNN pretrain done: {result}", flush=True)


def _build_fresh_model(config) -> AspectTEAGLM:
    import torch

    model_config = config.section("model")
    aspect_config = config.section("aspect_readout")
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
    model = AspectTEAGLM.from_scratch(
        str(model_config["lm_name"]),
        gnn_config=graph_config,
        tensorizer_config=tensor_config,
        prefix_tokens=int(model_config.get("prefix_tokens", 8)),
        projector_hidden_dim=int(model_config.get("projector_hidden_size", 512)),
        projector_num_layers=int(model_config.get("projector_num_layers", 1)),
        num_aspects=int(aspect_config.get("num_aspects", 6)),
        num_selected=int(aspect_config.get("num_selected", 3)),
        head_hidden_dim=int(aspect_config.get("head_hidden_dim", 512)),
        router_hidden_dim=int(aspect_config.get("router_hidden_dim", 64)),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        torch_dtype=dtype,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def _train_adapter(model: AspectTEAGLM, config, epochs: int, batch_size: int, learning_rate: float) -> None:
    import torch

    train_tuples = load_static_tuples(_static_path(config, "train"))
    rng = random.Random(int(config.seed))
    rng.shuffle(train_tuples)
    train_examples = _static_examples(train_tuples)
    print(f"[aspect-readout] {len(train_examples)} training examples", flush=True)

    trainable_params = [
        *model.projector.parameters(),
        *model.aspect_heads.parameters(),
        *model.router.parameters(),
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
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
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            if step % max(1, steps_per_epoch // 10) == 0:
                print(
                    f"[aspect-readout] epoch {epoch + 1}/{epochs} step {step}/{steps_per_epoch} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )
        mean_loss = epoch_loss / max(1, steps_per_epoch)
        print(f"[aspect-readout] epoch {epoch + 1}/{epochs} done mean_loss={mean_loss:.4f}", flush=True)


def _save_checkpoint(model: AspectTEAGLM, config) -> None:
    import torch
    from graph_modi.models.tea_glm import model_checkpoint_metadata, save_checkpoint_metadata

    checkpoint_dir = Path(config.output_dir) / "aspect_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.gnn.state_dict(), checkpoint_dir / "gnn.pt")
    torch.save(model.projector.state_dict(), checkpoint_dir / "projector.pt")
    # Use the same metadata format/fields (incl. state_sha256 hashes) that
    # load_external_component()/assert_checkpoint_compatible() require --
    # matches what scripts/run_dynamic_eval_variant.py's --variant aspect
    # path expects when loading this checkpoint later.
    metadata = model_checkpoint_metadata(
        model, training_config={}, data_split={}, data_sha256="", step=0
    )
    save_checkpoint_metadata(checkpoint_dir / "metadata.json", metadata)
    save_aspect_checkpoint(model.aspect_heads, model.router, checkpoint_dir / "aspect_adapter")
    print(f"[aspect-readout] saved checkpoint to {checkpoint_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.5)
    parser.add_argument("--force-gnn-pretrain", action="store_true")
    parser.add_argument("--adapter-epochs", type=int, default=None)
    parser.add_argument("--adapter-batch-size", type=int, default=8)
    parser.add_argument("--adapter-learning-rate", type=float, default=5e-4)
    args = parser.parse_args()

    _apply_gpu_memory_cap(args.gpu_memory_fraction)

    config = load_config(Path(args.config))

    print("[aspect-readout] phase 1: GNN pretrain from scratch", flush=True)
    _run_gnn_pretrain(config, force=args.force_gnn_pretrain)

    print("[aspect-readout] phase 2: building fresh model (frozen LM only)", flush=True)
    model = _build_fresh_model(config)
    _load_local_gnn(model, config)  # loads phase-1 GNN weights, freezes GNN

    adapter_epochs = args.adapter_epochs or int(config.section("training").get("epochs", 10))
    print("[aspect-readout] phase 2: training projector + aspect_heads + router jointly", flush=True)
    _train_adapter(
        model,
        config,
        epochs=adapter_epochs,
        batch_size=args.adapter_batch_size,
        learning_rate=args.adapter_learning_rate,
    )

    _save_checkpoint(model, config)

    print("[aspect-readout] evaluating on static exact-uniform validation/test...", flush=True)
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
        out_path = Path(config.output_dir) / f"aspect_static_eval_{split}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"[aspect-readout] wrote {out_path}")


if __name__ == "__main__":
    main()
