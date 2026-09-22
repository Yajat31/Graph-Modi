#!/usr/bin/env python3
"""Train the aspect-readout architecture (src/graph_modi/models/aspect_readout.py
-- MHLA-style sparse-routed heads, fresh GNN, no hand-fed task/neighbor
signal) from scratch on the REAL CLEGR dataset, for a fair, apples-to-apples
comparison against scripts/train_clegr_real.py's TEA/GraphToken runs: same
data, same GNN-pretrain regime, same epochs/batch size, only the readout
architecture differs.

Two-phase, same shape as train_clegr_real.py and
train_aspect_readout_from_scratch.py:
  Phase 1 -- pretrain a fresh GNN (pretrain_graph_encoder) on the CLEGR
             training split. Uses its own checkpoint dir (not shared with
             TEA/GraphToken's, since aspect-readout's summary projector
             also trains from scratch here, unlike the synthetic-benchmark
             version which reused an already-trained v2 projector).
  Phase 2 -- freeze that GNN, jointly train the summary projector +
             AspectHeads + TaskConditionedRouter (custom loop, mirroring
             train_aspect_readout_from_scratch.py's `_train_adapter`).
  Phase 3 -- evaluate on the CLEGR test split, same format-inferring scorer
             as train_clegr_real.py, bucketed by clegr_type/group/subgroup.

Usage:
  CUDA_VISIBLE_DEVICES=<mig-uuid> HF_HOME=... .venv/bin/python \
    scripts/train_aspect_readout_clegr_real.py \
    --gpu-memory-fraction 0.9 --data-dir datasets/clegr_real \
    --output-dir outputs/clegr_real_aspect
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graph_modi.models.aspect_readout import AspectTEAGLM, save_aspect_checkpoint  # noqa: E402
from graph_modi.schema import AttributedGraph  # noqa: E402
from graph_modi.training import TrainingExample, pretrain_graph_encoder  # noqa: E402


def _apply_gpu_memory_cap(fraction: float) -> None:
    import torch

    if fraction >= 1.0 or not torch.cuda.is_available():
        return
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(f"[clegr-aspect] capped this process to {fraction:.0%} of GPU memory", flush=True)


def _load_examples(path: Path) -> list[TrainingExample]:
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            examples.append(
                TrainingExample(
                    graph=AttributedGraph.from_dict(record["graph"]),
                    prompt=record["prompt"],
                    answer=record["answer"],
                    example_id=record["example_id"],
                    split=record["split"],
                    metadata=record["metadata"],
                )
            )
    return examples


def _build_fresh_gnn(node_feature_size: int, graph_hidden_size: int, graph_layers: int, graph_aggregation: str):
    from graph_modi.models.tea_glm import GraphSAGEConfig, GraphSAGEEncoder, NodeTensorizerConfig

    tensor_config = NodeTensorizerConfig(feature_dim=node_feature_size)
    graph_config = GraphSAGEConfig(
        input_dim=tensor_config.feature_dim,
        hidden_dim=graph_hidden_size,
        output_dim=graph_hidden_size,
        num_layers=graph_layers,
        aggregation=graph_aggregation,
    )
    return graph_config, GraphSAGEEncoder(graph_config), tensor_config


def _run_gnn_pretrain(train_examples, gnn_dir: Path, args) -> None:
    from graph_modi.models.tea_glm import DeterministicNodeTensorizer

    if (gnn_dir / "gnn.pt").exists() and not args.force_gnn_pretrain:
        print(f"[clegr-aspect] GNN checkpoint already exists at {gnn_dir}, skipping pretrain", flush=True)
        return
    _, gnn, tensor_config = _build_fresh_gnn(
        args.node_feature_size, args.graph_hidden_size, args.graph_layers, args.graph_aggregation
    )
    tensorizer = DeterministicNodeTensorizer(tensor_config)
    print(f"[clegr-aspect] pretraining GNN from scratch: {len(train_examples)} examples, epochs={args.gnn_epochs}", flush=True)
    result = pretrain_graph_encoder(
        gnn,
        tensorizer,
        train_examples,
        output_dir=gnn_dir,
        epochs=args.gnn_epochs,
        batch_size=args.gnn_batch_size,
        learning_rate=args.gnn_learning_rate,
        seed=args.seed,
    )
    print(f"[clegr-aspect] GNN pretrain done: {result}", flush=True)


def _build_model(args) -> AspectTEAGLM:
    import torch

    graph_config, _, tensor_config = _build_fresh_gnn(
        args.node_feature_size, args.graph_hidden_size, args.graph_layers, args.graph_aggregation
    )
    model = AspectTEAGLM.from_scratch(
        args.lm_name,
        gnn_config=graph_config,
        tensorizer_config=tensor_config,
        prefix_tokens=args.prefix_tokens,
        projector_hidden_dim=args.projector_hidden_size,
        projector_num_layers=args.projector_num_layers,
        num_aspects=args.num_aspects,
        num_selected=args.num_selected,
        head_hidden_dim=args.head_hidden_size,
        router_hidden_dim=args.router_hidden_size,
        torch_dtype=torch.bfloat16,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def _load_gnn_checkpoint(model: AspectTEAGLM, gnn_dir: Path) -> None:
    import torch

    model.gnn.load_state_dict(torch.load(gnn_dir / "gnn.pt", map_location="cpu", weights_only=True), strict=True)
    model.set_gnn_frozen(True)


def _train_adapter(model: AspectTEAGLM, train_examples, epochs: int, batch_size: int, learning_rate: float) -> None:
    import torch

    rng = random.Random(42)
    examples = list(train_examples)
    rng.shuffle(examples)

    trainable_params = [
        *model.projector.parameters(),
        *model.aspect_heads.parameters(),
        *model.router.parameters(),
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    steps_per_epoch = (len(examples) + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * epochs
    print(f"[clegr-aspect] training: examples={len(examples)} steps/epoch={steps_per_epoch} epochs={epochs} total_steps={total_steps}", flush=True)
    global_step = 0
    for epoch in range(epochs):
        rng.shuffle(examples)
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            batch = examples[step * batch_size : (step + 1) * batch_size]
            if not batch:
                continue
            output = model(
                graphs=[example.graph for example in batch],
                prompts=[example.prompt for example in batch],
                answers=[example.answer for example in batch],
                source_ids=[None for _ in batch],
                target_ids=[None for _ in batch],
                hops=[None for _ in batch],
                reasoning_types=[None for _ in batch],
            )
            loss = output.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            global_step += 1
            if global_step % max(1, steps_per_epoch // 20) == 0:
                print(f"[clegr-aspect] step {global_step}/{total_steps} (epoch {epoch + 1}/{epochs}) loss={loss.item():.4f}", flush=True)
        print(f"[clegr-aspect] epoch {epoch + 1}/{epochs} done mean_loss={epoch_loss / max(1, steps_per_epoch):.4f}", flush=True)


_BOOL_TRUE = {"true", "yes", "1"}
_BOOL_FALSE = {"false", "no", "0"}


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s,.-]", "", text.strip().lower())


def _score(gold: str, pred: str) -> bool:
    gold_norm, pred_norm = _normalize(gold), _normalize(pred)
    if gold_norm in _BOOL_TRUE or gold_norm in _BOOL_FALSE:
        pred_bool = pred_norm in _BOOL_TRUE
        gold_bool = gold_norm in _BOOL_TRUE
        return pred_bool == gold_bool and (pred_norm in _BOOL_TRUE or pred_norm in _BOOL_FALSE)
    try:
        return abs(float(gold_norm) - float(re.sub(r"[^\d.\-]", "", pred_norm))) < 1e-6
    except ValueError:
        pass
    if "," in gold:
        gold_set = {item.strip().lower() for item in gold.split(",") if item.strip()}
        pred_set = {item.strip().lower() for item in pred.split(",") if item.strip()}
        return gold_set == pred_set
    return gold_norm == pred_norm


def _evaluate(model, test_examples, batch_size: int, max_new_tokens: int) -> dict[str, Any]:
    rows = []
    for offset in range(0, len(test_examples), batch_size):
        batch = test_examples[offset : offset + batch_size]
        predictions = model.generate_batch(
            graphs=[example.graph for example in batch],
            prompts=[example.prompt for example in batch],
            max_new_tokens=max_new_tokens,
        )
        for example, predicted in zip(batch, predictions, strict=True):
            rows.append(
                {
                    "clegr_type": example.metadata.get("clegr_type"),
                    "clegr_group": example.metadata.get("clegr_group"),
                    "clegr_subgroup": example.metadata.get("clegr_subgroup"),
                    "gold": example.answer,
                    "predicted": predicted,
                    "correct": _score(example.answer, predicted),
                }
            )
        if (offset // batch_size) % 20 == 0:
            done = min(offset + batch_size, len(test_examples))
            running_acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"[clegr-aspect] eval {done}/{len(test_examples)} running_acc={running_acc:.4f}", flush=True)

    def _bucket(key: str) -> dict[str, float]:
        groups: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row["correct"])
        return {name: sum(vals) / len(vals) for name, vals in sorted(groups.items())}

    overall = sum(row["correct"] for row in rows) / len(rows)
    return {
        "overall_accuracy": overall,
        "n": len(rows),
        "by_type": _bucket("clegr_type"),
        "by_group": _bucket("clegr_group"),
        "by_subgroup": _bucket("clegr_subgroup"),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="datasets/clegr_real")
    parser.add_argument("--output-dir", default="outputs/clegr_real_aspect")
    parser.add_argument("--gnn-dir", default=None, help="defaults to <output-dir>/gnn")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.9)
    parser.add_argument("--force-gnn-pretrain", action="store_true")
    parser.add_argument("--lm-name", default="/data/lmw/models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--node-feature-size", type=int, default=64)
    parser.add_argument("--graph-hidden-size", type=int, default=4096)
    parser.add_argument("--graph-layers", type=int, default=8)
    parser.add_argument("--graph-aggregation", default="sum")
    parser.add_argument("--prefix-tokens", type=int, default=10)
    parser.add_argument("--projector-hidden-size", type=int, default=512)
    parser.add_argument("--projector-num-layers", type=int, default=1)
    parser.add_argument("--num-aspects", type=int, default=6)
    parser.add_argument("--num-selected", type=int, default=3)
    parser.add_argument("--head-hidden-size", type=int, default=512)
    parser.add_argument("--router-hidden-size", type=int, default=64)
    parser.add_argument("--gnn-epochs", type=int, default=30)
    parser.add_argument("--gnn-batch-size", type=int, default=64)
    parser.add_argument("--gnn-learning-rate", type=float, default=0.002)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _apply_gpu_memory_cap(args.gpu_memory_fraction)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    gnn_dir = Path(args.gnn_dir) if args.gnn_dir else output_dir / "gnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[clegr-aspect] loading examples...", flush=True)
    train_examples = _load_examples(data_dir / "train.jsonl")
    test_examples = _load_examples(data_dir / "test.jsonl")
    print(f"[clegr-aspect] train={len(train_examples)} test={len(test_examples)}", flush=True)

    print("[clegr-aspect] phase 1: GNN pretrain", flush=True)
    _run_gnn_pretrain(train_examples, gnn_dir, args)

    print("[clegr-aspect] phase 2: building model + loading GNN", flush=True)
    model = _build_model(args)
    _load_gnn_checkpoint(model, gnn_dir)

    print("[clegr-aspect] phase 2: training projector + aspect_heads + router", flush=True)
    _train_adapter(model, train_examples, args.epochs, args.batch_size, args.learning_rate)

    checkpoint_dir = output_dir / "aspect_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    import torch
    from dataclasses import asdict

    torch.save(model.gnn.state_dict(), checkpoint_dir / "gnn.pt")
    torch.save(model.projector.state_dict(), checkpoint_dir / "projector.pt")
    from graph_modi.models.tea_glm import model_checkpoint_metadata, save_checkpoint_metadata

    save_checkpoint_metadata(checkpoint_dir / "metadata.json", model_checkpoint_metadata(
        model, training_config={}, data_split={}, data_sha256="", step=0
    ))
    save_aspect_checkpoint(model.aspect_heads, model.router, checkpoint_dir / "aspect_adapter")
    print(f"[clegr-aspect] saved checkpoint to {checkpoint_dir}", flush=True)

    print("[clegr-aspect] phase 3: evaluating on real CLEGR test split", flush=True)
    model.eval()
    with torch.no_grad():
        result = _evaluate(model, test_examples, args.eval_batch_size, args.max_new_tokens)

    print(f"\n=== aspect-readout on real CLEGR test ({result['n']} examples) ===")
    print("overall_accuracy:", round(result["overall_accuracy"], 4))
    print("by_group:", {k: round(v, 4) for k, v in result["by_group"].items()})
    print("by_subgroup:", {k: round(v, 4) for k, v in result["by_subgroup"].items()})
    for type_name, acc in sorted(result["by_type"].items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {type_name}: {round(acc, 4)}")

    out_path = output_dir / "clegr_real_eval.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"[clegr-aspect] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
