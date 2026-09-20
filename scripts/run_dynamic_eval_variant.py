#!/usr/bin/env python3
"""Full dynamic (multi-turn) evaluate for a TEA/GraphToken checkpoint, in
either its pooled (v2, current) readout or the multi-neighbor (v3,
exploratory) readout from src/graph_modi/models/multi_neighbor_readout.py.

Mirrors cli.py's command_evaluate (same evaluate_sessions call, same
conditions/batch_size/splits config keys) but adds the multi_neighbor path,
which the main _backend()/command_evaluate do not support -- kept as a
separate script rather than touching the shared, validated cli.py path.

Usage (on the remote box, matching the main pipeline's env):
  CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/lmw/hf_cache/huggingface \
    .venv/bin/python scripts/run_dynamic_eval_variant.py \
    --config configs/v2_clegr_extended_exact2x_tea.yaml --variant pooled

  CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/lmw/hf_cache/huggingface \
    .venv/bin/python scripts/run_dynamic_eval_variant.py \
    --config configs/v2_clegr_extended_exact2x_tea.yaml --variant multi_neighbor \
    --neighbor-checkpoint outputs/v2_clegr_extended_exact2x_tea/multi_neighbor_projector
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graph_modi.cli import _data_path, _ensure_data, _tea_model, _write_json  # noqa: E402
from graph_modi.config import ExperimentConfig, load_config  # noqa: E402
from graph_modi.data.multiturn import load_sessions  # noqa: E402
from graph_modi.evaluation.runner import CONDITIONS, evaluate_sessions  # noqa: E402
from graph_modi.models.multi_neighbor_readout import (  # noqa: E402
    MultiNeighborTEAGLM,
    load_neighbor_checkpoint,
)
from graph_modi.models.tea_glm import TEAGLMBackend, load_external_component  # noqa: E402


def _build_backend(config: ExperimentConfig, variant: str, neighbor_checkpoint: str | None) -> Any:
    model_config = config.section("model")
    model = _tea_model(config)
    checkpoint_dir = Path(
        model_config.get("checkpoint", config.output_dir / "projector" / "checkpoint-final")
    )
    metadata_path = checkpoint_dir / "metadata.json"
    load_external_component(
        model, component="gnn", weights_path=checkpoint_dir / "gnn.pt", metadata_path=metadata_path
    )
    load_external_component(
        model,
        component="projector",
        weights_path=checkpoint_dir / "projector.pt",
        metadata_path=metadata_path,
    )

    backend_cls = TEAGLMBackend
    if str(model_config.get("architecture", "tea")) == "graph_token":
        from graph_modi.models.graph_token import GraphTokenBackend

        backend_cls = GraphTokenBackend

    max_new_tokens = int(config.section("evaluation").get("max_new_tokens", 96))

    if variant == "pooled":
        return backend_cls(model, max_new_tokens=max_new_tokens)
    if variant == "multi_neighbor":
        if not neighbor_checkpoint:
            raise ValueError("--neighbor-checkpoint is required for --variant multi_neighbor")
        neighbor_projector = load_neighbor_checkpoint(neighbor_checkpoint).to(model.device)
        multi_model = MultiNeighborTEAGLM(
            base=model,
            neighbor_projector=neighbor_projector,
            neighbor_slots=neighbor_projector.config.neighbor_slots,
        )
        return backend_cls(multi_model, max_new_tokens=max_new_tokens)
    raise ValueError(f"Unknown variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=["pooled", "multi_neighbor"], required=True)
    parser.add_argument("--neighbor-checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument(
        "--output-suffix", default=None, help="output filename is evaluation<suffix>.json"
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    _ensure_data(config)
    evaluation = config.section("evaluation")
    conditions = [str(value) for value in evaluation.get("conditions", CONDITIONS)]
    batch_size = args.batch_size or int(evaluation.get("batch_size", 16))
    splits = args.splits or list(evaluation.get("splits", ["test"]))

    print(f"[dynamic-eval] variant={args.variant} config={args.config} splits={splits}", flush=True)
    backend = _build_backend(config, args.variant, args.neighbor_checkpoint)

    result: dict[str, Any] = {}
    for split in splits:
        print(f"[dynamic-eval] loading sessions for split={split}", flush=True)
        sessions = load_sessions(_data_path(config, str(split)))
        print(f"[dynamic-eval] evaluating split={split} sessions={len(sessions)}", flush=True)
        result[str(split)] = evaluate_sessions(
            sessions, backend, conditions=conditions, batch_size=batch_size, progress=True
        )

    suffix = args.output_suffix if args.output_suffix is not None else f"_{args.variant}"
    path = Path(config.output_dir) / f"evaluation{suffix}.json"
    _write_json(path, result)
    print(f"[dynamic-eval] wrote {path}", flush=True)
    print(json.dumps({key: value["summary"] for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
