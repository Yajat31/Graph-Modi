"""Command-line entry points for reproducible GraphModi experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from graph_modi.config import ExperimentConfig, load_config
from graph_modi.data.multiturn import (
    DEFAULT_GRAPH_DEGREE,
    DEFAULT_LINE_COUNT,
    DEFAULT_REWIRE_PROBABILITY,
    DISTRIBUTION,
    generate_dataset,
    load_sessions,
    save_sessions,
)
from graph_modi.data.validation import audit_sessions
from graph_modi.evaluation.runner import CONDITIONS, evaluate_sessions
from graph_modi.graph.solvers import render_question
from graph_modi.models.base import GraphBackend, SymbolicMockBackend
from graph_modi.pipeline.multiturn import materialize_states, run_session
from graph_modi.schema import Session
from graph_modi.training import (
    ProjectorTrainingConfig,
    TrainingExample,
    pretrain_graph_encoder,
    run_mock_training,
    train_projector,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _data_path(config: ExperimentConfig, split: str) -> Path:
    return config.data_dir / f"{split}.jsonl"


def _audit_path(config: ExperimentConfig) -> Path:
    return config.data_dir / "audit.json"


def _generate(config: ExperimentConfig) -> dict[str, Any]:
    data = config.section("data")
    counts = {
        "train": int(data.get("train_sessions", 0)),
        "validation": int(data.get("validation_sessions", 0)),
        "test": int(data.get("test_sessions", 0)),
    }
    dataset = generate_dataset(
        counts=counts,
        seed=config.seed,
        node_count_min=int(data.get("node_count_min", 15)),
        node_count_max=int(data.get("node_count_max", 30)),
        turns=[int(value) for value in data.get("turns", [1, 2, 3, 5])],
        reasoning_types=[
            str(value)
            for value in data.get(
                "reasoning_types",
                ["shortest_path", "filtered_neighbor_count"],
            )
        ],
        distribution=str(data.get("distribution", DISTRIBUTION)),
        degree=int(data.get("graph_degree", DEFAULT_GRAPH_DEGREE)),
        rewire_probability=float(data.get("rewire_probability", DEFAULT_REWIRE_PROBABILITY)),
        line_count=int(data.get("line_count", DEFAULT_LINE_COUNT)),
    )
    report: dict[str, Any] = {}
    for split, sessions in dataset.items():
        save_sessions(_data_path(config, split), sessions)
        report[split] = audit_sessions(sessions)
        if not report[split]["valid"]:
            raise RuntimeError(f"Generated invalid {split} data: {report[split]['failures']}")
    _write_json(_audit_path(config), report)
    return report


def _ensure_data(config: ExperimentConfig) -> None:
    if not _data_path(config, "test").exists():
        _generate(config)


def _examples(sessions: list[Session]) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for session in sessions:
        for turn, state in zip(
            session.turns,
            materialize_states(session),
            strict=True,
        ):
            examples.append(
                TrainingExample(
                    graph=state.after,
                    prompt=render_question(turn.query, state.after),
                    answer=turn.gold_answer,
                    example_id=f"{session.session_id}-turn-{turn.turn_index}",
                    split=session.split,
                    metadata={
                        "reasoning_type": turn.query.reasoning_type.value,
                        "graph_fingerprint": turn.after_fingerprint,
                        "source_id": turn.query.source,
                        "target_id": turn.query.target,
                    },
                )
            )
    return examples


def _neural_components(config: ExperimentConfig) -> tuple[Any, Any, Any]:
    try:
        from graph_modi.models.tea_glm import (
            DeterministicNodeTensorizer,
            GraphSAGEConfig,
            GraphSAGEEncoder,
            NodeTensorizerConfig,
        )
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Install graph-modi[tea] for neural commands") from error
    model = config.section("model")
    tensor_config = NodeTensorizerConfig(feature_dim=int(model.get("node_feature_size", 256)))
    graph_config = GraphSAGEConfig(
        input_dim=tensor_config.feature_dim,
        hidden_dim=int(model.get("graph_hidden_size", 256)),
        output_dim=int(model.get("graph_hidden_size", 256)),
        num_layers=int(model.get("graph_layers", 4)),
        aggregation=str(model.get("graph_aggregation", "mean")),
    )
    return graph_config, GraphSAGEEncoder(graph_config), DeterministicNodeTensorizer(tensor_config)


def _load_local_gnn(model: Any, config: ExperimentConfig) -> None:
    model_config = config.section("model")
    path = Path(model_config.get("gnn_checkpoint", config.output_dir / "gnn" / "gnn.pt"))
    metadata_path = path.with_name("metadata.json")
    if not path.exists():
        raise FileNotFoundError(f"GNN checkpoint not found: {path}. Run pretrain-gnn first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "graph-modi-gnn-v1":
        raise ValueError(f"Unsupported GNN checkpoint metadata: {metadata_path}")
    if metadata.get("gnn_config") != asdict(model.gnn.config):
        raise ValueError("GNN checkpoint dimensions do not match this model")
    if metadata.get("tensorizer_config") != asdict(model.tensorizer.config):
        raise ValueError("GNN checkpoint tensorizer does not match this model")
    import torch

    model.gnn.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True),
        strict=True,
    )
    model.set_gnn_frozen(True)


def _tea_model(config: ExperimentConfig) -> Any:
    import torch

    from graph_modi.models.tea_glm import (
        TEAGLM,
        DeterministicNodeTensorizer,
        GraphSAGEConfig,
        NodeTensorizerConfig,
        TEAGLMConfig,
    )

    model = config.section("model")
    tensor_config = NodeTensorizerConfig(feature_dim=int(model.get("node_feature_size", 256)))
    graph_config = GraphSAGEConfig(
        input_dim=tensor_config.feature_dim,
        hidden_dim=int(model.get("graph_hidden_size", 256)),
        output_dim=int(model.get("graph_hidden_size", 256)),
        num_layers=int(model.get("graph_layers", 4)),
        aggregation=str(model.get("graph_aggregation", "mean")),
    )
    dtype_name = str(model.get("torch_dtype", "float32"))
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise ValueError(f"Unknown torch dtype: {dtype_name}")
    tea_model = TEAGLM.from_pretrained(
        str(model["lm_name"]),
        gnn_config=graph_config,
        tensorizer_config=tensor_config,
        config=TEAGLMConfig(freeze_gnn=True),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        torch_dtype=dtype,
        prefix_tokens=int(model.get("prefix_tokens", 8)),
        projector_hidden_dim=int(model.get("projector_hidden_size", 512)),
        projector_num_layers=int(model.get("projector_num_layers", 1)),
    )
    if not isinstance(tea_model.tensorizer, DeterministicNodeTensorizer):
        raise TypeError("Unexpected graph tensorizer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tea_model.to(device)


def command_generate(config: ExperimentConfig, _args: argparse.Namespace) -> None:
    report = _generate(config)
    print(json.dumps(report, indent=2))


def command_pretrain(config: ExperimentConfig, _args: argparse.Namespace) -> None:
    _ensure_data(config)
    model_config = config.section("model")
    if model_config.get("backend", "mock") == "mock":
        path = run_mock_training(config.output_dir / "gnn", seed=config.seed)
        print(path)
        return
    _, gnn, tensorizer = _neural_components(config)
    examples = _examples(load_sessions(_data_path(config, "train")))
    training = config.section("training")
    result = pretrain_graph_encoder(
        gnn,
        tensorizer,
        examples,
        output_dir=config.output_dir / "gnn",
        epochs=int(training.get("gnn_epochs", training.get("epochs", 5))),
        batch_size=int(training.get("gnn_batch_size", training.get("batch_size", 4))),
        learning_rate=float(training.get("gnn_learning_rate", 0.002)),
        seed=int(training.get("seed", config.seed)),
    )
    print(json.dumps(asdict(result), indent=2, default=str))


def command_train(config: ExperimentConfig, _args: argparse.Namespace) -> None:
    _ensure_data(config)
    model_config = config.section("model")
    if model_config.get("backend", "mock") == "mock":
        path = run_mock_training(config.output_dir / "projector", seed=config.seed)
        print(path)
        return
    model = _tea_model(config)
    _load_local_gnn(model, config)
    training = config.section("training")
    result = train_projector(
        model,
        _examples(load_sessions(_data_path(config, "train"))),
        ProjectorTrainingConfig(
            output_dir=config.output_dir / "projector",
            epochs=int(training.get("epochs", 5)),
            batch_size=int(training.get("batch_size", 1)),
            gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
            learning_rate=float(training.get("learning_rate", 5e-4)),
            weight_decay=float(training.get("weight_decay", 0.0)),
            mixed_precision=str(training.get("mixed_precision", "no")),
            save_every_steps=int(training.get("save_every_steps", 500)),
            seed=int(training.get("seed", config.seed)),
            resume_from=training.get("resume_from"),
        ),
    )
    print(json.dumps(asdict(result), indent=2, default=str))


def _backend(config: ExperimentConfig) -> GraphBackend:
    model_config = config.section("model")
    if model_config.get("backend", "mock") == "mock":
        return SymbolicMockBackend()
    from graph_modi.models.tea_glm import TEAGLMBackend, load_external_component

    model = _tea_model(config)
    checkpoint = Path(
        model_config.get(
            "checkpoint",
            config.output_dir / "projector" / "checkpoint-final",
        )
    )
    metadata = checkpoint / "metadata.json"
    load_external_component(
        model,
        component="gnn",
        weights_path=checkpoint / "gnn.pt",
        metadata_path=metadata,
    )
    load_external_component(
        model,
        component="projector",
        weights_path=checkpoint / "projector.pt",
        metadata_path=metadata,
    )
    return TEAGLMBackend(
        model,
        max_new_tokens=int(config.section("evaluation").get("max_new_tokens", 32)),
    )


def command_evaluate(config: ExperimentConfig, _args: argparse.Namespace) -> None:
    _ensure_data(config)
    evaluation = config.section("evaluation")
    conditions = [str(value) for value in evaluation.get("conditions", CONDITIONS)]
    batch_size = int(evaluation.get("batch_size", 16))
    result: dict[str, Any] = {}
    backend = _backend(config)
    for split in evaluation.get("splits", ["test"]):
        sessions = load_sessions(_data_path(config, str(split)))
        print(f"[evaluate] split={split!r}", flush=True)
        result[str(split)] = evaluate_sessions(
            sessions,
            backend,
            conditions=conditions,
            batch_size=batch_size,
            progress=True,
        )
    path = config.output_dir / "evaluation.json"
    _write_json(path, result)
    print(json.dumps({key: value["summary"] for key, value in result.items()}, indent=2))


def command_session(config: ExperimentConfig, args: argparse.Namespace) -> None:
    sessions = load_sessions(args.session)
    if args.index < 0 or args.index >= len(sessions):
        raise IndexError(f"Session index {args.index} is out of range")
    session = sessions[args.index]
    outputs = run_session(
        session,
        _backend(config),
        predicted_edits=args.predicted_edits,
        include_history=args.include_history,
    )
    print(json.dumps([asdict(output) for output in outputs], indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graph-modi")
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = {
        "generate": command_generate,
        "pretrain-gnn": command_pretrain,
        "train-projector": command_train,
        "evaluate": command_evaluate,
    }
    for name, function in commands.items():
        child = subparsers.add_parser(name)
        child.add_argument("--config", required=True, type=Path)
        child.set_defaults(handler=function)
    session = subparsers.add_parser("run-session")
    session.add_argument("--config", required=True, type=Path)
    session.add_argument("--session", required=True, type=Path)
    session.add_argument("--index", type=int, default=0)
    session.add_argument("--predicted-edits", action="store_true")
    session.add_argument("--include-history", action="store_true")
    session.set_defaults(handler=command_session)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    args.handler(config, args)


if __name__ == "__main__":
    main()
