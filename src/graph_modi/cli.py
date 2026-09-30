"""Command-line entry points for reproducible GraphModi experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from graph_modi.config import ExperimentConfig, load_config
from graph_modi.data.freeze_v1 import freeze_v1_manifest
from graph_modi.data.multiturn import (
    DEFAULT_GRAPH_DEGREE,
    DEFAULT_LINE_COUNT,
    DEFAULT_REWIRE_PROBABILITY,
    DISTRIBUTION,
    generate_dataset,
    load_sessions,
    save_sessions,
)
from graph_modi.data.v2 import (
    CLEGR_EXTENDED_STATIC_TASKS,
    DISTRIBUTION_V2,
    generate_counterfactual_static_pairs,
    generate_dataset_v2,
    generate_exact_uniform_sessions,
    generate_factorial_sessions,
    generate_static_corpus,
    generate_static_exact_uniform,
    load_static_tuples,
    save_static_tuples,
)
from graph_modi.data import v3 as data_v3
from graph_modi.data.validation import audit_sessions
from graph_modi.evaluation.runner import CONDITIONS, evaluate_sessions
from graph_modi.evaluation.static_eval import evaluate_static_oracle
from graph_modi.graph.serialization import csv_graph_prompt
from graph_modi.graph.solvers import render_question
from graph_modi.models.base import GraphBackend, SymbolicMockBackend
from graph_modi.pipeline.multiturn import materialize_states, run_session
from graph_modi.schema import ReasoningType, Session, StaticQATuple, TopologyFamily
from graph_modi.training import (
    ProjectorTrainingConfig,
    TrainingExample,
    pretrain_graph_encoder,
    pretrain_graph_encoder_contrastive,
    run_mock_training,
    train_projector,
    train_soft_prompt,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _data_path(config: ExperimentConfig, split: str) -> Path:
    return config.data_dir / f"{split}.jsonl"


def _audit_path(config: ExperimentConfig) -> Path:
    return config.data_dir / "audit.json"


def _static_path(config: ExperimentConfig, split: str) -> Path:
    return config.data_dir / f"static_{split}.jsonl"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _experiment_log(config: ExperimentConfig, payload: dict[str, Any]) -> None:
    log_dir = Path("documents/experiments")
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "git_sha": _git_sha(),
        "branch": "test",
        "config": str(config.path),
        "output_dir": str(config.output_dir),
        **payload,
    }
    log_path = log_dir / "log.md"
    phase = payload.get("phase", "run")
    payload_json = json.dumps(payload, sort_keys=True)
    line = f"- `{entry['git_sha'][:8]}` {phase}: {payload_json}\n"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _generate(config: ExperimentConfig, *, progress: bool = True) -> dict[str, Any]:
    data = config.section("data")
    distribution = str(data.get("distribution", DISTRIBUTION))
    if distribution == DISTRIBUTION_V2:
        return _generate_v2(config, progress=progress)
    if distribution == data_v3.DISTRIBUTION_V3:
        return _generate_v3(config, progress=progress)
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
    _experiment_log(config, {"phase": "generate-v1", "report": report})
    return report


def _reasoning_types(data: dict[str, Any], key: str) -> list[ReasoningType] | None:
    """Optional task-list override for v2 generation; None preserves the module defaults."""
    values = data.get(key)
    if not values:
        return None
    return [ReasoningType(str(value)) for value in values]


def _generate_v2(config: ExperimentConfig, *, progress: bool = True) -> dict[str, Any]:
    data = config.section("data")
    dynamic_only = bool(data.get("dynamic_only", False))
    session_layout = str(data.get("session_layout", "default"))
    dynamic_tasks = _reasoning_types(data, "dynamic_tasks")

    if session_layout == "factorial":
        per_cell_cfg = data.get("sessions_per_cell", {})
        if not isinstance(per_cell_cfg, dict) or not per_cell_cfg:
            raise ValueError(
                "data.session_layout=factorial requires data.sessions_per_cell "
                "mapping (e.g. {validation: 2, test: 20})"
            )
        sessions = generate_factorial_sessions(
            sessions_per_cell={str(key): int(value) for key, value in per_cell_cfg.items()},
            seed=config.seed,
            dynamic_tasks=dynamic_tasks,
            progress=progress,
        )
        static: dict[str, list[StaticQATuple]] = {}
    elif session_layout == "exact_uniform":
        eu_cfg = data.get("exact_uniform", {})
        if not isinstance(eu_cfg, dict):
            raise ValueError("data.exact_uniform must be a mapping")
        replicates = {
            "validation": int(eu_cfg.get("validation_replicates", 0)),
            "test": int(eu_cfg.get("test_replicates", 0)),
            "train": int(eu_cfg.get("train_replicates", 0)),
            "ood": int(eu_cfg.get("ood_replicates", 0)),
        }
        replicates = {key: value for key, value in replicates.items() if value > 0}
        if not replicates:
            raise ValueError(
                "data.session_layout=exact_uniform requires at least one of "
                "exact_uniform.{validation,test}_replicates > 0"
            )
        sessions = generate_exact_uniform_sessions(
            replicates_per_split=replicates,
            seed=config.seed,
            dynamic_tasks=dynamic_tasks,
            progress=progress,
        )
        static = {}
    else:
        counts = {
            "train": int(data.get("train_sessions", 0)),
            "validation": int(data.get("validation_sessions", 0)),
            "test": int(data.get("test_sessions", 0)),
            "ood": int(data.get("ood_sessions", 0)),
        }
        counts = {key: value for key, value in counts.items() if value > 0}
        if dynamic_only:
            sessions, static = generate_dataset_v2(
                counts=counts,
                seed=config.seed,
                static_graph_counts={},
                static_tuples_per_graph=int(data.get("static_tuples_per_graph", 12)),
                static_tasks=_reasoning_types(data, "static_tasks"),
                dynamic_tasks=dynamic_tasks,
                counterfactual_static_train=False,
                progress=progress,
            )
        else:
            sessions, static = generate_dataset_v2(
                counts=counts,
                seed=config.seed,
                static_graph_counts={
                    "train": int(data.get("static_train_graphs", 80)),
                    "validation": int(data.get("static_validation_graphs", 10)),
                    "test": int(data.get("static_test_graphs", 20)),
                },
                static_tuples_per_graph=int(data.get("static_tuples_per_graph", 12)),
                static_tasks=_reasoning_types(data, "static_tasks"),
                dynamic_tasks=dynamic_tasks,
                counterfactual_static_train=bool(data.get("counterfactual_static_train", False)),
                progress=progress,
            )

    static_layout = str(data.get("static_layout", "default"))
    if static_layout == "exact_uniform":
        static_tasks = _reasoning_types(data, "static_tasks")
        su_cfg = data.get("static_exact_uniform", {})
        if not isinstance(su_cfg, dict):
            raise ValueError("data.static_exact_uniform must be a mapping")
        eval_replicates = {
            "validation": int(su_cfg.get("validation_replicates", 0)),
            "test": int(su_cfg.get("test_replicates", 0)),
        }
        eval_replicates = {key: value for key, value in eval_replicates.items() if value > 0}
        if not eval_replicates:
            raise ValueError(
                "data.static_layout=exact_uniform requires at least one of "
                "static_exact_uniform.{validation,test}_replicates > 0"
            )
        static = generate_static_exact_uniform(
            replicates_per_split=eval_replicates,
            seed=config.seed,
            tasks=static_tasks,
            progress=progress,
        )
        # Validation/test are exact-uniform; train stays free-sampled, matching
        # generate_exact_uniform_sessions' own train-split convention.
        train_graphs = int(data.get("static_train_graphs", 0))
        if train_graphs > 0:
            if bool(data.get("counterfactual_static_train", False)):
                static["train"] = generate_counterfactual_static_pairs(
                    graph_count=train_graphs,
                    pairs_per_graph=max(1, int(data.get("static_tuples_per_graph", 12)) // 2),
                    seed=config.seed,
                    node_count_range=(16, 32),
                    topologies=(TopologyFamily.WATTS_STROGATZ, TopologyFamily.SBM),
                    tasks=tuple(static_tasks) if static_tasks else CLEGR_EXTENDED_STATIC_TASKS,
                    split="train",
                    progress=progress,
                )
            else:
                train_static = generate_static_corpus(
                    graph_counts={"train": train_graphs},
                    tuples_per_graph=int(data.get("static_tuples_per_graph", 12)),
                    seed=config.seed,
                    node_count_ranges={"train": (16, 32)},
                    topologies=(TopologyFamily.WATTS_STROGATZ, TopologyFamily.SBM),
                    tasks=static_tasks,
                    progress=progress,
                )
                static["train"] = train_static.get("train", [])

    report: dict[str, Any] = {}
    if dynamic_only and _audit_path(config).exists():
        try:
            report = json.loads(_audit_path(config).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {}

    for split, split_sessions in sessions.items():
        save_sessions(_data_path(config, split), split_sessions)
        report[split] = audit_sessions(split_sessions, allow_noop=True)
        if not report[split]["valid"]:
            raise RuntimeError(f"Generated invalid {split} data: {report[split]['failures'][:5]}")
        report[split]["session_layout"] = session_layout

    if static:
        # Either a full generate-v2 run, or static_layout=exact_uniform generated
        # fresh static data on top of a dynamic_only=true session run: save it.
        for split, tuples in static.items():
            save_static_tuples(_static_path(config, split), tuples)
            report[f"static_{split}"] = {"tuples": len(tuples)}
    else:
        # Nothing generated this run; preserve static audit entries, refreshing
        # tuple counts from files already on disk (e.g. a copied static corpus).
        for split in ("train", "validation", "test"):
            path = _static_path(config, split)
            if path.exists():
                report[f"static_{split}"] = {"tuples": sum(1 for _ in path.open(encoding="utf-8"))}

    # Drop empty ood key noise when unused.
    if int(data.get("ood_sessions", 0)) == 0 and "ood" not in sessions:
        report.pop("ood", None)

    _write_json(_audit_path(config), report)
    _experiment_log(
        config,
        {
            "phase": "generate-v2",
            "report": {key: (value if key.startswith("static_") else {"sessions": value.get("sessions"), "turns": value.get("turns"), "valid": value.get("valid")}) for key, value in report.items()},
            "distribution": DISTRIBUTION_V2,
            "session_layout": session_layout,
            "dynamic_only": dynamic_only,
        },
    )
    return report

def _generate_v3(config: ExperimentConfig, *, progress: bool = True) -> dict[str, Any]:
    """Compact CLEGR-style subway graphs, label-balanced (see data/v3.py)."""
    data = config.section("data")
    static_tasks = _reasoning_types(data, "static_tasks") or list(data_v3.STATIC_TASKS)
    dynamic_tasks = _reasoning_types(data, "dynamic_tasks") or list(data_v3.DYNAMIC_TASKS)
    cell_count = len(data_v3.NODE_COUNTS) * len(data_v3.DENSITIES)
    seed = config.seed

    static: dict[str, list[StaticQATuple]] = {}
    static["train"] = data_v3.generate_static_v3(
        split="train",
        graph_count=int(data.get("static_train_graphs", 2000)),
        tasks_per_graph=int(data.get("static_tuples_per_graph", 9)),
        seed=seed,
        sampler=data_v3.BalancedSampler(),
        tasks=static_tasks,
        progress=progress,
    )
    counterfactual_graphs = int(data.get("counterfactual_graphs", 0))
    if counterfactual_graphs:
        static["train"] += data_v3.generate_counterfactual_pairs_v3(
            graph_count=counterfactual_graphs,
            pairs_per_graph=int(data.get("counterfactual_pairs_per_graph", 3)),
            seed=seed,
            sampler=data_v3.BalancedSampler(),
            tasks=[task for task in dynamic_tasks if task in data_v3.DYNAMIC_TASKS],
            progress=progress,
        )
    for split in ("validation", "test"):
        replicates = int(data.get(f"static_{split}_replicates", 0))
        if replicates:
            static[split] = data_v3.generate_static_v3(
                split=split,
                graph_count=cell_count * replicates,
                tasks_per_graph=None,
                seed=seed,
                sampler=data_v3.BalancedSampler(),
                tasks=static_tasks,
                progress=progress,
            )

    session_cfg = data.get("sessions", {})
    replicates = {
        "validation": int(session_cfg.get("validation_replicates", 0)),
        "test": int(session_cfg.get("test_replicates", 0)),
    }
    replicates = {key: value for key, value in replicates.items() if value > 0}
    sessions = data_v3.generate_sessions_v3(
        replicates_per_split=replicates, seed=seed, tasks=dynamic_tasks, progress=progress
    )

    report: dict[str, Any] = {"distribution": data_v3.DISTRIBUTION_V3}
    for split, split_sessions in sessions.items():
        save_sessions(_data_path(config, split), split_sessions)
        report[split] = audit_sessions(split_sessions, allow_noop=True)
        if not report[split]["valid"]:
            raise RuntimeError(f"Generated invalid {split} data: {report[split]['failures'][:5]}")
        report[split]["labels"] = data_v3.session_label_audit(split_sessions)
        report[split]["graphs"] = data_v3.graph_audit(s.initial_graph for s in split_sessions)
    for split, tuples in static.items():
        save_static_tuples(_static_path(config, split), tuples)
        report[f"static_{split}"] = {
            "tuples": len(tuples),
            "labels": data_v3.label_audit(tuples),
            "graphs": data_v3.graph_audit(item.graph for item in tuples),
        }
    for split in ("validation", "test"):
        if split in static and "train" in static:
            report[f"static_{split}"]["text_leak_audit"] = data_v3.text_leak_audit(
                static["train"], static[split]
            )
    _write_json(_audit_path(config), report)
    _experiment_log(config, {"phase": "generate-v3", "distribution": data_v3.DISTRIBUTION_V3})
    return report


def _ensure_data(config: ExperimentConfig) -> None:
    if not _data_path(config, "test").exists():
        _generate(config)


def _static_examples(
    tuples: list[StaticQATuple], prompt_format: str = "wfi"
) -> list[TrainingExample]:
    def prompt_for(item: StaticQATuple) -> str:
        question = render_question(item.query, item.graph)
        return csv_graph_prompt(item.graph, question) if prompt_format == "csv" else question

    def answer_for(item: StaticQATuple) -> str:
        # CLEGR asks boolean questions with 'True'/'False'; scoring maps them back to yes/no.
        if prompt_format == "csv" and item.answer in ("yes", "no"):
            return "True" if item.answer == "yes" else "False"
        return item.answer

    return [
        TrainingExample(
            graph=item.graph,
            prompt=prompt_for(item),
            answer=answer_for(item),
            example_id=item.tuple_id,
            split=item.split,
            metadata={
                "reasoning_type": item.query.reasoning_type.value,
                "topology": item.topology.value,
                "density_bin": item.density_bin.value,
                "hop_depth": item.hop_depth.value,
                "scale_bin": item.scale_bin,
                "source_id": item.query.source,
                "target_id": item.query.target,
                "hops": item.query.hops,
            },
        )
        for item in tuples
    ]


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
                        "hops": turn.query.hops,
                    },
                )
            )
    return examples


def _training_examples(config: ExperimentConfig) -> list[TrainingExample]:
    data = config.section("data")
    if str(data.get("distribution", DISTRIBUTION)) in (DISTRIBUTION_V2, data_v3.DISTRIBUTION_V3):
        static_path = _static_path(config, "train")
        if static_path.exists():
            return _static_examples(
                load_static_tuples(static_path), str(data.get("prompt_format", "wfi"))
            )
    return _examples(load_sessions(_data_path(config, "train")))


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
    tensorizer_kind = str(model.get("tensorizer", "hash"))
    graph_config = GraphSAGEConfig(
        input_dim=int(model.get("node_feature_size", 256)),
        hidden_dim=int(model.get("graph_hidden_size", 256)),
        output_dim=int(model.get("graph_hidden_size", 256)),
        num_layers=int(model.get("graph_layers", 4)),
        dropout=float(model.get("graph_dropout", 0.0)),
        aggregation=str(model.get("graph_aggregation", "mean")),
    )
    if tensorizer_kind == "bert":
        from graph_modi.models.bert_tensorizer import BertNodeTensorizer, BertNodeTensorizerConfig

        tensorizer = BertNodeTensorizer(
            BertNodeTensorizerConfig(
                bert_model_name=str(model.get("bert_model_name", "bert-base-uncased")),
                feature_dim=graph_config.input_dim,
            )
        )
        return graph_config, GraphSAGEEncoder(graph_config), tensorizer
    tensor_config = NodeTensorizerConfig(feature_dim=graph_config.input_dim)
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
    tensorizer_kind = str(model.get("tensorizer", "hash"))
    tensor_config = NodeTensorizerConfig(feature_dim=int(model.get("node_feature_size", 256)))
    graph_config = GraphSAGEConfig(
        input_dim=tensor_config.feature_dim,
        hidden_dim=int(model.get("graph_hidden_size", 256)),
        output_dim=int(model.get("graph_hidden_size", 256)),
        num_layers=int(model.get("graph_layers", 4)),
        dropout=float(model.get("graph_dropout", 0.0)),
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
        config=TEAGLMConfig(
            freeze_gnn=True,
            max_sequence_length=int(model.get("max_sequence_length", 512)),
            prompt_template=str(model.get("prompt_template", "Question: {question}\nAnswer:")),
        ),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        torch_dtype=dtype,
        prefix_tokens=int(model.get("prefix_tokens", 8)),
        projector_hidden_dim=int(model.get("projector_hidden_size", 512)),
        projector_num_layers=int(model.get("projector_num_layers", 1)),
    )
    if tensorizer_kind == "bert":
        from graph_modi.models.bert_tensorizer import BertNodeTensorizer, BertNodeTensorizerConfig

        tea_model.tensorizer = BertNodeTensorizer(
            BertNodeTensorizerConfig(
                bert_model_name=str(model.get("bert_model_name", "bert-base-uncased")),
                feature_dim=graph_config.input_dim,
            )
        )
    elif not isinstance(tea_model.tensorizer, DeterministicNodeTensorizer):
        raise TypeError("Unexpected graph tensorizer")
    tea_model.graph_token_position = str(model.get("graph_token_position", "before_answer"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tea_model = tea_model.to(device)
    if bool(model.get("gradient_checkpointing", False)):
        tea_model.language_model.gradient_checkpointing_enable()
        tea_model.language_model.enable_input_require_grads()
    return tea_model


def _soft_prompt_model(config: ExperimentConfig) -> Any:
    import torch

    from graph_modi.models.soft_prompt import SoftPromptConfig, SoftPromptGLM

    model = config.section("model")
    dtype_name = str(model.get("torch_dtype", "float32"))
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise ValueError(f"Unknown torch dtype: {dtype_name}")
    soft_model = SoftPromptGLM.from_pretrained(
        str(model["lm_name"]),
        prefix_tokens=int(model.get("prefix_tokens", 10)),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        torch_dtype=dtype,
        config=SoftPromptConfig(
            prefix_tokens=int(model.get("prefix_tokens", 10)),
            max_sequence_length=int(model.get("max_sequence_length", 512)),
            prompt_template=str(model.get("prompt_template", "Question: {question}\nAnswer:")),
        ),
    )
    soft_model.graph_token_position = str(model.get("graph_token_position", "before_answer"))
    if bool(model.get("gradient_checkpointing", False)):
        soft_model.language_model.gradient_checkpointing_enable()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return soft_model.to(device)


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
    if model_config.get("backend") == "soft_prompt":
        print(json.dumps({"skipped": "soft_prompt has no GNN to pretrain"}, indent=2))
        return
    examples = _training_examples(config)
    training = config.section("training")
    objective = str(training.get("pretrain_objective", "classification"))
    if objective == "contrastive":
        # TEA-GLM's own pretraining objective (arXiv:2408.14512), reused by the
        # CLEGR paper -- needs the LM+tokenizer to embed node text, so build
        # the full TEAGLM (which already respects model.tensorizer=bert) and
        # reuse its gnn/tensorizer/language_model/tokenizer rather than
        # duplicating LM-loading logic here.
        tea_model = _tea_model(config)
        graphs = list({example.graph.graph_id: example.graph for example in examples}.values())
        result = pretrain_graph_encoder_contrastive(
            tea_model.gnn,
            tea_model.tensorizer,
            graphs,
            tea_model.language_model,
            tea_model.tokenizer,
            output_dir=config.output_dir / "gnn",
            epochs=int(training.get("gnn_epochs", training.get("epochs", 5))),
            graphs_per_batch=int(training.get("gnn_graphs_per_batch", 8)),
            learning_rate=float(training.get("gnn_learning_rate", 0.002)),
            temperature=float(training.get("contrastive_temperature", 0.07)),
            seed=int(training.get("seed", config.seed)),
        )
    else:
        _, gnn, tensorizer = _neural_components(config)
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
    training = config.section("training")
    train_config = ProjectorTrainingConfig(
        output_dir=config.output_dir / ("soft_prompt" if model_config.get("backend") == "soft_prompt" else "projector"),
        epochs=int(training.get("epochs", 5)),
        batch_size=int(training.get("batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        learning_rate=float(training.get("learning_rate", 5e-4)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        mixed_precision=str(training.get("mixed_precision", "no")),
        save_every_steps=int(training.get("save_every_steps", 500)),
        seed=int(training.get("seed", config.seed)),
        resume_from=training.get("resume_from"),
    )
    examples = _training_examples(config)
    if model_config.get("backend") == "soft_prompt":
        soft_model = _soft_prompt_model(config)
        result = train_soft_prompt(soft_model, examples, train_config)
        print(json.dumps(asdict(result), indent=2, default=str))
        return
    model = _tea_model(config)
    _load_local_gnn(model, config)
    architecture = str(model_config.get("architecture", "tea"))
    if architecture == "graph_token":
        from graph_modi.models.graph_token import train_graph_token

        result = train_graph_token(model, examples, train_config)
    else:
        result = train_projector(model, examples, train_config)
    print(json.dumps(asdict(result), indent=2, default=str))


def _backend(config: ExperimentConfig) -> GraphBackend:
    model_config = config.section("model")
    backend_name = str(model_config.get("backend", "mock"))
    if backend_name == "mock":
        return SymbolicMockBackend()
    if backend_name == "soft_prompt":
        from graph_modi.models.soft_prompt import SoftPromptBackend, load_soft_prompt_checkpoint

        soft_model = _soft_prompt_model(config)
        checkpoint = Path(
            model_config.get("checkpoint", config.output_dir / "soft_prompt" / "checkpoint-final")
        )
        load_soft_prompt_checkpoint(soft_model, checkpoint_dir=checkpoint)
        return SoftPromptBackend(
            soft_model,
            max_new_tokens=int(config.section("evaluation").get("max_new_tokens", 32)),
        )
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
    backend_cls = TEAGLMBackend
    if str(model_config.get("architecture", "tea")) == "graph_token":
        from graph_modi.models.graph_token import GraphTokenBackend

        backend_cls = GraphTokenBackend
    return backend_cls(
        model,
        max_new_tokens=int(config.section("evaluation").get("max_new_tokens", 32)),
    )


def command_freeze_v1(_config: ExperimentConfig, _args: argparse.Namespace) -> None:
    manifest = freeze_v1_manifest(Path.cwd())
    print(json.dumps(manifest, indent=2))


def command_static_eval(config: ExperimentConfig, args: argparse.Namespace) -> None:
    evaluation = config.section("evaluation")
    split = str(getattr(args, "split", evaluation.get("static_split", "test")))
    tuples = load_static_tuples(_static_path(config, split))
    result = evaluate_static_oracle(
        tuples,
        _backend(config),
        batch_size=int(evaluation.get("batch_size", 16)),
        progress=not getattr(args, "no_progress", False),
        prompt_format=str(config.section("data").get("prompt_format", "wfi")),
    )
    path = config.output_dir / f"static_eval_{split}.json"
    _write_json(path, result)
    _experiment_log(config, {"phase": "static-eval", "split": split, "passed": result["passed"]})
    summary = {key: result[key] for key in ("overall_accuracy", "passed", "failing_tasks")}
    print(json.dumps(summary, indent=2))


def command_pilot(config: ExperimentConfig, args: argparse.Namespace) -> None:
    _ensure_data(config)
    evaluation = config.section("evaluation")
    pilot_sessions = int(evaluation.get("pilot_sessions", 20))
    sessions = load_sessions(_data_path(config, "test"))[:pilot_sessions]
    static_path = _static_path(config, "test")
    static = load_static_tuples(static_path) if static_path.exists() else []
    backend = _backend(config)
    static_result = (
        evaluate_static_oracle(
            static,
            backend,
            batch_size=int(evaluation.get("batch_size", 16)),
            progress=not getattr(args, "no_progress", False),
        )
        if static
        else {"passed": True, "overall_accuracy": 1.0}
    )
    dynamic = evaluate_sessions(
        sessions,
        backend,
        conditions=[str(value) for value in evaluation.get("conditions", CONDITIONS)],
        batch_size=int(evaluation.get("batch_size", 16)),
        progress=not getattr(args, "no_progress", False),
    )
    report = {
        "git_sha": _git_sha(),
        "static_gate": static_result.get("passed", False),
        "static_accuracy": static_result.get("overall_accuracy"),
        "dynamic_summary": dynamic["summary"],
        "pilot_sessions": len(sessions),
    }
    path = config.output_dir / "pilot_report.json"
    _write_json(path, report)
    _experiment_log(config, {"phase": "pilot-gates", **report})
    print(json.dumps(report, indent=2))


def command_final_study(config: ExperimentConfig, args: argparse.Namespace) -> None:
    evaluation = config.section("evaluation")
    seeds = [int(value) for value in evaluation.get("seeds", [config.seed])]
    splits = [str(value) for value in evaluation.get("splits", ["test", "ood"])]
    conditions = [str(value) for value in evaluation.get("conditions", CONDITIONS)]
    batch_size = int(evaluation.get("batch_size", 16))
    matrix: dict[str, Any] = {"git_sha": _git_sha(), "seeds": {}}
    for seed in seeds:
        backend = _backend(config)
        matrix["seeds"][str(seed)] = {}
        for split in splits:
            path = _data_path(config, split)
            if not path.exists():
                continue
            sessions = load_sessions(path)
            matrix["seeds"][str(seed)][split] = evaluate_sessions(
                sessions,
                backend,
                conditions=conditions,
                batch_size=batch_size,
                progress=not getattr(args, "no_progress", False),
            )["summary"]
    out = config.output_dir / "final_study.json"
    _write_json(out, matrix)
    _experiment_log(config, {"phase": "final-study", "seeds": seeds, "splits": splits})
    print(json.dumps(matrix, indent=2))


def command_evaluate(config: ExperimentConfig, args: argparse.Namespace) -> None:
    _ensure_data(config)
    evaluation = config.section("evaluation")
    conditions = [str(value) for value in evaluation.get("conditions", CONDITIONS)]
    batch_size = int(evaluation.get("batch_size", 16))
    result: dict[str, Any] = {}
    backend = _backend(config)
    for split in evaluation.get("splits", ["test"]):
        sessions = load_sessions(_data_path(config, str(split)))
        result[str(split)] = evaluate_sessions(
            sessions,
            backend,
            conditions=conditions,
            batch_size=batch_size,
            progress=not getattr(args, "no_progress", False),
        )
    path = config.output_dir / "evaluation.json"
    _write_json(path, result)
    _experiment_log(config, {"phase": "evaluate", "splits": list(result)})
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
        "pilot-gates": command_pilot,
        "final-study": command_final_study,
    }
    for name, function in commands.items():
        child = subparsers.add_parser(name)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument("--no-progress", action="store_true")
        child.set_defaults(handler=function)
    static = subparsers.add_parser("static-eval")
    static.add_argument("--config", required=True, type=Path)
    static.add_argument("--split", default="test")
    static.add_argument("--no-progress", action="store_true")
    static.set_defaults(handler=command_static_eval)
    freeze = subparsers.add_parser("freeze-v1")
    freeze.add_argument("--config", required=True, type=Path)
    freeze.set_defaults(handler=command_freeze_v1)
    session = subparsers.add_parser("run-session")
    session.add_argument("--config", required=True, type=Path)
    session.add_argument("--session", required=True, type=Path)
    session.add_argument("--index", type=int, default=0)
    session.add_argument("--predicted-edits", action="store_true")
    session.add_argument("--include-history", action="store_true")
    session.set_defaults(handler=command_session)
    return parser


def _apply_gpu_memory_cap(fraction: float | None) -> None:
    if fraction is None or fraction >= 1.0:
        return
    import torch

    if not torch.cuda.is_available():
        return
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(f"[cli] capped this process to {fraction:.0%} of GPU memory", flush=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    fraction = config.values.get("gpu_memory_fraction")
    _apply_gpu_memory_cap(float(fraction) if fraction is not None else None)
    args.handler(config, args)


if __name__ == "__main__":
    main()
