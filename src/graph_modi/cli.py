"""Command-line entry points for reproducible GraphModi experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from graph_modi.config import ExperimentConfig, load_config
from graph_modi.data import v3 as data_v3
from graph_modi.data import v4 as data_v4
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
from graph_modi.data.validation import audit_sessions
from graph_modi.evaluation.runner import CONDITIONS, evaluate_sessions
from graph_modi.evaluation.static_eval import evaluate_static_oracle
from graph_modi.graph.solvers import render_question
from graph_modi.models.base import GraphBackend, SymbolicMockBackend
from graph_modi.pipeline.multiturn import materialize_states, run_session
from graph_modi.schema import ReasoningType, Session, StaticQATuple, TopologyFamily
from graph_modi.training import (
    ProjectorTrainingConfig,
    TrainingExample,
    pretrain_graph_encoder,
    pretrain_graph_encoder_contrastive,
    pretrain_graph_encoder_tea_glm,
    run_mock_training,
    set_deterministic_seed,
    train_projector,
    train_soft_prompt,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _data_path(config: ExperimentConfig, split: str) -> Path:
    # metro_v4 sessions use the names in datasets/metro_v4/manifest.json; older
    # distributions keep their original <split>.jsonl layout.
    if str(config.section("data").get("distribution", DISTRIBUTION)) == data_v4.DISTRIBUTION_V4:
        return config.data_dir / f"sessions_{split}.jsonl"
    return config.data_dir / f"{split}.jsonl"


def _audit_path(config: ExperimentConfig) -> Path:
    return config.data_dir / "audit.json"


def _static_path(config: ExperimentConfig, split: str) -> Path:
    return config.data_dir / f"static_{split}.jsonl"


def _model_device() -> Any:
    """CUDA for real model runs; refuse to fall back to CPU unless GRAPHMODI_ALLOW_CPU=1.

    A CPU fallback on a GPU box (e.g. a torch wheel built for a newer CUDA than the driver) would
    otherwise "train" the 8B model on CPU for hours without any error."""
    import os

    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("GRAPHMODI_ALLOW_CPU") == "1":
        return torch.device("cpu")
    raise RuntimeError(
        "CUDA is not available (torch "
        f"{torch.__version__}, built for CUDA {torch.version.cuda}). Check the NVIDIA driver and "
        "the torch wheel, or set GRAPHMODI_ALLOW_CPU=1 to run on CPU deliberately."
    )


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_sha() -> str:
    return _git("rev-parse", "HEAD")


def _git_dirty() -> bool | None:
    """True/False for tracked-file changes; None when git state cannot be read."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return None if status == "unknown" else status != ""


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(
    config: ExperimentConfig,
    *,
    data_files: list[Path],
    checkpoint: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Identity of a result: code, config, data files and checkpoint it came from."""
    return {
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "config": str(config.path),
        "output_dir": str(config.output_dir),
        "data_sha256": {str(path): _sha256_file(path) for path in data_files if path.exists()},
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        **extra,
    }


def _experiment_log(config: ExperimentConfig, payload: dict[str, Any]) -> None:
    """Append one line per phase to ``<output_dir>/experiment_log.md`` (not the tracked log)."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    phase = payload.get("phase", "run")
    payload_json = json.dumps(payload, sort_keys=True, default=str)
    line = f"- `{_git_sha()[:8]}` {phase} ({config.path}): {payload_json}\n"
    with (config.output_dir / "experiment_log.md").open("a", encoding="utf-8") as handle:
        handle.write(line)


def _generate(config: ExperimentConfig, *, progress: bool = True) -> dict[str, Any]:
    data = config.section("data")
    distribution = str(data.get("distribution", DISTRIBUTION))
    if distribution == DISTRIBUTION_V2:
        return _generate_v2(config, progress=progress)
    if distribution == data_v3.DISTRIBUTION_V3:
        return _generate_v3(config, progress=progress)
    if distribution == data_v4.DISTRIBUTION_V4:
        return _generate_v4(config, progress=progress)
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
            "report": {
                key: (
                    value
                    if key.startswith("static_")
                    else {
                        "sessions": value.get("sessions"),
                        "turns": value.get("turns"),
                        "valid": value.get("valid"),
                    }
                )
                for key, value in report.items()
            },
            "distribution": DISTRIBUTION_V2,
            "session_layout": session_layout,
            "dynamic_only": dynamic_only,
        },
    )
    return report


def _generate_v3(config: ExperimentConfig, *, progress: bool = True) -> dict[str, Any]:
    """Compact CLEGR-style subway graphs, label-balanced (see data/v3.py)."""
    data = config.section("data")
    data_v3.set_degree_balance(bool(data.get("degree_balance", False)))
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


def _generate_v4(config: ExperimentConfig, *, progress: bool = True) -> dict[str, Any]:
    """Corrected metro_v4 benchmark (see data/v4.py and the 2026-10 audits)."""
    data = config.section("data")
    static_tasks = _reasoning_types(data, "static_tasks") or list(data_v4.STATIC_TASKS)
    dynamic_tasks = _reasoning_types(data, "dynamic_tasks") or list(data_v4.DYNAMIC_TASKS)
    cell_count = len(data_v4.NODE_COUNTS) * len(data_v4.DENSITIES)
    seed = config.seed

    static: dict[str, list[StaticQATuple]] = {}
    static["train"] = data_v4.generate_static_v4(
        split="train",
        graph_count=int(data.get("static_train_graphs", 2000)),
        tasks_per_graph=int(data.get("static_tuples_per_graph", 9)),
        seed=seed,
        sampler=data_v4.BalancedSampler(),
        tasks=static_tasks,
        progress=progress,
    )
    counterfactual_graphs = int(data.get("counterfactual_graphs", 0))
    if counterfactual_graphs:
        static["train"] += data_v4.generate_counterfactual_pairs_v4(
            graph_count=counterfactual_graphs,
            pairs_per_graph=int(data.get("counterfactual_pairs_per_graph", 3)),
            seed=seed,
            sampler=data_v4.BalancedSampler(),
            tasks=[task for task in dynamic_tasks if task in data_v4.DYNAMIC_TASKS],
            progress=progress,
        )
    for split in ("validation", "test"):
        replicates = int(data.get(f"static_{split}_replicates", 0))
        if replicates:
            static[split] = data_v4.generate_static_v4(
                split=split,
                graph_count=cell_count * replicates,
                tasks_per_graph=None,
                seed=seed,
                sampler=data_v4.BalancedSampler(),
                tasks=static_tasks,
                progress=progress,
            )

    session_cfg = data.get("sessions", {})
    replicates = {
        "validation": int(session_cfg.get("validation_replicates", 0)),
        "test": int(session_cfg.get("test_replicates", 0)),
    }
    replicates = {key: value for key, value in replicates.items() if value > 0}
    sessions = data_v4.generate_sessions_v4(
        replicates_per_split=replicates, seed=seed, tasks=dynamic_tasks, progress=progress
    )

    report: dict[str, Any] = {"distribution": data_v4.DISTRIBUTION_V4}
    for split, split_sessions in sessions.items():
        save_sessions(_data_path(config, split), split_sessions)
        report[split] = audit_sessions(split_sessions, allow_noop=True)
        if not report[split]["valid"]:
            raise RuntimeError(f"Generated invalid {split} data: {report[split]['failures'][:5]}")
        report[split]["labels"] = data_v4.session_label_audit(split_sessions)
        report[split]["graphs"] = data_v4.graph_audit(s.initial_graph for s in split_sessions)
    for split, tuples in static.items():
        save_static_tuples(_static_path(config, split), tuples)
        report[f"static_{split}"] = {
            "tuples": len(tuples),
            "labels": data_v4.label_audit(tuples),
            "graphs": data_v4.graph_audit(item.graph for item in tuples),
        }
    for split in ("validation", "test"):
        if split in static and "train" in static:
            report[f"static_{split}"]["text_leak_audit"] = data_v4.text_leak_audit(
                static["train"], static[split]
            )
    _write_json(_audit_path(config), report)
    _experiment_log(config, {"phase": "generate-v4", "distribution": data_v4.DISTRIBUTION_V4})
    return report


def _ensure_data(config: ExperimentConfig) -> None:
    if _data_path(config, "test").exists():
        return
    if config.section("model").get("backend", "mock") == "mock":
        _generate(config)
        return
    # Real runs must never regenerate silently: a regenerated set need not match the manifest.
    raise FileNotFoundError(
        f"{_data_path(config, 'test')} is missing. Generate it explicitly "
        f"(`graph-modi generate --config {config.path}`) or copy the dataset and verify "
        f"its sha256 against {config.data_dir / 'manifest.json'}."
    )


def _static_examples(tuples: list[StaticQATuple]) -> list[TrainingExample]:
    return [
        TrainingExample(
            graph=item.graph,
            prompt=render_question(item.query, item.graph),
            answer=item.answer,
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


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _validation_examples(config: ExperimentConfig) -> list[TrainingExample]:
    """Held-out examples (same construction as training) for the validation-loss curve."""
    data = config.section("data")
    if str(data.get("distribution", DISTRIBUTION)) in (
        DISTRIBUTION_V2,
        data_v3.DISTRIBUTION_V3,
        data_v4.DISTRIBUTION_V4,
    ):
        static_path = _static_path(config, "validation")
        if static_path.exists():
            examples = _static_examples(load_static_tuples(static_path))
            limit = int(config.section("training").get("val_max_examples", 1000))
            # Fixed stride subsample: deterministic and spread over every task/graph.
            stride = max(1, len(examples) // max(1, limit))
            return examples[::stride][:limit]
    return _examples(load_sessions(_data_path(config, "validation")))[:1000]


def _training_examples(config: ExperimentConfig) -> list[TrainingExample]:
    data = config.section("data")
    if str(data.get("distribution", DISTRIBUTION)) in (
        DISTRIBUTION_V2,
        data_v3.DISTRIBUTION_V3,
        data_v4.DISTRIBUTION_V4,
    ):
        static_path = _static_path(config, "train")
        if static_path.exists():
            return _static_examples(load_static_tuples(static_path))
    return _examples(load_sessions(_data_path(config, "train")))


def _validation_graphs(config: ExperimentConfig, limit: int = 300) -> list[Any]:
    """Distinct held-out graphs for scoring pretraining (never used to train it)."""
    path = _static_path(config, "validation")
    if not path.exists():
        return []
    graphs = {item.graph.graph_id: item.graph for item in load_static_tuples(path)}
    return list(graphs.values())[:limit]


def _embedding_only_lm(lm_name: str) -> tuple[Any, Any]:
    """The LLM's input-embedding table (read from its safetensors shards) and its tokenizer."""
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer

    root = Path(lm_name)
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    key = "model.embed_tokens.weight"
    with safe_open(str(root / index["weight_map"][key]), framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(key)
    device = _model_device()
    embedding = torch.nn.Embedding.from_pretrained(weight.to(device), freeze=True)

    class _EmbeddingOnlyLM:
        def get_input_embeddings(self) -> Any:
            return embedding

    return _EmbeddingOnlyLM(), AutoTokenizer.from_pretrained(lm_name)


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
        output_dim=int(model.get("graph_output_size", model.get("graph_hidden_size", 256))),
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
    # Dropout does not change the weights' shapes, so a checkpoint pretrained with one dropout can
    # be fine-tuned with another.
    saved_config = {k: v for k, v in (metadata.get("gnn_config") or {}).items() if k != "dropout"}
    if saved_config != {k: v for k, v in asdict(model.gnn.config).items() if k != "dropout"}:
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
        output_dim=int(model.get("graph_output_size", model.get("graph_hidden_size", 256))),
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
    ablation = str(model.get("graph_ablation", "none"))
    if ablation not in {"none", "zero_gnn"}:
        raise ValueError("model.graph_ablation must be one of: none, zero_gnn")
    tea_model.graph_ablation = ablation
    vector_norm = str(model.get("graph_vector_norm", "none"))
    if vector_norm not in {"none", "standardize"}:
        raise ValueError("model.graph_vector_norm must be one of: none, standardize")
    tea_model.graph_vector_norm = vector_norm
    device = _model_device()
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
    if bool(model.get("gradient_checkpointing", False)):
        soft_model.language_model.gradient_checkpointing_enable()
    device = _model_device()
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
    if model_config.get("architecture") == "graph_token":
        print(json.dumps({"skipped": "graph_token trains its GNN end-to-end"}, indent=2))
        return
    examples = _training_examples(config)
    training = config.section("training")
    objective = str(training.get("pretrain_objective", "classification"))
    if objective == "tea_glm":
        # TEA-GLM stage 1: self-supervised graph contrastive learning plus feature-wise alignment
        # with the LLM's token-embedding principal components; only the embedding table is loaded.
        set_deterministic_seed(int(training.get("seed", config.seed)))
        _, gnn, tensorizer = _neural_components(config)
        embedding_lm, _ = _embedding_only_lm(str(model_config["lm_name"]))
        graphs = list({example.graph.graph_id: example.graph for example in examples}.values())
        result = pretrain_graph_encoder_tea_glm(
            gnn,
            tensorizer,
            graphs,
            embedding_lm,
            output_dir=config.output_dir / "gnn",
            seed=int(training.get("seed", config.seed)),
            epochs=int(training.get("gnn_epochs", 70)),
            patience=int(training.get("gnn_patience", 10)),
            graphs_per_batch=int(training.get("gnn_graphs_per_batch", 32)),
            learning_rate=float(training.get("gnn_learning_rate", 2e-3)),
            tau=float(training.get("contrastive_temperature", 0.4)),
            principal_components=int(training.get("principal_components", 1000)),
            drop_edge_rates=tuple(training.get("drop_edge_rates", (0.3, 0.4))),
            drop_feature_rates=tuple(training.get("drop_feature_rates", (0.0, 0.1))),
            dropout=_optional_float(training.get("pretrain_dropout", 0.5)),
            eval_graphs=_validation_graphs(config),
        )
    elif objective == "contrastive":
        # Node-text alignment (pretrain_graph_encoder_contrastive): needs the LM's tokenizer and
        # embedding table to embed node text.
        # Only the LLM's input-embedding table and tokenizer are needed, so the 8B model is
        # not loaded. The GNN is built after seeding, so a given seed gives one initialisation.
        set_deterministic_seed(int(training.get("seed", config.seed)))
        _, gnn, tensorizer = _neural_components(config)
        embedding_lm, tokenizer = _embedding_only_lm(str(model_config["lm_name"]))
        graphs = list({example.graph.graph_id: example.graph for example in examples}.values())
        result = pretrain_graph_encoder_contrastive(
            gnn,
            tensorizer,
            graphs,
            embedding_lm,
            tokenizer,
            output_dir=config.output_dir / "gnn",
            epochs=int(training.get("gnn_epochs", training.get("epochs", 5))),
            graphs_per_batch=int(training.get("gnn_graphs_per_batch", 8)),
            learning_rate=float(training.get("gnn_learning_rate", 0.002)),
            temperature=float(training.get("contrastive_temperature", 0.07)),
            seed=int(training.get("seed", config.seed)),
            target=str(training.get("contrastive_target", "mean")),
            status_weight=float(training.get("contrastive_status_weight", 4.0)),
            dropout=_optional_float(training.get("pretrain_dropout")),
            eval_graphs=_validation_graphs(config),
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
        output_dir=config.output_dir
        / ("soft_prompt" if model_config.get("backend") == "soft_prompt" else "projector"),
        epochs=int(training.get("epochs", 5)),
        batch_size=int(training.get("batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        learning_rate=float(training.get("learning_rate", 5e-4)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        mixed_precision=str(training.get("mixed_precision", "no")),
        save_every_steps=int(training.get("save_every_steps", 500)),
        seed=int(training.get("seed", config.seed)),
        resume_from=training.get("resume_from"),
        gnn_learning_rate=_optional_float(training.get("gnn_learning_rate_joint")),
        gnn_weight_decay=_optional_float(training.get("gnn_weight_decay_joint")),
        lr_schedule=str(training.get("lr_schedule", "constant")),
        warmup_steps=int(training.get("warmup_steps", 0)),
        min_lr_ratio=float(training.get("min_lr_ratio", 0.0)),
        eval_every_steps=int(training.get("eval_every_steps", 0)),
        val_max_examples=int(training.get("val_max_examples", 1000)),
        init_checkpoint=training.get("init_checkpoint"),
        wrap_prompts=bool(training.get("wrap_prompts", False)),
    )
    examples = _training_examples(config)
    # Every model is built after seeding, so the projector (and a GraphToken GNN) start from a
    # seeded initialisation rather than whatever the global RNG held.
    set_deterministic_seed(train_config.seed)
    if model_config.get("backend") == "soft_prompt":
        soft_model = _soft_prompt_model(config)
        result = train_soft_prompt(soft_model, examples, train_config)
        print(json.dumps(asdict(result), indent=2, default=str))
        return
    architecture = str(model_config.get("architecture", "tea"))
    val_examples = _validation_examples(config) if train_config.eval_every_steps else None
    if architecture == "graph_token":
        from graph_modi.models.graph_token import train_graph_token

        model = _tea_model(config)
        if model_config.get("gnn_checkpoint"):
            # Warm start: pretrained (TEA contrastive) GNN, then trained jointly.
            _load_local_gnn(model, config)
        result = train_graph_token(model, examples, train_config, val_examples=val_examples)
    else:
        model = _tea_model(config)
        if str(model_config.get("gnn_init", "pretrained")) == "random":
            # No-pretraining control: the seeded random GNN stays frozen and is saved with the
            # projector, so evaluation loads exactly these weights.
            model.set_gnn_frozen(True)
        else:
            _load_local_gnn(model, config)
        result = train_projector(model, examples, train_config, val_examples=val_examples)
    print(json.dumps(asdict(result), indent=2, default=str))


def _load_tea_checkpoint(model: Any, checkpoint: Path) -> None:
    from graph_modi.models.tea_glm import load_external_component

    metadata = checkpoint / "metadata.json"
    # GNN first: the projector's metadata binds to this exact GNN state.
    for component in ("gnn", "projector"):
        load_external_component(
            model,
            component=component,
            weights_path=checkpoint / f"{component}.pt",
            metadata_path=metadata,
        )


def _encoder_autocast_dtype(config: ExperimentConfig, checkpoint: Path) -> Any:
    """Autocast dtype for the GNN + projector at evaluation time.

    ``evaluation.encoder_autocast``: ``fp32`` (default, the historical behaviour: only the LM runs
    in bf16), ``match_training`` (the dtype recorded in the checkpoint's training config, which is
    what the weights were optimised under), or ``bf16``.
    """
    import torch

    from graph_modi.models.tea_glm import load_checkpoint_metadata

    # Default: evaluate the GNN + projector under the dtype they were trained in.
    mode = str(config.section("evaluation").get("encoder_autocast", "match_training"))
    if mode == "fp32":
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode != "match_training":
        raise ValueError("evaluation.encoder_autocast must be one of: fp32, match_training, bf16")
    recorded = load_checkpoint_metadata(checkpoint / "metadata.json").get("training_config", {})
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
        str(recorded.get("mixed_precision", "no"))
    )


def _answer_prior(config: ExperimentConfig, conditions: list[str]) -> dict[str, str] | None:
    """Most common training gold answer per reasoning type, for the ``majority_answer`` baseline."""
    if "majority_answer" not in conditions:
        return None
    from collections import Counter, defaultdict

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    static_path = _static_path(config, "train")
    if static_path.exists():
        for item in load_static_tuples(static_path):
            counts[item.query.reasoning_type.value][item.answer] += 1
    else:
        for session in load_sessions(_data_path(config, "train")):
            for turn in session.turns:
                counts[turn.query.reasoning_type.value][turn.gold_answer] += 1
    return {task: counter.most_common(1)[0][0] for task, counter in counts.items()}


def _checkpoint_path(config: ExperimentConfig, selection: str | None = None) -> Path | None:
    """Checkpoint to evaluate: ``model.checkpoint`` if set, else ``checkpoint-<selection>``.

    ``selection`` (CLI ``--checkpoint``) overrides ``evaluation.checkpoint``; both default to
    ``final``. ``best`` is the lowest validation-loss checkpoint (needs ``eval_every_steps``).
    """
    model_config = config.section("model")
    backend_name = str(model_config.get("backend", "mock"))
    if backend_name == "mock":
        return None
    if model_config.get("checkpoint") and selection is None:
        return Path(model_config["checkpoint"])
    selection = selection or str(config.section("evaluation").get("checkpoint", "final"))
    if "/" in selection:
        return Path(selection)
    folder = "soft_prompt" if backend_name == "soft_prompt" else "projector"
    path = config.output_dir / folder / f"checkpoint-{selection}"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint {path} does not exist")
    return path


def _checkpoint_tag(checkpoint: Path | None) -> str:
    if checkpoint is None:
        return "mock"
    return checkpoint.name.removeprefix("checkpoint-")


def _backend(config: ExperimentConfig, checkpoint: Path | None = None) -> GraphBackend:
    model_config = config.section("model")
    backend_name = str(model_config.get("backend", "mock"))
    if backend_name == "mock":
        return SymbolicMockBackend()
    checkpoint = checkpoint or _checkpoint_path(config)
    assert checkpoint is not None
    if backend_name == "soft_prompt":
        from graph_modi.models.soft_prompt import SoftPromptBackend, load_soft_prompt_checkpoint

        soft_model = _soft_prompt_model(config)
        load_soft_prompt_checkpoint(soft_model, checkpoint_dir=checkpoint)
        return SoftPromptBackend(
            soft_model,
            max_new_tokens=int(config.section("evaluation").get("max_new_tokens", 32)),
        )
    from graph_modi.models.tea_glm import TEAGLMBackend

    model = _tea_model(config)
    _load_tea_checkpoint(model, checkpoint)
    model.encoder_autocast_dtype = _encoder_autocast_dtype(config, checkpoint)
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
    data_file = _static_path(config, split)
    tuples = load_static_tuples(data_file)
    checkpoint = _checkpoint_path(config, getattr(args, "checkpoint", None))
    backend = _backend(config, checkpoint)
    result = evaluate_static_oracle(
        tuples,
        backend,
        batch_size=int(evaluation.get("batch_size", 16)),
        progress=not getattr(args, "no_progress", False),
    )
    result["provenance"] = _provenance(
        config,
        data_files=[data_file],
        checkpoint=checkpoint,
        split=split,
        tuples=len(tuples),
        encoder_autocast=str(
            getattr(getattr(backend, "model", None), "encoder_autocast_dtype", None)
        ),
    )
    tag = _checkpoint_tag(checkpoint)
    path = config.output_dir / f"static_eval_{split}_{tag}.json"
    _write_json(path, result)
    _experiment_log(
        config,
        {"phase": "static-eval", "split": split, "checkpoint": str(checkpoint), "path": str(path)},
    )
    summary = {key: result[key] for key in ("overall_accuracy", "passed", "failing_tasks")}
    summary["written"] = str(path)
    print(json.dumps(summary, indent=2))


def command_sweep_checkpoints(config: ExperimentConfig, args: argparse.Namespace) -> None:
    """Static-eval every saved checkpoint of a run (one model load) to see when, if ever, the
    graph signal appears: per-task accuracy plus how constant the predictions are."""
    from collections import Counter, defaultdict

    evaluation = config.section("evaluation")
    split = str(args.split)
    tuples = load_static_tuples(_static_path(config, split))
    if args.limit:
        stride = max(1, len(tuples) // args.limit)
        tuples = tuples[::stride][: args.limit]
    backend = _backend(config)
    root = Path(args.checkpoints) if args.checkpoints else config.output_dir / "projector"
    checkpoints = sorted(path for path in root.glob("checkpoint-*") if (path / "gnn.pt").exists())
    results = []
    for checkpoint in checkpoints:
        _load_tea_checkpoint(backend.model, checkpoint)
        outcome = evaluate_static_oracle(
            tuples,
            backend,
            batch_size=int(evaluation.get("batch_size", 16)),
            progress=False,
        )
        by_task: dict[str, Counter[str]] = defaultdict(Counter)
        for row in outcome["rows"]:
            by_task[row["reasoning_type"]][str(row["predicted_answer"])] += 1
        # Share of the modal answer per task: 1.0 means the model ignores its input.
        modal_share = {
            task: counter.most_common(1)[0][1] / sum(counter.values())
            for task, counter in sorted(by_task.items())
        }
        results.append(
            {
                "checkpoint": checkpoint.name,
                "overall_accuracy": outcome["overall_accuracy"],
                "task_accuracy": outcome["task_accuracy"],
                "modal_answer_share": modal_share,
            }
        )
        print(
            f"[sweep] {checkpoint.name} overall={outcome['overall_accuracy']:.3f} "
            f"mean_modal_share={sum(modal_share.values()) / len(modal_share):.2f}",
            flush=True,
        )
    path = config.output_dir / f"checkpoint_sweep_{split}.json"
    _write_json(path, {"split": split, "tuples": len(tuples), "checkpoints": results})
    print(f"wrote {path}")


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
        answer_prior=_answer_prior(
            config, [str(value) for value in evaluation.get("conditions", CONDITIONS)]
        ),
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
                answer_prior=_answer_prior(config, conditions),
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
    checkpoint = _checkpoint_path(config, getattr(args, "checkpoint", None))
    backend = _backend(config, checkpoint)
    splits = [str(split) for split in evaluation.get("splits", ["test"])]
    for split in splits:
        sessions = load_sessions(_data_path(config, split))
        result[split] = evaluate_sessions(
            sessions,
            backend,
            conditions=conditions,
            batch_size=batch_size,
            progress=not getattr(args, "no_progress", False),
            answer_prior=_answer_prior(config, conditions),
        )
        result[split]["provenance"] = _provenance(
            config,
            data_files=[_data_path(config, split)],
            checkpoint=checkpoint,
            split=split,
            sessions=len(sessions),
            turns=sum(len(session.turns) for session in sessions),
            conditions=conditions,
            encoder_autocast=str(evaluation.get("encoder_autocast", "match_training")),
        )
    tag = f"_{_checkpoint_tag(checkpoint)}" if checkpoint is not None else ""
    path = config.output_dir / f"evaluation{tag}.json"
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
    checkpoint_help = "best | final | path to a checkpoint dir (default: evaluation.checkpoint)"
    for name, function in commands.items():
        child = subparsers.add_parser(name)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument("--no-progress", action="store_true")
        if name == "evaluate":
            child.add_argument("--checkpoint", default=None, help=checkpoint_help)
        child.set_defaults(handler=function)
    static = subparsers.add_parser("static-eval")
    static.add_argument("--config", required=True, type=Path)
    static.add_argument("--split", default="test")
    static.add_argument("--checkpoint", default=None, help=checkpoint_help)
    static.add_argument("--no-progress", action="store_true")
    static.set_defaults(handler=command_static_eval)
    sweep = subparsers.add_parser("sweep-checkpoints")
    sweep.add_argument("--config", required=True, type=Path)
    sweep.add_argument("--split", default="validation")
    sweep.add_argument("--checkpoints", default=None, help="directory holding checkpoint-*")
    sweep.add_argument("--limit", type=int, default=0, help="strided subsample size (0 = all)")
    sweep.add_argument("--no-progress", action="store_true")
    sweep.set_defaults(handler=command_sweep_checkpoints)
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
    try:
        import torch
    except ImportError:  # CPU-only tools (e.g. data generation) don't need torch
        return

    if not torch.cuda.is_available():
        return
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(f"[cli] capped this process to {fraction:.0%} of GPU memory", flush=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    fraction = config.values.get("gpu_memory_fraction")
    if args.command != "generate":  # data generation is CPU-only
        _apply_gpu_memory_cap(float(fraction) if fraction is not None else None)
    args.handler(config, args)


if __name__ == "__main__":
    main()
