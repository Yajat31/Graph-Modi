"""Projector training and checkpointing for the optional TEA-GLM model."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from graph_modi.models.tea_glm import (
    TEA_CHECKPOINT_FORMAT,
    TEAGLM,
    DeterministicNodeTensorizer,
    GraphSAGEEncoder,
    assert_checkpoint_compatible,
    data_records_sha256,
    load_checkpoint_metadata,
    model_checkpoint_metadata,
    require_tea_dependencies,
    save_checkpoint_metadata,
)
from graph_modi.schema import AttributedGraph


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One graph-conditioned, teacher-forced causal-LM example."""

    graph: AttributedGraph
    prompt: str
    answer: str
    example_id: str
    split: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "split": self.split,
            "graph": self.graph.to_dict(),
            "prompt": self.prompt,
            "answer": self.answer,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ProjectorTrainingConfig:
    """Accelerate settings for training only the graph prefix projector."""

    output_dir: str | Path
    epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    mixed_precision: str = "no"
    save_every_steps: int = 100
    seed: int = 42
    num_workers: int = 0
    resume_from: str | Path | None = None

    def __post_init__(self) -> None:
        if (
            min(
                self.epochs,
                self.batch_size,
                self.gradient_accumulation_steps,
                self.save_every_steps,
            )
            <= 0
        ):
            raise ValueError("epochs, batch size, accumulation, and save interval must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0 or self.max_grad_norm < 0:
            raise ValueError("weight_decay and max_grad_norm must be non-negative")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of: no, fp16, bf16")

    def metadata(self) -> dict[str, Any]:
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        values["resume_from"] = str(self.resume_from) if self.resume_from else None
        return values


@dataclass(frozen=True, slots=True)
class TrainingResult:
    output_dir: Path
    final_checkpoint: Path
    global_step: int
    mean_loss: float
    data_sha256: str


@dataclass(frozen=True, slots=True)
class GNNPretrainingResult:
    checkpoint: Path
    metadata: Path
    labels: tuple[str, ...]
    training_accuracy: float


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic kernels."""
    require_tea_dependencies()
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _data_identity(
    examples: Sequence[TrainingExample],
    explicit_split: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    split_counts: dict[str, int] = {}
    for example in examples:
        split_counts[example.split] = split_counts.get(example.split, 0) + 1
    split = {
        "counts": dict(sorted(split_counts.items())),
        "example_ids_sha256": data_records_sha256(
            {"example_id": example.example_id, "split": example.split} for example in examples
        ),
    }
    if explicit_split:
        split["declared"] = dict(explicit_split)
    data_hash = data_records_sha256(example.record() for example in examples)
    return split, data_hash


def pretrain_graph_encoder(
    gnn: GraphSAGEEncoder,
    tensorizer: DeterministicNodeTensorizer,
    examples: Sequence[TrainingExample],
    *,
    output_dir: str | Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> GNNPretrainingResult:
    """Pretrain the GNN on solver labels before freezing it for projection."""
    require_tea_dependencies()
    if not examples:
        raise ValueError("At least one GNN pretraining example is required")
    if min(epochs, batch_size) <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    import torch
    from torch import nn

    set_deterministic_seed(seed)
    labels = tuple(sorted({example.answer for example in examples}))
    label_to_id = {label: index for index, label in enumerate(labels)}
    head = nn.Linear(gnn.config.output_dim, len(labels))
    optimizer = torch.optim.AdamW(
        [*gnn.parameters(), *head.parameters()],
        lr=learning_rate,
    )
    rng = random.Random(seed)
    ordered = list(examples)
    for _ in range(epochs):
        rng.shuffle(ordered)
        gnn.train()
        head.train()
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start : start + batch_size]
            vectors = torch.stack([gnn.pooled(tensorizer(example.graph)) for example in batch])
            targets = torch.tensor(
                [label_to_id[example.answer] for example in batch],
                dtype=torch.long,
            )
            loss = nn.functional.cross_entropy(head(vectors), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    gnn.eval()
    head.eval()
    correct = 0
    with torch.no_grad():
        for example in examples:
            logits = head(gnn.pooled(tensorizer(example.graph)))
            correct += int(labels[int(logits.argmax())] == example.answer)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "gnn.pt"
    head_checkpoint = destination / "gnn_head.pt"
    torch.save(gnn.state_dict(), checkpoint)
    torch.save(head.state_dict(), head_checkpoint)
    _, data_hash = _data_identity(examples, None)
    metadata_path = destination / "metadata.json"
    _write_json(
        metadata_path,
        {
            "format": "graph-modi-gnn-v1",
            "gnn_config": asdict(gnn.config),
            "tensorizer_config": asdict(tensorizer.config),
            "labels": labels,
            "data_sha256": data_hash,
            "seed": seed,
            "epochs": epochs,
            "training_accuracy": correct / len(examples),
        },
    )
    return GNNPretrainingResult(
        checkpoint=checkpoint,
        metadata=metadata_path,
        labels=labels,
        training_accuracy=correct / len(examples),
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _save_training_checkpoint(
    *,
    accelerator: Any,
    model: Any,
    optimizer: Any,
    checkpoint_dir: Path,
    config: ProjectorTrainingConfig,
    data_split: Mapping[str, Any],
    data_sha256: str,
    step: int,
    epoch: int,
    next_batch: int,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        accelerator.save(unwrapped.gnn.state_dict(), checkpoint_dir / "gnn.pt")
        accelerator.save(unwrapped.projector.state_dict(), checkpoint_dir / "projector.pt")
        accelerator.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        metadata = model_checkpoint_metadata(
            unwrapped,
            training_config=config.metadata(),
            data_split=data_split,
            data_sha256=data_sha256,
            step=step,
        )
        save_checkpoint_metadata(checkpoint_dir / "metadata.json", metadata)
        _write_json(
            checkpoint_dir / "trainer_state.json",
            {"global_step": step, "epoch": epoch, "next_batch": next_batch},
        )
    accelerator.wait_for_everyone()


def _validate_resume(
    model: TEAGLM,
    checkpoint_dir: Path,
    *,
    data_split: Mapping[str, Any],
    data_sha256: str,
) -> tuple[int, int, int]:
    metadata = load_checkpoint_metadata(checkpoint_dir / "metadata.json")
    assert_checkpoint_compatible(model, metadata, component="projector")
    if metadata.get("data_sha256") != data_sha256:
        raise ValueError("Resume checkpoint was created from different training examples")
    if metadata.get("data_split") != dict(data_split):
        raise ValueError("Resume checkpoint uses different data-split metadata")
    state = json.loads((checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8"))
    return int(state["global_step"]), int(state["epoch"]), int(state["next_batch"])


def train_projector(
    model: TEAGLM,
    examples: Sequence[TrainingExample],
    config: ProjectorTrainingConfig,
    *,
    data_split: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """Train graph prefix tokens with full answer loss and frozen GNN/causal LM."""
    require_tea_dependencies()
    if not examples:
        raise ValueError("At least one training example is required")
    try:
        import torch
        from accelerate import Accelerator
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "Projector training requires Accelerate. "
            "Install dependencies with `pip install 'graph-modi[tea]'`."
        ) from exc

    set_deterministic_seed(config.seed)
    model.set_gnn_frozen(True)
    for parameter in model.language_model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.projector.parameters():
        parameter.requires_grad_(True)

    split_identity, data_sha256 = _data_identity(examples, data_split)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        list(examples),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda batch: batch,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.projector.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    global_step = 0
    first_epoch = 0
    skip_batches = 0
    if config.resume_from is not None:
        checkpoint_dir = Path(config.resume_from)
        unwrapped = accelerator.unwrap_model(model)
        global_step, first_epoch, skip_batches = _validate_resume(
            unwrapped,
            checkpoint_dir,
            data_split=split_identity,
            data_sha256=data_sha256,
        )
        unwrapped.projector.load_state_dict(
            torch.load(
                checkpoint_dir / "projector.pt",
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        unwrapped.gnn.load_state_dict(
            torch.load(
                checkpoint_dir / "gnn.pt",
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        optimizer.load_state_dict(
            torch.load(
                checkpoint_dir / "optimizer.pt",
                map_location="cpu",
                weights_only=True,
            )
        )

    observed_loss = 0.0
    observed_updates = 0
    for epoch in range(first_epoch, config.epochs):
        model.train()
        epoch_loader = loader
        batch_offset = 0
        if epoch == first_epoch and skip_batches:
            epoch_loader = accelerator.skip_first_batches(loader, skip_batches)
            batch_offset = skip_batches
        for batch_index, batch in enumerate(epoch_loader, start=batch_offset):
            with accelerator.accumulate(model):
                output = model(
                    graphs=[example.graph for example in batch],
                    prompts=[example.prompt for example in batch],
                    answers=[example.answer for example in batch],
                )
                loss = output.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients and config.max_grad_norm:
                    accelerator.clip_grad_norm_(
                        (
                            parameter
                            for group in optimizer.param_groups
                            for parameter in group["params"]
                        ),
                        config.max_grad_norm,
                    )
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1
                reduced_loss = accelerator.reduce(loss.detach(), reduction="mean")
                observed_loss += float(reduced_loss.item())
                observed_updates += 1
                if global_step % config.save_every_steps == 0:
                    _save_training_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        checkpoint_dir=output_dir / f"checkpoint-{global_step:08d}",
                        config=config,
                        data_split=split_identity,
                        data_sha256=data_sha256,
                        step=global_step,
                        epoch=epoch,
                        next_batch=batch_index + 1,
                    )
        skip_batches = 0

    final_checkpoint = output_dir / "checkpoint-final"
    _save_training_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        checkpoint_dir=final_checkpoint,
        config=config,
        data_split=split_identity,
        data_sha256=data_sha256,
        step=global_step,
        epoch=config.epochs,
        next_batch=0,
    )
    mean_loss = observed_loss / observed_updates if observed_updates else float("nan")
    if accelerator.is_main_process:
        _write_json(
            output_dir / "training_result.json",
            {
                "global_step": global_step,
                "mean_loss": mean_loss,
                "data_sha256": data_sha256,
                "final_checkpoint": str(final_checkpoint),
            },
        )
    accelerator.wait_for_everyone()
    return TrainingResult(
        output_dir=output_dir,
        final_checkpoint=final_checkpoint,
        global_step=global_step,
        mean_loss=mean_loss,
        data_sha256=data_sha256,
    )


def run_mock_training(
    output_dir: str | Path,
    *,
    seed: int = 42,
    split_name: str = "mock",
) -> Path:
    """Write a deterministic, dependency- and download-free training checkpoint.

    This small function is intentionally CLI-friendly: pass an output path and it
    returns the metadata path while exercising the checkpoint metadata contract.
    """
    destination = Path(output_dir) / "checkpoint-mock"
    record = {
        "example_id": "mock-0000",
        "split": split_name,
        "graph": {
            "graph_id": "mock-graph",
            "nodes": [{"id": "n0", "label": "mock", "attributes": {}}],
            "edges": [],
            "directed": False,
            "metadata": {},
        },
        "prompt": "Question: Is the mock node present?\nAnswer:",
        "answer": "yes",
    }
    data_sha256 = data_records_sha256([record])
    metadata = {
        "format": TEA_CHECKPOINT_FORMAT,
        "step": 0,
        "mock": True,
        "gnn": {
            "class": "mock.GraphSAGEEncoder",
            "config": {
                "input_dim": 8,
                "hidden_dim": 8,
                "output_dim": 8,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "state_sha256": "download-free",
        },
        "language_model": {
            "class": "mock.FrozenCausalLM",
            "name_or_path": "download-free",
            "config_sha256": "download-free",
            "state_sha256": "download-free",
        },
        "tokenizer": {
            "class": "mock.ByteTokenizer",
            "name_or_path": "download-free",
            "vocab_sha256": "download-free",
            "vocab_size": 256,
            "backend_sha256": None,
            "special_tokens": {},
        },
        "projector": {
            "class": "mock.GraphPrefixProjector",
            "config": {
                "graph_dim": 8,
                "lm_hidden_dim": 8,
                "prefix_tokens": 2,
                "hidden_dim": 8,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "state_sha256": "download-free",
        },
        "tensorizer_config": {
            "feature_dim": 8,
            "hash_salt": "mock",
            "include_node_id": True,
            "include_label": True,
            "include_attributes": True,
            "l2_normalize": True,
        },
        "tea_glm_config": {"mock": True},
        "training_config": {"seed": seed, "mode": "mock"},
        "data_split": {"counts": {split_name: 1}},
        "data_sha256": data_sha256,
    }
    metadata_path = save_checkpoint_metadata(destination / "metadata.json", metadata)
    _write_json(
        Path(output_dir) / "training_result.json",
        {
            "global_step": 0,
            "mean_loss": 0.0,
            "data_sha256": data_sha256,
            "final_checkpoint": str(destination),
            "mock": True,
        },
    )
    return metadata_path
