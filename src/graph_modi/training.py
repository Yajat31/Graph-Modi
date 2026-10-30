"""Projector training and checkpointing for the optional TEA-GLM model."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from graph_modi.models.soft_prompt import (
    SoftPromptGLM,
    load_soft_prompt_checkpoint_metadata,
    save_soft_prompt_checkpoint,
    soft_prompt_checkpoint_metadata,
)
from graph_modi.models.tea_glm import (
    TEA_CHECKPOINT_FORMAT,
    TEAGLM,
    DeterministicNodeTensorizer,
    GraphSAGEEncoder,
    GraphTensor,
    assert_checkpoint_compatible,
    data_records_sha256,
    load_checkpoint_metadata,
    model_checkpoint_metadata,
    require_tea_dependencies,
    save_checkpoint_metadata,
)
from graph_modi.schema import AttributedGraph
from graph_modi.utils.progress import ProgressTracker


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
    # GraphToken trains the GNN jointly with the projector; TEA keeps it frozen.
    train_gnn: bool = False
    # Optimiser groups: the GNN may use its own lr / weight decay (None = share the projector's).
    gnn_learning_rate: float | None = None
    gnn_weight_decay: float | None = None
    # Learning-rate schedule applied to every group as a multiplier of its base lr.
    # Stateless in ``global_step``, so it resumes exactly.
    lr_schedule: str = "constant"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0
    # Held-out loss: every ``eval_every_steps`` updates (0 = off) on up to ``val_max_examples``
    # validation examples; the best-loss weights are kept in ``checkpoint-best``.
    eval_every_steps: int = 0
    val_max_examples: int = 1000
    # Warm start: load GNN + projector weights from this checkpoint before training.
    init_checkpoint: str | Path | None = None
    # Wrap each training prompt in the model's ``prompt_template`` (``Question: ...\nAnswer:``),
    # which inference always applies. False reproduces the pre-v4.1 bare-question training.
    wrap_prompts: bool = False

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
        if self.lr_schedule not in {"constant", "linear", "cosine"}:
            raise ValueError("lr_schedule must be one of: constant, linear, cosine")
        if self.warmup_steps < 0 or self.eval_every_steps < 0 or self.val_max_examples < 0:
            raise ValueError("warmup_steps, eval_every_steps and val_max_examples must be >= 0")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        for value in (self.gnn_learning_rate, self.gnn_weight_decay):
            if value is not None and value < 0:
                raise ValueError("gnn_learning_rate and gnn_weight_decay must be non-negative")
        if self.gnn_learning_rate is not None and self.gnn_learning_rate == 0:
            raise ValueError("gnn_learning_rate must be positive (freeze the GNN via train_gnn)")

    def metadata(self) -> dict[str, Any]:
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        values["resume_from"] = str(self.resume_from) if self.resume_from else None
        values["init_checkpoint"] = str(self.init_checkpoint) if self.init_checkpoint else None
        return values


def lr_multiplier(
    step: int,
    total_steps: int,
    *,
    schedule: str = "constant",
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> float:
    """Learning-rate multiplier for optimiser update number ``step`` (0-based)."""
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    if schedule == "linear":
        shape = 1.0 - progress
    else:
        import math

        shape = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * shape


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


def _collate_batch(
    tensorizer: DeterministicNodeTensorizer,
    batch: Sequence[TrainingExample],
    device: Any,
) -> tuple[GraphTensor, Any, list[dict[str, int]]]:
    """Merge a batch of graphs into one block-diagonal graph.

    Concatenating disjoint graphs with offset edge indices is mathematically
    equivalent to running GraphSAGE on each graph independently, since no
    edge ever crosses between the original graphs — but it replaces one
    forward pass per example with a single batched forward pass.
    """
    import torch

    tensor_graphs = [tensorizer(example.graph).to(device) for example in batch]
    node_maps: list[dict[str, int]] = []
    xs, edge_sources, edge_targets, edge_weights, node_batch = [], [], [], [], []
    offset = 0
    for graph_index, tensor_graph in enumerate(tensor_graphs):
        count = tensor_graph.x.shape[0]
        node_maps.append(
            {node_id: offset + position for position, node_id in enumerate(tensor_graph.node_ids)}
        )
        xs.append(tensor_graph.x)
        edge_sources.append(tensor_graph.edge_index[0] + offset)
        edge_targets.append(tensor_graph.edge_index[1] + offset)
        edge_weights.append(tensor_graph.edge_weight)
        node_batch.append(torch.full((count,), graph_index, dtype=torch.long, device=device))
        offset += count
    merged = GraphTensor(
        x=torch.cat(xs, dim=0),
        edge_index=torch.stack([torch.cat(edge_sources), torch.cat(edge_targets)], dim=0),
        edge_weight=torch.cat(edge_weights),
        node_ids=(),
        graph_id="batch",
    )
    return merged, torch.cat(node_batch), node_maps


def _batch_query_vectors(
    gnn: GraphSAGEEncoder,
    tensorizer: DeterministicNodeTensorizer,
    batch: Sequence[TrainingExample],
    device: Any,
) -> Any:
    """Query-conditioned graph vector concatenated with the query's
    source/target node representations, so the classifier can see *which*
    nodes a question refers to rather than only the graph as an
    undifferentiated whole. Computed for the whole batch via one merged
    forward pass. Pooling is task-conditional (zeroed whenever a source or
    target node anchor is present) to match tea_glm.py::TEAGLM.encode_graphs,
    which this classification head's representation is meant to mirror --
    see that function's docstring for why (arXiv:2508.20583's node-level vs.
    graph-level readout switch)."""
    import torch

    merged, node_batch, node_maps = _collate_batch(tensorizer, batch, device)
    node_repr = gnn(merged)
    dim = node_repr.shape[-1]
    count = len(batch)
    pooled_sum = torch.zeros((count, dim), dtype=node_repr.dtype, device=device)
    pooled_sum.index_add_(0, node_batch, node_repr)
    node_count = torch.zeros((count, 1), dtype=node_repr.dtype, device=device)
    node_count.index_add_(
        0, node_batch, torch.ones((node_repr.shape[0], 1), dtype=node_repr.dtype, device=device)
    )
    pooled = pooled_sum / node_count.clamp_min(1.0)
    zeros = torch.zeros(dim, dtype=node_repr.dtype, device=device)
    source_rows = []
    target_rows = []
    has_anchor = []
    for example, node_map in zip(batch, node_maps, strict=True):
        source_id = example.metadata.get("source_id")
        target_id = example.metadata.get("target_id")
        source_rows.append(node_repr[node_map[source_id]] if source_id in node_map else zeros)
        target_rows.append(node_repr[node_map[target_id]] if target_id in node_map else zeros)
        has_anchor.append(source_id in node_map or target_id in node_map)
    anchor_mask = torch.tensor(has_anchor, dtype=torch.bool, device=device).unsqueeze(-1)
    pooled = pooled.masked_fill(anchor_mask, 0.0)
    return torch.cat([pooled, torch.stack(source_rows), torch.stack(target_rows)], dim=-1)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn.to(device)

    labels = tuple(sorted({example.answer for example in examples}))
    label_to_id = {label: index for index, label in enumerate(labels)}
    head = nn.Linear(gnn.config.output_dim * 3, len(labels)).to(device)
    optimizer = torch.optim.AdamW(
        [*gnn.parameters(), *head.parameters()],
        lr=learning_rate,
    )
    rng = random.Random(seed)
    ordered = list(examples)
    batches_per_epoch = (len(ordered) + batch_size - 1) // batch_size
    total_batches = batches_per_epoch * epochs
    tracker = ProgressTracker("[pretrain-gnn]", total_batches, phase="gnn")
    tracker.banner(examples=len(ordered), epochs=epochs, batch_size=batch_size, device=str(device))
    batch_counter = 0
    for epoch in range(epochs):
        rng.shuffle(ordered)
        gnn.train()
        head.train()
        epoch_loss = 0.0
        epoch_correct = 0
        for batch_index, start in enumerate(range(0, len(ordered), batch_size)):
            batch = ordered[start : start + batch_size]
            vectors = _batch_query_vectors(gnn, tensorizer, batch, device)
            targets = torch.tensor(
                [label_to_id[example.answer] for example in batch],
                dtype=torch.long,
                device=device,
            )
            logits = head(vectors)
            loss = nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([*gnn.parameters(), *head.parameters()], max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch)
            epoch_correct += int((logits.argmax(dim=-1) == targets).sum().item())
            batch_counter += 1
            tracker.tick(
                loss=f"{loss.item():.4f}",
                acc=f"{epoch_correct / max(1, (batch_index + 1) * batch_size):.4f}",
                epoch=f"{epoch + 1}/{epochs}",
            )
        tracker.end(
            mean_loss=f"{epoch_loss / len(ordered):.4f}",
            train_acc=f"{epoch_correct / len(ordered):.4f}",
        )
    gnn.eval()
    head.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            logits = head(_batch_query_vectors(gnn, tensorizer, chunk, device))
            predictions = logits.argmax(dim=-1).tolist()
            correct += sum(
                labels[prediction] == example.answer
                for example, prediction in zip(chunk, predictions, strict=True)
            )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "gnn.pt"
    head_checkpoint = destination / "gnn_head.pt"
    torch.save({key: value.cpu() for key, value in gnn.state_dict().items()}, checkpoint)
    torch.save({key: value.cpu() for key, value in head.state_dict().items()}, head_checkpoint)
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


def pretrain_graph_encoder_contrastive(
    gnn: GraphSAGEEncoder,
    tensorizer: DeterministicNodeTensorizer,
    graphs: Sequence[AttributedGraph],
    language_model: Any,
    tokenizer: Any,
    *,
    output_dir: str | Path,
    epochs: int,
    graphs_per_batch: int,
    learning_rate: float,
    temperature: float = 0.07,
    seed: int,
    target: str = "mean",
    status_weight: float = 4.0,
    dropout: float | None = None,
    eval_graphs: Sequence[AttributedGraph] = (),
) -> GNNPretrainingResult:
    """TEA-GLM's own pretraining objective (arXiv:2408.14512, reused by the
    CLEGR paper as its TEA-GLM baseline): align each node's GNN embedding
    with that node's own textual description, embedded via the frozen LLM's
    word-embedding matrix, via a symmetric InfoNCE contrastive loss --
    instead of this project's earlier proxy of classifying the exact answer
    string from pooled+source+target (a proxy the classification head could
    never solve for most of the ~1800 real-CLEGR answer classes, and which
    the CLEGR paper's own TEA-GLM baseline never uses).

    Self-supervised: needs only the graphs themselves, no QA labels, so
    every node in every training graph is a training signal (not just the
    ones some QA example happens to name as source/target).

    A throwaway linear "alignment head" (gnn.output_dim -> LLM hidden dim)
    is trained jointly with the GNN to make the contrastive similarity
    well-defined across the two different dimensionalities, then discarded
    (its weights are saved next to the GNN for probing): gnn.pt is saved in the
    exact same format pretrain_graph_encoder already produces, so nothing
    downstream needs to change to consume it.

    ``target`` picks what a node is aligned to: ``mean`` (mean LLM input embedding of
    "name: attrs", the original), ``attrs_only`` (the same without the unique station name,
    which otherwise dominates the match), or ``status_weighted`` (the ``mean`` target plus
    ``status_weight`` times the embedding of the status value, since the multi-turn task turns
    on status). ``dropout`` overrides the GNN's dropout for pretraining only. ``eval_graphs``
    (held out) are scored for node retrieval at the end; the numbers go into metadata.json.
    ``language_model`` only needs ``get_input_embeddings()``.
    """
    require_tea_dependencies()
    if not graphs:
        raise ValueError("At least one graph is required for contrastive pretraining")
    if min(epochs, graphs_per_batch) <= 0 or learning_rate <= 0:
        raise ValueError("epochs, graphs_per_batch, and learning_rate must be positive")
    if target not in {"mean", "attrs_only", "status_weighted"}:
        raise ValueError("target must be one of: mean, attrs_only, status_weighted")

    import torch
    from torch import nn

    set_deterministic_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn.to(device)

    embedding_layer = language_model.get_input_embeddings()
    for parameter in embedding_layer.parameters():
        parameter.requires_grad_(False)
    lm_hidden_dim = int(embedding_layer.embedding_dim)

    def _node_text(node: Any) -> str:
        attrs = ", ".join(f"{key}={value}" for key, value in sorted(node.attributes.items()))
        if target == "attrs_only":
            return attrs or node.label
        return f"{node.label}: {attrs}" if attrs else node.label

    text_cache: dict[str, Any] = {}

    def _mean_embedding(text: str) -> Any:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            token_ids = [tokenizer.unk_token_id or 0]
        with torch.no_grad():
            # Index on the embedding table's own device (it may differ from the GNN's device).
            table_device = embedding_layer.weight.device
            token_embeds = embedding_layer(torch.tensor(token_ids, device=table_device))
            return token_embeds.mean(dim=0).float()

    def _text_embedding(text: str, node: Any = None) -> Any:
        key = f"{text}|{node.attributes.get('status') if node is not None else ''}"
        if key not in text_cache:
            vector = _mean_embedding(text)
            if target == "status_weighted" and node is not None and "status" in node.attributes:
                status = _mean_embedding(str(node.attributes["status"]))
                vector = nn.functional.normalize(vector, dim=0) + status_weight * (
                    nn.functional.normalize(status, dim=0)
                )
            text_cache[key] = vector
        return text_cache[key]

    def _node_targets(batch_graphs: Sequence[AttributedGraph]) -> Any:
        nodes = [node for graph in batch_graphs for node in sorted(graph.nodes, key=lambda n: n.id)]
        return nn.functional.normalize(
            torch.stack([_text_embedding(_node_text(node), node) for node in nodes]).to(device),
            dim=-1,
        )

    alignment_head = nn.Linear(gnn.config.output_dim, lm_hidden_dim).to(device)
    # TEAGLM freezes its GNN on construction, and the CLI hands that frozen GNN to this routine.
    # Without this the optimiser would only ever move the throwaway alignment head and the saved
    # "pretrained" GNN would be its random initialisation.
    original_requires_grad = [parameter.requires_grad for parameter in gnn.parameters()]
    gnn.requires_grad_(True)
    original_dropout = gnn.dropout.p
    if dropout is not None:
        gnn.dropout.p = dropout
    optimizer = torch.optim.AdamW(
        [*gnn.parameters(), *alignment_head.parameters()], lr=learning_rate
    )

    rng = random.Random(seed)
    ordered_graphs = list(graphs)
    batches_per_epoch = (len(ordered_graphs) + graphs_per_batch - 1) // graphs_per_batch
    total_batches = batches_per_epoch * epochs
    tracker = ProgressTracker("[pretrain-gnn-contrastive]", total_batches, phase="gnn_contrastive")
    tracker.banner(
        graphs=len(ordered_graphs),
        epochs=epochs,
        graphs_per_batch=graphs_per_batch,
        device=str(device),
    )

    class _GraphExample:
        __slots__ = ("graph",)

        def __init__(self, graph: AttributedGraph) -> None:
            self.graph = graph

    batch_counter = 0
    final_loss = 0.0
    loss_history: list[float] = []
    chance_sum = 0.0
    chance_count = 0
    for epoch in range(epochs):
        rng.shuffle(ordered_graphs)
        gnn.train()
        alignment_head.train()
        epoch_loss = 0.0
        epoch_batches = 0
        for start in range(0, len(ordered_graphs), graphs_per_batch):
            batch_graphs = ordered_graphs[start : start + graphs_per_batch]
            wrapped = [_GraphExample(graph) for graph in batch_graphs]
            merged, node_batch, node_maps = _collate_batch(tensorizer, wrapped, device)
            node_repr = gnn(merged)
            projected = alignment_head(node_repr)
            projected = nn.functional.normalize(projected, dim=-1)

            text_embeds = _node_targets(batch_graphs)

            logits = projected @ text_embeds.T / temperature
            labels = torch.arange(logits.shape[0], device=device)
            loss = (
                nn.functional.cross_entropy(logits, labels)
                + nn.functional.cross_entropy(logits.T, labels)
            ) / 2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [*gnn.parameters(), *alignment_head.parameters()], max_norm=1.0
            )
            optimizer.step()
            epoch_loss += float(loss.item())
            epoch_batches += 1
            batch_counter += 1
            chance_sum += math.log(logits.shape[0])
            chance_count += 1
            tracker.tick(loss=f"{loss.item():.4f}", epoch=f"{epoch + 1}/{epochs}")
        final_loss = epoch_loss / max(1, epoch_batches)
        loss_history.append(final_loss)
        tracker.end(mean_loss=f"{final_loss:.4f}")

    gnn.eval()
    for parameter, flag in zip(gnn.parameters(), original_requires_grad, strict=True):
        parameter.requires_grad_(flag)
    gnn.dropout.p = original_dropout

    def _retrieval(held_out: Sequence[AttributedGraph]) -> dict[str, float]:
        """Top-1 node retrieval on held-out graphs: against the nodes of the 8-graph batch
        (the training task) and against only the nodes of the node's own graph."""
        in_batch = within = total = 0
        with torch.no_grad():
            for start in range(0, len(held_out), graphs_per_batch):
                batch_graphs = list(held_out[start : start + graphs_per_batch])
                wrapped = [_GraphExample(graph) for graph in batch_graphs]
                merged, _, _ = _collate_batch(tensorizer, wrapped, device)
                projected = nn.functional.normalize(alignment_head(gnn(merged)), dim=-1)
                targets = _node_targets(batch_graphs)
                scores = projected @ targets.T
                in_batch += int(
                    (scores.argmax(dim=1) == torch.arange(len(scores), device=device)).sum()
                )
                total += len(scores)
                offset = 0
                for graph in batch_graphs:
                    size = len(graph.nodes)
                    block = scores[offset : offset + size, offset : offset + size]
                    within += int((block.argmax(dim=1) == torch.arange(size, device=device)).sum())
                    offset += size
        return {
            "retrieval_top1_in_batch": in_batch / max(1, total),
            "retrieval_top1_within_graph": within / max(1, total),
            "retrieval_nodes": total,
        }

    retrieval = _retrieval(list(eval_graphs)) if eval_graphs else {}
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(
        {key: value.cpu() for key, value in alignment_head.state_dict().items()},
        destination / "alignment_head.pt",
    )
    checkpoint = destination / "gnn.pt"
    torch.save({key: value.cpu() for key, value in gnn.state_dict().items()}, checkpoint)
    metadata_path = destination / "metadata.json"
    _write_json(
        metadata_path,
        {
            "format": "graph-modi-gnn-v1",
            "gnn_config": asdict(gnn.config),
            "tensorizer_config": asdict(tensorizer.config),
            "labels": (),
            # Hash of the graphs' content (not just their ids).
            "data_sha256": data_records_sha256([g.to_dict() for g in graphs]),
            "seed": seed,
            "epochs": epochs,
            "objective": "contrastive",
            "target": target,
            "status_weight": status_weight if target == "status_weighted" else None,
            "temperature": temperature,
            "dropout": dropout if dropout is not None else original_dropout,
            "learning_rate": learning_rate,
            "final_loss": final_loss,
            "loss_history": loss_history,
            # InfoNCE loss of a model that matches nodes at random (mean over batches).
            "chance_loss": chance_sum / max(1, chance_count),
            **retrieval,
        },
    )
    return GNNPretrainingResult(
        checkpoint=checkpoint,
        metadata=metadata_path,
        labels=(),
        training_accuracy=final_loss,
    )


def _write_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value) if isinstance(value, Mapping) else list(value)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
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
    include_optimizer: bool = True,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        accelerator.save(unwrapped.gnn.state_dict(), checkpoint_dir / "gnn.pt")
        accelerator.save(unwrapped.projector.state_dict(), checkpoint_dir / "projector.pt")
        if include_optimizer:
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
    """Restore GNN + projector weights from ``checkpoint_dir`` and return the trainer position.

    The GNN weights are loaded *before* the compatibility check: the projector metadata binds to
    the GNN's state hash, which for a jointly trained GNN (GraphToken) differs from the freshly
    initialised one the model was built with.
    """
    import torch

    metadata = load_checkpoint_metadata(checkpoint_dir / "metadata.json")
    if metadata.get("data_sha256") != data_sha256:
        raise ValueError("Resume checkpoint was created from different training examples")
    if metadata.get("data_split") != dict(data_split):
        raise ValueError("Resume checkpoint uses different data-split metadata")
    model.gnn.load_state_dict(
        torch.load(checkpoint_dir / "gnn.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    assert_checkpoint_compatible(model, metadata, component="projector")
    model.projector.load_state_dict(
        torch.load(checkpoint_dir / "projector.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    state = json.loads((checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8"))
    return int(state["global_step"]), int(state["epoch"]), int(state["next_batch"])


def wrap_prompts(examples: Sequence[TrainingExample], template: str) -> list[TrainingExample]:
    """Apply an inference prompt template to every example's prompt."""
    return [
        replace(example, prompt=template.format(question=example.prompt)) for example in examples
    ]


def _validation_loss(
    model: Any,
    val_examples: Sequence[TrainingExample],
    batch_size: int,
) -> float:
    """Mean answer loss over ``val_examples`` (teacher-forced, no gradient, dropout off)."""
    import torch

    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(val_examples), batch_size):
            batch = val_examples[start : start + batch_size]
            output = model(
                graphs=[example.graph for example in batch],
                prompts=[example.prompt for example in batch],
                answers=[example.answer for example in batch],
                source_ids=[example.metadata.get("source_id") for example in batch],
                target_ids=[example.metadata.get("target_id") for example in batch],
                hops=[example.metadata.get("hops") for example in batch],
                reasoning_types=[example.metadata.get("reasoning_type") for example in batch],
            )
            total += float(output.loss.item()) * len(batch)
            count += len(batch)
    if was_training:
        model.train()
    return total / max(1, count)


def train_projector(
    model: TEAGLM,
    examples: Sequence[TrainingExample],
    config: ProjectorTrainingConfig,
    *,
    data_split: Mapping[str, Any] | None = None,
    val_examples: Sequence[TrainingExample] | None = None,
) -> TrainingResult:
    """Train graph prefix tokens with full answer loss and a frozen causal LM.

    The GNN is frozen unless ``config.train_gnn`` is set (GraphToken), in which case it is
    optimised jointly with the projector (its own lr / weight decay, clipped separately).
    ``val_examples`` enables the held-out loss curve and ``checkpoint-best``.
    """
    require_tea_dependencies()
    if not examples:
        raise ValueError("At least one training example is required")
    if config.wrap_prompts:
        examples = wrap_prompts(examples, model.config.prompt_template)
        if val_examples:
            val_examples = wrap_prompts(val_examples, model.config.prompt_template)
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
    if config.init_checkpoint is not None:
        if config.resume_from is not None:
            raise ValueError("init_checkpoint and resume_from are mutually exclusive")
        from graph_modi.models.tea_glm import load_external_component

        init_dir = Path(config.init_checkpoint)
        for component in ("gnn", "projector"):
            load_external_component(
                model,
                component=component,
                weights_path=init_dir / f"{component}.pt",
                metadata_path=init_dir / "metadata.json",
            )
    model.set_gnn_frozen(not config.train_gnn)
    for parameter in model.language_model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.projector.parameters():
        parameter.requires_grad_(True)
    param_groups: list[dict[str, Any]] = [
        {
            "params": list(model.projector.parameters()),
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
    ]
    if config.train_gnn:
        param_groups.append(
            {
                "params": list(model.gnn.parameters()),
                "lr": config.gnn_learning_rate or config.learning_rate,
                "weight_decay": (
                    config.weight_decay
                    if config.gnn_weight_decay is None
                    else config.gnn_weight_decay
                ),
            }
        )
    base_lrs = [group["lr"] for group in param_groups]
    trainable = [parameter for group in param_groups for parameter in group["params"]]

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
    optimizer = torch.optim.AdamW(param_groups)
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
        optimizer.load_state_dict(
            torch.load(
                checkpoint_dir / "optimizer.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        # Replay the shuffles of the finished epochs so the resumed epoch sees the same
        # example order an uninterrupted run would (the sampler draws from ``generator``).
        for _ in range(first_epoch):
            for _ in loader:
                pass

    import time

    micro_batches_per_epoch = (len(examples) + config.batch_size - 1) // config.batch_size
    steps_per_epoch = max(1, micro_batches_per_epoch // config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.epochs
    log_every = max(1, steps_per_epoch // 20)
    start_time = time.monotonic()
    if accelerator.is_main_process:
        print(
            f"[train-projector] device={accelerator.device} examples={len(examples)} "
            f"steps/epoch={steps_per_epoch} epochs={config.epochs} total_steps={total_steps} "
            f"train_gnn={config.train_gnn} "
            f"trainable_params={sum(parameter.numel() for parameter in trainable)} "
            f"lr={config.learning_rate} gnn_lr={param_groups[-1]['lr']} "
            f"schedule={config.lr_schedule} warmup={config.warmup_steps}",
            flush=True,
        )

    observed_loss = 0.0
    observed_updates = 0
    epoch_loss = 0.0
    epoch_updates = 0
    val_examples = list(val_examples)[: config.val_max_examples] if val_examples else []
    val_history: list[dict[str, float]] = []
    best_val = float("inf")
    if val_examples and config.eval_every_steps and accelerator.is_main_process:
        print(
            f"[train-projector] validation loss every {config.eval_every_steps} steps "
            f"on {len(val_examples)} examples",
            flush=True,
        )
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
                    source_ids=[example.metadata.get("source_id") for example in batch],
                    target_ids=[example.metadata.get("target_id") for example in batch],
                    hops=[example.metadata.get("hops") for example in batch],
                    reasoning_types=[example.metadata.get("reasoning_type") for example in batch],
                )
                loss = output.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    multiplier = lr_multiplier(
                        global_step,
                        total_steps,
                        schedule=config.lr_schedule,
                        warmup_steps=config.warmup_steps,
                        min_lr_ratio=config.min_lr_ratio,
                    )
                    for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
                        group["lr"] = base_lr * multiplier
                    if config.max_grad_norm:
                        # Clip each group on its own so a 100M+ parameter GNN cannot dominate
                        # (and so shrink) the projector's gradient.
                        for group in optimizer.param_groups:
                            accelerator.clip_grad_norm_(group["params"], config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1
                reduced_loss = accelerator.reduce(loss.detach(), reduction="mean")
                loss_value = float(reduced_loss.item())
                observed_loss += loss_value
                observed_updates += 1
                epoch_loss += loss_value
                epoch_updates += 1
                if accelerator.is_main_process and (
                    global_step % log_every == 0 or global_step == total_steps
                ):
                    elapsed = time.monotonic() - start_time
                    print(
                        f"[train-projector] step {global_step}/{total_steps} "
                        f"(epoch {epoch + 1}/{config.epochs}) loss={loss_value:.4f} "
                        f"elapsed={elapsed:.0f}s",
                        flush=True,
                    )
                if (
                    val_examples
                    and config.eval_every_steps
                    and (global_step % config.eval_every_steps == 0 or global_step == total_steps)
                ):
                    val_loss = _validation_loss(model, val_examples, config.batch_size)
                    val_history.append({"step": global_step, "val_loss": val_loss})
                    if accelerator.is_main_process:
                        print(
                            f"[train-projector] step {global_step} val_loss={val_loss:.4f} "
                            f"(n={len(val_examples)}, best={min(best_val, val_loss):.4f})",
                            flush=True,
                        )
                        _write_json(output_dir / "val_history.json", val_history)
                    if val_loss < best_val:
                        best_val = val_loss
                        _save_training_checkpoint(
                            accelerator=accelerator,
                            model=model,
                            optimizer=optimizer,
                            checkpoint_dir=output_dir / "checkpoint-best",
                            config=config,
                            data_split=split_identity,
                            data_sha256=data_sha256,
                            step=global_step,
                            epoch=epoch,
                            next_batch=batch_index + 1,
                            include_optimizer=False,
                        )
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
        if accelerator.is_main_process and epoch_updates:
            print(
                f"[train-projector] epoch {epoch + 1}/{config.epochs} done "
                f"mean_loss={epoch_loss / epoch_updates:.4f} "
                f"elapsed={time.monotonic() - start_time:.0f}s",
                flush=True,
            )
        epoch_loss = 0.0
        epoch_updates = 0
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
                "best_val_loss": best_val if val_history else None,
                "val_history": val_history,
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


def _save_soft_prompt_training_checkpoint(
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
        metadata = soft_prompt_checkpoint_metadata(
            unwrapped,
            training_config=config.metadata(),
            data_split=data_split,
            data_sha256=data_sha256,
            step=step,
        )
        save_soft_prompt_checkpoint(checkpoint_dir, unwrapped, metadata)
        accelerator.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        _write_json(
            checkpoint_dir / "trainer_state.json",
            {"global_step": step, "epoch": epoch, "next_batch": next_batch},
        )
    accelerator.wait_for_everyone()


def _validate_soft_prompt_resume(
    checkpoint_dir: Path,
    *,
    data_split: Mapping[str, Any],
    data_sha256: str,
) -> tuple[int, int, int]:
    metadata = load_soft_prompt_checkpoint_metadata(checkpoint_dir / "metadata.json")
    if metadata.get("data_sha256") != data_sha256:
        raise ValueError("Resume checkpoint was created from different training examples")
    if metadata.get("data_split") != dict(data_split):
        raise ValueError("Resume checkpoint uses different data-split metadata")
    state = json.loads((checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8"))
    return int(state["global_step"]), int(state["epoch"]), int(state["next_batch"])


def train_soft_prompt(
    model: SoftPromptGLM,
    examples: Sequence[TrainingExample],
    config: ProjectorTrainingConfig,
    *,
    data_split: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """Train only the shared soft-prompt matrix; frozen LM, no GNN, no graph batching."""
    require_tea_dependencies()
    if not examples:
        raise ValueError("At least one training example is required")
    if config.wrap_prompts:
        examples = wrap_prompts(examples, model.config.prompt_template)
    try:
        import torch
        from accelerate import Accelerator
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "Soft-prompt training requires Accelerate. "
            "Install dependencies with `pip install 'graph-modi[tea]'`."
        ) from exc

    set_deterministic_seed(config.seed)
    for parameter in model.language_model.parameters():
        parameter.requires_grad_(False)
    model.soft_prompt.requires_grad_(True)

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
        [model.soft_prompt], lr=config.learning_rate, weight_decay=config.weight_decay
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    global_step = 0
    first_epoch = 0
    skip_batches = 0
    if config.resume_from is not None:
        checkpoint_dir = Path(config.resume_from)
        unwrapped = accelerator.unwrap_model(model)
        global_step, first_epoch, skip_batches = _validate_soft_prompt_resume(
            checkpoint_dir,
            data_split=split_identity,
            data_sha256=data_sha256,
        )
        state = torch.load(checkpoint_dir / "soft_prompt.pt", map_location="cpu", weights_only=True)
        with torch.no_grad():
            unwrapped.soft_prompt.copy_(state.to(unwrapped.soft_prompt.device))
        optimizer.load_state_dict(
            torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=True)
        )

    import time

    micro_batches_per_epoch = (len(examples) + config.batch_size - 1) // config.batch_size
    steps_per_epoch = max(1, micro_batches_per_epoch // config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.epochs
    log_every = max(1, steps_per_epoch // 20)
    start_time = time.monotonic()
    if accelerator.is_main_process:
        print(
            f"[train-projector] stage=soft_prompt device={accelerator.device} "
            f"examples={len(examples)} steps/epoch={steps_per_epoch} epochs={config.epochs} "
            f"total_steps={total_steps}",
            flush=True,
        )

    observed_loss = 0.0
    observed_updates = 0
    epoch_loss = 0.0
    epoch_updates = 0
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
                    prompts=[example.prompt for example in batch],
                    answers=[example.answer for example in batch],
                )
                loss = output.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients and config.max_grad_norm:
                    accelerator.clip_grad_norm_([model.soft_prompt], config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1
                reduced_loss = accelerator.reduce(loss.detach(), reduction="mean")
                loss_value = float(reduced_loss.item())
                observed_loss += loss_value
                observed_updates += 1
                epoch_loss += loss_value
                epoch_updates += 1
                if accelerator.is_main_process and (
                    global_step % log_every == 0 or global_step == total_steps
                ):
                    elapsed = time.monotonic() - start_time
                    print(
                        f"[train-projector] stage=soft_prompt step {global_step}/{total_steps} "
                        f"(epoch {epoch + 1}/{config.epochs}) loss={loss_value:.4f} "
                        f"elapsed={elapsed:.0f}s",
                        flush=True,
                    )
                if global_step % config.save_every_steps == 0:
                    _save_soft_prompt_training_checkpoint(
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
        if accelerator.is_main_process and epoch_updates:
            print(
                f"[train-projector] stage=soft_prompt epoch {epoch + 1}/{config.epochs} done "
                f"mean_loss={epoch_loss / epoch_updates:.4f} "
                f"elapsed={time.monotonic() - start_time:.0f}s",
                flush=True,
            )
        epoch_loss = 0.0
        epoch_updates = 0
        skip_batches = 0

    final_checkpoint = output_dir / "checkpoint-final"
    _save_soft_prompt_training_checkpoint(
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
