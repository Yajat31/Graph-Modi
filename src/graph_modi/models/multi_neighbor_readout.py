"""Exploratory: per-neighbor graph-prefix tokens instead of one pooled vector.

Throwaway probe for the aggregation-heavy task families that stayed near floor
even after the hops/reasoning_type fixes (`filtered_neighbor_count`,
`most_common_attribute_within_hops`, `within_hops_count`, `within_hops_list`).
Those all require aggregating/comparing several neighbor nodes; the production
readout (`TEAGLM.encode_graphs`) collapses the whole neighborhood into one
pooled vector before the frozen LLM ever sees it, so there is nothing left for
the LLM to aggregate over. This module instead emits one graph-prefix token
per 1-hop neighbor (zero-padded/truncated to a fixed slot count) alongside the
existing pooled+source+target+hops+task summary tokens, so the LLM can attend
over individual neighbors itself.

This is NOT wired into the main train/eval pipeline (`cli.py`/`training.py`)
and does not retrain the GNN: it reuses an already-trained TEA checkpoint's
frozen GNN/tensorizer/LM verbatim and trains only the new neighbor projector
from scratch. Driven by `scripts/explore_multi_neighbor_readout.py`.

If this measurably helps the aggregation tasks without regressing the
topology tasks that already work, promote it out of "experimental" into a
real, configurable readout mode on both TEA and GraphToken -- do not delete
the pooled path, keep both available.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from graph_modi.models.tea_glm import (
    TEAGLM,
    GraphTensor,
    _Module,
    _reasoning_type_onehot,
    require_tea_dependencies,
)
from graph_modi.schema import AttributedGraph

_TEA_IMPORT_ERROR: ImportError | None = None
try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on optional environment
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TEA_IMPORT_ERROR = exc


@dataclass(frozen=True, slots=True)
class MultiNeighborConfig:
    graph_dim: int
    lm_hidden_dim: int
    neighbor_slots: int = 8
    hidden_dim: int = 512
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(self.graph_dim, self.lm_hidden_dim, self.neighbor_slots, self.hidden_dim) <= 0:
            raise ValueError("MultiNeighborConfig dimensions and neighbor_slots must be positive")


class NeighborTokenProjector(_Module):
    """Maps K individual neighbor embeddings to K separate LM-embedding tokens
    (one shared MLP applied per slot) instead of pooling them into one vector."""

    def __init__(self, config: MultiNeighborConfig) -> None:
        require_tea_dependencies()
        super().__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(config.graph_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.lm_hidden_dim),
        )

    def forward(self, neighbor_vectors: Any) -> Any:
        if neighbor_vectors.ndim != 3 or neighbor_vectors.shape[-1] != self.config.graph_dim:
            raise ValueError(
                f"Expected neighbor vectors [B, {self.config.neighbor_slots}, "
                f"{self.config.graph_dim}], got {tuple(neighbor_vectors.shape)}"
            )
        return self.mlp(neighbor_vectors)


def _gather_neighbor_rows(
    node_repr: Any,
    tensor_graph: GraphTensor,
    source_id: str | None,
    slots: int,
) -> Any:
    """First-hop neighbor embeddings of ``source_id``, zero-padded/truncated to ``slots``.

    Deliberately 1-hop only for this probe regardless of the query's own
    ``hops`` value (2 in the current task set) -- a quick way to test whether
    per-neighbor tokens help at all before investing in a hop-aware version.
    """
    zeros = torch.zeros(node_repr.shape[-1], dtype=node_repr.dtype, device=node_repr.device)
    node_index = {node_id: position for position, node_id in enumerate(tensor_graph.node_ids)}
    if source_id not in node_index:
        return torch.stack([zeros] * slots)
    source_position = node_index[source_id]
    sources, destinations = tensor_graph.edge_index
    neighbor_positions = sorted(
        {
            int(destinations[i])
            for i in range(sources.shape[0])
            if int(sources[i]) == source_position
        }
    )
    rows = [node_repr[position] for position in neighbor_positions[:slots]]
    while len(rows) < slots:
        rows.append(zeros)
    return torch.stack(rows[:slots])


class MultiNeighborTEAGLM(TEAGLM):
    """TEAGLM variant whose graph prefix is [summary tokens] + [K neighbor
    tokens] instead of just [summary tokens] from one pooled vector.

    Reuses an already-trained base TEAGLM's frozen LM/GNN/tensorizer/summary
    projector verbatim; only ``graph_prefix`` differs, and only
    ``neighbor_projector`` is trainable.
    """

    def __init__(
        self,
        *,
        base: TEAGLM,
        neighbor_projector: NeighborTokenProjector,
        neighbor_slots: int,
    ) -> None:
        require_tea_dependencies()
        # Deliberately skip TEAGLM.__init__: its graph_dim validation assumes
        # the pooled-only architecture. Reuse the base model's already-valid
        # components directly instead.
        nn.Module.__init__(self)
        self.gnn = base.gnn
        self.projector = base.projector
        self.language_model = base.language_model
        self.tokenizer = base.tokenizer
        self.tensorizer = base.tensorizer
        self.config = base.config
        self.neighbor_projector = neighbor_projector
        self.neighbor_slots = neighbor_slots
        # Bookkeeping-only: _truncate()/forward() read
        # self.projector.config.prefix_tokens purely as "how many slots does
        # the graph prefix occupy" for masking/truncation math -- they never
        # rebuild self.projector's layers (already built for the *summary*
        # token count alone). Reporting the combined total here keeps that
        # accounting correct without duplicating forward()/generate().
        # graph_prefix() below bypasses self.projector.forward()'s own
        # reshape (which would otherwise use this same faked number and
        # break) by calling self.projector.network(...) directly.
        combined_tokens = self.projector.config.prefix_tokens + neighbor_slots
        self.projector.config = replace(self.projector.config, prefix_tokens=combined_tokens)
        self._gnn_frozen = True
        for parameter in self.gnn.parameters():
            parameter.requires_grad_(False)
        for parameter in self.projector.parameters():
            parameter.requires_grad_(False)
        for parameter in self.language_model.parameters():
            parameter.requires_grad_(False)
        self.language_model.eval()

    def graph_prefix(
        self,
        graphs: Sequence[AttributedGraph | GraphTensor],
        *,
        source_ids: Sequence[str | None] | None = None,
        target_ids: Sequence[str | None] | None = None,
        hops: Sequence[int | None] | None = None,
        reasoning_types: Sequence[str | None] | None = None,
    ) -> Any:
        if source_ids is None:
            source_ids = [None] * len(graphs)
        if target_ids is None:
            target_ids = [None] * len(graphs)
        if hops is None:
            hops = [None] * len(graphs)
        if reasoning_types is None:
            reasoning_types = [None] * len(graphs)
        embedding_dtype = self.language_model.get_input_embeddings().weight.dtype
        summary_vectors = []
        neighbor_batches = []
        for graph, source_id, target_id, hop_count, reasoning_type in zip(
            graphs, source_ids, target_ids, hops, reasoning_types, strict=True
        ):
            tensor_graph = self.tensorizer(graph) if isinstance(graph, AttributedGraph) else graph
            tensor_graph = tensor_graph.to(self.device)
            with torch.no_grad():
                node_repr = self.gnn(tensor_graph)
            pooled = node_repr.mean(dim=0)
            zeros = torch.zeros(node_repr.shape[-1], dtype=node_repr.dtype, device=node_repr.device)
            node_index = {
                node_id: position for position, node_id in enumerate(tensor_graph.node_ids)
            }
            source_row = node_repr[node_index[source_id]] if source_id in node_index else zeros
            target_row = node_repr[node_index[target_id]] if target_id in node_index else zeros
            hop_feature = torch.tensor(
                [float(hop_count) if hop_count is not None else 0.0],
                dtype=node_repr.dtype,
                device=node_repr.device,
            )
            task_feature = _reasoning_type_onehot(
                reasoning_type, dtype=node_repr.dtype, device=node_repr.device
            )
            summary_vectors.append(
                torch.cat([pooled, source_row, target_row, hop_feature, task_feature], dim=-1)
            )
            neighbor_batches.append(
                _gather_neighbor_rows(node_repr, tensor_graph, source_id, self.neighbor_slots)
            )
        summary_vectors = torch.stack(summary_vectors)
        neighbor_vectors = torch.stack(neighbor_batches)
        summary_tokens = self.projector.network(summary_vectors).view(
            summary_vectors.shape[0], -1, self.projector.config.lm_hidden_dim
        )
        neighbor_tokens = self.neighbor_projector(neighbor_vectors)
        return torch.cat([summary_tokens, neighbor_tokens], dim=1).to(dtype=embedding_dtype)


_NEIGHBOR_CHECKPOINT_FORMAT = "graph-modi-multi-neighbor-readout-v1"


def save_neighbor_checkpoint(
    neighbor_projector: NeighborTokenProjector, checkpoint_dir: str | Path
) -> None:
    """Persist the trained neighbor projector -- the exploratory harness
    originally trained and evaluated it in one process without ever writing
    it to disk, so a v3 run couldn't be reused for a later, separate dynamic
    eval. Mirrors the format/metadata style of the main checkpoint saver in
    training.py, scoped down to this one extra component."""
    require_tea_dependencies()
    output = Path(checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(neighbor_projector.state_dict(), output / "neighbor_projector.pt")
    metadata = {
        "format": _NEIGHBOR_CHECKPOINT_FORMAT,
        "config": asdict(neighbor_projector.config),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_neighbor_checkpoint(checkpoint_dir: str | Path) -> NeighborTokenProjector:
    """Reconstruct a trained NeighborTokenProjector from
    ``save_neighbor_checkpoint``'s output."""
    require_tea_dependencies()
    checkpoint = Path(checkpoint_dir)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != _NEIGHBOR_CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported neighbor-projector checkpoint metadata: {checkpoint}")
    config = MultiNeighborConfig(**metadata["config"])
    projector = NeighborTokenProjector(config)
    projector.load_state_dict(torch.load(checkpoint / "neighbor_projector.pt", map_location="cpu"))
    return projector
