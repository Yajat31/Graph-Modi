"""Exploratory: MHLA-style multi-aspect graph readout with task-conditioned
routing, as an alternative/complement to the multi-neighbor readout in
``multi_neighbor_readout.py``.

Motivation (user proposal, see documents/experiments/results/
v2_clegr_extended-20260919/report.md §8): v2's graph_prefix collapses the
whole neighborhood into ONE pooled vector before the frozen LLM ever sees it
-- multi_neighbor_readout.py fixed part of this by adding per-*neighbor*
tokens (indexed by node), but every neighbor still gets the same single
representation. This module instead indexes by *aspect*: it decomposes the
same pooled+source+target+hop+task vector v2 already computes into several
independent learned "views" (analogous to DeepSeek's Multi-Head Latent
Attention -- one shared compressed latent, several per-head up-projections),
then a small task-conditioned router selects which ``num_selected`` of the
``num_aspects`` views are actually worth spending context on for this
query's reasoning_type, instead of always paying for all of them.

Sparse top-k routing (not just K parallel heads averaged/concatenated) is
the deliberate choice: with plain parallel heads trained against the same
downstream LM loss, gradient descent has no reason to make head A specialize
in anything different from head B (representation collapse). Routing only
a subset per example means each head's parameters only receive gradient
from the examples actually routed to it, which is what gives Mixture-of-
Experts-style architectures (Shazeer 2017, Switch Transformer, DeepSeek-MoE)
their specialization pressure "for free" -- no explicit diversity loss or
hand-designed input slicing needed as a first attempt.

Unlike multi_neighbor_readout.py (which reused v2's already-trained, frozen
GNN verbatim), this is trained with the GNN itself **from scratch** too --
decided upfront rather than as a fallback, since a GNN already optimized
for v2's single-pooled-vector readout has no particular reason to produce
node embeddings whose *linear combinations* separate into meaningfully
different aspects; better to let the GNN and the adapter learn a
representation and a routing scheme jointly from the start. Two supported
constructions, both in this module:

- ``AspectTEAGLM(base=..., aspect_heads=..., router=..., freeze_base=True)``
  -- reuse-frozen-components mode (mirrors ``MultiNeighborTEAGLM``), kept for
  comparison / in case the from-scratch run underperforms this cheaper
  option.
- ``AspectTEAGLM.from_scratch(lm_name_or_path, ...)`` -- the mode actually
  used here: builds a randomly-initialized GNN + summary projector (via
  ``TEAGLM.from_pretrained`` with ``freeze_gnn=False``) alongside fresh
  ``AspectHeads``/``TaskConditionedRouter``, all trainable, LM still frozen.
  Training script: ``scripts/train_aspect_readout_from_scratch.py`` (GNN
  pretrain phase, reusing the existing ``pretrain_graph_encoder`` path,
  followed by a joint projector+aspect_heads+router phase).

Deliberately NOT fed to the new adapter: the reasoning_type one-hot ("task
vector") and the source/target node rows ("neighbour representation") that
v2/v3 hand-engineer into ``encode_graphs``'s concatenated vector. Both stay
exactly as they are on the frozen summary path (unchanged from v2) -- but
the new AspectHeads/TaskConditionedRouter see only the raw mean-pooled GNN
vector, nothing else. The point of this probe is to test whether a learned
adapter can pick out useful, task-relevant structure on its own from the
graph representation alone, rather than being told the task or handed
specific node picks -- a strictly harder and more general setup than v2/v3.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from graph_modi.models.tea_glm import (
    TEAGLM,
    GraphTensor,
    _Module,
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
class AspectReadoutConfig:
    pooled_dim: int
    lm_hidden_dim: int
    num_aspects: int = 6
    num_selected: int = 3
    head_hidden_dim: int = 512
    router_hidden_dim: int = 64
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(self.pooled_dim, self.lm_hidden_dim, self.head_hidden_dim, self.router_hidden_dim) <= 0:
            raise ValueError("AspectReadoutConfig dimensions must be positive")
        if not (0 < self.num_selected <= self.num_aspects):
            raise ValueError("num_selected must be in (0, num_aspects]")


class AspectHeads(_Module):
    """K independent MLPs ("up-projection heads") mapping the same shared
    *raw pooled GNN vector* (no task one-hot, no source/target rows -- see
    module docstring) to K candidate graph-prefix tokens -- the
    decompression side of the MHLA analogy. Each head is free to specialize
    once routing (below) makes it see a non-uniform slice of the training
    distribution."""

    def __init__(self, config: AspectReadoutConfig) -> None:
        require_tea_dependencies()
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.pooled_dim, config.head_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.head_hidden_dim, config.lm_hidden_dim),
                )
                for _ in range(config.num_aspects)
            ]
        )

    def forward(self, pooled: Any) -> Any:
        if pooled.ndim != 2 or pooled.shape[-1] != self.config.pooled_dim:
            raise ValueError(f"Expected pooled vector [B, {self.config.pooled_dim}], got {tuple(pooled.shape)}")
        return torch.stack([head(pooled) for head in self.heads], dim=1)  # [B, K, lm_hidden_dim]


class TaskConditionedRouter(_Module):
    """Selects the top ``num_selected`` of K aspect heads per example from
    the same raw pooled GNN vector the heads see -- no explicit task label,
    no source/target rows. The router has to infer what's worth selecting
    from the graph's own pooled representation alone (it is "task-
    conditioned" only in the sense that different graphs/queries route
    differently as a side effect of pooled-vector differences, not because
    it is told the task). Standard sparse top-k softmax gating, as in
    Sparse MoE / Switch Transformer / Mixtral -- gradients flow to the
    router's logits and to the selected heads only; unselected heads get no
    gradient from that example, which is what creates specialization
    pressure without an explicit diversity loss."""

    def __init__(self, config: AspectReadoutConfig) -> None:
        require_tea_dependencies()
        super().__init__()
        self.config = config
        self.gate = nn.Sequential(
            nn.Linear(config.pooled_dim, config.router_hidden_dim),
            nn.GELU(),
            nn.Linear(config.router_hidden_dim, config.num_aspects),
        )

    def forward(self, pooled: Any) -> tuple[Any, Any]:
        logits = self.gate(pooled)  # [B, K]
        weights = torch.softmax(logits, dim=-1)
        topk_weights, topk_indices = weights.topk(self.config.num_selected, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights, topk_indices  # [B, M], [B, M]


class AspectTEAGLM(TEAGLM):
    """TEAGLM variant whose graph prefix is [v2's existing summary tokens]
    + [M task-routed aspect tokens] instead of just the summary tokens.

    Reuses the base TEAGLM's frozen LM/GNN/tensorizer/summary projector
    verbatim -- graph_prefix's summary-token computation is unchanged from
    v2; only aspect_heads and router are new and trainable.
    """

    def __init__(
        self,
        *,
        base: TEAGLM,
        aspect_heads: AspectHeads,
        router: TaskConditionedRouter,
        freeze_base: bool = True,
    ) -> None:
        """``freeze_base=True`` reproduces MultiNeighborTEAGLM's reuse-frozen-
        checkpoint mode. ``freeze_base=False`` (used by ``from_scratch``)
        leaves ``base``'s GNN and summary projector trainable -- for a fresh
        model those are exactly as randomly-initialized as aspect_heads/
        router, so there is nothing meaningful to freeze yet."""
        require_tea_dependencies()
        # Deliberately skip TEAGLM.__init__: see MultiNeighborTEAGLM for why
        # (its graph_dim validation assumes the pooled-only architecture).
        nn.Module.__init__(self)
        self.gnn = base.gnn
        self.projector = base.projector
        self.language_model = base.language_model
        self.tokenizer = base.tokenizer
        self.tensorizer = base.tensorizer
        self.config = base.config
        self.aspect_heads = aspect_heads
        self.router = router
        # Bookkeeping-only, same trick as MultiNeighborTEAGLM: _truncate()/
        # forward() read self.projector.config.prefix_tokens purely as "how
        # many slots does the graph prefix occupy" for masking/truncation
        # math. graph_prefix() below bypasses self.projector.forward()'s own
        # reshape (which would otherwise use this same faked number and
        # break) by calling self.projector.network(...) directly.
        combined_tokens = self.projector.config.prefix_tokens + router.config.num_selected
        self.projector.config = replace(self.projector.config, prefix_tokens=combined_tokens)
        self._gnn_frozen = freeze_base
        for parameter in self.gnn.parameters():
            parameter.requires_grad_(not freeze_base)
        for parameter in self.projector.parameters():
            parameter.requires_grad_(not freeze_base)
        if freeze_base:
            self.gnn.eval()
        for parameter in self.language_model.parameters():
            parameter.requires_grad_(False)
        self.language_model.eval()

    @classmethod
    def from_scratch(
        cls,
        lm_name_or_path: str,
        *,
        gnn_config: Any = None,
        projector_config: Any = None,
        tensorizer_config: Any = None,
        prefix_tokens: int = 8,
        projector_hidden_dim: int = 512,
        projector_num_layers: int = 1,
        num_aspects: int = 6,
        num_selected: int = 3,
        head_hidden_dim: int = 512,
        router_hidden_dim: int = 64,
        **from_pretrained_kwargs: Any,
    ) -> AspectTEAGLM:
        """Fresh GNN + summary projector (both randomly initialized, both
        trainable) via ``TEAGLM.from_pretrained(..., freeze_gnn=False)``,
        plus fresh ``AspectHeads``/``TaskConditionedRouter``. Only the LM is
        pretrained/frozen -- everything graph-side starts from scratch and
        is meant to be trained jointly (see
        scripts/train_aspect_readout_from_scratch.py)."""
        require_tea_dependencies()
        from graph_modi.models.tea_glm import TEAGLMConfig

        base = TEAGLM.from_pretrained(
            lm_name_or_path,
            gnn_config=gnn_config,
            projector_config=projector_config,
            tensorizer_config=tensorizer_config,
            config=TEAGLMConfig(freeze_gnn=False),
            prefix_tokens=prefix_tokens,
            projector_hidden_dim=projector_hidden_dim,
            projector_num_layers=projector_num_layers,
            **from_pretrained_kwargs,
        )
        lm_hidden_dim = int(base.language_model.get_input_embeddings().embedding_dim)
        aspect_config = AspectReadoutConfig(
            pooled_dim=base.gnn.config.output_dim,
            lm_hidden_dim=lm_hidden_dim,
            num_aspects=num_aspects,
            num_selected=num_selected,
            head_hidden_dim=head_hidden_dim,
            router_hidden_dim=router_hidden_dim,
        )
        return cls(
            base=base,
            aspect_heads=AspectHeads(aspect_config),
            router=TaskConditionedRouter(aspect_config),
            freeze_base=False,
        )

    def train(self, mode: bool = True) -> AspectTEAGLM:
        nn.Module.train(self, mode)
        self.language_model.eval()
        if self._gnn_frozen:
            self.gnn.eval()
        return self

    def graph_prefix(
        self,
        graphs: Any,
        *,
        source_ids: Any = None,
        target_ids: Any = None,
        hops: Any = None,
        reasoning_types: Any = None,
    ) -> Any:
        embedding_dtype = self.language_model.get_input_embeddings().weight.dtype
        latent = self.encode_graphs(
            graphs,
            source_ids=source_ids,
            target_ids=target_ids,
            hops=hops,
            reasoning_types=reasoning_types,
        )  # [B, graph_dim] = [pooled | source_row | target_row | hop | task]
        base_prefix = self.projector.network(latent).view(
            latent.shape[0], -1, self.projector.config.lm_hidden_dim
        )  # [B, T_summary, lm_hidden] -- summary path unchanged from v2, still
        # sees the full latent including task/source/target.

        # Aspect heads/router see ONLY the raw pooled vector -- no task
        # one-hot, no source/target rows (see module docstring).
        pooled = latent[:, : self.aspect_heads.config.pooled_dim]
        candidates = self.aspect_heads(pooled)  # [B, K, lm_hidden]
        weights, indices = self.router(pooled)  # [B, M], [B, M]
        selected = torch.gather(
            candidates, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, candidates.shape[-1])
        )  # [B, M, lm_hidden]
        aspect_tokens = selected * weights.unsqueeze(-1)

        return torch.cat([base_prefix, aspect_tokens], dim=1).to(dtype=embedding_dtype)


_ASPECT_CHECKPOINT_FORMAT = "graph-modi-aspect-readout-v1"


def save_aspect_checkpoint(
    aspect_heads: AspectHeads, router: TaskConditionedRouter, checkpoint_dir: str | Path
) -> None:
    """Persist the trained aspect heads + router. Learned from
    multi_neighbor_readout.py's first exploratory run, which trained and
    evaluated in one process without ever writing the result to disk --
    saving from the start this time so a later separate dynamic eval can
    reuse this checkpoint without retraining."""
    require_tea_dependencies()
    output = Path(checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(aspect_heads.state_dict(), output / "aspect_heads.pt")
    torch.save(router.state_dict(), output / "router.pt")
    metadata = {
        "format": _ASPECT_CHECKPOINT_FORMAT,
        "config": asdict(aspect_heads.config),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_aspect_checkpoint(checkpoint_dir: str | Path) -> tuple[AspectHeads, TaskConditionedRouter]:
    """Reconstruct trained AspectHeads/TaskConditionedRouter from
    ``save_aspect_checkpoint``'s output."""
    require_tea_dependencies()
    checkpoint = Path(checkpoint_dir)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != _ASPECT_CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported aspect-readout checkpoint metadata: {checkpoint}")
    config = AspectReadoutConfig(**metadata["config"])
    aspect_heads = AspectHeads(config)
    aspect_heads.load_state_dict(torch.load(checkpoint / "aspect_heads.pt", map_location="cpu"))
    router = TaskConditionedRouter(config)
    router.load_state_dict(torch.load(checkpoint / "router.pt", map_location="cpu"))
    return aspect_heads, router
