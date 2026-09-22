"""Exploratory: Perceiver/Q-Former-style cross-attention resampler readout,
the third probe in this line of work after multi_neighbor_readout.py (v3)
and aspect_readout.py (the "matrix" / MHLA-style probe).

Context: v2's readout mean-pools every node into one vector; v3 fixes part
of this with per-neighbor tokens (indexed by node, 1-hop only); aspect
readout instead decomposes one pooled vector into several learned "views"
selected by a sparse router. All three still funnel through either a single
pooled vector or a hand-designed slice of the node matrix.

This module asks a different question, raised directly by the user: TEA and
GraphToken currently project graph features into the LLM's *input embedding
space* and concatenate them into the token sequence -- the same pattern
LLaVA uses for images. The alternative used by BLIP-2's Q-Former / the
Perceiver Resampler (as opposed to Flamingo's heavier cross-attention layers
spliced *into* the frozen LLM, which we are deliberately not attempting here
given how small this project's training corpus is relative to what
Flamingo-style architectures are normally trained on) is: keep a small
number of *learned query vectors*, and let them cross-attend over the
*entire* per-node embedding matrix to decide what to extract, rather than
mean-pooling it or picking fixed neighbor slots by hand. The output is still
just N tokens prepended to the LLM's input -- no surgery on the frozen LLM's
own attention, so `forward`/`generate`/`generate_batch` need no changes,
exactly like v2/v3/aspect_readout.

Consistent with aspect_readout.py's "let the adapter learn it" change: the
query vectors are plain learned parameters, not conditioned on
reasoning_type or hops. Source/target identity is not hand-picked by row
index either (v2's `source_row`/`target_row` mechanism) -- instead each
node's row is tagged with a 2-dim [is_source, is_target] indicator before
attention, so the resampler has to *attend* to the source/target nodes
itself if they matter for the query, rather than being handed them.
Task-conditioning is not injected here at all: the frozen LLM sees the
resampled tokens immediately followed by the question text, so its own
self-attention over [prefix; question] is trusted to do that integration --
same reasoning as why aspect_readout's router isn't task-conditioned either.

Two supported constructions, mirroring aspect_readout.py:

- ``PerceiverTEAGLM(base=..., resampler=...)`` -- reuse v2's already-trained
  frozen GNN. Architecturally this is arguably sufficient on its own (unlike
  aspect readout's ceiling problem, where K MLP heads reading the same
  *collapsed pooled vector* have no route to genuine diversity, this
  resampler attends over the *full per-node matrix*, which already has real
  per-node diversity from message passing -- no obvious pooling bottleneck
  forcing a retrain).
- ``PerceiverTEAGLM.from_scratch(...)`` -- train a fresh GNN for this probe
  too, run in **parallel** with aspect_readout's from-scratch GNN pretrain
  (GPU 1 had spare capacity under the 50% cap, and GNN pretraining alone --
  before the LLM is loaded -- is cheap). Chosen over reuse for this run
  mainly for a fair comparison: if aspect readout and this probe used
  different GNN-training regimes, an accuracy difference between them could
  be the GNN, not the readout mechanism being tested. Training script:
  ``scripts/train_perceiver_readout_from_scratch.py``.
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
class PerceiverResamplerConfig:
    node_dim: int  # GNN per-node output_dim + 2 (source/target tag)
    lm_hidden_dim: int
    num_queries: int = 10
    num_heads: int = 4
    num_layers: int = 2
    ff_hidden_dim: int = 512
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(self.node_dim, self.lm_hidden_dim, self.ff_hidden_dim) <= 0:
            raise ValueError("PerceiverResamplerConfig dimensions must be positive")
        if self.num_queries <= 0 or self.num_heads <= 0 or self.num_layers <= 0:
            raise ValueError("num_queries, num_heads, num_layers must be positive")
        if self.lm_hidden_dim % self.num_heads != 0:
            raise ValueError("lm_hidden_dim must be divisible by num_heads")


class _CrossAttentionBlock(_Module):
    """Pre-norm cross-attention (queries attend over node keys/values) +
    a feed-forward block, both with residual connections -- one Perceiver
    "layer". Stacking a few of these lets later layers refine what earlier
    layers extracted, instead of a single attention pass."""

    def __init__(self, config: PerceiverResamplerConfig) -> None:
        require_tea_dependencies()
        super().__init__()
        self.query_norm = nn.LayerNorm(config.lm_hidden_dim)
        self.kv_norm = nn.LayerNorm(config.lm_hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.lm_hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(config.lm_hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.lm_hidden_dim, config.ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_hidden_dim, config.lm_hidden_dim),
        )

    def forward(self, queries: Any, kv: Any, key_padding_mask: Any) -> Any:
        normed_queries = self.query_norm(queries)
        normed_kv = self.kv_norm(kv)
        attended, _ = self.attention(
            normed_queries, normed_kv, normed_kv, key_padding_mask=key_padding_mask, need_weights=False
        )
        queries = queries + attended
        queries = queries + self.feed_forward(self.ff_norm(queries))
        return queries


class PerceiverResampler(_Module):
    """K learned query vectors that cross-attend over the full per-node
    embedding matrix (all nodes, padded/masked to the batch's max node
    count) to produce exactly ``num_queries`` graph-prefix tokens."""

    def __init__(self, config: PerceiverResamplerConfig) -> None:
        require_tea_dependencies()
        super().__init__()
        self.config = config
        self.queries = nn.Parameter(torch.randn(config.num_queries, config.lm_hidden_dim) * 0.02)
        self.kv_proj = nn.Linear(config.node_dim, config.lm_hidden_dim)
        self.blocks = nn.ModuleList([_CrossAttentionBlock(config) for _ in range(config.num_layers)])

    def forward(self, node_matrix: Any, key_padding_mask: Any) -> Any:
        """``node_matrix``: [B, N_max, node_dim] (zero-padded).
        ``key_padding_mask``: [B, N_max] bool, True at PAD positions
        (``nn.MultiheadAttention`` convention -- True means "ignore")."""
        if node_matrix.ndim != 3 or node_matrix.shape[-1] != self.config.node_dim:
            raise ValueError(
                f"Expected node matrix [B, N, {self.config.node_dim}], got {tuple(node_matrix.shape)}"
            )
        batch_size = node_matrix.shape[0]
        kv = self.kv_proj(node_matrix)  # [B, N_max, lm_hidden_dim]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_queries, lm_hidden_dim]
        for block in self.blocks:
            queries = block(queries, kv, key_padding_mask)
        return queries  # [B, num_queries, lm_hidden_dim]


def _node_matrix_and_mask(
    node_repr: Any, tensor_graph: GraphTensor, source_id: str | None, target_id: str | None, max_nodes: int
) -> tuple[Any, Any]:
    """Build one example's [max_nodes, gnn_out_dim + 2] tagged, zero-padded
    node matrix (columns: node embedding, is_source, is_target) plus a
    [max_nodes] boolean pad mask (True at pad positions)."""
    node_count = node_repr.shape[0]
    node_index = {node_id: position for position, node_id in enumerate(tensor_graph.node_ids)}
    tags = torch.zeros(node_count, 2, dtype=node_repr.dtype, device=node_repr.device)
    if source_id in node_index:
        tags[node_index[source_id], 0] = 1.0
    if target_id in node_index:
        tags[node_index[target_id], 1] = 1.0
    tagged = torch.cat([node_repr, tags], dim=-1)  # [node_count, gnn_out_dim + 2]
    pad_count = max_nodes - node_count
    if pad_count > 0:
        padding = torch.zeros(pad_count, tagged.shape[-1], dtype=tagged.dtype, device=tagged.device)
        tagged = torch.cat([tagged, padding], dim=0)
    mask = torch.zeros(max_nodes, dtype=torch.bool, device=node_repr.device)
    if pad_count > 0:
        mask[node_count:] = True
    return tagged, mask


class PerceiverTEAGLM(TEAGLM):
    """TEAGLM variant whose entire graph prefix comes from a
    ``PerceiverResampler`` cross-attending over the full node matrix,
    replacing v2's mean-pool-then-project summary path outright (not
    additive, unlike v3/aspect_readout) -- this probe is testing the
    resampler as a full alternative readout, not an addition to the
    existing one.

    Reuses the base TEAGLM's frozen GNN/LM/tensorizer/tokenizer verbatim
    (see module docstring for why the GNN isn't retrained here); the base
    model's own summary ``projector`` is unused and not carried over.
    """

    def __init__(self, *, base: TEAGLM, resampler: PerceiverResampler, freeze_base: bool = True) -> None:
        require_tea_dependencies()
        nn.Module.__init__(self)
        self.gnn = base.gnn
        self.language_model = base.language_model
        self.tokenizer = base.tokenizer
        self.tensorizer = base.tensorizer
        self.config = base.config
        self.resampler = resampler
        # _truncate()/forward() only need something with a `.config`
        # carrying `prefix_tokens`/`lm_hidden_dim` for budget bookkeeping --
        # reuse base's projector object purely as that config holder, faked
        # to the resampler's actual output token count. graph_prefix() below
        # never calls its forward(), so its (mismatched) weights are inert
        # and it is never added to any optimizer.
        self.projector = base.projector
        self.projector.config = replace(
            self.projector.config,
            prefix_tokens=resampler.config.num_queries,
            lm_hidden_dim=resampler.config.lm_hidden_dim,
        )
        self._gnn_frozen = freeze_base
        for parameter in self.gnn.parameters():
            parameter.requires_grad_(not freeze_base)
        if freeze_base:
            self.gnn.eval()
        for parameter in self.projector.parameters():
            parameter.requires_grad_(False)
        for parameter in self.language_model.parameters():
            parameter.requires_grad_(False)
        self.language_model.eval()

    @classmethod
    def from_scratch(
        cls,
        lm_name_or_path: str,
        *,
        gnn_config: Any = None,
        tensorizer_config: Any = None,
        num_queries: int = 10,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_hidden_dim: int = 512,
        **from_pretrained_kwargs: Any,
    ) -> PerceiverTEAGLM:
        """Fresh, trainable GNN (via ``TEAGLM.from_pretrained(...,
        freeze_gnn=False)``) plus a fresh ``PerceiverResampler``. The base
        model's own summary projector is built but never trained or used
        for prefix computation (see ``__init__``'s note) -- only its config
        object is reused as bookkeeping."""
        require_tea_dependencies()
        from graph_modi.models.tea_glm import TEAGLMConfig

        base = TEAGLM.from_pretrained(
            lm_name_or_path,
            gnn_config=gnn_config,
            tensorizer_config=tensorizer_config,
            config=TEAGLMConfig(freeze_gnn=False),
            **from_pretrained_kwargs,
        )
        lm_hidden_dim = int(base.language_model.get_input_embeddings().embedding_dim)
        resampler_config = PerceiverResamplerConfig(
            node_dim=base.gnn.config.output_dim + 2,
            lm_hidden_dim=lm_hidden_dim,
            num_queries=num_queries,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_hidden_dim=ff_hidden_dim,
        )
        return cls(base=base, resampler=PerceiverResampler(resampler_config), freeze_base=False)

    def train(self, mode: bool = True) -> PerceiverTEAGLM:
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
        if source_ids is None:
            source_ids = [None] * len(graphs)
        if target_ids is None:
            target_ids = [None] * len(graphs)
        embedding_dtype = self.language_model.get_input_embeddings().weight.dtype

        node_reprs = []
        tensor_graphs = []
        for graph in graphs:
            tensor_graph = self.tensorizer(graph) if isinstance(graph, AttributedGraph) else graph
            tensor_graph = tensor_graph.to(self.device)
            tensor_graphs.append(tensor_graph)
            if self._gnn_frozen:
                with torch.no_grad():
                    node_reprs.append(self.gnn(tensor_graph))
            else:
                node_reprs.append(self.gnn(tensor_graph))

        max_nodes = max(repr_.shape[0] for repr_ in node_reprs)
        matrices, masks = [], []
        for node_repr, tensor_graph, source_id, target_id in zip(
            node_reprs, tensor_graphs, source_ids, target_ids, strict=True
        ):
            matrix, mask = _node_matrix_and_mask(node_repr, tensor_graph, source_id, target_id, max_nodes)
            matrices.append(matrix)
            masks.append(mask)
        node_matrix = torch.stack(matrices)  # [B, max_nodes, gnn_out_dim + 2]
        key_padding_mask = torch.stack(masks)  # [B, max_nodes]

        prefix = self.resampler(node_matrix, key_padding_mask)  # [B, num_queries, lm_hidden_dim]
        return prefix.to(dtype=embedding_dtype)


_PERCEIVER_CHECKPOINT_FORMAT = "graph-modi-perceiver-readout-v1"


def save_perceiver_checkpoint(resampler: PerceiverResampler, checkpoint_dir: str | Path) -> None:
    require_tea_dependencies()
    output = Path(checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(resampler.state_dict(), output / "resampler.pt")
    metadata = {"format": _PERCEIVER_CHECKPOINT_FORMAT, "config": asdict(resampler.config)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_perceiver_checkpoint(checkpoint_dir: str | Path) -> PerceiverResampler:
    require_tea_dependencies()
    checkpoint = Path(checkpoint_dir)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != _PERCEIVER_CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported perceiver-readout checkpoint metadata: {checkpoint}")
    config = PerceiverResamplerConfig(**metadata["config"])
    resampler = PerceiverResampler(config)
    resampler.load_state_dict(torch.load(checkpoint / "resampler.pt", map_location="cpu"))
    return resampler
