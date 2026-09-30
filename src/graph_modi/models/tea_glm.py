"""TEA-GLM-compatible graph encoder, prefix projector, and causal LM wrapper.

The module itself is importable without the optional ``tea`` dependencies. Neural
components fail at construction time with an actionable installation message.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from graph_modi.graph.edits import parse_edit_program
from graph_modi.models.base import ModelInput
from graph_modi.schema import AttributedGraph, ReasoningType

# WITHIN_HOPS_COUNT/WITHIN_HOPS_LIST/MOST_COMMON_ATTRIBUTE_WITHIN_HOPS queries
# name a hop radius (2 or 3) that the GraphSAGE readout otherwise has no way
# to see: node_repr is the *final*, num_layers-mixed embedding, which
# conflates every radius up to num_layers into one vector per node. Appending
# the raw hop count lets the projector at least condition on which radius a
# query means, instead of the frozen LM having to infer it purely from the
# question text.
HOPS_FEATURE_DIM = 1

# Training on many heterogeneous task types (boolean, numeric, list, string)
# with no explicit task-identity signal causes answer-format leakage: e.g. a
# cycle_membership (yes/no) query answered with a bare number, or a
# most_common_attribute_within_hops (categorical) query answered with a
# station-ID list -- the model apparently can't reliably infer which of the
# ~12 task formats applies from the question text plus a graph prefix alone,
# and defaults to whatever format is best-represented in the training mix.
# A one-hot task-identity feature (covering every ReasoningType, not just the
# ones observed to leak) removes that ambiguity the same way HOPS_FEATURE_DIM
# does for hop radius.
_REASONING_TYPE_VALUES = tuple(member.value for member in ReasoningType)
_REASONING_TYPE_INDEX = {value: index for index, value in enumerate(_REASONING_TYPE_VALUES)}
REASONING_TYPE_FEATURE_DIM = len(_REASONING_TYPE_VALUES)


def _reasoning_type_onehot(reasoning_type: str | None, *, dtype: Any, device: Any) -> Any:
    vector = torch.zeros(REASONING_TYPE_FEATURE_DIM, dtype=dtype, device=device)
    index = _REASONING_TYPE_INDEX.get(reasoning_type) if reasoning_type is not None else None
    if index is not None:
        vector[index] = 1.0
    return vector

_TEA_IMPORT_ERROR: ImportError | None = None
try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on optional environment
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TEA_IMPORT_ERROR = exc

_Module = nn.Module if nn is not None else object

TEA_CHECKPOINT_FORMAT = "graph-modi-tea-glm-v1"


def require_tea_dependencies() -> None:
    """Raise an actionable error when PyTorch/Transformers dependencies are absent."""
    if torch is None or nn is None:
        raise RuntimeError(
            "TEA-GLM support requires the optional dependencies. "
            "Install them with `pip install 'graph-modi[tea]'`."
        ) from _TEA_IMPORT_ERROR


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _state_dict_sha256(module: Any) -> str:
    require_tea_dependencies()
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json(list(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _object_name(value: Any) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


@dataclass(frozen=True, slots=True)
class NodeTensorizerConfig:
    """Stable feature-hashing configuration for attributed graph nodes."""

    feature_dim: int = 256
    hash_salt: str = "graph-modi-node-v1"
    include_node_id: bool = True
    include_label: bool = True
    include_attributes: bool = True
    l2_normalize: bool = True

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")


@dataclass(frozen=True, slots=True)
class GraphTensor:
    """A deterministic tensor representation independent of torch-geometric."""

    x: Any
    edge_index: Any
    edge_weight: Any
    node_ids: tuple[str, ...]
    graph_id: str

    def to(self, device: Any) -> GraphTensor:
        return GraphTensor(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            edge_weight=self.edge_weight.to(device),
            node_ids=self.node_ids,
            graph_id=self.graph_id,
        )


class DeterministicNodeTensorizer:
    """Feature-hash node labels and attributes with stable ordering and hashing."""

    def __init__(self, config: NodeTensorizerConfig | None = None) -> None:
        require_tea_dependencies()
        self.config = config or NodeTensorizerConfig()

    def _add_feature(self, row: list[float], key: str, magnitude: float = 1.0) -> None:
        payload = f"{self.config.hash_salt}\0{key}".encode()
        raw = hashlib.sha256(payload).digest()
        bucket = int.from_bytes(raw[:8], "big") % self.config.feature_dim
        sign = 1.0 if raw[8] & 1 else -1.0
        row[bucket] += sign * magnitude

    def _node_row(self, node: Any) -> list[float]:
        row = [0.0] * self.config.feature_dim
        if self.config.include_node_id:
            self._add_feature(row, f"id={node.id}")
        if self.config.include_label:
            self._add_feature(row, f"label={node.label}")
        if self.config.include_attributes:
            for key, value in sorted(node.attributes.items()):
                rendered = _canonical_json(value)
                self._add_feature(row, f"attr:{key}={rendered}")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric = float(value)
                    if math.isfinite(numeric):
                        self._add_feature(row, f"numeric:{key}", math.tanh(numeric))
        if self.config.l2_normalize:
            norm = math.sqrt(sum(value * value for value in row))
            if norm:
                row = [value / norm for value in row]
        return row

    def __call__(self, graph: AttributedGraph) -> GraphTensor:
        node_map = graph.node_map()
        node_ids = tuple(sorted(node_map))
        if not node_ids:
            raise ValueError(f"Graph {graph.graph_id!r} has no nodes")
        index = {node_id: position for position, node_id in enumerate(node_ids)}
        rows = [self._node_row(node_map[node_id]) for node_id in node_ids]
        directed_edges: list[tuple[int, int, float, str]] = []
        for edge in graph.edges:
            if edge.source not in index or edge.target not in index:
                raise ValueError(
                    f"Edge {edge.source!r}->{edge.target!r} references an unknown node"
                )
            directed_edges.append(
                (index[edge.source], index[edge.target], float(edge.weight), edge.relation)
            )
            if not graph.directed and edge.source != edge.target:
                directed_edges.append(
                    (index[edge.target], index[edge.source], float(edge.weight), edge.relation)
                )
        directed_edges.sort(key=lambda item: (item[0], item[1], item[3], item[2]))
        if directed_edges:
            edge_index = torch.tensor(
                [[item[0] for item in directed_edges], [item[1] for item in directed_edges]],
                dtype=torch.long,
            )
            edge_weight = torch.tensor([item[2] for item in directed_edges], dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float32)
        return GraphTensor(
            x=torch.tensor(rows, dtype=torch.float32),
            edge_index=edge_index,
            edge_weight=edge_weight,
            node_ids=node_ids,
            graph_id=graph.graph_id,
        )


@dataclass(frozen=True, slots=True)
class GraphSAGEConfig:
    input_dim: int = 256
    hidden_dim: int = 256
    output_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    aggregation: str = "mean"

    def __post_init__(self) -> None:
        if min(self.input_dim, self.hidden_dim, self.output_dim, self.num_layers) <= 0:
            raise ValueError("GraphSAGE dimensions and num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.aggregation not in {"mean", "sum", "max"}:
            raise ValueError("aggregation must be one of: mean, sum, max")


class GraphSAGEEncoder(_Module):
    """GraphSAGE implemented using only core PyTorch, with mean/sum/max aggregation.

    Mean aggregation is invariant to neighbor count, which discards exactly the
    signal that counting and distance-based tasks (shortest path, neighbor
    counts) depend on. Sum preserves it; max is a permutation-invariant
    alternative that instead reports the strongest neighbor signal per feature.
    """

    def __init__(self, config: GraphSAGEConfig) -> None:
        require_tea_dependencies()
        super().__init__()
        self.config = config
        dimensions = [config.input_dim]
        dimensions.extend([config.hidden_dim] * (config.num_layers - 1))
        dimensions.append(config.output_dim)
        self.self_layers = nn.ModuleList(
            nn.Linear(dimensions[i], dimensions[i + 1]) for i in range(config.num_layers)
        )
        self.neighbor_layers = nn.ModuleList(
            nn.Linear(dimensions[i], dimensions[i + 1], bias=False)
            for i in range(config.num_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(dimensions[i + 1]) for i in range(config.num_layers)
        )
        self.dropout = nn.Dropout(config.dropout)

    def _aggregate(self, x: Any, sources: Any, destinations: Any, edge_weight: Any) -> Any:
        if not sources.numel():
            return torch.zeros_like(x)
        weights = edge_weight.to(dtype=x.dtype).unsqueeze(-1)
        messages = x[sources] * weights
        if self.config.aggregation == "max":
            index = destinations.unsqueeze(-1).expand_as(messages)
            aggregate = torch.full_like(x, float("-inf"))
            aggregate = aggregate.scatter_reduce(0, index, messages, reduce="amax")
            return torch.where(torch.isinf(aggregate), torch.zeros_like(aggregate), aggregate)
        aggregate = torch.zeros_like(x)
        aggregate.index_add_(0, destinations, messages)
        if self.config.aggregation == "sum":
            return aggregate
        degree = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        degree.index_add_(0, destinations, weights.abs())
        return aggregate / degree.clamp_min(1.0)

    def forward(self, graph: GraphTensor) -> Any:
        x = graph.x
        if x.ndim != 2 or x.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Expected node features [N, {self.config.input_dim}], got {tuple(x.shape)}"
            )
        sources, destinations = graph.edge_index
        for layer_index, (self_layer, neighbor_layer) in enumerate(
            zip(self.self_layers, self.neighbor_layers, strict=True)
        ):
            aggregate = self._aggregate(x, sources, destinations, graph.edge_weight)
            x = self_layer(x) + neighbor_layer(aggregate)
            x = self.norms[layer_index](x)
            if layer_index + 1 < self.config.num_layers:
                x = self.dropout(torch.relu(x))
        return x

    def pooled(self, graph: GraphTensor) -> Any:
        return self.forward(graph).mean(dim=0)


@dataclass(frozen=True, slots=True)
class PrefixProjectorConfig:
    graph_dim: int = 256
    lm_hidden_dim: int = 768
    prefix_tokens: int = 8
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if (
            min(
                self.graph_dim,
                self.lm_hidden_dim,
                self.prefix_tokens,
                self.hidden_dim,
                self.num_layers,
            )
            <= 0
        ):
            raise ValueError("Projector dimensions, layers, and prefix_tokens must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class GraphPrefixProjector(_Module):
    """Map one pooled graph vector to a configurable sequence of LM embeddings."""

    def __init__(self, config: PrefixProjectorConfig) -> None:
        require_tea_dependencies()
        super().__init__()
        self.config = config
        layers: list[Any] = []
        input_dim = config.graph_dim
        for _ in range(config.num_layers - 1):
            layers.extend(
                [
                    nn.Linear(input_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            input_dim = config.hidden_dim
        layers.append(nn.Linear(input_dim, config.prefix_tokens * config.lm_hidden_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, graph_vectors: Any) -> Any:
        if graph_vectors.ndim != 2 or graph_vectors.shape[-1] != self.config.graph_dim:
            raise ValueError(
                f"Expected graph vectors [B, {self.config.graph_dim}], "
                f"got {tuple(graph_vectors.shape)}"
            )
        projected = self.network(graph_vectors)
        return projected.reshape(
            graph_vectors.shape[0],
            self.config.prefix_tokens,
            self.config.lm_hidden_dim,
        )


@dataclass(frozen=True, slots=True)
class TEAGLMConfig:
    max_sequence_length: int = 512
    prompt_template: str = "Question: {question}\nAnswer:"
    add_bos_token: bool = True
    add_eos_token: bool = True
    freeze_gnn: bool = True

    def __post_init__(self) -> None:
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if "{question}" not in self.prompt_template:
            raise ValueError("prompt_template must contain {question}")


class TEAGLM(_Module):
    """Frozen causal LM conditioned on GraphSAGE-derived prefix embeddings."""

    def __init__(
        self,
        *,
        gnn: GraphSAGEEncoder,
        projector: GraphPrefixProjector,
        language_model: Any,
        tokenizer: Any,
        tensorizer: DeterministicNodeTensorizer,
        config: TEAGLMConfig | None = None,
    ) -> None:
        require_tea_dependencies()
        super().__init__()
        self.gnn = gnn
        self.projector = projector
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.tensorizer = tensorizer
        self.config = config or TEAGLMConfig()
        lm_hidden_dim = int(language_model.get_input_embeddings().embedding_dim)
        expected_graph_dim = (
            gnn.config.output_dim * 3 + HOPS_FEATURE_DIM + REASONING_TYPE_FEATURE_DIM
        )
        if expected_graph_dim != projector.config.graph_dim:
            raise ValueError(
                "projector graph_dim must equal 3 * GNN output_dim + HOPS_FEATURE_DIM "
                "+ REASONING_TYPE_FEATURE_DIM (pooled graph vector concatenated with "
                "source and target node embeddings, plus the query's hop-radius "
                "scalar and one-hot task identity)"
            )
        if projector.config.lm_hidden_dim != lm_hidden_dim:
            raise ValueError(
                "Projector lm_hidden_dim does not match LM input embedding dimension "
                f"({projector.config.lm_hidden_dim} != {lm_hidden_dim})"
            )
        for parameter in self.language_model.parameters():
            parameter.requires_grad_(False)
        self.language_model.eval()
        self._gnn_frozen = False
        self.set_gnn_frozen(self.config.freeze_gnn)

    @classmethod
    def from_pretrained(
        cls,
        lm_name_or_path: str,
        *,
        gnn_config: GraphSAGEConfig | None = None,
        projector_config: PrefixProjectorConfig | None = None,
        tensorizer_config: NodeTensorizerConfig | None = None,
        config: TEAGLMConfig | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        torch_dtype: Any = None,
        prefix_tokens: int = 8,
        projector_hidden_dim: int = 512,
        projector_num_layers: int = 1,
    ) -> TEAGLM:
        require_tea_dependencies()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("TEA-GLM requires Transformers. Install `graph-modi[tea]`.") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            lm_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        language_model = AutoModelForCausalLM.from_pretrained(
            lm_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define either pad_token_id or eos_token_id")
            tokenizer.pad_token = tokenizer.eos_token
        tensor_config = tensorizer_config or NodeTensorizerConfig()
        graph_config = gnn_config or GraphSAGEConfig(input_dim=tensor_config.feature_dim)
        if graph_config.input_dim != tensor_config.feature_dim:
            raise ValueError("Tensorizer feature_dim must equal GNN input_dim")
        lm_hidden_dim = int(language_model.get_input_embeddings().embedding_dim)
        prefix_config = projector_config or PrefixProjectorConfig(
            graph_dim=graph_config.output_dim * 3 + HOPS_FEATURE_DIM + REASONING_TYPE_FEATURE_DIM,
            lm_hidden_dim=lm_hidden_dim,
            prefix_tokens=prefix_tokens,
            hidden_dim=projector_hidden_dim,
            num_layers=projector_num_layers,
        )
        return cls(
            gnn=GraphSAGEEncoder(graph_config),
            projector=GraphPrefixProjector(prefix_config),
            language_model=language_model,
            tokenizer=tokenizer,
            tensorizer=DeterministicNodeTensorizer(tensor_config),
            config=config,
        )

    def train(self, mode: bool = True) -> TEAGLM:
        super().train(mode)
        self.language_model.eval()
        if self._gnn_frozen:
            self.gnn.eval()
        return self

    def set_gnn_frozen(self, frozen: bool = True) -> None:
        self._gnn_frozen = frozen
        for parameter in self.gnn.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            self.gnn.eval()

    @property
    def device(self) -> Any:
        return next(self.projector.parameters()).device

    def _lm_autocast(self) -> Any:
        """Autocast to the language model's dtype outside of Accelerate.

        Training goes through ``accelerator.prepare(..., mixed_precision=...)``,
        which autocasts the forward pass automatically, so the fp32 GNN/projector
        can feed a bf16 language model without a dtype mismatch. Inference paths
        (generate/forward called directly, e.g. from evaluation) get no such
        wrapper, so they need this explicitly or every call crashes as soon as a
        graph-conditioned prefix (fp32) is concatenated with bf16 token embeddings.
        """
        lm_dtype = next(self.language_model.parameters()).dtype
        if self.device.type == "cuda" and lm_dtype in (torch.bfloat16, torch.float16):
            return torch.autocast(device_type="cuda", dtype=lm_dtype)
        return contextlib.nullcontext()

    def encode_graphs(
        self,
        graphs: Sequence[AttributedGraph | GraphTensor],
        *,
        source_ids: Sequence[str | None] | None = None,
        target_ids: Sequence[str | None] | None = None,
        hops: Sequence[int | None] | None = None,
        reasoning_types: Sequence[str | None] | None = None,
    ) -> Any:
        """Query-conditioned graph vector: a specific node's own embedding for
        node/edge-level queries, or a whole-graph mean pool only when there is
        no node to point at -- plus the query's hop radius and task identity.

        Mean pooling alone collapses every node into one vector, discarding
        which node(s) a query is about -- fatal for path/reachability/
        neighbor-count tasks that name specific nodes (README section 5:
        "task-appropriate readouts ... do not represent every problem with
        one repeated mean-pooled vector"). Matches the source CLEGR paper's
        own GLM baseline (arXiv:2508.20583), which switches between
        M_g(graph, n_i) (a specific node's representation, no pooling) for
        node-level questions and Pool(M_g(graph)) only for graph-level
        questions with no single relevant node -- rather than unconditionally
        concatenating both, which dilutes the node-specific signal with an
        undifferentiated average over every other node for exactly the
        questions that least need it. This mirrors the pooled+source+
        target representation pretrain_graph_encoder's classification head
        already uses (training.py::_batch_query_vectors); zero-filled when a
        task has no source/target (e.g. pure graph-level tasks). ``hops`` is
        appended as a raw scalar (0.0 when absent) for WITHIN_HOPS_*/
        MOST_COMMON_ATTRIBUTE_WITHIN_HOPS queries, whose answer depends on a
        radius node_repr alone cannot distinguish (see HOPS_FEATURE_DIM).
        ``reasoning_types`` is appended as a one-hot over every ReasoningType
        (all zeros when absent) so the model has an explicit task-identity
        signal instead of inferring answer format from question text plus an
        otherwise task-agnostic graph prefix -- observed to leak badly across
        the ~12-task mix (e.g. cycle_membership answered with a bare number;
        see REASONING_TYPE_FEATURE_DIM).
        """
        if source_ids is None:
            source_ids = [None] * len(graphs)
        if target_ids is None:
            target_ids = [None] * len(graphs)
        if hops is None:
            hops = [None] * len(graphs)
        if reasoning_types is None:
            reasoning_types = [None] * len(graphs)
        vectors = []
        for graph, source_id, target_id, hop_count, reasoning_type in zip(
            graphs, source_ids, target_ids, hops, reasoning_types, strict=True
        ):
            tensor_graph = self.tensorizer(graph) if isinstance(graph, AttributedGraph) else graph
            tensor_graph = tensor_graph.to(self.device)
            if self.config.freeze_gnn:
                with torch.no_grad():
                    node_repr = self.gnn(tensor_graph)
            else:
                node_repr = self.gnn(tensor_graph)
            zeros = torch.zeros(node_repr.shape[-1], dtype=node_repr.dtype, device=node_repr.device)
            node_index = {node_id: position for position, node_id in enumerate(tensor_graph.node_ids)}
            source_row = node_repr[node_index[source_id]] if source_id in node_index else zeros
            target_row = node_repr[node_index[target_id]] if target_id in node_index else zeros
            # Task-conditional pooling: the real CLEGR paper's own GLM baseline switches
            # between M_g(graph, n_i) (a specific node's own representation, no pooling) for
            # node/edge-level questions and Pool(M_g(graph)) only for graph-level questions
            # that have no single node to point at (arXiv:2508.20583). Unconditionally
            # concatenating a whole-graph mean pool alongside source/target rows -- as this
            # function did previously -- dilutes the node-specific signal with an
            # undifferentiated average over every other node in the graph, for exactly the
            # questions that least need it. Pool only when there is no node anchor at all.
            has_node_anchor = source_id in node_index or target_id in node_index
            pooled = zeros if has_node_anchor else node_repr.mean(dim=0)
            hop_feature = torch.tensor(
                [float(hop_count) if hop_count is not None else 0.0],
                dtype=node_repr.dtype,
                device=node_repr.device,
            )
            task_feature = _reasoning_type_onehot(
                reasoning_type, dtype=node_repr.dtype, device=node_repr.device
            )
            vectors.append(
                torch.cat([pooled, source_row, target_row, hop_feature, task_feature], dim=-1)
            )
        return torch.stack(vectors)

    def graph_prefix(
        self,
        graphs: Sequence[AttributedGraph | GraphTensor],
        *,
        source_ids: Sequence[str | None] | None = None,
        target_ids: Sequence[str | None] | None = None,
        hops: Sequence[int | None] | None = None,
        reasoning_types: Sequence[str | None] | None = None,
    ) -> Any:
        """Project pooled+source+target graph vectors, then cast to the LM's embedding dtype.

        Training goes through Accelerate's autocast (accelerator.prepare(...,
        mixed_precision=...)), which reconciles the fp32 GNN/projector output
        with the bf16 frozen LM automatically. Plain inference (generate/
        generate_batch) has no such wrapper, so without this explicit cast,
        concatenating the fp32 prefix with bf16 token embeddings promotes the
        whole sequence to fp32 and crashes deep inside the LM's attention
        projections, which run at the LM's native (bf16) parameter dtype.
        """
        embedding_dtype = self.language_model.get_input_embeddings().weight.dtype
        vectors = self.encode_graphs(
            graphs,
            source_ids=source_ids,
            target_ids=target_ids,
            hops=hops,
            reasoning_types=reasoning_types,
        )
        return self.projector(vectors).to(dtype=embedding_dtype)

    def _encode_text(self, text: str, *, answer: bool) -> list[int]:
        token_ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        if not answer and self.config.add_bos_token and self.tokenizer.bos_token_id is not None:
            token_ids.insert(0, int(self.tokenizer.bos_token_id))
        if answer and self.config.add_eos_token and self.tokenizer.eos_token_id is not None:
            token_ids.append(int(self.tokenizer.eos_token_id))
        return token_ids

    # "before_answer": [prompt][graph tokens][answer] (this repo's original layout);
    # "prefix": [BOS][graph tokens][prompt][answer], as in CLEGR Eq. 2. Kept as a plain attribute
    # (set by the CLI), not a config field, so existing checkpoint metadata stays comparable.
    graph_token_position: str = "before_answer"

    def _assemble_prompt(self, prompt_ids: list[int], prefix: Any, embedding: Any) -> list[Any]:
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        if self.graph_token_position == "prefix":
            has_bos = bool(
                prompt_ids
                and self.config.add_bos_token
                and prompt_ids[0] == self.tokenizer.bos_token_id
            )
            lead = 1 if has_bos else 0
            parts = [embedding(prompt_tensor[:lead]), prefix, embedding(prompt_tensor[lead:])]
        else:
            parts = [embedding(prompt_tensor), prefix]
        return [part for part in parts if part.shape[0] > 0]

    def _truncate(
        self, prompt_ids: list[int], answer_ids: list[int]
    ) -> tuple[list[int], list[int]]:
        available = self.config.max_sequence_length - self.projector.config.prefix_tokens
        if available <= 0:
            raise ValueError("max_sequence_length must exceed prefix_tokens")
        if len(answer_ids) >= available:
            return [], answer_ids[:available]
        prompt_budget = available - len(answer_ids)
        return prompt_ids[-prompt_budget:], answer_ids

    def forward(
        self,
        *,
        graphs: Sequence[AttributedGraph | GraphTensor],
        prompts: Sequence[str],
        answers: Sequence[str],
        source_ids: Sequence[str | None] | None = None,
        target_ids: Sequence[str | None] | None = None,
        hops: Sequence[int | None] | None = None,
        reasoning_types: Sequence[str | None] | None = None,
    ) -> Any:
        """Compute causal loss over every answer token, masking prompt and graph prefix."""
        if not graphs or not (len(graphs) == len(prompts) == len(answers)):
            raise ValueError("graphs, prompts, and answers must be non-empty and equally sized")
        embedding = self.language_model.get_input_embeddings()
        prefixes = self.graph_prefix(
            graphs,
            source_ids=source_ids,
            target_ids=target_ids,
            hops=hops,
            reasoning_types=reasoning_types,
        )
        sequences: list[Any] = []
        label_rows: list[Any] = []
        for index, (prompt, answer) in enumerate(zip(prompts, answers, strict=True)):
            prompt_ids, answer_ids = self._truncate(
                self._encode_text(prompt, answer=False),
                self._encode_text(answer, answer=True),
            )
            if not answer_ids:
                raise ValueError("An answer produced no tokens")
            answer_tensor = torch.tensor(answer_ids, dtype=torch.long, device=self.device)
            text_parts = self._assemble_prompt(prompt_ids, prefixes[index], embedding)
            text_parts.append(embedding(answer_tensor))
            sequences.append(torch.cat(text_parts, dim=0))
            masked = len(prompt_ids) + self.projector.config.prefix_tokens
            label_rows.append(
                torch.cat(
                    [
                        torch.full((masked,), -100, dtype=torch.long, device=self.device),
                        answer_tensor,
                    ]
                )
            )
        max_length = max(sequence.shape[0] for sequence in sequences)
        hidden_dim = sequences[0].shape[-1]
        inputs_embeds = torch.zeros(
            (len(sequences), max_length, hidden_dim),
            dtype=sequences[0].dtype,
            device=self.device,
        )
        attention_mask = torch.zeros(
            (len(sequences), max_length), dtype=torch.long, device=self.device
        )
        labels = torch.full(
            (len(sequences), max_length), -100, dtype=torch.long, device=self.device
        )
        for index, (sequence, row_labels) in enumerate(zip(sequences, label_rows, strict=True)):
            length = sequence.shape[0]
            inputs_embeds[index, :length] = sequence
            attention_mask[index, :length] = 1
            labels[index, :length] = row_labels
        with self._lm_autocast():
            return self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )

    @torch.no_grad() if torch is not None else (lambda function: function)
    def generate(
        self,
        graph: AttributedGraph | GraphTensor,
        prompt: str,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        hops: int | None = None,
        reasoning_type: str | None = None,
        max_new_tokens: int = 64,
        **generation_kwargs: Any,
    ) -> str:
        prompt_ids = self._encode_text(prompt, answer=False)
        prompt_ids, _ = self._truncate(prompt_ids, [])
        prefix = self.graph_prefix(
            [graph],
            source_ids=[source_id],
            target_ids=[target_id],
            hops=[hops],
            reasoning_types=[reasoning_type],
        )[0]
        embedding = self.language_model.get_input_embeddings()
        inputs_embeds = torch.cat(
            self._assemble_prompt(prompt_ids, prefix, embedding), dim=0
        ).unsqueeze(0)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)
        with self._lm_autocast():
            generated = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **{"do_sample": False, **generation_kwargs},
            )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    @torch.no_grad() if torch is not None else (lambda function: function)
    def generate_text_only(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        **generation_kwargs: Any,
    ) -> str:
        token_ids = self._encode_text(prompt, answer=False)
        token_ids, _ = self._truncate(token_ids, [])
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        with self._lm_autocast():
            generated = self.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **{"do_sample": False, **generation_kwargs},
            )
        completion = generated[0, input_ids.shape[1] :]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()

    @staticmethod
    def _left_pad_embeds(sequences: list[Any], device: Any) -> tuple[Any, Any]:
        """Left-pad a list of [L_i, D] embedding tensors to [B, L_max, D] plus a
        matching attention mask. Left-padding keeps every sequence's last real
        token aligned at the same position, which is what causal-LM `.generate()`
        expects for a batch of unequal-length prompts."""
        max_length = max(sequence.shape[0] for sequence in sequences)
        hidden_dim = sequences[0].shape[-1]
        batch = torch.zeros(
            (len(sequences), max_length, hidden_dim),
            dtype=sequences[0].dtype,
            device=device,
        )
        mask = torch.zeros((len(sequences), max_length), dtype=torch.long, device=device)
        for index, sequence in enumerate(sequences):
            length = sequence.shape[0]
            batch[index, max_length - length :] = sequence
            mask[index, max_length - length :] = 1
        return batch, mask

    @torch.no_grad() if torch is not None else (lambda function: function)
    def generate_batch(
        self,
        graphs: Sequence[AttributedGraph | GraphTensor],
        prompts: Sequence[str],
        *,
        source_ids: Sequence[str | None] | None = None,
        target_ids: Sequence[str | None] | None = None,
        hops: Sequence[int | None] | None = None,
        reasoning_types: Sequence[str | None] | None = None,
        max_new_tokens: int = 64,
        **generation_kwargs: Any,
    ) -> list[str]:
        """Batched, graph-conditioned generation. Equivalent to calling
        ``generate`` once per example, but runs the whole batch through the
        language model in a single forward pass."""
        if not graphs or len(graphs) != len(prompts):
            raise ValueError("graphs and prompts must be non-empty and equally sized")
        embedding = self.language_model.get_input_embeddings()
        prefixes = self.graph_prefix(
            graphs,
            source_ids=source_ids,
            target_ids=target_ids,
            hops=hops,
            reasoning_types=reasoning_types,
        )
        sequences = []
        for index, prompt in enumerate(prompts):
            prompt_ids, _ = self._truncate(self._encode_text(prompt, answer=False), [])
            sequences.append(
                torch.cat(self._assemble_prompt(prompt_ids, prefixes[index], embedding), dim=0)
            )
        inputs_embeds, attention_mask = self._left_pad_embeds(sequences, self.device)
        with self._lm_autocast():
            generated = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **{"do_sample": False, **generation_kwargs},
            )
        return [self.tokenizer.decode(row, skip_special_tokens=True).strip() for row in generated]

    @torch.no_grad() if torch is not None else (lambda function: function)
    def generate_text_only_batch(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 64,
        **generation_kwargs: Any,
    ) -> list[str]:
        """Batched text-only generation (no graph tokens). Equivalent to
        calling ``generate_text_only`` once per prompt, batched into one
        forward pass."""
        if not prompts:
            raise ValueError("prompts must be non-empty")
        pad_id = int(self.tokenizer.pad_token_id)
        token_id_lists = [
            self._truncate(self._encode_text(prompt, answer=False), [])[0] for prompt in prompts
        ]
        max_length = max(len(ids) for ids in token_id_lists)
        input_ids = torch.full(
            (len(prompts), max_length), pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            (len(prompts), max_length), dtype=torch.long, device=self.device
        )
        for index, ids in enumerate(token_id_lists):
            length = len(ids)
            input_ids[index, max_length - length :] = torch.tensor(
                ids, dtype=torch.long, device=self.device
            )
            attention_mask[index, max_length - length :] = 1
        with self._lm_autocast():
            generated = self.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **{"do_sample": False, **generation_kwargs},
            )
        completions = generated[:, max_length:]
        return [self.tokenizer.decode(row, skip_special_tokens=True).strip() for row in completions]


def tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    """Return an exact, serializable tokenizer identity including its vocabulary."""
    vocab = tokenizer.get_vocab()
    backend = getattr(tokenizer, "backend_tokenizer", None)
    return {
        "class": _object_name(tokenizer),
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_sha256": _sha256_json(vocab),
        "vocab_size": len(vocab),
        "backend_sha256": (
            hashlib.sha256(backend.to_str().encode("utf-8")).hexdigest()
            if backend is not None
            else None
        ),
        "special_tokens": {
            key: getattr(tokenizer, key, None)
            for key in (
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "unk_token_id",
            )
        },
    }


def model_checkpoint_metadata(
    model: TEAGLM,
    *,
    training_config: Mapping[str, Any],
    data_split: Mapping[str, Any],
    data_sha256: str,
    step: int,
) -> dict[str, Any]:
    """Build metadata binding a checkpoint to exact backbones and training data."""
    lm_config = model.language_model.config.to_dict()
    return {
        "format": TEA_CHECKPOINT_FORMAT,
        "step": step,
        "gnn": {
            "class": _object_name(model.gnn),
            "config": asdict(model.gnn.config),
            "state_sha256": _state_dict_sha256(model.gnn),
        },
        "language_model": {
            "class": _object_name(model.language_model),
            "name_or_path": str(getattr(model.language_model.config, "_name_or_path", "")),
            "config_sha256": _sha256_json(lm_config),
        },
        "tokenizer": tokenizer_identity(model.tokenizer),
        "projector": {
            "class": _object_name(model.projector),
            "config": asdict(model.projector.config),
            "state_sha256": _state_dict_sha256(model.projector),
        },
        "tensorizer_config": asdict(model.tensorizer.config),
        "tea_glm_config": asdict(model.config),
        "training_config": dict(training_config),
        "data_split": dict(data_split),
        "data_sha256": data_sha256,
    }


def save_checkpoint_metadata(path: str | Path, metadata: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_canonical_json(dict(metadata)) + "\n", encoding="utf-8")
    return destination


def load_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format") != TEA_CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported TEA-GLM checkpoint metadata: {path}")
    return value


def assert_checkpoint_compatible(
    model: TEAGLM,
    metadata: Mapping[str, Any],
    *,
    component: str,
) -> None:
    """Reject external GNN/projector weights from incompatible shapes or backbones."""
    if component not in {"gnn", "projector"}:
        raise ValueError("component must be 'gnn' or 'projector'")
    expected = model_checkpoint_metadata(
        model,
        training_config={},
        data_split={},
        data_sha256="",
        step=0,
    )
    actual_component = metadata.get(component)
    if not isinstance(actual_component, Mapping):
        raise ValueError(f"Checkpoint has no {component} metadata")
    expected_component = expected[component]
    if actual_component.get("class") != expected_component["class"]:
        raise ValueError(f"Incompatible {component} class")
    if actual_component.get("config") != expected_component["config"]:
        raise ValueError(f"Incompatible {component} shape/configuration")
    for backbone in (
        "language_model",
        "tokenizer",
        "tensorizer_config",
        "tea_glm_config",
    ):
        if metadata.get(backbone) != expected[backbone]:
            raise ValueError(f"Checkpoint {backbone} does not match the active model")
    if component == "projector" and metadata.get("gnn") != expected["gnn"]:
        raise ValueError("Projector checkpoint was trained against a different GNN")


def load_external_component(
    model: TEAGLM,
    *,
    component: str,
    weights_path: str | Path,
    metadata_path: str | Path,
) -> None:
    """Load an external component only after strict metadata and tensor checks."""
    require_tea_dependencies()
    metadata = load_checkpoint_metadata(metadata_path)
    assert_checkpoint_compatible(model, metadata, component=component)
    target = model.gnn if component == "gnn" else model.projector
    checkpoint_path = Path(weights_path)
    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "Loading safetensors checkpoints requires `graph-modi[tea]`."
            ) from exc
        state_dict = load_file(str(checkpoint_path), device="cpu")
    else:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, Mapping):
        raise ValueError("Component checkpoint must contain a state dictionary")
    target.load_state_dict(state_dict, strict=True)
    expected_hash = metadata[component].get("state_sha256")
    if not expected_hash or _state_dict_sha256(target) != expected_hash:
        raise ValueError(f"{component} weights do not match checkpoint metadata")


class TEAGLMBackend:
    """GraphBackend adapter for inference through a trained TEA-GLM model."""

    def __init__(self, model: TEAGLM, *, max_new_tokens: int = 64) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.encode_calls = 0

    def encode(self, graph: AttributedGraph) -> GraphTensor:
        self.encode_calls += 1
        return self.model.tensorizer(graph)

    _EDIT_PROMPT_PREFIX = (
        "You maintain a metro graph. A revision may describe one or several changes "
        "at once, separated by \" ; \". Output every edit it describes, each in one "
        "of these formats, separated by \" ; \", and always finish with END:\n"
        "  SET NODE <station> status open\n"
        "  SET NODE <station> status closed\n"
        "  ADD EDGE <station> <station> transfer\n"
        "  DEL EDGE <station> <station> track\n"
        "  DEL EDGE <station> <station> transfer\n"
        "  NOOP <reason>\n"
        "Use the station name exactly as it appears in the revision. If the revision "
        "does not describe any change to the graph, output NOOP with a short reason. "
        "Always end your output with END.\n\n"
        "Revision: Elm is closed now.\n"
        "Edit: SET NODE Elm status closed ; END\n\n"
        "Revision: Service at Oak has resumed.\n"
        "Edit: SET NODE Oak status open ; END\n\n"
        "Revision: Please mark Pine as closed.\n"
        "Edit: SET NODE Pine status closed ; END\n\n"
        "Revision: ADD EDGE Elm Cedar transfer\n"
        "Edit: ADD EDGE Elm Cedar transfer ; END\n\n"
        "Revision: DEL EDGE Oak Pine track\n"
        "Edit: DEL EDGE Oak Pine track ; END\n\n"
        "Revision: NOOP: no graph update this turn.\n"
        "Edit: NOOP no change mentioned ; END\n\n"
        "Revision: Elm is closed now. ; ADD EDGE Elm Cedar transfer\n"
        "Edit: SET NODE Elm status closed ; ADD EDGE Elm Cedar transfer ; END\n\n"
        "Revision: Please mark Pine as closed. ; DEL EDGE Oak Pine track ; Service at Oak has resumed.\n"
        "Edit: SET NODE Pine status closed ; DEL EDGE Oak Pine track ; SET NODE Oak status open ; END\n\n"
    )

    _EDIT_MAX_NEW_TOKENS = 64

    @staticmethod
    def _extract_edit_program_text(generated: str) -> str:
        """Truncate at the first END marker.

        The model has no stop token for this format and will keep rambling
        into hallucinated "Revision:/Edit:" continuations past the program it
        actually meant to output; everything after the first END is noise.
        Falls back to the first line if the model never emits END (matches
        the old single-edit behavior for malformed output).
        """
        upper = generated.upper()
        end_index = upper.find("END")
        if end_index != -1:
            return generated[:end_index]
        return generated.split("\n", 1)[0]

    def predict_edit(self, utterance: str, graph: AttributedGraph) -> Any:
        """Extract the ordered edit program named by a revision sentence.

        Text-only generation, deliberately not graph-conditioned: the target
        node is always named explicitly in the utterance, so no graph
        information is needed to determine the edit. Empirically, running this
        through the graph-conditioned path corrupts generation instead of
        helping it — the graph prefix embeddings were only ever trained
        immediately before short QA-style prompts, so appending them after
        this longer few-shot instruction prompt is out of distribution for the
        frozen LM and produces incoherent output.
        """
        prompt = f"{self._EDIT_PROMPT_PREFIX}Revision: {utterance}\nEdit:"
        generated = self.model.generate_text_only(prompt, max_new_tokens=self._EDIT_MAX_NEW_TOKENS)
        program_text = self._extract_edit_program_text(generated)
        try:
            program = parse_edit_program(
                program_text,
                {node.label.casefold(): node.id for node in graph.nodes},
            )
        except ValueError:
            return None
        return program.edits or None

    def answer(self, model_input: ModelInput) -> str | None:
        text_only_conditions = {
            "question_only",
            "serialized_initial_history",
            "token_matched_history",
            "serialized_current_graph",
        }
        prompt = self.model.config.prompt_template.format(question=model_input.question)
        graph_once_after_first_turn = (
            model_input.condition == "graph_once_then_text" and model_input.turn_index > 0
        )
        if model_input.condition in text_only_conditions or graph_once_after_first_turn:
            return self.model.generate_text_only(
                prompt,
                max_new_tokens=self.max_new_tokens,
            )
        graph = model_input.encoded_graph
        if not isinstance(graph, GraphTensor):
            graph = self.encode(model_input.current_graph)
        return self.model.generate(
            graph,
            prompt,
            source_id=model_input.query.source,
            target_id=model_input.query.target,
            hops=model_input.query.hops,
            reasoning_type=model_input.query.reasoning_type.value,
            max_new_tokens=self.max_new_tokens,
        )

    def predict_edit_batch(self, items: Sequence[tuple[str, AttributedGraph]]) -> list[Any]:
        prompts = [
            f"{self._EDIT_PROMPT_PREFIX}Revision: {utterance}\nEdit:" for utterance, _ in items
        ]
        generated = self.model.generate_text_only_batch(
            prompts, max_new_tokens=self._EDIT_MAX_NEW_TOKENS
        )
        results: list[Any] = []
        for (_, graph), text in zip(items, generated, strict=True):
            program_text = self._extract_edit_program_text(text)
            try:
                program = parse_edit_program(
                    program_text, {node.label.casefold(): node.id for node in graph.nodes}
                )
            except ValueError:
                results.append(None)
                continue
            results.append(program.edits or None)
        return results

    def answer_batch(self, model_inputs: Sequence[ModelInput]) -> list[str | None]:
        text_only_conditions = {
            "question_only",
            "serialized_initial_history",
            "token_matched_history",
            "serialized_current_graph",
        }
        results: list[str | None] = [None] * len(model_inputs)
        text_indices: list[int] = []
        text_prompts: list[str] = []
        graph_indices: list[int] = []
        graph_graphs: list[Any] = []
        graph_prompts: list[str] = []
        graph_source_ids: list[str | None] = []
        graph_target_ids: list[str | None] = []
        graph_hops: list[int | None] = []
        graph_reasoning_types: list[str | None] = []
        for index, model_input in enumerate(model_inputs):
            prompt = self.model.config.prompt_template.format(question=model_input.question)
            graph_once_after_first_turn = (
                model_input.condition == "graph_once_then_text" and model_input.turn_index > 0
            )
            if model_input.condition in text_only_conditions or graph_once_after_first_turn:
                text_indices.append(index)
                text_prompts.append(prompt)
                continue
            graph = model_input.encoded_graph
            if not isinstance(graph, GraphTensor):
                graph = self.encode(model_input.current_graph)
            graph_indices.append(index)
            graph_graphs.append(graph)
            graph_prompts.append(prompt)
            graph_source_ids.append(model_input.query.source)
            graph_target_ids.append(model_input.query.target)
            graph_hops.append(model_input.query.hops)
            graph_reasoning_types.append(model_input.query.reasoning_type.value)
        if text_prompts:
            generated = self.model.generate_text_only_batch(
                text_prompts,
                max_new_tokens=self.max_new_tokens,
            )
            for index, output in zip(text_indices, generated, strict=True):
                results[index] = output
        if graph_prompts:
            for index, output in zip(
                graph_indices,
                self.model.generate_batch(
                    graph_graphs,
                    graph_prompts,
                    source_ids=graph_source_ids,
                    target_ids=graph_target_ids,
                    hops=graph_hops,
                    reasoning_types=graph_reasoning_types,
                    max_new_tokens=self.max_new_tokens,
                ),
                strict=True,
            ):
                results[index] = output
        return results


def data_records_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash canonicalized records without retaining an in-memory serialized corpus."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(_canonical_json(dict(record)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
