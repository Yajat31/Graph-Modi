"""BERT-768 node tensorizer, matching the CLEGR paper's own GLM input features
(clegr.md Table 3: In Dim 768, from bert-base-uncased sentence representations
of each node's attributes) -- a drop-in replacement for
DeterministicNodeTensorizer's arbitrary feature-hashing scheme. Produces the
exact same GraphTensor interface, so no downstream code (GraphSAGEEncoder,
TEAGLM, training.py) needs to change to consume it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graph_modi.models.tea_glm import GraphTensor, require_tea_dependencies
from graph_modi.schema import AttributedGraph

_TEA_IMPORT_ERROR: ImportError | None = None
try:
    import torch
except ImportError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TEA_IMPORT_ERROR = exc


@dataclass(frozen=True, slots=True)
class BertNodeTensorizerConfig:
    bert_model_name: str = "bert-base-uncased"
    feature_dim: int = 768
    include_label: bool = True
    include_attributes: bool = True
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")


def _node_text(node: Any, config: BertNodeTensorizerConfig) -> str:
    parts = []
    if config.include_label:
        parts.append(node.label)
    if config.include_attributes:
        attrs = ", ".join(f"{key}={value}" for key, value in sorted(node.attributes.items()))
        if attrs:
            parts.append(attrs)
    return ": ".join(parts) if parts else node.label


class BertNodeTensorizer:
    """Encodes each node's textual description with a frozen BERT encoder,
    mean-pooling the last hidden state -- matching CLEGR's "sentence
    representation of each node... encoded into a 768-dimensional embedding
    using bert-base-uncased" (clegr.md section 2.2). BERT itself is never
    trained; this is a fixed feature extractor, same role as the paper's.

    Caches by exact node text (label+attributes), since the same text
    recurs across many training examples referencing the same graph, and
    running BERT is far more expensive than the deterministic hash it
    replaces.
    """

    def __init__(self, config: BertNodeTensorizerConfig | None = None) -> None:
        require_tea_dependencies()
        if _TEA_IMPORT_ERROR is not None:
            raise RuntimeError("BertNodeTensorizer requires torch") from _TEA_IMPORT_ERROR
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BertNodeTensorizer requires transformers. Install `graph-modi[tea]`."
            ) from exc

        self.config = config or BertNodeTensorizerConfig()
        self._device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.bert_model_name)
        self._bert = AutoModel.from_pretrained(self.config.bert_model_name).to(self._device)
        self._bert.eval()
        for parameter in self._bert.parameters():
            parameter.requires_grad_(False)
        self._cache: dict[str, Any] = {}

    def _embed_texts(self, texts: list[str]) -> Any:
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            batch = self._tokenizer(
                uncached, padding=True, truncation=True, max_length=64, return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                output = self._bert(**batch)
                mask = batch["attention_mask"].unsqueeze(-1).to(output.last_hidden_state.dtype)
                summed = (output.last_hidden_state * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp_min(1.0)
                pooled = (summed / counts).cpu()
            for text, vector in zip(uncached, pooled, strict=True):
                self._cache[text] = vector
        return torch.stack([self._cache[t] for t in texts])

    def __call__(self, graph: AttributedGraph) -> GraphTensor:
        node_map = graph.node_map()
        node_ids = tuple(sorted(node_map))
        if not node_ids:
            raise ValueError(f"Graph {graph.graph_id!r} has no nodes")
        index = {node_id: position for position, node_id in enumerate(node_ids)}
        texts = [_node_text(node_map[node_id], self.config) for node_id in node_ids]
        x = self._embed_texts(texts)

        directed_edges: list[tuple[int, int, float, str]] = []
        for edge in graph.edges:
            if edge.source not in index or edge.target not in index:
                raise ValueError(f"Edge {edge.source!r}->{edge.target!r} references an unknown node")
            directed_edges.append((index[edge.source], index[edge.target], float(edge.weight), edge.relation))
            if not graph.directed and edge.source != edge.target:
                directed_edges.append((index[edge.target], index[edge.source], float(edge.weight), edge.relation))
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
            x=x.to(dtype=torch.float32),
            edge_index=edge_index,
            edge_weight=edge_weight,
            node_ids=node_ids,
            graph_id=graph.graph_id,
        )
