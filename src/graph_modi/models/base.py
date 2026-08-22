"""Backend protocol and a deterministic, download-free test backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from graph_modi.graph.edits import parse_edit
from graph_modi.graph.solvers import answer_query
from graph_modi.schema import AttributedGraph, GraphEdit, GraphQuery


@dataclass(frozen=True, slots=True)
class ModelInput:
    initial_graph: AttributedGraph
    current_graph: AttributedGraph
    query: GraphQuery
    question: str
    history: tuple[str, ...]
    condition: str
    encoded_graph: Any = None
    turn_index: int = 0


class GraphBackend(Protocol):
    encode_calls: int

    def encode(self, graph: AttributedGraph) -> Any:
        """Encode one graph state into model-consumable graph tokens."""

    def predict_edit(
        self,
        utterance: str,
        graph: AttributedGraph,
    ) -> GraphEdit | None:
        """Convert one natural-language revision into a typed edit."""

    def answer(self, model_input: ModelInput) -> str | None:
        """Answer a graph question under an explicit evaluation condition."""


class SymbolicMockBackend:
    """A transparent oracle used by unit tests and the CPU smoke path."""

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, graph: AttributedGraph) -> str:
        from graph_modi.graph.executor import graph_fingerprint

        self.encode_calls += 1
        return graph_fingerprint(graph)

    def predict_edit(
        self,
        utterance: str,
        graph: AttributedGraph,
    ) -> GraphEdit | None:
        names = {node.label.casefold(): node.id for node in graph.nodes}
        normalized = utterance.strip().rstrip(".")
        lowered = normalized.casefold()
        status = None
        if any(token in lowered for token in ("closed", "suspended")):
            status = "closed"
        elif any(token in lowered for token in ("open", "reopened", "resumed")):
            status = "open"
        if status is not None:
            for label, node_id in names.items():
                if label in lowered:
                    return parse_edit(f"SET NODE {node_id} status {status}")
        try:
            return parse_edit(normalized, names)
        except ValueError:
            return None

    def answer(self, model_input: ModelInput) -> str | None:
        if model_input.condition == "question_only":
            return None
        if model_input.condition in {
            "frozen_graph_history",
            "cached_no_reencode",
            "graph_once_then_text",
        }:
            graph = model_input.initial_graph
        else:
            graph = model_input.current_graph
        return answer_query(graph, model_input.query)
