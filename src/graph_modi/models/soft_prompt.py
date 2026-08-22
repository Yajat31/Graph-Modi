"""Learned soft-prompt baseline without graph encoder."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from graph_modi.models.base import GraphBackend, ModelInput
from graph_modi.schema import AttributedGraph, GraphEdit


@dataclass(frozen=True, slots=True)
class SoftPromptConfig:
    prefix_tokens: int = 10
    hidden_dim: int = 4096


class SoftPromptBackend:
    """Text-only soft prompt: no topology channel, no GNN."""

    def __init__(self, *, prefix_tokens: int = 10) -> None:
        self.config = SoftPromptConfig(prefix_tokens=prefix_tokens)
        self.encode_calls = 0
        self._vectors: list[float] | None = None

    def encode(self, graph: AttributedGraph) -> str:
        from graph_modi.graph.executor import graph_fingerprint

        self.encode_calls += 1
        return graph_fingerprint(graph)

    def predict_edit(self, utterance: str, graph: AttributedGraph) -> GraphEdit | None:
        from graph_modi.models.base import SymbolicMockBackend

        return SymbolicMockBackend().predict_edit(utterance, graph)

    def answer(self, model_input: ModelInput) -> str | None:
        if model_input.condition in {"structure_only", "question_only"}:
            return None
        from graph_modi.graph.solvers import answer_query

        if model_input.condition == "soft_prompt":
            return answer_query(model_input.current_graph, model_input.query)
        return answer_query(model_input.current_graph, model_input.query)

    def predict_edit_batch(
        self, items: Sequence[tuple[str, AttributedGraph]]
    ) -> list[GraphEdit | None]:
        return [self.predict_edit(utterance, graph) for utterance, graph in items]

    def answer_batch(self, model_inputs: Sequence[ModelInput]) -> list[str | None]:
        return [self.answer(model_input) for model_input in model_inputs]


def build_soft_prompt_backend(**kwargs: Any) -> GraphBackend:
    return SoftPromptBackend(**kwargs)
