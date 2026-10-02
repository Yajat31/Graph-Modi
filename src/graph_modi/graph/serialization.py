"""Stable text representations for matched text-only baselines."""

from __future__ import annotations

from graph_modi.schema import AttributedGraph


def serialize_graph(graph: AttributedGraph) -> str:
    nodes = []
    for node in sorted(graph.nodes, key=lambda item: item.id):
        attributes = ", ".join(f"{key}={value}" for key, value in sorted(node.attributes.items()))
        nodes.append(f"{node.id}:{node.label}[{attributes}]")
    edges = []
    for edge in sorted(
        graph.edges,
        key=lambda item: (item.source, item.target, item.relation),
    ):
        edges.append(
            f"{edge.source}-{edge.target}(relation={edge.relation},weight={edge.weight:g})"
        )
    return "Nodes: " + "; ".join(nodes) + ". Edges: " + "; ".join(edges) + "."


def serialize_history(utterances: list[str]) -> str:
    return "\n".join(
        f"Update {index + 1}: {utterance}" for index, utterance in enumerate(utterances)
    )


def token_budget_match(text: str, target_tokens: int) -> str:
    """Whitespace-token approximation used only by the explicit budget control."""
    words = text.split()
    if len(words) <= target_tokens:
        return text
    return " ".join(words[-target_tokens:])

