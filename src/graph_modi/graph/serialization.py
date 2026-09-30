"""Stable text representations for matched text-only baselines."""

from __future__ import annotations

import re

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


_NODE_COLUMNS = (
    "disabled_access", "has_rail", "architecture", "cleanliness", "music", "size", "line", "status",
)
_EDGE_COLUMNS = ("line_color", "line_stroke", "has_aircon", "built")
CSV_INSTRUCTION = (
    "Above is the representation of a synthetic subway network. All stations and lines are "
    "completely fictional. Keep in mind that the subway network is not real. All information "
    "necessary to answer the question is present in the above representation."
)


def plain_question(question: str) -> str:
    """The question sentence without the leading W(f_i) node-text lines."""
    return question.rsplit("\n", 1)[-1]


def _csv_cell(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return '""'
    if isinstance(value, (int, float)):
        return f"{value:g}" if isinstance(value, float) else str(value)
    return '"' + str(value).replace('"', "'") + '"'


def _natural(identifier: str) -> tuple[int, str]:
    return (len(identifier), identifier)


def graph_to_csv(graph: AttributedGraph) -> tuple[str, str]:
    """CLEGR-style node and edge CSV (id/name first, then attribute columns)."""
    present = {key for node in graph.nodes for key in node.attributes}
    node_columns = [c for c in _NODE_COLUMNS if c in present] + sorted(present - set(_NODE_COLUMNS))
    node_lines = [",".join(f'"{c}"' for c in ("id", "name", *node_columns))]
    for node in sorted(graph.nodes, key=lambda item: _natural(item.id)):
        cells = [_csv_cell(node.id), _csv_cell(node.label)]
        cells += [_csv_cell(node.attributes.get(column)) for column in node_columns]
        node_lines.append(",".join(cells))
    present_edge = {key for edge in graph.edges for key in edge.attributes}
    edge_columns = [c for c in _EDGE_COLUMNS if c in present_edge] + sorted(
        present_edge - set(_EDGE_COLUMNS)
    )
    edge_lines = [",".join(f'"{c}"' for c in ("source_id", "target_id", *edge_columns))]
    for edge in sorted(graph.edges, key=lambda item: (_natural(item.source), _natural(item.target))):
        cells = [_csv_cell(edge.source), _csv_cell(edge.target)]
        cells += [_csv_cell(edge.attributes.get(column)) for column in edge_columns]
        edge_lines.append(",".join(cells))
    return "\n".join(node_lines), "\n".join(edge_lines)


_ANSWER_TAIL = re.compile(
    r"\s*Answer (?P<kind>yes or no|with a number or unreachable|with a number|directly)\.?\s*$"
)
# CLEGR Appendix E.1.7 answer-format suffixes (unreachable is our one addition to the count suffix).
_SUFFIX_BOOL = "Answer with 'True' or 'False':\n\nAnswer:"
_SUFFIX_CYCLE = "Answer with 'True' if it is in a cycle, otherwise 'False':\n\nAnswer:"
_SUFFIX_COUNT = "Answer with a number:\n\nAnswer:"
_SUFFIX_COUNT_UNREACHABLE = "Answer with a number or 'unreachable':\n\nAnswer:"
_SUFFIX_STRING = "Answer directly:"


def split_answer_format(question: str) -> tuple[str, str]:
    """Split our question sentence from its trailing instruction and pick the CLEGR suffix."""
    core = plain_question(question).strip()
    match = _ANSWER_TAIL.search(core)
    kind = match.group("kind") if match else "directly"
    if match:
        core = core[: match.start()].rstrip()
    if kind == "yes or no":
        suffix = _SUFFIX_CYCLE if "part of any cycle" in core else _SUFFIX_BOOL
    elif kind == "with a number":
        suffix = _SUFFIX_COUNT
    elif kind == "with a number or unreachable":
        suffix = _SUFFIX_COUNT_UNREACHABLE
    else:
        suffix = _SUFFIX_STRING
    return core, suffix


def csv_graph_prompt(graph: AttributedGraph, question: str, *, updates: list[str] | None = None) -> str:
    """CLEGR Appendix E.1.7 prompt body: node CSV, edge CSV, instruction, the question, then the
    answer-format suffix. The ``[INST] ... [/INST]`` wrapper comes from the model's prompt template.

    ``updates`` (edit utterances since the CSV was recorded) go between the edge CSV and the
    instruction, for the initial-graph-plus-history condition."""
    nodes, edges = graph_to_csv(graph)
    core, suffix = split_answer_format(question)
    parts = ["--- Nodes ---", nodes, "--- Edges ---", edges, CSV_INSTRUCTION + " The question is:"]
    if updates:
        parts.insert(-1, "Updates since this representation was recorded:")
        for offset, text in enumerate(updates):
            parts.insert(-1, f"Update {offset + 1}: {text}")
    parts.append(core)
    parts.append("")
    parts.append(suffix)
    return "\n".join(parts)
