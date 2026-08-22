"""Validated immutable application of graph edits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from graph_modi.schema import (
    AttributedGraph,
    Edge,
    EditOperation,
    EditProgram,
    EditTarget,
    GraphEdit,
    Node,
)


@dataclass(frozen=True, slots=True)
class EditResult:
    graph: AttributedGraph
    applied: bool
    error: str | None = None


def _edge_key(edge: Edge, directed: bool) -> tuple[str, str, str]:
    source, target = edge.source, edge.target
    if not directed and source > target:
        source, target = target, source
    return source, target, edge.relation


def validate_graph(graph: AttributedGraph) -> None:
    ids = [node.id for node in graph.nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Graph contains duplicate node IDs")
    known = set(ids)
    seen: set[tuple[str, str, str]] = set()
    for edge in graph.edges:
        if edge.source not in known or edge.target not in known:
            raise ValueError(f"Edge references unknown node: {edge}")
        if edge.source == edge.target:
            raise ValueError("Self-loops are not supported")
        key = _edge_key(edge, graph.directed)
        if key in seen:
            raise ValueError(f"Duplicate edge: {key}")
        seen.add(key)


def graph_fingerprint(graph: AttributedGraph) -> str:
    def scalar_pairs(values: dict[str, Any]) -> list[tuple[str, Any]]:
        return sorted(values.items())

    nodes = sorted(
        (
            node.id,
            node.label,
            scalar_pairs(node.attributes),
        )
        for node in graph.nodes
    )
    edges = sorted(
        (
            *_edge_key(edge, graph.directed),
            edge.weight,
            scalar_pairs(edge.attributes),
        )
        for edge in graph.edges
    )
    payload = {
        "directed": graph.directed,
        "nodes": nodes,
        "edges": edges,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _failure(graph: AttributedGraph, message: str, strict: bool) -> EditResult:
    if strict:
        raise ValueError(message)
    return EditResult(graph=graph, applied=False, error=message)


def apply_edit_program(
    graph: AttributedGraph,
    program: EditProgram | Sequence[GraphEdit],
    *,
    strict: bool = False,
) -> EditResult:
    """Apply an ordered edit program sequentially."""
    edits = program.edits if isinstance(program, EditProgram) else tuple(program)
    current = graph
    any_applied = False
    for edit in edits:
        result = apply_edit(current, edit, strict=strict)
        current = result.graph
        any_applied = any_applied or result.applied
    return EditResult(graph=current, applied=any_applied)


def apply_edit(
    graph: AttributedGraph,
    edit: GraphEdit,
    *,
    strict: bool = False,
) -> EditResult:
    validate_graph(graph)
    if edit.operation is EditOperation.NOOP:
        return EditResult(graph=graph, applied=False)

    nodes = list(graph.nodes)
    edges = list(graph.edges)
    node_index = {node.id: index for index, node in enumerate(nodes)}

    if edit.target is EditTarget.NODE:
        if not edit.node_id:
            return _failure(graph, "Node edit requires node_id", strict)
        exists = edit.node_id in node_index
        if edit.operation is EditOperation.ADD:
            if exists:
                return _failure(graph, f"Node already exists: {edit.node_id}", strict)
            nodes.append(Node(edit.node_id, edit.label or edit.node_id))
        elif edit.operation is EditOperation.DEL:
            if not exists:
                return _failure(graph, f"Unknown node: {edit.node_id}", strict)
            nodes.pop(node_index[edit.node_id])
            edges = [edge for edge in edges if edit.node_id not in {edge.source, edge.target}]
        elif edit.operation is EditOperation.SET:
            if not exists:
                return _failure(graph, f"Unknown node: {edit.node_id}", strict)
            if not edit.attribute:
                return _failure(graph, "SET NODE requires an attribute", strict)
            old = nodes[node_index[edit.node_id]]
            attributes = dict(old.attributes)
            if attributes.get(edit.attribute) == edit.value:
                return EditResult(graph=graph, applied=False, error="no_state_change")
            attributes[edit.attribute] = edit.value
            nodes[node_index[edit.node_id]] = replace(old, attributes=attributes)
        else:
            return _failure(graph, f"Unsupported node edit: {edit.operation}", strict)

    elif edit.target is EditTarget.EDGE:
        if not edit.source or not edit.destination:
            return _failure(graph, "Edge edit requires source and destination", strict)
        if edit.source == edit.destination:
            return _failure(graph, "Self-loops are not supported", strict)
        if edit.source not in node_index or edit.destination not in node_index:
            return _failure(graph, "Edge references an unknown node", strict)
        candidate = Edge(
            source=edit.source,
            target=edit.destination,
            relation=edit.relation,
            weight=edit.weight,
        )
        key = _edge_key(candidate, graph.directed)
        matches = [
            index for index, edge in enumerate(edges) if _edge_key(edge, graph.directed) == key
        ]
        if edit.operation is EditOperation.ADD:
            if matches:
                return _failure(graph, f"Edge already exists: {key}", strict)
            edges.append(candidate)
        elif edit.operation is EditOperation.DEL:
            if not matches:
                return _failure(graph, f"Unknown edge: {key}", strict)
            edges.pop(matches[0])
        else:
            return _failure(graph, f"Unsupported edge edit: {edit.operation}", strict)
    else:
        return _failure(graph, "Non-NOOP edit requires a target", strict)

    updated = replace(graph, nodes=tuple(nodes), edges=tuple(edges))
    validate_graph(updated)
    return EditResult(graph=updated, applied=True)
