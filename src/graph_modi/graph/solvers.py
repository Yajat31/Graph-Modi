"""Deterministic symbolic solvers used to generate and audit labels."""

from __future__ import annotations

import heapq
from collections import deque

from graph_modi.schema import AttributedGraph, GraphQuery, ReasoningType, Scalar


def _available(attributes: dict[str, Scalar]) -> bool:
    return (
        attributes.get("status", "open") != "closed" and attributes.get("open", True) is not False
    )


def adjacency(
    graph: AttributedGraph,
    *,
    available_only: bool = True,
) -> dict[str, list[tuple[str, float]]]:
    nodes = graph.node_map()
    result: dict[str, list[tuple[str, float]]] = {node_id: [] for node_id in nodes}
    for edge in graph.edges:
        if available_only and (
            not _available(nodes[edge.source].attributes)
            or not _available(nodes[edge.target].attributes)
        ):
            continue
        result[edge.source].append((edge.target, edge.weight))
        if not graph.directed:
            result[edge.target].append((edge.source, edge.weight))
    for neighbors in result.values():
        neighbors.sort()
    return result


def shortest_path(
    graph: AttributedGraph,
    source: str,
    target: str,
) -> list[str]:
    nodes = graph.node_map()
    if source not in nodes or target not in nodes:
        return []
    if not _available(nodes[source].attributes) or not _available(nodes[target].attributes):
        return []
    neighbors = adjacency(graph)
    distances: dict[str, float] = {source: 0.0}
    paths: dict[str, tuple[str, ...]] = {source: (source,)}
    queue: list[tuple[float, tuple[str, ...], str]] = [(0.0, (source,), source)]
    while queue:
        distance, path_tuple, node = heapq.heappop(queue)
        if distance != distances.get(node) or path_tuple != paths.get(node):
            continue
        if node == target:
            return list(path_tuple)
        for neighbor, weight in neighbors[node]:
            candidate_distance = distance + weight
            candidate_path = (*path_tuple, neighbor)
            old_distance = distances.get(neighbor, float("inf"))
            old_path = paths.get(neighbor)
            if candidate_distance < old_distance or (
                candidate_distance == old_distance
                and (old_path is None or candidate_path < old_path)
            ):
                distances[neighbor] = candidate_distance
                paths[neighbor] = candidate_path
                heapq.heappush(queue, (candidate_distance, candidate_path, neighbor))
    return []


def reachable(graph: AttributedGraph, source: str, target: str) -> bool:
    return bool(shortest_path(graph, source, target))


def _cycle_member(graph: AttributedGraph, source: str) -> bool:
    if graph.directed or source not in graph.node_map():
        return False
    neighbors = [node for node, _ in adjacency(graph, available_only=False)[source]]
    if len(neighbors) < 2:
        return False
    blocked = source
    target_set = set(neighbors[1:])
    queue: deque[str] = deque([neighbors[0]])
    visited = {blocked, neighbors[0]}
    graph_adjacency = adjacency(graph, available_only=False)
    while queue:
        node = queue.popleft()
        if node in target_set:
            return True
        for neighbor, _ in graph_adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def answer_query(graph: AttributedGraph, query: GraphQuery) -> str:
    nodes = graph.node_map()
    if query.source not in nodes:
        return "invalid"
    if query.reasoning_type is ReasoningType.SHORTEST_PATH:
        if query.target is None:
            return "invalid"
        path = shortest_path(graph, query.source, query.target)
        return str(len(path)) if path else "unreachable"
    if query.reasoning_type is ReasoningType.REACHABILITY:
        if query.target is None:
            return "invalid"
        return "yes" if reachable(graph, query.source, query.target) else "no"
    if query.reasoning_type is ReasoningType.FILTERED_NEIGHBOR_COUNT:
        graph_adjacency = adjacency(graph, available_only=False)
        count = sum(
            nodes[neighbor].attributes.get(query.attribute or "") == query.value
            for neighbor, _ in graph_adjacency[query.source]
        )
        return str(count)
    if query.reasoning_type is ReasoningType.FILTERED_PATH_COUNT:
        if query.target is None:
            return "invalid"
        path = shortest_path(graph, query.source, query.target)
        if not path:
            return "unreachable"
        return str(
            sum(nodes[node].attributes.get(query.attribute or "") == query.value for node in path)
        )
    if query.reasoning_type is ReasoningType.CYCLE_MEMBERSHIP:
        return "yes" if _cycle_member(graph, query.source) else "no"
    if query.reasoning_type is ReasoningType.EDGE_EXISTS:
        if query.target is None:
            return "invalid"
        graph_adjacency = adjacency(graph, available_only=False)
        return (
            "yes"
            if any(neighbor == query.target for neighbor, _ in graph_adjacency[query.source])
            else "no"
        )
    if query.reasoning_type is ReasoningType.NODE_DEGREE:
        graph_adjacency = adjacency(graph, available_only=False)
        return str(len(graph_adjacency[query.source]))
    if query.reasoning_type is ReasoningType.NODE_COUNT:
        return str(len(graph.nodes))
    if query.reasoning_type is ReasoningType.PATH_COST:
        if query.target is None:
            return "invalid"
        path = shortest_path(graph, query.source, query.target)
        if not path:
            return "unreachable"
        total = 0.0
        neighbors = adjacency(graph)
        for left, right in zip(path, path[1:], strict=False):
            for neighbor, weight in neighbors[left]:
                if neighbor == right:
                    total += weight
                    break
        return str(int(total) if total == int(total) else round(total, 2))
    if query.reasoning_type is ReasoningType.LINK_PREDICTION:
        if query.target is None:
            return "invalid"
        graph_adjacency = adjacency(graph, available_only=False)
        return (
            "yes"
            if any(neighbor == query.target for neighbor, _ in graph_adjacency[query.source])
            else "no"
        )
    raise ValueError(f"Unsupported reasoning type: {query.reasoning_type}")


def render_question(query: GraphQuery, graph: AttributedGraph) -> str:
    if query.question:
        return query.question
    labels = {node.id: node.label for node in graph.nodes}
    source = labels.get(query.source, query.source)
    target = labels.get(query.target or "", query.target or "")
    if query.reasoning_type is ReasoningType.SHORTEST_PATH:
        return (
            f"How many stations, including endpoints, are on the shortest open "
            f"path from {source} to {target}? Answer with a number or unreachable."
        )
    if query.reasoning_type is ReasoningType.REACHABILITY:
        return f"Can {target} be reached from {source} using open stations? Answer yes or no."
    if query.reasoning_type is ReasoningType.FILTERED_NEIGHBOR_COUNT:
        return (
            f"How many stations directly connected to {source} have "
            f"{query.attribute} equal to {query.value}? Answer with a number."
        )
    if query.reasoning_type is ReasoningType.FILTERED_PATH_COUNT:
        return (
            f"On the shortest open path from {source} to {target}, how many stations "
            f"have {query.attribute} equal to {query.value}?"
        )
    if query.reasoning_type is ReasoningType.CYCLE_MEMBERSHIP:
        return f"Is {source} part of any cycle? Answer yes or no."
    if query.reasoning_type is ReasoningType.NODE_DEGREE:
        return f"What is the degree of {source}? Answer with a number."
    if query.reasoning_type is ReasoningType.NODE_COUNT:
        return "How many stations are in the network? Answer with a number."
    if query.reasoning_type is ReasoningType.PATH_COST:
        return (
            f"What is the total travel cost along the shortest open path from "
            f"{source} to {target}? Answer with a number or unreachable."
        )
    return f"Is there a direct edge between {source} and {target}? Answer yes or no."
