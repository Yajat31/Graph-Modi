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


def reachable_avoiding(
    graph: AttributedGraph,
    source: str,
    target: str,
    attribute: str,
    value: Scalar,
) -> bool:
    """Reachability that additionally excludes intermediate nodes matching a filter.

    Endpoints are always exempt from the exclusion, so a query about a source or
    target that happens to match the avoided class still has a well-defined answer.
    """
    nodes = graph.node_map()
    if source not in nodes or target not in nodes:
        return False

    def allowed(node_id: str) -> bool:
        if node_id in (source, target):
            return True
        return nodes[node_id].attributes.get(attribute) != value

    neighbors = adjacency(graph, available_only=False)
    visited = {source}
    queue: deque[str] = deque([source])
    while queue:
        node = queue.popleft()
        if node == target:
            return True
        for neighbor, _ in neighbors[node]:
            if neighbor not in visited and allowed(neighbor):
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def nodes_within_hops(
    graph: AttributedGraph,
    source: str,
    hops: int,
    *,
    available_only: bool = False,
) -> dict[str, int]:
    """Map of node id -> hop distance from ``source``, excluding ``source`` itself."""
    neighbors = adjacency(graph, available_only=available_only)
    distances: dict[str, int] = {source: 0}
    queue: deque[str] = deque([source])
    while queue:
        node = queue.popleft()
        if distances[node] >= hops:
            continue
        for neighbor, _ in neighbors[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    distances.pop(source, None)
    return distances


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
    if query.reasoning_type is ReasoningType.CONSTRAINED_REACHABILITY:
        if query.target is None or query.attribute is None:
            return "invalid"
        return (
            "yes"
            if reachable_avoiding(graph, query.source, query.target, query.attribute, query.value)
            else "no"
        )
    if query.reasoning_type is ReasoningType.WITHIN_HOPS_COUNT:
        hops = query.hops or 2
        in_range = nodes_within_hops(graph, query.source, hops)
        if query.attribute is not None:
            return str(
                sum(
                    nodes[node_id].attributes.get(query.attribute) == query.value
                    for node_id in in_range
                )
            )
        return str(len(in_range))
    if query.reasoning_type is ReasoningType.WITHIN_HOPS_LIST:
        hops = query.hops or 2
        in_range = nodes_within_hops(graph, query.source, hops)
        matching = [
            node_id
            for node_id in in_range
            if query.attribute is None
            or nodes[node_id].attributes.get(query.attribute) == query.value
        ]
        labels = sorted(nodes[node_id].label for node_id in matching)
        return ", ".join(labels) if labels else "none"
    if query.reasoning_type is ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS:
        if query.attribute is None:
            return "invalid"
        hops = query.hops or 2
        sample_ids = set(nodes_within_hops(graph, query.source, hops)) | {query.source}
        counts: dict[str, int] = {}
        for node_id in sample_ids:
            attribute_value = nodes[node_id].attributes.get(query.attribute)
            if attribute_value is None:
                continue
            key = str(attribute_value)
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return "none"
        best_count = max(counts.values())
        winners = sorted(key for key, count in counts.items() if count == best_count)
        return winners[0]
    if query.reasoning_type is ReasoningType.ATTRIBUTE_LOOKUP:
        if query.attribute is None:
            return "invalid"
        value = nodes[query.source].attributes.get(query.attribute)
        return "none" if value is None else str(value)
    if query.reasoning_type is ReasoningType.ATTRIBUTE_CHECK:
        if query.attribute is None:
            return "invalid"
        return "yes" if nodes[query.source].attributes.get(query.attribute) == query.value else "no"
    raise ValueError(f"Unsupported reasoning type: {query.reasoning_type}")


def _node_wfi_text(
    graph: AttributedGraph,
    node_id: str | None,
    exclude: frozenset[str] = frozenset(),
) -> str | None:
    """W(f_i): the node's own textual attributes, as the CLEGR paper's Eq. 1
    concatenates alongside the graph-encoded prefix
    (clegr.md section 0: "M_l(M_P(M_g(G,n_i)) || W(f_i) || W(q))").
    None when node_id is absent, so graph-level questions get no W(f_i) text.
    ``exclude`` lists attribute names withheld from the text (they stay in the
    graph itself): dynamic state such as ``status`` and, for Facts questions,
    the attribute being asked about."""
    if not node_id:
        return None
    node = next((n for n in graph.nodes if n.id == node_id), None)
    if node is None:
        return None
    attrs = ", ".join(
        f"{key}={value}" for key, value in sorted(node.attributes.items()) if key not in exclude
    )
    return f"{node.label}: {attrs}." if attrs else f"{node.label}."


def _prepend_wfi(
    question: str,
    graph: AttributedGraph,
    source_id: str | None,
    target_id: str | None,
    exclude: frozenset[str] = frozenset(),
) -> str:
    texts = [
        t
        for t in (
            _node_wfi_text(graph, source_id, exclude),
            _node_wfi_text(graph, target_id, exclude),
        )
        if t
    ]
    if not texts:
        return question
    return " ".join(dict.fromkeys(texts)) + "\n" + question


# Tasks whose answer is (or, for the mode, is largely decided by) the source's own value of the
# queried attribute: that attribute is withheld from the W(f_i) text so the text cannot give it away.
_FACTS_TYPES = frozenset(
    {
        ReasoningType.ATTRIBUTE_LOOKUP,
        ReasoningType.ATTRIBUTE_CHECK,
        ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS,
    }
)


def wfi_excluded_attributes(query: GraphQuery, graph: AttributedGraph) -> frozenset[str]:
    """Attributes withheld from the W(f_i) text: the graph's ``wfi_exclude`` list
    (comma-separated in metadata) plus the queried attribute on Facts questions."""
    excluded = {name for name in str(graph.metadata.get("wfi_exclude", "")).split(",") if name}
    if query.reasoning_type in _FACTS_TYPES and query.attribute:
        excluded.add(query.attribute)
    return frozenset(excluded)


def render_question(query: GraphQuery, graph: AttributedGraph) -> str:
    if query.question:
        return query.question
    return _prepend_wfi(
        _render_question_text(query, graph),
        graph,
        query.source,
        query.target,
        wfi_excluded_attributes(query, graph),
    )


def _render_question_text(query: GraphQuery, graph: AttributedGraph) -> str:
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
    if query.reasoning_type is ReasoningType.CONSTRAINED_REACHABILITY:
        return (
            f"Can {target} be reached from {source} without passing through any station "
            f"where {query.attribute} is {query.value}? Answer yes or no."
        )
    if query.reasoning_type is ReasoningType.WITHIN_HOPS_COUNT:
        hops = query.hops or 2
        if query.attribute is not None:
            return (
                f"How many other stations within {hops} hops of {source} have "
                f"{query.attribute} equal to {query.value}? Answer with a number."
            )
        return f"How many other stations are within {hops} hops of {source}? Answer with a number."
    if query.reasoning_type is ReasoningType.WITHIN_HOPS_LIST:
        hops = query.hops or 2
        if query.attribute is not None:
            return (
                f"Which stations within {hops} hops of {source} have {query.attribute} "
                f"equal to {query.value}? Answer with a comma-separated list, or 'none'."
            )
        return (
            f"Which stations are within {hops} hops of {source}? "
            "Answer with a comma-separated list, or 'none'."
        )
    if query.reasoning_type is ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS:
        hops = query.hops or 2
        return (
            f"What is the most common {query.attribute} among stations within {hops} "
            f"hops of {source}, including {source} itself?"
        )
    if query.reasoning_type is ReasoningType.ATTRIBUTE_LOOKUP:
        return f"What is the {query.attribute} of {source}? Answer directly."
    if query.reasoning_type is ReasoningType.ATTRIBUTE_CHECK:
        return f"Does {source} have {query.attribute} equal to {query.value}? Answer yes or no."
    return f"Is there a direct edge between {source} and {target}? Answer yes or no."
