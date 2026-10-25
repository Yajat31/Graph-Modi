"""Regression tests for the cycle_membership solver (2026-10-22 benchmark audit, finding B1).

The old solver only tried the first neighbour as a start, so it said "no" for a station on a cycle
whenever its first neighbour was a dead-end or bridge branch.
"""

from __future__ import annotations

import random
from collections import deque

from graph_modi.graph.solvers import answer_query
from graph_modi.schema import AttributedGraph, Edge, GraphQuery, Node, ReasoningType


def _graph(node_ids: list[str], pairs: list[tuple[str, str]]) -> AttributedGraph:
    return AttributedGraph(
        graph_id="t",
        nodes=tuple(Node(i, i) for i in node_ids),
        edges=tuple(Edge(a, b) for a, b in pairs),
    )


def _cycle(graph: AttributedGraph, source: str) -> str:
    return answer_query(graph, GraphQuery(ReasoningType.CYCLE_MEMBERSHIP, source))


def test_station_on_triangle_behind_a_dead_end_neighbour() -> None:
    # "a" sorts first and is a pendant; "b" and "c" close a triangle with "s". The old solver
    # started from "a", could not reach "b" or "c" without "s", and answered "no".
    graph = _graph(["s", "a", "b", "c"], [("s", "a"), ("s", "b"), ("s", "c"), ("b", "c")])
    assert _cycle(graph, "s") == "yes"


def test_bridge_stations_and_leaves_are_not_on_cycles() -> None:
    graph = _graph(
        ["a", "b", "c", "d", "e", "f"],
        [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "c"), ("c", "f")],
    )
    assert _cycle(graph, "a") == "no"  # leaf
    assert _cycle(graph, "b") == "no"  # degree 2 on a bridge chain
    assert _cycle(graph, "f") == "no"  # leaf
    assert _cycle(graph, "c") == "yes"  # triangle c-d-e
    assert _cycle(graph, "d") == "yes"


def test_two_cycles_sharing_a_station_and_a_cut_vertex_with_a_late_cycle() -> None:
    # the only cycle through "s" uses its last two neighbours
    graph = _graph(
        ["s", "a", "b", "c", "d", "e"],
        [("s", "a"), ("a", "b"), ("s", "c"), ("c", "d"), ("d", "e"), ("e", "s")],
    )
    assert _cycle(graph, "s") == "yes"
    assert _cycle(graph, "a") == "no"
    assert _cycle(graph, "b") == "no"


def _witness_exists(graph: AttributedGraph, source: str) -> bool:
    """Reference: two neighbours of ``source`` joined by a path that avoids ``source``."""
    adjacency: dict[str, set[str]] = {n.id: set() for n in graph.nodes}
    for edge in graph.edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
    neighbors = sorted(adjacency[source])
    for index, first in enumerate(neighbors):
        for second in neighbors[index + 1 :]:
            seen = {first}
            queue = deque([first])
            while queue:
                node = queue.popleft()
                for nxt in adjacency[node]:
                    if nxt != source and nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            if second in seen:
                return True
    return False


def test_solver_matches_reference_on_random_graphs() -> None:
    rng = random.Random(0)
    checked = yes = 0
    for _ in range(300):
        n = rng.randint(4, 12)
        ids = [f"n{i}" for i in range(n)]
        pairs = {tuple(sorted(rng.sample(ids, 2))) for _ in range(rng.randint(n - 1, 2 * n))}
        graph = _graph(ids, sorted(pairs))
        for source in ids:
            expected = "yes" if _witness_exists(graph, source) else "no"
            assert _cycle(graph, source) == expected, (sorted(pairs), source)
            checked += 1
            yes += expected == "yes"
    assert checked > 2000 and 0 < yes < checked  # both answers are exercised
