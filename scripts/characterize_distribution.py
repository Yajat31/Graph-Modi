#!/usr/bin/env python3
"""Report structural statistics for the locked graph distribution.

The numbers this prints are what a decision record should cite: whether degree
stays constant across node counts, whether closing a station on the shortest
path lengthens the trip instead of disconnecting it, and whether path lengths
have any spread. Runs on CPU with no downloads.
"""

from __future__ import annotations

import argparse
import json
import random
from statistics import mean, median
from typing import Any

from graph_modi.data.multiturn import (
    DEFAULT_GRAPH_DEGREE,
    DEFAULT_LINE_COUNT,
    DEFAULT_REWIRE_PROBABILITY,
    _make_graph,
)
from graph_modi.graph.executor import apply_edit
from graph_modi.graph.solvers import adjacency, shortest_path
from graph_modi.schema import (
    AttributedGraph,
    EditOperation,
    EditTarget,
    GraphEdit,
)

CLOSE_OUTCOMES = ("unreachable", "longer", "same_length")


def _neighbor_map(graph: AttributedGraph) -> dict[str, list[str]]:
    return {
        node_id: [neighbor for neighbor, _ in neighbors]
        for node_id, neighbors in adjacency(graph, available_only=False).items()
    }


def _reachable_count(
    neighbors: dict[str, list[str]],
    start: str,
    *,
    blocked_node: str | None = None,
    blocked_edge: frozenset[str] | None = None,
) -> int:
    seen = {start}
    frontier = [start]
    while frontier:
        node = frontier.pop()
        for neighbor in neighbors[node]:
            if neighbor == blocked_node or neighbor in seen:
                continue
            if blocked_edge is not None and frozenset((node, neighbor)) == blocked_edge:
                continue
            seen.add(neighbor)
            frontier.append(neighbor)
    return len(seen)


def _eccentricities(neighbors: dict[str, list[str]]) -> list[int]:
    result = []
    for source in neighbors:
        distances = {source: 0}
        frontier = [source]
        while frontier:
            node = frontier.pop(0)
            for neighbor in neighbors[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    frontier.append(neighbor)
        if len(distances) == len(neighbors):
            result.append(max(distances.values()))
    return result


def _local_clustering(neighbors: dict[str, list[str]]) -> float:
    coefficients = []
    for node_neighbors in neighbors.values():
        degree = len(node_neighbors)
        if degree < 2:
            coefficients.append(0.0)
            continue
        links = 0
        for index, left in enumerate(node_neighbors):
            left_neighbors = set(neighbors[left])
            links += sum(1 for right in node_neighbors[index + 1 :] if right in left_neighbors)
        coefficients.append(2 * links / (degree * (degree - 1)))
    return mean(coefficients) if coefficients else 0.0


def _close_station(graph: AttributedGraph, node_id: str) -> AttributedGraph:
    return apply_edit(
        graph,
        GraphEdit(
            operation=EditOperation.SET,
            target=EditTarget.NODE,
            node_id=node_id,
            attribute="status",
            value="closed",
        ),
    ).graph


def _summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": round(mean(values), 3),
        "median": round(median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def characterize(
    *,
    node_count_min: int,
    node_count_max: int,
    graphs: int,
    degree: int,
    rewire_probability: float,
    line_count: int,
    pair_sample: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    degrees: list[float] = []
    edge_counts: list[float] = []
    diameters: list[float] = []
    bridges: list[float] = []
    articulation_points: list[float] = []
    clustering: list[float] = []
    alternate_path_fraction: list[float] = []
    path_lengths: list[float] = []
    close_outcomes = dict.fromkeys(CLOSE_OUTCOMES, 0)
    line_sizes: list[float] = []

    for index in range(graphs):
        node_count = rng.randint(node_count_min, node_count_max)
        graph = _make_graph(
            rng,
            "train",
            index,
            node_count,
            degree=degree,
            rewire_probability=rewire_probability,
            line_count=line_count,
        )
        neighbors = _neighbor_map(graph)
        node_ids = list(neighbors)
        degrees.append(mean(len(neighbors[node_id]) for node_id in node_ids))
        edge_counts.append(len(graph.edges))
        eccentricities = _eccentricities(neighbors)
        if eccentricities:
            diameters.append(max(eccentricities))
        clustering.append(_local_clustering(neighbors))
        bridges.append(
            sum(
                1
                for edge in graph.edges
                if _reachable_count(
                    neighbors,
                    edge.source,
                    blocked_edge=frozenset((edge.source, edge.target)),
                )
                < node_count
            )
        )
        articulation_points.append(
            sum(
                1
                for node_id in node_ids
                if _reachable_count(
                    neighbors,
                    next(other for other in node_ids if other != node_id),
                    blocked_node=node_id,
                )
                < node_count - 1
            )
        )
        line_counts: dict[str, int] = {}
        for node in graph.nodes:
            line = str(node.attributes.get("line", ""))
            line_counts[line] = line_counts.get(line, 0) + 1
        line_sizes.extend(float(value) for value in line_counts.values())

        pairs = [
            (left, right)
            for left_index, left in enumerate(node_ids)
            for right in node_ids[left_index + 1 :]
        ]
        rng.shuffle(pairs)
        pairs = pairs[:pair_sample]
        with_alternate = 0
        counted = 0
        for source, target in pairs:
            path = shortest_path(graph, source, target)
            if not path:
                continue
            path_lengths.append(len(path))
            interior = path[1:-1]
            if not interior:
                continue
            counted += 1
            updated = _close_station(graph, interior[0])
            rerouted = shortest_path(updated, source, target)
            if rerouted:
                with_alternate += 1
                close_outcomes["longer" if len(rerouted) > len(path) else "same_length"] += 1
            else:
                close_outcomes["unreachable"] += 1
        if counted:
            alternate_path_fraction.append(with_alternate / counted)

    total_closures = sum(close_outcomes.values()) or 1
    return {
        "parameters": {
            "distribution": "watts_strogatz_metro_v1",
            "node_count_min": node_count_min,
            "node_count_max": node_count_max,
            "graph_degree": degree,
            "rewire_probability": rewire_probability,
            "line_count": line_count,
            "graphs": graphs,
            "seed": seed,
        },
        "mean_degree": _summarize(degrees),
        "edge_count": _summarize(edge_counts),
        "diameter": _summarize(diameters),
        "bridges": _summarize(bridges),
        "articulation_points": _summarize(articulation_points),
        "clustering": _summarize(clustering),
        "fraction_pairs_with_alternate_path": _summarize(alternate_path_fraction),
        "shortest_path_stations": _summarize(path_lengths),
        "stations_per_line": _summarize(line_sizes),
        "close_station_on_shortest_path": {
            outcome: round(count / total_closures, 3) for outcome, count in close_outcomes.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-count-min", type=int, default=15)
    parser.add_argument("--node-count-max", type=int, default=30)
    parser.add_argument("--graphs", type=int, default=200)
    parser.add_argument("--graph-degree", type=int, default=DEFAULT_GRAPH_DEGREE)
    parser.add_argument("--rewire-probability", type=float, default=DEFAULT_REWIRE_PROBABILITY)
    parser.add_argument("--line-count", type=int, default=DEFAULT_LINE_COUNT)
    parser.add_argument("--pair-sample", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = characterize(
        node_count_min=args.node_count_min,
        node_count_max=args.node_count_max,
        graphs=args.graphs,
        degree=args.graph_degree,
        rewire_probability=args.rewire_probability,
        line_count=args.line_count,
        pair_sample=args.pair_sample,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
