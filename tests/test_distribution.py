import random

import pytest

from graph_modi.data.multiturn import (
    DEFAULT_GRAPH_DEGREE,
    DEFAULT_LINE_COUNT,
    DEFAULT_REWIRE_PROBABILITY,
    DISTRIBUTION,
    TRACK_RELATION,
    TRANSFER_RELATION,
    _candidate_edit,
    _make_graph,
    _queries,
    _utterance,
    generate_dataset,
)
from graph_modi.graph.edits import execution_equivalent, parse_edit
from graph_modi.graph.executor import apply_edit, validate_graph
from graph_modi.graph.solvers import adjacency, answer_query
from graph_modi.schema import (
    AttributedGraph,
    EditOperation,
    EditTarget,
    GraphEdit,
    ReasoningType,
)


def build(
    node_count: int = 20,
    *,
    seed: int = 3,
    degree: int = DEFAULT_GRAPH_DEGREE,
    rewire_probability: float = DEFAULT_REWIRE_PROBABILITY,
    line_count: int = DEFAULT_LINE_COUNT,
) -> AttributedGraph:
    return _make_graph(
        random.Random(seed),
        "train",
        0,
        node_count,
        degree=degree,
        rewire_probability=rewire_probability,
        line_count=line_count,
    )


def neighbors_of(graph: AttributedGraph) -> dict[str, list[str]]:
    return {
        node_id: [neighbor for neighbor, _ in entries]
        for node_id, entries in adjacency(graph, available_only=False).items()
    }


def is_connected(graph: AttributedGraph) -> bool:
    neighbors = neighbors_of(graph)
    start = graph.nodes[0].id
    seen = {start}
    frontier = [start]
    while frontier:
        for neighbor in neighbors[frontier.pop()]:
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return len(seen) == len(graph.nodes)


@pytest.mark.parametrize("node_count", [8, 15, 20, 30])
def test_degree_is_constant_across_node_counts(node_count: int) -> None:
    graph = build(node_count)
    assert len(graph.edges) == node_count * DEFAULT_GRAPH_DEGREE // 2
    validate_graph(graph)
    assert is_connected(graph)


def test_relations_are_track_or_transfer() -> None:
    graph = build()
    relations = {edge.relation for edge in graph.edges}
    assert relations <= {TRACK_RELATION, TRANSFER_RELATION}
    assert graph.metadata["distribution"] == DISTRIBUTION


def test_ring_adjacency_is_not_recoverable_from_node_ids() -> None:
    """Positions are permuted onto ids, so id arithmetic must not reveal edges."""
    offset = DEFAULT_GRAPH_DEGREE // 2
    total = 0
    id_adjacent = 0
    for seed in range(8):
        graph = build(20, seed=seed)
        for edge in graph.edges:
            total += 1
            difference = abs(int(edge.source[1:]) - int(edge.target[1:]))
            if difference <= offset:
                id_adjacent += 1
    assert total
    assert id_adjacent / total < 0.5


def test_lines_are_contiguous_ring_arcs() -> None:
    graph = build(20, rewire_probability=0.0)
    by_line: dict[str, list[str]] = {}
    for node in graph.nodes:
        by_line.setdefault(str(node.attributes["line"]), []).append(node.id)
    assert len(by_line) == DEFAULT_LINE_COUNT
    sizes = [len(members) for members in by_line.values()]
    assert max(sizes) - min(sizes) <= 1

    # An arc of an unrewired ring lattice is connected within itself.
    neighbors = neighbors_of(graph)
    for members in by_line.values():
        group = set(members)
        seen = {members[0]}
        frontier = [members[0]]
        while frontier:
            for neighbor in neighbors[frontier.pop()]:
                if neighbor in group and neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append(neighbor)
        assert seen == group


def test_added_edges_carry_an_explicit_relation() -> None:
    graph = build()
    edit = _candidate_edit(graph, EditOperation.ADD, random.Random(0))
    assert edit is not None
    assert edit.relation == TRANSFER_RELATION
    assert apply_edit(graph, edit).applied


@pytest.mark.parametrize("operation", [EditOperation.ADD, EditOperation.DEL])
def test_edge_utterances_round_trip_with_non_default_relations(
    operation: EditOperation,
) -> None:
    graph = build()
    edit = _candidate_edit(graph, operation, random.Random(1))
    assert edit is not None
    utterance = _utterance(edit, graph, "direct")
    parsed = parse_edit(utterance, {node.label.casefold(): node.id for node in graph.nodes})
    assert execution_equivalent(parsed, edit)
    assert apply_edit(graph, parsed).applied


def test_line_valued_neighbor_queries_are_generated_and_solvable() -> None:
    graph = build()
    queries = list(_queries(graph, ReasoningType.FILTERED_NEIGHBOR_COUNT))
    line_queries = [query for query in queries if query.attribute == "line"]
    assert line_queries
    for query in line_queries[:5]:
        assert answer_query(graph, query).isdigit()


def test_node_degree_queries_are_generated_and_solvable() -> None:
    graph = build()
    queries = list(_queries(graph, ReasoningType.NODE_DEGREE))
    assert len(queries) == len(graph.nodes)
    for query in queries[:5]:
        assert answer_query(graph, query).isdigit()


def test_constrained_reachability_queries_are_generated_and_solvable() -> None:
    graph = build()
    queries = list(_queries(graph, ReasoningType.CONSTRAINED_REACHABILITY))
    assert queries
    for query in queries[:5]:
        assert answer_query(graph, query) in {"yes", "no"}


@pytest.mark.parametrize(
    "reasoning_type", [ReasoningType.WITHIN_HOPS_COUNT, ReasoningType.WITHIN_HOPS_LIST]
)
def test_within_hops_queries_are_generated_and_solvable(reasoning_type: ReasoningType) -> None:
    graph = build()
    queries = list(_queries(graph, reasoning_type))
    assert queries
    assert all(query.hops == 2 for query in queries)
    for query in queries[:5]:
        answer = answer_query(graph, query)
        assert answer  # every query on a connected graph yields a non-empty answer


def test_most_common_attribute_within_hops_queries_are_generated_and_solvable() -> None:
    graph = build()
    queries = list(_queries(graph, ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS))
    assert queries
    for query in queries[:5]:
        assert answer_query(graph, query) != "invalid"


def test_closing_a_station_usually_reroutes_rather_than_disconnecting() -> None:
    """The ring backbone should make closures lengthen trips, not sever the map."""
    graph = build(20)
    severed = 0
    for node in graph.nodes:
        updated = apply_edit(
            graph,
            GraphEdit(
                operation=EditOperation.SET,
                target=EditTarget.NODE,
                node_id=node.id,
                attribute="status",
                value="closed",
            ),
        ).graph
        open_neighbors = {
            node_id: [neighbor for neighbor, _ in entries]
            for node_id, entries in adjacency(updated).items()
            if node_id != node.id
        }
        start = next(iter(open_neighbors))
        seen = {start}
        frontier = [start]
        while frontier:
            for neighbor in open_neighbors[frontier.pop()]:
                if neighbor != node.id and neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append(neighbor)
        if len(seen) < len(graph.nodes) - 1:
            severed += 1
    assert severed <= len(graph.nodes) // 4


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"degree": 3}, "even"),
        ({"degree": 20}, "more than"),
        ({"rewire_probability": 1.5}, "rewire_probability"),
        ({"line_count": 9}, "line_count"),
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build(20, **kwargs)  # type: ignore[arg-type]


def test_unknown_distribution_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported graph distribution"):
        generate_dataset(
            counts={"train": 1},
            seed=1,
            node_count_min=12,
            node_count_max=12,
            turns=[1],
            reasoning_types=["shortest_path"],
            distribution="synthetic_metro_v1",
        )


def test_node_count_must_exceed_degree() -> None:
    with pytest.raises(ValueError, match="must exceed graph_degree"):
        generate_dataset(
            counts={"train": 1},
            seed=1,
            node_count_min=4,
            node_count_max=8,
            turns=[1],
            reasoning_types=["shortest_path"],
        )
