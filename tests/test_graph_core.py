from dataclasses import replace

from graph_modi.graph.edits import execution_equivalent, parse_edit
from graph_modi.graph.executor import apply_edit, graph_fingerprint
from graph_modi.graph.solvers import answer_query
from graph_modi.schema import (
    AttributedGraph,
    Edge,
    GraphQuery,
    Node,
    ReasoningType,
)


def sample_graph() -> AttributedGraph:
    return AttributedGraph(
        graph_id="g",
        nodes=(
            Node("a", "Ashford", {"status": "open", "accessible": True}),
            Node("b", "Belmont", {"status": "open", "accessible": False}),
            Node("c", "Crount", {"status": "open", "accessible": True}),
        ),
        edges=(Edge("a", "b"), Edge("b", "c")),
    )


def test_parse_apply_and_fingerprint_are_execution_equivalent() -> None:
    graph = sample_graph()
    by_name = parse_edit(
        "SET NODE Belmont status closed",
        {"belmont": "b"},
    )
    by_id = parse_edit("SET NODE b status closed")
    assert execution_equivalent(by_name, by_id)

    result = apply_edit(graph, by_name)
    assert result.applied
    assert graph_fingerprint(result.graph) != graph_fingerprint(graph)
    assert graph.node_map()["b"].attributes["status"] == "open"


def test_invalid_edit_does_not_mutate_graph() -> None:
    graph = sample_graph()
    result = apply_edit(graph, parse_edit("DEL EDGE a c"))
    assert not result.applied
    assert graph_fingerprint(result.graph) == graph_fingerprint(graph)


def test_symbolic_solvers_follow_open_state() -> None:
    graph = sample_graph()
    query = GraphQuery(ReasoningType.SHORTEST_PATH, "a", "c")
    assert answer_query(graph, query) == "3"
    updated = apply_edit(graph, parse_edit("SET NODE b status closed")).graph
    assert answer_query(updated, query) == "unreachable"


def test_node_degree_solver() -> None:
    graph = sample_graph()
    assert answer_query(graph, GraphQuery(ReasoningType.NODE_DEGREE, "b")) == "2"
    assert answer_query(graph, GraphQuery(ReasoningType.NODE_DEGREE, "a")) == "1"


def test_constrained_reachability_avoids_the_excluded_class() -> None:
    graph = sample_graph()
    query = GraphQuery(
        ReasoningType.CONSTRAINED_REACHABILITY, "a", "c", attribute="accessible", value=False
    )
    # The only a-c path runs through b, which is inaccessible.
    assert answer_query(graph, query) == "no"
    # b's accessible is False, not True, so avoiding "accessible == True" leaves it usable.
    assert answer_query(graph, replace(query, attribute="accessible", value=True)) == "yes"
    unconstrained = GraphQuery(ReasoningType.REACHABILITY, "a", "c")
    assert answer_query(graph, unconstrained) == "yes"


def test_within_hops_count_and_list() -> None:
    graph = sample_graph()
    count_query = GraphQuery(ReasoningType.WITHIN_HOPS_COUNT, "a", hops=2)
    assert answer_query(graph, count_query) == "2"
    filtered_count = replace(count_query, attribute="accessible", value=True)
    assert answer_query(graph, filtered_count) == "1"

    list_query = GraphQuery(ReasoningType.WITHIN_HOPS_LIST, "a", hops=2)
    assert answer_query(graph, list_query) == "Belmont, Crount"
    filtered_list = replace(list_query, attribute="accessible", value=True)
    assert answer_query(graph, filtered_list) == "Crount"


def test_most_common_attribute_within_hops() -> None:
    graph = sample_graph()
    query = GraphQuery(
        ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS, "a", attribute="accessible", hops=2
    )
    assert answer_query(graph, query) == "True"
