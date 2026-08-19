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
