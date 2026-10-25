"""Regression tests for the metro_v4 corrected benchmark."""

from __future__ import annotations

import random
from collections import Counter

from graph_modi.data import v4
from graph_modi.schema import DensityBin, EditOperation, EditTarget, ReasoningType


def test_pendant_chains_produce_nonleaf_off_cycle_stations() -> None:
    rng = random.Random(0)
    nonleaf_no = 0
    for i in range(40):
        graph = v4.make_metro_graph_v4(rng, "train", i, 16, DensityBin.MEDIUM)
        degree = v4.V3._degrees(graph)
        for node_id, deg in degree.items():
            if deg < 2:
                continue
            from graph_modi.graph.solvers import answer_query
            from graph_modi.schema import GraphQuery

            if answer_query(graph, GraphQuery(ReasoningType.CYCLE_MEMBERSHIP, node_id)) == "no":
                nonleaf_no += 1
    assert nonleaf_no > 0  # chains create bridge-chain internals


def test_node_order_does_not_park_terminals_at_the_end() -> None:
    rng = random.Random(1)
    parked = 0
    for i in range(30):
        graph = v4.make_metro_graph_v4(rng, "train", i, 18, DensityBin.SPARSE)
        degree = v4.V3._degrees(graph)
        last = [n.id for n in graph.nodes[-3:]]
        if all(degree[i] == 1 for i in last):
            parked += 1
    assert parked < 5  # chance alone would be rare; v3 had 30/30


def test_closed_count_always_in_range() -> None:
    rng = random.Random(2)
    for i in range(20):
        graph = v4.make_metro_graph_v4(rng, "train", i, 14, DensityBin.DENSE)
        closed = sum(n.attributes["status"] == "closed" for n in graph.nodes)
        assert 1 <= closed <= 3


def test_add_edge_carries_attributes() -> None:
    rng = random.Random(3)
    graph = v4.make_metro_graph_v4(rng, "train", 0, 12, DensityBin.MEDIUM)
    for _ in range(20):
        program = v4.sample_edit_program_v4(graph, rng, force_noop=False)
        if any(e.operation is EditOperation.ADD for e in program.edits):
            after = v4.apply_program_v4(graph, program, rng)
            for edge in after.edges:
                assert edge.attributes
                assert "line_color" in edge.attributes
            return
    raise AssertionError("no ADD edit sampled")


def test_del_never_disconnects() -> None:
    rng = random.Random(4)
    graph = v4.make_metro_graph_v4(rng, "train", 0, 14, DensityBin.SPARSE)
    for _ in range(30):
        program = v4.sample_edit_program_v4(graph, rng, force_noop=False)
        after = v4.apply_program_v4(graph, program, rng)
        # connected: BFS from first node reaches all
        adj = {n.id: set() for n in after.nodes}
        for e in after.edges:
            adj[e.source].add(e.target)
            adj[e.target].add(e.source)
        start = next(iter(adj))
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in adj[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        assert len(seen) == len(adj)


def test_noop_rate_near_target() -> None:
    rng = random.Random(5)
    graph = v4.make_metro_graph_v4(rng, "train", 0, 14, DensityBin.MEDIUM)
    noops = 0
    n = 400
    for _ in range(n):
        force = rng.random() < v4.NOOP_RATE
        program = v4.sample_edit_program_v4(graph, rng, force_noop=force)
        noops += program.edits[0].operation is EditOperation.NOOP
    rate = noops / n
    assert 0.12 <= rate <= 0.25, rate


def test_endpoint_open_queries() -> None:
    rng = random.Random(6)
    graph = v4.make_metro_graph_v4(rng, "train", 0, 16, DensityBin.MEDIUM, closed_count=3)
    for task in (ReasoningType.REACHABILITY, ReasoningType.SHORTEST_PATH):
        for query in v4.candidate_queries_v4(graph, task, rng):
            assert v4._is_open(graph, query.source)
            assert v4._is_open(graph, query.target)


def test_cycle_queries_are_degree_two() -> None:
    rng = random.Random(7)
    graph = v4.make_metro_graph_v4(rng, "train", 0, 16, DensityBin.MEDIUM)
    degree = v4.V3._degrees(graph)
    queries = v4.candidate_queries_v4(graph, ReasoningType.CYCLE_MEMBERSHIP, rng)
    assert queries, "chains should produce degree-2 stations"
    for query in queries:
        assert degree[query.source] == 2


def test_constrained_question_states_exemption() -> None:
    rng = random.Random(8)
    graph = v4.make_metro_graph_v4(rng, "train", 0, 12, DensityBin.MEDIUM)
    queries = v4.candidate_queries_v4(graph, ReasoningType.CONSTRAINED_REACHABILITY, rng)
    text = v4.render_question_v4(queries[0], graph)
    assert "exempt" in text.lower()


def test_session_unique_mode_and_metadata() -> None:
    session = v4.generate_session_v4(
        split="validation",
        index=0,
        seed=42,
        node_count=14,
        density=DensityBin.MEDIUM,
        turn_count=4,
        task=ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS,
        sampler=v4.BalancedSampler(),
    )
    assert len(session.turns) == 4
    assert 1 <= sum(n.attributes["status"] == "closed" for n in session.initial_graph.nodes) <= 3
    for turn in session.turns:
        assert "mentions_edit" in turn.complexity
        assert turn.complexity["session_length"] == 4
        assert "ADD EDGE" not in turn.utterance or "transfer" not in turn.utterance.split("EDGE")[0]


def test_utterances_are_paraphrased_not_canonical() -> None:
    rng = random.Random(9)
    graph = v4.make_metro_graph_v4(rng, "train", 0, 12, DensityBin.MEDIUM)
    for family in v4.PARAPHRASE_FAMILIES:
        for _ in range(10):
            program = v4.sample_edit_program_v4(graph, rng, force_noop=False)
            for edit in program.edits:
                text = v4._utterance_v4(edit, graph, family)
                assert not text.startswith("ADD EDGE")
                assert not text.startswith("DEL EDGE")


def test_determinism() -> None:
    a = v4.generate_static_v4(
        split="validation", graph_count=33, tasks_per_graph=None, seed=42,
        sampler=v4.BalancedSampler(), progress=False,
    )
    b = v4.generate_static_v4(
        split="validation", graph_count=33, tasks_per_graph=None, seed=42,
        sampler=v4.BalancedSampler(), progress=False,
    )
    assert [x.tuple_id for x in a] == [x.tuple_id for x in b]
    assert [x.answer for x in a] == [x.answer for x in b]
