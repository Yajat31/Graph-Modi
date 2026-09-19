"""Tests for dataset v2 generation and validation."""

from graph_modi.data.v2 import DISTRIBUTION_V2, generate_dataset_v2
from graph_modi.data.validation import audit_sessions
from graph_modi.graph.edits import parse_edit_program
from graph_modi.schema import EditOperation
from graph_modi.utils.progress import ProgressTracker


def test_v2_smoke_dataset() -> None:
    sessions, static = generate_dataset_v2(
        counts={"train": 2, "test": 2, "ood": 1},
        seed=11,
        static_graph_counts={"train": 2, "test": 1},
        static_tuples_per_graph=3,
        progress=False,
    )
    for split_sessions in sessions.values():
        report = audit_sessions(split_sessions, allow_noop=True)
        assert report["valid"], report["failures"][:3]
    assert sum(len(items) for items in static.values()) >= 3
    assert sessions["train"][0].initial_graph.metadata["distribution"] == DISTRIBUTION_V2


def test_factorial_sessions_cross_scale_and_turns() -> None:
    from collections import Counter

    from graph_modi.data.v2 import generate_factorial_sessions
    from graph_modi.schema import ReasoningType

    sessions = generate_factorial_sessions(
        sessions_per_cell={"validation": 2},
        seed=7,
        dynamic_tasks=[
            ReasoningType.EDGE_EXISTS,
            ReasoningType.REACHABILITY,
            ReasoningType.CYCLE_MEMBERSHIP,
        ],
        progress=False,
    )
    assert set(sessions) == {"validation"}
    assert len(sessions["validation"]) == 2 * 3 * 4  # per_cell × scales × turns
    report = audit_sessions(sessions["validation"], allow_noop=True)
    assert report["valid"], report["failures"][:3]

    cells = Counter()
    dens_by_cell: dict[str, Counter] = {}
    for session in sessions["validation"]:
        turn0 = session.turns[0]
        cell = turn0.complexity["factorial_cell"]
        cells[cell] += 1
        dens_by_cell.setdefault(cell, Counter())[turn0.complexity["density"]] += 1
        assert turn0.complexity["session_length"] == len(session.turns)
        assert turn0.complexity["scale_bin"] in {
            "scale_small",
            "scale_medium",
            "scale_large",
        }
    assert len(cells) == 12
    assert all(count == 2 for count in cells.values())
    all_targets = {
        turn.complexity.get("target_density")
        for s in sessions["validation"]
        for turn in s.turns
    }
    assert all_targets == {"sparse", "medium"}  # 2 sessions/cell → first two bins
    # Across the full grid, realized densities should include sparse (large/small) and medium.
    all_densities = {turn.complexity["density"] for s in sessions["validation"] for turn in s.turns}
    assert "sparse" in all_densities
    assert "medium" in all_densities


def test_exact_uniform_sessions_grid() -> None:
    from collections import Counter

    from graph_modi.data.v2 import generate_exact_uniform_sessions
    from graph_modi.schema import ReasoningType

    sessions = generate_exact_uniform_sessions(
        replicates_per_split={"validation": 1},
        seed=3,
        dynamic_tasks=[
            ReasoningType.EDGE_EXISTS,
            ReasoningType.REACHABILITY,
            ReasoningType.CYCLE_MEMBERSHIP,
        ],
        progress=False,
    )
    assert len(sessions["validation"]) == 26 * 4 * 3 * 3
    report = audit_sessions(sessions["validation"], allow_noop=True)
    assert report["valid"], report["failures"][:3]

    cells = Counter()
    n_by_bin: dict[str, Counter] = {"scale_small": Counter(), "scale_medium": Counter(), "scale_large": Counter()}
    for session in sessions["validation"]:
        types = {turn.query.reasoning_type.value for turn in session.turns}
        assert len(types) == 1
        t0 = session.turns[0]
        cells[t0.complexity["factorial_cell"]] += 1
        n = len(session.initial_graph.nodes)
        n_by_bin[t0.complexity["scale_bin"]][n] += 1
        assert t0.complexity["session_length"] == len(session.turns)
    assert len(cells) == 26 * 4 * 3 * 3
    assert all(v == 1 for v in cells.values())
    assert len(n_by_bin["scale_small"]) == 9
    assert len(n_by_bin["scale_medium"]) == 8
    assert len(n_by_bin["scale_large"]) == 9
    # each exact n appears equally often within a bin (4 lengths × 3 dens × 3 tasks = 36)
    assert set(n_by_bin["scale_small"].values()) == {36}
    assert set(n_by_bin["scale_medium"].values()) == {36}
    assert set(n_by_bin["scale_large"].values()) == {36}


def test_watts_strogatz_edges_get_nontrivial_weights() -> None:
    import random

    from graph_modi.data.v2 import make_graph
    from graph_modi.schema import TopologyFamily

    graph = make_graph(random.Random(5), "train", 0, 24, TopologyFamily.WATTS_STROGATZ)
    weights = {edge.weight for edge in graph.edges}
    # PATH_COST is degenerate (== hop count) if every edge stays at the schema
    # default of 1.0; Watts-Strogatz graphs must get the same weight range as
    # SBM/Erdos-Renyi for weighted-path tasks to be meaningful on this topology.
    assert weights != {1.0}
    assert all(1.0 <= weight <= 5.0 for weight in weights)


def test_generate_static_corpus_no_longer_drops_node_degree() -> None:
    from graph_modi.data.v2 import generate_static_corpus
    from graph_modi.schema import ReasoningType, TopologyFamily

    static = generate_static_corpus(
        graph_counts={"train": 2},
        tuples_per_graph=3,
        seed=9,
        node_count_ranges={"train": (16, 24)},
        topologies=(TopologyFamily.WATTS_STROGATZ,),
        tasks=[ReasoningType.NODE_DEGREE],
        progress=False,
    )
    # NODE_DEGREE previously had no _queries() branch, so every candidate list
    # was empty and generate_static_corpus silently skipped it (0 tuples).
    assert len(static["train"]) == 2 * 3
    assert all(item.metadata["reasoning_type"] == "node_degree" for item in static["train"])


def test_static_exact_uniform_grid_covers_every_cell_including_new_tasks() -> None:
    from collections import Counter

    from graph_modi.data.v2 import generate_static_exact_uniform
    from graph_modi.schema import ReasoningType

    tasks = [
        ReasoningType.EDGE_EXISTS,
        ReasoningType.NODE_DEGREE,
        ReasoningType.CONSTRAINED_REACHABILITY,
        ReasoningType.WITHIN_HOPS_COUNT,
        ReasoningType.WITHIN_HOPS_LIST,
        ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS,
        ReasoningType.PATH_COST,
    ]
    tuples = generate_static_exact_uniform(
        replicates_per_split={"validation": 1},
        seed=5,
        tasks=tasks,
        progress=False,
    )["validation"]
    assert len(tuples) == 26 * 3 * len(tasks)
    seen_tasks = {item.metadata["reasoning_type"] for item in tuples}
    assert seen_tasks == {task.value for task in tasks}
    cells = Counter(item.metadata["factorial_cell"] for item in tuples)
    assert len(cells) == 26 * 3 * len(tasks)
    assert all(count == 1 for count in cells.values())


def test_progress_tracker_emits(monkeypatch) -> None:
    lines: list[str] = []

    def capture(value: str = "", **kwargs) -> None:
        lines.append(value)

    monkeypatch.setattr("builtins.print", capture)
    tracker = ProgressTracker("[generate]", 10, enabled=True)
    tracker.banner(phase="test")
    tracker.tick(5)
    tracker.end()
    assert any("[generate]" in line for line in lines)


def test_edit_program_roundtrip() -> None:
    program = parse_edit_program("SET NODE n0 status closed ; NOOP ; END")
    assert len(program.edits) == 2
    assert program.edits[0].operation is EditOperation.SET
    assert program.edits[1].operation is EditOperation.NOOP
