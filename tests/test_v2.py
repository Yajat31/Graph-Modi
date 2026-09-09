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


def test_target_density_hits_bin() -> None:
    import random

    from graph_modi.data.v2 import make_graph, _graph_density
    from graph_modi.schema import DensityBin, TopologyFamily

    rng = random.Random(0)
    for target in DensityBin:
        graph = make_graph(
            rng,
            "test",
            0,
            40,
            TopologyFamily.WATTS_STROGATZ,
            target_density=target,
        )
        assert _graph_density(graph) is target, (target, _graph_density(graph), len(graph.edges))


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
