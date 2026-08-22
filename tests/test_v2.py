"""Tests for dataset v2 generation and validation."""

from graph_modi.data.v2 import DISTRIBUTION_V2, generate_dataset_v2
from graph_modi.data.validation import audit_sessions
from graph_modi.graph.edits import parse_edit_program
from graph_modi.graph.executor import apply_edit_program
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
