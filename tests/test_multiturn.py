from graph_modi.data.multiturn import generate_dataset
from graph_modi.data.validation import audit_sessions
from graph_modi.models.base import SymbolicMockBackend
from graph_modi.pipeline.multiturn import materialize_states, run_session


def tiny_dataset():
    return generate_dataset(
        counts={"train": 4, "validation": 2, "test": 2},
        seed=7,
        node_count_min=8,
        node_count_max=10,
        turns=[3],
        reasoning_types=["shortest_path", "filtered_neighbor_count"],
    )


def test_generated_splits_are_valid_and_disjoint() -> None:
    dataset = tiny_dataset()
    for sessions in dataset.values():
        report = audit_sessions(sessions)
        assert report["valid"], report["failures"]
        assert report["counterfactual_pairs"] == report["turns"]


def test_turn_three_operates_on_graph_two() -> None:
    session = tiny_dataset()["test"][0]
    states = materialize_states(session)
    assert states[1].after == states[2].before
    assert states[0].before == session.initial_graph
    assert states[2].after != session.initial_graph


def test_reencode_occurs_after_every_accepted_edit() -> None:
    session = tiny_dataset()["test"][0]
    backend = SymbolicMockBackend()
    outputs = run_session(session, backend)
    assert len(outputs) == 3
    assert outputs[-1].encode_calls == 4  # initial graph plus three updates
    assert [output.predicted_answer for output in outputs] == [
        turn.gold_answer for turn in session.turns
    ]
