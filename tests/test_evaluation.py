import pytest

from graph_modi.data.multiturn import generate_dataset
from graph_modi.evaluation.metrics import answers_match, set_f1
from graph_modi.evaluation.runner import evaluate_sessions
from graph_modi.models.base import SymbolicMockBackend


class RecordingBackend(SymbolicMockBackend):
    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[tuple[str, str]] = []

    def answer(self, model_input):
        self.prompts.append((model_input.condition, model_input.question))
        return super().answer(model_input)


def test_paired_conditions_expose_stale_graph_control() -> None:
    sessions = generate_dataset(
        counts={"test": 4},
        seed=22,
        node_count_min=8,
        node_count_max=10,
        turns=[2],
        reasoning_types=["shortest_path"],
    )["test"]
    report = evaluate_sessions(
        sessions,
        SymbolicMockBackend(),
        conditions=[
            "question_only",
            "frozen_graph_history",
            "oracle_updated_graph",
            "predicted_updated_graph",
            "serialized_current_graph",
            "tool_solver",
        ],
    )
    summary = report["summary"]
    assert summary["question_only"]["answer_accuracy"] == 0
    assert summary["oracle_updated_graph"]["answer_accuracy"] == 1
    assert summary["serialized_current_graph"]["answer_accuracy"] == 1
    assert summary["tool_solver"]["answer_accuracy"] == 1
    assert summary["frozen_graph_history"]["stale_rate"] == 1
    assert summary["predicted_updated_graph"]["execution_equivalent_edit_accuracy"] == 1


def test_new_baseline_conditions_exist() -> None:
    from graph_modi.evaluation.runner import CONDITIONS

    for name in ("soft_prompt", "structure_only", "modify_and_print", "majority_prior"):
        assert name in CONDITIONS


def test_answers_match_applies_numeric_tolerance_for_path_cost() -> None:
    assert answers_match("12.0", "12", reasoning_type="path_cost")
    assert answers_match("12.5", "12.49", reasoning_type="path_cost")
    assert not answers_match("12.5", "14", reasoning_type="path_cost")


def test_answers_match_is_exact_for_boolean_tasks() -> None:
    assert answers_match("yes", "yes", reasoning_type="reachability")
    assert not answers_match("yes", "no", reasoning_type="reachability")
    # No numeric tolerance should leak into non-numeric task types.
    assert not answers_match("2", "2.4", reasoning_type="reachability")


def test_answers_match_compares_lists_as_sets() -> None:
    assert answers_match(
        "Belmont, Crount", "Crount, Belmont", reasoning_type="within_hops_list"
    )
    assert not answers_match(
        "Belmont, Crount", "Belmont", reasoning_type="within_hops_list"
    )
    assert answers_match("none", "none", reasoning_type="within_hops_list")


def test_set_f1_partial_credit() -> None:
    assert set_f1("Belmont, Crount", "Crount, Belmont") == 1.0
    assert set_f1("none", "none") == 1.0
    assert set_f1("Belmont, Crount, Ashford", "Belmont") == pytest.approx(0.5)
    assert set_f1("Belmont", "Crount") == 0.0


def test_oracle_graph_prompt_does_not_include_revision_history() -> None:
    session = generate_dataset(
        counts={"test": 1},
        seed=88,
        node_count_min=8,
        node_count_max=8,
        turns=[2],
        reasoning_types=["shortest_path"],
    )["test"][0]
    backend = RecordingBackend()
    evaluate_sessions([session], backend, conditions=["oracle_updated_graph"])
    revisions = {turn.utterance for turn in session.turns}
    assert all(revision not in prompt for _, prompt in backend.prompts for revision in revisions)
