from __future__ import annotations

import random

import pytest

from graph_modi.data import v3
from graph_modi.data.validation import audit_sessions
from graph_modi.graph.solvers import answer_query, render_question
from graph_modi.schema import DensityBin, GraphQuery, ReasoningType


def _graph(density: DensityBin = DensityBin.SPARSE, n: int = 16, seed: int = 1):
    return v3.make_metro_graph(random.Random(seed), "test", 0, n, density)


@pytest.mark.parametrize("density", list(v3.DENSITIES))
@pytest.mark.parametrize("n", [12, 17, 22])
def test_graph_is_connected_compact_and_semantic(density: DensityBin, n: int) -> None:
    graph = _graph(density, n)
    assert len(graph.nodes) == n
    assert graph.metadata["diameter"] <= v3.MAX_DIAMETER[density]
    assert len({node.label for node in graph.nodes}) == n
    attrs = graph.nodes[0].attributes
    for key in ("disabled_access", "has_rail", "architecture", "cleanliness", "music", "size", "line", "status"):
        assert key in attrs
    assert {"line_color", "line_stroke", "has_aircon", "built"} <= set(graph.edges[0].attributes)
    degrees = {node.id: 0 for node in graph.nodes}
    for edge in graph.edges:
        degrees[edge.source] += 1
        degrees[edge.target] += 1
    assert sum(1 for d in degrees.values() if d == 1) >= 2  # terminal stations exist


def test_status_is_hidden_from_wfi_text_but_kept_in_graph() -> None:
    graph = _graph()
    query = GraphQuery(ReasoningType.REACHABILITY, graph.nodes[0].id, graph.nodes[1].id)
    text = render_question(query, graph)
    assert "status" not in text
    assert "status" in graph.nodes[0].attributes


def test_facts_questions_hide_the_queried_attribute() -> None:
    graph = _graph()
    node = graph.nodes[0]
    lookup = GraphQuery(ReasoningType.ATTRIBUTE_LOOKUP, node.id, attribute="architecture")
    text = render_question(lookup, graph)
    assert "architecture=" not in text.split("\n")[0]
    assert "cleanliness=" in text.split("\n")[0]
    assert answer_query(graph, lookup) == node.attributes["architecture"]
    check = GraphQuery(ReasoningType.ATTRIBUTE_CHECK, node.id, attribute="has_rail", value=node.attributes["has_rail"])
    assert answer_query(graph, check) == "yes"
    assert "has_rail=" not in render_question(check, graph).split("\n")[0]


def test_static_labels_are_balanced_and_deterministic() -> None:
    def make():
        return v3.generate_static_v3(
            split="validation", graph_count=33 * 6, tasks_per_graph=None, seed=7,
            sampler=v3.BalancedSampler(), progress=False,
        )

    first, second = make(), make()
    assert [t.answer for t in first] == [t.answer for t in second]
    audit = v3.label_audit(first)
    for task in v3.YES_NO_TASKS:
        assert abs(audit[task.value]["majority_accuracy"] - 0.5) <= 0.03
    for task in v3.STATIC_TASKS:
        if task not in v3.YES_NO_TASKS:
            assert audit[task.value]["majority_accuracy"] <= 0.30
    for item in first:
        assert item.answer == answer_query(item.graph, item.query)


def test_sessions_are_valid_balanced_and_use_the_exact_grid() -> None:
    sessions = v3.generate_sessions_v3(
        replicates_per_split={"validation": 1}, seed=3, progress=False
    )["validation"]
    assert len(sessions) == len(v3.NODE_COUNTS) * 3 * len(v3.SESSION_LENGTHS) * len(v3.DYNAMIC_TASKS)
    assert audit_sessions(sessions, allow_noop=True)["valid"]
    audit = v3.session_label_audit(sessions)
    for task in v3.YES_NO_TASKS & set(v3.DYNAMIC_TASKS):
        assert abs(audit[task.value]["majority_accuracy"] - 0.5) <= 0.03
    assert {len(s.initial_graph.nodes) for s in sessions} == set(v3.NODE_COUNTS)


def test_counterfactual_pairs_share_text_but_differ_in_graph_and_answer() -> None:
    pairs = v3.generate_counterfactual_pairs_v3(
        graph_count=20, pairs_per_graph=2, seed=5, sampler=v3.BalancedSampler(), progress=False
    )
    by_id: dict[str, list] = {}
    for item in pairs:
        by_id.setdefault(item.tuple_id.rsplit("-", 1)[0], []).append(item)
    assert by_id
    for group in by_id.values():
        before, after = sorted(group, key=lambda t: t.metadata["counterfactual"])[::-1]
        assert before.query.question == after.query.question
        assert before.graph.graph_id != after.graph.graph_id
        assert before.answer != after.answer


def test_no_text_leak_beyond_question_prior() -> None:
    train = v3.generate_static_v3(
        split="train", graph_count=600, tasks_per_graph=9, seed=11, sampler=v3.BalancedSampler(), progress=False
    )
    evaluate = v3.generate_static_v3(
        split="validation", graph_count=33 * 4, tasks_per_graph=None, seed=11,
        sampler=v3.BalancedSampler(), progress=False,
    )
    report = v3.text_leak_audit(train, evaluate)
    for task in v3.YES_NO_TASKS:
        assert report[task.value]["with_node_text_rule_accuracy"] <= 0.60


def test_csv_prompt_is_clegr_format_and_includes_state() -> None:
    from graph_modi.graph.serialization import csv_graph_prompt, plain_question

    graph = _graph(DensityBin.MEDIUM, 14)
    query = GraphQuery(ReasoningType.REACHABILITY, graph.nodes[0].id, graph.nodes[1].id)
    question = render_question(query, graph)
    prompt = csv_graph_prompt(graph, question, updates=["X is closed now."])
    assert prompt.startswith("--- Nodes ---")
    assert '"id","name","disabled_access"' in prompt and '"source_id","target_id"' in prompt
    assert prompt.count("\n") > len(graph.nodes) + len(graph.edges)
    assert '"open"' in prompt or '"closed"' in prompt  # state stays visible in the text track
    assert "Update 1: X is closed now." in prompt
    assert plain_question(question).split(" Answer")[0] in prompt  # question sentence, instruction stripped
    assert "status=" not in prompt  # W(f_i) node text is not repeated in the CSV track


def test_csv_prompt_uses_clegr_answer_suffixes() -> None:
    from graph_modi.graph.serialization import csv_graph_prompt

    graph = _graph(DensityBin.MEDIUM, 14)
    a, b = graph.nodes[0].id, graph.nodes[1].id
    cases = {
        ReasoningType.REACHABILITY: GraphQuery(ReasoningType.REACHABILITY, a, b),
        ReasoningType.CYCLE_MEMBERSHIP: GraphQuery(ReasoningType.CYCLE_MEMBERSHIP, a),
        ReasoningType.NODE_DEGREE: GraphQuery(ReasoningType.NODE_DEGREE, a),
        ReasoningType.ATTRIBUTE_LOOKUP: GraphQuery(ReasoningType.ATTRIBUTE_LOOKUP, a, attribute="size"),
    }
    expected = {
        ReasoningType.REACHABILITY: "Answer with 'True' or 'False':\n\nAnswer:",
        ReasoningType.CYCLE_MEMBERSHIP: "Answer with 'True' if it is in a cycle, otherwise 'False':\n\nAnswer:",
        ReasoningType.NODE_DEGREE: "Answer with a number:\n\nAnswer:",
        ReasoningType.ATTRIBUTE_LOOKUP: "Answer directly:",
    }
    for task, query in cases.items():
        prompt = csv_graph_prompt(graph, render_question(query, graph))
        assert prompt.endswith(expected[task]), task
        assert "The question is:\n" in prompt
        assert "Answer yes or no" not in prompt and "Answer with a number." not in prompt


def test_csv_track_targets_are_true_false_but_still_score_as_yes_no() -> None:
    from graph_modi.cli import _static_examples
    from graph_modi.evaluation.metrics import answers_match

    static = v3.generate_static_v3(
        split="validation", graph_count=33, tasks_per_graph=None, seed=2,
        sampler=v3.BalancedSampler(), progress=False,
    )
    plain = _static_examples(static, "wfi")
    csv = _static_examples(static, "csv")
    for before, after in zip(plain, csv, strict=True):
        if before.answer in ("yes", "no"):
            assert after.answer == ("True" if before.answer == "yes" else "False")
            assert answers_match(before.answer, after.answer, reasoning_type=before.metadata["reasoning_type"])
        else:
            assert after.answer == before.answer


def test_graph_token_position_prefix_places_tokens_after_bos() -> None:
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from graph_modi.models.soft_prompt import SoftPromptGLM
    from graph_modi.models.tea_glm import TEAGLM

    embedding = torch.nn.Embedding(20, 4)
    prefix = torch.full((3, 4), 7.0)
    for cls, config in (
        (SoftPromptGLM, SimpleNamespace(add_bos_token=True)),
        (TEAGLM, SimpleNamespace(add_bos_token=True)),
    ):
        stub = SimpleNamespace(
            device="cpu", config=config, tokenizer=SimpleNamespace(bos_token_id=1), graph_token_position="prefix"
        )
        parts = cls._assemble_prompt(stub, [1, 5, 6], prefix, embedding)
        assert [p.shape[0] for p in parts] == [1, 3, 2]  # BOS, graph tokens, rest of prompt
        stub.graph_token_position = "before_answer"
        parts = cls._assemble_prompt(stub, [1, 5, 6], prefix, embedding)
        assert [p.shape[0] for p in parts] == [3, 3]  # prompt, graph tokens


def test_edit_parser_resolves_multi_word_station_names() -> None:
    from graph_modi.graph.edits import parse_edit_program
    from graph_modi.schema import EditOperation, EditTarget

    names = {"vale hill": "n3", "quill cross": "n7", "elm": "n1"}
    program = parse_edit_program(
        "SET NODE Vale Hill status closed ; DEL EDGE Quill Cross Vale Hill transfer ; ADD EDGE Elm Vale Hill transfer ; END",
        names,
    )
    node_edit, delete, add = program.edits
    assert (node_edit.node_id, node_edit.attribute, node_edit.value) == ("n3", "status", "closed")
    assert (delete.operation, delete.source, delete.destination, delete.relation) == (
        EditOperation.DEL, "n7", "n3", "transfer",
    )
    assert (add.source, add.destination) == ("n1", "n3")
    # plain single-token IDs and unknown names keep the old one-word behaviour
    plain = parse_edit_program("SET NODE n3 status closed ; DEL EDGE n7 n3 track ; END", None).edits
    assert plain[0].node_id == "n3" and plain[1].source == "n7" and plain[1].destination == "n3"
    assert parse_edit_program("SET NODE Nowhere status open ; END", names).edits[0].node_id == "Nowhere"
