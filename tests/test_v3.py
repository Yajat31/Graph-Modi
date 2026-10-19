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


def test_graph_tokens_follow_the_prompt() -> None:
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from graph_modi.models.soft_prompt import SoftPromptGLM
    from graph_modi.models.tea_glm import TEAGLM

    embedding = torch.nn.Embedding(20, 4)
    prefix = torch.full((3, 4), 7.0)
    for cls in (SoftPromptGLM, TEAGLM):
        stub = SimpleNamespace(device="cpu")
        parts = cls._assemble_prompt(stub, [1, 5, 6], prefix, embedding)
        assert [p.shape[0] for p in parts] == [3, 3]  # prompt, graph tokens


def test_unfrozen_gnn_receives_gradients_through_graph_encoding() -> None:
    torch = pytest.importorskip("torch")
    from graph_modi.models.tea_glm import (
        HOPS_FEATURE_DIM,
        REASONING_TYPE_FEATURE_DIM,
        TEAGLM,
        DeterministicNodeTensorizer,
        GraphPrefixProjector,
        GraphSAGEConfig,
        GraphSAGEEncoder,
        NodeTensorizerConfig,
        PrefixProjectorConfig,
    )
    from graph_modi.training import ProjectorTrainingConfig

    class StubLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(10, 8)

        def get_input_embeddings(self):
            return self.embed

    gnn_config = GraphSAGEConfig(input_dim=16, hidden_dim=8, output_dim=8)
    model = TEAGLM(
        gnn=GraphSAGEEncoder(gnn_config),
        projector=GraphPrefixProjector(
            PrefixProjectorConfig(
                graph_dim=3 * 8 + HOPS_FEATURE_DIM + REASONING_TYPE_FEATURE_DIM,
                lm_hidden_dim=8,
                prefix_tokens=2,
            )
        ),
        language_model=StubLM(),
        tokenizer=None,
        tensorizer=DeterministicNodeTensorizer(NodeTensorizerConfig(feature_dim=16)),
    )
    graph = _graph()
    assert not ProjectorTrainingConfig(output_dir="unused").train_gnn
    assert not model.encode_graphs([graph]).requires_grad  # TEA default: frozen GNN

    model.set_gnn_frozen(False)  # what train_projector does when train_gnn=True
    model.encode_graphs([graph]).sum().backward()
    assert all(p.grad is not None for p in model.gnn.parameters())


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
