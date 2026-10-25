"""Regression tests for the 2026-10 audit fixes (parser, statistics, trainer, ablations)."""

from __future__ import annotations

import random

import pytest

from graph_modi.data import v3
from graph_modi.data.multiturn import generate_dataset
from graph_modi.evaluation.metrics import (
    aggregate_rows,
    cluster_bootstrap_ci,
    normalize_answer,
    paired_cluster_bootstrap,
    wilson_interval,
)
from graph_modi.evaluation.runner import CONDITIONS, evaluate_sessions
from graph_modi.models.base import SymbolicMockBackend
from graph_modi.schema import DensityBin
from graph_modi.training import ProjectorTrainingConfig, lr_multiplier

# --------------------------------------------------------------------------- answer parser


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("No, there are 2 routes.", "no"),  # trailing number must not override the answer
        ("Yes, 3 stations are open", "yes"),
        ("False, because 7 hops", "no"),
        ("True", "yes"),
        ("3", "3"),
        ("3 stations", "3"),
        ("It is 4", "4"),  # no leading answer: first number
        ("The answer is no", "no"),
        ("no path", "unreachable"),
        ("No path exists between them", "unreachable"),
        ("unreachable", "unreachable"),
        ("4.0", "4.0"),
        ("-2", "-2"),
        ("modern", "modern"),
        ("Yes.", "yes"),
        (None, None),
    ],
)
def test_normalize_answer_uses_the_leading_token(text: str | None, expected: str | None) -> None:
    assert normalize_answer(text) == expected


def test_trained_model_outputs_are_parsed_as_before() -> None:
    # Trained models emit a bare answer then EOS; the leading-token rule must not change them.
    for bare in ("yes", "no", "0", "12", "unreachable", "jazz", "dirty"):
        assert normalize_answer(bare) == bare


# ------------------------------------------------------------------------ lr schedule / config


def test_lr_multiplier_constant_warmup_and_decay() -> None:
    assert lr_multiplier(0, 100) == 1.0
    assert lr_multiplier(0, 100, warmup_steps=10) == pytest.approx(0.1)
    assert lr_multiplier(9, 100, warmup_steps=10) == pytest.approx(1.0)
    assert lr_multiplier(10, 100, schedule="cosine", warmup_steps=10) == pytest.approx(1.0)
    assert lr_multiplier(55, 100, schedule="cosine", warmup_steps=10) == pytest.approx(0.5)
    assert lr_multiplier(100, 100, schedule="cosine", warmup_steps=10) == pytest.approx(0.0)
    assert lr_multiplier(100, 100, schedule="linear", min_lr_ratio=0.1) == pytest.approx(0.1)
    values = [
        lr_multiplier(step, 100, schedule="cosine", warmup_steps=10) for step in range(10, 101)
    ]
    assert values == sorted(values, reverse=True)


def test_training_config_rejects_bad_schedule_settings() -> None:
    with pytest.raises(ValueError):
        ProjectorTrainingConfig(output_dir="x", lr_schedule="step")
    with pytest.raises(ValueError):
        ProjectorTrainingConfig(output_dir="x", min_lr_ratio=1.5)
    with pytest.raises(ValueError):
        ProjectorTrainingConfig(output_dir="x", gnn_learning_rate=0.0)
    assert ProjectorTrainingConfig(output_dir="x").metadata()["lr_schedule"] == "constant"


# ------------------------------------------------------------------------------- statistics


def _rows(per_session: dict[str, list[bool]], condition: str = "c") -> list[dict]:
    return [
        {"session_id": session, "turn_index": index, "condition": condition, "answer_correct": ok}
        for session, outcomes in per_session.items()
        for index, ok in enumerate(outcomes)
    ]


def test_cluster_bootstrap_is_wider_than_wilson_when_sessions_are_correlated() -> None:
    # Whole sessions are either all right or all wrong: turns are far from independent.
    rng = random.Random(0)
    per_session = {f"s{i}": [rng.random() < 0.5] * 6 for i in range(60)}
    rows = _rows(per_session)
    successes = sum(row["answer_correct"] for row in rows)
    wilson = wilson_interval(successes, len(rows))
    low, high = cluster_bootstrap_ci(rows, resamples=500)
    assert (high - low) > 1.5 * (wilson[1] - wilson[0])


def test_cluster_bootstrap_degenerate_cases() -> None:
    assert cluster_bootstrap_ci([]) == (0.0, 0.0)
    assert cluster_bootstrap_ci(_rows({"a": [True, True], "b": [True]})) == (1.0, 1.0)


def test_aggregate_rows_reports_session_bootstrap_ci() -> None:
    rows = [
        {
            **row,
            "stale": False,
            "state_exact": True,
            "edit_correct": None,
            "latency_seconds": 0.1,
            "input_tokens": 5,
        }
        for row in _rows({f"s{i}": [i % 2 == 0, True] for i in range(10)})
    ]
    report = aggregate_rows(rows)["c"]
    assert report["answer_accuracy_ci95_session_bootstrap"][0] <= report["answer_accuracy"]
    assert report["answer_accuracy_ci95_session_bootstrap"][1] >= report["answer_accuracy"]
    assert report["mean_graph_prefix_tokens"] == 0.0


def test_paired_bootstrap_detects_a_real_gap_and_not_a_null_one() -> None:
    rng = random.Random(1)
    base = {f"s{i}": [rng.random() < 0.5 for _ in range(4)] for i in range(80)}
    better = {
        session: [True if rng.random() < 0.3 else ok for ok in outcomes]
        for session, outcomes in base.items()
    }
    gap = paired_cluster_bootstrap(_rows(better), _rows(base), resamples=1000)
    assert gap["mean_delta"] > 0.05
    assert gap["ci95"][0] > 0 and gap["p_value"] < 0.01
    null = paired_cluster_bootstrap(_rows(base), _rows(base), resamples=1000)
    assert null["mean_delta"] == 0.0 and null["p_value"] == 1.0
    assert null["wins"] == null["loses"] == 0


# ----------------------------------------------------------------------------- runner rows


def _sessions():
    return generate_dataset(
        counts={"test": 4},
        seed=22,
        node_count_min=8,
        node_count_max=10,
        turns=[2],
        reasoning_types=["shortest_path"],
    )["test"]


def test_rows_keep_the_raw_generation_and_count_prefix_tokens() -> None:
    class Prose(SymbolicMockBackend):
        def answer(self, model_input):
            return f"No, there are {super().answer(model_input)} routes."

        def prefix_token_count(self, model_input) -> int:
            return 10

    report = evaluate_sessions(_sessions(), Prose(), conditions=["oracle_updated_graph"])
    row = report["rows"][0]
    assert row["raw_answer"].startswith("No, there are")
    assert row["predicted_answer"] == "no"  # parsed from the lead, not the number inside
    assert row["graph_prefix_tokens"] == 10
    assert row["latency_batch_size"] >= 1
    assert report["summary"]["oracle_updated_graph"]["mean_graph_prefix_tokens"] == 10


def test_stale_answer_and_majority_answer_are_distinct_baselines() -> None:
    assert "stale_answer" in CONDITIONS and "majority_answer" in CONDITIONS
    sessions = _sessions()
    prior = {"shortest_path": "4"}
    report = evaluate_sessions(
        sessions,
        SymbolicMockBackend(),
        conditions=["stale_answer", "majority_answer", "tool_solver"],
        answer_prior=prior,
    )
    summary = report["summary"]
    assert summary["stale_answer"]["stale_rate"] == 1.0
    assert summary["tool_solver"]["answer_accuracy"] == 1.0
    gold = [row["gold_answer"] for row in report["rows"] if row["condition"] == "majority_answer"]
    assert summary["majority_answer"]["answer_accuracy"] == pytest.approx(
        sum(answer == "4" for answer in gold) / len(gold)
    )
    with pytest.raises(ValueError, match="answer_prior"):
        evaluate_sessions(sessions, SymbolicMockBackend(), conditions=["majority_answer"])


def test_soft_prompt_no_history_prompt_is_the_bare_question() -> None:
    assert "soft_prompt_no_history" in CONDITIONS
    sessions = _sessions()
    report = evaluate_sessions(
        sessions, SymbolicMockBackend(), conditions=["soft_prompt", "soft_prompt_no_history"]
    )
    tokens = {
        condition: report["summary"][condition]["mean_input_tokens"]
        for condition in ("soft_prompt", "soft_prompt_no_history")
    }
    assert tokens["soft_prompt_no_history"] < tokens["soft_prompt"]  # history text is absent


def test_legacy_majority_prior_name_still_runs_as_stale_answer() -> None:
    report = evaluate_sessions(_sessions(), SymbolicMockBackend(), conditions=["majority_prior"])
    assert set(report["summary"]) == {"stale_answer"}


# ------------------------------------------------------------------ torch: ablation, trainer


def _tiny_model(*, gnn_dropout: float = 0.0, seed: int = 0):
    """A CPU TEAGLM around a 2-layer random GPT-2 and a word-level tokenizer."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
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

    words = ["<pad>", "<bos>", "<eos>", "<unk>", "question", ":", "answer", "yes", "no", "is", "?"]
    tok = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({w: i for i, w in enumerate(words)}, "<unk>")
    )
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    torch.manual_seed(seed)
    lm = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=len(words),
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    return TEAGLM(
        gnn=GraphSAGEEncoder(
            GraphSAGEConfig(
                input_dim=16, hidden_dim=8, output_dim=8, num_layers=2, dropout=gnn_dropout
            )
        ),
        projector=GraphPrefixProjector(
            PrefixProjectorConfig(
                graph_dim=3 * 8 + HOPS_FEATURE_DIM + REASONING_TYPE_FEATURE_DIM,
                lm_hidden_dim=16,
                prefix_tokens=2,
                hidden_dim=16,
            )
        ),
        language_model=lm,
        tokenizer=tokenizer,
        tensorizer=DeterministicNodeTensorizer(NodeTensorizerConfig(feature_dim=16)),
    )


def _examples(count: int = 6):
    from graph_modi.training import TrainingExample

    out = []
    for index in range(count):
        graph = v3.make_metro_graph(random.Random(index), "train", index, 12, DensityBin.SPARSE)
        out.append(
            TrainingExample(
                graph=graph,
                prompt="question : is yes ?",
                answer="yes" if index % 2 else "no",
                example_id=f"e{index}",
                split="train",
                metadata={
                    "reasoning_type": "edge_exists",
                    "source_id": graph.nodes[0].id,
                    "target_id": graph.nodes[1].id,
                },
            )
        )
    return out


def test_zero_gnn_ablation_hides_the_graph_but_keeps_the_task() -> None:
    torch = pytest.importorskip("torch")
    model = _tiny_model()
    graphs = [e.graph for e in _examples(2)]
    kwargs = {"reasoning_types": ["edge_exists"], "hops": [None]}

    def vector(graph, ablation, task="edge_exists"):
        model.graph_ablation = ablation
        return model.encode_graphs(
            [graph],
            source_ids=[graph.nodes[0].id],
            target_ids=[graph.nodes[1].id],
            hops=[None],
            reasoning_types=[task],
        )

    assert not torch.equal(vector(graphs[0], "none"), vector(graphs[1], "none"))
    assert torch.equal(vector(graphs[0], "zero_gnn"), vector(graphs[1], "zero_gnn"))
    assert not torch.equal(
        vector(graphs[0], "zero_gnn"), vector(graphs[0], "zero_gnn", "reachability")
    )
    del kwargs


def test_joint_training_uses_two_groups_and_the_schedule(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from graph_modi.training import train_projector

    model = _tiny_model()
    before = {k: v.clone() for k, v in model.gnn.state_dict().items()}
    config = ProjectorTrainingConfig(
        output_dir=tmp_path,
        epochs=2,
        batch_size=2,
        learning_rate=1e-2,
        gnn_learning_rate=1e-3,
        lr_schedule="cosine",
        warmup_steps=2,
        train_gnn=True,
        eval_every_steps=3,
        save_every_steps=100,
    )
    examples = _examples()
    result = train_projector(model, examples, config, val_examples=examples[:2])
    assert any(not torch.equal(before[k], v) for k, v in model.gnn.state_dict().items())
    history = (tmp_path / "val_history.json").read_text()
    assert "val_loss" in history
    assert (tmp_path / "checkpoint-best" / "gnn.pt").exists()
    assert not (tmp_path / "checkpoint-best" / "optimizer.pt").exists()
    assert result.final_checkpoint.exists()


def test_graphtoken_resume_matches_an_uninterrupted_run(tmp_path) -> None:
    """Resuming a jointly trained GNN used to fail the GNN-hash check, and a resumed run
    reshuffled the data from scratch; the resumed weights must equal the uninterrupted ones."""
    torch = pytest.importorskip("torch")
    from graph_modi.training import train_projector

    examples = _examples(8)

    def config(out, **extra):
        return ProjectorTrainingConfig(
            output_dir=out,
            epochs=3,
            batch_size=2,
            learning_rate=1e-2,
            gnn_learning_rate=1e-3,
            lr_schedule="linear",
            warmup_steps=1,
            train_gnn=True,
            save_every_steps=4,
            **extra,
        )

    full = _tiny_model()
    train_projector(full, examples, config(tmp_path / "full"))
    # steps/epoch = 4, so checkpoint-00000004 is the end of epoch 1.
    resumed = _tiny_model()  # fresh random GNN: its hash differs from the checkpoint's
    train_projector(
        resumed,
        examples,
        config(tmp_path / "resumed", resume_from=tmp_path / "full" / "checkpoint-00000004"),
    )
    for name, value in full.projector.state_dict().items():
        assert torch.allclose(value, resumed.projector.state_dict()[name], atol=1e-6), name
    for name, value in full.gnn.state_dict().items():
        assert torch.allclose(value, resumed.gnn.state_dict()[name], atol=1e-6), name


def test_warm_start_loads_gnn_and_projector(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from graph_modi.training import train_projector

    source = _tiny_model(seed=1)
    train_projector(
        source,
        _examples(),
        ProjectorTrainingConfig(output_dir=tmp_path / "a", epochs=1, batch_size=2, train_gnn=True),
    )
    target = _tiny_model(seed=2)
    train_projector(
        target,
        _examples(),
        ProjectorTrainingConfig(
            output_dir=tmp_path / "b",
            epochs=1,
            batch_size=2,
            learning_rate=1e-12,
            train_gnn=True,
            init_checkpoint=tmp_path / "a" / "checkpoint-final",
        ),
    )
    for name, value in source.gnn.state_dict().items():  # lr ~ 0: weights stay at the warm start
        assert torch.allclose(value, target.gnn.state_dict()[name], atol=1e-5), name


def test_contrastive_pretraining_actually_updates_the_gnn(tmp_path) -> None:
    """TEAGLM freezes its GNN on construction; the pretraining routine used to inherit that, so
    only the throwaway alignment head trained and the saved 'pretrained' GNN equalled its random
    initialisation."""
    torch = pytest.importorskip("torch")
    from graph_modi.training import pretrain_graph_encoder_contrastive

    model = _tiny_model()
    assert not any(p.requires_grad for p in model.gnn.parameters())  # frozen by TEAGLM.__init__
    before = {k: v.clone() for k, v in model.gnn.state_dict().items()}
    graphs = [e.graph for e in _examples(6)]
    for target in ("mean", "attrs_only", "status_weighted"):
        pretrain_graph_encoder_contrastive(
            model.gnn,
            model.tensorizer,
            graphs,
            model.language_model,
            model.tokenizer,
            output_dir=tmp_path / target,
            epochs=2,
            graphs_per_batch=2,
            learning_rate=1e-2,
            seed=0,
            target=target,
            dropout=0.0,
            eval_graphs=graphs[:4],
        )
        saved = torch.load(tmp_path / target / "gnn.pt", map_location="cpu", weights_only=True)
        assert any(not torch.equal(before[k], saved[k]) for k in before)
        import json

        meta = json.loads((tmp_path / target / "metadata.json").read_text())
        assert "training_accuracy" not in meta
        assert meta["final_loss"] > 0 and meta["chance_loss"] > meta["final_loss"] - 5
        assert 0.0 <= meta["retrieval_top1_within_graph"] <= 1.0
        assert meta["target"] == target and len(meta["loss_history"]) == 2
        assert (tmp_path / target / "alignment_head.pt").exists()
    assert model.gnn.dropout.p == 0.0  # restored (it was 0.0); override must not leak


def test_vector_standardisation_is_a_noop_option_that_runs() -> None:
    torch = pytest.importorskip("torch")
    model = _tiny_model()
    graphs = [e.graph for e in _examples(2)]
    kwargs = {
        "reasoning_types": ["edge_exists"] * 2,
        "hops": [None, None],
        "source_ids": [g.nodes[0].id for g in graphs],
        "target_ids": [g.nodes[1].id for g in graphs],
    }
    plain = model.graph_prefix(graphs, **kwargs)
    model.graph_vector_norm = "standardize"
    standardized = model.graph_prefix(graphs, **kwargs)
    assert plain.shape == standardized.shape and not torch.equal(plain, standardized)


def test_sum_mlp_aggregation_adds_modules_only_for_itself_and_keeps_degree() -> None:
    torch = pytest.importorskip("torch")
    from graph_modi.models.tea_glm import GraphSAGEConfig, GraphSAGEEncoder

    def build(aggregation):
        torch.manual_seed(0)
        return GraphSAGEEncoder(
            GraphSAGEConfig(
                input_dim=8, hidden_dim=8, output_dim=8, num_layers=2, aggregation=aggregation
            )
        )

    assert not any(k.startswith("message_mlps") for k in build("sum").state_dict())
    mlp = build("sum_mlp")
    assert any(k.startswith("message_mlps") for k in mlp.state_dict())
    from graph_modi.models.tea_glm import DeterministicNodeTensorizer, NodeTensorizerConfig

    graphs = [e.graph for e in _examples(2)]
    tensorizer = DeterministicNodeTensorizer(NodeTensorizerConfig(feature_dim=8))
    out = mlp(tensorizer(graphs[0]))
    assert out.shape[1] == 8 and torch.isfinite(out).all()
    out.sum().backward()
    assert all(p.grad is not None for p in mlp.message_mlps.parameters())


def test_degree_balanced_generation_removes_the_degree_shortcut() -> None:
    from collections import Counter, defaultdict

    def build(balanced: bool):
        v3.set_degree_balance(balanced)
        try:
            return v3.generate_static_v3(
                split="validation",
                graph_count=33 * 12,
                tasks_per_graph=None,
                seed=7,
                sampler=v3.BalancedSampler(),
                progress=False,
            )
        finally:
            v3.set_degree_balance(False)

    def best_rule_accuracy(items, task):
        """Accuracy of the best lookup (source degree bin, target degree bin) -> majority answer."""
        table = defaultdict(Counter)
        for item in items:
            if item.query.reasoning_type.value != task:
                continue
            degree = {n.id: 0 for n in item.graph.nodes}
            for e in item.graph.edges:
                degree[e.source] += 1
                degree[e.target] += 1
            cap = 3 if item.query.target else 4  # the bins the sampler balances on
            key = (
                min(degree[item.query.source], cap),
                min(degree[item.query.target], cap) if item.query.target else -1,
            )
            table[key][item.answer] += 1
        total = sum(sum(c.values()) for c in table.values())
        return sum(c.most_common(1)[0][1] for c in table.values()) / total

    plain, balanced = build(False), build(True)
    # Original generator: degree predicts cycle membership almost perfectly.
    assert best_rule_accuracy(plain, "cycle_membership") > 0.85
    # Balanced: only degree >= 2 stations are asked about, and degree no longer predicts answers.
    for item in balanced:
        if item.query.reasoning_type.value == "cycle_membership":
            degree = sum(1 for e in item.graph.edges if item.query.source in (e.source, e.target))
            assert degree >= 2
    for task in ("cycle_membership", "edge_exists", "constrained_reachability", "reachability"):
        assert best_rule_accuracy(balanced, task) < 0.62, task
        yes = sum(i.answer == "yes" for i in balanced if i.query.reasoning_type.value == task)
        total = sum(1 for i in balanced if i.query.reasoning_type.value == task)
        assert abs(yes / total - 0.5) < 0.06, task
