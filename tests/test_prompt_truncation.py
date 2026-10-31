"""Prompt truncation: BOS survives, text-only prompts get the whole budget, truncation is loud."""

import warnings
from types import SimpleNamespace

import pytest

from graph_modi.data.multiturn import generate_dataset
from graph_modi.evaluation.runner import evaluate_sessions
from graph_modi.models.base import SymbolicMockBackend
from graph_modi.models.soft_prompt import SoftPromptGLM
from graph_modi.models.tea_glm import TEAGLM, TEAGLMBackend, fit_prompt, warn_truncated

BOS = 1


class WordTokenizer:
    """One token per whitespace word (ids from 10 up), plus BOS/EOS ids."""

    bos_token_id = BOS
    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [10 + index for index, _ in enumerate(text.split())]


def _tea_stub(*, max_sequence_length: int, prefix_tokens: int = 3):
    stub = SimpleNamespace(
        config=SimpleNamespace(
            max_sequence_length=max_sequence_length,
            add_bos_token=True,
            add_eos_token=True,
            prompt_template="Question: {question}\nAnswer:",
        ),
        projector=SimpleNamespace(config=SimpleNamespace(prefix_tokens=prefix_tokens)),
        tokenizer=WordTokenizer(),
    )
    stub._encode_text = lambda text, answer: TEAGLM._encode_text(stub, text, answer=answer)
    stub._truncate = lambda *args, **kwargs: TEAGLM._truncate(stub, *args, **kwargs)
    return stub


def test_truncation_keeps_the_leading_bos() -> None:
    prompt = [BOS, *range(10, 30)]
    kept, answer = fit_prompt(prompt, [], available=8, bos_token_id=BOS)
    assert kept == [BOS, *range(23, 30)]  # BOS plus the last 7 tokens
    assert answer == []
    # The answer-overflow branch keeps BOS too.
    kept, answer = fit_prompt(prompt, list(range(40, 60)), available=8, bos_token_id=BOS)
    assert kept == [BOS] and answer == list(range(40, 47))
    # Short prompts are untouched; without a BOS the plain left cut applies.
    assert fit_prompt([BOS, 10, 11], [], available=8, bos_token_id=BOS)[0] == [BOS, 10, 11]
    assert fit_prompt(list(range(10, 30)), [], available=4, bos_token_id=BOS)[0] == [26, 27, 28, 29]


def test_soft_prompt_truncation_keeps_the_leading_bos() -> None:
    stub = SimpleNamespace(
        config=SimpleNamespace(max_sequence_length=8, prefix_tokens=3),
        tokenizer=WordTokenizer(),
    )
    kept, _ = SoftPromptGLM._truncate(stub, [BOS, *range(10, 30)], [])
    assert kept == [BOS, *range(26, 30)]  # 8 - 3 soft-prompt tokens


def test_text_only_prompts_get_the_full_budget() -> None:
    stub = _tea_stub(max_sequence_length=8, prefix_tokens=3)
    prompt = [BOS, *range(10, 30)]
    assert len(TEAGLM._truncate(stub, prompt, [], graph_prefix=False)[0]) == 8
    assert len(TEAGLM._truncate(stub, prompt, [])[0]) == 5  # graph tokens take 3


def test_truncation_is_warned_once_per_batch_with_a_count() -> None:
    full = [[BOS, *range(10, 30)], [BOS, 10], [BOS, *range(10, 20)]]
    kept = [fit_prompt(ids, [], available=8, bos_token_id=BOS)[0] for ids in full]
    with pytest.warns(UserWarning, match="2 of 3 prompts") as caught:
        assert warn_truncated(full, kept) == 2
    assert len(caught) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert warn_truncated(full[1:2], kept[1:2]) == 0


def test_backend_reports_prompt_length_and_truncation() -> None:
    stub = _tea_stub(max_sequence_length=12, prefix_tokens=3)
    stub.prompt_length = lambda prompt, **kwargs: TEAGLM.prompt_length(stub, prompt, **kwargs)
    backend = object.__new__(TEAGLMBackend)  # no LM: only the model's tokenizer path is used
    backend.model = stub
    long_question = " ".join(["w"] * 9)  # BOS + "Question:" + 9 words + "Answer:" = 12 tokens

    def count(condition: str) -> tuple[int, bool]:
        model_input = SimpleNamespace(question=long_question, condition=condition, turn_index=0)
        return backend.prompt_token_count(model_input)

    assert count("serialized_current_graph") == (12, False)  # text-only: 12-token budget
    assert count("oracle_updated_graph") == (12, True)  # graph prefix leaves 9


class CountingBackend(SymbolicMockBackend):
    def prompt_token_count(self, model_input) -> tuple[int, bool]:
        tokens = len(model_input.question.split())
        return tokens, tokens > 20


def test_evaluation_rows_carry_prompt_tokens_and_truncated_flag() -> None:
    sessions = generate_dataset(
        counts={"test": 2},
        seed=5,
        node_count_min=8,
        node_count_max=9,
        turns=[2],
        reasoning_types=["node_degree"],
    )["test"]
    conditions = ["serialized_current_graph", "oracle_updated_graph"]
    rows = evaluate_sessions(sessions, CountingBackend(), conditions=conditions)["rows"]
    assert rows and all({"prompt_tokens", "truncated", "input_tokens"} <= set(row) for row in rows)
    by_condition = {row["condition"]: row for row in rows}
    assert by_condition["serialized_current_graph"]["truncated"] is True
    assert by_condition["oracle_updated_graph"]["truncated"] is False
    # Backends without a tokenizer leave the fields empty.
    plain = evaluate_sessions(sessions, SymbolicMockBackend(), conditions=conditions)["rows"]
    assert all(row["prompt_tokens"] is None and row["truncated"] is None for row in plain)
