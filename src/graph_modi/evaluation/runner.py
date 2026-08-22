"""Paired evaluation conditions for graph tokens, text history, and tools."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from graph_modi.evaluation.metrics import aggregate_rows, normalize_answer
from graph_modi.graph.edits import execution_equivalent
from graph_modi.graph.executor import apply_edit, graph_fingerprint
from graph_modi.graph.serialization import (
    serialize_graph,
    serialize_history,
    token_budget_match,
)
from graph_modi.graph.solvers import answer_query, render_question
from graph_modi.models.base import GraphBackend, ModelInput
from graph_modi.pipeline.multiturn import materialize_states
from graph_modi.schema import AttributedGraph, Session

CONDITIONS = (
    "question_only",
    "frozen_graph_history",
    "graph_once_then_text",
    "shuffled_graph",
    "oracle_updated_graph",
    "cached_no_reencode",
    "predicted_updated_graph",
    "serialized_initial_history",
    "token_matched_history",
    "serialized_current_graph",
    "tool_solver",
)


def _prompt(
    condition: str,
    initial: AttributedGraph,
    current: AttributedGraph,
    history: list[str],
    question: str,
) -> str:
    history_text = serialize_history(history)
    if condition == "serialized_initial_history":
        return f"Initial graph: {serialize_graph(initial)}\n{history_text}\n{question}"
    if condition == "serialized_current_graph":
        return f"Current graph: {serialize_graph(current)}\n{question}"
    if condition == "token_matched_history":
        matched = token_budget_match(history_text, target_tokens=64)
        return f"{matched}\n{question}"
    if condition in {"frozen_graph_history", "cached_no_reencode", "graph_once_then_text"}:
        return f"{history_text}\n{question}"
    return question


def evaluate_sessions(
    sessions: Sequence[Session],
    backend: GraphBackend,
    *,
    conditions: Sequence[str] = CONDITIONS,
) -> dict[str, Any]:
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown evaluation conditions: {sorted(unknown)}")
    materialized = {session.session_id: materialize_states(session) for session in sessions}
    rows: list[dict[str, Any]] = []
    shuffled_initials = (
        [sessions[(index + 1) % len(sessions)].initial_graph for index in range(len(sessions))]
        if sessions
        else []
    )

    for session_index, session in enumerate(sessions):
        for condition in conditions:
            expected_states = materialized[session.session_id]
            current = session.initial_graph
            shuffled = shuffled_initials[session_index]
            encoded = backend.encode(
                shuffled if condition == "shuffled_graph" else session.initial_graph
            )
            history: list[str] = []
            for turn, expected in zip(session.turns, expected_states, strict=True):
                started = time.perf_counter()
                history.append(turn.utterance)
                predicted_edit = None
                edit_correct: bool | None = None
                if condition == "predicted_updated_graph":
                    predicted_edit = backend.predict_edit(turn.utterance, current)
                    edit_correct = execution_equivalent(predicted_edit, turn.gold_edit)
                    result = (
                        apply_edit(current, predicted_edit) if predicted_edit is not None else None
                    )
                else:
                    result = apply_edit(current, turn.gold_edit)
                if result is not None and result.applied:
                    current = result.graph
                    if condition in {
                        "oracle_updated_graph",
                        "predicted_updated_graph",
                    }:
                        encoded = backend.encode(current)

                question = render_question(turn.query, expected.after)
                predicted_answer: str | None
                if condition == "tool_solver":
                    predicted_answer = answer_query(current, turn.query)
                else:
                    model_current = shuffled if condition == "shuffled_graph" else current
                    predicted_answer = backend.answer(
                        ModelInput(
                            initial_graph=session.initial_graph,
                            current_graph=model_current,
                            query=turn.query,
                            question=_prompt(
                                condition,
                                session.initial_graph,
                                current,
                                history,
                                question,
                            ),
                            history=tuple(history),
                            condition=condition,
                            encoded_graph=encoded,
                            turn_index=turn.turn_index,
                        )
                    )
                predicted_normalized = normalize_answer(predicted_answer)
                gold = normalize_answer(turn.gold_answer)
                stale = normalize_answer(turn.stale_answer)
                state_exact = graph_fingerprint(current) == turn.after_fingerprint
                prompt = _prompt(
                    condition,
                    session.initial_graph,
                    current,
                    history,
                    question,
                )
                rows.append(
                    {
                        "session_id": session.session_id,
                        "turn_index": turn.turn_index,
                        "condition": condition,
                        "gold_answer": gold,
                        "stale_answer": stale,
                        "predicted_answer": predicted_normalized,
                        "answer_correct": predicted_normalized == gold,
                        "stale": predicted_normalized == stale,
                        "edit_correct": edit_correct,
                        "state_exact": state_exact,
                        "input_tokens": len(prompt.split()),
                        "latency_seconds": time.perf_counter() - started,
                        "graph_fingerprint": graph_fingerprint(current),
                    }
                )
    return {"summary": aggregate_rows(rows), "rows": rows}
