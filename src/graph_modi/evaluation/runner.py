"""Paired evaluation conditions for graph tokens, text history, and tools."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from graph_modi.evaluation.metrics import aggregate_rows, normalize_answer
from graph_modi.graph.edits import execution_equivalent_program
from graph_modi.graph.executor import apply_edit, apply_edit_program, graph_fingerprint
from graph_modi.graph.serialization import (
    serialize_graph,
    serialize_history,
    token_budget_match,
)
from graph_modi.graph.solvers import answer_query, render_question
from graph_modi.models.base import GraphBackend, ModelInput
from graph_modi.pipeline.multiturn import materialize_states
from graph_modi.schema import AttributedGraph, GraphEdit, Session
from graph_modi.utils.progress import ProgressTracker

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
    "soft_prompt",
    "structure_only",
    "modify_and_print",
    "majority_prior",
)

_REENCODE_CONDITIONS = {"oracle_updated_graph", "predicted_updated_graph"}


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
    if condition == "modify_and_print":
        return f"Current graph: {serialize_graph(current)}\n{history_text}\n{question}"
    if condition == "token_matched_history":
        matched = token_budget_match(history_text, target_tokens=64)
        return f"{matched}\n{question}"
    if condition in {"frozen_graph_history", "cached_no_reencode", "graph_once_then_text"}:
        return f"{history_text}\n{question}"
    if condition == "soft_prompt":
        return f"{history_text}\n{question}"
    if condition == "structure_only":
        return question
    return question


@dataclass
class _Task:
    """Per-(session, condition) state, evolving one round (turn index) at a time."""

    session: Session
    condition: str
    expected_states: Sequence[Any]
    current: AttributedGraph
    shuffled: AttributedGraph
    encoded: Any
    history: list[str] = field(default_factory=list)
    question: str = ""
    predicted_edit: GraphEdit | None = None
    edit_correct: bool | None = None
    predicted_answer: str | None = None


def _advance_round(
    condition: str,
    round_index: int,
    chunk: list[_Task],
    backend: GraphBackend,
) -> None:
    """Advance one batch of tasks through round ``round_index``: append history,
    predict/apply the edit, re-encode if needed, and get an answer — batched
    across every task in the chunk wherever the backend supports it."""
    for task in chunk:
        task.history.append(task.session.turns[round_index].utterance)

    if condition == "predicted_updated_graph":
        items = [(task.session.turns[round_index].utterance, task.current) for task in chunk]
        predicted_edits = backend.predict_edit_batch(items)
        for task, edit in zip(chunk, predicted_edits, strict=True):
            turn = task.session.turns[round_index]
            gold_edits = turn.gold_edits or (
                turn.edit_program.edits if turn.edit_program else (turn.gold_edit,)
            )
            predicted = edit if isinstance(edit, tuple) else ((edit,) if edit is not None else None)
            task.predicted_edit = edit
            task.edit_correct = execution_equivalent_program(predicted, gold_edits)
    else:
        for task in chunk:
            task.predicted_edit = None
            task.edit_correct = None

    for task in chunk:
        turn = task.session.turns[round_index]
        if condition == "predicted_updated_graph":
            predicted = task.predicted_edit
            if isinstance(predicted, tuple):
                result = apply_edit_program(task.current, predicted) if predicted else None
            else:
                result = apply_edit(task.current, predicted) if predicted else None
        else:
            edits = turn.gold_edits or (
                turn.edit_program.edits if turn.edit_program else (turn.gold_edit,)
            )
            result = apply_edit_program(task.current, edits)
        if result is not None and result.applied:
            task.current = result.graph
            if condition in _REENCODE_CONDITIONS:
                task.encoded = backend.encode(task.current)

    for task in chunk:
        turn = task.session.turns[round_index]
        expected = task.expected_states[round_index]
        task.question = render_question(turn.query, expected.after)

    if condition == "tool_solver":
        for task in chunk:
            turn = task.session.turns[round_index]
            task.predicted_answer = answer_query(task.current, turn.query)
        return

    if condition == "majority_prior":
        for task in chunk:
            turn = task.session.turns[round_index]
            task.predicted_answer = turn.stale_answer
        return

    model_inputs = []
    for task in chunk:
        turn = task.session.turns[round_index]
        model_current = task.shuffled if condition == "shuffled_graph" else task.current
        model_inputs.append(
            ModelInput(
                initial_graph=task.session.initial_graph,
                current_graph=model_current,
                query=turn.query,
                question=_prompt(
                    condition, task.session.initial_graph, task.current, task.history, task.question
                ),
                history=tuple(task.history),
                condition=condition,
                encoded_graph=task.encoded,
                turn_index=turn.turn_index,
            )
        )
    answers = backend.answer_batch(model_inputs)
    for task, answer in zip(chunk, answers, strict=True):
        task.predicted_answer = answer


def _row_for(task: _Task, round_index: int, latency_seconds: float) -> dict[str, Any]:
    turn = task.session.turns[round_index]
    predicted_normalized = normalize_answer(task.predicted_answer)
    gold = normalize_answer(turn.gold_answer)
    stale = normalize_answer(turn.stale_answer)
    prompt = _prompt(
        task.condition, task.session.initial_graph, task.current, task.history, task.question
    )
    return {
        "session_id": task.session.session_id,
        "turn_index": turn.turn_index,
        "condition": task.condition,
        "gold_answer": gold,
        "stale_answer": stale,
        "predicted_answer": predicted_normalized,
        "answer_correct": predicted_normalized == gold,
        "stale": predicted_normalized == stale,
        "edit_correct": task.edit_correct,
        "state_exact": graph_fingerprint(task.current) == turn.after_fingerprint,
        "input_tokens": len(prompt.split()),
        "latency_seconds": latency_seconds,
        "graph_fingerprint": graph_fingerprint(task.current),
    }


def evaluate_sessions(
    sessions: Sequence[Session],
    backend: GraphBackend,
    *,
    conditions: Sequence[str] = CONDITIONS,
    batch_size: int = 16,
    progress: bool = False,
) -> dict[str, Any]:
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown evaluation conditions: {sorted(unknown)}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    materialized = {session.session_id: materialize_states(session) for session in sessions}
    shuffled_initials = (
        [sessions[(index + 1) % len(sessions)].initial_graph for index in range(len(sessions))]
        if sessions
        else []
    )

    total_turns = len(conditions) * sum(len(session.turns) for session in sessions)
    tracker = ProgressTracker(
        "[evaluate]",
        total_turns,
        phase="dynamic",
        enabled=progress,
    )
    start_time = time.monotonic()
    if progress:
        tracker.banner(
            sessions=len(sessions),
            conditions=len(conditions),
            total_turns=total_turns,
            batch_size=batch_size,
        )

    rows: list[dict[str, Any]] = []
    for condition in conditions:
        tasks = []
        for session_index, session in enumerate(sessions):
            shuffled = shuffled_initials[session_index]
            if condition == "shuffled_graph":
                initial_for_encode = shuffled
            else:
                initial_for_encode = session.initial_graph
            tasks.append(
                _Task(
                    session=session,
                    condition=condition,
                    expected_states=materialized[session.session_id],
                    current=session.initial_graph,
                    shuffled=shuffled,
                    encoded=backend.encode(initial_for_encode),
                )
            )
        max_turns = max((len(task.session.turns) for task in tasks), default=0)
        for round_index in range(max_turns):
            active = [task for task in tasks if round_index < len(task.session.turns)]
            for start in range(0, len(active), batch_size):
                chunk = active[start : start + batch_size]
                batch_started = time.perf_counter()
                _advance_round(condition, round_index, chunk, backend)
                batch_latency = (time.perf_counter() - batch_started) / len(chunk)
                rows.extend(_row_for(task, round_index, batch_latency) for task in chunk)
                tracker.tick(
                    len(chunk),
                    condition=condition,
                    round=f"{round_index + 1}/{max_turns}",
                )
    if progress:
        tracker.end(elapsed=f"{time.monotonic() - start_time:.0f}s")
    return {"summary": aggregate_rows(rows), "rows": rows}
