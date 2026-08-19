"""Edit, apply, re-encode, and answer over cumulative graph state."""

from __future__ import annotations

import time
from dataclasses import dataclass

from graph_modi.graph.executor import apply_edit, graph_fingerprint
from graph_modi.graph.solvers import render_question
from graph_modi.models.base import GraphBackend, ModelInput
from graph_modi.schema import AttributedGraph, Session, TurnOutput


@dataclass(frozen=True, slots=True)
class MaterializedTurn:
    before: AttributedGraph
    after: AttributedGraph


def materialize_states(session: Session) -> tuple[MaterializedTurn, ...]:
    current = session.initial_graph
    states: list[MaterializedTurn] = []
    for turn in session.turns:
        before = current
        if graph_fingerprint(before) != turn.before_fingerprint:
            raise ValueError(
                f"Session {session.session_id} turn {turn.turn_index} does not "
                "continue from the preceding graph"
            )
        result = apply_edit(before, turn.gold_edit, strict=True)
        current = result.graph
        if graph_fingerprint(current) != turn.after_fingerprint:
            raise ValueError(
                f"Session {session.session_id} turn {turn.turn_index} has an "
                "incorrect after_fingerprint"
            )
        states.append(MaterializedTurn(before=before, after=current))
    return tuple(states)


def run_session(
    session: Session,
    backend: GraphBackend,
    *,
    predicted_edits: bool = False,
    include_history: bool = False,
) -> tuple[TurnOutput, ...]:
    current = session.initial_graph
    encoded = backend.encode(current)
    history: list[str] = []
    outputs: list[TurnOutput] = []
    for turn in session.turns:
        started = time.perf_counter()
        predicted = (
            backend.predict_edit(turn.utterance, current) if predicted_edits else turn.gold_edit
        )
        if predicted is None:
            outputs.append(
                TurnOutput(
                    turn_index=turn.turn_index,
                    predicted_edit=None,
                    edit_applied=False,
                    predicted_answer=None,
                    graph_fingerprint=graph_fingerprint(current),
                    encode_calls=backend.encode_calls,
                    latency_seconds=time.perf_counter() - started,
                    error="edit_parse_failed",
                )
            )
            continue
        result = apply_edit(current, predicted)
        if result.applied:
            current = result.graph
            encoded = backend.encode(current)
        history.append(turn.utterance)
        question = render_question(turn.query, current)
        answer = backend.answer(
            ModelInput(
                initial_graph=session.initial_graph,
                current_graph=current,
                query=turn.query,
                question=question,
                history=tuple(history if include_history else ()),
                condition=(
                    "predicted_updated_graph" if predicted_edits else "oracle_updated_graph"
                ),
                encoded_graph=encoded,
            )
        )
        outputs.append(
            TurnOutput(
                turn_index=turn.turn_index,
                predicted_edit=predicted,
                edit_applied=result.applied,
                predicted_answer=answer,
                graph_fingerprint=graph_fingerprint(current),
                encode_calls=backend.encode_calls,
                latency_seconds=time.perf_counter() - started,
                error=result.error,
            )
        )
    return tuple(outputs)
