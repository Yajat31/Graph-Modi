"""Dataset validity, leakage, and balance reports."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from graph_modi.graph.executor import apply_edit, graph_fingerprint
from graph_modi.graph.solvers import answer_query
from graph_modi.schema import Session


def audit_sessions(sessions: Sequence[Session]) -> dict[str, Any]:
    operations: Counter[str] = Counter()
    reasoning: Counter[str] = Counter()
    answers: Counter[str] = Counter()
    failures: list[str] = []
    turns = 0
    for session in sessions:
        current = session.initial_graph
        for turn in session.turns:
            turns += 1
            prefix = f"{session.session_id}/{turn.turn_index}"
            if graph_fingerprint(current) != turn.before_fingerprint:
                failures.append(f"{prefix}: non-cumulative before state")
            result = apply_edit(current, turn.gold_edit)
            if not result.applied:
                failures.append(f"{prefix}: gold edit rejected ({result.error})")
                continue
            current = result.graph
            observed = answer_query(current, turn.query)
            if observed != turn.gold_answer:
                failures.append(f"{prefix}: oracle label mismatch")
            if turn.gold_answer == turn.stale_answer:
                failures.append(f"{prefix}: unchanged answer shortcut")
            if graph_fingerprint(current) != turn.after_fingerprint:
                failures.append(f"{prefix}: after fingerprint mismatch")
            tokens = set(re.findall(r"\b[\w-]+\b", turn.utterance.casefold()))
            if turn.gold_answer.casefold() in tokens:
                failures.append(f"{prefix}: answer leaked in revision")
            operations[turn.gold_edit.operation.value] += 1
            reasoning[turn.query.reasoning_type.value] += 1
            answers[turn.gold_answer] += 1
    majority = max(answers.values(), default=0) / max(1, turns)
    return {
        "sessions": len(sessions),
        "turns": turns,
        "valid": not failures,
        "failures": failures,
        "operation_distribution": dict(operations),
        "reasoning_distribution": dict(reasoning),
        "answer_distribution": dict(answers),
        "majority_answer_accuracy": majority,
        "counterfactual_pairs": turns,
    }
