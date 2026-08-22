"""Dataset validity, leakage, and balance reports."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from graph_modi.graph.executor import apply_edit_program, graph_fingerprint
from graph_modi.graph.solvers import answer_query
from graph_modi.schema import EditOperation, Session


def audit_sessions(sessions: Sequence[Session], *, allow_noop: bool = False) -> dict[str, Any]:
    operations: Counter[str] = Counter()
    reasoning: Counter[str] = Counter()
    answers: Counter[str] = Counter()
    failures: list[str] = []
    turns = 0
    noop_turns = 0
    for session in sessions:
        current = session.initial_graph
        for turn in session.turns:
            turns += 1
            prefix = f"{session.session_id}/{turn.turn_index}"
            if graph_fingerprint(current) != turn.before_fingerprint:
                failures.append(f"{prefix}: non-cumulative before state")
            edits = turn.gold_edits or (turn.gold_edit,)
            if len(edits) == 1 and edits[0].operation is EditOperation.NOOP:
                noop_turns += 1
                updated = current
            else:
                result = apply_edit_program(current, edits)
                if not result.applied and not allow_noop:
                    failures.append(f"{prefix}: gold edit rejected")
                    continue
                updated = result.graph
            current = updated
            observed = answer_query(current, turn.query)
            if observed != turn.gold_answer:
                failures.append(f"{prefix}: oracle label mismatch")
            is_noop = edits and edits[0].operation is EditOperation.NOOP
            if turn.gold_answer == turn.stale_answer and not is_noop and not allow_noop:
                failures.append(f"{prefix}: unchanged answer shortcut")
            if graph_fingerprint(current) != turn.after_fingerprint:
                failures.append(f"{prefix}: after fingerprint mismatch")
            if not is_noop:
                tokens = set(re.findall(r"\b[\w-]+\b", turn.utterance.casefold()))
                if turn.gold_answer.casefold() in tokens:
                    failures.append(f"{prefix}: answer leaked in revision")
            for edit in edits:
                operations[edit.operation.value] += 1
            reasoning[turn.query.reasoning_type.value] += 1
            answers[turn.gold_answer] += 1
    majority = max(answers.values(), default=0) / max(1, turns)
    return {
        "sessions": len(sessions),
        "turns": turns,
        "noop_turns": noop_turns,
        "valid": not failures,
        "failures": failures,
        "operation_distribution": dict(operations),
        "reasoning_distribution": dict(reasoning),
        "answer_distribution": dict(answers),
        "majority_answer_accuracy": majority,
        "counterfactual_pairs": turns - noop_turns,
    }
