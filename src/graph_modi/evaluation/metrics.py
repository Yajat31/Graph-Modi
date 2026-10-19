"""Metric aggregation for multi-turn graph modification."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from math import sqrt
from typing import Any


_ALIASES = {
    "true": "yes",
    "false": "no",
    "no path": "unreachable",
    "not reachable": "unreachable",
}
_NUMBER = r"-?\d+(?:\.\d+)?"


def _leading_answer(text: str) -> str | None:
    """The answer a generation *leads with*, or None if it does not lead with one.

    Free-form text-only baselines answer in prose ("No, there are 2 routes."). Scanning the whole
    string for the first number would read that as "2"; the answer is the leading token.
    """
    if re.match(r"(?:unreachable|no path|not reachable)\b", text):
        return "unreachable"
    leading_boolean = re.match(r"(yes|no|true|false)\b", text)
    if leading_boolean:
        return _ALIASES.get(leading_boolean.group(1), leading_boolean.group(1))
    leading_number = re.match(rf"({_NUMBER})(?![\w.-])", text)
    if leading_number:
        return leading_number.group(1)
    return None


def normalize_answer(value: str | None) -> str | None:
    """Canonicalise a gold or generated answer.

    Order: exact alias -> answer the text leads with (first token) -> keyword anywhere
    ("unreachable"/"no path") -> first number -> first yes/no word -> the text itself.
    The leading-token rule comes first so trailing prose cannot change the answer.
    """
    if value is None:
        return None
    normalized = value.strip().casefold().rstrip(".")
    if normalized in _ALIASES:
        return _ALIASES[normalized]
    leading = _leading_answer(normalized)
    if leading is not None:
        return leading
    if "unreachable" in normalized or "no path" in normalized:
        return "unreachable"
    number = re.search(rf"(?<![\w.-]){_NUMBER}(?![\w.-])", normalized)
    if number:
        return number.group(0)
    boolean = re.search(r"\b(yes|no|true|false)\b", normalized)
    if boolean:
        return _ALIASES.get(boolean.group(1), boolean.group(1))
    return normalized


# Reasoning types whose answer is a number, compared with tolerance rather
# than exact string equality (LLM-generated formatting varies, e.g. "12" vs
# "12.0"; PATH_COST in particular can be a non-integer weighted-path cost).
_NUMERIC_REASONING_TYPES = {
    "node_degree",
    "node_count",
    "path_cost",
    "filtered_neighbor_count",
    "filtered_path_count",
    "within_hops_count",
    "shortest_path",
}
# Reasoning types whose answer is a comma-separated set of entities.
_LIST_REASONING_TYPES = {"within_hops_list"}


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _list_answer_set(value: str) -> frozenset[str]:
    if value in {"none", ""}:
        return frozenset()
    return frozenset(token.strip() for token in value.split(",") if token.strip())


def answers_match(
    gold: str | None,
    predicted: str | None,
    *,
    reasoning_type: str | None = None,
    numeric_tolerance: float = 0.01,
) -> bool:
    """Task-aware correctness check, replacing blind exact-match for numeric/list tasks.

    Boolean and categorical tasks keep exact string equality (unchanged
    behavior); numeric tasks (e.g. PATH_COST) get a tolerance; list tasks
    (e.g. WITHIN_HOPS_LIST) compare as sets rather than literal strings.
    """
    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(predicted)
    if gold_norm is None or pred_norm is None:
        return gold_norm == pred_norm
    if reasoning_type in _LIST_REASONING_TYPES:
        return _list_answer_set(gold_norm) == _list_answer_set(pred_norm)
    if reasoning_type in _NUMERIC_REASONING_TYPES:
        gold_value = _as_float(gold_norm)
        pred_value = _as_float(pred_norm)
        if gold_value is not None and pred_value is not None:
            return abs(gold_value - pred_value) <= numeric_tolerance
    return gold_norm == pred_norm


def set_f1(gold: str | None, predicted: str | None) -> float:
    """Set-valued F1 for list answers (e.g. WITHIN_HOPS_LIST); README's target metric."""
    gold_set = _list_answer_set(normalize_answer(gold) or "")
    pred_set = _list_answer_set(normalize_answer(predicted) or "")
    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0
    overlap = len(gold_set & pred_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_set)
    recall = overlap / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z * sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _session_totals(rows: list[dict[str, Any]], key: str = "answer_correct") -> dict[str, list[int]]:
    """Per-session [number correct, number of turns]; turns of one session are not independent."""
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        entry = totals[str(row["session_id"])]
        entry[0] += int(bool(row.get(key)))
        entry[1] += 1
    return totals


def _percentile(sorted_values: list[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def cluster_bootstrap_ci(
    rows: list[dict[str, Any]],
    *,
    resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for accuracy, resampling whole sessions (the unit of dependence)."""
    totals = list(_session_totals(rows).values())
    if not totals:
        return 0.0, 0.0
    rng = random.Random(seed)
    count = len(totals)
    estimates = []
    for _ in range(resamples):
        correct = turns = 0
        for _ in range(count):
            entry = totals[rng.randrange(count)]
            correct += entry[0]
            turns += entry[1]
        estimates.append(correct / max(1, turns))
    estimates.sort()
    return _percentile(estimates, alpha / 2), _percentile(estimates, 1 - alpha / 2)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    report: dict[str, dict[str, Any]] = {}
    for condition, condition_rows in grouped.items():
        count = len(condition_rows)
        final_rows: dict[str, dict[str, Any]] = {}
        for row in condition_rows:
            final_rows[str(row["session_id"])] = row
        answer_successes = sum(bool(row["answer_correct"]) for row in condition_rows)

        def rate(
            key: str,
            rows: list[dict[str, Any]] = condition_rows,
        ) -> float:
            available = [row[key] for row in rows if row.get(key) is not None]
            return sum(bool(value) for value in available) / max(1, len(available))

        report[condition] = {
            "turns": count,
            "sessions": len(final_rows),
            "answer_accuracy": answer_successes / max(1, count),
            "answer_accuracy_ci95": list(wilson_interval(answer_successes, count)),
            # Wilson treats turns as independent; this resamples sessions instead.
            "answer_accuracy_ci95_session_bootstrap": list(cluster_bootstrap_ci(condition_rows)),
            "final_state_accuracy": (
                sum(bool(row["answer_correct"]) for row in final_rows.values())
                / max(1, len(final_rows))
            ),
            "stale_rate": rate("stale"),
            "exact_graph_state": rate("state_exact"),
            "execution_equivalent_edit_accuracy": rate("edit_correct"),
            "mean_latency_seconds": (
                sum(float(row["latency_seconds"]) for row in condition_rows) / max(1, count)
            ),
            # Whitespace-split words of the text prompt (graph prefix tokens excluded).
            "mean_input_tokens": (
                sum(int(row["input_tokens"]) for row in condition_rows) / max(1, count)
            ),
            "mean_graph_prefix_tokens": (
                sum(int(row.get("graph_prefix_tokens", 0)) for row in condition_rows)
                / max(1, count)
            ),
        }
    return report


def stratified_accuracy(
    rows: list[dict[str, Any]],
    *,
    stratum_key: str,
) -> dict[str, dict[str, Any]]:
    """CLEGR-style per-stratum answer accuracy with Wilson CIs.

    ``rows`` must already carry the stratum field (joined from session metadata).
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if stratum_key not in row or row[stratum_key] is None:
            continue
        buckets[str(row[stratum_key])].append(row)
    out: dict[str, dict[str, Any]] = {}
    for label, bucket in sorted(buckets.items(), key=lambda item: item[0]):
        successes = sum(bool(row.get("answer_correct")) for row in bucket)
        total = len(bucket)
        out[label] = {
            "n": total,
            "answer_accuracy": successes / max(1, total),
            "answer_accuracy_ci95": list(wilson_interval(successes, total)),
            "stale_rate": (
                sum(bool(row.get("stale")) for row in bucket if row.get("stale") is not None)
                / max(1, sum(row.get("stale") is not None for row in bucket))
            ),
            "execution_equivalent_edit_accuracy": (
                sum(bool(row.get("edit_correct")) for row in bucket if row.get("edit_correct") is not None)
                / max(1, sum(row.get("edit_correct") is not None for row in bucket))
            ),
        }
    return out


def paired_accuracy_delta(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...] = ("session_id", "turn_index"),
) -> dict[str, Any]:
    """Mean paired Δ (left − right) on aligned turns, with Wilson-free summary."""
    right_map = {
        tuple(row[field] for field in key_fields): row
        for row in right_rows
    }
    deltas: list[float] = []
    wins = loses = ties = 0
    for row in left_rows:
        key = tuple(row[field] for field in key_fields)
        other = right_map.get(key)
        if other is None:
            continue
        left_ok = bool(row.get("answer_correct"))
        right_ok = bool(other.get("answer_correct"))
        deltas.append(float(left_ok) - float(right_ok))
        if left_ok and not right_ok:
            wins += 1
        elif right_ok and not left_ok:
            loses += 1
        else:
            ties += 1
    n = len(deltas)
    mean = sum(deltas) / max(1, n)
    return {
        "n": n,
        "mean_delta": mean,
        "wins": wins,
        "loses": loses,
        "ties": ties,
    }


def paired_cluster_bootstrap(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...] = ("session_id", "turn_index"),
    resamples: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired accuracy difference (left - right) with a session-clustered bootstrap.

    Turns are aligned on ``key_fields``; sessions are resampled as a whole. ``p_value`` is the
    two-sided bootstrap probability of a difference at least as far from zero on the other side.
    """
    right_map = {tuple(row[field] for field in key_fields): row for row in right_rows}
    per_session: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    wins = loses = ties = 0
    for row in left_rows:
        other = right_map.get(tuple(row[field] for field in key_fields))
        if other is None:
            continue
        left_ok, right_ok = bool(row.get("answer_correct")), bool(other.get("answer_correct"))
        entry = per_session[str(row["session_id"])]
        entry[0] += float(left_ok) - float(right_ok)
        entry[1] += 1.0
        if left_ok and not right_ok:
            wins += 1
        elif right_ok and not left_ok:
            loses += 1
        else:
            ties += 1
    sessions = list(per_session.values())
    turns = sum(entry[1] for entry in sessions)
    if not sessions or not turns:
        return {"n": 0, "mean_delta": 0.0, "ci95": [0.0, 0.0], "p_value": 1.0}
    mean = sum(entry[0] for entry in sessions) / turns
    rng = random.Random(seed)
    count = len(sessions)
    deltas = []
    for _ in range(resamples):
        total = weight = 0.0
        for _ in range(count):
            entry = sessions[rng.randrange(count)]
            total += entry[0]
            weight += entry[1]
        deltas.append(total / max(1.0, weight))
    deltas.sort()
    at_most_zero = sum(delta <= 0 for delta in deltas) / resamples
    at_least_zero = sum(delta >= 0 for delta in deltas) / resamples
    return {
        "n": int(turns),
        "sessions": count,
        "mean_delta": mean,
        "ci95": [_percentile(deltas, 0.025), _percentile(deltas, 0.975)],
        "p_value": min(1.0, 2 * min(at_most_zero, at_least_zero)),
        "wins": wins,
        "loses": loses,
        "ties": ties,
    }
