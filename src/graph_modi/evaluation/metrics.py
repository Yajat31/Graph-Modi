"""Metric aggregation for multi-turn graph modification."""

from __future__ import annotations

import re
from collections import defaultdict
from math import sqrt
from typing import Any


def normalize_answer(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().casefold().rstrip(".")
    aliases = {
        "true": "yes",
        "false": "no",
        "no path": "unreachable",
        "not reachable": "unreachable",
    }
    if normalized in aliases:
        return aliases[normalized]
    if "unreachable" in normalized or "no path" in normalized:
        return "unreachable"
    number = re.search(r"(?<![\w.-])-?\d+(?:\.\d+)?(?![\w.-])", normalized)
    if number:
        return number.group(0)
    boolean = re.search(r"\b(yes|no|true|false)\b", normalized)
    if boolean:
        return aliases.get(boolean.group(1), boolean.group(1))
    return normalized


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
            "mean_input_tokens": (
                sum(int(row["input_tokens"]) for row in condition_rows) / max(1, count)
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
