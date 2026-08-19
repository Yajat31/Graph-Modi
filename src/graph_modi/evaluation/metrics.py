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
