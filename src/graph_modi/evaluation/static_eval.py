"""Static oracle-QA capability gate before dynamic evaluation."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from graph_modi.evaluation.metrics import answers_match, normalize_answer
from graph_modi.graph.solvers import render_question
from graph_modi.models.base import GraphBackend, ModelInput, SymbolicMockBackend
from graph_modi.schema import StaticQATuple
from graph_modi.utils.progress import ProgressTracker

GATE_OVERALL_THRESHOLD = 0.70
GATE_PER_TASK_THRESHOLD = 0.50
GATE_UNIMODAL_MARGIN = 0.10


def _oracle_backend() -> GraphBackend:
    return SymbolicMockBackend()


def evaluate_static_oracle(
    tuples: Sequence[StaticQATuple],
    backend: GraphBackend | None = None,
    *,
    batch_size: int = 16,
    progress: bool = True,
) -> dict[str, Any]:
    """Batched static QA evaluation with per-task/complexity breakdown."""
    solver = backend or _oracle_backend()
    total = len(tuples)
    tracker = ProgressTracker("[static-eval]", total, phase="oracle-qa", enabled=progress)
    tracker.banner(total=total, batch_size=batch_size)
    rows: list[dict[str, Any]] = []
    start = time.monotonic()
    for offset in range(0, total, batch_size):
        chunk = tuples[offset : offset + batch_size]
        inputs = [
            ModelInput(
                initial_graph=item.graph,
                current_graph=item.graph,
                query=item.query,
                question=render_question(item.query, item.graph),
                history=(),
                condition="oracle_updated_graph",
            )
            for item in chunk
        ]
        predictions = solver.answer_batch(inputs)
        for item, predicted in zip(chunk, predictions, strict=True):
            reasoning_type = str(
                item.metadata.get("reasoning_type", item.query.reasoning_type.value)
            )
            gold = normalize_answer(item.answer)
            pred = normalize_answer(predicted)
            rows.append(
                {
                    "tuple_id": item.tuple_id,
                    "split": item.split,
                    "reasoning_type": reasoning_type,
                    "topology": item.topology.value,
                    "density_bin": item.density_bin.value,
                    "hop_depth": item.hop_depth.value,
                    "scale_bin": item.scale_bin,
                    "gold_answer": gold,
                    "predicted_answer": pred,
                    "correct": answers_match(
                        item.answer, predicted, reasoning_type=reasoning_type
                    ),
                }
            )
        tracker.tick(batch=f"{min(offset + batch_size, total)}/{total}")
    tracker.end(correct=sum(row["correct"] for row in rows), total=total)
    return summarize_static_gate(rows, elapsed_seconds=time.monotonic() - start)


def summarize_static_gate(
    rows: list[dict[str, Any]],
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    overall = sum(row["correct"] for row in rows) / max(1, len(rows))
    by_task: dict[str, list[bool]] = defaultdict(list)
    by_scale: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_task[str(row["reasoning_type"])].append(bool(row["correct"]))
        by_scale[str(row["scale_bin"])].append(bool(row["correct"]))
    task_accuracy = {task: sum(values) / len(values) for task, values in sorted(by_task.items())}
    scale_accuracy = {
        scale: sum(values) / len(values) for scale, values in sorted(by_scale.items())
    }
    failing_tasks = [
        task for task, accuracy in task_accuracy.items() if accuracy < GATE_PER_TASK_THRESHOLD
    ]
    passed = overall >= GATE_OVERALL_THRESHOLD and not failing_tasks
    return {
        "overall_accuracy": overall,
        "task_accuracy": task_accuracy,
        "scale_accuracy": scale_accuracy,
        "passed": passed,
        "thresholds": {
            "overall": GATE_OVERALL_THRESHOLD,
            "per_task": GATE_PER_TASK_THRESHOLD,
            "unimodal_margin": GATE_UNIMODAL_MARGIN,
        },
        "failing_tasks": failing_tasks,
        "rows": rows,
        "elapsed_seconds": elapsed_seconds,
    }


def compare_multimodal_gain(
    full_rows: list[dict[str, Any]],
    text_rows: list[dict[str, Any]],
    structure_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    full = sum(row["correct"] for row in full_rows) / max(1, len(full_rows))
    text = sum(row["correct"] for row in text_rows) / max(1, len(text_rows))
    structure = sum(row["correct"] for row in structure_rows) / max(1, len(structure_rows))
    unimodal = max(text, structure)
    gain = full - unimodal
    return {
        "full_accuracy": full,
        "text_accuracy": text,
        "structure_accuracy": structure,
        "multimodal_gain": gain,
        "passed": gain >= GATE_UNIMODAL_MARGIN,
    }
