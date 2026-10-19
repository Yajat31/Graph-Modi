"""Session-clustered confidence intervals and paired tests from `evaluation.json` files.

The turns of one session share a graph and a revision history, so they are not independent; the
per-turn Wilson interval in the evaluation summaries is therefore too narrow. This script
resamples whole sessions, and compares conditions / models on the *same* (session, turn) cells.

    python scripts/stats_v3.py --split test --out stats_test.json \
        tea=outputs/v3_exact2x_tea/evaluation.json \
        tea_run2=outputs/v3_exact2x_graphtoken/evaluation.json \
        gt_e2e=outputs/v3_exact2x_graphtoken_e2e/evaluation.json \
        soft_prompt=outputs/v3_exact2x_soft_prompt/evaluation.json

Each ``label=path`` is one model. Within a model it compares the graph conditions with each
other; across models it compares the same condition (the cross-model comparison between two
models of the same architecture is the run-to-run noise floor).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_modi.evaluation.metrics import (  # noqa: E402
    cluster_bootstrap_ci,
    paired_cluster_bootstrap,
    wilson_interval,
)

LEGACY = {"majority_prior": "stale_answer"}
WITHIN_MODEL = [  # (left, right): is the first condition better than the second?
    ("oracle_updated_graph", "frozen_graph_history"),
    ("oracle_updated_graph", "shuffled_graph"),
    ("oracle_updated_graph", "structure_only"),
    ("oracle_updated_graph", "question_only"),
    ("frozen_graph_history", "question_only"),
    ("frozen_graph_history", "shuffled_graph"),
]
ACROSS_MODELS = ("oracle_updated_graph", "frozen_graph_history", "shuffled_graph", "structure_only")


def pc(value: float) -> str:
    return f"{100 * value:.1f}"


def load_rows(path: str, split: str) -> dict[str, list[dict]]:
    data = json.loads(Path(path).read_text())
    rows_by_condition: dict[str, list[dict]] = {}
    for row in data[split]["rows"]:
        condition = LEGACY.get(row["condition"], row["condition"])
        rows_by_condition.setdefault(condition, []).append(row)
    return rows_by_condition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument(
        "--pair", action="append", default=[],
        help="extra comparison modelA:conditionA:modelB:conditionB (A minus B)",
    )
    parser.add_argument("models", nargs="+", help="label=path/to/evaluation.json")
    args = parser.parse_args()

    models = {}
    for spec in args.models:
        label, _, path = spec.partition("=")
        models[label] = load_rows(path, args.split)

    report: dict = {"split": args.split, "accuracy": {}, "within_model": {}, "across_models": {}}
    for label, by_condition in models.items():
        report["accuracy"][label] = {}
        for condition, rows in sorted(by_condition.items()):
            correct = sum(bool(row["answer_correct"]) for row in rows)
            low, high = cluster_bootstrap_ci(rows, resamples=args.resamples)
            wilson = wilson_interval(correct, len(rows))
            report["accuracy"][label][condition] = {
                "turns": len(rows),
                "sessions": len({row["session_id"] for row in rows}),
                "accuracy": correct / max(1, len(rows)),
                "wilson_ci95": list(wilson),
                "session_bootstrap_ci95": [low, high],
            }
        report["within_model"][label] = {}
        for left, right in WITHIN_MODEL:
            if left in by_condition and right in by_condition:
                report["within_model"][label][f"{left} - {right}"] = paired_cluster_bootstrap(
                    by_condition[left], by_condition[right], resamples=args.resamples
                )
    for (a, rows_a), (b, rows_b) in itertools.combinations(models.items(), 2):
        for condition in ACROSS_MODELS:
            if condition in rows_a and condition in rows_b:
                report["across_models"].setdefault(f"{a} - {b}", {})[condition] = (
                    paired_cluster_bootstrap(rows_a[condition], rows_b[condition],
                                             resamples=args.resamples)
                )
    report["custom_pairs"] = {}
    for spec in args.pair:
        a, condition_a, b, condition_b = spec.split(":")
        report["custom_pairs"][spec] = paired_cluster_bootstrap(
            models[a][condition_a], models[b][condition_b], resamples=args.resamples
        )
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"split={args.split}")
    for label, conditions in report["accuracy"].items():
        print(f"\n{label}")
        for condition, entry in conditions.items():
            low, high = entry["session_bootstrap_ci95"]
            wl, wh = entry["wilson_ci95"]
            print(f"  {condition:26s} {pc(entry['accuracy'])}  session-CI [{pc(low)}, {pc(high)}]"
                  f"  (per-turn Wilson [{pc(wl)}, {pc(wh)}])")
    print("\nPaired differences (left - right), accuracy points, session bootstrap:")
    for label, comparisons in report["within_model"].items():
        for name, result in comparisons.items():
            print(f"  {label:12s} {name:52s} {100 * result['mean_delta']:+5.1f}  "
                  f"CI [{100 * result['ci95'][0]:+.1f}, {100 * result['ci95'][1]:+.1f}]  p={result['p_value']:.3f}")
    for pair, comparisons in report["across_models"].items():
        for condition, result in comparisons.items():
            print(f"  {pair:24s} {condition:24s} {100 * result['mean_delta']:+5.1f}  "
                  f"CI [{100 * result['ci95'][0]:+.1f}, {100 * result['ci95'][1]:+.1f}]  p={result['p_value']:.3f}")
    for spec, result in report["custom_pairs"].items():
        print(f"  {spec:60s} {100 * result['mean_delta']:+5.1f}  "
              f"CI [{100 * result['ci95'][0]:+.1f}, {100 * result['ci95'][1]:+.1f}]  p={result['p_value']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
