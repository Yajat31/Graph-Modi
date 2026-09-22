#!/usr/bin/env python3
"""Statistical rigor pass for the CLEGR-extended dynamic (multi-turn) eval
results -- ports the same methodology already used for the older 3-task
exact2x benchmark (Wilson 95% CIs, majority-class-shortcut audit; see
scripts/analyze_exact2x_eval.py and documents/experiments/STATUS_REPORT.md
section 8.5) to the newer 12-static/8-dynamic CLEGR-extended task family,
where it has never been applied -- every number reported in report.md
sections 7-10 so far is a single-seed point estimate with no CI and no
majority-baseline check.

Two things this computes per (architecture, task):
  1. Wilson 95% CI on answer accuracy (via graph_modi.evaluation.metrics),
     averaged across the 12 real conditions (excludes tool_solver/
     majority_prior, which are an oracle ceiling and a stale-answer-repeat
     baseline respectively -- see runner.py -- not majority-CLASS baselines).
  2. A genuine majority-class-shortcut audit: for each task, the single
     most common GOLD answer's own frequency (e.g. if 92% of
     `node_count` answers are "0", a model can hit 92% by ALWAYS predicting
     "0" without any real capability) -- flags any task where the model's
     accuracy is within a few points of (or below) this trivial ceiling.

Run once per already-completed dynamic eval JSON (these are ~130MB each,
analysis stays on the remote box rather than downloading them):
  .venv/bin/python scripts/analyze_clegr_extended_dynamic.py \
    --config configs/v2_clegr_extended_exact2x_tea.yaml \
    --evaluation outputs/v2_clegr_extended_exact2x_tea/evaluation_pooled.json \
    --label "TEA v2 (pooled)"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graph_modi.cli import _data_path  # noqa: E402
from graph_modi.config import load_config  # noqa: E402
from graph_modi.data.multiturn import load_sessions  # noqa: E402
from graph_modi.evaluation.metrics import stratified_accuracy, wilson_interval  # noqa: E402

_BASELINE_CONDITIONS = {"tool_solver", "majority_prior"}


def _task_lookup(config, split: str) -> dict[tuple[str, int], str]:
    sessions = load_sessions(_data_path(config, split))
    lookup: dict[tuple[str, int], str] = {}
    for session in sessions:
        for turn in session.turns:
            lookup[(session.session_id, turn.turn_index)] = turn.query.reasoning_type.value
    return lookup


def _majority_class_baseline(rows: list[dict]) -> tuple[str, float]:
    """The single most common gold answer's own frequency in this task's
    rows -- the accuracy a model gets for FREE by always predicting one
    fixed string, with zero graph reasoning. Not the same as the
    `majority_prior` CONDITION (which repeats each turn's own stale
    pre-update answer, a staleness/tracking diagnostic, not a global
    label-skew check)."""
    counts = Counter(row["gold_answer"] for row in rows)
    label, count = counts.most_common(1)[0]
    return label, count / len(rows)


def _majority_labeling_diagnostic(rows: list[dict], majority_label: str) -> dict:
    """Same check as documents/experiments/STATUS_REPORT.md section 8.5:
    a model tied with (or below) the majority-class baseline on raw
    accuracy is NOT necessarily majority-labeling -- it could be
    systematically strong on the minority class and weak on the majority
    class, netting out to a similar or lower number while doing real work.
    Distinguish the two via P(predict=majority_label), accuracy conditioned
    on gold=majority_label vs gold!=majority_label, and balanced accuracy
    (their mean) -- true majority-labeling shows P(predict=majority_label)
    near 1.0 and near-zero minority-class accuracy."""
    predicted_majority = sum(1 for row in rows if row["predicted_answer"] == majority_label)
    majority_gold_rows = [row for row in rows if row["gold_answer"] == majority_label]
    minority_gold_rows = [row for row in rows if row["gold_answer"] != majority_label]
    acc_given_majority = (
        sum(row["answer_correct"] for row in majority_gold_rows) / len(majority_gold_rows)
        if majority_gold_rows
        else float("nan")
    )
    acc_given_minority = (
        sum(row["answer_correct"] for row in minority_gold_rows) / len(minority_gold_rows)
        if minority_gold_rows
        else float("nan")
    )
    return {
        "p_predict_majority": predicted_majority / len(rows),
        "acc_given_gold_majority": acc_given_majority,
        "acc_given_gold_minority": acc_given_minority,
        "balanced_accuracy": (acc_given_majority + acc_given_minority) / 2
        if majority_gold_rows and minority_gold_rows
        else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation", required=True, help="path to evaluation_*.json")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    data = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))

    print(f"\n{'=' * 70}\n{args.label}\n{'=' * 70}")
    for split in ("validation", "test"):
        if split not in data:
            continue
        lookup = _task_lookup(config, split)
        rows = data[split]["rows"]
        real_rows = [row for row in rows if row["condition"] not in _BASELINE_CONDITIONS]
        for row in real_rows:
            row["reasoning_type"] = lookup.get((row["session_id"], row["turn_index"]), "UNKNOWN")

        stats = stratified_accuracy(real_rows, stratum_key="reasoning_type")
        by_task_rows: dict[str, list[dict]] = defaultdict(list)
        for row in real_rows:
            by_task_rows[row["reasoning_type"]].append(row)

        print(f"\n-- {split} (n={len(real_rows)} turns, 12 real conditions) --")
        header = (
            f"{'task':34} {'acc':>7} {'95% CI':>16} {'majority':>9} {'lift':>7} "
            f"{'P(pred=maj)':>11} {'acc|maj':>8} {'acc|min':>8} {'balanced':>9}  verdict"
        )
        print(header)
        print("-" * len(header))
        total_correct = sum(1 for row in real_rows if row["answer_correct"])
        for task in sorted(stats):
            s = stats[task]
            majority_label, majority_acc = _majority_class_baseline(by_task_rows[task])
            lift = s["answer_accuracy"] - majority_acc
            diag = _majority_labeling_diagnostic(by_task_rows[task], majority_label)
            # True majority-labeling: predicts the majority label almost
            # always AND is near-zero on the minority class -- lift alone
            # can't tell these apart from "legitimately reasoning but netting
            # out near/below majority due to being weak on the MAJORITY
            # class specifically" (the pattern found for the older 3-task
            # benchmark in STATUS_REPORT.md section 8.5).
            if diag["p_predict_majority"] > 0.85 and diag["acc_given_gold_minority"] < 0.15:
                verdict = "MAJORITY-LABELING"
            elif lift <= 0.02:
                verdict = "weak, not majority-labeling"
            else:
                verdict = "real lift"
            ci_lo, ci_hi = s["answer_accuracy_ci95"]
            print(
                f"{task:34} {s['answer_accuracy']:6.1%} "
                f"[{ci_lo:5.1%},{ci_hi:5.1%}] {majority_acc:8.1%} {lift:+6.1%} "
                f"{diag['p_predict_majority']:10.1%} {diag['acc_given_gold_majority']:7.1%} "
                f"{diag['acc_given_gold_minority']:7.1%} {diag['balanced_accuracy']:8.1%}  {verdict}"
            )
        overall_acc = total_correct / len(real_rows)
        ci_lo, ci_hi = wilson_interval(total_correct, len(real_rows))
        print("-" * len(header))
        print(f"{'OVERALL':38} {overall_acc:6.1%} [{ci_lo:5.1%},{ci_hi:5.1%}]")


if __name__ == "__main__":
    main()
