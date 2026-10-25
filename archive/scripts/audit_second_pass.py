"""Second-pass checks on documents/experiments/results/audit-20261016 (JSON only, no GPU).

Prints the tables cited in documents/experiments/experimental_audit_addendum_2026-10-21.md:
  A. agreement with the degree+status shortcut on items where the shortcut is wrong
  B. split of the re-encoding gain into shortcut-right ("easy") and shortcut-wrong ("hard") turns
  C. models vs the shortcut tracker, with unpaired binomial half-widths
  D. multiplicity in stats_test_final.json
  E. per-seed slice behaviour of the three scratch GraphTokens

Usage: python scripts/audit_second_pass.py [results_dir]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "documents/experiments/results/audit-20261016")
BINARY = ("constrained_reachability", "cycle_membership", "edge_exists", "reachability")
MODELS = ("gt_sum", "gt_sum_s43", "gt_warm", "tea_sum_random", "tea_sum", "old_tea")


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def half_width(p: float, n: int) -> float:
    return 1.96 * math.sqrt(p * (1 - p) / n)


def section_a(shortcut: dict) -> None:
    print("A. Binary tasks, items where the shortcut is wrong: agreement with the shortcut = 1 - accuracy")
    print(f"{'model':16}" + "".join(f"{t[:14]:>16}" for t in BINARY))
    for model in MODELS:
        hard = shortcut["static"][model]["hard_per_task"]
        cells = []
        for task in BINARY:
            acc, n = hard[task]
            cells.append(f"{1 - acc:6.2f} (n={n:3d})")
        print(f"{model:16}" + "".join(f"{c:>16}" for c in cells))
    print("   chance agreement = 0.50; values near 0.8 mean the model follows the shortcut\n")


def section_b(shortcut: dict) -> None:
    print("B. Re-encoding gain (updated minus frozen graph) on shortcut-right vs shortcut-wrong turns")
    multi = shortcut["multi"]
    print(f"{'model':16}{'all':>8}{'easy':>8}{'hard':>8}   hard turns n={multi['gt_sum']['oracle_updated_graph']['hard'][1]}")
    for model in ("gt_sum", "gt_warm", "tea_sum_random"):
        up, fr = multi[model]["oracle_updated_graph"], multi[model]["frozen_graph_history"]
        gains = [100 * (up[k][0] - fr[k][0]) for k in ("all", "easy", "hard")]
        print(f"{model:16}" + "".join(f"{g:8.1f}" for g in gains))
    sc = multi["shortcut"]
    print(f"{'shortcut':16}{100 * (sc['updated_graph'] - sc['stale_graph']):8.1f}   (tracker, all turns)\n")


def section_c(shortcut: dict) -> None:
    print("C. Models vs shortcut tracker (8 non-attribute tasks)")
    multi, static = shortcut["multi"], shortcut["static"]
    sc_multi = multi["shortcut"]["updated_graph"]
    n = multi["shortcut"]["turns"]
    for model in ("gt_sum", "gt_warm", "tea_sum_random"):
        acc = multi[model]["oracle_updated_graph"]["all"][0]
        print(
            f"  multi  {model:16} {100 * acc:5.1f} vs {100 * sc_multi:5.1f}  "
            f"diff {100 * (acc - sc_multi):+5.1f}  (unpaired +/-{100 * half_width(acc, n):.1f})"
        )
    sc_static = static["shortcut"]["task_mean"]
    items = 8 * 396
    for model in ("gt_sum", "gt_warm", "tea_sum_random"):
        acc = static[model]["overall"]
        print(
            f"  static {model:16} {100 * acc:5.1f} vs {100 * sc_static:5.1f}  "
            f"diff {100 * (acc - sc_static):+5.1f}  (unpaired +/-{100 * half_width(acc, items):.1f})"
        )
    print()


def section_d(stats: dict) -> None:
    tests = []
    for group in ("across_models", "within_model"):
        for pair, conditions in stats[group].items():
            for condition, row in conditions.items():
                if isinstance(row, dict) and "p_value" in row:
                    tests.append((pair, condition, row["p_value"], row["mean_delta"]))
    total = len(tests)
    below = sum(p < 0.05 for _, _, p, _ in tests)
    print(f"D. stats_test_final.json: {total} tests, {below} with p < 0.05, Bonferroni alpha = {0.05 / total:.5f}")
    print("   p in [0.01, 0.05):")
    for pair, condition, p, delta in tests:
        if 0.01 <= p < 0.05:
            print(f"     {pair} / {condition}: p={p}, delta={100 * delta:+.1f} pts")
    print("   p == 0.0 means below the 1/5000 resolution of the bootstrap; report as p < 0.0002\n")


def section_e(shortcut: dict) -> None:
    print("E. Reachability slices by scratch seed (static test accuracy)")
    for model in ("gt_sum", "gt_sum_s43", "gt_warm", "tea_sum_random"):
        sl = shortcut["static"][model]["slices"]
        print(
            f"  {model:16} both endpoints open {100 * sl['reach_both_open'][0]:5.1f}   "
            f"an endpoint closed {100 * sl['reach_endpoint_closed'][0]:5.1f}"
        )
    print()


def main() -> None:
    shortcut = load("shortcut_audit.json")
    section_a(shortcut)
    section_b(shortcut)
    section_c(shortcut)
    section_d(load("stats_test_final.json"))
    section_e(shortcut)


if __name__ == "__main__":
    main()
