#!/usr/bin/env python3
"""Figures for the CLEGR-extended task-family work (static gate history,
multi-neighbor-readout static probe, and the full 4-way dynamic eval).

Unlike analyze_exact2x_eval.py, this does not join raw per-turn rows (the
dynamic eval JSONs are ~130MB each and live only on the remote GPU box) —
the per-task/per-condition accuracies below are the already-aggregated
numbers reported in documents/experiments/results/v2_clegr_extended-20260919/
report.md (computed there from evaluation_{pooled,multi_neighbor}.json on the
remote box). Re-run the aggregation queries in that report's §7.2-7.4 if the
underlying evals are ever redone.

Writes documents/experiments/figures/clegr_extended_*.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "documents" / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#33415c",
        "axes.labelcolor": "#1b263b",
        "text.color": "#1b263b",
        "xtick.color": "#1b263b",
        "ytick.color": "#1b263b",
        "axes.grid": True,
        "grid.color": "#e0e4e8",
        "grid.linewidth": 0.8,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
    }
)

STATIC_TASKS = [
    "reachability",
    "cycle_membership",
    "edge_exists",
    "constrained_reachability",
    "filtered_neighbor_count",
    "filtered_path_count",
    "node_degree",
    "within_hops_count",
    "within_hops_list",
    "most_common_attribute_within_hops",
    "path_cost",
    "shortest_path",
]

# task -> (validation, test), percent
STATIC_HISTORY = {
    "Before": {
        "reachability": (53.9, 46.2),
        "cycle_membership": (60.3, 55.1),
        "edge_exists": (51.3, 56.4),
        "constrained_reachability": (52.6, 59.0),
        "filtered_neighbor_count": (17.9, 30.8),
        "filtered_path_count": (17.9, 15.4),
        "node_degree": (15.4, 26.9),
        "within_hops_count": (2.6, 3.8),
        "within_hops_list": (0.0, 0.0),
        "most_common_attribute_within_hops": (23.1, 21.8),
        "path_cost": (2.6, 0.0),
        "shortest_path": (10.3, 7.7),
        "Overall": (25.6, 26.9),
    },
    "v1 (+hops)": {
        "reachability": (68.0, 70.5),
        "cycle_membership": (21.8, 24.4),
        "edge_exists": (46.2, 43.6),
        "constrained_reachability": (41.0, 44.9),
        "filtered_neighbor_count": (14.1, 24.4),
        "filtered_path_count": (20.5, 28.2),
        "node_degree": (20.5, 20.5),
        "within_hops_count": (16.7, 11.5),
        "within_hops_list": (2.6, 1.3),
        "most_common_attribute_within_hops": (9.0, 14.1),
        "path_cost": (1.3, 1.3),
        "shortest_path": (5.1, 9.0),
        "Overall": (22.2, 24.5),
    },
    "v2 (+reasoning_type)": {
        "reachability": (56.4, 53.8),
        "cycle_membership": (80.8, 70.5),
        "edge_exists": (51.3, 51.3),
        "constrained_reachability": (30.8, 38.5),
        "filtered_neighbor_count": (10.3, 21.8),
        "filtered_path_count": (19.2, 24.4),
        "node_degree": (20.5, 25.6),
        "within_hops_count": (7.7, 11.5),
        "within_hops_list": (2.6, 1.3),
        "most_common_attribute_within_hops": (7.7, 19.2),
        "path_cost": (2.6, 1.3),
        "shortest_path": (7.7, 14.1),
        "Overall": (24.8, 27.8),
    },
}
STATIC_HISTORY_COLORS = {
    "Before": "#adb5bd",
    "v1 (+hops)": "#ca6702",
    "v2 (+reasoning_type)": "#2a9d8f",
}

# TEA, static exact-uniform eval: pooled (v2) vs multi-neighbor (v3), percent
STATIC_MULTI_NEIGHBOR = {
    "v2 pooled": {
        "reachability": (56.4, 53.8),
        "constrained_reachability": (30.8, 38.5),
        "cycle_membership": (80.8, 70.5),
        "edge_exists": (51.3, 51.3),
        "path_cost": (2.6, 1.3),
        "shortest_path": (7.7, 14.1),
        "filtered_neighbor_count": (10.3, 21.8),
        "filtered_path_count": (19.2, 24.4),
        "node_degree": (20.5, 25.6),
        "most_common_attribute_within_hops": (7.7, 19.2),
        "within_hops_count": (7.7, 11.5),
        "within_hops_list": (2.6, 2.6),
        "Overall": (24.8, 27.8),
    },
    "v3 multi-neighbor": {
        "reachability": (93.6, 87.2),
        "constrained_reachability": (71.8, 69.2),
        "cycle_membership": (85.9, 85.9),
        "edge_exists": (66.7, 62.8),
        "path_cost": (15.4, 14.1),
        "shortest_path": (17.9, 23.1),
        "filtered_neighbor_count": (21.8, 30.8),
        "filtered_path_count": (29.5, 20.5),
        "node_degree": (28.2, 29.5),
        "most_common_attribute_within_hops": (10.3, 16.7),
        "within_hops_count": (10.3, 14.1),
        "within_hops_list": (2.6, 2.6),
        "Overall": (37.8, 38.0),
    },
}
MULTI_NEIGHBOR_COLORS = {"v2 pooled": "#2a9d8f", "v3 multi-neighbor": "#264653"}
MULTI_NEIGHBOR_TASK_ORDER = [
    "reachability",
    "constrained_reachability",
    "cycle_membership",
    "edge_exists",
    "path_cost",
    "shortest_path",
    "filtered_neighbor_count",
    "filtered_path_count",
    "node_degree",
    "most_common_attribute_within_hops",
    "within_hops_count",
    "within_hops_list",
    "Overall",
]

# Full dynamic (multi-turn) eval, 8-task roster actually present in the
# dynamic session data on disk (see report.md §7 task-roster note).
DYNAMIC_TASK_ORDER = [
    "constrained_reachability",
    "reachability",
    "cycle_membership",
    "edge_exists",
    "filtered_neighbor_count",
    "within_hops_count",
    "most_common_attribute_within_hops",
    "node_count",
    "Overall",
]
DYNAMIC_4WAY = {
    "TEA v2 (pooled)": {
        "constrained_reachability": (38.7, 38.8),
        "reachability": (47.9, 47.6),
        "cycle_membership": (53.3, 53.7),
        "edge_exists": (46.6, 45.5),
        "filtered_neighbor_count": (17.0, 16.0),
        "within_hops_count": (9.6, 8.5),
        "most_common_attribute_within_hops": (8.4, 9.5),
        "node_count": (1.3, 1.2),
        "Overall": (27.8, 27.6),
    },
    "TEA v3 (multi-neighbor)": {
        "constrained_reachability": (55.7, 56.9),
        "reachability": (52.6, 51.9),
        "cycle_membership": (57.4, 57.8),
        "edge_exists": (48.0, 47.8),
        "filtered_neighbor_count": (20.0, 18.9),
        "within_hops_count": (9.7, 9.2),
        "most_common_attribute_within_hops": (9.0, 9.2),
        "node_count": (1.2, 1.2),
        "Overall": (31.7, 31.6),
    },
    "GraphToken v2 (pooled)": {
        "constrained_reachability": (40.1, 40.9),
        "reachability": (47.9, 48.2),
        "cycle_membership": (37.1, 37.7),
        "edge_exists": (43.1, 43.0),
        "filtered_neighbor_count": (18.3, 17.1),
        "within_hops_count": (8.7, 8.3),
        "most_common_attribute_within_hops": (12.6, 10.8),
        "node_count": (1.1, 1.2),
        "Overall": (26.1, 25.9),
    },
    "GraphToken v3 (multi-neighbor)": {
        "constrained_reachability": (51.6, 51.7),
        "reachability": (50.2, 48.7),
        "cycle_membership": (46.2, 46.7),
        "edge_exists": (45.3, 44.7),
        "filtered_neighbor_count": (19.5, 19.2),
        "within_hops_count": (10.2, 10.0),
        "most_common_attribute_within_hops": (8.1, 7.4),
        "node_count": (1.1, 1.0),
        "Overall": (29.0, 28.7),
    },
}
DYNAMIC_4WAY_COLORS = {
    "TEA v2 (pooled)": "#8ecae6",
    "TEA v3 (multi-neighbor)": "#023047",
    "GraphToken v2 (pooled)": "#ffb703",
    "GraphToken v3 (multi-neighbor)": "#bb3e03",
}

# TEA v2 (pooled), dynamic eval, per condition, percent (val, test)
TEA_V2_CONDITIONS = {
    "tool_solver": (100.0, 100.0),
    "majority_prior": (69.5, 68.3),
    "oracle_updated_graph": (36.1, 35.5),
    "predicted_updated_graph": (36.0, 35.3),
    "shuffled_graph": (30.6, 30.5),
    "cached_no_reencode": (30.2, 30.4),
    "modify_and_print": (30.1, 29.8),
    "structure_only": (29.9, 29.9),
    "frozen_graph_history": (30.5, 30.0),
    "graph_once_then_text": (23.8, 24.3),
    "question_only": (24.6, 24.2),
    "token_matched_history": (22.8, 22.0),
    "serialized_initial_history": (21.4, 21.5),
    "serialized_current_graph": (18.0, 17.7),
}

# Headline overall-accuracy story: static gate history -> static
# multi-neighbor probe -> full dynamic eval, val/test averaged for a single
# trajectory value per point.
PROGRESSION = [
    ("Before\n(static, TEA)", 26.25),
    ("v1 +hops\n(static, TEA)", 23.35),
    ("v2 +task-id\n(static, TEA)", 26.30),
    ("v3 multi-neigh.\n(static, TEA)", 37.90),
    ("v2 pooled\n(dynamic, TEA)", 27.70),
    ("v3 multi-neigh.\n(dynamic, TEA)", 31.65),
    ("v2 pooled\n(dynamic, GraphToken)", 26.00),
    ("v3 multi-neigh.\n(dynamic, GraphToken)", 28.85),
]


def _grouped_bar(ax, tasks, series: dict[str, dict[str, tuple[float, float]]], colors, split_index, title):
    n_groups = len(tasks)
    n_series = len(series)
    width = 0.8 / n_series
    x = np.arange(n_groups)
    for i, (name, values) in enumerate(series.items()):
        heights = [values[t][split_index] for t in tasks]
        offset = (i - (n_series - 1) / 2) * width
        bars = ax.bar(x + offset, heights, width=width * 0.92, label=name, color=colors[name])
        for bar, h in zip(bars, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.2,
                f"{h:.0f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#1b263b",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=38, ha="right", fontsize=8)
    ax.set_ylim(0, max(100, max(v[split_index] for values in series.values() for v in values.values()) + 12))
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.axhline(0, color="#33415c", linewidth=0.8)


def fig_static_history() -> None:
    tasks = STATIC_TASKS + ["Overall"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    for ax, split_index, split_name in zip(axes, (0, 1), ("Validation", "Test")):
        _grouped_bar(ax, tasks, STATIC_HISTORY, STATIC_HISTORY_COLORS, split_index, f"Static gate — {split_name}")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(
        "CLEGR-extended static oracle-QA gate: Before → v1 (+hops) → v2 (+reasoning_type)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "clegr_extended_static_history.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_multi_neighbor_static() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, split_index, split_name in zip(axes, (0, 1), ("Validation", "Test")):
        _grouped_bar(
            ax,
            MULTI_NEIGHBOR_TASK_ORDER,
            STATIC_MULTI_NEIGHBOR,
            MULTI_NEIGHBOR_COLORS,
            split_index,
            f"Static probe (TEA) — {split_name}",
        )
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.suptitle(
        "Multi-neighbor readout, static exact-uniform eval (TEA): pooled (v2) vs. multi-neighbor (v3)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "clegr_extended_multi_neighbor_static.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_dynamic_4way() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10.5), sharex=True)
    for ax, split_index, split_name in zip(axes, (0, 1), ("Validation", "Test")):
        _grouped_bar(
            ax,
            DYNAMIC_TASK_ORDER,
            DYNAMIC_4WAY,
            DYNAMIC_4WAY_COLORS,
            split_index,
            f"Full dynamic (multi-turn) eval — {split_name}",
        )
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9, ncol=2)
    fig.suptitle(
        "TEA vs. GraphToken × pooled (v2) vs. multi-neighbor (v3), full dynamic eval\n"
        "(8-task dynamic roster; averaged across 12 real conditions, excludes tool_solver/majority_prior)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "clegr_extended_dynamic_4way.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_tea_v2_conditions() -> None:
    conditions = sorted(TEA_V2_CONDITIONS, key=lambda c: -sum(TEA_V2_CONDITIONS[c]))
    val = [TEA_V2_CONDITIONS[c][0] for c in conditions]
    test = [TEA_V2_CONDITIONS[c][1] for c in conditions]
    y = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(y + 0.18, val, height=0.34, color="#2a9d8f", label="Validation")
    ax.barh(y - 0.18, test, height=0.34, color="#264653", label="Test")
    ax.set_yticks(y)
    ax.set_yticklabels(conditions, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Answer accuracy (%)")
    ax.set_title("TEA v2 (pooled), full dynamic eval — accuracy by condition", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    for yi, (v, t) in enumerate(zip(val, test)):
        ax.text(v + 1, yi + 0.18, f"{v:.0f}", va="center", fontsize=7.5)
        ax.text(t + 1, yi - 0.18, f"{t:.0f}", va="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "clegr_extended_tea_v2_conditions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_progression() -> None:
    labels = [p[0] for p in PROGRESSION]
    values = [p[1] for p in PROGRESSION]
    regime = ["static"] * 4 + ["dynamic"] * 4
    colors = ["#2a9d8f" if r == "static" else "#bb3e03" for r in regime]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    bars = ax.bar(x, values, color=colors, width=0.62)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.2, rotation=12, ha="right")
    ax.set_ylabel("Overall accuracy (%), val/test averaged")
    ax.set_title(
        "Overall accuracy across the CLEGR-extended iteration history\n"
        "(static probe overstated the multi-neighbor readout's real-world gain)",
        fontsize=12,
        fontweight="bold",
    )
    ax.axvline(3.5, color="#33415c", linestyle="--", linewidth=1)
    ax.text(1.5, max(values) + 6, "static oracle-QA gate", ha="center", fontsize=9, color="#2a9d8f", fontweight="bold")
    ax.text(5.5, max(values) + 6, "full dynamic (multi-turn) eval", ha="center", fontsize=9, color="#bb3e03", fontweight="bold")
    ax.set_ylim(0, max(values) + 12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "clegr_extended_progression.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_static_history()
    fig_multi_neighbor_static()
    fig_dynamic_4way()
    fig_tea_v2_conditions()
    fig_progression()
    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
