"""Figures for the mid-project report (ANLP.pdf) -> documents/experiments/figures/report/.

Two sets, each as PNG (for review) and PDF (vector, for LaTeX):

  main/      Figures 1-3 of the report, now also showing TEA with model-written (predicted)
             edits and GraphToken with the re-encoded gold graph.
  appendix/  The same three views with every condition of Table 1: the graph-token
             conditions on one row, the text-only conditions and the two reference points
             (stale answer, per-task majority) on the other.

Reads documents/experiments/results/v3-20260929/. The per-task majority is not in the
exact2x analysis files, so it is computed from the regenerated v3 sessions (seed 42) and
cached in per_task_majority_cuts.json; pass --rebuild-majority to regenerate it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "documents/experiments/results/v3-20260929"
OUT = ROOT / "documents/experiments/figures/report"
MAJORITY_CACHE = RES / "per_task_majority_cuts.json"

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SPLITS = ("validation", "test")

# Categorical slots from the validated default palette; each condition keeps its colour in
# every figure. Adjacent pairs pass the CVD (>=8) and normal-vision (>=15) checks per panel;
# every series also has its own marker and dash so identity never rests on colour alone.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

# (label, source file key, condition, colour, dash, marker)
SERIES = {
    "tea_oracle": ("TEA updated graph (gold edits)", "tea", "oracle_updated_graph", BLUE, "-", "o"),
    "tea_predicted": ("TEA predicted graph (model-written edits)", "tea_predfix", "predicted_updated_graph", BLUE, (0, (4, 2)), "s"),
    "gt_oracle": ("GraphToken updated graph (gold edits)", "graphtoken", "oracle_updated_graph", ORANGE, "-", "^"),
    "tea_frozen": ("TEA frozen graph + history", "tea", "frozen_graph_history", VIOLET, "-", "v"),
    "tea_shuffled": ("TEA shuffled graph", "tea", "shuffled_graph", YELLOW, "-.", "X"),
    "tea_question": ("TEA question only", "tea", "question_only", MUTED, (0, (3, 2)), "D"),
    "soft": ("soft prompt (history text)", "soft_prompt", "soft_prompt", AQUA, "-", "P"),
    # appendix-only conditions (Table 1)
    "tea_structure": ("TEA structure only (stale G0, no history)", "tea", "structure_only", GREEN, (0, (6, 2)), "h"),
    "tea_frozen_cached": ("TEA frozen graph history / cached no re-encode", "tea", "frozen_graph_history", VIOLET, "-", "v"),
    "tea_once": ("TEA graph once, then text", "tea", "graph_once_then_text", RED, (0, (1, 1.5)), "<"),
    "tea_token_matched": ("TEA token-matched history (no graph)", "tea", "token_matched_history", MAGENTA, (0, (5, 1, 1, 1)), ">"),
    "oracle_context": ("TEA updated graph (for reference)", "tea", "oracle_updated_graph", BLUE, "-", None),
    "stale": ("stale answer (majority_prior, no model)", "tea", "majority_prior", INK, (0, (1, 1)), None),
    "majority": ("per-task majority (no model)", "majority", None, INK2, (0, (6, 3)), None),
}

MAIN_LINES = ("tea_oracle", "tea_predicted", "gt_oracle", "tea_frozen", "tea_shuffled", "tea_question", "soft")
MAIN_BARS = ("tea_oracle", "tea_predicted", "gt_oracle", "tea_frozen", "tea_question", "soft")
APPX_GRAPH = ("tea_oracle", "tea_structure", "tea_frozen_cached", "tea_shuffled", "tea_once")
APPX_TEXT = ("oracle_context", "soft", "tea_token_matched", "tea_question", "stale", "majority")
APPX_BARS = ("tea_oracle", "tea_structure", "tea_frozen_cached", "tea_once", "tea_shuffled",
             "tea_question", "tea_token_matched", "soft", "majority", "stale")

DIMS = {
    "session_length": (["1", "2", "4", "8"], "session length (number of turns)", "accuracy by session length"),
    "turn_index": ([str(i) for i in range(8)], "turn index within the session (turn 0 = first question)",
                   "accuracy by turn index (later turns only exist in 8-turn sessions)"),
}
TURN_NOTE = ("Sessions reaching each turn: turn 0 = 1,188; turn 1 = 891; turns 2-3 = 594; turns 4-7 = 297 "
             "(about +/-5 points at the 95% level, so late-turn wiggles are mostly noise).")


# ------------------------------------------------------------------ data
def build_majority_cache() -> dict:
    """Per-task majority (each task's most common gold answer on the split) per cut."""
    sys.path.insert(0, str(ROOT / "src"))
    from graph_modi.data import v3
    from graph_modi.evaluation.metrics import normalize_answer

    sessions = v3.generate_sessions_v3(replicates_per_split={"validation": 1, "test": 1}, seed=42,
                                       tasks=list(v3.DYNAMIC_TASKS), progress=False)
    result = {}
    for split in SPLITS:
        rows = [
            {"task": turn.query.reasoning_type.value, "gold": normalize_answer(turn.gold_answer),
             "stale": normalize_answer(turn.stale_answer), "session_length": str(len(session.turns)),
             "turn_index": str(turn.turn_index)}
            for session in sessions[split] for turn in session.turns
        ]
        counts: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            counts[row["task"]][row["gold"]] += 1
        majority = {task: c.most_common(1)[0][0] for task, c in counts.items()}
        for row in rows:
            row["correct"] = row["gold"] == majority[row["task"]]
            row["changed"] = "answer_changed" if row["gold"] != row["stale"] else "answer_unchanged"
        cuts = {}
        for dim in ("session_length", "turn_index", "changed"):
            groups: dict[str, list[bool]] = defaultdict(list)
            for row in rows:
                groups[row[dim]].append(row["correct"])
            cuts[f"majority::{dim}"] = {k: {"acc": sum(v) / len(v), "n": len(v)} for k, v in sorted(groups.items())}
        result[split] = {"overall": sum(r["correct"] for r in rows) / len(rows), "turns": len(rows), "cuts": cuts}
    MAJORITY_CACHE.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result


def load_all(rebuild_majority: bool) -> dict:
    files = {key: RES / f"exact2x_{key}_analysis.json" for key in ("tea", "tea_predfix", "graphtoken", "soft_prompt")}
    data = {key: json.loads(path.read_text()) for key, path in files.items()}
    data["majority"] = build_majority_cache() if rebuild_majority or not MAJORITY_CACHE.exists() else json.loads(MAJORITY_CACHE.read_text())
    return data


def cut(data: dict, key: str, split: str, dim: str) -> dict:
    _, source, cond, *_ = SERIES[key]
    name = f"majority::{dim}" if source == "majority" else f"{cond}::{dim}"
    return data[source][split]["cuts"][name]


# ------------------------------------------------------------------ drawing
def style(ax, grid_axis: str = "both") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def handle(key: str) -> Line2D:
    label, _, _, color, dash, marker = SERIES[key]
    context = key == "oracle_context"
    return Line2D([], [], color=color, linestyle=dash, marker=marker, markersize=5, linewidth=1.2 if context else 2,
                  alpha=0.45 if context else 1.0, markeredgecolor=SURFACE, markeredgewidth=0.8, label=label)


def draw_lines(ax, data: dict, keys, split: str, dim: str) -> None:
    order = DIMS[dim][0]
    for key in keys:
        _, _, _, color, dash, marker = SERIES[key]
        values = cut(data, key, split, dim)
        xs = [int(k) for k in order if k in values]
        ys = [100 * values[k]["acc"] for k in order if k in values]
        context = key == "oracle_context"
        ax.plot(xs, ys, color=color, linestyle=dash, linewidth=1.2 if context else 2,
                alpha=0.45 if context else 1.0, marker=marker, markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=2 if context else 3)
    ax.set_xticks([int(k) for k in order])


def save(fig, folder: str, stem: str) -> None:
    target = OUT / folder
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / f"{stem}.png", dpi=200, facecolor=SURFACE)
    fig.savefig(target / f"{stem}.pdf", facecolor=SURFACE)
    plt.close(fig)


def fig_main_lines(data: dict, dim: str, stem: str) -> None:
    order, xlabel, title = DIMS[dim]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), facecolor=SURFACE, sharey=True)
    for ax, split in zip(axes, SPLITS):
        style(ax)
        draw_lines(ax, data, MAIN_LINES, split, dim)
        ax.set_title(split, color=INK, fontsize=10, loc="left")
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
        ax.set_ylim(20, 42)
    axes[0].set_ylabel("accuracy (%)", color=INK2, fontsize=9)
    fig.suptitle(f"Exact2x: {title}", color=INK, fontsize=11, x=0.01, ha="left")
    note = dim == "turn_index"
    fig.legend(handles=[handle(k) for k in MAIN_LINES], loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.045 if note else 0.0))
    if note:
        fig.text(0.01, 0.008, TURN_NOTE, fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.15 if note else 0.11, 1, 0.95))
    save(fig, "main", stem)


def fig_appendix_lines(data: dict, dim: str, stem: str) -> None:
    order, xlabel, title = DIMS[dim]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), facecolor=SURFACE, sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.25]})
    rows = (("graph-token conditions (TEA)", APPX_GRAPH, (20, 42)),
            ("text-only conditions and reference points", APPX_TEXT, (0, 66)))
    for r, (row_title, keys, ylim) in enumerate(rows):
        for c, split in enumerate(SPLITS):
            ax = axes[r, c]
            style(ax)
            draw_lines(ax, data, keys, split, dim)
            ax.set_ylim(*ylim)
            ax.set_title(f"{split}: {row_title}", color=INK, fontsize=9.5, loc="left")
            if c == 0:
                ax.set_ylabel("accuracy (%)", color=INK2, fontsize=9)
            if r == 1:
                ax.set_xlabel(xlabel, color=INK2, fontsize=9)
            else:
                ax.tick_params(labelbottom=True)
        axes[r, 1].sharey(axes[r, 0])
        axes[r, 1].tick_params(labelleft=False)
    fig.suptitle(f"Exact2x, all Table 1 conditions: {title}", color=INK, fontsize=11, x=0.01, ha="left")
    graph_handles = [handle(k) for k in APPX_GRAPH]
    text_handles = [handle(k) for k in APPX_TEXT]
    fig.legend(handles=graph_handles + text_handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.018 if dim == "turn_index" else 0.0))
    bottom = 0.105
    if dim == "turn_index":
        fig.text(0.01, 0.003, TURN_NOTE, fontsize=7.5, color=INK2)
        bottom = 0.125
    fig.tight_layout(rect=(0, bottom, 1, 0.96))
    save(fig, "appendix", stem)


def se(p: float, n: int) -> float:
    return 100 * math.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))


GROUPS = (("answer_changed", "turns where the edit changed the answer"),
          ("answer_unchanged", "turns where the answer did not change"))


def bar_handle(key: str) -> Patch:
    label, _, _, color, *_ = SERIES[key]
    hatched, hollow = key == "tea_predicted", key == "majority"
    return Patch(facecolor=SURFACE if hatched or hollow else color, edgecolor=color, hatch="////" if hatched else None,
                 linewidth=1.2, label=label)


def draw_bars(ax, data: dict, keys, split: str, ymax: float) -> None:
    width = 0.84 / len(keys)
    for j, key in enumerate(keys):
        _, _, _, color, *_ = SERIES[key]
        values = cut(data, key, split, "changed")
        for i, (group, _) in enumerate(GROUPS):
            share = values[group]
            value = 100 * share["acc"]
            x = i + (j - (len(keys) - 1) / 2) * width
            hatched, hollow = key == "tea_predicted", key == "majority"
            reference = key in ("stale", "majority")
            ax.bar(x, min(value, ymax), width=width - 0.02, color=SURFACE if hatched or hollow else color,
                   edgecolor=color if hatched or hollow else SURFACE, hatch="////" if hatched else None,
                   linewidth=1.2 if hatched or hollow else 0.5,
                   yerr=None if reference else 1.96 * se(share["acc"], share["n"]),
                   error_kw={"ecolor": INK2, "elinewidth": 0.7, "capsize": 1.5})
            if key == "stale":
                if value > ymax:  # truncated bar: mark the break and print the true value
                    ax.plot([x - width * 0.45, x + width * 0.45], [ymax - 1.2, ymax - 0.4], color=SURFACE, linewidth=2.2)
                ax.text(x, min(value, ymax) + 0.6, f"{value:.0f}", ha="center", va="bottom", fontsize=7, color=INK)


def group_labels(ax, data: dict, split: str) -> None:
    counts = cut(data, "tea_oracle", split, "changed")
    total = sum(v["n"] for v in counts.values())
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{name}\n({100 * counts[g]['n'] / total:.0f}% of turns, n={counts[g]['n']:,})" for g, name in GROUPS],
                       fontsize=8, color=INK)


def fig_main_bars(data: dict, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    style(ax, "y")
    draw_bars(ax, data, MAIN_BARS, "test", 45)
    group_labels(ax, data, "test")
    ax.set_ylim(0, 45)
    ax.set_ylabel("accuracy (%), test split, 95% CI", color=INK2, fontsize=9)
    ax.set_title("Exact2x: edit tracking", color=INK, fontsize=11, loc="left")
    ax.legend(handles=[bar_handle(k) for k in MAIN_BARS], frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    save(fig, "main", stem)


def fig_appendix_bars(data: dict, stem: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), facecolor=SURFACE, sharey=True)
    for ax, split in zip(axes, SPLITS):
        style(ax, "y")
        draw_bars(ax, data, APPX_BARS, split, 45)
        group_labels(ax, data, split)
        ax.set_ylim(0, 47)
        ax.set_title(split, color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("accuracy (%), 95% CI for model conditions", color=INK2, fontsize=9)
    fig.suptitle("Exact2x, all Table 1 conditions: edit tracking  (stale-answer bar is 0 / 100 by definition; truncated)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.legend(handles=[bar_handle(k) for k in APPX_BARS], loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    save(fig, "appendix", stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-majority", action="store_true")
    args = parser.parse_args()
    data = load_all(args.rebuild_majority)
    fig_main_lines(data, "session_length", "fig1_accuracy_by_session_length")
    fig_main_lines(data, "turn_index", "fig2_accuracy_by_turn_index")
    fig_main_bars(data, "fig3_edit_tracking")
    fig_appendix_lines(data, "session_length", "figA1_all_conditions_by_session_length")
    fig_appendix_lines(data, "turn_index", "figA2_all_conditions_by_turn_index")
    fig_appendix_bars(data, "figA3_all_conditions_edit_tracking")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
