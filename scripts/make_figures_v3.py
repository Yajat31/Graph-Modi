"""Figures for the v3 benchmark results (static + exact2x) -> documents/experiments/figures/v3_*.png.

Reads documents/experiments/results/v3-20260929/. GraphToken's exact2x file is optional and is
picked up automatically once it exists.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "documents/experiments/results/v3-20260929"
FIG = ROOT / "documents/experiments/figures"

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"soft_prompt": "#1baf7a", "TEA": "#2a78d6", "GraphToken": "#eb6834"}
MODELS = (("soft_prompt", "soft_prompt"), ("TEA", "tea"), ("GraphToken", "graphtoken"))
LINE_STYLE = {
    "oracle_updated_graph": ("#104281", "-"),
    "frozen_graph_history": ("#5598e7", "-"),
    "shuffled_graph": ("#c98500", "-."),
    "question_only": ("#8b8a85", "--"),
    "soft_prompt": ("#1baf7a", "-"),
}


def style(ax, grid_axis="x") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def short(task: str) -> str:
    return task.replace("_", " ").replace("most common attribute within hops", "most common attr.")


def se(p: float, n: int) -> float:
    return 100 * math.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))


def load(name: str):
    path = RES / name
    return json.loads(path.read_text()) if path.exists() else None


def static_results(split: str) -> dict[str, dict]:
    return {label: load(f"static_{key}_{split}.json") for label, key in MODELS if load(f"static_{key}_{split}.json")}


def exact2x() -> dict[str, dict]:
    return {label: load(f"exact2x_{key}_analysis.json") for label, key in MODELS if load(f"exact2x_{key}_analysis.json")}


def condition_of(label: str, name: str) -> str:
    return "soft_prompt" if label == "soft_prompt" else name


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ static
def fig_static_by_task() -> None:
    results = static_results("test")
    audit = load("dataset_audit.json")["static_test"]["text_leak_audit"]
    tasks = sorted(next(iter(results.values()))["task_accuracy"], key=lambda t: -results["GraphToken"]["task_accuracy"][t])
    fig, ax = plt.subplots(figsize=(8.6, 5.6), facecolor=SURFACE)
    style(ax)
    height = 0.8 / len(results)
    for j, (label, data) in enumerate(results.items()):
        for i, task in enumerate(tasks):
            n = sum(1 for r in data["rows"] if r["reasoning_type"] == task)
            value = 100 * data["task_accuracy"][task]
            y = i + (j - (len(results) - 1) / 2) * height
            ax.barh(y, value, height=height - 0.03, color=COLORS[label], xerr=1.96 * se(value / 100, n),
                    error_kw={"ecolor": INK2, "elinewidth": 0.7, "capsize": 1.5}, label=label if i == 0 else None)
    for i, task in enumerate(tasks):
        ax.plot([100 * audit[task]["question_only_rule_accuracy"]] * 2, [i - 0.42, i + 0.42], color=INK, linewidth=1.6,
                label="question-only prior" if i == 0 else None)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([short(t) for t in tasks], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("accuracy (%), static test split, 95% CI", color=INK2, fontsize=9)
    ax.set_title("Static eval: accuracy by task", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "v3_static_by_task.png")


def fig_static_by_group() -> None:
    results = static_results("test")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE, sharey=True)
    for ax, key, order, title in (
        (axes[0], "density_bin", ("sparse", "medium", "dense"), "by graph density"),
        (axes[1], "scale_bin", ("scale_small", "scale_medium", "scale_large"), "by graph size (12-15 / 16-19 / 20-22 stations)"),
    ):
        style(ax, "y")
        width = 0.8 / len(results)
        for j, (label, data) in enumerate(results.items()):
            for i, group in enumerate(order):
                rows = [r["correct"] for r in data["rows"] if r[key] == group]
                p = sum(rows) / len(rows)
                x = i + (j - (len(results) - 1) / 2) * width
                ax.bar(x, 100 * p, width=width - 0.03, color=COLORS[label], yerr=1.96 * se(p, len(rows)),
                       error_kw={"ecolor": INK2, "elinewidth": 0.7, "capsize": 1.5}, label=label if i == 0 else None)
        ax.axhline(33.1, color=INK, linestyle=":", linewidth=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([o.replace("scale_", "") for o in order], fontsize=8, color=INK)
        ax.set_title(f"Static test accuracy {title}", color=INK, fontsize=10, loc="left")
        ax.set_ylim(0, 48)
    axes[0].set_ylabel("accuracy (%)", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].text(0.02, 34.2, "question-only prior 33.1%", fontsize=7, color=INK2, transform=axes[0].get_yaxis_transform())
    fig.tight_layout()
    save(fig, "v3_static_by_density_scale.png")


# ------------------------------------------------------------------ exact2x
def fig_exact2x_conditions() -> None:
    data = exact2x()
    graph_models = [m for m in ("TEA", "GraphToken") if m in data]
    conds = ["oracle_updated_graph", "structure_only", "frozen_graph_history", "cached_no_reencode",
             "shuffled_graph", "graph_once_then_text", "token_matched_history", "question_only"]
    audit = load("dataset_audit.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE, sharey=True)
    for ax, split in zip(axes, ("validation", "test")):
        style(ax)
        height = 0.8 / len(graph_models)
        for j, model in enumerate(graph_models):
            for i, cond in enumerate(conds):
                s = data[model][split]["summary"][cond]
                value = 100 * s["answer_accuracy"]
                y = i + (j - (len(graph_models) - 1) / 2) * height
                ax.barh(y, value, height=height - 0.04, color=COLORS[model],
                        xerr=[[value - 100 * s["answer_accuracy_ci95"][0]], [100 * s["answer_accuracy_ci95"][1] - value]],
                        error_kw={"ecolor": INK2, "elinewidth": 0.7, "capsize": 1.5}, label=model if i == 0 else None)
        if "soft_prompt" in data:
            sp = 100 * data["soft_prompt"][split]["summary"]["soft_prompt"]["answer_accuracy"]
            ax.axvline(sp, color=COLORS["soft_prompt"], linestyle="--", linewidth=1.5, label=f"soft_prompt (history text) {sp:.1f}")
        ax.axvline(100 * audit[split]["labels"]["_overall_majority_accuracy"], color=INK, linestyle=":", linewidth=1.2, label="majority baseline")
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels([c.replace("_", " ") for c in conds], fontsize=8, color=INK)
        ax.set_xlim(20, 40)
        ax.set_xlabel("accuracy (%)", color=INK2, fontsize=9)
        ax.set_title(f"Exact2x {split}", color=INK, fontsize=10, loc="left")
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Exact2x (multi-turn, edited graphs): accuracy by evaluation condition", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "v3_exact2x_conditions.png")


def line_series(data: dict, dim: str, split: str = "test") -> list[tuple[str, str, str, dict]]:
    series = []
    for model in ("TEA", "GraphToken"):
        if model not in data:
            continue
        for cond in ("oracle_updated_graph", "frozen_graph_history", "shuffled_graph", "question_only"):
            series.append((f"{model} {cond.replace('_', ' ')}", model, cond, data[model][split]["cuts"][f"{cond}::{dim}"]))
    if "soft_prompt" in data:
        series.append(("soft_prompt (history text)", "soft_prompt", "soft_prompt", data["soft_prompt"][split]["cuts"][f"soft_prompt::{dim}"]))
    return series


def draw_lines(ax, data: dict, dim: str, order: list[str], split: str, only_tea_and_soft: bool = False) -> None:
    for label, model, cond, cut in line_series(data, dim, split):
        if model == "GraphToken" and cond != "oracle_updated_graph":
            continue
        if cond in ("question_only",) and model == "GraphToken":
            continue
        color, dash = LINE_STYLE[cond if cond in LINE_STYLE else "oracle_updated_graph"]
        if model == "GraphToken":
            color = COLORS["GraphToken"]
        keys = [k for k in order if k in cut]
        ax.plot([int(k) for k in keys], [100 * cut[k]["acc"] for k in keys], color=color, linestyle=dash,
                marker="o", markersize=3.5, linewidth=1.8, label=label)


def fig_exact2x_lines(dim: str, order: list[str], xlabel: str, title: str, filename: str) -> None:
    data = exact2x()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE, sharey=True)
    for ax, split in zip(axes, ("validation", "test")):
        style(ax, "both")
        draw_lines(ax, data, dim, order, split)
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
        ax.set_title(f"{split}", color=INK, fontsize=10, loc="left")
        ax.set_ylim(20, 42)
        ax.set_xticks([int(k) for k in order])
    axes[0].set_ylabel("accuracy (%)", color=INK2, fontsize=9)
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    if dim == "turn_index":
        fig.text(0.01, 0.005, "Sessions reaching each turn: turn 0 = 1,188; turn 1 = 891; turns 2-3 = 594; turns 4-7 = 297 "
                 "(about +/-5 points at the 95% level, so late-turn wiggles are mostly noise).", fontsize=7.5, color=INK2)
        fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, filename)


def fig_exact2x_by_task() -> None:
    data = exact2x()
    tasks = sorted(data["TEA"]["test"]["cuts"]["oracle_updated_graph::task"])
    series = [("TEA updated graph", "TEA", "oracle_updated_graph"), ("TEA frozen graph", "TEA", "frozen_graph_history"),
              ("TEA shuffled graph", "TEA", "shuffled_graph")]
    if "GraphToken" in data:
        series.append(("GraphToken updated graph", "GraphToken", "oracle_updated_graph"))
    if "soft_prompt" in data:
        series.append(("soft_prompt (history text)", "soft_prompt", "soft_prompt"))
    colors = {"TEA updated graph": "#104281", "TEA frozen graph": "#86b6ef", "TEA shuffled graph": "#eb6834",
              "GraphToken updated graph": "#eb6834", "soft_prompt (history text)": COLORS["soft_prompt"]}
    colors["TEA shuffled graph"] = "#8b8a85"
    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
    style(ax)
    height = 0.84 / len(series)
    for j, (label, model, cond) in enumerate(series):
        for i, task in enumerate(tasks):
            cut = data[model]["test"]["cuts"][f"{cond}::task"][task]
            y = i + (j - (len(series) - 1) / 2) * height
            ax.barh(y, 100 * cut["acc"], height=height - 0.03, color=colors[label], label=label if i == 0 else None)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([short(t) for t in tasks], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.axvline(50, color=INK2, linestyle=":", linewidth=0.8)
    ax.set_xlabel("accuracy (%), test split (yes/no chance = 50)", color=INK2, fontsize=9)
    ax.set_title("Exact2x test: accuracy by task", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "v3_exact2x_by_task.png")


def fig_exact2x_density() -> None:
    data = exact2x()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=SURFACE, sharey=True)
    series = [("TEA updated graph", "TEA", "oracle_updated_graph", "#104281"), ("TEA frozen graph", "TEA", "frozen_graph_history", "#86b6ef")]
    if "GraphToken" in data:
        series.append(("GraphToken updated graph", "GraphToken", "oracle_updated_graph", COLORS["GraphToken"]))
    if "soft_prompt" in data:
        series.append(("soft_prompt (history text)", "soft_prompt", "soft_prompt", COLORS["soft_prompt"]))
    for ax, (dim, order, title) in zip(axes, (("density", ["sparse", "medium", "dense"], "by graph density"),
                                             ("scale", ["scale_small", "scale_medium", "scale_large"], "by graph size (12-15 / 16-19 / 20-22)"))):
        style(ax, "y")
        width = 0.8 / len(series)
        for j, (label, model, cond, color) in enumerate(series):
            cut = data[model]["test"]["cuts"][f"{cond}::{dim}"]
            for i, group in enumerate(order):
                x = i + (j - (len(series) - 1) / 2) * width
                ax.bar(x, 100 * cut[group]["acc"], width=width - 0.03, color=color, yerr=1.96 * se(cut[group]["acc"], cut[group]["n"]),
                       error_kw={"ecolor": INK2, "elinewidth": 0.7, "capsize": 1.5}, label=label if i == 0 else None)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([o.replace("scale_", "") for o in order], fontsize=8, color=INK)
        ax.set_ylim(20, 42)
        ax.set_title(f"Exact2x test accuracy {title}", color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("accuracy (%)", color=INK2, fontsize=9)
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "v3_exact2x_by_density_size.png")


def fig_exact2x_edit_tracking() -> None:
    data = exact2x()
    series = [("TEA updated graph", "TEA", "oracle_updated_graph", "#104281"), ("TEA frozen graph", "TEA", "frozen_graph_history", "#86b6ef"),
              ("TEA no graph tokens", "TEA", "question_only", "#8b8a85")]
    if "GraphToken" in data:
        series.append(("GraphToken updated graph", "GraphToken", "oracle_updated_graph", COLORS["GraphToken"]))
    if "soft_prompt" in data:
        series.append(("soft_prompt (history text)", "soft_prompt", "soft_prompt", COLORS["soft_prompt"]))
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=SURFACE)
    style(ax, "y")
    width = 0.8 / len(series)
    for j, (label, model, cond, color) in enumerate(series):
        cut = data[model]["test"]["cuts"][f"{cond}::changed"]
        for i, group in enumerate(("answer_changed", "answer_unchanged")):
            x = i + (j - (len(series) - 1) / 2) * width
            ax.bar(x, 100 * cut[group]["acc"], width=width - 0.03, color=color, yerr=1.96 * se(cut[group]["acc"], cut[group]["n"]),
                   error_kw={"ecolor": INK2, "elinewidth": 0.7, "capsize": 1.5}, label=label if i == 0 else None)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["turns where the edit changed the answer (41%)", "turns where the answer did not change (59%)"], fontsize=8, color=INK)
    ax.set_ylabel("accuracy (%), test split", color=INK2, fontsize=9)
    ax.set_title("Exact2x: edit tracking", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    save(fig, "v3_exact2x_edit_tracking.png")


def fig_dataset() -> None:
    audit = load("dataset_audit.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    style(ax)
    static = audit["static_test"]["labels"]
    dynamic = audit["test"]["labels"]
    tasks = sorted(t for t in static if not t.startswith("_"))
    for j, (name, labels, color) in enumerate((("static test", static, "#5598e7"), ("exact2x test", dynamic, "#104281"))):
        for i, task in enumerate(tasks):
            if task in labels:
                y = i + (j - 0.5) * 0.38
                ax.barh(y, 100 * labels[task]["majority_accuracy"], height=0.34, color=color, label=name if task == "edge_exists" else None)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([short(t) for t in tasks], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.axvline(50, color=INK2, linestyle=":", linewidth=0.8)
    ax.set_xlabel("majority-answer accuracy (%) - yes/no tasks are 50 by construction", color=INK2, fontsize=8)
    ax.set_title("Label balance per task", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[1]
    style(ax, "y")
    hist = audit["test"]["graphs"]["node_count_histogram"]
    ax.bar([int(k) for k in hist], list(hist.values()), color="#5598e7")
    ax.set_xlabel("stations in the session's initial graph", color=INK2, fontsize=9)
    ax.set_ylabel("sessions (test)", color=INK2, fontsize=9)
    ax.set_title("Graph sizes (12-22, exact grid)", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    save(fig, "v3_dataset_composition.png")


RES_X = ROOT / "documents/experiments/results/v3x-20260929"


def fig_predicted_edits() -> None:
    """Model-written edits (parsed, applied, graph re-encoded) vs the oracle edits, from the same run."""
    runs = [(label, load(f"exact2x_{key}_predfix_analysis.json")) for label, key in (("TEA", "tea"), ("GraphToken", "graphtoken"))]
    runs = [(label, data) for label, data in runs if data]
    kinds = load("predfix_tea_edit_kinds_test.json")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.1, 1.3, 1.3]})
    ax = axes[0]
    style(ax, "y")
    width = 0.8 / (2 * len(runs))
    for j, (label, data) in enumerate(runs):
        for k, (cond, hatch, name) in enumerate((("oracle_updated_graph", None, "oracle edits"), ("predicted_updated_graph", "//", "model-written edits"))):
            for i, split in enumerate(("validation", "test")):
                s = data[split]["summary"][cond]
                x = i + (j * 2 + k - (2 * len(runs) - 1) / 2) * width
                ax.bar(x, 100 * s["answer_accuracy"], width=width - 0.02, color=COLORS[label], hatch=hatch, alpha=1.0 if hatch is None else 0.7,
                       yerr=[[100 * (s["answer_accuracy"] - s["answer_accuracy_ci95"][0])], [100 * (s["answer_accuracy_ci95"][1] - s["answer_accuracy"])]],
                       error_kw={"ecolor": INK2, "elinewidth": 0.7, "capsize": 1.5}, label=f"{label}, {name}" if i == 0 else None)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["validation", "test"], fontsize=9, color=INK)
    ax.set_ylim(25, 43)
    ax.set_ylabel("answer accuracy (%)", color=INK2, fontsize=9)
    ax.set_title("Model-written vs oracle edits", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)
    ax = axes[1]
    style(ax, "both")
    tea = runs[0][1]
    for cond, color, dash, name in (("oracle_updated_graph", "#104281", "-", "oracle edits"), ("predicted_updated_graph", "#eb6834", "--", "model-written edits")):
        cut = tea["test"]["cuts"][f"{cond}::session_length"]
        ax.plot([1, 2, 4, 8], [100 * cut[str(k)]["acc"] for k in (1, 2, 4, 8)], color=color, linestyle=dash, marker="o", linewidth=1.8, label=f"TEA {name}")
    ax.set_xticks([1, 2, 4, 8])
    ax.set_ylim(25, 40)
    ax.set_xlabel("session length (turns)", color=INK2, fontsize=9)
    ax.set_title("Errors would accumulate in long sessions - they do not", color=INK, fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax = axes[2]
    style(ax)
    if kinds:
        order = sorted(kinds["by_kind"], key=lambda k: -kinds["by_kind"][k]["n"])
        values = [100 * kinds["by_kind"][k]["correct"] / kinds["by_kind"][k]["n"] for k in order]
        ax.barh(range(len(order)), values, color="#5598e7")
        for i, (k, v) in enumerate(zip(order, values)):
            ax.text(v + 0.5, i, f"{v:.0f}%  (n={kinds['by_kind'][k]['n']})", va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([k.replace("NODE", "node").replace("EDGE", "edge") for k in order], fontsize=8, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, 120)
    ax.set_xlabel("edit reproduced exactly (%) - TEA, test", color=INK2, fontsize=9)
    ax.set_title("Edit accuracy by edit kind (frozen Llama, few-shot)", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    save(fig, "v3_predicted_edits.png")


def fig_variants() -> None:
    names = [("base3", "baseline\n3 layers, linear\nprojector, 3 epochs"), ("deep", "6-layer GNN"), ("proj", "MLP projector\n3x4096, lr 1e-3\n(collapsed)"),
             ("proj2", "MLP projector\n2x2048, lr 2e-4"), ("data", "3x data\n(1 epoch)")]
    static = {k: json.loads((RES_X / f"static_{k}_test.json").read_text()) for k, _ in names if (RES_X / f"static_{k}_test.json").exists()}
    exact = {k: json.loads((RES_X / f"exact2x_{k}_analysis.json").read_text()) for k, _ in names if (RES_X / f"exact2x_{k}_analysis.json").exists()}
    audit = load("dataset_audit.json")["static_test"]["text_leak_audit"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.5, 1]})
    keys = [k for k, _ in names if k in static]
    labels = dict(names)
    ax = axes[0]
    style(ax, "y")
    ax.bar(range(len(keys)), [100 * static[k]["overall_accuracy"] for k in keys], color=["#104281" if k == "base3" else "#8b8a85" for k in keys])
    for i, k in enumerate(keys):
        ax.text(i, 100 * static[k]["overall_accuracy"] + 0.5, f"{100 * static[k]['overall_accuracy']:.1f}", ha="center", fontsize=8, color=INK2)
    ax.axhline(33.1, color=INK, linestyle=":", linewidth=1, label="question-only prior (33.1)")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([labels[k].replace("\n", "\n") for k in keys], fontsize=6.5, color=INK)
    ax.set_ylim(25, 43)
    ax.set_ylabel("accuracy (%)", color=INK2, fontsize=9)
    ax.set_title("Static test, overall", color=INK, fontsize=10, loc="left")
    ax = axes[1]
    style(ax, "y")
    tasks = ["reachability", "constrained_reachability", "attribute_lookup", "most_common_attribute_within_hops"]
    palette = ["#104281", "#86b6ef", "#eb6834", "#1baf7a", "#8b8a85"]
    width = 0.8 / len(keys)
    for j, k in enumerate(keys):
        ax.bar([i + (j - (len(keys) - 1) / 2) * width for i in range(len(tasks))], [100 * static[k]["task_accuracy"][t] for t in tasks],
               width=width - 0.02, color=palette[j], label=labels[k].replace("\n", " "))
    for i, t in enumerate(tasks):
        ax.plot([i - 0.42, i + 0.42], [100 * audit[t]["question_only_rule_accuracy"]] * 2, color=INK, linewidth=1.6, label="question-only prior" if i == 0 else None)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([short(t) for t in tasks], fontsize=8, color=INK)
    ax.set_ylabel("accuracy (%), static test", color=INK2, fontsize=9)
    ax.set_ylim(0, 88)
    ax.set_title("The tasks where the baseline had signal", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2)
    ax = axes[2]
    style(ax, "y")
    ekeys = [k for k in keys if k in exact]
    width = 0.8 / 3
    for j, (cond, color, name) in enumerate((("oracle_updated_graph", "#104281", "updated graph"), ("frozen_graph_history", "#86b6ef", "frozen graph"), ("shuffled_graph", "#8b8a85", "other session's graph"))):
        ax.bar([i + (j - 1) * width for i in range(len(ekeys))], [100 * exact[k]["test"]["summary"][cond]["answer_accuracy"] for k in ekeys],
               width=width - 0.02, color=color, label=name)
    ax.set_xticks(range(len(ekeys)))
    short_names = {"base3": "baseline", "deep": "6-layer\nGNN", "proj": "MLP 3x4096\n(collapsed)", "proj2": "MLP 2x2048", "data": "3x data"}
    ax.set_xticklabels([short_names[k] for k in ekeys], fontsize=7, color=INK)
    ax.set_ylim(25, 40)
    ax.set_title("Exact2x test, overall", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Does a deeper GNN, a larger projector or more data help TEA?  (no)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "v3x_variants.png")


if __name__ == "__main__":
    fig_dataset()
    fig_static_by_task()
    fig_static_by_group()
    fig_exact2x_conditions()
    fig_exact2x_lines("turn_index", [str(i) for i in range(8)], "turn index within the session (turn 0 = first question)",
                      "Exact2x: accuracy by turn index (later turns only exist in 8-turn sessions)", "v3_exact2x_by_turn.png")
    fig_exact2x_lines("session_length", ["1", "2", "4", "8"], "session length (number of turns)",
                      "Exact2x: accuracy by session length", "v3_exact2x_by_session_length.png")
    fig_exact2x_lines("n_nodes", [str(n) for n in range(12, 23)], "stations in the graph",
                      "Exact2x: accuracy by graph size", "v3_exact2x_by_graph_size.png")
    fig_exact2x_by_task()
    fig_exact2x_density()
    fig_exact2x_edit_tracking()
    fig_predicted_edits()
    fig_variants()
    print("figures written to", FIG)
