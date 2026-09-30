"""Figures for documents/experiments/status_report_fixed_final.md.

Reads documents/experiments/results/fixed_final-20260929/ (analysis + static eval JSONs) and
writes documents/experiments/figures/ff_*.png. soft_prompt files are optional.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "documents/experiments/results/fixed_final-20260929"
FIG = ROOT / "documents/experiments/figures"

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"TEA": "#2a78d6", "GraphToken": "#eb6834", "soft_prompt": "#1baf7a"}
DENSITY_COLORS = {"sparse": "#86b6ef", "medium": "#2a78d6", "dense": "#104281"}
TASKS = [
    "edge_exists",
    "reachability",
    "cycle_membership",
    "constrained_reachability",
    "filtered_neighbor_count",
    "node_count",
    "within_hops_count",
    "most_common_attribute_within_hops",
]
MODEL_CONDITIONS = [
    "oracle_updated_graph",
    "predicted_updated_graph",
    "modify_and_print",
    "cached_no_reencode",
    "frozen_graph_history",
    "structure_only",
    "shuffled_graph",
    "graph_once_then_text",
    "token_matched_history",
    "question_only",
    "serialized_initial_history",
    "serialized_current_graph",
]


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def load(name: str) -> dict | None:
    path = RES / name
    return json.loads(path.read_text()) if path.exists() else None


def cut(analysis: dict, split: str, cond: str, dim: str) -> dict:
    return analysis[split]["cuts"][f"{cond}::{dim}"]


def pct(value: float | None) -> float:
    return 100 * value if value is not None else float("nan")


def short(task: str) -> str:
    return task.replace("_", " ").replace("most common attribute within hops", "most common attr.")


def models() -> dict[str, dict]:
    out = {}
    for label, file in (("TEA", "exact2x_tea_analysis.json"), ("GraphToken", "exact2x_graphtoken_analysis.json")):
        data = load(file)
        if data:
            out[label] = data
    return out


def fig_conditions(split: str = "test") -> None:
    data = models()
    soft = load("exact2x_soft_prompt_analysis.json")
    fig, ax = plt.subplots(figsize=(8.2, 5.6), facecolor=SURFACE)
    style(ax)
    height = 0.38
    for i, cond in enumerate(MODEL_CONDITIONS):
        for j, (name, analysis) in enumerate(data.items()):
            value = pct(cut(analysis, split, cond, "overall")["all"]["acc"])
            y = i + (j - 0.5) * height
            ax.barh(y, value, height=height - 0.04, color=COLORS[name], label=name if i == 0 else None)
            ax.text(value + 0.5, y, f"{value:.1f}", va="center", fontsize=7, color=INK2)
    if soft:
        value = pct(cut(soft, split, "soft_prompt", "overall")["all"]["acc"])
        ax.axvline(value, color=COLORS["soft_prompt"], linestyle="--", linewidth=1.4, label=f"soft_prompt ({value:.1f})")
    ax.set_yticks(range(len(MODEL_CONDITIONS)))
    ax.set_yticklabels([c.replace("_", " ") for c in MODEL_CONDITIONS], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 43)
    ax.set_xlabel("answer accuracy (%)", color=INK2, fontsize=9)
    ax.set_title(f"Exact2x, {split} split: accuracy by evaluation condition", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(0.01, 0.005, "Excluded: tool_solver (100%) and stale-answer baseline (68%).", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / f"ff_conditions_{split}.png", dpi=160)
    plt.close(fig)


def fig_graph_gain(split: str = "test") -> None:
    data = models()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), facecolor=SURFACE, sharey=True)
    panels = (
        ("structure_only", "question_only", "Graph tokens vs. no graph tokens\n(structure_only - question_only)"),
        ("oracle_updated_graph", "shuffled_graph", "Correct vs. wrong session's graph\n(oracle_updated_graph - shuffled_graph)"),
    )
    height = 0.38
    for ax, (a, b, title) in zip(axes, panels):
        style(ax)
        for j, (name, analysis) in enumerate(data.items()):
            for i, task in enumerate(TASKS):
                delta = pct(cut(analysis, split, a, "task")[task]["acc"]) - pct(cut(analysis, split, b, "task")[task]["acc"])
                y = i + (j - 0.5) * height
                ax.barh(y, delta, height=height - 0.04, color=COLORS[name], label=name if i == 0 else None)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("accuracy difference (percentage points)", color=INK2, fontsize=8)
    axes[0].set_yticks(range(len(TASKS)))
    axes[0].set_yticklabels([short(t) for t in TASKS], fontsize=8, color=INK)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Exact2x, {split} split: how much do the graph tokens actually contribute?", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / f"ff_graph_gain_{split}.png", dpi=160)
    plt.close(fig)


def fig_task_density(split: str = "test", cond: str = "oracle_updated_graph") -> None:
    data = dict(models())
    entries = [(name, analysis, cond) for name, analysis in data.items()]
    soft = load("exact2x_soft_prompt_analysis.json")
    if soft:
        entries.append(("soft_prompt (no graph)", soft, "soft_prompt"))
    fig, axes = plt.subplots(1, len(entries), figsize=(5.0 * len(entries), 4.8), facecolor=SURFACE, sharey=True, sharex=True)
    axes = [axes] if len(entries) == 1 else list(axes)
    width = 0.26
    for ax, (name, analysis, c) in zip(axes, entries):
        style(ax)
        table = cut(analysis, split, c, "task|density")
        for j, dens in enumerate(("sparse", "medium", "dense")):
            for i, task in enumerate(TASKS):
                value = pct(table[f"{task}|{dens}"]["acc"])
                y = i + (j - 1) * width
                ax.barh(y, value, height=width - 0.03, color=DENSITY_COLORS[dens], label=dens if i == 0 else None)
        ax.set_title(name, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("answer accuracy (%)", color=INK2, fontsize=8)
    axes[0].set_yticks(range(len(TASKS)))
    axes[0].set_yticklabels([short(t) for t in TASKS], fontsize=8, color=INK)
    axes[0].invert_yaxis()
    axes[-1].legend(title="graph density", frameon=False, fontsize=8, title_fontsize=8, loc="lower right")
    fig.suptitle(f"Exact2x, {split} split: accuracy by task and graph density (graph models: {cond})", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / f"ff_task_density_{split}.png", dpi=160)
    plt.close(fig)


def fig_pathcost(split: str = "test") -> None:
    series = {}
    for name, key in (("TEA", "tea"), ("GraphToken", "graphtoken"), ("soft_prompt", "soft_prompt")):
        data = load(f"pathcost_{key}_analysis.json")
        if data:
            series[name] = data
    if not series:
        return
    conds = ["oracle_updated_graph", "frozen_graph_history", "structure_only", "question_only", "majority_prior"]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style(ax)
    height = 0.8 / max(1, len(series))
    for j, (name, analysis) in enumerate(series.items()):
        use = ["soft_prompt"] if name == "soft_prompt" else conds
        for cond in use:
            i = conds.index(cond) if cond in conds else len(conds)
            value = pct(cut(analysis, split, cond, "overall")["all"]["acc"])
            y = i + (j - (len(series) - 1) / 2) * height
            ax.barh(y, value, height=height - 0.03, color=COLORS[name], label=name if (cond == use[0]) else None)
            ax.text(value + 0.5, y, f"{value:.1f}", va="center", fontsize=7, color=INK2)
    labels = [("stale-answer baseline" if c == "majority_prior" else c.replace("_", " ")) for c in conds] + ["soft_prompt"]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("answer accuracy (%)", color=INK2, fontsize=9)
    ax.set_title(f"path_cost supplement, {split} split", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(0.01, 0.005, "question_only always answers 'unreachable' (29.7% of gold answers); stale-answer baseline shown for reference.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / f"ff_pathcost_{split}.png", dpi=160)
    plt.close(fig)


def fig_static(split: str = "test") -> None:
    series = {}
    for name, key in (("TEA", "tea"), ("GraphToken", "graphtoken"), ("soft_prompt", "soft_prompt")):
        data = load(f"static_{key}_{split}.json")
        if data:
            series[name] = data["task_accuracy"]
    tasks = sorted(next(iter(series.values())), key=lambda t: -series["TEA"][t])
    fig, ax = plt.subplots(figsize=(8.2, 5.2), facecolor=SURFACE)
    style(ax)
    height = 0.8 / len(series)
    for j, (name, table) in enumerate(series.items()):
        for i, task in enumerate(tasks):
            value = 100 * table[task]
            y = i + (j - (len(series) - 1) / 2) * height
            ax.barh(y, value, height=height - 0.03, color=COLORS[name], label=name if i == 0 else None)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([short(t) for t in tasks], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("static oracle-QA accuracy (%)", color=INK2, fontsize=9)
    ax.set_title(f"Static eval, {split} split: accuracy by task", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / f"ff_static_{split}.png", dpi=160)
    plt.close(fig)


def fig_diameter() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    import networkx as nx

    from graph_modi.data.v2 import TopologyFamily, generate_session_v2
    from graph_modi.schema import DensityBin, ReasoningType

    cache = RES / "diameter_by_density.json"
    if cache.exists():
        stats = json.loads(cache.read_text())
    else:
        stats = {}
        for density in (DensityBin.SPARSE, DensityBin.MEDIUM, DensityBin.DENSE):
            stats[density.value] = {}
            for n in range(16, 49, 4):
                diameters = []
                for seed in range(30):
                    session = generate_session_v2(
                        split="test", index=seed, seed=seed * 7 + n, node_count=n, turn_count=1,
                        topology=TopologyFamily.WATTS_STROGATZ,
                        reasoning_types=(ReasoningType.REACHABILITY,),
                        scale_bin="scale_small", target_density=density,
                    )
                    graph = session.initial_graph
                    g = nx.Graph()
                    g.add_nodes_from(node.id for node in graph.nodes)
                    g.add_edges_from((e.source, e.target) for e in graph.edges)
                    if not nx.is_connected(g):
                        g = g.subgraph(max(nx.connected_components(g), key=len))
                    diameters.append(nx.diameter(g))
                stats[density.value][str(n)] = statistics.mean(diameters)
        cache.write_text(json.dumps(stats, indent=1))
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for dens, color in DENSITY_COLORS.items():
        xs = sorted(int(n) for n in stats[dens])
        ax.plot(xs, [stats[dens][str(n)] for n in xs], color=color, linewidth=2, marker="o", markersize=5, label=dens)
    ax.axhline(3, color=INK2, linestyle="--", linewidth=1)
    ax.text(16.3, 3.6, "3-layer GraphSAGE receptive field (3 hops)", fontsize=8, color=INK2)
    ax.set_xlabel("number of nodes", color=INK2, fontsize=9)
    ax.set_ylabel("mean graph diameter (hops)", color=INK2, fontsize=9)
    ax.set_title("Graph diameter vs. GNN receptive field", color=INK, fontsize=11, loc="left")
    ax.legend(title="density", frameon=False, fontsize=8, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "ff_diameter.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    for split in ("test", "validation"):
        fig_conditions(split)
        fig_graph_gain(split)
        fig_task_density(split)
        fig_static(split)
        fig_pathcost(split)
    fig_diameter()
    print("figures written to", FIG)
