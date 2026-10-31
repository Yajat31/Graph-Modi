"""Report figures for the v4.1 runs (CPU only; needs matplotlib).

  PYTHONPATH=src python3 scripts/make_figures_v4.py --runs "$ART" --out documents/mid-report/figures

Reads the per-turn rows in <run>/evaluation_best.json (evaluation_final.json for the soft prompt)
and <run>/static_eval_test_best.json, joins them with the test sessions for task, session length
and answer-changing flags, and writes one figure set per model:

  <model>_fig1_accuracy_by_session_length.png   main conditions vs session length
  <model>_fig2_accuracy_by_turn_index.png       main conditions vs turn index
  <model>_fig3_edit_tracking.png                answer-changing vs unchanged turns
  <model>_figA1/A2/A3_all_conditions_*.png      the same with every condition (by_session_length,
                                                by_turn_index, edit_tracking)
  <model>_figA4_static_accuracy_by_task.png     static test per task vs the local ceiling
  v4_combined_{conditions,by_session_length,edit_tracking}.png   models side by side

<model> is graphtoken and tea (TEA-style: pretrained GNN, frozen; each the mean over the
s42/s43/s44 runs present), warm (warm-started GraphToken) and random (frozen random GNN).
Missing runs are skipped.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from graph_modi.data.multiturn import load_sessions  # noqa: E402
from graph_modi.evaluation.metrics import answers_match, normalize_answer  # noqa: E402

MODELS = {
    "graphtoken": ("GraphToken", ("v4_graphtoken_s42", "v4_graphtoken_s43", "v4_graphtoken_s44")),
    "warm": ("GraphToken, warm start", ("v4_graphtoken_warm_s42",)),
    "tea": ("TEA-style", ("v4_tea_s42", "v4_tea_s43", "v4_tea_s44")),
    "random": ("Frozen random GNN", ("v4_tea_random_sum",)),
}
SOFT_RUN = "v4_soft_prompt"

# Categorical slots in fixed order (reference palette, light mode); baselines stay gray.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#52514e"
LABELS = {
    "oracle_updated_graph": "Updated graph (oracle edits)",
    "predicted_updated_graph": "Updated graph (predicted edits)",
    "structure_only": "Structure only",
    "frozen_graph_history": "Frozen graph + history",
    "shuffled_graph": "Shuffled graph",
    "question_only": "Question only",
    "soft_prompt": "Soft prompt + history",
    "soft_prompt_no_history": "Soft prompt",
    "tool_solver": "Solver on current graph",
    "stale_answer": "Stale oracle",
    "majority_answer": "Per-task majority",
}
MAIN = [
    "oracle_updated_graph",
    "predicted_updated_graph",
    "frozen_graph_history",
    "shuffled_graph",
    "question_only",
    "soft_prompt",
]
ALL = MAIN[:2] + ["structure_only"] + MAIN[2:] + ["soft_prompt_no_history"]
REFERENCE = {"stale_answer": "--"}
TASK_ORDER = [
    "node_degree",
    "edge_exists",
    "reachability",
    "constrained_reachability",
    "cycle_membership",
    "shortest_path",
    "within_hops_count",
    "filtered_neighbor_count",
    "attribute_lookup",
    "attribute_check",
    "most_common_attribute_within_hops",
]

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#b5b4ad",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def turn_meta(sessions) -> dict[tuple[str, int], dict]:
    meta = {}
    for session in sessions:
        for turn in session.turns:
            meta[(session.session_id, turn.turn_index)] = {
                "task": turn.query.reasoning_type.value,
                "length": int(turn.complexity.get("session_length", len(session.turns))),
                "turn": turn.turn_index,
                "changed": normalize_answer(turn.gold_answer)
                != normalize_answer(turn.stale_answer),
            }
    return meta


def load_rows(run: Path, ckpt: str, meta: dict) -> dict[str, list[dict]]:
    path = run / f"evaluation_{ckpt}.json"
    if not path.exists():
        path = run / "evaluation_final.json"
    if not path.exists():
        return {}
    grouped = defaultdict(list)
    for row in json.loads(path.read_text())["test"]["rows"]:
        task = meta[(row["session_id"], row["turn_index"])]["task"]
        if row.get("raw_answer") is not None:  # rescore with the current scorer
            row["answer_correct"] = answers_match(
                row["gold_answer"], row["raw_answer"], reasoning_type=task
            )
        grouped[row["condition"]].append(row)
    return grouped


def accuracy_by(rows: list[dict], meta: dict, key) -> dict:
    buckets = defaultdict(list)
    for r in rows:
        k = key(meta[(r["session_id"], r["turn_index"])])
        if k is not None:
            buckets[k].append(bool(r["answer_correct"]))
    return {k: 100 * sum(v) / len(v) for k, v in buckets.items()}


def averaged(per_run: list[dict]) -> dict:
    """Mean and (min, max) over seed runs for each bucket present in all of them."""
    keys = set.intersection(*(set(d) for d in per_run)) if per_run else set()
    return {
        k: (
            statistics.fmean(d[k] for d in per_run),
            min(d[k] for d in per_run),
            max(d[k] for d in per_run),
        )
        for k in keys
    }


def model_curves(runs: list[dict], soft: dict, conditions, meta, key) -> dict[str, dict]:
    curves = {}
    for cond in conditions:
        source = [soft] if cond.startswith("soft_prompt") else runs
        per_run = [accuracy_by(r[cond], meta, key) for r in source if r.get(cond)]
        if per_run:
            curves[cond] = averaged(per_run)
    for cond in list(REFERENCE) + ["tool_solver", "majority_answer"]:
        per_run = [accuracy_by(r[cond], meta, key) for r in runs if r.get(cond)]
        if per_run:
            curves[cond] = averaged(per_run)
    return curves


def line_plot(
    curves, conditions, xs, xlabel, title, path, xticklabels=None, references=True
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, cond in enumerate(c for c in conditions if c in curves):
        c = curves[cond]
        x = [v for v in xs if v in c]
        mean = [c[v][0] for v in x]
        ax.plot(x, mean, color=SERIES[i], lw=2, marker="o", ms=4, label=LABELS[cond])
        low, high = [c[v][1] for v in x], [c[v][2] for v in x]
        if low != high:
            ax.fill_between(x, low, high, color=SERIES[i], alpha=0.15, lw=0)
    for cond, style in REFERENCE.items():
        if references and cond in curves:
            x = [v for v in xs if v in curves[cond]]
            y = [curves[cond][v][0] for v in x]
            ax.plot(x, y, color=GRAY, lw=1.2, ls=style, label=LABELS[cond])
    ax.set_xticks(xs, xticklabels or [str(v) for v in xs])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Answer accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def bar_plot(groups: dict[str, dict], series: list[tuple[str, str]], title, path, ylabel) -> None:
    """groups: {group label: {series key: (mean, low, high)}}; series: [(key, legend label)]."""
    names = list(groups)
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(max(6.4, 0.75 * len(names) + 2), 3.4))
    for i, (key, label) in enumerate(series):
        xs = [g + (i - (len(series) - 1) / 2) * width for g in range(len(names))]
        vals = [groups[n].get(key, (float("nan"),) * 3) for n in names]
        err = [[max(0.0, v[0] - v[1]) for v in vals], [max(0.0, v[2] - v[0]) for v in vals]]
        ax.bar(
            xs,
            [v[0] for v in vals],
            width * 0.92,
            color=SERIES[i],
            label=label,
            yerr=err if any(e > 0 for e in err[0] + err[1]) else None,
            error_kw={"lw": 0.8, "ecolor": GRAY, "capsize": 2},
        )
    ax.set_xticks(
        range(len(names)),
        names,
        rotation=30 if len(names) > 6 else 0,
        ha="right" if len(names) > 6 else "center",
    )
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def edit_tracking(curves, conditions, title, path) -> None:
    groups = {
        LABELS[c]: {
            "changed": curves[c].get(True, (float("nan"),) * 3),
            "unchanged": curves[c].get(False, (float("nan"),) * 3),
        }
        for c in conditions
        if c in curves
    }
    bar_plot(
        groups,
        [("changed", "Answer-changing turns"), ("unchanged", "Unchanged turns")],
        title,
        path,
        "Answer accuracy (%)",
    )


def static_by_task(runs: list[Path], soft: Path, ceilings: dict, title, path) -> None:
    def per_task(run: Path, ckpt: str) -> dict:
        p = run / f"static_eval_test_{ckpt}.json"
        if not p.exists():
            p = run / "static_eval_test_final.json"
        return {t: 100 * a for t, a in json.loads(p.read_text())["task_accuracy"].items()}

    model = averaged([per_task(r, "best") for r in runs])
    base = per_task(soft, "final") if (soft / "static_eval_test_final.json").exists() else {}
    groups = {}
    for task in TASK_ORDER:
        if task not in model:
            continue
        g = {"model": model[task]}
        if task in base:
            g["soft"] = (base[task],) * 3
        if task in ceilings:
            g["ceiling"] = (100 * ceilings[task],) * 3
        groups[task.replace("_", " ")] = g
    bar_plot(
        groups,
        [
            ("model", title.split(":")[0]),
            ("soft", "Soft prompt"),
            ("ceiling", "Local ceiling (1-/2-hop classifier)"),
        ],
        title,
        path,
        "Static test accuracy (%)",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="directory holding the v4_* run dirs")
    parser.add_argument("--data", default="datasets/metro_v4_1")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument(
        "--ceilings", default="documents/experiments/results/v4_1-analysis/shortcut_ceilings.json"
    )
    parser.add_argument("--out", default="documents/mid-report/figures")
    args = parser.parse_args()
    root, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = turn_meta(load_sessions(Path(args.data) / "sessions_test.jsonl"))
    ceilings = json.loads(Path(args.ceilings).read_text())["local_ceiling_per_task"]["static"]
    soft = load_rows(root / SOFT_RUN, "final", meta)
    lengths = sorted({m["length"] for m in meta.values()})
    turns = sorted({m["turn"] for m in meta.values()})
    by_length = lambda m: m["length"]  # noqa: E731
    by_turn = lambda m: m["turn"]  # noqa: E731
    by_changed = lambda m: m["changed"]  # noqa: E731

    combined: dict[str, dict] = {}
    for slug, (name, run_names) in MODELS.items():
        run_dirs = [root / r for r in run_names if (root / r / "evaluation_best.json").exists()]
        if not run_dirs:
            print(f"skip {slug}: no runs under {root}")
            continue
        runs = [load_rows(r, args.checkpoint, meta) for r in run_dirs]
        seeds = f" (mean of {len(runs)} seeds)" if len(runs) > 1 else ""
        for conds, prefix, tag in ((MAIN, "fig", "accuracy_"), (ALL, "figA", "all_conditions_")):
            n = {"fig": ("1", "2", "3"), "figA": ("A1", "A2", "A3")}[prefix]
            c_len = model_curves(runs, soft, conds, meta, by_length)
            line_plot(
                c_len,
                conds,
                lengths,
                "Session length (turns)",
                f"{name}: accuracy by session length{seeds}",
                out / f"{slug}_fig{n[0]}_{tag}by_session_length.png",
                references=False,
            )
            c_turn = model_curves(runs, soft, conds, meta, by_turn)
            line_plot(
                c_turn,
                conds,
                turns,
                "Turn index",
                f"{name}: accuracy by turn{seeds}",
                out / f"{slug}_fig{n[1]}_{tag}by_turn_index.png",
                [str(t + 1) for t in turns],
                references=False,
            )
            c_chg = model_curves(runs, soft, conds, meta, by_changed)
            edit_tracking(
                c_chg,
                conds,
                f"{name}: answer-changing vs unchanged turns{seeds}",
                out / f"{slug}_fig{n[2]}_{tag.replace('accuracy_', '')}edit_tracking.png",
            )
            if prefix == "fig":
                combined[name] = {
                    "length": c_len,
                    "changed": c_chg,
                    "all": model_curves(runs, soft, ALL, meta, lambda m: "all"),
                }
        static_by_task(
            run_dirs,
            root / SOFT_RUN,
            ceilings,
            f"{name}: static test accuracy by task",
            out / f"{slug}_figA4_static_accuracy_by_task.png",
        )
        print(f"wrote {slug} figures ({len(runs)} run(s))")

    if not combined:
        return
    oracle = "oracle_updated_graph"
    groups = {
        LABELS[c]: {m: v["all"][c]["all"] for m, v in combined.items() if c in v["all"]}
        for c in ALL
    }
    bar_plot(
        groups,
        [(m, m) for m in combined],
        "Test-session accuracy by condition",
        out / "v4_combined_conditions.png",
        "Answer accuracy (%)",
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (m, v) in enumerate(combined.items()):
        c = v["length"][oracle]
        ax.plot(
            lengths, [c[x][0] for x in lengths], color=SERIES[i], lw=2, marker="o", ms=4, label=m
        )
    soft_len = combined[next(iter(combined))]["length"].get("soft_prompt")
    if soft_len:
        ax.plot(
            lengths,
            [soft_len[x][0] for x in lengths],
            color=GRAY,
            lw=1.2,
            ls="--",
            label=LABELS["soft_prompt"],
        )
    ax.set_xticks(lengths, [str(x) for x in lengths])
    ax.set_xlabel("Session length (turns)")
    ax.set_ylabel("Answer accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Updated graph (oracle edits) by session length", loc="left", fontsize=10)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.savefig(out / "v4_combined_by_session_length.png")
    plt.close(fig)
    groups = {
        m: {"changed": v["changed"][oracle][True], "unchanged": v["changed"][oracle][False]}
        for m, v in combined.items()
    }
    bar_plot(
        groups,
        [("changed", "Answer-changing turns"), ("unchanged", "Unchanged turns")],
        "Updated graph (oracle edits): answer-changing vs unchanged turns",
        out / "v4_combined_edit_tracking.png",
        "Answer accuracy (%)",
    )
    print("wrote combined figures")


if __name__ == "__main__":
    main()
