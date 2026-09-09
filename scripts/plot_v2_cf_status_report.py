#!/usr/bin/env python3
"""Generate STATUS_REPORT figures for v2_gate_variant_cf-20260823.

Reads TEA / GraphToken / soft_prompt evaluation dumps, joins session complexity
metadata, writes PNGs under documents/experiments/figures/, and prints a small
cross-check summary against the canonical report numbers.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "metro_v2_gate_variant_cf"
OUT_TEA = ROOT / "outputs" / "v2_gate_variant_cf_tea"
OUT_GT = ROOT / "outputs" / "v2_gate_variant_cf_graphtoken"
OUT_SP = ROOT / "outputs" / "v2_gate_variant_cf_soft_prompt"
FIG_DIR = ROOT / "documents" / "experiments" / "figures"

KEY_CONDITIONS = [
    "question_only",
    "majority_prior",
    "soft_prompt",
    "shuffled_graph",
    "structure_only",
    "serialized_current_graph",
    "frozen_graph_history",
    "graph_once_then_text",
    "cached_no_reencode",
    "modify_and_print",
    "oracle_updated_graph",
    "predicted_updated_graph",
    "tool_solver",
]

TREND_CONDS = [
    "question_only",
    "majority_prior",
    "oracle_updated_graph",
    "predicted_updated_graph",
]

SCALE_ORDER = ["scale_small", "scale_medium", "scale_ood"]
DENSITY_ORDER = ["medium", "dense"]
REASON_ORDER = ["edge_exists", "reachability", "cycle_membership"]
SPLIT_ORDER = ["validation", "test", "ood"]


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_sessions() -> pd.DataFrame:
    rows: list[dict] = []
    for split in SPLIT_ORDER:
        path = DATA_DIR / f"{split}.jsonl"
        with path.open() as f:
            for line in f:
                session = json.loads(line)
                n_turns = len(session["turns"])
                for turn in session["turns"]:
                    cx = turn["complexity"]
                    rows.append(
                        {
                            "session_id": session["session_id"],
                            "turn_index": turn["turn_index"],
                            "split": split,
                            "session_length": n_turns,
                            "scale_bin": cx["scale_bin"],
                            "density": cx["density"],
                            "edit_count": cx["edit_count"],
                            "reasoning_type": turn["query"]["reasoning_type"],
                            "gold_answer": turn["gold_answer"],
                            "stale_answer": turn["stale_answer"],
                            "answer_changing": turn["gold_answer"] != turn["stale_answer"],
                        }
                    )
    return pd.DataFrame(rows)


def load_eval(path: Path, model: str) -> pd.DataFrame:
    with path.open() as f:
        data = json.load(f)
    rows: list[dict] = []
    for split, payload in data.items():
        for row in payload["rows"]:
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "session_id": row["session_id"],
                    "turn_index": row["turn_index"],
                    "condition": row["condition"],
                    "answer_correct": bool(row["answer_correct"]),
                    "edit_correct": row.get("edit_correct"),
                    "predicted_answer": row.get("predicted_answer"),
                    "gold_answer": row.get("gold_answer"),
                    "stale_answer": row.get("stale_answer"),
                    "stale": row.get("stale"),
                }
            )
    return pd.DataFrame(rows)


def load_static(path: Path, model: str) -> dict:
    with path.open() as f:
        data = json.load(f)
    return {
        "model": model,
        "overall_accuracy": data["overall_accuracy"],
        "passed": data["passed"],
        "failing_tasks": data.get("failing_tasks", []),
        "task_accuracy": data.get("task_accuracy", {}),
    }


def accuracy_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False)["answer_correct"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .rename(columns={"sum": "correct", "count": "n", "mean": "accuracy"})
    )
    cis = grouped.apply(
        lambda r: _wilson_ci(int(r["correct"]), int(r["n"])),
        axis=1,
        result_type="expand",
    )
    grouped["ci_lo"] = cis[0]
    grouped["ci_hi"] = cis[1]
    return grouped


def savefig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"wrote {path.relative_to(ROOT)}")


def plot_static_gate(statics: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.1, 1.4]})
    models = [s["model"] for s in statics]
    overall = [s["overall_accuracy"] for s in statics]
    colors = ["#2a6f97", "#01497c", "#9a031e"]
    axes[0].bar(models, overall, color=colors[: len(models)], edgecolor="black", linewidth=0.6)
    axes[0].axhline(0.70, color="gray", linestyle="--", linewidth=1, label="gate ≥70%")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Overall accuracy")
    axes[0].set_title("Static oracle-QA gate")
    for i, (m, v, s) in enumerate(zip(models, overall, statics)):
        mark = "pass" if s["passed"] else "fail"
        axes[0].text(i, v + 0.02, f"{v:.1%}\n({mark})", ha="center", va="bottom", fontsize=9)
    axes[0].legend(loc="lower right", fontsize=8)

    task_rows = []
    for s in statics:
        for task, acc in s["task_accuracy"].items():
            task_rows.append({"model": s["model"], "task": task, "accuracy": acc})
    tdf = pd.DataFrame(task_rows)
    sns.barplot(
        data=tdf,
        x="task",
        y="accuracy",
        hue="model",
        order=REASON_ORDER,
        palette=dict(zip(models, colors[: len(models)])),
        ax=axes[1],
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1].axhline(0.50, color="gray", linestyle="--", linewidth=1, label="per-task ≥50%")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Per-task static accuracy")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=8)
    savefig("static_gate.png")


def plot_conditions_overall(joined: pd.DataFrame, soft_prompt: pd.DataFrame) -> None:
    glm = joined[joined["condition"].isin(KEY_CONDITIONS)].copy()
    # soft_prompt is its own model/condition; append as a synthetic condition series
    sp = soft_prompt.copy()
    sp["condition"] = "soft_prompt"
    # Replicate soft_prompt rows once per GLM model for side-by-side? Better: one bar group.
    # Use turn-weighted mean across all splits for TEA/GT conditions, plus soft_prompt once.
    tea_gt = accuracy_table(glm, ["model", "condition"])
    sp_acc = accuracy_table(sp, ["condition"])
    sp_acc["model"] = "soft_prompt"

    # Order conditions for display; put soft_prompt near text baselines
    order = [c for c in KEY_CONDITIONS if c != "soft_prompt"]
    # insert soft_prompt after majority_prior
    if "majority_prior" in order:
        idx = order.index("majority_prior") + 1
        order = order[:idx] + ["soft_prompt"] + order[idx:]

    plot_df = pd.concat([tea_gt, sp_acc], ignore_index=True)
    plot_df = plot_df[plot_df["condition"].isin(order)]

    plt.figure(figsize=(13, 5.5))
    ax = sns.barplot(
        data=plot_df,
        x="condition",
        y="accuracy",
        hue="model",
        order=order,
        hue_order=["TEA", "GraphToken", "soft_prompt"],
        palette={"TEA": "#2a6f97", "GraphToken": "#01497c", "soft_prompt": "#9a031e"},
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Answer accuracy (turn-weighted)")
    ax.set_xlabel("")
    ax.set_title("Dynamic eval: condition accuracy (validation+test+ood)")
    ax.tick_params(axis="x", rotation=55)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(title="", fontsize=9)
    savefig("conditions_overall.png")


def plot_conditions_by_split(joined: pd.DataFrame) -> None:
    focus = [
        "question_only",
        "majority_prior",
        "frozen_graph_history",
        "oracle_updated_graph",
        "predicted_updated_graph",
        "tool_solver",
    ]
    sub = joined[joined["condition"].isin(focus)]
    tab = accuracy_table(sub, ["model", "split", "condition"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, model in zip(axes, ["TEA", "GraphToken"]):
        mdf = tab[tab["model"] == model].pivot(index="condition", columns="split", values="accuracy")
        mdf = mdf.reindex(index=focus, columns=SPLIT_ORDER)
        sns.heatmap(
            mdf,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            vmin=0.3,
            vmax=1.0,
            ax=ax,
            cbar=ax is axes[-1],
        )
        ax.set_title(model)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Per-split answer accuracy for key conditions", y=1.02)
    savefig("conditions_by_split.png")


def _trend_plot(
    tab: pd.DataFrame,
    x: str,
    order: list,
    title: str,
    fname: str,
    xlabel: str,
) -> None:
    sub = tab[tab["condition"].isin(TREND_CONDS)].copy()
    # categorical x
    sub[x] = pd.Categorical(sub[x], categories=order, ordered=True)
    sub = sub.dropna(subset=[x])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    style = {
        "oracle_updated_graph": {"marker": "o", "linestyle": "-", "lw": 2.2},
        "predicted_updated_graph": {"marker": "s", "linestyle": "--", "lw": 2.0},
        "question_only": {"marker": "^", "linestyle": ":", "lw": 1.5},
        "majority_prior": {"marker": "v", "linestyle": ":", "lw": 1.5},
    }
    colors = {
        "oracle_updated_graph": "#0a9396",
        "predicted_updated_graph": "#005f73",
        "question_only": "#bb3e03",
        "majority_prior": "#9b2226",
    }
    for ax, model in zip(axes, ["TEA", "GraphToken"]):
        mdf = sub[sub["model"] == model]
        for cond in TREND_CONDS:
            cdf = mdf[mdf["condition"] == cond].sort_values(x)
            if cdf.empty:
                continue
            st = style[cond]
            xs = list(range(len(cdf)))
            ax.plot(
                xs,
                cdf["accuracy"].to_numpy(),
                label=cond,
                color=colors[cond],
                marker=st["marker"],
                linestyle=st["linestyle"],
                linewidth=st["lw"],
            )
            ax.fill_between(
                xs,
                cdf["ci_lo"].to_numpy(),
                cdf["ci_hi"].to_numpy(),
                color=colors[cond],
                alpha=0.12,
            )
        ax.set_title(model)
        ax.set_ylim(0.2, 1.0)
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Answer accuracy")
        ax.legend(fontsize=7, loc="lower left")
        oracle_ns = (
            mdf[mdf["condition"] == "oracle_updated_graph"]
            .sort_values(x)
            .set_index(x)["n"]
            .reindex(order)
        )
        labels = []
        for lab in order:
            n = oracle_ns.get(lab, float("nan"))
            labels.append(f"{lab}\n(n={int(n)})" if pd.notna(n) else str(lab))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels)
    fig.suptitle(title, y=1.02)
    savefig(fname)


def plot_complexity_trends(joined: pd.DataFrame) -> None:
    # session_length is constant per session; already on each turn
    for x, order, title, fname, xlabel in [
        (
            "scale_bin",
            SCALE_ORDER,
            "Accuracy vs graph scale",
            "trend_scale.png",
            "Scale bin",
        ),
        (
            "density",
            DENSITY_ORDER,
            "Accuracy vs graph density",
            "trend_density.png",
            "Density bin",
        ),
        (
            "session_length",
            [1, 2, 4, 8],
            "Accuracy vs session length (turn count)",
            "trend_session_length.png",
            "Session length",
        ),
    ]:
        sub = joined[joined["condition"].isin(TREND_CONDS)]
        tab = accuracy_table(sub, ["model", "condition", x])
        _trend_plot(tab, x, order, title, fname, xlabel)

    # turn index
    sub = joined[joined["condition"].isin(["oracle_updated_graph", "predicted_updated_graph"])]
    tab = accuracy_table(sub, ["model", "condition", "turn_index"])
    order = sorted(tab["turn_index"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, model in zip(axes, ["TEA", "GraphToken"]):
        mdf = tab[tab["model"] == model]
        for cond, color, ls in [
            ("oracle_updated_graph", "#0a9396", "-"),
            ("predicted_updated_graph", "#005f73", "--"),
        ]:
            cdf = mdf[mdf["condition"] == cond].sort_values("turn_index")
            ax.plot(
                cdf["turn_index"],
                cdf["accuracy"],
                marker="o",
                color=color,
                linestyle=ls,
                label=cond,
                linewidth=2,
            )
        ax.set_title(model)
        ax.set_xlabel("Turn index (0-based)")
        ax.set_ylim(0.5, 0.9)
        ax.grid(True, axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Answer accuracy")
        ax.legend(fontsize=8)
    fig.suptitle("Accuracy vs turn position within session", y=1.02)
    savefig("trend_turn_index.png")


def plot_reasoning_type(joined: pd.DataFrame) -> None:
    focus = TREND_CONDS
    sub = joined[joined["condition"].isin(focus)]
    tab = accuracy_table(sub, ["model", "condition", "reasoning_type"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, model in zip(axes, ["TEA", "GraphToken"]):
        mdf = tab[tab["model"] == model]
        sns.barplot(
            data=mdf,
            x="reasoning_type",
            y="accuracy",
            hue="condition",
            order=REASON_ORDER,
            hue_order=focus,
            ax=ax,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_title(model)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("")
        if ax is axes[0]:
            ax.set_ylabel("Answer accuracy")
        ax.legend(fontsize=7, loc="upper left")
        # annotate n from oracle rows
        for i, task in enumerate(REASON_ORDER):
            n = mdf[(mdf["condition"] == "oracle_updated_graph") & (mdf["reasoning_type"] == task)]["n"]
            if len(n):
                ax.text(i, -0.08, f"n={int(n.iloc[0])}", ha="center", fontsize=8, transform=ax.get_xaxis_transform())
    fig.suptitle("Accuracy by reasoning type", y=1.05)
    savefig("reasoning_type.png")


def plot_edit_ops(joined: pd.DataFrame) -> None:
    # predicted only for edit accuracy; answer accuracy for predicted/oracle
    pred = joined[joined["condition"] == "predicted_updated_graph"].copy()
    pred = pred[pred["edit_correct"].notna()]
    edit_tab = (
        pred.groupby(["model", "edit_count"])["edit_correct"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .rename(columns={"sum": "correct", "count": "n", "mean": "edit_accuracy"})
    )
    ans = joined[joined["condition"].isin(["oracle_updated_graph", "predicted_updated_graph"])]
    ans_tab = accuracy_table(ans, ["model", "condition", "edit_count"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.barplot(
        data=edit_tab,
        x="edit_count",
        y="edit_accuracy",
        hue="model",
        palette={"TEA": "#2a6f97", "GraphToken": "#01497c"},
        ax=axes[0],
        edgecolor="black",
        linewidth=0.4,
    )
    axes[0].set_ylim(0.7, 1.02)
    axes[0].set_title("Edit-prediction accuracy by op count")
    axes[0].set_xlabel("Gold edit_count")
    axes[0].set_ylabel("Edit accuracy")
    axes[0].legend(fontsize=8)

    sns.lineplot(
        data=ans_tab,
        x="edit_count",
        y="accuracy",
        hue="model",
        style="condition",
        markers=True,
        ax=axes[1],
    )
    axes[1].set_ylim(0.5, 0.9)
    axes[1].set_title("Answer accuracy by op count")
    axes[1].set_xlabel("Gold edit_count")
    axes[1].set_ylabel("Answer accuracy")
    axes[1].legend(fontsize=7)
    savefig("edit_ops.png")


def plot_answer_changing(joined: pd.DataFrame) -> None:
    # Use oracle_updated_graph predictions: answer_changing from gold!=stale on session meta
    oracle = joined[joined["condition"] == "oracle_updated_graph"].copy()
    ac = oracle[oracle["answer_changing"]].copy()
    ac["matches_stale"] = ac["predicted_answer"] == ac["stale_answer"]

    rows = []
    for model in ["TEA", "GraphToken"]:
        for split in SPLIT_ORDER + ["all"]:
            if split == "all":
                sub = ac[ac["model"] == model]
            else:
                sub = ac[(ac["model"] == model) & (ac["split"] == split)]
            n = len(sub)
            if n == 0:
                continue
            acc = sub["answer_correct"].mean()
            stale = sub["matches_stale"].mean()
            rows.append({"model": model, "split": split, "metric": "accuracy", "value": acc, "n": n})
            rows.append({"model": model, "split": split, "metric": "pred_matches_stale", "value": stale, "n": n})
    rdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, model in zip(axes, ["TEA", "GraphToken"]):
        mdf = rdf[rdf["model"] == model]
        sns.barplot(
            data=mdf,
            x="split",
            y="value",
            hue="metric",
            order=SPLIT_ORDER + ["all"],
            palette={"accuracy": "#0a9396", "pred_matches_stale": "#9b2226"},
            ax=ax,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_title(model)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("")
        if ax is axes[0]:
            ax.set_ylabel("Rate on answer-changing turns")
        ax.legend(fontsize=8)
        # n labels
        for i, split in enumerate(SPLIT_ORDER + ["all"]):
            n = mdf[(mdf["split"] == split) & (mdf["metric"] == "accuracy")]["n"]
            if len(n):
                ax.text(i, -0.08, f"n={int(n.iloc[0])}", ha="center", fontsize=8, transform=ax.get_xaxis_transform())
    fig.suptitle("Oracle accuracy on turns where the edit changes the answer", y=1.05)
    savefig("answer_changing.png")


def plot_oracle_vs_predicted_gap(joined: pd.DataFrame) -> None:
    pred = joined[joined["condition"] == "predicted_updated_graph"].copy()
    oracle = joined[joined["condition"] == "oracle_updated_graph"][
        ["model", "session_id", "turn_index", "answer_correct", "predicted_answer"]
    ].rename(columns={"answer_correct": "oracle_correct", "predicted_answer": "oracle_pred"})
    merged = pred.merge(oracle, on=["model", "session_id", "turn_index"], how="inner")
    merged["edit_ok"] = merged["edit_correct"].fillna(False).astype(bool)
    merged["same_answer"] = merged["predicted_answer"] == merged["oracle_pred"]

    rows = []
    for model in ["TEA", "GraphToken"]:
        m = merged[merged["model"] == model]
        for label, mask in [
            ("all turns", slice(None)),
            ("edit correct", m["edit_ok"]),
            ("edit wrong", ~m["edit_ok"]),
        ]:
            if label == "all turns":
                sub = m
            else:
                sub = m[mask]
            n = len(sub)
            rows.append(
                {
                    "model": model,
                    "slice": label,
                    "metric": "predicted_acc",
                    "value": sub["answer_correct"].mean() if n else float("nan"),
                    "n": n,
                }
            )
            rows.append(
                {
                    "model": model,
                    "slice": label,
                    "metric": "oracle_acc",
                    "value": sub["oracle_correct"].mean() if n else float("nan"),
                    "n": n,
                }
            )
        # agreement on edit-correct
        ok = m[m["edit_ok"]]
        rows.append(
            {
                "model": model,
                "slice": "edit correct",
                "metric": "answer_agreement",
                "value": ok["same_answer"].mean() if len(ok) else float("nan"),
                "n": len(ok),
            }
        )

    rdf = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, model in zip(axes, ["TEA", "GraphToken"]):
        mdf = rdf[(rdf["model"] == model) & (rdf["metric"].isin(["oracle_acc", "predicted_acc"]))]
        sns.barplot(
            data=mdf,
            x="slice",
            y="value",
            hue="metric",
            order=["all turns", "edit correct", "edit wrong"],
            palette={"oracle_acc": "#0a9396", "predicted_acc": "#005f73"},
            ax=ax,
            edgecolor="black",
            linewidth=0.3,
        )
        agree = rdf[(rdf["model"] == model) & (rdf["metric"] == "answer_agreement")]["value"]
        if len(agree):
            ax.axhline(float(agree.iloc[0]), color="#9a031e", linestyle="--", linewidth=1.2, label=f"answer agree (edit ok)={agree.iloc[0]:.1%}")
        ax.set_title(model)
        ax.set_ylim(0.5, 1.0)
        ax.set_xlabel("")
        if ax is axes[0]:
            ax.set_ylabel("Accuracy")
        ax.legend(fontsize=7)
    fig.suptitle("Oracle vs predicted gap (and residual on identical graphs)", y=1.03)
    savefig("oracle_vs_predicted_gap.png")


def plot_dataset_audit(sessions: pd.DataFrame) -> None:
    with (DATA_DIR / "audit.json").open() as f:
        audit = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

    # answer distribution by split
    ans_rows = []
    for split in SPLIT_ORDER:
        dist = audit[split]["answer_distribution"]
        for ans, n in dist.items():
            ans_rows.append({"split": split, "answer": ans, "n": n})
    sns.barplot(data=pd.DataFrame(ans_rows), x="split", y="n", hue="answer", order=SPLIT_ORDER, ax=axes[0, 0], edgecolor="black", linewidth=0.3)
    axes[0, 0].set_title("Answer labels")
    axes[0, 0].set_xlabel("")

    # operations
    op_rows = []
    for split in SPLIT_ORDER:
        dist = audit[split]["operation_distribution"]
        for op, n in dist.items():
            op_rows.append({"split": split, "op": op, "n": n})
    sns.barplot(data=pd.DataFrame(op_rows), x="split", y="n", hue="op", order=SPLIT_ORDER, ax=axes[0, 1], edgecolor="black", linewidth=0.3)
    axes[0, 1].set_title("Edit operations")
    axes[0, 1].set_xlabel("")

    # reasoning
    r_rows = []
    for split in SPLIT_ORDER:
        dist = audit[split]["reasoning_distribution"]
        for task, n in dist.items():
            r_rows.append({"split": split, "task": task, "n": n})
    sns.barplot(data=pd.DataFrame(r_rows), x="split", y="n", hue="task", order=SPLIT_ORDER, hue_order=REASON_ORDER, ax=axes[0, 2], edgecolor="black", linewidth=0.3)
    axes[0, 2].set_title("Reasoning type")
    axes[0, 2].set_xlabel("")

    # scale / density / session length from sessions (unique turns)
    sns.countplot(data=sessions, x="scale_bin", order=SCALE_ORDER, ax=axes[1, 0], color="#2a6f97", edgecolor="black", linewidth=0.3)
    axes[1, 0].set_title("Scale bin (turns)")
    axes[1, 0].set_xlabel("")
    sns.countplot(data=sessions, x="density", order=DENSITY_ORDER, ax=axes[1, 1], color="#0a9396", edgecolor="black", linewidth=0.3)
    axes[1, 1].set_title("Density bin (turns)")
    axes[1, 1].set_xlabel("")
    # session length: count sessions not turns
    sess = sessions.drop_duplicates("session_id")
    sns.countplot(data=sess, x="session_length", order=[1, 2, 4, 8], ax=axes[1, 2], color="#01497c", edgecolor="black", linewidth=0.3)
    axes[1, 2].set_title("Session length (sessions)")
    axes[1, 2].set_xlabel("")

    fig.suptitle("Dataset audit & complexity distributions (metro_v2_gate_variant_cf)", y=1.01)
    savefig("dataset_audit.png")


def print_cross_check(joined: pd.DataFrame, soft_prompt: pd.DataFrame, statics: list[dict]) -> None:
    print("\n=== CROSS-CHECK vs report.md ===")
    for s in statics:
        print(f"static {s['model']}: {s['overall_accuracy']:.4f} passed={s['passed']}")

    # coarse turn-weighted overall for TEA/GT key conditions
    for model in ["TEA", "GraphToken"]:
        sub = joined[joined["model"] == model]
        for cond in ["oracle_updated_graph", "predicted_updated_graph", "question_only", "majority_prior"]:
            m = sub[sub["condition"] == cond]["answer_correct"].mean()
            n = (sub["condition"] == cond).sum()
            print(f"{model} {cond}: {m:.4f} (n={n})")
        edit = sub[sub["condition"] == "predicted_updated_graph"]["edit_correct"].dropna().mean()
        print(f"{model} edit_acc: {edit:.4f}")

    sp_m = soft_prompt["answer_correct"].mean()
    print(f"soft_prompt overall: {sp_m:.4f} (n={len(soft_prompt)})")

    # scale strata TEA oracle
    for scale in SCALE_ORDER:
        for model in ["TEA", "GraphToken"]:
            for cond in ["oracle_updated_graph", "predicted_updated_graph"]:
                m = joined[
                    (joined["model"] == model)
                    & (joined["condition"] == cond)
                    & (joined["scale_bin"] == scale)
                ]["answer_correct"].mean()
                n = (
                    (joined["model"] == model)
                    & (joined["condition"] == cond)
                    & (joined["scale_bin"] == scale)
                ).sum()
                print(f"{model} {cond} {scale}: {m:.4f} (n={n})")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.7)
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    print("Loading sessions...")
    sessions = load_sessions()
    print(f"  turns={len(sessions)} sessions={sessions['session_id'].nunique()}")

    print("Loading evaluations...")
    tea = load_eval(OUT_TEA / "evaluation.json", "TEA")
    gt = load_eval(OUT_GT / "evaluation.json", "GraphToken")
    sp = load_eval(OUT_SP / "evaluation.json", "soft_prompt")
    glm = pd.concat([tea, gt], ignore_index=True)
    sess_meta = sessions.drop(columns=["gold_answer", "stale_answer"])
    joined = glm.merge(sess_meta, on=["session_id", "turn_index", "split"], how="left")
    if joined["scale_bin"].isna().any():
        missing = joined["scale_bin"].isna().sum()
        raise RuntimeError(f"Failed to join complexity for {missing} rows")
    soft_joined = sp.merge(sess_meta, on=["session_id", "turn_index", "split"], how="left")

    statics = [
        load_static(OUT_TEA / "static_eval_validation.json", "TEA"),
        load_static(OUT_GT / "static_eval_validation.json", "GraphToken"),
        load_static(OUT_SP / "static_eval_validation.json", "soft_prompt"),
    ]

    print("Plotting...")
    plot_static_gate(statics)
    plot_conditions_overall(joined, soft_joined)
    plot_conditions_by_split(joined)
    plot_complexity_trends(joined)
    plot_reasoning_type(joined)
    plot_edit_ops(joined)
    plot_answer_changing(joined)
    plot_oracle_vs_predicted_gap(joined)
    plot_dataset_audit(sessions)
    print_cross_check(joined, soft_joined, statics)
    print("\nDone.")


if __name__ == "__main__":
    main()
