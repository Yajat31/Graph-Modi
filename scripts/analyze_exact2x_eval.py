#!/usr/bin/env python3
"""CLEGR-style stratified analysis + figures for the exact-uniform 2x eval set.

Joins evaluation.json rows to session complexity metadata, writes:
  - documents/experiments/results/v2_gate_variant_cf_exact2x-*/strata.json
  - documents/experiments/figures/exact2x_*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from graph_modi.evaluation.metrics import paired_accuracy_delta, stratified_accuracy, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "metro_v2_gate_variant_cf_exact2x"
OUT = {
    "TEA": ROOT / "outputs" / "v2_gate_variant_cf_exact2x_tea",
    "GraphToken": ROOT / "outputs" / "v2_gate_variant_cf_exact2x_graphtoken",
    "soft_prompt": ROOT / "outputs" / "v2_gate_variant_cf_exact2x_soft_prompt",
}
FIG_DIR = ROOT / "documents" / "experiments" / "figures"
RESULT_DIR = ROOT / "documents" / "experiments" / "results" / "v2_gate_variant_cf_exact2x-20260911"

FOCUS = [
    "question_only",
    "majority_prior",
    "structure_only",
    "shuffled_graph",
    "frozen_graph_history",
    "graph_once_then_text",
    "cached_no_reencode",
    "serialized_current_graph",
    "serialized_initial_history",
    "token_matched_history",
    "modify_and_print",
    "oracle_updated_graph",
    "predicted_updated_graph",
]
# soft_prompt is a separate checkpoint; included in overall / trend overlays.
KEY_CONDITIONS = FOCUS + ["soft_prompt"]
SPLIT_ORDER = ["validation", "test"]
SCALE_ORDER = ["scale_small", "scale_medium", "scale_large"]
TURN_ORDER = [1, 2, 4, 8]
DENSITY_ORDER = ["sparse", "medium", "dense"]
REASON_ORDER = ["edge_exists", "reachability", "cycle_membership"]

CONDITION_COLORS = {
    "question_only": "#9b2226",
    "majority_prior": "#ae2012",
    "soft_prompt": "#bb3e03",
    "structure_only": "#ca6702",
    "shuffled_graph": "#ee9b00",
    "frozen_graph_history": "#e9d8a6",
    "graph_once_then_text": "#94d2bd",
    "cached_no_reencode": "#0a9396",
    "serialized_current_graph": "#005f73",
    "serialized_initial_history": "#001219",
    "token_matched_history": "#3d405b",
    "modify_and_print": "#81b29a",
    "oracle_updated_graph": "#2a9d8f",
    "predicted_updated_graph": "#264653",
}


def load_sessions() -> pd.DataFrame:
    rows: list[dict] = []
    for split in ("validation", "test"):
        path = DATA_DIR / f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                session = json.loads(line)
                for turn in session["turns"]:
                    cx = turn["complexity"]
                    rows.append(
                        {
                            "session_id": session["session_id"],
                            "turn_index": turn["turn_index"],
                            "split": split,
                            "session_length": cx.get("session_length", len(session["turns"])),
                            "scale_bin": cx["scale_bin"],
                            "node_count": len(session["initial_graph"]["nodes"]),
                            "density": cx["density"],
                            "target_density": cx.get("target_density", cx["density"]),
                            "edit_count": cx["edit_count"],
                            "hop_depth": cx.get("hop_depth"),
                            "topology": cx.get("topology"),
                            "factorial_cell": cx.get("factorial_cell"),
                            "reasoning_type": turn["query"]["reasoning_type"],
                            "gold_answer": turn["gold_answer"],
                            "stale_answer": turn["stale_answer"],
                            "answer_changing": turn["gold_answer"] != turn["stale_answer"],
                            "is_noop": any(
                                edit.get("operation") == "NOOP"
                                for edit in (turn.get("gold_edits") or [])
                            )
                            or (
                                turn.get("gold_edit", {}).get("operation") == "NOOP"
                                if turn.get("gold_edit")
                                else False
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def load_eval(path: Path, model: str) -> pd.DataFrame:
    with path.open() as handle:
        data = json.load(handle)
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
                    "state_exact": row.get("state_exact"),
                }
            )
    return pd.DataFrame(rows)


def savefig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"wrote {path.relative_to(ROOT)}")


def _acc_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False)["answer_correct"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .rename(columns={"sum": "correct", "count": "n", "mean": "accuracy"})
    )
    cis = grouped.apply(
        lambda row: wilson_interval(int(row["correct"]), int(row["n"])),
        axis=1,
        result_type="expand",
    )
    grouped["ci_lo"] = cis[0]
    grouped["ci_hi"] = cis[1]
    return grouped


def plot_conditions_overall(joined: pd.DataFrame, soft: pd.DataFrame) -> None:
    glm = joined[joined["condition"].isin(FOCUS)].copy()
    tea_gt = _acc_table(glm, ["model", "condition"])
    sp = soft.copy()
    sp_acc = _acc_table(sp, ["condition"])
    sp_acc["model"] = "soft_prompt"
    order = list(FOCUS)
    if "majority_prior" in order:
        idx = order.index("majority_prior") + 1
        order = order[:idx] + ["soft_prompt"] + order[idx:]
    plot_df = pd.concat([tea_gt, sp_acc], ignore_index=True)
    plot_df = plot_df[plot_df["condition"].isin(order)]
    plt.figure(figsize=(14, 5.8))
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
    ax.set_title("Exact2x dynamic eval: condition accuracy (validation+test; no tool_solver)")
    ax.tick_params(axis="x", rotation=55)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(title="", fontsize=9)
    savefig("exact2x_conditions_overall.png")


def plot_conditions_by_split(joined: pd.DataFrame, soft: pd.DataFrame) -> None:
    glm = joined[joined["condition"].isin(FOCUS)].copy()
    sp = soft.copy()
    sp["model"] = "soft_prompt"
    combined = pd.concat([glm, sp], ignore_index=True)
    order = list(FOCUS)
    if "majority_prior" in order:
        idx = order.index("majority_prior") + 1
        order = order[:idx] + ["soft_prompt"] + order[idx:]
    tab = _acc_table(combined, ["model", "split", "condition"])
    models = ["TEA", "GraphToken", "soft_prompt"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.5), sharey=True)
    for ax, model in zip(axes, models):
        mdf = tab[tab["model"] == model].pivot(index="condition", columns="split", values="accuracy")
        mdf = mdf.reindex(index=order, columns=SPLIT_ORDER)
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
    fig.suptitle("Exact2x: per-split answer accuracy (all frameworks except tool_solver)", y=1.01)
    savefig("exact2x_conditions_by_split.png")


def _plot_condition_lines(ax, tab: pd.DataFrame, model: str, x: str, order: list, conditions: list[str]) -> None:
    mdf = tab[tab["model"] == model]
    for cond in conditions:
        cdf = mdf[mdf["condition"] == cond].copy()
        if cdf.empty:
            continue
        cdf[x] = pd.Categorical(cdf[x], categories=order, ordered=True)
        cdf = cdf.dropna(subset=[x]).sort_values(x)
        if cdf.empty:
            continue
        xs = list(range(len(cdf)))
        color = CONDITION_COLORS.get(cond, "#333333")
        lw = 2.4 if cond in ("oracle_updated_graph", "predicted_updated_graph") else 1.4
        ax.plot(xs, cdf["accuracy"], marker="o", label=cond, color=color, linewidth=lw, markersize=4)
        ax.fill_between(xs, cdf["ci_lo"], cdf["ci_hi"], color=color, alpha=0.06)


def plot_marginal_trends(joined: pd.DataFrame, soft: pd.DataFrame) -> None:
    sp = soft.copy()
    sp["model"] = "TEA"  # overlay same soft_prompt curve on TEA panel
    sp2 = soft.copy()
    sp2["model"] = "GraphToken"
    trend = pd.concat([joined[joined["condition"].isin(FOCUS)], sp, sp2], ignore_index=True)
    conditions = list(FOCUS)
    if "majority_prior" in conditions:
        idx = conditions.index("majority_prior") + 1
        conditions = conditions[:idx] + ["soft_prompt"] + conditions[idx:]

    for x, order, title, fname, xlabel in [
        ("scale_bin", SCALE_ORDER, "Exact2x: accuracy vs scale (all frameworks)", "exact2x_trend_scale.png", "Scale"),
        ("session_length", TURN_ORDER, "Exact2x: accuracy vs session length (all frameworks)", "exact2x_trend_session_length.png", "Session length"),
        ("target_density", DENSITY_ORDER, "Exact2x: accuracy vs target density (all frameworks)", "exact2x_trend_density.png", "Target density"),
    ]:
        tab = _acc_table(trend, ["model", "condition", x])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
        for ax, model in zip(axes, ["TEA", "GraphToken"]):
            _plot_condition_lines(ax, tab, model, x, order, conditions)
            ax.set_title(model)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([str(v) for v in order])
            ax.set_ylim(0.2, 1.0)
            ax.set_xlabel(xlabel)
            ax.grid(True, axis="y", alpha=0.3)
            if ax is axes[0]:
                ax.set_ylabel("Answer accuracy")
            ax.legend(fontsize=6, loc="lower left", ncol=2, framealpha=0.9)
        fig.suptitle(title, y=1.02)
        savefig(fname)

    # exact n within bins — all frameworks (TEA / GraphToken panels)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True)
    for ax, model in zip(axes, ["TEA", "GraphToken"]):
        mdf = trend[(trend["model"] == model) & (trend["condition"].isin(conditions))]
        tab = _acc_table(mdf, ["condition", "node_count"])
        for cond in conditions:
            cdf = tab[tab["condition"] == cond].sort_values("node_count")
            if cdf.empty:
                continue
            color = CONDITION_COLORS.get(cond, "#333333")
            lw = 2.2 if cond in ("oracle_updated_graph", "predicted_updated_graph") else 1.2
            ax.plot(cdf["node_count"], cdf["accuracy"], marker="o", label=cond, color=color, linewidth=lw, markersize=3)
        ax.set_title(model)
        ax.set_xlabel("Exact node count")
        ax.set_ylim(0.2, 1.0)
        ax.grid(True, axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Answer accuracy")
        ax.legend(fontsize=5.5, loc="lower left", ncol=2, framealpha=0.9)
    fig.suptitle("Exact2x: accuracy vs exact n (all frameworks except tool_solver)", y=1.02)
    savefig("exact2x_trend_exact_n.png")


def plot_interaction_heatmap(joined: pd.DataFrame) -> None:
    # One small heatmap per condition (TEA and GraphToken as two pages/files).
    n = len(FOCUS)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    for model in ["TEA", "GraphToken"]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.1 * nrows), sharex=True, sharey=True)
        axes_flat = axes.flatten()
        for i, cond in enumerate(FOCUS):
            ax = axes_flat[i]
            mdf = joined[(joined["model"] == model) & (joined["condition"] == cond)]
            tab = _acc_table(mdf, ["scale_bin", "session_length"])
            pivot = tab.pivot(index="scale_bin", columns="session_length", values="accuracy")
            pivot = pivot.reindex(index=SCALE_ORDER, columns=TURN_ORDER)
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", vmin=0.3, vmax=1.0, ax=ax, cbar=False, annot_kws={"size": 7})
            ax.set_title(cond, fontsize=8)
            ax.set_xlabel("L" if i >= n - ncols else "")
            ax.set_ylabel("Scale" if i % ncols == 0 else "")
        for j in range(n, len(axes_flat)):
            axes_flat[j].axis("off")
        fig.suptitle(f"Exact2x interaction: {model} scale × length (all frameworks except tool_solver)", y=1.01)
        fname = "exact2x_interaction_heatmap.png" if model == "TEA" else "exact2x_interaction_heatmap_graphtoken.png"
        savefig(fname)


def plot_turn_index(joined: pd.DataFrame, soft: pd.DataFrame) -> None:
    sp = soft.copy()
    sp["model"] = "TEA"
    sp2 = soft.copy()
    sp2["model"] = "GraphToken"
    trend = pd.concat([joined[joined["condition"].isin(FOCUS)], sp, sp2], ignore_index=True)
    conditions = list(FOCUS)
    if "majority_prior" in conditions:
        idx = conditions.index("majority_prior") + 1
        conditions = conditions[:idx] + ["soft_prompt"] + conditions[idx:]
    tab = _acc_table(trend, ["model", "condition", "turn_index"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, model in zip(axes, ["TEA", "GraphToken"]):
        mdf = tab[tab["model"] == model]
        for cond in conditions:
            cdf = mdf[mdf["condition"] == cond].sort_values("turn_index")
            if cdf.empty:
                continue
            color = CONDITION_COLORS.get(cond, "#333333")
            lw = 2.4 if cond in ("oracle_updated_graph", "predicted_updated_graph") else 1.3
            ax.plot(cdf["turn_index"], cdf["accuracy"], marker="o", color=color, label=cond, linewidth=lw, markersize=4)
        ax.set_title(model)
        ax.set_xlabel("Turn index")
        ax.set_ylim(0.25, 1.0)
        ax.grid(True, axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Answer accuracy")
        ax.legend(fontsize=5.5, loc="lower left", ncol=2, framealpha=0.9)
    fig.suptitle("Exact2x: turn-index (all frameworks except tool_solver)", y=1.02)
    savefig("exact2x_trend_turn_index.png")


def plot_reasoning_and_answer_changing(joined: pd.DataFrame, soft: pd.DataFrame) -> None:
    sp = soft.copy()
    sp["model"] = "TEA"
    sp2 = soft.copy()
    sp2["model"] = "GraphToken"
    trend = pd.concat([joined[joined["condition"].isin(FOCUS)], sp, sp2], ignore_index=True)
    conditions = list(FOCUS)
    if "majority_prior" in conditions:
        idx = conditions.index("majority_prior") + 1
        conditions = conditions[:idx] + ["soft_prompt"] + conditions[idx:]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    tab = _acc_table(trend, ["model", "condition", "reasoning_type"])
    for ax, model in zip(axes, ["TEA", "GraphToken"]):
        mdf = tab[tab["model"] == model]
        sns.barplot(
            data=mdf,
            x="reasoning_type",
            y="accuracy",
            hue="condition",
            hue_order=conditions,
            order=REASON_ORDER,
            palette={c: CONDITION_COLORS.get(c, "#333") for c in conditions},
            ax=ax,
            edgecolor="black",
            linewidth=0.2,
        )
        ax.set_title(model)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("")
        if ax is axes[0]:
            ax.set_ylabel("Accuracy")
        ax.legend(fontsize=5.5, loc="lower left", ncol=2, framealpha=0.9)
    fig.suptitle("Exact2x: task strata (all frameworks except tool_solver)", y=1.02)
    savefig("exact2x_reasoning_type.png")

    # answer-changing vs stable for every framework
    rows = []
    for cond in conditions:
        if cond == "soft_prompt":
            src = trend[trend["condition"] == "soft_prompt"]
        else:
            src = joined[joined["condition"] == cond]
        if src.empty:
            continue
        tmp = src.copy()
        tmp["slice"] = tmp["answer_changing"].map({True: "answer_changing", False: "answer_stable"})
        tmp["condition"] = cond
        rows.append(tmp)
    ac = pd.concat(rows, ignore_index=True)
    # pool models for TEA/GT conditions; soft_prompt already tagged
    # plot TEA and GraphToken separately
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True)
    for ax, model in zip(axes, ["TEA", "GraphToken"]):
        mdf = ac[ac["model"] == model]
        tab = _acc_table(mdf, ["condition", "slice"])
        sns.barplot(
            data=tab,
            x="slice",
            y="accuracy",
            hue="condition",
            hue_order=conditions,
            palette={c: CONDITION_COLORS.get(c, "#333") for c in conditions},
            ax=ax,
            edgecolor="black",
            linewidth=0.2,
        )
        ax.set_title(model)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("")
        if ax is axes[0]:
            ax.set_ylabel("Accuracy")
        ax.legend(fontsize=5.5, loc="lower left", ncol=2, framealpha=0.9)
    fig.suptitle("Exact2x: answer-changing vs stable (all frameworks except tool_solver)", y=1.02)
    savefig("exact2x_answer_changing.png")


def build_strata(joined: pd.DataFrame, soft: pd.DataFrame) -> dict:
    strata: dict = {"models": {}, "paired_deltas": {}, "multimodal_gain": {}}
    for model in ["TEA", "GraphToken"]:
        mdf = joined[joined["model"] == model]
        model_out: dict = {"conditions": {}, "by_stratum": {}}
        for cond in mdf["condition"].unique():
            rows = mdf[mdf["condition"] == cond].to_dict("records")
            successes = sum(bool(r["answer_correct"]) for r in rows)
            model_out["conditions"][cond] = {
                "n": len(rows),
                "answer_accuracy": successes / max(1, len(rows)),
                "answer_accuracy_ci95": list(wilson_interval(successes, len(rows))),
            }
        for cond in FOCUS:
            cdf = mdf[mdf["condition"] == cond]
            rows = cdf.to_dict("records")
            model_out["by_stratum"][cond] = {
                "scale_bin": stratified_accuracy(rows, stratum_key="scale_bin"),
                "session_length": stratified_accuracy(
                    [{**r, "session_length": str(r["session_length"])} for r in rows],
                    stratum_key="session_length",
                ),
                "target_density": stratified_accuracy(rows, stratum_key="target_density"),
                "density": stratified_accuracy(rows, stratum_key="density"),
                "reasoning_type": stratified_accuracy(rows, stratum_key="reasoning_type"),
                "hop_depth": stratified_accuracy(rows, stratum_key="hop_depth"),
                "node_count": stratified_accuracy(
                    [{**r, "node_count": str(r["node_count"])} for r in rows],
                    stratum_key="node_count",
                ),
                "edit_count": stratified_accuracy(
                    [{**r, "edit_count": str(r["edit_count"])} for r in rows],
                    stratum_key="edit_count",
                ),
                "answer_changing": stratified_accuracy(
                    [{**r, "answer_changing": str(r["answer_changing"])} for r in rows],
                    stratum_key="answer_changing",
                ),
                "turn_index": stratified_accuracy(
                    [{**r, "turn_index": str(r["turn_index"])} for r in rows],
                    stratum_key="turn_index",
                ),
            }
        # paired deltas
        oracle = mdf[mdf["condition"] == "oracle_updated_graph"].to_dict("records")
        for baseline in ["question_only", "majority_prior", "frozen_graph_history", "predicted_updated_graph"]:
            right = mdf[mdf["condition"] == baseline].to_dict("records")
            strata["paired_deltas"][f"{model}:oracle_vs_{baseline}"] = paired_accuracy_delta(oracle, right)
        strata["models"][model] = model_out

        soft_acc = float(soft["answer_correct"].mean()) if len(soft) else float("nan")
        struct = float(mdf[mdf["condition"] == "structure_only"]["answer_correct"].mean())
        oracle_acc = float(mdf[mdf["condition"] == "oracle_updated_graph"]["answer_correct"].mean())
        strata["multimodal_gain"][model] = {
            "oracle": oracle_acc,
            "soft_prompt": soft_acc,
            "structure_only": struct,
            "gain": oracle_acc - max(soft_acc, struct),
        }
    if len(soft):
        successes = int(soft["answer_correct"].sum())
        strata["soft_prompt"] = {
            "n": len(soft),
            "answer_accuracy": successes / len(soft),
            "answer_accuracy_ci95": list(wilson_interval(successes, len(soft))),
        }
    return strata


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.7)
    missing = [name for name, path in OUT.items() if not (path / "evaluation.json").exists()]
    if missing:
        raise SystemExit(f"Missing evaluation.json for: {missing}. Run evaluate first.")

    sessions = load_sessions()
    frames = []
    for model in ["TEA", "GraphToken"]:
        frames.append(load_eval(OUT[model] / "evaluation.json", model))
    glm = pd.concat(frames, ignore_index=True)
    meta = sessions.drop(columns=["gold_answer", "stale_answer"], errors="ignore")
    joined = glm.merge(meta, on=["session_id", "turn_index", "split"], how="left")
    if joined["scale_bin"].isna().any():
        raise RuntimeError("Failed to join complexity metadata")
    soft = load_eval(OUT["soft_prompt"] / "evaluation.json", "soft_prompt")
    soft = soft.merge(meta, on=["session_id", "turn_index", "split"], how="left")

    plot_conditions_overall(joined, soft)
    plot_conditions_by_split(joined, soft)
    plot_marginal_trends(joined, soft)
    plot_interaction_heatmap(joined)
    plot_turn_index(joined, soft)
    plot_reasoning_and_answer_changing(joined, soft)

    strata = build_strata(joined, soft)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / "strata.json"
    out_path.write_text(json.dumps(strata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)}")
    print("multimodal_gain", json.dumps(strata["multimodal_gain"], indent=2))


if __name__ == "__main__":
    main()
