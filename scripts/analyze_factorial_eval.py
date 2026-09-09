#!/usr/bin/env python3
"""CLEGR-style stratified analysis + figures for the factorial CF eval set.

Joins evaluation.json rows to session complexity metadata, writes:
  - documents/experiments/results/v2_gate_variant_cf_factorial-*/strata.json
  - documents/experiments/figures/factorial_*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from graph_modi.evaluation.metrics import paired_accuracy_delta, stratified_accuracy, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "metro_v2_gate_variant_cf_factorial"
OUT = {
    "TEA": ROOT / "outputs" / "v2_gate_variant_cf_factorial_tea",
    "GraphToken": ROOT / "outputs" / "v2_gate_variant_cf_factorial_graphtoken",
    "soft_prompt": ROOT / "outputs" / "v2_gate_variant_cf_factorial_soft_prompt",
}
FIG_DIR = ROOT / "documents" / "experiments" / "figures"
RESULT_DIR = ROOT / "documents" / "experiments" / "results" / "v2_gate_variant_cf_factorial-20260909"

FOCUS = [
    "question_only",
    "majority_prior",
    "oracle_updated_graph",
    "predicted_updated_graph",
]
SCALE_ORDER = ["scale_small", "scale_medium", "scale_large"]
TURN_ORDER = [1, 2, 4, 8]
DENSITY_ORDER = ["sparse", "medium", "dense"]
REASON_ORDER = ["edge_exists", "reachability", "cycle_membership"]


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
                            "density": cx["density"],
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


def plot_marginal_trends(joined: pd.DataFrame) -> None:
    for x, order, title, fname, xlabel in [
        ("scale_bin", SCALE_ORDER, "Disentangled: accuracy vs scale (pooled turns)", "factorial_trend_scale.png", "Scale"),
        ("session_length", TURN_ORDER, "Disentangled: accuracy vs session length (pooled scale)", "factorial_trend_session_length.png", "Session length"),
        ("density", DENSITY_ORDER, "Accuracy vs density (balanced within cells)", "factorial_trend_density.png", "Density"),
    ]:
        sub = joined[joined["condition"].isin(FOCUS)]
        tab = _acc_table(sub, ["model", "condition", x])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        colors = {
            "oracle_updated_graph": "#0a9396",
            "predicted_updated_graph": "#005f73",
            "question_only": "#bb3e03",
            "majority_prior": "#9b2226",
        }
        for ax, model in zip(axes, ["TEA", "GraphToken"]):
            mdf = tab[tab["model"] == model]
            for cond in FOCUS:
                cdf = mdf[mdf["condition"] == cond].copy()
                cdf[x] = pd.Categorical(cdf[x], categories=order, ordered=True)
                cdf = cdf.dropna(subset=[x]).sort_values(x)
                if cdf.empty:
                    continue
                xs = list(range(len(cdf)))
                ax.plot(xs, cdf["accuracy"], marker="o", label=cond, color=colors[cond], linewidth=2)
                ax.fill_between(xs, cdf["ci_lo"], cdf["ci_hi"], color=colors[cond], alpha=0.12)
            ax.set_title(model)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([str(v) for v in order])
            ax.set_ylim(0.2, 1.0)
            ax.set_xlabel(xlabel)
            ax.grid(True, axis="y", alpha=0.3)
            if ax is axes[0]:
                ax.set_ylabel("Answer accuracy")
            ax.legend(fontsize=7, loc="lower left")
        fig.suptitle(title, y=1.02)
        savefig(fname)


def plot_interaction_heatmap(joined: pd.DataFrame) -> None:
    sub = joined[joined["condition"] == "oracle_updated_graph"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, model in zip(axes, ["TEA", "GraphToken"]):
        mdf = sub[sub["model"] == model]
        tab = _acc_table(mdf, ["scale_bin", "session_length"])
        pivot = tab.pivot(index="scale_bin", columns="session_length", values="accuracy")
        pivot = pivot.reindex(index=SCALE_ORDER, columns=TURN_ORDER)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", vmin=0.4, vmax=1.0, ax=ax)
        ax.set_title(f"{model} oracle × scale × turns")
        ax.set_xlabel("Session length")
        ax.set_ylabel("Scale")
    fig.suptitle("Factorial interaction: oracle accuracy", y=1.02)
    savefig("factorial_interaction_heatmap.png")


def plot_reasoning_and_answer_changing(joined: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    sub = joined[joined["condition"].isin(["oracle_updated_graph", "majority_prior"])]
    tab = _acc_table(sub, ["model", "condition", "reasoning_type"])
    for ax, model in zip(axes, ["TEA", "GraphToken"]):
        sns.barplot(
            data=tab[tab["model"] == model],
            x="reasoning_type",
            y="accuracy",
            hue="condition",
            order=REASON_ORDER,
            ax=ax,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_title(model)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("")
        if ax is axes[0]:
            ax.set_ylabel("Accuracy")
        ax.legend(fontsize=7)
    fig.suptitle("CLEGR-style task strata (oracle vs majority)", y=1.03)
    savefig("factorial_reasoning_type.png")

    oracle = joined[joined["condition"] == "oracle_updated_graph"].copy()
    oracle["slice"] = oracle["answer_changing"].map({True: "answer_changing", False: "answer_stable"})
    tab = _acc_table(oracle, ["model", "slice"])
    plt.figure(figsize=(7, 4.2))
    sns.barplot(data=tab, x="slice", y="accuracy", hue="model", edgecolor="black", linewidth=0.3)
    plt.ylim(0, 1.05)
    plt.title("Oracle accuracy: answer-changing vs stable turns")
    plt.xlabel("")
    savefig("factorial_answer_changing.png")


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
                "density": stratified_accuracy(rows, stratum_key="density"),
                "reasoning_type": stratified_accuracy(rows, stratum_key="reasoning_type"),
                "hop_depth": stratified_accuracy(rows, stratum_key="hop_depth"),
                "edit_count": stratified_accuracy(
                    [{**r, "edit_count": str(r["edit_count"])} for r in rows],
                    stratum_key="edit_count",
                ),
                "answer_changing": stratified_accuracy(
                    [{**r, "answer_changing": str(r["answer_changing"])} for r in rows],
                    stratum_key="answer_changing",
                ),
                "factorial_cell": stratified_accuracy(rows, stratum_key="factorial_cell"),
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

    plot_marginal_trends(joined)
    plot_interaction_heatmap(joined)
    plot_reasoning_and_answer_changing(joined)

    strata = build_strata(joined, soft)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / "strata.json"
    out_path.write_text(json.dumps(strata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)}")
    print("multimodal_gain", json.dumps(strata["multimodal_gain"], indent=2))


if __name__ == "__main__":
    main()
