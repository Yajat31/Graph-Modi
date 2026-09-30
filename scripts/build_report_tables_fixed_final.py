"""Print the markdown tables used in status_report_fixed_final.md, straight from the result JSONs."""

from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "documents/experiments/results/fixed_final-20260929"
MODELS = (("TEA", "tea"), ("GraphToken", "graphtoken"), ("soft_prompt", "soft_prompt"))
MAIN_TASKS = [
    "reachability", "edge_exists", "cycle_membership", "constrained_reachability",
    "filtered_neighbor_count", "within_hops_count", "most_common_attribute_within_hops", "node_count",
]
STATIC_TASKS = [
    "reachability", "constrained_reachability", "cycle_membership", "edge_exists",
    "most_common_attribute_within_hops", "filtered_path_count", "node_degree", "shortest_path",
    "filtered_neighbor_count", "path_cost", "within_hops_count", "within_hops_list",
]


def load(name: str):
    path = RES / name
    return json.loads(path.read_text()) if path.exists() else None


def p(x, nd=1):
    return "n/a" if x is None else f"{100 * x:.{nd}f}"


def static_tables() -> str:
    data = {label: {s: load(f"static_{key}_{s}.json") for s in ("validation", "test")} for label, key in MODELS}
    out = ["| Split | " + " | ".join(l for l, _ in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for s in ("validation", "test"):
        out.append(f"| {s} | " + " | ".join(p(data[l][s]["overall_accuracy"]) + "%" for l, _ in MODELS) + " |")
    out += ["", "| Task | " + " | ".join(f"{l} val / test" for l, _ in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for t in STATIC_TASKS:
        cells = [f"{p(data[l]['validation']['task_accuracy'][t])} / {p(data[l]['test']['task_accuracy'][t])}" for l, _ in MODELS]
        out.append(f"| {t} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def cond_table(split: str) -> str:
    a = {label: load(f"exact2x_{key}_analysis.json") for label, key in MODELS}
    conds = [
        "oracle_updated_graph", "predicted_updated_graph", "modify_and_print", "cached_no_reencode",
        "frozen_graph_history", "structure_only", "shuffled_graph", "graph_once_then_text",
        "token_matched_history", "question_only", "serialized_initial_history", "serialized_current_graph",
        "majority_prior", "tool_solver",
    ]
    out = ["| Condition | TEA | GraphToken |", "|---|---|---|"]
    for c in conds:
        cells = []
        for label in ("TEA", "GraphToken"):
            s = a[label][split]["summary"][c]
            lo, hi = s["answer_accuracy_ci95"]
            cells.append(f"{p(s['answer_accuracy'])} [{p(lo)}-{p(hi)}]")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    s = a["soft_prompt"][split]["summary"]["soft_prompt"]
    lo, hi = s["answer_accuracy_ci95"]
    out.append(f"| **soft_prompt (single condition, no graph)** | {p(s['answer_accuracy'])} [{p(lo)}-{p(hi)}] | (same run) |")
    return "\n".join(out)


def task_table(split: str) -> str:
    a = {label: load(f"exact2x_{key}_analysis.json") for label, key in MODELS}
    out = ["| Task | TEA (oracle graph) | GraphToken (oracle graph) | soft_prompt | question_only (TEA) | n turns |", "|---|---|---|---|---|---|"]
    for t in MAIN_TASKS:
        tea = a["TEA"][split]["cuts"]["oracle_updated_graph::task"][t]
        gt = a["GraphToken"][split]["cuts"]["oracle_updated_graph::task"][t]
        sp = a["soft_prompt"][split]["cuts"]["soft_prompt::task"][t]
        qo = a["TEA"][split]["cuts"]["question_only::task"][t]
        out.append(f"| {t} | {p(tea['acc'])} | {p(gt['acc'])} | {p(sp['acc'])} | {p(qo['acc'])} | {tea['n']} |")
    return "\n".join(out)


def density_table() -> str:
    a = {label: load(f"exact2x_{key}_analysis.json") for label, key in MODELS}
    cond = {"TEA": "oracle_updated_graph", "GraphToken": "oracle_updated_graph", "soft_prompt": "soft_prompt"}
    out = ["| Model | Split | sparse | medium | dense |", "|---|---|---|---|---|"]
    for label, _ in MODELS:
        for split in ("validation", "test"):
            c = a[label][split]["cuts"][f"{cond[label]}::density"]
            out.append(f"| {label} | {split} | {p(c['sparse']['acc'])} | {p(c['medium']['acc'])} | {p(c['dense']['acc'])} |")
    out += ["", "| Model | Split | small (16-24) | medium (25-32) | large (40-48) |", "|---|---|---|---|---|"]
    for label, _ in MODELS:
        for split in ("validation", "test"):
            c = a[label][split]["cuts"][f"{cond[label]}::scale"]
            out.append(f"| {label} | {split} | {p(c['scale_small']['acc'])} | {p(c['scale_medium']['acc'])} | {p(c['scale_large']['acc'])} |")
    return "\n".join(out)


def pathcost_table() -> str:
    out = ["| Condition | TEA val / test | GraphToken val / test | soft_prompt val / test |", "|---|---|---|---|"]
    a = {label: load(f"pathcost_{key}_analysis.json") for label, key in MODELS}
    conds = ["oracle_updated_graph", "frozen_graph_history", "structure_only", "shuffled_graph", "question_only", "majority_prior"]
    for c in conds:
        cells = []
        for label in ("TEA", "GraphToken", "soft_prompt"):
            if a[label] is None or label == "soft_prompt":
                cells.append("n/a" if label == "soft_prompt" else "pending")
                continue
            cells.append(f"{p(a[label]['validation']['summary'][c]['answer_accuracy'])} / {p(a[label]['test']['summary'][c]['answer_accuracy'])}")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    if a["soft_prompt"]:
        s = a["soft_prompt"]
        out.append(f"| soft_prompt | | | {p(s['validation']['summary']['soft_prompt']['answer_accuracy'])} / {p(s['test']['summary']['soft_prompt']['answer_accuracy'])} |")
    else:
        out.append("| soft_prompt | | | pending |")
    return "\n".join(out)


def answer_changing_table() -> str:
    d = load("answer_changing.json")
    out = ["| Model / condition | turns whose answer changed (n=2963) | turns whose answer did not change (n=6397) |", "|---|---|---|"]
    for label, key in MODELS:
        conds = ("oracle_updated_graph", "frozen_graph_history", "question_only") if key != "soft_prompt" else ("soft_prompt",)
        for c in conds:
            s = d[key]["test"][c]
            out.append(f"| {label} / {c} | {p(s['chg'][0])} | {p(s['same'][0])} |")
    return "\n".join(out)


if __name__ == "__main__":
    for title, fn in (
        ("STATIC", static_tables), ("COND-TEST", lambda: cond_table("test")), ("COND-VAL", lambda: cond_table("validation")),
        ("TASK-TEST", lambda: task_table("test")), ("TASK-VAL", lambda: task_table("validation")),
        ("DENSITY", density_table), ("PATHCOST", pathcost_table), ("ANSWER-CHANGING", answer_changing_table),
    ):
        print(f"\n### {title}\n{fn()}")
