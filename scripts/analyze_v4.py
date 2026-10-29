"""Pre-registered analysis of the frozen v4.1 runs (CPU only).

  PYTHONPATH=src python3 scripts/analyze_v4.py --runs outputs \
      --out documents/experiments/results/v4_1-analysis/analysis.json
  PYTHONPATH=src python3 scripts/analyze_v4.py --shortcut-only     # benchmark ceilings, no runs

1. Learned shortcut ceiling: one gradient-boosted classifier per task, fit on static train, seeing
   only node-local features of the queried stations (degree, status, number of leaf neighbours,
   smallest neighbour degree, closed neighbours, hop radius, graph size, closed count). No
   adjacency beyond one hop, no text. Scored on static test and on the test sessions' updated
   graphs; this is the "degree / status reader" a GNN with a linear readout can reach.
2. For every run directory under --runs (evaluation_<ckpt>.json, static_eval_test_<ckpt>.json):
   accuracy with session-clustered bootstrap CIs, overall and on the pre-registered slices
   (answer-changing turns, mentions_edit == false, shortcut-wrong turns, paraphrase family,
   per task next to its shortcut ceiling).
3. Paired, session-clustered differences: within a run (oracle vs shuffled / frozen / structure
   only) and across runs (GraphToken vs soft prompt without history, warm start, random GNN).
4. Seed mean and spread for the GraphToken seeds.

Slices and comparisons are fixed in the run checklist before any v4.1 result exists.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402

from graph_modi.data.multiturn import load_sessions  # noqa: E402
from graph_modi.data.v2 import load_static_tuples  # noqa: E402
from graph_modi.evaluation.metrics import (  # noqa: E402
    answers_match,
    cluster_bootstrap_ci,
    normalize_answer,
    paired_cluster_bootstrap,
)
from graph_modi.pipeline.multiturn import materialize_states  # noqa: E402

ATTRIBUTE_TASKS = {"attribute_lookup", "attribute_check", "most_common_attribute_within_hops"}
GRAPH_CONDITIONS = (
    "oracle_updated_graph",
    "predicted_updated_graph",
    "structure_only",
    "frozen_graph_history",
    "shuffled_graph",
    "question_only",
)
WITHIN_RUN_PAIRS = (
    ("oracle_updated_graph", "shuffled_graph"),
    ("oracle_updated_graph", "frozen_graph_history"),
    ("oracle_updated_graph", "structure_only"),
    ("oracle_updated_graph", "predicted_updated_graph"),
)
# (left run prefix, left condition, right run prefix, right condition); prefixes match run dirs.
ACROSS_RUN_PAIRS = (
    ("v4_graphtoken_s", "oracle_updated_graph", "v4_soft_prompt", "soft_prompt_no_history"),
    ("v4_graphtoken_s", "oracle_updated_graph", "v4_soft_prompt", "soft_prompt"),
    ("v4_graphtoken_s42", "oracle_updated_graph", "v4_tea_random_sum", "oracle_updated_graph"),
    ("v4_graphtoken_warm_s42", "oracle_updated_graph", "v4_graphtoken_s42", "oracle_updated_graph"),
)


def features(graph, query) -> list[float]:
    neighbours: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        neighbours[edge.source].add(edge.target)
        neighbours[edge.target].add(edge.source)
    closed = {n.id for n in graph.nodes if n.attributes.get("status") == "closed"}

    def of(node_id: str | None) -> list[float]:
        if (
            node_id is None
            or node_id not in neighbours
            and node_id not in {n.id for n in graph.nodes}
        ):
            return [-1.0] * 5
        around = neighbours[node_id]
        degrees = [len(neighbours[other]) for other in around]
        return [
            float(len(around)),
            float(node_id in closed),
            float(sum(d == 1 for d in degrees)),
            float(min(degrees, default=0)),
            float(len(around & closed)),
        ]

    return [
        *of(query.source),
        *of(query.target),
        float(query.hops or 0),
        float(len(graph.nodes)),
        float(len(closed)),
    ]


class Shortcut:
    def __init__(self, train) -> None:
        by_task: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        for item in train:
            task = item.query.reasoning_type.value
            if task in ATTRIBUTE_TASKS:
                continue
            xs, ys = by_task[task]
            xs.append(features(item.graph, item.query))
            ys.append(normalize_answer(item.answer))
        self.models: dict[str, object] = {}
        for task, (xs, ys) in by_task.items():
            if len(set(ys)) == 1:
                self.models[task] = ys[0]
                continue
            model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0)
            self.models[task] = model.fit(np.array(xs), np.array(ys))

    def correct(self, task: str, graph, query, gold) -> bool | None:
        model = self.models.get(task)
        if model is None:
            return None
        guess = (
            model
            if isinstance(model, str)
            else str(model.predict(np.array([features(graph, query)]))[0])
        )
        return answers_match(gold, guess, reasoning_type=task)


def acc(flags) -> float | None:
    flags = list(flags)
    return sum(flags) / len(flags) if flags else None


def summarize(rows: list[dict], resamples: int) -> dict:
    if not rows:
        return {"n": 0}
    low, high = cluster_bootstrap_ci(rows, resamples=resamples)
    return {"acc": acc(r["answer_correct"] for r in rows), "ci95": [low, high], "n": len(rows)}


def turn_metadata(sessions, shortcut: Shortcut) -> dict[tuple[str, int], dict]:
    meta = {}
    for session in sessions:
        for turn, state in zip(session.turns, materialize_states(session), strict=True):
            task = turn.query.reasoning_type.value
            meta[(session.session_id, turn.turn_index)] = {
                "task": task,
                "changed": normalize_answer(turn.gold_answer)
                != normalize_answer(turn.stale_answer),
                "mentions_edit": bool(turn.complexity.get("mentions_edit")),
                "family": str(turn.complexity.get("paraphrase_family", "?")),
                "session_length": int(turn.complexity.get("session_length", 0)),
                "shortcut_ok": shortcut.correct(task, state.after, turn.query, turn.gold_answer),
            }
    return meta


SLICES = {
    "all": lambda m: True,
    "answer_changing": lambda m: m["changed"],
    "answer_unchanged": lambda m: not m["changed"],
    "no_mention": lambda m: not m["mentions_edit"],
    "mention": lambda m: m["mentions_edit"],
    "shortcut_wrong": lambda m: m["shortcut_ok"] is False,
    "shortcut_right": lambda m: m["shortcut_ok"] is True,
    "changed_and_no_mention": lambda m: m["changed"] and not m["mentions_edit"],
    "family_direct": lambda m: m["family"] == "direct",
    "family_service": lambda m: m["family"] == "service",
    "family_notice": lambda m: m["family"] == "notice",
}


def sliced(rows: list[dict], meta: dict, resamples: int) -> dict:
    keyed = [
        (meta[(r["session_id"], r["turn_index"])], r)
        for r in rows
        if (r["session_id"], r["turn_index"]) in meta
    ]
    out = {
        name: summarize([r for m, r in keyed if keep(m)], resamples)
        for name, keep in SLICES.items()
    }
    by_task = defaultdict(list)
    for m, r in keyed:
        by_task[m["task"]].append(r)
    out["per_task"] = {
        task: summarize(task_rows, resamples) for task, task_rows in sorted(by_task.items())
    }
    return out


def static_report(path: Path, test_items: dict, shortcut_ok: dict) -> dict:
    rows = json.loads(path.read_text())["rows"]
    correct = {r["tuple_id"]: bool(r["correct"]) for r in rows}
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["reasoning_type"]].append(bool(r["correct"]))
    hard = [t for t in correct if shortcut_ok.get(t) is False]
    return {
        "overall": acc(correct.values()),
        "task_mean": acc(acc(v) for v in by_task.values()),
        "per_task": {t: acc(v) for t, v in sorted(by_task.items())},
        "shortcut_wrong": {"acc": acc(correct[t] for t in hard), "n": len(hard)},
        "n": len(correct),
    }


def condition_rows(evaluation: dict, split: str) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in evaluation.get(split, {}).get("rows", []):
        grouped[row["condition"]].append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/metro_v4_1")
    parser.add_argument("--runs", default="outputs", help="directory holding the v4_* run dirs")
    parser.add_argument("--checkpoint", default="best", help="checkpoint tag of the session eval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--shortcut-only", action="store_true")
    parser.add_argument(
        "--out", default="documents/experiments/results/v4_1-analysis/analysis.json"
    )
    args = parser.parse_args()
    data = Path(args.data)

    shortcut = Shortcut(load_static_tuples(data / "static_train.jsonl"))
    test_items = {t.tuple_id: t for t in load_static_tuples(data / f"static_{args.split}.jsonl")}
    static_ok = {
        tid: shortcut.correct(t.query.reasoning_type.value, t.graph, t.query, t.answer)
        for tid, t in test_items.items()
    }
    per_task_static = defaultdict(list)
    for tid, ok in static_ok.items():
        if ok is not None:
            per_task_static[test_items[tid].query.reasoning_type.value].append(ok)
    sessions = load_sessions(data / f"sessions_{args.split}.jsonl")
    meta = turn_metadata(sessions, shortcut)
    per_task_multi = defaultdict(list)
    for m in meta.values():
        if m["shortcut_ok"] is not None:
            per_task_multi[m["task"]].append(m["shortcut_ok"])
    report: dict = {
        "shortcut": {
            "features": "src/tgt degree, status, leaf neighbours, min neighbour degree, closed "
            "neighbours; hop radius; graph size; closed count",
            "static_overall_non_attribute": acc(v for v in static_ok.values() if v is not None),
            "static_per_task": {t: acc(v) for t, v in sorted(per_task_static.items())},
            "sessions_updated_graph_non_attribute": acc(
                m["shortcut_ok"] for m in meta.values() if m["shortcut_ok"] is not None
            ),
            "sessions_per_task": {t: acc(v) for t, v in sorted(per_task_multi.items())},
        },
        "slice_sizes": {name: sum(keep(m) for m in meta.values()) for name, keep in SLICES.items()},
        "runs": {},
        "paired": {},
        "seeds": {},
    }

    if not args.shortcut_only:
        runs_root = Path(args.runs)
        all_rows: dict[str, dict[str, list[dict]]] = {}
        for run in sorted(p for p in runs_root.glob("v4_*") if p.is_dir()):
            entry: dict = {}
            for ckpt in ("best", "final"):
                path = run / f"static_eval_{args.split}_{ckpt}.json"
                if path.exists():
                    entry[f"static_{ckpt}"] = static_report(path, test_items, static_ok)
            eval_path = run / f"evaluation_{args.checkpoint}.json"
            if not eval_path.exists():
                eval_path = run / "evaluation_final.json"
            if eval_path.exists():
                evaluation = json.loads(eval_path.read_text())
                grouped = condition_rows(evaluation, args.split)
                all_rows[run.name] = grouped
                entry["sessions_file"] = eval_path.name
                entry["provenance"] = evaluation.get(args.split, {}).get("provenance")
                entry["sessions"] = {
                    c: sliced(rows, meta, args.resamples) for c, rows in sorted(grouped.items())
                }
                entry["within_run_paired"] = {
                    f"{a} - {b}": paired_cluster_bootstrap(
                        grouped[a], grouped[b], resamples=args.resamples
                    )
                    for a, b in WITHIN_RUN_PAIRS
                    if grouped.get(a) and grouped.get(b)
                }
            if entry:
                report["runs"][run.name] = entry

        for left_prefix, left_cond, right_prefix, right_cond in ACROSS_RUN_PAIRS:
            rights = [name for name in all_rows if name.startswith(right_prefix)]
            for left in (name for name in all_rows if name.startswith(left_prefix)):
                for right in rights:
                    if (
                        left == right
                        or not all_rows[left].get(left_cond)
                        or not all_rows[right].get(right_cond)
                    ):
                        continue
                    for slice_name in ("all", "answer_changing", "no_mention", "shortcut_wrong"):
                        keep = SLICES[slice_name]
                        pick = [
                            r
                            for r in all_rows[left][left_cond]
                            if keep(meta[(r["session_id"], r["turn_index"])])
                        ]
                        report["paired"][
                            f"{left}:{left_cond} - {right}:{right_cond} [{slice_name}]"
                        ] = paired_cluster_bootstrap(
                            pick, all_rows[right][right_cond], resamples=args.resamples
                        )

        seed_runs = [n for n in report["runs"] if n.startswith("v4_graphtoken_s")]
        for condition in GRAPH_CONDITIONS:
            for slice_name in ("all", "answer_changing", "no_mention", "shortcut_wrong"):
                values = [
                    report["runs"][n]["sessions"][condition][slice_name]["acc"]
                    for n in seed_runs
                    if condition in report["runs"][n].get("sessions", {})
                ]
                if values:
                    report["seeds"][f"{condition} [{slice_name}]"] = {
                        "runs": seed_runs,
                        "mean": statistics.mean(values),
                        "sd": statistics.stdev(values) if len(values) > 1 else None,
                        "values": values,
                    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    s = report["shortcut"]
    print(
        f"learned shortcut: static {100 * s['static_overall_non_attribute']:.1f}  "
        f"sessions {100 * s['sessions_updated_graph_non_attribute']:.1f}  (non-attribute tasks)"
    )
    for task, value in s["static_per_task"].items():
        print(
            f"   {task:34s} static {100 * value:5.1f}  sessions {100 * (s['sessions_per_task'].get(task) or 0):5.1f}"
        )
    print("slice sizes:", report["slice_sizes"])
    for name, entry in report["runs"].items():
        line = f"{name:28s}"
        for ckpt in ("best", "final"):
            if f"static_{ckpt}" in entry:
                line += f" static_{ckpt} {100 * entry[f'static_{ckpt}']['overall']:5.1f}"
        for condition in ("oracle_updated_graph", "soft_prompt", "soft_prompt_no_history"):
            got = entry.get("sessions", {}).get(condition, {}).get("all", {})
            if got.get("acc") is not None:
                line += f" | {condition} {100 * got['acc']:5.1f}"
        print(line)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
