"""How much of each model's score does a node-local shortcut explain?

Fits a gradient-boosted classifier per task on the *static train* split using only
  - degree of the source and target station,
  - open/closed status of the source and target station,
  - graph size, and whether the endpoints share a line.
It sees no adjacency, no attributes beyond `status`/`line`, and no text. Then it
  (1) reports its accuracy next to the models' on the static test split,
  (2) re-scores each model's saved rows on the items where the shortcut is wrong ("hard"),
  (3) does the same for the multi-turn sessions, applying the shortcut to the updated graph
      (what a status/degree tracker would know) and to the stale initial graph.

    python scripts/shortcut_audit.py --out shortcut_audit.json \
        static:gt_sum=outputs/audit/gt_sum/static_eval_test.json ... \
        multi:gt_sum=outputs/audit/gt_sum_exact2x/evaluation.json ...

Tasks with a queried attribute (attribute_lookup / attribute_check / most_common_attribute_within_hops)
are excluded: this baseline has no attribute features.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402

from graph_modi.data.multiturn import load_sessions  # noqa: E402
from graph_modi.data.v2 import load_static_tuples  # noqa: E402
from graph_modi.evaluation.metrics import answers_match, normalize_answer  # noqa: E402
from graph_modi.pipeline.multiturn import materialize_states  # noqa: E402

ATTRIBUTE_TASKS = {"attribute_lookup", "attribute_check", "most_common_attribute_within_hops"}


def features(graph, query) -> list[float]:
    degree: dict[str, int] = defaultdict(int)
    for edge in graph.edges:
        degree[edge.source] += 1
        degree[edge.target] += 1
    nodes = {node.id: node for node in graph.nodes}

    def of(node_id):
        node = nodes.get(node_id) if node_id is not None else None
        if node is None:
            return -1.0, -1.0, None
        status = 1.0 if node.attributes.get("status") == "closed" else 0.0
        return float(degree[node_id]), status, node.attributes.get("line")

    d_src, s_src, line_src = of(query.source)
    d_tgt, s_tgt, line_tgt = of(query.target)
    same_line = -1.0 if line_src is None or line_tgt is None else float(line_src == line_tgt)
    return [d_src, d_tgt, s_src, s_tgt, float(len(graph.nodes)), same_line]


class Tracker:
    """One gradient-boosted classifier per task over the shortcut features."""

    def __init__(self, train):
        by_task = defaultdict(lambda: ([], []))
        for item in train:
            task = item.query.reasoning_type.value
            if task in ATTRIBUTE_TASKS:
                continue
            xs, ys = by_task[task]
            xs.append(features(item.graph, item.query))
            ys.append(normalize_answer(item.answer))
        self.models = {}
        for task, (xs, ys) in by_task.items():
            classes = sorted(set(ys))
            if len(classes) == 1:
                self.models[task] = ("const", classes[0])
                continue
            model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0)
            model.fit(np.array(xs), np.array(ys))
            self.models[task] = ("gbm", model)

    def predict(self, task: str, graph, query):
        kind, model = self.models[task]
        if kind == "const":
            return model
        return str(model.predict(np.array([features(graph, query)]))[0])


def acc(flags) -> float:
    flags = list(flags)
    return sum(flags) / len(flags) if flags else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/metro_v3")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("specs", nargs="+", help="static:label=path | multi:label=path")
    args = parser.parse_args()
    data = Path(args.data)

    tracker = Tracker(load_static_tuples(data / "static_train.jsonl"))
    report: dict = {"static": {}, "multi": {}}

    # ---- static --------------------------------------------------------------------------
    test = {
        t.tuple_id: t
        for t in load_static_tuples(data / f"static_{args.split}.jsonl")
        if t.query.reasoning_type.value not in ATTRIBUTE_TASKS
    }
    shortcut_ok = {}
    for tid, item in test.items():
        task = item.query.reasoning_type.value
        shortcut_ok[tid] = answers_match(
            item.answer, tracker.predict(task, item.graph, item.query), reasoning_type=task
        )
    per_task = defaultdict(list)
    for tid, ok in shortcut_ok.items():
        per_task[test[tid].query.reasoning_type.value].append(ok)
    report["static"]["shortcut"] = {
        "overall": acc(shortcut_ok.values()),
        "task_mean": acc(acc(v) for v in per_task.values()),
        "per_task": {t: acc(v) for t, v in sorted(per_task.items())},
    }

    # hard slices named by the audit
    def slice_ids(name):
        out = []
        for tid, item in test.items():
            task = item.query.reasoning_type.value
            f = features(item.graph, item.query)
            if name == "cycle_deg>=2" and task == "cycle_membership" and f[0] >= 2:
                out.append(tid)
            if name == "reach_both_open" and task == "reachability" and f[2] == 0 and f[3] == 0:
                out.append(tid)
            if (
                name == "reach_endpoint_closed"
                and task == "reachability"
                and (f[2] == 1 or f[3] == 1)
            ):
                out.append(tid)
        return out

    slices = {n: slice_ids(n) for n in ("cycle_deg>=2", "reach_both_open", "reach_endpoint_closed")}
    for spec in [s for s in args.specs if s.startswith("static:")]:
        label, _, path = spec[len("static:") :].partition("=")
        rows = {r["tuple_id"]: r for r in json.loads(Path(path).read_text())["rows"]}
        rows = {tid: r for tid, r in rows.items() if tid in test}
        correct = {tid: bool(r["correct"]) for tid, r in rows.items()}
        hard = [tid for tid in correct if not shortcut_ok[tid]]
        easy = [tid for tid in correct if shortcut_ok[tid]]
        by_task_hard = defaultdict(list)
        for tid in hard:
            by_task_hard[test[tid].query.reasoning_type.value].append(correct[tid])
        report["static"][label] = {
            "overall": acc(correct.values()),
            "on_shortcut_wrong_items": acc(correct[t] for t in hard),
            "n_hard": len(hard),
            "on_shortcut_right_items": acc(correct[t] for t in easy),
            "hard_per_task": {t: [acc(v), len(v)] for t, v in sorted(by_task_hard.items())},
            "slices": {
                n: [acc(correct[t] for t in ids if t in correct), len(ids)]
                for n, ids in slices.items()
            },
        }

    # ---- multi-turn ----------------------------------------------------------------------
    sessions = [s for s in load_sessions(data / f"{args.split}.jsonl")]
    updated_ok, stale_ok, key_task = {}, {}, {}
    for session in sessions:
        for turn, state in zip(session.turns, materialize_states(session), strict=True):
            task = turn.query.reasoning_type.value
            if task in ATTRIBUTE_TASKS or task not in tracker.models:
                continue
            key = (session.session_id, turn.turn_index)
            key_task[key] = task
            updated_ok[key] = answers_match(
                turn.gold_answer,
                tracker.predict(task, state.after, turn.query),
                reasoning_type=task,
            )
            stale_ok[key] = answers_match(
                turn.gold_answer,
                tracker.predict(task, session.initial_graph, turn.query),
                reasoning_type=task,
            )
    report["multi"]["shortcut"] = {
        "updated_graph": acc(updated_ok.values()),
        "stale_graph": acc(stale_ok.values()),
        "turns": len(updated_ok),
    }
    hard_turns = {k for k, ok in updated_ok.items() if not ok}
    for spec in [s for s in args.specs if s.startswith("multi:")]:
        label, _, path = spec[len("multi:") :].partition("=")
        rows = json.loads(Path(path).read_text())[args.split]["rows"]
        per = defaultdict(lambda: {"all": [], "hard": [], "easy": []})
        for r in rows:
            key = (r["session_id"], r["turn_index"])
            if key not in updated_ok:
                continue
            bucket = per[r["condition"]]
            bucket["all"].append(bool(r["answer_correct"]))
            bucket["hard" if key in hard_turns else "easy"].append(bool(r["answer_correct"]))
        report["multi"][label] = {
            c: {k: [acc(v), len(v)] for k, v in b.items()}
            for c, b in sorted(per.items())
            if c
            in (
                "oracle_updated_graph",
                "frozen_graph_history",
                "shuffled_graph",
                "structure_only",
                "soft_prompt",
                "soft_prompt_no_history",
            )
        }
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        "static shortcut baseline: overall %.3f, task-mean %.3f"
        % (report["static"]["shortcut"]["overall"], report["static"]["shortcut"]["task_mean"])
    )
    for t, v in report["static"]["shortcut"]["per_task"].items():
        print(f"   {t:28s} {100 * v:5.1f}")
    for label, v in report["static"].items():
        if label == "shortcut":
            continue
        print(
            f"{label:16s} overall {100 * v['overall']:5.1f} | shortcut-wrong items {100 * v['on_shortcut_wrong_items']:5.1f} (n={v['n_hard']}) "
            f"| shortcut-right {100 * v['on_shortcut_right_items']:5.1f} | "
            + " ".join(f"{n}={100 * a:.0f}(n={m})" for n, (a, m) in v["slices"].items())
        )
    m = report["multi"]["shortcut"]
    print(
        "multi-turn shortcut: updated graph %.3f, stale graph %.3f (%d turns, %d hard)"
        % (m["updated_graph"], m["stale_graph"], m["turns"], len(hard_turns))
    )
    for label, v in report["multi"].items():
        if label == "shortcut":
            continue
        for c, b in v.items():
            print(
                f"   {label:16s} {c:22s} all {100 * b['all'][0]:5.1f} | hard turns {100 * b['hard'][0]:5.1f} (n={b['hard'][1]}) | easy {100 * b['easy'][0]:5.1f}"
            )


if __name__ == "__main__":
    main()
