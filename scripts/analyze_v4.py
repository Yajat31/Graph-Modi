"""Pre-registered analysis of the frozen v4.1 runs (CPU only).

  PYTHONPATH=src python3 scripts/analyze_v4.py --runs outputs \
      --out documents/experiments/results/v4_1-analysis/analysis.json
  PYTHONPATH=src python3 scripts/analyze_v4.py --shortcut-only     # benchmark ceilings, no runs

1. Learned shortcut ceiling: one gradient-boosted classifier per task, fit on static train, seeing
   only node-local features of the queried stations (degree, status, number of leaf neighbours,
   smallest neighbour degree, closed neighbours, hop radius, graph size, closed count). No
   adjacency beyond one hop, no text. Scored on static test and on the test sessions' updated
   graphs; this is the "degree / status reader" a GNN with a linear readout can reach.
   A second classifier adds two-hop features (nodes, leaves and closed stations two hops away,
   triangles, max / summed neighbour degree, common neighbours). A 3-layer GNN sees all of these,
   so the per-task local ceiling is the better of the two, and "local-wrong" turns (both wrong) are
   the evidence for graph use beyond local reading (amended 2026-10-30, before any v4.1 result).
2. Text baselines fit on validation sessions, no graph: "text_only" (what the revision text says
   about the queried stations, this turn and earlier) and "stale_plus_text" (the pre-edit answer
   plus that text). Revisions are templated and parse exactly into the gold edits, so the gold
   edits stand in for the text.
3. For every run directory under --runs (evaluation_<ckpt>.json, static_eval_test_<ckpt>.json):
   accuracy with session-clustered bootstrap CIs, overall and on the pre-registered slices
   (answer-changing turns, answer-changing turns that do not mention the edit, mentions_edit ==
   false, shortcut-wrong and local-wrong turns, paraphrase family, per task next to its ceiling).
   Do not slice reachability or shortest_path by hop_depth: that tag encodes the answer.
4. Paired, session-clustered differences: within a run (oracle vs shuffled / frozen / structure
   only) and across runs (GraphToken vs soft prompt without history, warm start, random GNN).
5. Seed mean and spread for the GraphToken seeds.

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


def features_two_hop(graph, query) -> list[float]:
    """One-hop features plus two-hop structure around each queried station (no distances)."""
    neighbours: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        neighbours[edge.source].add(edge.target)
        neighbours[edge.target].add(edge.source)
    closed = {n.id for n in graph.nodes if n.attributes.get("status") == "closed"}
    ids = {n.id for n in graph.nodes}

    def of(node_id: str | None) -> list[float]:
        if node_id is None or node_id not in ids:
            return [-1.0] * 6
        around = neighbours[node_id]
        ring2 = (
            set().union(*(neighbours[m] for m in around)) - around - {node_id} if around else set()
        )
        around_list = sorted(around)
        triangles = sum(
            1 for i, a in enumerate(around_list) for b in around_list[i + 1 :] if b in neighbours[a]
        )
        return [
            float(len(ring2)),
            float(sum(len(neighbours[m]) == 1 for m in ring2)),
            float(triangles),
            float(max((len(neighbours[m]) for m in around), default=0)),
            float(sum(len(neighbours[m]) for m in around)),
            float(len(ring2 & closed)),
        ]

    common = (
        float(len(neighbours[query.source] & neighbours[query.target]))
        if query.target is not None
        else -1.0
    )
    return [*features(graph, query), *of(query.source), *of(query.target), common]


class Shortcut:
    def __init__(self, train, featurize=features) -> None:
        self.featurize = featurize
        by_task: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        for item in train:
            task = item.query.reasoning_type.value
            if task in ATTRIBUTE_TASKS:
                continue
            xs, ys = by_task[task]
            xs.append(self.featurize(item.graph, item.query))
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
            else str(model.predict(np.array([self.featurize(graph, query)]))[0])
        )
        return answers_match(gold, guess, reasoning_type=task)


def _edit_events(turn, labels: dict[str, str]) -> list[tuple]:
    """(kind, station...) events the revision text states; kinds close/open/add/del."""
    events = []
    for edit in turn.gold_edits or (turn.gold_edit,):
        op = edit.operation.value.upper()
        if op == "SET":
            events.append(("close" if edit.value == "closed" else "open", labels[edit.node_id]))
        elif op in ("ADD", "DEL"):
            events.append((op.lower(), labels[edit.source], labels[edit.destination]))
    return events


def text_features(session) -> dict[int, dict]:
    """Per turn: what the revision history says about the queried station(s), no graph."""
    labels = {n.id: n.label for n in session.initial_graph.nodes}
    history: list[list[tuple]] = []
    out = {}
    for turn in session.turns:
        events = _edit_events(turn, labels)
        src = labels.get(turn.query.source)
        tgt = labels.get(turn.query.target) if turn.query.target else None

        def current(label):
            if label is None:
                return "-"
            return "+".join(sorted({e[0] for e in events if label in e[1:]})) or "none"

        def previous(label):
            if label is None:
                return "-"
            for past in reversed(history):
                kinds = sorted({e[0] for e in past if label in e[1:]})
                if kinds:
                    return "+".join(kinds)
            return "none"

        def pair(event_lists):
            if tgt is None:
                return "-"
            for past in reversed(event_lists):
                for e in past:
                    if e[0] in ("add", "del") and {e[1], e[2]} == {src, tgt}:
                        return e[0]
            return "none"

        out[turn.turn_index] = {
            "sc": current(src),
            "tc": current(tgt),
            "sp": previous(src),
            "tp": previous(tgt),
            "pc": pair([events]),
            "pp": pair(history),
            "noop": not events,
        }
        history.append(events)
    return out


class TextLookup:
    """Back-off lookup table over text features, fit on validation sessions (per task)."""

    TEXT = [
        ("sc", "tc", "sp", "tp", "pc", "pp", "noop"),
        ("sc", "tc", "pc", "pp"),
        ("sc", "tc"),
        (),
    ]
    STALE = [
        ("stale", "sc", "tc", "sp", "tp", "pc", "pp", "noop"),
        ("stale", "sc", "tc", "pc"),
        ("stale", "sc", "tc"),
        ("stale",),
        (),
    ]

    def __init__(self, ladder, min_count: int = 3) -> None:
        self.ladder, self.min_count = ladder, min_count
        self.tables = [defaultdict(lambda: defaultdict(int)) for _ in ladder]

    def fit(self, rows) -> TextLookup:
        for f, y in rows:
            for keys, table in zip(self.ladder, self.tables):
                table[tuple(f[k] for k in keys)][y] += 1
        return self

    def predict(self, f) -> str:
        for keys, table in zip(self.ladder, self.tables):
            counts = table.get(tuple(f[k] for k in keys))
            if counts and sum(counts.values()) >= self.min_count:
                return max(sorted(counts), key=counts.get)
        last = self.tables[-1][()]
        return max(sorted(last), key=last.get)


def text_baseline_rows(val_sessions, test_sessions) -> dict[str, list[dict]]:
    """Pseudo-condition rows (answer_correct per test turn) for the two text baselines."""

    def collect(sessions):
        rows = []
        for session in sessions:
            feats = text_features(session)
            for turn in session.turns:
                f = dict(feats[turn.turn_index], stale=normalize_answer(turn.stale_answer))
                rows.append((session.session_id, turn, f))
        return rows

    val_rows, test_rows = collect(val_sessions), collect(test_sessions)
    out: dict[str, list[dict]] = {"text_only": [], "stale_plus_text": []}
    for task in sorted({t.query.reasoning_type.value for _, t, _ in test_rows}):
        fit_rows = [
            (f, normalize_answer(t.gold_answer))
            for _, t, f in val_rows
            if t.query.reasoning_type.value == task
        ]
        if not fit_rows:
            continue
        models = {
            "text_only": TextLookup(TextLookup.TEXT).fit(fit_rows),
            "stale_plus_text": TextLookup(TextLookup.STALE).fit(fit_rows),
        }
        for sid, turn, f in test_rows:
            if turn.query.reasoning_type.value != task:
                continue
            for name, model in models.items():
                out[name].append(
                    {
                        "session_id": sid,
                        "turn_index": turn.turn_index,
                        "answer_correct": answers_match(
                            turn.gold_answer, model.predict(f), reasoning_type=task
                        ),
                    }
                )
    return out


def acc(flags) -> float | None:
    flags = list(flags)
    return sum(flags) / len(flags) if flags else None


def summarize(rows: list[dict], resamples: int) -> dict:
    if not rows:
        return {"n": 0}
    low, high = cluster_bootstrap_ci(rows, resamples=resamples)
    return {"acc": acc(r["answer_correct"] for r in rows), "ci95": [low, high], "n": len(rows)}


def turn_metadata(sessions, shortcut: Shortcut, shortcut2: Shortcut) -> dict[tuple[str, int], dict]:
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
                "shortcut2_ok": shortcut2.correct(task, state.after, turn.query, turn.gold_answer),
            }
    return meta


# Slices used for paired comparisons and seed summaries (pre-registered).
HEADLINE_SLICES = (
    "all",
    "answer_changing",
    "changed_and_no_mention",
    "no_mention",
    "shortcut_wrong",
    "local_wrong",
)

SLICES = {
    "all": lambda m: True,
    "answer_changing": lambda m: m["changed"],
    "answer_unchanged": lambda m: not m["changed"],
    "no_mention": lambda m: not m["mentions_edit"],
    "mention": lambda m: m["mentions_edit"],
    "shortcut_wrong": lambda m: m["shortcut_ok"] is False,
    "shortcut_right": lambda m: m["shortcut_ok"] is True,
    "changed_and_no_mention": lambda m: m["changed"] and not m["mentions_edit"],
    # wrong under both local models (one-hop and two-hop): the main graph-use slice
    "local_wrong": lambda m: m["shortcut_ok"] is False and m["shortcut2_ok"] is False,
    "local_right": lambda m: m["shortcut_ok"] is True or m["shortcut2_ok"] is True,
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


def static_report(path: Path, test_items: dict, shortcut_ok: dict, local_ok: dict) -> dict:
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
        "local_wrong": {
            "acc": acc(correct[t] for t in correct if local_ok.get(t) is False),
            "n": sum(local_ok.get(t) is False for t in correct),
        },
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

    train_items = load_static_tuples(data / "static_train.jsonl")
    shortcut = Shortcut(train_items)
    shortcut2 = Shortcut(train_items, featurize=features_two_hop)
    test_items = {t.tuple_id: t for t in load_static_tuples(data / f"static_{args.split}.jsonl")}
    static_ok = {
        tid: shortcut.correct(t.query.reasoning_type.value, t.graph, t.query, t.answer)
        for tid, t in test_items.items()
    }
    static_ok2 = {
        tid: shortcut2.correct(t.query.reasoning_type.value, t.graph, t.query, t.answer)
        for tid, t in test_items.items()
    }
    # local_ok: right under at least one local model; False only when both are wrong
    static_local_ok = {
        tid: None if static_ok[tid] is None else bool(static_ok[tid] or static_ok2[tid])
        for tid in test_items
    }
    per_task_static = defaultdict(list)
    per_task_static2 = defaultdict(list)
    for tid, ok in static_ok.items():
        if ok is not None:
            per_task_static[test_items[tid].query.reasoning_type.value].append(ok)
            per_task_static2[test_items[tid].query.reasoning_type.value].append(static_ok2[tid])
    sessions = load_sessions(data / f"sessions_{args.split}.jsonl")
    meta = turn_metadata(sessions, shortcut, shortcut2)
    per_task_multi = defaultdict(list)
    per_task_multi2 = defaultdict(list)
    for m in meta.values():
        if m["shortcut_ok"] is not None:
            per_task_multi[m["task"]].append(m["shortcut_ok"])
            per_task_multi2[m["task"]].append(m["shortcut2_ok"])

    def no_degree(per_task):
        return acc(v for t, vals in per_task.items() if t != "node_degree" for v in vals)

    val_sessions = load_sessions(data / "sessions_validation.jsonl")
    baselines = text_baseline_rows(val_sessions, sessions)
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
            "static_non_attribute_excl_node_degree": no_degree(per_task_static),
            "sessions_non_attribute_excl_node_degree": no_degree(per_task_multi),
        },
        "shortcut_two_hop": {
            "features": "one-hop features plus, per queried station, nodes / leaves / closed "
            "stations at two hops, triangles, max and summed neighbour degree; common neighbours",
            "static_overall_non_attribute": acc(
                v for vals in per_task_static2.values() for v in vals
            ),
            "static_per_task": {t: acc(v) for t, v in sorted(per_task_static2.items())},
            "sessions_updated_graph_non_attribute": acc(
                v for vals in per_task_multi2.values() for v in vals
            ),
            "sessions_per_task": {t: acc(v) for t, v in sorted(per_task_multi2.items())},
            "static_non_attribute_excl_node_degree": no_degree(per_task_static2),
            "sessions_non_attribute_excl_node_degree": no_degree(per_task_multi2),
        },
        "local_ceiling_per_task": {
            "static": {
                t: max(acc(per_task_static[t]), acc(per_task_static2[t])) for t in per_task_static
            },
            "sessions": {
                t: max(acc(per_task_multi[t]), acc(per_task_multi2[t])) for t in per_task_multi
            },
        },
        "text_baselines": {
            name: sliced(rows, meta, args.resamples) for name, rows in sorted(baselines.items())
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
                    entry[f"static_{ckpt}"] = static_report(
                        path, test_items, static_ok, static_local_ok
                    )
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
                    for slice_name in HEADLINE_SLICES:
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
            for slice_name in HEADLINE_SLICES:
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
    s2 = report["shortcut_two_hop"]
    print(
        f"two-hop local shortcut: static {100 * s2['static_overall_non_attribute']:.1f}  "
        f"sessions {100 * s2['sessions_updated_graph_non_attribute']:.1f}  (non-attribute tasks)"
    )
    for name, entry in report["text_baselines"].items():
        print(f"text baseline {name}: sessions {100 * entry['all']['acc']:.1f}")
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
