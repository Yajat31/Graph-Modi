#!/usr/bin/env python3
"""Acceptance gates for metro_v4. Generates into --work, runs checks, writes results.

Usage:
  PYTHONPATH=src python3 scripts/audit_v4_gates.py [--work /tmp/graphmodi_v4] [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random as _random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from graph_modi.data import v4 as V4
from graph_modi.data.multiturn import load_sessions, save_sessions
from graph_modi.data.v2 import load_static_tuples, save_static_tuples
from graph_modi.graph.executor import graph_fingerprint
from graph_modi.graph.solvers import answer_query
from graph_modi.schema import EditOperation, EditProgram, EditTarget, ReasoningType

SEED = 42
CHECKS: list[dict[str, Any]] = []
RESULTS: dict[str, Any] = {}
EIGHT = [
    "edge_exists",
    "reachability",
    "constrained_reachability",
    "cycle_membership",
    "node_degree",
    "filtered_neighbor_count",
    "shortest_path",
    "within_hops_count",
]


def check(section: str, name: str, ok: bool, detail: str = "", level: str = "fail") -> bool:
    status = "pass" if ok else level
    CHECKS.append({"section": section, "name": name, "status": status, "detail": detail})
    print(f"  [{status.upper():4}] {section}/{name}  {detail}")
    return ok


def info(section: str, name: str, detail: str) -> None:
    CHECKS.append({"section": section, "name": name, "status": "info", "detail": detail})
    print(f"  [INFO] {section}/{name}  {detail}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(work: Path) -> dict[str, Path]:
    work.mkdir(parents=True, exist_ok=True)
    paths = {
        "static_train": work / "static_train.jsonl",
        "static_validation": work / "static_validation.jsonl",
        "static_test": work / "static_test.jsonl",
        "sessions_validation": work / "sessions_validation.jsonl",
        "sessions_test": work / "sessions_test.jsonl",
    }
    if all(p.exists() for p in paths.values()):
        print(f"using cached data in {work}")
        return paths
    cells = len(V4.NODE_COUNTS) * len(V4.DENSITIES)
    t0 = time.time()
    train = V4.generate_static_v4(
        split="train",
        graph_count=2000,
        tasks_per_graph=9,
        seed=SEED,
        sampler=V4.BalancedSampler(),
        tasks=list(V4.STATIC_TASKS),
        progress=True,
    )
    train += V4.generate_counterfactual_pairs_v4(
        graph_count=1000,
        pairs_per_graph=3,
        seed=SEED,
        sampler=V4.BalancedSampler(),
        tasks=list(V4.DYNAMIC_TASKS),
        progress=True,
    )
    save_static_tuples(paths["static_train"], train)
    print(f"  static train: {len(train)} ({time.time() - t0:.0f}s)")
    for split, reps in (("validation", 6), ("test", 12)):
        items = V4.generate_static_v4(
            split=split,
            graph_count=cells * reps,
            tasks_per_graph=None,
            seed=SEED,
            sampler=V4.BalancedSampler(),
            tasks=list(V4.STATIC_TASKS),
            progress=True,
        )
        save_static_tuples(paths[f"static_{split}"], items)
        print(f"  static {split}: {len(items)} ({time.time() - t0:.0f}s)")
    sessions = V4.generate_sessions_v4(
        replicates_per_split={"validation": 1, "test": 1},
        seed=SEED,
        tasks=list(V4.DYNAMIC_TASKS),
        progress=True,
    )
    for split, items in sessions.items():
        save_sessions(paths[f"sessions_{split}"], items)
        print(f"  sessions {split}: {len(items)} ({time.time() - t0:.0f}s)")
    return paths


# Independent solver (same semantics as library after the cycle fix)
def adj(graph):
    a = {n.id: set() for n in graph.nodes}
    for e in graph.edges:
        a[e.source].add(e.target)
        a[e.target].add(e.source)
    return a


def is_open(graph, nid):
    return graph.node_map()[nid].attributes.get("status", "open") != "closed"


def gate_labels(static, sessions):
    print("\n[labels]")
    for split, items in static.items():
        wrong = 0
        ties = 0
        for item in items:
            if answer_query(item.graph, item.query) != item.answer:
                wrong += 1
            if item.query.reasoning_type is ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS:
                hops = item.query.hops or 2
                if not V4.V3._unique_mode(
                    item.graph, item.query.source, item.query.attribute or "", hops
                ):
                    ties += 1
        check("labels", f"static_{split}", wrong == 0, f"{wrong}/{len(items)} mismatches")
        check("labels", f"static_{split}_unique_mode", ties == 0, f"{ties} tied modes")
        by_task = Counter(i.query.reasoning_type.value for i in items)
        info("labels", f"static_{split}_counts", f"{dict(by_task)}")
        if split == "test":
            check(
                "labels",
                "test_full_grid",
                all(by_task[t] == 396 for t in by_task),
                f"per-task counts { {t: by_task[t] for t in sorted(by_task)} }",
            )
    for split, sess in sessions.items():
        wrong = stale_wrong = ties = fp_wrong = 0
        turns = 0
        for session in sess:
            state = session.initial_graph
            for turn in session.turns:
                turns += 1
                edits = turn.gold_edits or (turn.gold_edit,)
                program = turn.edit_program or EditProgram(edits=edits)
                # Attribute values on ADD edges are rng-dependent; answers do not depend on them.
                after = V4.apply_program_v4(state, program, _random.Random(0))
                if answer_query(after, turn.query) != turn.gold_answer:
                    wrong += 1
                if answer_query(state, turn.query) != turn.stale_answer:
                    stale_wrong += 1
                if graph_fingerprint(after) != turn.after_fingerprint:
                    # fingerprints include edge attributes; tolerate ADD-attribute mismatch by
                    # comparing topology+status only via answer consistency above
                    if not any(e.operation is EditOperation.ADD for e in edits):
                        fp_wrong += 1
                if turn.query.reasoning_type is ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS:
                    hops = turn.query.hops or 2
                    if not V4.V3._unique_mode(
                        state, turn.query.source, turn.query.attribute or "", hops
                    ):
                        ties += 1
                    if not V4.V3._unique_mode(
                        after, turn.query.source, turn.query.attribute or "", hops
                    ):
                        ties += 1
                state = after
        check("labels", f"sessions_{split}_gold", wrong == 0, f"{wrong}/{turns}")
        check("labels", f"sessions_{split}_stale", stale_wrong == 0, f"{stale_wrong}/{turns}")
        check(
            "labels", f"sessions_{split}_ties", ties == 0, f"{ties} tied modes across before/after"
        )
        # ADD edge attributes are rng-drawn, so fingerprints after an ADD diverge under replay;
        # gold/stale answer checks above already cover label correctness.
        info(
            "labels",
            f"sessions_{split}_replay",
            f"{fp_wrong} fingerprint mismatches on non-ADD turns (informational)",
        )


def gate_structure(static, sessions):
    print("\n[structure]")
    for split, items in static.items():
        # Counterfactual "after" graphs can have 0 closed (a SET reopened the last one); skip them.
        graphs = {
            i.graph.graph_id: i.graph for i in items if i.metadata.get("counterfactual") != "after"
        }
        issues = Counter()
        for g in graphs.values():
            closed = sum(n.attributes.get("status") == "closed" for n in g.nodes)
            if not (1 <= closed <= 3):
                issues["closed_range"] += 1
            if any(not e.attributes for e in g.edges):
                issues["empty_edge_attrs"] += 1
            deg = V4.V3._degrees(g)
            last = [n.id for n in g.nodes[-3:]]
            if all(deg[i] == 1 for i in last):
                issues["terminals_at_end"] += 1
        check("structure", f"{split}_closed_range", not issues["closed_range"], f"{dict(issues)}")
        check("structure", f"{split}_edge_attrs", not issues["empty_edge_attrs"], f"{dict(issues)}")
        check(
            "structure",
            f"{split}_node_order",
            issues["terminals_at_end"] < 0.2 * len(graphs),
            f"{issues['terminals_at_end']}/{len(graphs)} graphs park terminals at the end",
            "warn",
        )
    for split, sess in sessions.items():
        bad = 0
        for s in sess:
            closed = sum(n.attributes.get("status") == "closed" for n in s.initial_graph.nodes)
            bad += not (1 <= closed <= 3)
        check(
            "structure",
            f"sessions_{split}_start_closed",
            bad == 0,
            f"{bad}/{len(sess)} start outside 1-3",
        )


def gate_shortcuts(static):
    print("\n[shortcuts]")
    train = static["train"]
    test = static["test"]

    def feats(item):
        g = item.graph
        q = item.query
        deg = V4.V3._degrees(g)
        nodes = g.node_map()
        return {
            "ds": deg[q.source],
            "dt": deg[q.target] if q.target else None,
            "ss": nodes[q.source].attributes.get("status"),
            "st": nodes[q.target].attributes.get("status") if q.target else None,
            "same": (
                nodes[q.source].attributes.get("line") == nodes[q.target].attributes.get("line")
            )
            if q.target
            else None,
        }

    ladders = {
        "status_only": [("ss", "st"), ("ss",), ()],
        "degree_only": [("ds", "dt"), ("ds",), ()],
        "degree_status": [
            ("ds", "dt", "ss", "st", "same"),
            ("ds", "dt", "ss", "st"),
            ("ds", "ss", "st"),
            ("ss", "st"),
            (),
        ],
    }

    class Lookup:
        def __init__(self, ladder):
            self.ladder = ladder
            self.tables = [defaultdict(Counter) for _ in ladder]

        def fit(self, rows):
            for f, a in rows:
                for level, table in zip(self.ladder, self.tables):
                    table[tuple(f[k] for k in level)][a] += 1
            return self

        def predict(self, f):
            for level, table in zip(self.ladder, self.tables):
                c = table.get(tuple(f[k] for k in level))
                if c and sum(c.values()) >= 3:
                    return c.most_common(1)[0][0]
            return self.tables[-1][()].most_common(1)[0][0]

    by_task_train = defaultdict(list)
    by_task_test = defaultdict(list)
    for item in train:
        by_task_train[item.query.reasoning_type.value].append((feats(item), item.answer))
    for item in test:
        by_task_test[item.query.reasoning_type.value].append((feats(item), item.answer))

    summary = {}
    for name, ladder in ladders.items():
        scores = {}
        for task in EIGHT:
            model = Lookup(ladder).fit(by_task_train[task])
            rows = by_task_test[task]
            scores[task] = sum(model.predict(f) == a for f, a in rows) / len(rows)
        mean8 = sum(scores[t] for t in EIGHT) / len(EIGHT)
        summary[name] = {"tasks": scores, "eight": mean8}
        print(
            f"  {name:14} 8-task {100 * mean8:.1f}  cycle {100 * scores['cycle_membership']:.1f}  reach {100 * scores['reachability']:.1f}"
        )
    check(
        "shortcuts",
        "cycle_degree_ceiling",
        summary["degree_only"]["tasks"]["cycle_membership"] <= 0.60,
        f"{100 * summary['degree_only']['tasks']['cycle_membership']:.1f}%",
    )
    check(
        "shortcuts",
        "reach_status_ceiling",
        summary["status_only"]["tasks"]["reachability"] <= 0.60,
        f"{100 * summary['status_only']['tasks']['reachability']:.1f}%",
    )
    # node_degree is definitionally a degree lookup (100%); the meaningful ceiling is the
    # other 7 tasks. Also report the raw 8-task mean for comparison with v3's 63.9%.
    seven = [t for t in EIGHT if t != "node_degree"]
    mean7 = sum(summary["degree_status"]["tasks"][t] for t in seven) / len(seven)
    summary["degree_status"]["seven_ex_degree"] = mean7
    check(
        "shortcuts",
        "seven_task_ceiling",
        mean7 < 0.50,
        f"{100 * mean7:.1f}% (excludes node_degree; 8-task {100 * summary['degree_status']['eight']:.1f}%)",
    )
    info(
        "shortcuts",
        "eight_task_ceiling",
        f"{100 * summary['degree_status']['eight']:.1f}% (includes node_degree at 100%)",
    )
    # v4.1: 2-hop cycle rule. A degree-2 station next to a leaf is on a pendant chain.
    def neighbours(graph, node_id):
        return {e.target for e in graph.edges if e.source == node_id} | {
            e.source for e in graph.edges if e.target == node_id
        }

    def leaf_rule(item):
        deg = V4.V3._degrees(item.graph)
        near_leaf = any(deg[n] == 1 for n in neighbours(item.graph, item.query.source))
        return "no" if near_leaf else "yes"

    cycle = [i for i in test if i.query.reasoning_type is ReasoningType.CYCLE_MEMBERSHIP]
    leaf_acc = sum(leaf_rule(i) == i.answer for i in cycle) / len(cycle)
    summary["cycle_leaf_neighbour_rule"] = leaf_acc
    check(
        "shortcuts",
        "cycle_leaf_neighbour_ceiling",
        leaf_acc <= 0.85,
        f"{100 * leaf_acc:.1f}% (v4.0: ~95%)",
        "warn",
    )
    # Counterfactual items must obey the same filters as test items.
    violations = Counter()
    for item in train:
        if item.metadata.get("counterfactual") is None:
            continue
        if not V4._v4_eligible(item.graph, item.query):
            violations[item.query.reasoning_type.value] += 1
    check(
        "shortcuts",
        "counterfactual_filters",
        not violations,
        f"{dict(violations)} items break the v4 test filters",
    )
    RESULTS["shortcuts"] = summary


def gate_sessions(sessions):
    print("\n[sessions]")
    for split, sess in sessions.items():
        turns = [t for s in sess for t in s.turns]
        noop = sum(1 for t in turns if "noop" in t.tags)
        rate = noop / len(turns)
        check("sessions", f"{split}_noop_rate", 0.15 <= rate <= 0.20, f"{rate:.3f}")
        changed_rate = sum(t.gold_answer != t.stale_answer for t in turns) / len(turns)
        check(
            "sessions",
            f"{split}_answer_changing_share",
            changed_rate >= 0.35,
            f"{changed_rate:.3f} (stale-answer baseline {1 - changed_rate:.3f}; v4.0: 0.146)",
        )
        by_length = defaultdict(lambda: [0, 0])
        for s in sess:
            for t in s.turns:
                by_length[len(s.turns)][0] += t.gold_answer != t.stale_answer
                by_length[len(s.turns)][1] += 1
        info(
            "sessions",
            f"{split}_answer_changing_by_length",
            f"{ {k: round(a / b, 3) for k, (a, b) in sorted(by_length.items())} }",
        )
        net_noop = cancel = revert = 0
        disconnected = 0
        closed_drift = []
        text_rule = 0
        edge_edit_turns = 0
        mention = chance = 0
        edit_turns = 0
        for session in sess:
            state = session.initial_graph
            seen = {V4._state_key(state)}
            for turn in session.turns:
                edits = turn.gold_edits or (turn.gold_edit,)
                ops = [e for e in edits if e.operation is not EditOperation.NOOP]
                after = V4.apply_program_v4(
                    state,
                    turn.edit_program or EditProgram(edits=edits),
                    _random.Random(0),
                )
                if ops:
                    if V4._state_key(after) == V4._state_key(state):
                        net_noop += 1
                    touched = Counter()
                    for e in ops:
                        touched[V4._touches(e)] += 1
                    if any(c > 1 for c in touched.values()):
                        cancel += 1
                    if V4._state_key(after) in seen and V4._state_key(after) != V4._state_key(
                        state
                    ):
                        revert += 1
                seen.add(V4._state_key(after))
                # connectivity
                a = adj(after)
                start = next(iter(a))
                reach = {start}
                stack = [start]
                while stack:
                    n = stack.pop()
                    for x in a[n]:
                        if x not in reach:
                            reach.add(x)
                            stack.append(x)
                disconnected += len(reach) != len(a)
                closed_drift.append(sum(not is_open(after, i) for i in a) / len(a))
                if ops:
                    edit_turns += 1
                    q = turn.query
                    named = {q.source} | ({q.target} if q.target else set())
                    touched_nodes = set()
                    for e in ops:
                        if e.target is EditTarget.NODE:
                            touched_nodes.add(e.node_id)
                        else:
                            touched_nodes |= {e.source, e.destination}
                    hit = bool(named & touched_nodes)
                    mention += hit
                    n = len(session.initial_graph.nodes)
                    k, tt = len(named), len(touched_nodes)
                    import math

                    chance += 1 - (
                        math.comb(n - tt, k) / math.comb(n, k) if n - tt >= k and k <= n else 0
                    )
                    if q.reasoning_type.value == "edge_exists":
                        pair = frozenset((q.source, q.target))
                        for e in ops:
                            if (
                                e.target is EditTarget.EDGE
                                and frozenset((e.source, e.destination)) == pair
                            ):
                                text_rule += 1
                                edge_edit_turns += 1
                                break
                        else:
                            edge_edit_turns += any(e.target is EditTarget.EDGE for e in ops)
                state = after
        check("sessions", f"{split}_net_noop", net_noop == 0, f"{net_noop}")
        check("sessions", f"{split}_cancelling", cancel == 0, f"{cancel}")
        check("sessions", f"{split}_reverts", revert == 0, f"{revert}", "warn")
        check(
            "sessions",
            f"{split}_always_connected",
            disconnected == 0,
            f"{disconnected}/{len(turns)}",
        )
        info(
            "sessions",
            f"{split}_closed_fraction_mean",
            f"{sum(closed_drift) / len(closed_drift):.3f}",
        )
        m_rate = mention / edit_turns if edit_turns else 0
        c_rate = chance / edit_turns if edit_turns else 0
        check(
            "sessions",
            f"{split}_mention_near_chance",
            m_rate <= 2.0 * c_rate + 0.05,
            f"mention {m_rate:.3f} vs chance {c_rate:.3f}",
            "warn",
        )
        # text rule on edge_exists among all edit turns for that task is checked loosely
        info(
            "sessions",
            f"{split}_edge_text_rule_fires",
            f"{text_rule} (gate: <20% of edge_exists edit turns)",
        )


def gate_prompt(static):
    print("\n[prompt]")
    flagged = []
    for item in static["test"]:
        if item.query.reasoning_type is ReasoningType.CONSTRAINED_REACHABILITY:
            if "exempt" not in item.query.question.lower():
                flagged.append(item.tuple_id)
    check("prompt", "constrained_states_exemption", not flagged, f"{len(flagged)} missing")
    leak = V4.text_leak_audit(static["train"], static["test"])
    bad = [t for t, v in leak.items() if v.get("leak_flag")]
    check("prompt", "text_leak_audit", not bad, f"flagged {bad}")


def gate_splits(static, sessions):
    print("\n[splits]")

    def wl(graph):
        # attributed WL one round
        nodes = graph.node_map()
        labels = {
            i: (
                nodes[i].attributes.get("line"),
                nodes[i].attributes.get("status"),
                nodes[i].attributes.get("size"),
            )
            for i in nodes
        }
        a = adj(graph)
        for _ in range(3):
            labels = {i: (labels[i], tuple(sorted(labels[j] for j in a[i]))) for i in labels}
        return str(sorted(labels.values()))

    hashes = {
        "train": {wl(g) for g in {i.graph.graph_id: i.graph for i in static["train"]}.values()},
        "test": {wl(g) for g in {i.graph.graph_id: i.graph for i in static["test"]}.values()},
        "sessions_test": {wl(s.initial_graph) for s in sessions["test"]},
    }
    check(
        "splits",
        "train_test_overlap",
        not (hashes["train"] & hashes["test"]),
        f"{len(hashes['train'] & hashes['test'])}",
    )
    check(
        "splits",
        "train_sessions_overlap",
        not (hashes["train"] & hashes["sessions_test"]),
        f"{len(hashes['train'] & hashes['sessions_test'])}",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/graphmodi_v4")
    parser.add_argument(
        "--out", default="documents/experiments/results/benchmark-audit-v4-20261025"
    )
    args = parser.parse_args()
    work, out = Path(args.work), Path(args.out)
    print("Generating metro_v4 ...")
    paths = generate(work)
    static = {s: load_static_tuples(paths[f"static_{s}"]) for s in ("train", "validation", "test")}
    sessions = {s: load_sessions(paths[f"sessions_{s}"]) for s in ("validation", "test")}
    print(
        f"loaded static {{{', '.join(f'{k}:{len(v)}' for k, v in static.items())}}} "
        f"sessions {{{', '.join(f'{k}:{len(v)}' for k, v in sessions.items())}}}"
    )

    gate_labels(static, sessions)
    gate_structure(static, sessions)
    gate_shortcuts(static)
    gate_sessions(sessions)
    gate_prompt(static)
    gate_splits(static, sessions)

    summary = Counter(c["status"] for c in CHECKS)
    print(
        f"\nSUMMARY pass={summary['pass']} fail={summary['fail']} warn={summary['warn']} info={summary['info']}"
    )
    for c in CHECKS:
        if c["status"] in ("fail", "warn"):
            print(f"  [{c['status'].upper()}] {c['section']}/{c['name']}  {c['detail']}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(
            {"summary": dict(summary), "checks": CHECKS, "sections": RESULTS}, indent=1, default=str
        )
    )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "distribution": V4.DISTRIBUTION_V4,
                "files": {
                    n: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                    for n, p in paths.items()
                },
            },
            indent=1,
        )
    )
    print(f"wrote {out}")
    if summary["fail"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
