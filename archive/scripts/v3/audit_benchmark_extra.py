#!/usr/bin/env python3
"""Third-pass data audit of metro_v3: semantics, metadata, ties and duplicates.

  X1  constrained reachability: does the unstated "endpoints are exempt" convention change answers?
  X2  query shape: pair tasks never ask about a station and itself
  X3  duplicates: identical (graph, question) with different answers; repeated items inside a split
  X4  names: can one station name be a substring of another (ambiguous in the question text)?
  X5  hop_depth tag: is it a difficulty measure or just a task label?
  X6  session metadata vs what the sessions actually contain
  X7  most_common_attribute ties in the *stale* (before) graph of session turns

Usage: PYTHONPATH=src python3 scripts/audit_benchmark_extra.py --work /tmp/gm_fixed [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_benchmark as A  # noqa: E402

from graph_modi.graph.executor import graph_fingerprint  # noqa: E402

RES: dict[str, Any] = {}


def x1_constrained(static: dict[str, list]) -> None:
    print("\n[X1] constrained_reachability: endpoint exemption")
    out = {}
    for split, items in static.items():
        total = exempt_relevant = changes = 0
        for item in items:
            if item.query.reasoning_type.value != "constrained_reachability":
                continue
            total += 1
            q = item.query
            st = A.State(item.graph)
            ends = [e for e in (q.source, q.target) if st.attrs[e].get(q.attribute) == q.value]
            if not ends:
                continue
            exempt_relevant += 1

            # strict reading: no station on the route, endpoints included, may have the value
            def allowed(node: str) -> bool:
                return st.attrs[node].get(q.attribute) != q.value

            strict = (
                "yes"
                if (
                    allowed(q.source)
                    and allowed(q.target)
                    and q.target in A.bfs(st.adj(), q.source, allowed=allowed)
                )
                else "no"
            )
            changes += strict != item.answer
        out[split] = {
            "items": total,
            "endpoint_has_avoided_value": exempt_relevant,
            "answer_differs_under_strict_reading": changes,
        }
        A.check(
            "x1",
            f"{split}_endpoint_exemption",
            changes == 0,
            f"{exempt_relevant}/{total} items have an endpoint with the avoided value; "
            f"{changes} of them flip to 'no' if endpoints are not exempt (the question text never says endpoints are exempt)",
            "warn",
        )
    RES["x1"] = out


def x2_x3_x4(static: dict[str, list], sessions: dict[str, list]) -> None:
    print("\n[X2] query shape  [X3] duplicates  [X4] names")
    self_pairs = 0
    for items in static.values():
        for item in items:
            if item.query.target is not None and item.query.target == item.query.source:
                self_pairs += 1
    for sess in sessions.values():
        for s in sess:
            for t in s.turns:
                if t.query.target is not None and t.query.target == t.query.source:
                    self_pairs += 1
    A.check(
        "x2",
        "no_self_pairs",
        self_pairs == 0,
        f"{self_pairs} queries ask about a station and itself",
    )

    conflicts = dup = 0
    for split, items in static.items():
        seen: dict[tuple[str, str], set[str]] = defaultdict(set)
        count: Counter[tuple[str, str]] = Counter()
        for item in items:
            key = (graph_fingerprint(item.graph), item.query.question)
            seen[key].add(item.answer)
            count[key] += 1
        conflicts += sum(len(v) > 1 for v in seen.values())
        dup += sum(c - 1 for c in count.values() if c > 1)
        A.info(
            "x3",
            f"{split}_repeated_items",
            f"{sum(c - 1 for c in count.values() if c > 1)} repeated (graph, question) items of {len(items)}",
        )
    A.check(
        "x3",
        "no_conflicting_duplicates",
        conflicts == 0,
        f"{conflicts} (graph, question) keys with more than one answer",
    )

    names = sorted({f"{a} {b}" for a in A.V3._NAME_A for b in A.V3._NAME_B})
    contained = [(x, y) for x in names for y in names if x != y and x in y]
    Counter(n.split()[0] for n in names)
    A.check(
        "x4",
        "no_name_is_a_substring_of_another",
        not contained,
        f"{len(names)} names; {len(contained)} containment pairs",
    )
    RES["x2_x3_x4"] = {
        "self_pairs": self_pairs,
        "conflicts": conflicts,
        "repeats": dup,
        "containment_pairs": len(contained),
    }


def x5_hops(static: dict[str, list], sessions: dict[str, list]) -> None:
    print("\n[X5] hop_depth tag")
    tag: dict[str, Counter] = defaultdict(Counter)
    for item in static["test"]:
        tag[item.query.reasoning_type.value][item.hop_depth.value] += 1
    for task, c in sorted(tag.items()):
        A.info("x5", task, f"{dict(c)}")
    constant = [t for t, c in tag.items() if len(c) == 1]
    A.check(
        "x5",
        "hop_depth_varies_within_task",
        len(constant) <= 2,
        f"{len(constant)} of {len(tag)} tasks always get the same tag {constant}: hop_depth is mostly a task label, not a difficulty measure",
        "warn",
    )
    # for shortest_path the tag should follow the answer
    mismatches = 0
    for item in static["test"]:
        if item.query.reasoning_type.value != "shortest_path":
            continue
        a = item.answer
        expect = (
            "global"
            if a == "unreachable"
            else "local"
            if int(a) <= 2
            else "mid"
            if int(a) <= 4
            else "global"
        )
        mismatches += expect != item.hop_depth.value
    A.check("x5", "shortest_path_tag_follows_answer", mismatches == 0, f"{mismatches} mismatches")
    RES["x5"] = {t: dict(c) for t, c in tag.items()}


def x6_x7_sessions(sessions: dict[str, list]) -> None:
    print("\n[X6] session metadata  [X7] ties in the stale graph")
    for split, sess in sessions.items():
        bad: Counter[str] = Counter()
        ties = most_common_turns = tie_changes_flag = 0
        for s in sess:
            state = A.State(s.initial_graph)
            length = len(s.turns)
            for turn in s.turns:
                c = turn.complexity
                edits = turn.gold_edits or (turn.gold_edit,)
                real_noop = all(e.operation.value == "NOOP" for e in edits)
                bad["session_length"] += c["session_length"] != length
                bad["node_count"] += c["node_count"] != len(s.initial_graph.nodes)
                bad["edit_count"] += c["edit_count"] != len(turn.edit_program.edits)
                bad["noop_tag"] += ("noop" in turn.tags) != real_noop
                bad["density_tags_differ"] += c["density"] != c["target_density"]
                after, _ = A.apply_indep(state, edits)
                if turn.query.reasoning_type.value == "most_common_attribute_within_hops":
                    most_common_turns += 1
                    if not A.mode_is_unique(state, turn.query):
                        ties += 1
                        if turn.stale_answer != turn.gold_answer:
                            tie_changes_flag += 1
                state = after
        A.check("x6", f"{split}_metadata_consistent", not any(bad.values()), f"{dict(bad)}")
        A.check(
            "x7",
            f"{split}_stale_mode_ties",
            ties == 0,
            f"{ties}/{most_common_turns} most_common turns have a tied mode in the stale graph (stale answer is an alphabetical tie-break); "
            f"{tie_changes_flag} of them are counted as 'answer changed'",
            "warn",
        )
        RES[f"x6_x7_{split}"] = {
            "bad": dict(bad),
            "stale_ties": ties,
            "turns": most_common_turns,
            "tie_flagged_changed": tie_changes_flag,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/gm_fixed")
    parser.add_argument("--degree-balance", action="store_true")
    parser.add_argument("--out", default="/tmp/gm_extra_out")
    args = parser.parse_args()
    work = Path(args.work)
    A.V3.set_degree_balance(args.degree_balance)
    static = {
        s: A.load_static_tuples(work / f"static_{s}.jsonl") for s in ("train", "validation", "test")
    }
    sessions = {s: A.load_sessions(work / f"sessions_{s}.jsonl") for s in ("validation", "test")}
    x1_constrained(static)
    x2_x3_x4(static, sessions)
    x5_hops(static, sessions)
    x6_x7_sessions(sessions)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = Counter(c["status"] for c in A.CHECKS)
    (out / "extra_results.json").write_text(
        json.dumps(
            {"summary": dict(summary), "checks": A.CHECKS, "sections": RES}, indent=1, default=str
        )
    )
    print(f"\nSUMMARY {dict(summary)}\nwrote {out}/extra_results.json")


if __name__ == "__main__":
    main()
