#!/usr/bin/env python3
"""Second-pass data audit of metro_v3 (run scripts/audit_benchmark.py first; it caches the data).

Looks for ways the *data* lets a model score without doing the intended work:

  A  write path     can the revision text be parsed back to the gold edit program by a regex?
  B  query/edit     does the question mention the edited station/edge, and does that decide the answer?
  C  shift          static training distribution vs the session distribution a static-trained model is tested on
  D  cycle truth    corrected cycle labels: what is left of the task once the label bug is fixed?
  E  edge hazard    original edges carry attributes, added edges carry none

Usage: PYTHONPATH=src python3 scripts/audit_benchmark_deep.py --work /tmp/graphmodi_bench [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_benchmark as A  # noqa: E402

from graph_modi.schema import EditOperation, EditTarget  # noqa: E402

RES: dict[str, Any] = {}


def touched(edits: Any) -> set[str]:
    out: set[str] = set()
    for e in edits:
        if e.operation is EditOperation.NOOP:
            continue
        if e.target is EditTarget.NODE:
            out.add(e.node_id)
        else:
            out |= {e.source, e.destination}
    return out


# ------------------------------------------------------------------------------ A
def parse_utterance(text: str, labels: dict[str, str]) -> set[tuple]:
    by_label = sorted(((lab, nid) for nid, lab in labels.items()), key=lambda x: -len(x[0]))
    parsed: set[tuple] = set()
    if text.startswith("NOOP"):
        return parsed
    for piece in text.split(" ; "):
        piece = piece.strip()
        if piece.endswith(" is closed now."):
            lab = piece[: -len(" is closed now.")]
            parsed.add(("SET", dict((l, n) for l, n in by_label)[lab], "closed"))
        elif piece.endswith(" has reopened."):
            lab = piece[: -len(" has reopened.")]
            parsed.add(("SET", dict((l, n) for l, n in by_label)[lab], "open"))
        elif piece.startswith(("ADD EDGE ", "DEL EDGE ")):
            op, rest = piece[:3], piece[9:]
            rel = rest.rsplit(" ", 1)[1]
            rest = rest.rsplit(" ", 1)[0]
            for l1, n1 in by_label:
                if rest.startswith(l1 + " "):
                    l2 = rest[len(l1) + 1 :]
                    n2 = dict(by_label).get(l2)
                    if n2:
                        parsed.add((op, frozenset((n1, n2)), rel))
                        break
        else:
            parsed.add(("UNPARSED", piece))
    return parsed


def gold_set(edits: Any) -> set[tuple]:
    out: set[tuple] = set()
    for e in edits:
        if e.operation is EditOperation.NOOP:
            continue
        if e.target is EditTarget.NODE:
            out.add(("SET", e.node_id, e.value))
        else:
            out.add((e.operation.value, frozenset((e.source, e.destination)), e.relation))
    return out


def section_write(sessions: dict[str, list]) -> None:
    print("\n[A] Write path: how hard is revision text -> edit program?")
    out: dict[str, Any] = {}
    for split, sess in sessions.items():
        skeletons: Counter[str] = Counter()
        exact = unparsed = turns = noop = copy_edges = edge_turns = 0
        for session in sess:
            state = A.State(session.initial_graph)
            labels = {n.id: n.label for n in session.initial_graph.nodes}
            for turn in session.turns:
                turns += 1
                edits = turn.gold_edits or (turn.gold_edit,)
                text = turn.utterance
                sk = text
                for lab in sorted(labels.values(), key=len, reverse=True):
                    sk = sk.replace(lab, "<S>")
                skeletons[sk] += 1
                parsed = parse_utterance(text, labels)
                exact += parsed == gold_set(edits)
                unparsed += any(p[0] == "UNPARSED" for p in parsed)
                noop += text.startswith("NOOP")
                if any(
                    e.target is EditTarget.EDGE
                    for e in edits
                    if e.operation is not EditOperation.NOOP
                ):
                    edge_turns += 1
                    copy_edges += any(p[0] in ("ADD", "DEL") for p in parsed)
                state, _ = A.apply_indep(state, edits)
        out[split] = {
            "turns": turns,
            "regex_exact": exact / turns,
            "unparsed": unparsed,
            "noop_turns": noop,
            "skeleton_count": len(skeletons),
            "top_skeletons": skeletons.most_common(8),
        }
        A.check(
            "write",
            f"{split}_regex_parser_recovers_gold",
            exact == turns,
            f"{exact}/{turns} turns parsed exactly by a ~25-line regex parser",
            "warn",
        )
        A.info(
            "write",
            f"{split}_utterance_skeletons",
            f"{len(skeletons)} distinct skeletons after masking station names; top {skeletons.most_common(6)}",
        )
        A.info(
            "write",
            f"{split}_edge_edits_are_copy_tasks",
            f"{copy_edges}/{edge_turns} edge-edit turns state the edit as the literal 'ADD/DEL EDGE <A> <B> <relation>' string",
        )
        A.info(
            "write",
            f"{split}_all_noop_baseline",
            f"predicting NOOP everywhere is exact on {noop}/{turns} = {noop / turns:.3f} of turns",
        )
    RES["write"] = out


# ------------------------------------------------------------------------------ B
def section_coupling(sessions: dict[str, list]) -> None:
    print("\n[B] Query/edit coupling: does the question mention what was edited?")
    out: dict[str, Any] = {}
    for split, sess in sessions.items():
        rows: dict[str, Counter] = defaultdict(Counter)
        chance: dict[str, float] = defaultdict(float)
        for session in sess:
            state = A.State(session.initial_graph)
            n = len(session.initial_graph.nodes)
            for turn in session.turns:
                edits = turn.gold_edits or (turn.gold_edit,)
                after, _ = A.apply_indep(state, edits)
                task = turn.query.reasoning_type.value
                t = touched(edits)
                if t:
                    q = {turn.query.source} | ({turn.query.target} if turn.query.target else set())
                    r = rows[task]
                    r["edit_turns"] += 1
                    hit = bool(q & t)
                    r["mentioned"] += hit
                    k, tt = len(q), len(t)
                    chance[task] += 1 - (
                        math.comb(n - tt, k) / math.comb(n, k) if n - tt >= k else 0
                    )
                    changed = turn.gold_answer != turn.stale_answer
                    r["changed"] += changed
                    r["changed_and_mentioned"] += changed and hit
                    # text-only rules that need no graph
                    ops = [e for e in edits if e.operation is not EditOperation.NOOP]
                    if task == "edge_exists":
                        pair = frozenset((turn.query.source, turn.query.target))
                        for e in ops:
                            if (
                                e.target is EditTarget.EDGE
                                and frozenset((e.source, e.destination)) == pair
                            ):
                                r["rule_fires"] += 1
                                guess = "yes" if e.operation is EditOperation.ADD else "no"
                                r["rule_correct"] += guess == turn.gold_answer
                                break
                    if task in ("reachability", "shortest_path"):
                        closing = [
                            e
                            for e in ops
                            if e.target is EditTarget.NODE
                            and e.value == "closed"
                            and e.node_id in q
                        ]
                        if closing:
                            r["rule_fires"] += 1
                            guess = "no" if task == "reachability" else "unreachable"
                            r["rule_correct"] += guess == turn.gold_answer
                    if task == "node_degree":
                        edge_ends = [
                            e
                            for e in ops
                            if e.target is EditTarget.EDGE
                            and turn.query.source in (e.source, e.destination)
                        ]
                        if edge_ends:
                            r["rule_fires"] += 1
                            before_deg = len(state.adj()[turn.query.source])
                            delta = sum(
                                1 if e.operation is EditOperation.ADD else -1 for e in edge_ends
                            )
                            r["rule_correct"] += str(before_deg + delta) == turn.gold_answer
                state = after
        table = {}
        for task, r in sorted(rows.items()):
            et = r["edit_turns"]
            table[task] = {
                "edit_turns": et,
                "mention_rate": round(r["mentioned"] / et, 3),
                "chance_mention_rate": round(chance[task] / et, 3),
                "answer_changing_given_mentioned": round(
                    r["changed_and_mentioned"] / r["mentioned"], 3
                )
                if r["mentioned"]
                else None,
                "answer_changing_given_not_mentioned": round(
                    (r["changed"] - r["changed_and_mentioned"]) / (et - r["mentioned"]), 3
                )
                if et > r["mentioned"]
                else None,
                "text_rule_fire_rate": round(r["rule_fires"] / et, 3),
                "text_rule_precision": round(r["rule_correct"] / r["rule_fires"], 3)
                if r["rule_fires"]
                else None,
            }
            A.info("coupling", f"{split}_{task}", f"{table[task]}")
        out[split] = table
    RES["coupling"] = out


# ------------------------------------------------------------------------------ C
def section_shift(static: dict[str, list], sessions: dict[str, list]) -> None:
    print("\n[C] Distribution shift: static training vs session evaluation")
    out: dict[str, Any] = {}
    closed_static = Counter(
        sum(a["status"] == "closed" for a in A.State(g).attrs.values())
        for g in {
            i.graph.graph_id: i.graph for i in static["train"] if "counterfactual" not in i.metadata
        }.values()
    )
    closed_sessions = Counter(
        sum(a["status"] == "closed" for a in A.State(s.initial_graph).attrs.values())
        for s in sessions["test"]
    )
    A.info(
        "shift",
        "initial_closed_count",
        f"static train {dict(sorted(closed_static.items()))}; session test starts {dict(sorted(closed_sessions.items()))}",
    )
    zero = closed_sessions[0] / sum(closed_sessions.values())
    A.check(
        "shift",
        "no_zero_closed_start_in_training",
        zero == 0,
        f"{100 * zero:.0f}% of test sessions start with 0 closed stations; static train never has 0",
        "warn",
    )
    out["closed"] = {"static": dict(closed_static), "sessions": dict(closed_sessions)}
    # answer structure inside sessions (after-state), compared with static test
    structure: dict[str, Any] = {}
    tally: dict[str, Counter] = defaultdict(Counter)
    answers: dict[str, Counter] = defaultdict(Counter)
    for session in sessions["test"]:
        state = A.State(session.initial_graph)
        for turn in session.turns:
            after, _ = A.apply_indep(state, turn.gold_edits or (turn.gold_edit,))
            task = turn.query.reasoning_type.value
            f = A.features(after, turn.query)
            if task == "reachability":
                tally["reach"]["n"] += 1
                tally["reach"]["no"] += turn.gold_answer == "no"
                tally["reach"]["no_closed"] += turn.gold_answer == "no" and "closed" in (
                    f["ss"],
                    f["st"],
                )
                tally["reach"]["closed_endpoint"] += "closed" in (f["ss"], f["st"])
            if task == "cycle_membership":
                tally["cycle"]["n"] += 1
                tally["cycle"]["no"] += turn.gold_answer == "no"
                tally["cycle"]["no_deg1"] += turn.gold_answer == "no" and f["ds"] == 1
            if task in A.YES_NO:
                answers[task][turn.gold_answer] += 1
            state = after
    structure["reachability"] = {k: v for k, v in tally["reach"].items()}
    structure["cycle"] = {k: v for k, v in tally["cycle"].items()}
    r, c = tally["reach"], tally["cycle"]
    A.info(
        "shift",
        "session_reachability",
        f"'no' share {r['no'] / r['n']:.3f}; 'no' with a closed endpoint {r['no_closed']}/{r['no']}; "
        f"queries with a closed endpoint {r['closed_endpoint'] / r['n']:.3f}",
    )
    A.info(
        "shift",
        "session_cycle",
        f"'no' share {c['no'] / c['n']:.3f}; 'no' on degree-1 stations {c['no_deg1']}/{c['no']}",
    )
    yn = {t: round(a["yes"] / sum(a.values()), 3) for t, a in sorted(answers.items())}
    A.check(
        "shift",
        "session_yes_rate_balanced",
        all(abs(v - 0.5) < 0.08 for v in yn.values()),
        f"P(yes) in session test turns {yn}",
        "warn",
    )
    out["structure"] = structure
    out["session_yes_rate"] = yn
    RES["shift"] = out


# ------------------------------------------------------------------------------ D
def section_cycle_truth(static: dict[str, list], sessions: dict[str, list]) -> None:
    print("\n[D] cycle_membership with corrected labels")
    out: dict[str, Any] = {}
    for split in ("train", "test"):
        deg_no: Counter[int] = Counter()
        deg_all: Counter[int] = Counter()
        for item in static[split]:
            if item.query.reasoning_type.value != "cycle_membership":
                continue
            st = A.State(item.graph)
            truth = A.solve(st, item.query)
            d = len(st.adj()[item.query.source])
            deg_all[d] += 1
            deg_no[d] += truth == "no"
        total = sum(deg_all.values())
        no = sum(deg_no.values())
        nonleaf_no = no - deg_no[1]
        out[split] = {
            "items": total,
            "true_no": no,
            "true_no_nonleaf": nonleaf_no,
            "no_by_degree": dict(sorted(deg_no.items())),
        }
        A.info(
            "cycle_truth",
            f"static_{split}",
            f"true 'no' = {no}/{total}; of those degree-1 {deg_no[1]}, non-leaf {nonleaf_no} ({100 * nonleaf_no / no:.1f}%); by degree {dict(sorted(deg_no.items()))}",
        )
    # session turns
    leaf = nonleaf = n = 0
    for session in sessions["test"]:
        state = A.State(session.initial_graph)
        for turn in session.turns:
            after, _ = A.apply_indep(state, turn.gold_edits or (turn.gold_edit,))
            if turn.query.reasoning_type.value == "cycle_membership":
                truth = A.solve(after, turn.query)
                n += 1
                if truth == "no":
                    if len(after.adj()[turn.query.source]) <= 1:
                        leaf += 1
                    else:
                        nonleaf += 1
            state = after
    A.info(
        "cycle_truth",
        "sessions_test",
        f"true 'no' turns: degree<=1 {leaf}, non-leaf {nonleaf} of {n} cycle turns",
    )
    out["sessions_test"] = {"leaf_no": leaf, "nonleaf_no": nonleaf, "turns": n}
    RES["cycle_truth"] = out


# ------------------------------------------------------------------------------ E
def section_edges(static: dict[str, list], sessions: dict[str, list]) -> None:
    print("\n[E] Edge attribute hazard")
    attrs = Counter()
    rel = Counter()
    lc = Counter()
    n = 0
    for item in static["test"][::11]:
        for e in item.graph.edges:
            n += 1
            attrs[bool(e.attributes)] += 1
            rel[e.relation] += 1
            lc[e.attributes.get("line_color") if e.attributes else None] += 1
    A.info(
        "edges",
        "initial_edges",
        f"{n} edges; with attributes {attrs[True]}, without {attrs[False]}; relations {dict(rel)}; line_color {dict(lc)}",
    )
    added = 0
    for session in sessions["test"]:
        for turn in session.turns:
            for e in turn.gold_edits or (turn.gold_edit,):
                if e.operation is EditOperation.ADD:
                    added += 1
    A.info(
        "edges",
        "added_edges",
        f"{added} ADD EDGE edits in test sessions create edges with relation 'transfer' and no attributes; "
        f"every original edge has {len(next(iter(static['test'][0].graph.edges)).attributes)} attributes",
    )
    RES["edges"] = {"initial": n, "attrs": dict(attrs), "added": added}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/graphmodi_bench")
    parser.add_argument("--degree-balance", action="store_true")
    parser.add_argument("--out", default="documents/experiments/results/benchmark-audit-20261021")
    args = parser.parse_args()
    work = Path(args.work)
    A.V3.set_degree_balance(args.degree_balance)
    static = {
        s: A.load_static_tuples(work / f"static_{s}.jsonl") for s in ("train", "validation", "test")
    }
    sessions = {s: A.load_sessions(work / f"sessions_{s}.jsonl") for s in ("validation", "test")}
    section_write(sessions)
    section_coupling(sessions)
    section_shift(static, sessions)
    section_cycle_truth(static, sessions)
    section_edges(static, sessions)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = Counter(c["status"] for c in A.CHECKS)
    (out / "deep_results.json").write_text(
        json.dumps(
            {"summary": dict(summary), "checks": A.CHECKS, "sections": RES}, indent=1, default=str
        )
    )
    print(f"\nSUMMARY {dict(summary)}\nwrote {out}/deep_results.json")


if __name__ == "__main__":
    main()
