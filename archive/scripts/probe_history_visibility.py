"""Does the accumulating edit history explain soft_prompt's rise with session length?

For every reachability / constrained_reachability turn: is a closed endpoint announced in the text
history so far? Then compare accuracy when it is announced vs not, and how often it is announced
by turn index. Usage (on the box, repo root):
  python scripts/probe_history_visibility.py <split> <evaluation.json> <condition> [<evaluation.json> <condition> ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

from graph_modi.data.multiturn import load_sessions
from graph_modi.pipeline.multiturn import materialize_states

split = sys.argv[1]
runs = list(zip(sys.argv[2::2], sys.argv[3::2]))
sessions = {s.session_id: s for s in load_sessions(f"datasets/metro_v3/{split}.jsonl")}

# per (session, turn): visibility of a closed endpoint in the history text, and the truth
info: dict[tuple[str, int], dict] = {}
for session in sessions.values():
    labels = {n.id: n.label for n in session.initial_graph.nodes}
    states = materialize_states(session)
    announced_closed: set[str] = set()
    for turn, state in zip(session.turns, states, strict=True):
        text = turn.utterance
        for node_id, label in labels.items():
            if f"{label} is closed now" in text:
                announced_closed.add(node_id)
            if f"{label} has reopened" in text:
                announced_closed.discard(node_id)
        query = turn.query
        if query.reasoning_type.value not in ("reachability",):
            continue
        after = {n.id: n.attributes.get("status") for n in state.after.nodes}
        endpoints = [query.source, query.target]
        info[(session.session_id, turn.turn_index)] = {
            "closed_true": any(after.get(e) == "closed" for e in endpoints),
            "closed_visible": any(e in announced_closed and after.get(e) == "closed" for e in endpoints),
            "length": len(session.turns),
            "gold": turn.gold_answer,
        }

for path, condition in runs:
    rows = json.load(open(path))[split]["rows"]
    buckets = defaultdict(lambda: [0, 0])
    by_turn = defaultdict(lambda: [0, 0, 0])  # n, visible closed, correct
    for row in rows:
        key = (row["session_id"], row["turn_index"])
        if row["condition"] != condition or key not in info:
            continue
        item = info[key]
        group = ("closed endpoint announced in history" if item["closed_visible"]
                 else "closed endpoint NOT announced (initially closed)" if item["closed_true"]
                 else "no closed endpoint")
        buckets[group][0] += int(bool(row["answer_correct"]))
        buckets[group][1] += 1
        t = by_turn[row["turn_index"]]
        t[0] += 1
        t[1] += int(item["closed_visible"])
        t[2] += int(bool(row["answer_correct"]))
    print(f"== {split} / {condition} (reachability turns)")
    for group, (ok, n) in sorted(buckets.items()):
        print(f"   {group:52s} n={n:4d} accuracy {100 * ok / n:5.1f}%")
    print("   by turn index: share with a closed endpoint announced | accuracy")
    print("   " + "  ".join(f"t{k}: {100 * v[1] / v[0]:3.0f}% | {100 * v[2] / v[0]:3.0f}%" for k, v in sorted(by_turn.items())))
