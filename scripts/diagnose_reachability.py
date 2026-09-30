"""Why do the graph models over-predict "no" on reachability? Static-set diagnosis (CPU only).

Usage: python scripts/diagnose_reachability.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict, deque

sys.path.insert(0, "src")
from graph_modi.data.v2 import load_static_tuples  # noqa: E402
from graph_modi.graph.solvers import adjacency  # noqa: E402
from graph_modi.schema import ReasoningType  # noqa: E402

R = "documents/experiments/results/v3-20260929/"


def hops_to_closed(graph, node_id) -> int | None:
    """Fewest hops from ``node_id`` to any closed node (0 = the node itself), ignoring status."""
    nodes = graph.node_map()
    adj = adjacency(graph, available_only=False)
    dist = {node_id: 0}
    queue = deque([node_id])
    while queue:
        cur = queue.popleft()
        if nodes[cur].attributes.get("status") == "closed":
            return dist[cur]
        for nxt, _ in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return None


def features(item):
    q, g = item.query, item.graph
    nearest = min(filter(lambda v: v is not None, [hops_to_closed(g, q.source), hops_to_closed(g, q.target)]), default=None)
    closed_total = sum(1 for n in g.nodes if n.attributes.get("status") == "closed")
    return nearest, closed_total


print("=== 1. training-set composition for reachability (static_train + counterfactual pairs)")
train = [t for t in load_static_tuples("datasets/metro_v3/static_train.jsonl") if t.query.reasoning_type is ReasoningType.REACHABILITY]
groups = Counter()
for t in train:
    nearest, _ = features(t)
    cf = t.metadata.get("counterfactual", "independent")
    groups[(t.answer, "endpoint closed" if nearest == 0 else "no endpoint closed", cf)] += 1
total = sum(groups.values())
for key, n in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
    print(f"   {key[2]:12s} answer={key[0]:3s} {key[1]:20s} {n:5d} ({100*n/total:4.1f}%)")
no_total = sum(n for k, n in groups.items() if k[0] == "no")
no_closed = sum(n for k, n in groups.items() if k[0] == "no" and k[1] == "endpoint closed")
print(f"   of the {no_total} 'no' training answers, {100*no_closed/no_total:.0f}% have a closed endpoint; "
      f"the rest are 'cut' cases; 'yes' answers: {sum(n for k, n in groups.items() if k[0]=='yes')}")

print("\n=== 2. static test reachability: predicted-'no' rate by how close the nearest closed station is")
tuples = {t.tuple_id: t for t in load_static_tuples("datasets/metro_v3/static_test.jsonl")}
for model in ("tea", "graphtoken", "soft_prompt"):
    rows = [r for r in json.load(open(R + f"static_{model}_test.json"))["rows"] if r["reasoning_type"] == "reachability"]
    stats = defaultdict(lambda: [0, 0, 0])  # n, predicted no, correct
    for r in rows:
        item = tuples[r["tuple_id"]]
        nearest, _ = features(item)
        bucket = "endpoint closed (gold no)" if nearest == 0 else (f"closed station {nearest} hop(s) away" if nearest is not None and nearest <= 3 else "closed station >=4 hops / none")
        s = stats[bucket]
        s[0] += 1
        s[1] += int(r["predicted_answer"] == "no")
        s[2] += int(r["correct"])
    print(f"  {model}:")
    for bucket in ("endpoint closed (gold no)", "closed station 1 hop(s) away", "closed station 2 hop(s) away", "closed station 3 hop(s) away", "closed station >=4 hops / none"):
        if bucket in stats:
            n, no, ok = stats[bucket]
            gold_yes = sum(1 for r in rows if False)
            print(f"     {bucket:34s} n={n:3d}  predicts 'no' {100*no/n:5.1f}%  accuracy {100*ok/n:5.1f}%")
