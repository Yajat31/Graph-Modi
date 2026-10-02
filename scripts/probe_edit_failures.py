"""Which edits does the model fail to reproduce? Joins predicted-edit rows with the session's gold edits.

Usage (box, repo root): python scripts/probe_edit_failures.py <predicted evaluation.json> <split> [<out.json>]"""
import json, sys
from collections import defaultdict
sys.path.insert(0, "src")
from graph_modi.data.multiturn import load_sessions

path, split = sys.argv[1], sys.argv[2]
sessions = {s.session_id: s for s in load_sessions(f"datasets/metro_v3/{split}.jsonl")}
rows = [r for r in json.load(open(path))[split]["rows"] if r["condition"] == "predicted_updated_graph"]
by_kind = defaultdict(lambda: [0, 0])
by_count = defaultdict(lambda: [0, 0])
for r in rows:
    turn = sessions[r["session_id"]].turns[r["turn_index"]]
    edits = turn.gold_edits or (turn.gold_edit,)
    ops = sorted({(e.operation.value + (" " + e.target.value if e.target else "")) for e in edits})
    kind = "NOOP" if ops == ["NOOP"] else " + ".join(ops)
    by_kind[kind][0] += int(bool(r["edit_correct"]))
    by_kind[kind][1] += 1
    count = 0 if kind == "NOOP" else len(edits)
    by_count[count][0] += int(bool(r["edit_correct"]))
    by_count[count][1] += 1
print(f"{split}: edit accuracy by edit kind")
for k, (ok, n) in sorted(by_kind.items(), key=lambda kv: -kv[1][1]):
    print(f"   {k:34s} n={n:4d}  correct {100 * ok / n:5.1f}%")
print("by number of edits in the turn:", {k: f"{100 * v[0] / v[1]:.0f}% (n={v[1]})" for k, v in sorted(by_count.items())})

if len(sys.argv) > 3:
    json.dump({"by_kind": {k: {"correct": v[0], "n": v[1]} for k, v in by_kind.items()},
               "by_count": {str(k): {"correct": v[0], "n": v[1]} for k, v in by_count.items()}}, open(sys.argv[3], "w"))
