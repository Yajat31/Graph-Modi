"""Aggregate an exact2x evaluation.json by condition x task x density x scale x session length.

Rows in evaluation.json carry only (session_id, turn_index, condition, correctness); the
task/density/scale metadata lives in the session files, so this joins them.

Usage: python scripts/analyze_exact2x.py <evaluation.json> <data_dir> <out.json>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_meta(path: Path) -> dict[tuple[str, int], dict]:
    meta: dict[tuple[str, int], dict] = {}
    with path.open() as handle:
        for line in handle:
            session = json.loads(line)
            n_nodes = len(session["initial_graph"]["nodes"])
            n_turns = len(session["turns"])
            for turn in session["turns"]:
                complexity = turn.get("complexity", {})
                meta[(session["session_id"], turn["turn_index"])] = {
                    "task": turn["query"]["reasoning_type"],
                    "density": complexity.get("density"),
                    "target_density": complexity.get("target_density"),
                    "scale": complexity.get("scale_bin"),
                    "n_nodes": n_nodes,
                    "session_length": n_turns,
                    "turn_index": turn["turn_index"],
                }
    return meta


def rate(bucket: list[int]) -> float | None:
    return bucket[0] / bucket[1] if bucket[1] else None


def main() -> None:
    eval_path, data_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    evaluation = json.load(eval_path.open())
    result: dict = {}
    for split, payload in evaluation.items():
        meta = load_meta(data_dir / f"{split}.jsonl")
        dims = ("task", "density", "scale", "session_length", "n_nodes", "turn_index")
        agg: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for row in payload["rows"]:
            info = meta[(row["session_id"], row["turn_index"])]
            correct = int(bool(row["answer_correct"]))
            cond = row["condition"]
            keys = {
                "overall": "all",
                "task": info["task"],
                "density": info["density"],
                "scale": info["scale"],
                "session_length": str(info["session_length"]),
                "n_nodes": str(info["n_nodes"]),
                "turn_index": str(info["turn_index"]),
                "changed": "answer_changed" if row["gold_answer"] != row["stale_answer"] else "answer_unchanged",
                "task|density": f"{info['task']}|{info['density']}",
                "density|scale": f"{info['density']}|{info['scale']}",
                "task|scale": f"{info['task']}|{info['scale']}",
            }
            for dim, key in keys.items():
                bucket = agg[f"{cond}::{dim}"][key]
                bucket[0] += correct
                bucket[1] += 1
        out: dict = {"summary": payload["summary"], "cuts": {}}
        for name, cuts in agg.items():
            out["cuts"][name] = {k: {"acc": rate(v), "n": v[1]} for k, v in cuts.items()}
        result[split] = out
        del dims
    out_path.write_text(json.dumps(result))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
