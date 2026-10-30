"""Every number quoted in the report, computed from the v4.1 run artifacts (CPU only).

  mkdir -p /tmp/v4_1 && tar xzf documents/experiments/results/v4_1-20261030/runs.tar.gz -C /tmp/v4_1
  PYTHONPATH=src python3 scripts/report_tables_v4.py --runs /tmp/v4_1 \
      --out documents/experiments/results/v4_1-analysis/report_tables.json

Rows are rescored from the saved raw generations with the current scorer (as in analyze_v4.py).
Model groups average their seeds: GraphToken (s42/s43/s44), TEA-style (s42/s43/s44), warm start,
frozen random GNN. Conditions that do not depend on the graph model (soft prompt, zero-shot text
baselines, stale and majority answers, tool solver) come from one run. CIs and paired differences
come from analyze_v4.py's analysis.json; the GNN probe from probe_gnn_v4.py.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_modi.data.multiturn import load_sessions  # noqa: E402
from graph_modi.data.v2 import load_static_tuples  # noqa: E402
from graph_modi.evaluation.metrics import answers_match, normalize_answer  # noqa: E402

GROUPS = {
    "graphtoken": ["v4_graphtoken_s42", "v4_graphtoken_s43", "v4_graphtoken_s44"],
    "tea": ["v4_tea_s42", "v4_tea_s43", "v4_tea_s44"],
    "warm": ["v4_graphtoken_warm_s42"],
    "random": ["v4_tea_random_sum"],
}
SHARED = {  # condition -> run that holds it
    "soft_prompt": "v4_soft_prompt",
    "soft_prompt_no_history": "v4_soft_prompt",
    "serialized_current_graph": "v4_text_baselines",
    "serialized_initial_history": "v4_text_baselines",
    "token_matched_history": "v4_text_baselines",
    "stale_answer": "v4_graphtoken_s42",
    "majority_answer": "v4_graphtoken_s42",
    "tool_solver": "v4_graphtoken_s42",
}
GRAPH_CONDITIONS = [
    "oracle_updated_graph",
    "predicted_updated_graph",
    "frozen_graph_history",
    "structure_only",
    "shuffled_graph",
    "question_only",
]


def pct(values: list[bool]) -> float | None:
    return round(100 * sum(values) / len(values), 1) if values else None


def turn_meta(sessions) -> dict:
    meta = {}
    for session in sessions:
        for turn in session.turns:
            ops = [
                "noop" if e.operation.value == "NOOP" else f"{e.operation.value} {e.target.value}"
                for e in (turn.gold_edits or ())
            ] or ["noop"]
            meta[(session.session_id, turn.turn_index)] = {
                "task": turn.query.reasoning_type.value,
                "length": int(turn.complexity.get("session_length", len(session.turns))),
                "turn": turn.turn_index + 1,
                "density": str(turn.complexity.get("density", "?")),
                "family": str(turn.complexity.get("paraphrase_family", "?")),
                "changed": normalize_answer(turn.gold_answer)
                != normalize_answer(turn.stale_answer),
                "mentions": bool(turn.complexity.get("mentions_edit")),
                "ops": "+".join(sorted(set(ops))),
            }
    return meta


def load_rows(path: Path, meta: dict) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in json.loads(path.read_text())["test"]["rows"]:
        m = meta[(row["session_id"], row["turn_index"])]
        if row.get("raw_answer") is not None:
            row["answer_correct"] = answers_match(
                row["gold_answer"], row["raw_answer"], reasoning_type=m["task"]
            )
        row["_m"] = m
        grouped[row["condition"]].append(row)
    return grouped


def breakdown(rows: list[dict], key) -> dict:
    buckets = defaultdict(list)
    for r in rows:
        k = key(r["_m"])
        if k is not None:
            buckets[k].append(bool(r["answer_correct"]))
    return {str(k): pct(v) for k, v in sorted(buckets.items(), key=lambda kv: str(kv[0]))}


def averaged(dicts: list[dict]) -> dict:
    keys = set.intersection(*(set(d) for d in dicts)) if dicts else set()
    out = {}
    for k in sorted(keys):
        values = [d[k] for d in dicts if d[k] is not None]
        if values:
            out[k] = {
                "mean": round(statistics.fmean(values), 1),
                "sd": round(statistics.stdev(values), 1) if len(values) > 1 else None,
            }
    return out


VIEWS = {
    "all": lambda m: "all",
    "by_length": lambda m: m["length"],
    "by_turn": lambda m: m["turn"],
    "by_density": lambda m: m["density"],
    "by_task": lambda m: m["task"],
    "by_family": lambda m: m["family"],
    "by_change": lambda m: "changed" if m["changed"] else "unchanged",
    "changed_no_mention": lambda m: (
        "changed_no_mention" if m["changed"] and not m["mentions"] else None
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--data", default="datasets/metro_v4_1")
    parser.add_argument(
        "--analysis", default="documents/experiments/results/v4_1-analysis/analysis.json"
    )
    parser.add_argument(
        "--probe", default="documents/experiments/results/v4_1-analysis/gnn_probe.json"
    )
    parser.add_argument(
        "--out", default="documents/experiments/results/v4_1-analysis/report_tables.json"
    )
    args = parser.parse_args()
    root, data = Path(args.runs), Path(args.data)
    sessions = load_sessions(data / "sessions_test.jsonl")
    meta = turn_meta(sessions)
    analysis = json.loads(Path(args.analysis).read_text())
    report: dict = {"benchmark": {}, "static": {}, "sessions": {}, "write_path": {}}

    # Benchmark statistics (test split).
    turns = list(meta.values())
    report["benchmark"] = {
        "sessions": len(sessions),
        "turns": len(turns),
        "answer_changing": pct([m["changed"] for m in turns]),
        "noop": pct([m["ops"] == "noop" for m in turns]),
        "mentions_edit": pct([m["mentions"] for m in turns]),
        "ops": dict(Counter(m["ops"] for m in turns)),
        "slice_sizes": analysis["slice_sizes"],
    }

    # Static QA.
    static_items = load_static_tuples(data / "static_test.jsonl")
    by_task = defaultdict(list)
    for item in static_items:
        by_task[item.query.reasoning_type.value].append(normalize_answer(item.answer))
    majority_static = {
        task: 100 * Counter(answers).most_common(1)[0][1] / len(answers)
        for task, answers in by_task.items()
    }
    report["static"]["majority_per_task_test"] = {
        t: round(v, 1) for t, v in sorted(majority_static.items())
    }
    report["static"]["majority_test"] = round(
        sum(majority_static[t] * len(by_task[t]) for t in by_task)
        / sum(len(v) for v in by_task.values()),
        1,
    )
    static_runs = {**GROUPS, "soft": ["v4_soft_prompt"]}
    for group, runs in static_runs.items():
        ckpt = "final" if group == "soft" else "best"
        entry = {}
        for split in ("validation", "test"):
            per_run = []
            for run in runs:
                path = root / run / f"static_eval_{split}_{ckpt}.json"
                result = json.loads(path.read_text())
                tasks = {k: 100 * v for k, v in result["task_accuracy"].items()}
                per_run.append({"overall": 100 * result["overall_accuracy"], **tasks})
            entry[split] = averaged(per_run)
        report["static"][group] = entry

    # Multi-turn sessions.
    rows = {
        run: load_rows(
            root
            / run
            / ("evaluation_final.json" if run == "v4_soft_prompt" else "evaluation_best.json"),
            meta,
        )
        for run in {r for runs in GROUPS.values() for r in runs} | set(SHARED.values())
    }
    for view, key in VIEWS.items():
        table: dict = {}
        for group, runs in GROUPS.items():
            table[group] = {
                cond: averaged([breakdown(rows[run][cond], key) for run in runs])
                for cond in GRAPH_CONDITIONS
                if all(rows[run].get(cond) for run in runs)
            }
        table["shared"] = {cond: breakdown(rows[run][cond], key) for cond, run in SHARED.items()}
        report["sessions"][view] = table
    report["sessions"]["local_ceiling_per_task"] = {
        t: round(100 * v, 1) for t, v in analysis["local_ceiling_per_task"]["sessions"].items()
    }
    report["sessions"]["text_lookup_baselines"] = {
        name: {
            s: round(100 * e[s]["acc"], 1)
            for s in ("all", "answer_changing", "changed_and_no_mention")
        }
        for name, e in analysis["text_baselines"].items()
    }
    report["sessions"]["slices"] = {
        run: {
            cond: {
                s: round(100 * e["sessions"][cond][s]["acc"], 1)
                for s in ("all", "answer_changing", "changed_and_no_mention", "local_wrong")
                if e["sessions"][cond][s].get("acc") is not None
            }
            for cond in e.get("sessions", {})
        }
        for run, e in analysis["runs"].items()
    }
    report["sessions"]["ci95"] = {
        run: {
            cond: [round(100 * x, 1) for x in e["sessions"][cond]["all"]["ci95"]]
            for cond in e.get("sessions", {})
        }
        for run, e in analysis["runs"].items()
    }
    report["sessions"]["seeds"] = {
        k: {"mean": round(100 * v["mean"], 1), "sd": round(100 * (v["sd"] or 0), 1)}
        for k, v in analysis["seeds"].items()
    }
    report["sessions"]["within_run_paired"] = {
        run: {
            pair: {
                "delta": round(100 * v["mean_delta"], 1),
                "ci95": [round(100 * x, 1) for x in v["ci95"]],
                "p": v["p_value"],
            }
            for pair, v in e.get("within_run_paired", {}).items()
        }
        for run, e in analysis["runs"].items()
    }
    report["sessions"]["paired"] = {
        k: {
            "delta": round(100 * v["mean_delta"], 1),
            "ci95": [round(100 * x, 1) for x in v["ci95"]],
        }
        for k, v in analysis["paired"].items()
        if k.endswith("[all]") or k.endswith("[answer_changing]") or k.endswith("[local_wrong]")
    }

    # Write path (edits come from the frozen LLM, so they are the same for every graph model).
    predicted = rows["v4_graphtoken_s42"]["predicted_updated_graph"]
    edit_rows = [r for r in predicted if r["edit_correct"] is not None]
    by_ops = defaultdict(list)
    by_family = defaultdict(list)
    for r in edit_rows:
        by_ops[r["_m"]["ops"]].append(bool(r["edit_correct"]))
        by_family[r["_m"]["family"]].append(bool(r["edit_correct"]))
    report["write_path"] = {
        "execution_equivalent": pct([bool(r["edit_correct"]) for r in edit_rows]),
        "state_exact": pct([bool(r["state_exact"]) for r in predicted]),
        "by_ops": {k: {"acc": pct(v), "n": len(v)} for k, v in sorted(by_ops.items())},
        "by_family": {k: pct(v) for k, v in sorted(by_family.items())},
        "same_edits_across_runs": all(
            [r["edit_correct"] for r in rows[run]["predicted_updated_graph"]]
            == [r["edit_correct"] for r in predicted]
            for run in ("v4_tea_s42", "v4_graphtoken_warm_s42")
        ),
    }
    report["gnn_probe"] = json.loads(Path(args.probe).read_text())

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
