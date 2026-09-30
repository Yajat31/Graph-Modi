"""Print headline numbers for a v3 exact2x run against its dataset baselines.

Usage: python scripts/summarize_v3_exact2x.py <analysis.json> <dataset audit.json>
(analysis.json comes from scripts/analyze_exact2x.py.)
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    analysis = json.load(open(sys.argv[1]))
    audit = json.load(open(sys.argv[2]))
    for split in ("validation", "test"):
        cuts = analysis[split]["cuts"]
        summary = analysis[split]["summary"]
        majority = audit[split]["labels"]["_overall_majority_accuracy"]
        changing = audit[split]["labels"]["_answer_changing_fraction"]
        print(f"== {split} (overall majority baseline {100 * majority:.1f}%, "
              f"answer-changing turns {100 * changing:.0f}%)")
        for condition, values in sorted(summary.items(), key=lambda item: -item[1]["answer_accuracy"]):
            low, high = values["answer_accuracy_ci95"]
            print(f"   {condition:28s} {100 * values['answer_accuracy']:5.1f}% "
                  f"[{100 * low:.1f}-{100 * high:.1f}]  stale-rate {100 * values['stale_rate']:.1f}%")
        for condition in summary:
            if condition in ("tool_solver", "majority_prior"):
                continue
            tasks = {k: round(100 * v["acc"], 1) for k, v in sorted(cuts[f"{condition}::task"].items())}
            print(f"   [{condition}] per task: {tasks}")
            if condition in ("soft_prompt", "oracle_updated_graph"):
                density = {k: round(100 * v["acc"], 1) for k, v in cuts[f"{condition}::density"].items()}
                length = {k: round(100 * v["acc"], 1) for k, v in sorted(cuts[f"{condition}::session_length"].items())}
                print(f"   [{condition}] by density: {density}  by session length: {length}")


if __name__ == "__main__":
    main()
