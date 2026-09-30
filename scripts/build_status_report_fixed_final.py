"""Write documents/experiments/status_report_fixed_final.md (tables come from the result JSONs)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report_tables_fixed_final as T  # noqa: E402

RES = ROOT / "documents/experiments/results/fixed_final-20260929"
diam = json.loads((RES / "diameter_by_density.json").read_text())
diam_rows = ["| density | " + " | ".join(f"n={n}" for n in ("16", "24", "32", "40", "48")) + " |", "|---|---|---|---|---|---|"]
for dens in ("sparse", "medium", "dense"):
    diam_rows.append(f"| {dens} | " + " | ".join(f"{diam[dens][n]:.1f}" for n in ("16", "24", "32", "40", "48")) + " |")

TEMPLATE = (ROOT / "scripts/status_report_fixed_final.template.md").read_text()
body = (
    TEMPLATE.replace("{{STATIC}}", T.static_tables())
    .replace("{{COND_TEST}}", T.cond_table("test"))
    .replace("{{COND_VAL}}", T.cond_table("validation"))
    .replace("{{TASK_TEST}}", T.task_table("test"))
    .replace("{{TASK_VAL}}", T.task_table("validation"))
    .replace("{{DENSITY}}", T.density_table())
    .replace("{{PATHCOST}}", T.pathcost_table())
    .replace("{{ANSWER_CHANGING}}", T.answer_changing_table())
    .replace("{{DIAMETER}}", "\n".join(diam_rows))
)
out = ROOT / "documents/experiments/status_report_fixed_final.md"
out.write_text(body)
print("wrote", out, len(body.splitlines()), "lines")
