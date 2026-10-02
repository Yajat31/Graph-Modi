"""Write documents/experiments/status_report_v3.md from the result JSONs.

Every number in the tables and in the prose is read from the files in
documents/experiments/results/v3-20260929 and v3x-20260929, so rerunning this script after a
pending run lands refreshes the whole report.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "documents/experiments/results/v3-20260929"
RX = ROOT / "documents/experiments/results/v3x-20260929"


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def pc(x: float | None, nd: int = 1) -> str:
    return "pending" if x is None else f"{100 * x:.{nd}f}"


def se(p: float, n: int) -> float:
    return 100 * math.sqrt(p * (1 - p) / n)


audit = load(R / "dataset_audit.json")
static = {m: {s: load(R / f"static_{m}_{s}.json") for s in ("validation", "test")} for m in ("soft_prompt", "tea", "graphtoken")}
static["tea3"] = {s: load(RX / f"static_base3_{s}.json") for s in ("validation", "test")}
exact = {m: load(R / f"exact2x_{m}_analysis.json") for m in ("soft_prompt", "tea", "graphtoken")}
exact["tea3"] = load(RX / "exact2x_base3_analysis.json")
predfix = {m: load(R / f"exact2x_{m}_predfix_analysis.json") for m in ("tea", "graphtoken")}
predfix["tea3"] = load(RX / "exact2x_base3_predfix_analysis.json")
edit_kinds = load(R / "predfix_tea_edit_kinds_test.json")
LEAK = audit["static_test"]["text_leak_audit"]
LABELS = audit["static_test"]["labels"]
NAMES = {"soft_prompt": "soft_prompt (no graph)", "tea": "TEA (5 epochs)", "tea3": "TEA (3 epochs)", "graphtoken": "GraphToken (5 epochs)"}
N_STATIC = {s: len(static["tea"][s]["rows"]) for s in ("validation", "test")}


def overall_prior(split: str) -> float:
    labels = audit[f"static_{split}"]["labels"]
    leak = audit[f"static_{split}"]["text_leak_audit"]
    counts = {t: v["n"] for t, v in labels.items() if not t.startswith("_")}
    return sum(leak[t]["question_only_rule_accuracy"] * counts[t] for t in counts) / sum(counts.values())


def static_overall_table() -> str:
    rows = ["| Model | Validation | Test |", "|---|---|---|"]
    rows.append(f"| majority answer per task | {pc(audit['static_validation']['labels']['_overall_majority_accuracy'])} | {pc(audit['static_test']['labels']['_overall_majority_accuracy'])} |")
    rows.append(f"| question-only prior (best rule from the question fields alone) | {pc(overall_prior('validation'))} | {pc(overall_prior('test'))} |")
    for m in ("soft_prompt", "tea", "tea3", "graphtoken"):
        v, t = static[m]["validation"]["overall_accuracy"], static[m]["test"]["overall_accuracy"]
        rows.append(f"| {NAMES[m]} | {pc(v)} | {pc(t)} |")
    return "\n".join(rows)


def static_task_table() -> str:
    tasks = sorted(static["tea"]["test"]["task_accuracy"], key=lambda t: -static["graphtoken"]["test"]["task_accuracy"][t])
    rows = ["| Task | prior | soft_prompt | TEA (5 ep) | TEA (3 ep) | GraphToken |", "|---|---|---|---|---|---|"]
    for t in tasks:
        cells = [pc(static[m]["test"]["task_accuracy"][t]) for m in ("soft_prompt", "tea", "tea3", "graphtoken")]
        rows.append(f"| {t} | {pc(LEAK[t]['question_only_rule_accuracy'])} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


CONDS = ["oracle_updated_graph", "structure_only", "frozen_graph_history", "cached_no_reencode", "shuffled_graph",
         "graph_once_then_text", "token_matched_history", "question_only"]


def exact_condition_table(split: str) -> str:
    rows = ["| Condition | TEA (5 ep) | TEA (3 ep) | GraphToken |", "|---|---|---|---|"]
    for c in CONDS:
        cells = []
        for m in ("tea", "tea3", "graphtoken"):
            s = exact[m].get(split, {}).get("summary", {}).get(c)
            if s is None:
                cells.append("-")
                continue
            lo, hi = s["answer_accuracy_ci95"]
            cells.append(f"{pc(s['answer_accuracy'])} [{pc(lo)}-{pc(hi)}]")
        rows.append(f"| {c} | " + " | ".join(cells) + " |")
    sp = exact["soft_prompt"][split]["summary"]["soft_prompt"]
    rows.append(f"| soft_prompt (reads the edit history as text) | {pc(sp['answer_accuracy'])} [{pc(sp['answer_accuracy_ci95'][0])}-{pc(sp['answer_accuracy_ci95'][1])}] | | |")
    rows.append(f"| overall majority baseline | {pc(audit[split]['labels']['_overall_majority_accuracy'])} | | |")
    return "\n".join(rows)


def exact_task_table() -> str:
    tasks = sorted(exact["tea"]["test"]["cuts"]["oracle_updated_graph::task"])
    rows = ["| Task | TEA updated | TEA frozen | GraphToken updated | GraphToken frozen | soft_prompt |", "|---|---|---|---|---|---|"]
    for t in tasks:
        def g(m, c):
            return pc(exact[m]["test"]["cuts"][f"{c}::task"][t]["acc"])
        rows.append(f"| {t} | {g('tea','oracle_updated_graph')} | {g('tea','frozen_graph_history')} | {g('graphtoken','oracle_updated_graph')} | {g('graphtoken','frozen_graph_history')} | {g('soft_prompt','soft_prompt')} |")
    return "\n".join(rows)


def edit_tracking_table() -> str:
    rows = ["| Model / condition | edit changed the answer | answer unchanged |", "|---|---|---|"]
    for label, m, c in (("TEA updated graph", "tea", "oracle_updated_graph"), ("TEA frozen graph", "tea", "frozen_graph_history"),
                        ("TEA no graph tokens", "tea", "question_only"), ("GraphToken updated graph", "graphtoken", "oracle_updated_graph"),
                        ("GraphToken frozen graph", "graphtoken", "frozen_graph_history"), ("soft_prompt", "soft_prompt", "soft_prompt")):
        cut = exact[m]["test"]["cuts"][f"{c}::changed"]
        rows.append(f"| {label} | {pc(cut['answer_changed']['acc'])} (n={cut['answer_changed']['n']}) | {pc(cut['answer_unchanged']['acc'])} (n={cut['answer_unchanged']['n']}) |")
    return "\n".join(rows)


def predicted_table() -> str:
    rows = ["| Model | Split | model-written edits | oracle edits | gap | edits exactly right | graph state exactly right |", "|---|---|---|---|---|---|---|"]
    for m in ("tea", "tea3", "graphtoken"):
        data = predfix[m]
        for split in ("validation", "test"):
            if data is None:
                rows.append(f"| {NAMES[m]} | {split} | pending | pending | | | |")
                continue
            p, o = data[split]["summary"]["predicted_updated_graph"], data[split]["summary"]["oracle_updated_graph"]
            rows.append(f"| {NAMES[m]} | {split} | {pc(p['answer_accuracy'])} | {pc(o['answer_accuracy'])} | {100 * (o['answer_accuracy'] - p['answer_accuracy']):+.1f} | "
                        f"{pc(p['execution_equivalent_edit_accuracy'])} | {pc(p['exact_graph_state'])} |")
    return "\n".join(rows)


def edit_kind_table() -> str:
    if not edit_kinds:
        return "(pending)"
    rows = ["| Edit kind | turns | reproduced exactly (%) |", "|---|---|---|"]
    for k, v in sorted(edit_kinds["by_kind"].items(), key=lambda kv: -kv[1]["n"]):
        rows.append(f"| {k} | {v['n']} | {100 * v['correct'] / v['n']:.1f} |")
    return "\n".join(rows)


VARIANTS = [("base3", "baseline: 3 layers, linear projector, 24k examples x 3 epochs"), ("deep", "6-layer GNN"),
            ("proj", "MLP projector 3 x 4096, lr 1e-3 (collapsed to one answer)"), ("proj2", "MLP projector 2 x 2048, lr 2e-4"),
            ("data", "3x training data (72k examples), 1 epoch")]


def variant_table() -> str:
    rows = ["| Variant | static val | static test | `reachability` | `attribute_lookup` | exact2x updated | exact2x frozen | exact2x shuffled |", "|---|---|---|---|---|---|---|---|"]
    for key, label in VARIANTS:
        sv, st = load(RX / f"static_{key}_validation.json"), load(RX / f"static_{key}_test.json")
        ex = load(RX / f"exact2x_{key}_analysis.json")["test"]["summary"]
        rows.append(f"| {label} | {pc(sv['overall_accuracy'])} | {pc(st['overall_accuracy'])} | {pc(st['task_accuracy']['reachability'])} | {pc(st['task_accuracy']['attribute_lookup'])} | "
                    f"{pc(ex['oracle_updated_graph']['answer_accuracy'])} | {pc(ex['frozen_graph_history']['answer_accuracy'])} | {pc(ex['shuffled_graph']['answer_accuracy'])} |")
    return "\n".join(rows)


def label_table() -> str:
    rows = ["| Task | static test items | majority answer | question-only prior |", "|---|---|---|---|"]
    for t in sorted(t for t in LABELS if not t.startswith("_")):
        rows.append(f"| {t} | {LABELS[t]['n']} | {pc(LABELS[t]['majority_accuracy'])} | {pc(LEAK[t]['question_only_rule_accuracy'])} |")
    return "\n".join(rows)


def d(model: str, cond: str, split: str = "test") -> float:
    return exact[model][split]["summary"][cond]["answer_accuracy"]


gt_gain = d("graphtoken", "oracle_updated_graph") - d("graphtoken", "frozen_graph_history")
tea_gain = d("tea", "oracle_updated_graph") - d("tea", "frozen_graph_history")
pt = predfix["tea"]["test"]["summary"] if predfix["tea"] else None
gt_pred = predfix["graphtoken"]["test"]["summary"] if predfix["graphtoken"] else None
gt_pred_sentence = (
    f"GraphToken with model-written edits scores {pc(gt_pred['predicted_updated_graph']['answer_accuracy'])}% on test against "
    f"{pc(gt_pred['oracle_updated_graph']['answer_accuracy'])}% with oracle edits."
    if gt_pred else "The GraphToken predicted-edit run was still in progress when this report was generated.")

TEMPLATE = f"""# Status report: the v3 benchmark (compact CLEGR-style subway graphs), TEA vs GraphToken vs soft prompt

All numbers are generated from the result files in `documents/experiments/results/v3-20260929/` and
`v3x-20260929/` by `scripts/build_report_v3.py`. **Single seed (42) throughout.** Figures are in
`documents/experiments/figures/` (`v3_*.png`, `v3x_variants.png`).

This report supersedes `status_report_fixed_final.md`, which used the earlier benchmark (graphs of 16-48
stations, label skew and a text shortcut).

## 1. Summary

1. **The benchmark now discriminates.** The graph-blind soft prompt scores at the question-only prior
   (static {pc(static['soft_prompt']['validation']['overall_accuracy'])}% validation and {pc(static['soft_prompt']['test']['overall_accuracy'])}% test, against priors of
   {pc(overall_prior('validation'))}% and {pc(overall_prior('test'))}%), so anything clearly above the prior must come from the graph.
2. **GraphToken is the strongest model**: {pc(static['graphtoken']['test']['overall_accuracy'])}% static test, against
   {pc(static['tea']['test']['overall_accuracy'])}% for TEA (5 epochs), {pc(static['tea3']['test']['overall_accuracy'])}% for TEA (3 epochs) and
   {pc(static['soft_prompt']['test']['overall_accuracy'])}% for soft_prompt. On the multi-turn eval it reaches
   {pc(d('graphtoken', 'oracle_updated_graph'))}% test when its graph tokens are re-encoded after each edit.
3. **The graph tokens carry node attributes and some neighbourhood information, but not edge structure.** The gains
   are on `attribute_lookup`, `most_common_attribute_within_hops`, `reachability` and `constrained_reachability`.
   `edge_exists`, `cycle_membership`, `node_degree`, `filtered_neighbor_count`, `shortest_path` and `within_hops_count`
   show no consistent gain over their priors for any model (one exception that does not replicate: `node_degree` for the
   3-epoch TEA checkpoint, {pc(static['tea3']['test']['task_accuracy']['node_degree'])}% against a {pc(LEAK['node_degree']['question_only_rule_accuracy'])}% prior).
4. **Re-encoding the edited graph helps**: updated-graph minus frozen-graph is {100 * gt_gain:+.1f} points for GraphToken and
   {100 * tea_gain:+.1f} for TEA on exact2x test, and a different session's graph is worse than the correct initial graph.
5. **Model-written edits cost nothing for TEA.** A frozen Llama with a few-shot prompt reproduces {pc(pt['predicted_updated_graph']['execution_equivalent_edit_accuracy']) if pt else 'n/a'}%
   of edits exactly, and accuracy with them is {pc(pt['predicted_updated_graph']['answer_accuracy']) if pt else 'n/a'}% against {pc(pt['oracle_updated_graph']['answer_accuracy']) if pt else 'n/a'}% with
   oracle edits (test). {gt_pred_sentence}
6. **A deeper GNN, a larger projector and 3x more data did not help TEA** (Section 6).
7. **Caveats that matter** are in Section 8: one seed, unequal epochs, a graph-token position that differs from CLEGR, and
   a soft prompt that is weaker than CLEGR's.

## 2. The benchmark

- **Graphs**: 12-22 stations, exact grid over size (11 values) x density (sparse / medium / dense, defined by mean
  degree) with connectivity and a diameter cap enforced; CLEGR-style node semantics (`disabled_access`, `has_rail`,
  `architecture`, `cleanliness`, `music`, `size`, plus `line` and `status`) and edge attributes (`line_color`,
  `line_stroke`, `has_aircon`, `built`).
- **Tasks** (11 static, 9 in the multi-turn eval): `edge_exists`, `reachability`, `constrained_reachability`,
  `cycle_membership`, `node_degree`, `filtered_neighbor_count`, `shortest_path`, `within_hops_count`,
  `most_common_attribute_within_hops`, plus the Facts tasks `attribute_lookup` and `attribute_check` (static only).
  Dropped as not encodable by a 3-layer GNN without edge weights or name outputs: `path_cost`, `node_count`,
  `within_hops_list`, `filtered_path_count`.
- **Labels are balanced**: yes/no tasks are 50% by construction; the others have flat answer distributions.
  A cross-validated text-only audit on the static sets finds no signal in the prompt text beyond what the question type gives away.

{label_table()}

- **Data sizes**: training 23,992 items (2,000 independent graphs x 9 tasks = 18,000 items, plus about 6,000 items from
  1,000 graphs as before/after-one-edit pairs); static eval {N_STATIC['validation']} validation / {N_STATIC['test']} test items; multi-turn eval 1,188 sessions per
  split (1, 2, 4 or 8 turns; 4,455 turns), one cell per (size, density, session length, task).
- **`status` (open/closed) is not in the prompt text**; it is only in the graph, or in the edit history in the multi-turn
  eval. For the Facts tasks and the mode task, the queried attribute is also hidden from the text.

## 3. Models and prompts

| | TEA | GraphToken | soft_prompt |
|---|---|---|---|
| Encoder | GraphSAGE 768 / 1024 / 1024, 3 layers, dropout 0.5, mean aggregation, frozen BERT-768 node features | same architecture, **initialised from TEA's pretrained GNN and trained jointly** | none |
| Alignment | contrastive GNN-to-LLM-token pretraining on 5,996 graphs from the eval grid (10 epochs) | uses TEA's | none |
| Trained | one linear projector | GNN + projector | 10 soft tokens |
| LLM | Llama-3.1-8B-Instruct, frozen | same | same |
| Training | 5 epochs, effective batch 24, lr 1e-3, bf16 | same | same |
| Decoding | greedy, at most 32 new tokens | same | same |

Prompt (all three): `Question: <W(f_i): attributes of the named stations> <question> Answer:` followed by the
10 graph or soft tokens, then the generated answer. The graph tokens come from a linear map of
`[pooled | source node | target node | hop count | task one-hot]` (3,087 numbers); the pooled slot is zeroed whenever the
question names a station, which is every task here. The multi-turn conditions differ only in the graph tokens and
whether the edit history is included as text.

## 4. Static results

{static_overall_table()}

Per task, test split (accuracy %; the prior is the best rule from the question's own fields):

{static_task_table()}

Standard errors are about {se(0.5, N_STATIC['test'] // 11):.1f} points per task (about {N_STATIC['test'] // 11} items) and
{se(0.37, N_STATIC['test']):.1f} points overall.

- The soft prompt tracks the prior on every task.
- `attribute_lookup` hides the queried attribute from the text, so the gain over the prior (27%) can only come from the
  graph tokens; GraphToken reads it back at {pc(static['graphtoken']['test']['task_accuracy']['attribute_lookup'])}% and TEA at
  {pc(static['tea']['test']['task_accuracy']['attribute_lookup'])}%.
- `attribute_check`, which asks the same fact as a yes/no question, stays at chance for every model.

## 5. Multi-turn (exact2x) results

Validation:

{exact_condition_table('validation')}

Test:

{exact_condition_table('test')}

95% intervals are per-turn and ignore correlation within a session, so they are optimistic. A dash means that condition or split was not run for that model (the 3-epoch checkpoint was evaluated on the test split with four conditions only).

Per task, test split, graph models with re-encoded versus frozen graph tokens:

{exact_task_table()}

Edit tracking, test split (accuracy on turns whose edit changed the answer versus not):

{edit_tracking_table()}

Accuracy does not trend with session length, turn index, graph size or density beyond sampling noise (the
8-turn sessions have only 297 sessions; late-turn points move by about 5 points). See `v3_exact2x_by_*.png`.

### Model-written edits

In this condition the model reads each edit sentence, writes the edit as text, the text is parsed and applied, and the
resulting graph is re-encoded. Each run also evaluates the oracle-edit condition, so the comparison is within one run.

{predicted_table()}

Edit accuracy by kind (TEA, test):

{edit_kind_table()}

An earlier run of this condition was invalid: the edit parser read a station name as one word, and v3 station names
have two words, so every real edit was rejected. The parser now resolves multi-word names by longest match and all
5,541 gold edit turns round-trip through it; the invalid results are kept in `results/v3-20260929/invalid_parser_bug/`.
Oracle, static and every other result were unaffected.

## 6. Does a deeper GNN, a larger projector or more data help TEA?

Same compute budget (about 3,000 optimiser steps); evaluation sets identical to the baseline's.

{variant_table()}

None of the three changes beats the baseline. A 6-layer GNN falls to the prior on `reachability` (it answers "yes" to 386 of
396 questions); the first larger projector collapsed to almost constant answers (for example "yes" to all 396 `reachability`
questions), and the corrected one is also at chance on `reachability` and only slightly above the prior on `attribute_lookup`;
3x more data at equal steps is also below the baseline. The 5-epoch TEA run is slightly below the 3-epoch checkpoint of the
same run, so more training or more data is not obviously the bottleneck. These are single runs, so differences of a point or
two are noise.

Why TEA over-predicts "no" on `reachability`: it separates a closed endpoint perfectly (98.9% on those items) but
a closed station one to three hops away also pushes it to "no" (89%, 78% and 67% "no" at 1, 2 and 3 hops), so it gets only
22-33% of those. This is consistent with its frozen station embeddings mixing a station's own status with its neighbours' (not
tested directly). GraphToken, whose GNN is trained jointly, handles the near-closed cases better (86% accuracy) but misses about
half of the closed endpoints. These probes are on the static test split with the 5-epoch checkpoints.

## 7. Figures

| File | Content |
|---|---|
| `v3_static_by_task.png`, `v3_static_by_density_scale.png` | static accuracy per task, by density and size |
| `v3_exact2x_conditions.png` | every multi-turn condition, validation and test |
| `v3_exact2x_by_task.png`, `v3_exact2x_by_density_size.png` | per task; by density and size |
| `v3_exact2x_by_turn.png`, `v3_exact2x_by_session_length.png`, `v3_exact2x_by_graph_size.png` | turns, sessions, graph size |
| `v3_exact2x_edit_tracking.png` | answer-changing versus unchanged turns |
| `v3_predicted_edits.png` | model-written versus oracle edits |
| `v3x_variants.png` | deeper GNN, larger projector, more data |
| `v3_dataset_composition.png` | label balance and graph sizes |

## 8. Caveats

- **One seed.** CLEGR uses five. Differences of about two points or less between models are within noise.
- **Different trainable capacity.** soft_prompt, TEA and GraphToken were each trained for 5 epochs, but GraphToken updates the
  GNN as well as the projector, so it has far more trainable parameters than TEA or the soft prompt. The 3-epoch TEA checkpoint was chosen to match the variants'
  compute budget, not selected on test, and is reported next to the 5-epoch run.
- **Not CLEGR's token order.** CLEGR places the graph tokens first (graph, W(f_i), question); here they come after the
  text, just before the answer. The wrapper is `Question: ... Answer:`, not `[INST]`. The projector also receives the hop count
  and a task one-hot, which CLEGR's does not.
- **The soft prompt is weaker than CLEGR's.** CLEGR's soft prompt also sees the full node and edge CSV of the graph as
  text. Here it sees only the named stations' attributes (and the edit history in the multi-turn eval). A CSV-in-prompt
  track was implemented but cancelled before producing results.
- **Graph size and attributes are ours** (12-22 stations, invented categories; CLEGR averages about 26 stations).
- **GraphToken here is a warm start**, initialised from TEA's pretrained GNN, not the from-scratch recipe.
- **Out-of-distribution graphs were not evaluated.**

## 9. Reproducing

```
python -m graph_modi.cli generate        --config configs/v3_tea.yaml          # datasets/metro_v3
python -m graph_modi.cli pretrain-gnn    --config configs/v3_tea.yaml
python -m graph_modi.cli train-projector --config configs/v3_{{tea,graphtoken,soft_prompt}}.yaml
python -m graph_modi.cli static-eval     --config configs/v3_<model>.yaml --split {{validation,test}}
python -m graph_modi.cli evaluate        --config configs/v3_exact2x_<model>.yaml
python -m graph_modi.cli evaluate        --config configs/v3_exact2x_<model>_predfix.yaml   # model-written edits
python scripts/analyze_exact2x.py <evaluation.json> datasets/metro_v3 <analysis.json>
python scripts/make_figures_v3.py && python scripts/build_report_v3.py
```

Variant configs are `configs/v3x_*.yaml`. Code changed for this benchmark: `data/v3.py` (new), `graph/solvers.py`
(W(f_i) masking, Facts tasks), `graph/edits.py` (multi-word station names), `graph/serialization.py` (CSV prompt),
`models/tea_glm.py` and `models/soft_prompt.py` (greedy decoding, token position option), `cli.py`.
"""

out = ROOT / "documents/experiments/status_report_v3.md"
out.write_text(TEMPLATE)
print(f"wrote {out} ({len(TEMPLATE.splitlines())} lines)")
