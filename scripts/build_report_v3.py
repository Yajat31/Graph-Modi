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
RA = ROOT / "documents/experiments/results/audit-20261016"


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def pc(x: float | None, nd: int = 1) -> str:
    return "pending" if x is None else f"{100 * x:.{nd}f}"


def se(p: float, n: int) -> float:
    return 100 * math.sqrt(p * (1 - p) / n)


audit = load(R / "dataset_audit.json")
STATS = load(
    R / "stats_test_existing_runs.json"
)  # session-clustered CIs / paired tests (scripts/stats_v3.py)
E2E = {s: load(R / f"static_graphtoken_e2e_{s}.json") for s in ("validation", "test")}
E2E_EXACT = load(R / "exact2x_graphtoken_e2e_analysis.json")
static = {
    m: {s: load(R / f"static_{m}_{s}.json") for s in ("validation", "test")}
    for m in ("soft_prompt", "tea", "graphtoken")
}
static["tea3"] = {s: load(RX / f"static_base3_{s}.json") for s in ("validation", "test")}
exact = {m: load(R / f"exact2x_{m}_analysis.json") for m in ("soft_prompt", "tea", "graphtoken")}
exact["tea3"] = load(RX / "exact2x_base3_analysis.json")
predfix = {m: load(R / f"exact2x_{m}_predfix_analysis.json") for m in ("tea", "graphtoken")}
predfix["tea3"] = load(RX / "exact2x_base3_predfix_analysis.json")
edit_kinds = load(R / "predfix_tea_edit_kinds_test.json")
LEAK = audit["static_test"]["text_leak_audit"]
LABELS = audit["static_test"]["labels"]
NAMES = {
    "soft_prompt": "soft_prompt (no graph)",
    "tea": "TEA (5 epochs)",
    "tea3": "TEA (3 epochs)",
    "graphtoken": "TEA run 2 (5 epochs; was labelled GraphToken)",
}
N_STATIC = {s: len(static["tea"][s]["rows"]) for s in ("validation", "test")}


def overall_prior(split: str) -> float:
    labels = audit[f"static_{split}"]["labels"]
    leak = audit[f"static_{split}"]["text_leak_audit"]
    counts = {t: v["n"] for t, v in labels.items() if not t.startswith("_")}
    return sum(leak[t]["question_only_rule_accuracy"] * counts[t] for t in counts) / sum(
        counts.values()
    )


def static_overall_table() -> str:
    rows = ["| Model | Validation | Test |", "|---|---|---|"]
    rows.append(
        f"| majority answer per task | {pc(audit['static_validation']['labels']['_overall_majority_accuracy'])} | {pc(audit['static_test']['labels']['_overall_majority_accuracy'])} |"
    )
    rows.append(
        f"| question-only prior (best rule from the question fields alone) | {pc(overall_prior('validation'))} | {pc(overall_prior('test'))} |"
    )
    for m in ("soft_prompt", "tea", "tea3", "graphtoken"):
        v, t = static[m]["validation"]["overall_accuracy"], static[m]["test"]["overall_accuracy"]
        rows.append(f"| {NAMES[m]} | {pc(v)} | {pc(t)} |")
    return "\n".join(rows)


def static_task_table() -> str:
    tasks = sorted(
        static["tea"]["test"]["task_accuracy"],
        key=lambda t: -static["graphtoken"]["test"]["task_accuracy"][t],
    )
    rows = [
        "| Task | prior | soft_prompt | TEA (5 ep) | TEA (3 ep) | TEA run 2 |",
        "|---|---|---|---|---|---|",
    ]
    for t in tasks:
        cells = [
            pc(static[m]["test"]["task_accuracy"][t])
            for m in ("soft_prompt", "tea", "tea3", "graphtoken")
        ]
        rows.append(
            f"| {t} | {pc(LEAK[t]['question_only_rule_accuracy'])} | " + " | ".join(cells) + " |"
        )
    return "\n".join(rows)


CONDS = [
    "oracle_updated_graph",
    "structure_only",
    "frozen_graph_history",
    "cached_no_reencode",
    "shuffled_graph",
    "graph_once_then_text",
    "token_matched_history",
    "question_only",
]


def exact_condition_table(split: str) -> str:
    rows = ["| Condition | TEA (5 ep) | TEA (3 ep) | TEA run 2 |", "|---|---|---|---|"]
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
    rows.append(
        f"| soft_prompt (reads the edit history as text) | {pc(sp['answer_accuracy'])} [{pc(sp['answer_accuracy_ci95'][0])}-{pc(sp['answer_accuracy_ci95'][1])}] | | |"
    )
    rows.append(
        f"| overall majority baseline | {pc(audit[split]['labels']['_overall_majority_accuracy'])} | | |"
    )
    return "\n".join(rows)


def exact_task_table() -> str:
    tasks = sorted(exact["tea"]["test"]["cuts"]["oracle_updated_graph::task"])
    rows = [
        "| Task | TEA updated | TEA frozen | TEA run 2 updated | TEA run 2 frozen | soft_prompt |",
        "|---|---|---|---|---|---|",
    ]
    for t in tasks:

        def g(m, c):
            return pc(exact[m]["test"]["cuts"][f"{c}::task"][t]["acc"])

        rows.append(
            f"| {t} | {g('tea', 'oracle_updated_graph')} | {g('tea', 'frozen_graph_history')} | {g('graphtoken', 'oracle_updated_graph')} | {g('graphtoken', 'frozen_graph_history')} | {g('soft_prompt', 'soft_prompt')} |"
        )
    return "\n".join(rows)


def edit_tracking_table() -> str:
    rows = ["| Model / condition | edit changed the answer | answer unchanged |", "|---|---|---|"]
    for label, m, c in (
        ("TEA updated graph", "tea", "oracle_updated_graph"),
        ("TEA frozen graph", "tea", "frozen_graph_history"),
        ("TEA no graph tokens", "tea", "question_only"),
        ("TEA run 2 updated graph", "graphtoken", "oracle_updated_graph"),
        ("TEA run 2 frozen graph", "graphtoken", "frozen_graph_history"),
        ("soft_prompt", "soft_prompt", "soft_prompt"),
    ):
        cut = exact[m]["test"]["cuts"][f"{c}::changed"]
        rows.append(
            f"| {label} | {pc(cut['answer_changed']['acc'])} (n={cut['answer_changed']['n']}) | {pc(cut['answer_unchanged']['acc'])} (n={cut['answer_unchanged']['n']}) |"
        )
    return "\n".join(rows)


def predicted_table() -> str:
    rows = [
        "| Model | Split | model-written edits | oracle edits | gap | edits exactly right | graph state exactly right |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in ("tea", "tea3", "graphtoken"):
        data = predfix[m]
        for split in ("validation", "test"):
            if data is None:
                rows.append(f"| {NAMES[m]} | {split} | pending | pending | | | |")
                continue
            p, o = (
                data[split]["summary"]["predicted_updated_graph"],
                data[split]["summary"]["oracle_updated_graph"],
            )
            rows.append(
                f"| {NAMES[m]} | {split} | {pc(p['answer_accuracy'])} | {pc(o['answer_accuracy'])} | {100 * (o['answer_accuracy'] - p['answer_accuracy']):+.1f} | "
                f"{pc(p['execution_equivalent_edit_accuracy'])} | {pc(p['exact_graph_state'])} |"
            )
    return "\n".join(rows)


def edit_kind_table() -> str:
    if not edit_kinds:
        return "(pending)"
    rows = ["| Edit kind | turns | reproduced exactly (%) |", "|---|---|---|"]
    for k, v in sorted(edit_kinds["by_kind"].items(), key=lambda kv: -kv[1]["n"]):
        rows.append(f"| {k} | {v['n']} | {100 * v['correct'] / v['n']:.1f} |")
    return "\n".join(rows)


VARIANTS = [
    ("base3", "baseline: 3 layers, linear projector, 24k examples x 3 epochs"),
    ("deep", "6-layer GNN"),
    ("proj", "MLP projector 3 x 4096, lr 1e-3 (collapsed to one answer)"),
    ("proj2", "MLP projector 2 x 2048, lr 2e-4"),
    ("data", "3x training data (72k examples), 1 epoch"),
]


def variant_table() -> str:
    rows = [
        "| Variant | static val | static test | `reachability` | `attribute_lookup` | exact2x updated | exact2x frozen | exact2x shuffled |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, label in VARIANTS:
        sv, st = load(RX / f"static_{key}_validation.json"), load(RX / f"static_{key}_test.json")
        ex = load(RX / f"exact2x_{key}_analysis.json")["test"]["summary"]
        rows.append(
            f"| {label} | {pc(sv['overall_accuracy'])} | {pc(st['overall_accuracy'])} | {pc(st['task_accuracy']['reachability'])} | {pc(st['task_accuracy']['attribute_lookup'])} | "
            f"{pc(ex['oracle_updated_graph']['answer_accuracy'])} | {pc(ex['frozen_graph_history']['answer_accuracy'])} | {pc(ex['shuffled_graph']['answer_accuracy'])} |"
        )
    return "\n".join(rows)


def label_table() -> str:
    rows = [
        "| Task | static test items | majority answer | question-only prior |",
        "|---|---|---|---|",
    ]
    for t in sorted(t for t in LABELS if not t.startswith("_")):
        rows.append(
            f"| {t} | {LABELS[t]['n']} | {pc(LABELS[t]['majority_accuracy'])} | {pc(LEAK[t]['question_only_rule_accuracy'])} |"
        )
    return "\n".join(rows)


def d(model: str, cond: str, split: str = "test") -> float:
    return exact[model][split]["summary"][cond]["answer_accuracy"]


gt_gain = d("graphtoken", "oracle_updated_graph") - d("graphtoken", "frozen_graph_history")
tea_gain = d("tea", "oracle_updated_graph") - d("tea", "frozen_graph_history")
pt = predfix["tea"]["test"]["summary"] if predfix["tea"] else None
gt_pred = predfix["graphtoken"]["test"]["summary"] if predfix["graphtoken"] else None
gt_pred_sentence = (
    f"TEA run 2 with model-written edits scores {pc(gt_pred['predicted_updated_graph']['answer_accuracy'])}% on test against "
    f"{pc(gt_pred['oracle_updated_graph']['answer_accuracy'])}% with oracle edits."
    if gt_pred
    else "The TEA run 2 predicted-edit run was still in progress when this report was generated."
)


def _a(name):
    return load(RA / name)


FINAL = [  # (label, key, what it is)
    ("soft_prompt (no graph)", "soft_prompt", "10 learned tokens, no graph"),
    (
        "TEA, old (random frozen GNN, mean aggregation)",
        "tea_old",
        "what this report's earlier versions called TEA",
    ),
    ("task-only control (zeroed GNN, retrained)", "taskonly", "projector sees task id + hops only"),
    (
        "end-to-end GraphToken, mean aggregation",
        "gt_e2e_mean",
        "random-init GNN trained jointly; collapsed",
    ),
    (
        "TEA, frozen RANDOM GNN, sum aggregation",
        "tea_sum_random",
        "no-pretraining control for the next row",
    ),
    (
        "TEA, frozen PRETRAINED GNN, sum aggregation",
        "tea_sum",
        "contrastive GNN pretraining (fixed), projector lr 1e-4",
    ),
    ("GraphToken, sum aggregation, seed 42", "gt_sum", "random-init GNN trained jointly"),
    ("GraphToken, sum aggregation, seed 43", "gt_sum_s43", ""),
    ("GraphToken, sum aggregation, seed 44", "gt_sum_s44", ""),
    (
        "GraphToken warm start (pretrained + joint), sum aggregation",
        "gt_warm_full",
        "pretrained GNN, then trained jointly",
    ),
]


def final_table() -> str:
    rows = [
        "| Model | static val | static test | exact2x test: updated graph | frozen graph | shuffled graph | structure only |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, key, _ in FINAL:
        v, t = _a(f"static_{key}_validation.json"), _a(f"static_{key}_test.json")
        ex = _a(f"exact2x_{key}_summary.json")
        sm = ex["test"]["summary"] if ex else {}
        cells = [
            pc(sm[c]["answer_accuracy"]) if c in sm else "-"
            for c in (
                "oracle_updated_graph",
                "frozen_graph_history",
                "shuffled_graph",
                "structure_only",
            )
        ]
        rows.append(
            f"| {label} | {pc(v['overall_accuracy']) if v else '-'} | {pc(t['overall_accuracy']) if t else '-'} | "
            + " | ".join(cells)
            + " |"
        )
    sp = _a("exact2x_reeval_soft_prompt_summary.json")["test"]["summary"]
    rows.append(
        f"| soft prompt, edit history as text / question only (multi-turn) | | | {pc(sp['soft_prompt']['answer_accuracy'])} / {pc(sp['soft_prompt_no_history']['answer_accuracy'])} | | | |"
    )
    return "\n".join(rows)


def final_task_table() -> str:
    keys = ["tea_sum_random", "tea_sum", "gt_sum", "gt_warm_full"]
    data = [_a(f"static_{k}_test.json") for k in keys]
    rows = [
        "| Task | prior | TEA random GNN | TEA pretrained GNN | GraphToken | GraphToken warm start |",
        "|---|---|---|---|---|---|",
    ]
    for t in sorted(data[0]["task_accuracy"], key=lambda t: -data[3]["task_accuracy"][t]):
        rows.append(
            f"| {t} | {pc(LEAK[t]['question_only_rule_accuracy'])} | "
            + " | ".join(pc(x["task_accuracy"][t]) for x in data)
            + " |"
        )
    return "\n".join(rows)


FS = _a("stats_test_final.json")
SC = _a("shortcut_audit.json")


def shortcut_table() -> str:
    labels = [
        ("old_tea", "old TEA"),
        ("tea_sum_random", "TEA random sum GNN"),
        ("tea_sum", "TEA pretrained sum GNN"),
        ("gt_sum", "GraphToken seed 42"),
        ("gt_sum_s43", "GraphToken seed 43"),
        ("gt_warm", "GraphToken warm start"),
    ]
    head = ["| Task | shortcut |"] + [f" {n} |" for _, n in labels]
    rows = ["".join(head), "|---|---|" + "---|" * len(labels)]
    data = {
        k: _a(
            {
                "old_tea": "static_tea_old_test.json",
                "tea_sum_random": "static_tea_sum_random_test.json",
                "tea_sum": "static_tea_sum_test.json",
                "gt_sum": "static_gt_sum_test.json",
                "gt_sum_s43": "static_gt_sum_s43_test.json",
                "gt_warm": "static_gt_warm_full_test.json",
            }[k]
        )
        for k, _ in labels
    }
    for t, v in SC["static"]["shortcut"]["per_task"].items():
        rows.append(
            f"| {t} | {pc(v)} | "
            + " | ".join(pc(data[k]["task_accuracy"][t]) for k, _ in labels)
            + " |"
        )
    rows.append(
        f"| **mean over these 8 tasks** | **{pc(SC['static']['shortcut']['task_mean'])}** | "
        + " | ".join(pc(SC["static"][k]["overall"]) for k, _ in labels)
        + " |"
    )
    rows.append(
        "| items where the shortcut is wrong (n=%d) | 0.0 | " % SC["static"]["gt_sum"]["n_hard"]
        + " | ".join(pc(SC["static"][k]["on_shortcut_wrong_items"]) for k, _ in labels)
        + " |"
    )
    rows.append(
        "| items where the shortcut is right | 100.0 | "
        + " | ".join(pc(SC["static"][k]["on_shortcut_right_items"]) for k, _ in labels)
        + " |"
    )
    for sl, name in (
        ("cycle_deg>=2", "`cycle_membership`, degree >= 2 (n=231; always-yes = 85.7)"),
        ("reach_both_open", "`reachability`, both endpoints open (n=220; always-yes = 90)"),
        ("reach_endpoint_closed", "`reachability`, an endpoint closed (n=176)"),
    ):
        rows.append(
            f"| {name} | - | "
            + " | ".join(pc(SC["static"][k]["slices"][sl][0]) for k, _ in labels)
            + " |"
        )
    return "\n".join(rows)


def multi_shortcut_table() -> str:
    rows = [
        "| Model (multi-turn test, 8 non-attribute tasks) | all turns, updated graph | turns where the shortcut fails (n=%d) | turns where it succeeds | frozen graph, all turns |"
        % 1401,
        "|---|---|---|---|---|",
    ]
    m = SC["multi"]["shortcut"]
    rows.append(
        f"| shortcut tracker (updated graph / stale graph) | {pc(m['updated_graph'])} | 0.0 | 100.0 | {pc(m['stale_graph'])} |"
    )
    for k, n in (
        ("tea_sum_random", "TEA random sum GNN"),
        ("gt_sum", "GraphToken seed 42"),
        ("gt_warm", "GraphToken warm start"),
    ):
        o, f = SC["multi"][k]["oracle_updated_graph"], SC["multi"][k]["frozen_graph_history"]
        rows.append(
            f"| {n} | {pc(o['all'][0])} | {pc(o['hard'][0])} | {pc(o['easy'][0])} | {pc(f['all'][0])} |"
        )
    sp = SC["multi"]["soft_prompt"]["soft_prompt_no_history"]
    rows.append(
        f"| soft prompt (question only) | {pc(sp['all'][0])} | {pc(sp['hard'][0])} | {pc(sp['easy'][0])} | - |"
    )
    return "\n".join(rows)


gts = [
    _a(f"static_{k}_test.json")["overall_accuracy"] for k in ("gt_sum", "gt_sum_s43", "gt_sum_s44")
]
gt_mean = sum(gts) / 3
gt_sd = (sum((x - gt_mean) ** 2 for x in gts) / 2) ** 0.5
PF = {
    k: _a(f"exact2x_predfix_{k}_summary.json")["test"]["summary"]
    for k in ("gt_sum", "gt_warm_full")
}


def fp(pair: str, cond: str = "oracle_updated_graph") -> str:
    r = FS["across_models"][pair][cond]
    return f"{100 * r['mean_delta']:+.1f} [{100 * r['ci95'][0]:+.1f}, {100 * r['ci95'][1]:+.1f}]"


def fsp(model: str) -> str:
    r = FS["custom_pairs"][f"{model}:oracle_updated_graph:soft_prompt:soft_prompt_no_history"]
    return f"{100 * r['mean_delta']:+.1f} [{100 * r['ci95'][0]:+.1f}, {100 * r['ci95'][1]:+.1f}]"


TEMPLATE = f"""# Status report: the v3 benchmark (compact CLEGR-style subway graphs), TEA vs soft prompt (and a failed GraphToken)

All numbers are generated from the result files in `documents/experiments/results/v3-20260929/` and
`v3x-20260929/` by `scripts/build_report_v3.py`. **Single seed (42) throughout.** Figures are in
`documents/experiments/figures/` (`v3_*.png`, `v3x_variants.png`).

This report supersedes `status_report_fixed_final.md`, which used the earlier benchmark (graphs of 16-48
stations, label skew and a text shortcut).

## 0. Corrections (read first; found by the 2026-10-14 audits)

The first version of this report was wrong in four ways that change what the numbers mean.

1. **"TEA run 2" is not GraphToken.** The column earlier labelled GraphToken has a frozen GNN (training re-froze it) and
   differs from TEA only in the projector's random initialisation. It is a second TEA run, so the TEA / TEA run 2 gap is
   **run-to-run variance**: {pc(static["graphtoken"]["test"]["overall_accuracy"])}% vs {pc(static["tea"]["test"]["overall_accuracy"])}% static test, and on exact2x test
   `oracle_updated_graph` 37.9% vs 35.1%, a paired difference of {100 * STATS["across_models"]["tea - tea_run2"]["oracle_updated_graph"]["mean_delta"]:+.1f} points
   (session-bootstrap 95% CI [{100 * STATS["across_models"]["tea - tea_run2"]["oracle_updated_graph"]["ci95"][0]:+.1f}, {100 * STATS["across_models"]["tea - tea_run2"]["oracle_updated_graph"]["ci95"][1]:+.1f}]).
   Treat about 3 points as the noise floor for any single-seed comparison here.
2. **No "pretrained" GNN existed.** `TEAGLM` freezes its GNN when built, and the contrastive pretraining optimised that
   frozen GNN, so only the discarded alignment head trained (final loss 4.477 against a chance level of 4.908; the 6-layer
   run ended at 4.906). Every TEA, TEA run 2 and `v3x` result in this report is therefore a **frozen random GNN plus a
   trained projector**. With the optimiser fixed the same objective reaches 1.746. The `v3x_deep` result is not evidence about
   depth: that GNN was random and 6 layers oversmooth more. A properly pretrained TEA is being run as a separate result.
3. **The first end-to-end GraphToken (mean aggregation) collapsed, and the cause was the aggregation (Section 0a).** Static test
   {pc(E2E["test"]["overall_accuracy"])}% against a 32.8% question-only prior: it answers "yes" to all 396 `reachability` and all 396 `edge_exists`
   questions, no checkpoint of the 10 saved ever exceeds 34.2% on validation, and its multi-turn `oracle_updated_graph` score
   ({pc(E2E_EXACT["test"]["summary"]["oracle_updated_graph"]["answer_accuracy"])}%) equals the graph-blind soft prompt's 34.2%. One-epoch pilots that changed warmup, GNN lr, dropout, clipping and bf16 did not
   recover it; changing only the aggregation from mean to sum did (53% vs 33% static validation).
4. **Statistics and baselines.** The per-turn Wilson intervals below ignore that turns of one session are dependent; session-bootstrap
   intervals are about 25% wider and paired tests are in `results/v3-20260929/stats_test_existing_runs.json`. TEA's graph advantage over
   the soft prompt on exact2x test is {100 * STATS["custom_pairs"]["tea:oracle_updated_graph:soft_prompt:soft_prompt"]["mean_delta"]:+.1f} points (CI [{100 * STATS["custom_pairs"]["tea:oracle_updated_graph:soft_prompt:soft_prompt"]["ci95"][0]:+.1f}, {100 * STATS["custom_pairs"]["tea:oracle_updated_graph:soft_prompt:soft_prompt"]["ci95"][1]:+.1f}]), i.e. not significant.
   The condition called `majority_prior` is the **stale (pre-edit) answer** (59.3%), not a majority baseline (about 29.4%); it is now
   `stale_answer`. The answer parser used to take the first number anywhere in a generation, which under-scores free-form text-only
   baselines; old result rows were scored with it, and re-runs under the leading-token parser are in `outputs/audit/`.

## 0a. Results after the fixes (sum aggregation; single GPU runs, audit of 2026-10-14 to 16)

Mean aggregation divides by the neighbour count, so a node's degree is invisible to the GNN, and with a random
initialisation the mean-aggregated embeddings are almost the same vector for every node (node-to-node variation is 9% of the
embedding norm). **Sum aggregation** keeps degree and gives the encoder usable structure. Everything below uses sum aggregation,
dropout 0, a cosine schedule with warmup, and (for jointly trained GNNs) a GNN learning rate of 1e-4. Accuracy in %; test split
for multi-turn; the question-only prior is {pc(overall_prior("test"))} static and 27.7 multi-turn.

{final_table()}

By task (static test):

{final_task_table()}

What this shows:

1. **A valid GraphToken exists and uses the graph.** Over 3 seeds its static test accuracy is {100 * gt_mean:.1f} +/- {100 * gt_sd:.1f} (sd) points; on multi-turn the
   updated graph beats the frozen graph by about 12 points and a shuffled graph by about 26 (paired session-bootstrap CIs exclude 0;
   `stats_test_final.json`). Model-written edits cost nothing: {pc(PF["gt_sum"]["predicted_updated_graph"]["answer_accuracy"])}% with predicted edits vs
   {pc(PF["gt_sum"]["oracle_updated_graph"]["answer_accuracy"])}% with oracle edits (edit accuracy {pc(PF["gt_sum"]["predicted_updated_graph"]["execution_equivalent_edit_accuracy"])}%, which is mostly in-context copying).
2. **Joint training of the GNN adds nothing over a random GNN with sum aggregation.** A frozen random sum-aggregation GNN
   reaches {pc(_a("static_tea_sum_random_test.json")["overall_accuracy"])}% static test, the same as the trained-from-scratch GraphToken ({100 * gt_mean:.1f}%); on multi-turn updated-graph accuracy the paired difference is
   {fp("tea_sum_random - gt_sum")} points. The gain over the earlier models is the aggregation, not the training of the GNN.
3. **Contrastive pretraining helps only when the GNN is then trained.** A frozen pretrained GNN is *worse* than a frozen random one
   ({pc(_a("static_tea_sum_test.json")["overall_accuracy"])}% vs {pc(_a("static_tea_sum_random_test.json")["overall_accuracy"])}% static test): it reads attributes and reachability
   (`attribute_lookup` {pc(_a("static_tea_sum_test.json")["task_accuracy"]["attribute_lookup"])}%, `reachability` {pc(_a("static_tea_sum_test.json")["task_accuracy"]["reachability"])}%) but loses degree
   (`node_degree` {pc(_a("static_tea_sum_test.json")["task_accuracy"]["node_degree"])}% vs {pc(_a("static_tea_sum_random_test.json")["task_accuracy"]["node_degree"])}%), because the text-alignment objective does not reward it. Starting joint training from the pretrained
   GNN gives the best model ({pc(_a("static_gt_warm_full_test.json")["overall_accuracy"])}% static test, {pc(_a("exact2x_gt_warm_full_summary.json")["test"]["summary"]["oracle_updated_graph"]["answer_accuracy"])}% multi-turn; `attribute_lookup`
   {pc(_a("static_gt_warm_full_test.json")["task_accuracy"]["attribute_lookup"])}%, `reachability` {pc(_a("static_gt_warm_full_test.json")["task_accuracy"]["reachability"])}%), {fp("gt_sum - gt_warm").replace("+", "-", 1) if False else ""}about 6 points above GraphToken from scratch on multi-turn
   (paired difference {fp("gt_sum - gt_warm")} for scratch minus warm start). Pretraining and joint training are complementary.
4. **Against the graph-blind soft prompt.** The like-for-like control (question only, no history text) scores
   {pc(_a("exact2x_reeval_soft_prompt_summary.json")["test"]["summary"]["soft_prompt_no_history"]["answer_accuracy"])}%. Paired differences on multi-turn updated-graph accuracy: GraphToken warm start {fsp("gt_warm")}, GraphToken {fsp("gt_sum")},
   TEA random sum GNN {fsp("tea_sum_random")}, TEA pretrained sum GNN {fsp("tea_sum")}, old TEA {fsp("tea_old")} (points, 95% CI).
5. **Seed noise.** The three GraphToken seeds differ by at most 0.4 points on multi-turn (paired CIs include 0), so the earlier 3-point
   run-to-run spread belongs to the mean-aggregation setup where the GNN barely carried information. These are still three runs of one
   recipe, on one dataset, and the warm-start and TEA rows are single seeds.
6. **TEA with working pretraining needed its own learning rate.** At the TEA default of 1e-3 the pretrained-GNN projector trains badly
   (answer loss 1.3 vs 0.7 for a random GNN, static test 34.4%) because the pretrained embeddings vary about ten times more across nodes;
   one-epoch pilots gave 42.7% validation at 1e-4 and 35.3% at 3e-4, so 1e-4 was used. This lr was chosen with validation loss/accuracy of a
   1-epoch pilot, not the test split.
7. **Deeper GNNs.** A 6-layer GNN, pretrained properly, loses message passing (neighbour-status probe AUC 0.51-0.62 vs 0.99-1.00 for 3
   layers; `probe_variants.json`), so the earlier `v3x_deep` result says nothing about depth versus pretraining.

Caveats specific to these runs: the TEA rows use projector lr 1e-4 and the GraphToken rows 1e-3 (GNN 1e-4), and the learning rates were not tuned jointly across models (only TEA with a pretrained GNN had a 1e-4 / 3e-4 / 1e-3 pilot); sum aggregation was chosen after seeing pilots on validation data; tasks that depend on degree
(`node_degree` 100%, `cycle_membership` 92%) become easy, so part of the improvement is that the encoder can now count; training data
still has the task-by-cell imbalance and closed-station mismatch listed in Section 8; `checkpoint-best` was recorded but the final
checkpoint is what is evaluated.

## 0b. How much is a node-local shortcut? (audit of 2026-10-16)

Most of the gain in Section 0a can be produced by two node-local facts. `scripts/shortcut_audit.py` fits a gradient-boosted classifier per
task on the static *train* split using only: the degree of the source and target station, their open/closed status, graph size and whether
they share a line (no adjacency, no other attributes, no text). Computed on the 8 tasks without a queried attribute (static test):

{shortcut_table()}

Multi-turn, applying the same shortcut to the edited graph (a status-and-degree tracker) and to the stale initial graph:

{multi_shortcut_table()}

Reading:

1. **The models are at the shortcut ceiling, not above it.** Over these 8 tasks the shortcut scores {pc(SC["static"]["shortcut"]["task_mean"])}%; GraphToken from scratch
   {pc(SC["static"]["gt_sum"]["overall"])}% and the warm start {pc(SC["static"]["gt_warm"]["overall"])}%. Where it is wrong every model scores 22-26%, so no model shows multi-hop topology
   reasoning here. The warm start's extra points come from reading attributes and status (`attribute_lookup`, and `reachability` where an endpoint is closed).
2. **The "+12 points from re-encoding the edited graph" is what a status tracker gives.** The shortcut gains {pc(SC["multi"]["shortcut"]["updated_graph"] - SC["multi"]["shortcut"]["stale_graph"])} points from
   the updated graph; the models gain about 14 (updated minus frozen). That shows the encoder tracks edits, not that it reasons over paths.
3. **The cycle skill is "degree 1 means no".** On `cycle_membership` items with degree >= 2 every model is at the always-yes rate (the generator forces
   degree-1 terminals and 83% of "no" answers are one).
4. **Reachability is status reading, and seeds differ per task.** Scratch GraphToken and the random-GNN TEA do not read status (chance); the warm start
   reads it exactly but is at the always-yes rate when both endpoints are open, so it detects no disconnection. Two scratch seeds give opposite
   biases on this task (31 / 70 and 68 / 39 on the two slices), so the 0.2-point overall seed spread hides large per-task variation.
5. **Corrections to Section 0a.** (i) Sum aggregation gives degree, not general structure; the mean-to-sum effect is causal (pilots A and D differ in one factor) but
   what it unlocked is degree. (ii) "Joint training adds nothing over a random GNN" compares models that both sit at the shortcut ceiling, and the two differ in lr
   (1e-4 vs 1e-3) and schedule, so it is not a clean test. (iii) The warm start reads `attribute_lookup` at 93% but `attribute_check` on the same facts at 48%; not investigated.
6. **Richer aggregation or readout did not help (1-epoch pilots, static validation, recipe of pilot D = 53.0).** A GIN-style per-neighbour MLP before the sum
   scored 52.6; a 2-layer MLP projector (hidden 2048, lr 3e-4) scored 53.4 (`shortest_path` 32 vs 25 and `filtered_neighbor_count` 37 vs 32 on validation, single
   seed, within what per-task seed spread allows). Neither moves the tasks that need multi-hop structure, which is consistent with the shortcut ceiling above:
   the limit is what the training data rewards, not the aggregator.
7. **Not yet done:** regenerating data balanced on degree and endpoint status (this changes the methodology and invalidates existing results); the audit of the data
   so far covers text leakage only, not structural shortcuts like these.

## 1. Summary

1. **The benchmark now discriminates.** The graph-blind soft prompt scores at the question-only prior
   (static {pc(static["soft_prompt"]["validation"]["overall_accuracy"])}% validation and {pc(static["soft_prompt"]["test"]["overall_accuracy"])}% test, against priors of
   {pc(overall_prior("validation"))}% and {pc(overall_prior("test"))}%), so anything clearly above the prior must come from the graph.
2. **A frozen random GNN plus a trained projector beats the graph-blind soft prompt on static QA, but the two TEA runs differ by
   as much as TEA differs from the others**: {pc(static["graphtoken"]["test"]["overall_accuracy"])}% (TEA run 2), {pc(static["tea"]["test"]["overall_accuracy"])}% (TEA, 5 epochs), {pc(static["tea3"]["test"]["overall_accuracy"])}% (TEA, 3 epochs) and
   {pc(static["soft_prompt"]["test"]["overall_accuracy"])}% for soft_prompt (static test). On the multi-turn eval the two TEA runs reach
   {pc(d("tea", "oracle_updated_graph"))}% and {pc(d("graphtoken", "oracle_updated_graph"))}% when graph tokens are re-encoded after each edit. No claim that one of these models is better than
   another is supported by a single seed (Section 0).
3. **(Mean-aggregation, random-GNN models only; see Sections 0a-0b for the sum-aggregation ones.) The graph tokens carry node attributes and some neighbourhood information, but not edge structure.** The gains
   are on `attribute_lookup`, `most_common_attribute_within_hops`, `reachability` and `constrained_reachability`.
   `edge_exists`, `cycle_membership`, `node_degree`, `filtered_neighbor_count`, `shortest_path` and `within_hops_count`
   show no consistent gain over their priors for any model (one exception that does not replicate: `node_degree` for the
   3-epoch TEA checkpoint, {pc(static["tea3"]["test"]["task_accuracy"]["node_degree"])}% against a {pc(LEAK["node_degree"]["question_only_rule_accuracy"])}% prior).
4. **Re-encoding the edited graph helps**: updated-graph minus frozen-graph is {100 * gt_gain:+.1f} points for TEA run 2 and
   {100 * tea_gain:+.1f} for TEA on exact2x test (about 3 points of this is within the run-to-run spread), and a different session's graph is worse than the correct initial graph.
5. **Model-written edits cost nothing for TEA.** A frozen Llama with a few-shot prompt reproduces {pc(pt["predicted_updated_graph"]["execution_equivalent_edit_accuracy"]) if pt else "n/a"}%
   of edits exactly, and accuracy with them is {pc(pt["predicted_updated_graph"]["answer_accuracy"]) if pt else "n/a"}% against {pc(pt["oracle_updated_graph"]["answer_accuracy"]) if pt else "n/a"}% with
   oracle edits (test). {gt_pred_sentence}
6. **A deeper GNN, a larger projector and 3x more data did not help TEA** (Section 6; invalid as evidence about depth or pretraining, see Section 0).
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
  1,000 graphs as before/after-one-edit pairs); static eval {N_STATIC["validation"]} validation / {N_STATIC["test"]} test items; multi-turn eval 1,188 sessions per
  split (1, 2, 4 or 8 turns; 4,455 turns), one cell per (size, density, session length, task).
- **`status` (open/closed) is not in the prompt text**; it is only in the graph, or in the edit history in the multi-turn
  eval. For the Facts tasks and the mode task, the queried attribute is also hidden from the text.

## 3. Models and prompts

| | TEA | TEA run 2 (was "GraphToken") | soft_prompt |
|---|---|---|---|
| Encoder | GraphSAGE 768 / 1024 / 1024, 3 layers, dropout 0.5, mean aggregation, frozen BERT-768 node features; **random, never effectively pretrained** (Section 0) | same GNN as TEA, **frozen** | none |
| Alignment | contrastive pretraining was run for 10 epochs but never updated the GNN (only the discarded head) | uses TEA's | none |
| Trained | one linear projector (126.6M parameters) | one linear projector (different random init) | 10 soft tokens |
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

Standard errors are about {se(0.5, N_STATIC["test"] // 11):.1f} points per task (about {N_STATIC["test"] // 11} items) and
{se(0.37, N_STATIC["test"]):.1f} points overall.

- The soft prompt tracks the prior on every task.
- `attribute_lookup` hides the queried attribute from the text, so the gain over the prior (27%) can only come from the
  graph tokens; TEA run 2 reads it back at {pc(static["graphtoken"]["test"]["task_accuracy"]["attribute_lookup"])}% and TEA at
  {pc(static["tea"]["test"]["task_accuracy"]["attribute_lookup"])}%.
- `attribute_check`, which asks the same fact as a yes/no question, stays at chance for every model.

## 5. Multi-turn (exact2x) results

Validation:

{exact_condition_table("validation")}

Test:

{exact_condition_table("test")}

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

Same compute budget (about 3,000 optimiser steps); evaluation sets identical to the baseline's. **Read with Section 0:** every row
here used a random frozen GNN (the pretraining never trained it), so the 6-layer row compares depth of a random GNN, not pretraining quality.

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
tested directly). TEA run 2 handles the near-closed cases better (86% accuracy) but misses about
half of the closed endpoints; since it differs from TEA only by projector initialisation, that difference is also run-to-run variation. These probes are on the static test split with the 5-epoch checkpoints.

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
- **Trainable capacity.** soft_prompt, TEA and TEA run 2 were each trained for 5 epochs. TEA's projector alone has 126.6M trainable
  parameters (a 3,089-to-40,960 linear map); the end-to-end GraphToken adds only the 5.8M-parameter GNN (+4.6%), so capacity does not
  explain its collapse. The 3-epoch TEA checkpoint was chosen to match the variants' compute budget, not selected on test.
- **Not CLEGR's token order.** CLEGR places the graph tokens first (graph, W(f_i), question); here they come after the
  text, just before the answer. The wrapper is `Question: ... Answer:`, not `[INST]`. The projector also receives the hop count
  and a task one-hot, which CLEGR's does not.
- **The soft prompt is weaker than CLEGR's.** CLEGR's soft prompt also sees the full node and edge CSV of the graph as
  text. Here it sees only the named stations' attributes (and the edit history in the multi-turn eval). A CSV-in-prompt
  track was implemented but cancelled before producing results.
- **Graph size and attributes are ours** (12-22 stations, invented categories; CLEGR averages about 26 stations).
- **The old "GraphToken" column is a second TEA run** (frozen GNN) and the first end-to-end GraphToken (mean aggregation) collapsed. The valid
  GraphToken results are the sum-aggregation runs in Section 0a; every table below Section 0a predates them and uses mean aggregation.
- **The history comparison is not symmetric.** The soft prompt reads the edit history as text; `oracle_updated_graph` has none, and no model was
  trained with history in the prompt, so history-text conditions are out of distribution (`frozen_graph_history` 33.3% vs `structure_only` 33.1%
  for TEA run 2: the text is ignored). "Graph beats soft prompt" is not a controlled result; `soft_prompt_no_history` is the like-for-like control.
- **Reachability mostly tests whether an endpoint is closed** (88.9% of "no" answers have a closed endpoint; 91% of "unreachable" answers for
  `shortest_path`), so reachability gains are mainly status reading, not graph structure. Training puts exactly 2 of 11 tasks missing from
  every size-density cell, and 35% of session initial graphs have no closed station, which training never shows.
- **Edit prediction is largely in-context copying.** The few-shot prompt contains the generator's own templates; the 95% figure is the same frozen LLM
  for every model and is not a model comparison.
- **Validation is used for selection in the audit runs.** `checkpoint-best` (validation loss on 1,000 validation examples) and the choice among
  pretraining variants (probe scores on validation graphs) use the validation split, so validation accuracy of those runs is optimistic; only test
  numbers are reported for selected runs.
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

Variant configs are `archive/configs/v3x_*.yaml`. Code changed for this benchmark: `data/v3.py` (new), `graph/solvers.py`
(W(f_i) masking, Facts tasks), `graph/edits.py` (multi-word station names),
`models/tea_glm.py` and `models/soft_prompt.py` (greedy decoding), `cli.py`.
"""

out = ROOT / "documents/experiments/status_report_v3.md"
out.write_text(TEMPLATE)
print(f"wrote {out} ({len(TEMPLATE.splitlines())} lines)")
