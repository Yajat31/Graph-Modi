#!/usr/bin/env python3
"""Cleanliness audit of the metro_v3 benchmark.

Regenerates the benchmark from the committed generator (configs/v3_tea.yaml parameters, seed 42)
into a scratch directory and audits it with code that does NOT call the generator's solvers:

  1  reproduction   regenerated data vs committed dataset_audit.json, and the pretraining data hash
  2  labels         every static tuple and every session turn re-solved by independent solvers
  3  structure      graph invariants (connectivity, diameter cap, density bins, names, lines, ...)
  4  split hygiene  seed/graph-id/isomorphism/question overlap between train, validation, test
  5  prompt text    W(f_i) contents, hidden attributes, status words, answer-in-prompt
  6  balance        answer balance inside prompt-visible and structural strata
  7  sessions       replayed edits, stale answers, NOOP / revert / cancelling edits, state drift
  8  counterfactual training pairs
  9  shortcuts      lookup-table classifiers on degree/status/size (no sklearn)
 10  latent hazards node/edge ordering, parallel edges, relation labels

Usage:
    PYTHONPATH=src python scripts/audit_benchmark.py [--work /tmp/graphmodi_bench] [--out DIR]
Writes DIR/results.json and DIR/manifest.json. Only the standard library is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from graph_modi.data import v3 as V3
from graph_modi.data.multiturn import load_sessions, save_sessions
from graph_modi.data.v2 import load_static_tuples, save_static_tuples
from graph_modi.graph.executor import apply_edit, graph_fingerprint
from graph_modi.graph.solvers import answer_query
from graph_modi.models.tea_glm import data_records_sha256
from graph_modi.schema import EditOperation, EditTarget, ReasoningType

RT = ReasoningType
SEED = 42
STATIC_TASKS = [t.value for t in V3.STATIC_TASKS]
DYNAMIC_TASKS = [t.value for t in V3.DYNAMIC_TASKS]
YES_NO = {t.value for t in V3.YES_NO_TASKS}
FACTS = {RT.ATTRIBUTE_LOOKUP.value, RT.ATTRIBUTE_CHECK.value, RT.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS.value}
EIGHT_TASKS = [
    "edge_exists", "reachability", "constrained_reachability", "cycle_membership",
    "node_degree", "filtered_neighbor_count", "shortest_path", "within_hops_count",
]
ORIGINAL_SHA = "80b1df4735af47f26f81ecc01b73db37833330a381642ee93b24fd87798f9b83"
COMMITTED_AUDIT = Path("documents/experiments/results/v3-20260929/dataset_audit.json")

RESULTS: dict[str, Any] = {}
CHECKS: list[dict[str, Any]] = []


def check(section: str, name: str, ok: bool, detail: str = "", level: str = "fail") -> bool:
    """Record a pass/fail check. ``level`` is what a failure counts as: fail, warn, or info."""
    status = "pass" if ok else level
    CHECKS.append({"section": section, "name": name, "status": status, "detail": detail})
    print(f"  [{status.upper():4}] {section}/{name}  {detail}")
    return ok


def info(section: str, name: str, detail: str) -> None:
    CHECKS.append({"section": section, "name": name, "status": "info", "detail": detail})
    print(f"  [INFO] {section}/{name}  {detail}")


# ------------------------------------------------------------------------------ generation
def generate(work: Path, degree_balance: bool = False) -> dict[str, Path]:
    """Same calls and parameters as cli._generate_v3 with configs/v3_tea.yaml, minus logging."""
    work.mkdir(parents=True, exist_ok=True)
    paths = {
        "static_train": work / "static_train.jsonl",
        "static_validation": work / "static_validation.jsonl",
        "static_test": work / "static_test.jsonl",
        "sessions_validation": work / "sessions_validation.jsonl",
        "sessions_test": work / "sessions_test.jsonl",
    }
    if all(p.exists() for p in paths.values()):
        print(f"using cached data in {work}")
        return paths
    V3.set_degree_balance(degree_balance)
    cells = len(V3.NODE_COUNTS) * len(V3.DENSITIES)
    t0 = time.time()
    train = V3.generate_static_v3(
        split="train", graph_count=2000, tasks_per_graph=9, seed=SEED,
        sampler=V3.BalancedSampler(), tasks=list(V3.STATIC_TASKS), progress=False,
    )
    train += V3.generate_counterfactual_pairs_v3(
        graph_count=1000, pairs_per_graph=3, seed=SEED, sampler=V3.BalancedSampler(),
        tasks=list(V3.DYNAMIC_TASKS), progress=False,
    )
    save_static_tuples(paths["static_train"], train)
    print(f"  static train + counterfactual: {len(train)} tuples ({time.time() - t0:.0f}s)")
    for split, reps in (("validation", 6), ("test", 12)):
        items = V3.generate_static_v3(
            split=split, graph_count=cells * reps, tasks_per_graph=None, seed=SEED,
            sampler=V3.BalancedSampler(), tasks=list(V3.STATIC_TASKS), progress=False,
        )
        save_static_tuples(paths[f"static_{split}"], items)
        print(f"  static {split}: {len(items)} tuples ({time.time() - t0:.0f}s)")
    sessions = V3.generate_sessions_v3(
        replicates_per_split={"validation": 1, "test": 1}, seed=SEED,
        tasks=list(V3.DYNAMIC_TASKS), progress=False,
    )
    for split, items in sessions.items():
        save_sessions(paths[f"sessions_{split}"], items)
        print(f"  sessions {split}: {len(items)} ({time.time() - t0:.0f}s)")
    return paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------------------ independent model
class State:
    """Minimal graph state: attributes by id, an undirected edge set. Built from a library graph
    but all reasoning on it is done here."""

    def __init__(self, graph: Any) -> None:
        self.attrs = {n.id: dict(n.attributes) for n in graph.nodes}
        self.label = {n.id: n.label for n in graph.nodes}
        self.edges = {frozenset((e.source, e.target)) for e in graph.edges}

    def copy(self) -> State:
        other = object.__new__(State)
        other.attrs = {k: dict(v) for k, v in self.attrs.items()}
        other.label = self.label
        other.edges = set(self.edges)
        return other

    def adj(self) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {i: set() for i in self.attrs}
        for edge in self.edges:
            a, b = tuple(edge)
            result[a].add(b)
            result[b].add(a)
        return result

    def is_open(self, node: str) -> bool:
        return self.attrs[node].get("status") != "closed"


def bfs(adj: dict[str, set[str]], start: str, allowed: Any = None, limit: int | None = None) -> dict[str, int]:
    dist = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if limit is not None and dist[node] >= limit:
            continue
        for nxt in adj[node]:
            if nxt not in dist and (allowed is None or allowed(nxt)):
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return dist


def bridges(adj: dict[str, set[str]]) -> set[frozenset[str]]:
    disc: dict[str, int] = {}
    low: dict[str, int] = {}
    found: set[frozenset[str]] = set()
    clock = [0]

    def visit(node: str, parent: str | None) -> None:
        disc[node] = low[node] = clock[0]
        clock[0] += 1
        for nxt in adj[node]:
            if nxt == parent:
                continue
            if nxt in disc:
                low[node] = min(low[node], disc[nxt])
            else:
                visit(nxt, node)
                low[node] = min(low[node], low[nxt])
                if low[nxt] > disc[node]:
                    found.add(frozenset((node, nxt)))

    for node in adj:
        if node not in disc:
            visit(node, None)
    return found


def solve(state: State, query: Any) -> str:
    """Independent re-implementation of the 11 task semantics (from the question text)."""
    kind = query.reasoning_type.value
    s, t = query.source, query.target
    adj = state.adj()
    attrs = state.attrs
    if kind == "edge_exists":
        return "yes" if t in adj[s] else "no"
    if kind == "reachability":
        if not (state.is_open(s) and state.is_open(t)):
            return "no"
        return "yes" if t in bfs(adj, s, allowed=state.is_open) else "no"
    if kind == "shortest_path":
        if not (state.is_open(s) and state.is_open(t)):
            return "unreachable"
        dist = bfs(adj, s, allowed=state.is_open)
        return str(dist[t] + 1) if t in dist else "unreachable"
    if kind == "constrained_reachability":
        def allowed(node: str) -> bool:
            return node in (s, t) or attrs[node].get(query.attribute) != query.value
        return "yes" if t in bfs(adj, s, allowed=allowed) else "no"
    if kind == "cycle_membership":
        cut = bridges(adj)
        return "yes" if any(frozenset((s, n)) not in cut for n in adj[s]) else "no"
    if kind == "node_degree":
        return str(len(adj[s]))
    if kind == "filtered_neighbor_count":
        return str(sum(attrs[n].get(query.attribute) == query.value for n in adj[s]))
    if kind == "within_hops_count":
        hops = query.hops or 2
        near = [n for n, d in bfs(adj, s, limit=hops).items() if 1 <= d <= hops]
        return str(sum(attrs[n].get(query.attribute) == query.value for n in near))
    if kind == "most_common_attribute_within_hops":
        hops = query.hops or 2
        near = [n for n, d in bfs(adj, s, limit=hops).items() if d <= hops]
        counts = Counter(str(attrs[n].get(query.attribute)) for n in near)
        best = max(counts.values())
        return sorted(v for v, c in counts.items() if c == best)[0]
    if kind == "attribute_lookup":
        return str(attrs[s].get(query.attribute))
    if kind == "attribute_check":
        return "yes" if attrs[s].get(query.attribute) == query.value else "no"
    raise ValueError(kind)


def mode_is_unique(state: State, query: Any) -> bool:
    adj = state.adj()
    hops = query.hops or 2
    near = [n for n, d in bfs(adj, query.source, limit=hops).items() if d <= hops]
    top = Counter(str(state.attrs[n].get(query.attribute)) for n in near).most_common(2)
    return len(top) == 1 or top[0][1] > top[1][1]


def apply_indep(state: State, edits: Any) -> tuple[State, list[str]]:
    """Apply gold edits with our own semantics; returns the new state and any anomalies."""
    state = state.copy()
    anomalies: list[str] = []
    for edit in edits:
        if edit.operation is EditOperation.NOOP:
            continue
        if edit.target is EditTarget.NODE and edit.operation is EditOperation.SET:
            if state.attrs[edit.node_id].get(edit.attribute) == edit.value:
                anomalies.append("set_noop")
            state.attrs[edit.node_id][edit.attribute] = edit.value
        elif edit.target is EditTarget.EDGE:
            key = frozenset((edit.source, edit.destination))
            if edit.operation is EditOperation.ADD:
                if key in state.edges:
                    anomalies.append("add_existing_edge")
                state.edges.add(key)
            elif edit.operation is EditOperation.DEL:
                if key not in state.edges:
                    anomalies.append("del_missing_edge")
                state.edges.discard(key)
        else:
            anomalies.append(f"unsupported:{edit.operation.value}:{edit.target}")
    return state, anomalies


# ------------------------------------------------------------------------------ 1 reproduction
def section_reproduction(paths: dict[str, Path], static: dict[str, list], sessions: dict[str, list]) -> None:
    print("\n[1] Reproduction")
    if V3.DEGREE_BALANCE:
        train_graphs = list({item.graph.graph_id: item.graph for item in static["train"]}.values())
        digest = data_records_sha256([g.to_dict() for g in train_graphs])
        RESULTS["reproduction"] = {"pretraining_graph_sha256": digest, "original": None}
        info("reproduction", "degree_balanced_variant", f"no committed audit to compare against; pretraining hash {digest[:12]}")
        return
    train_graphs = list({item.graph.graph_id: item.graph for item in static["train"]}.values())
    digest = data_records_sha256([g.to_dict() for g in train_graphs])
    RESULTS["reproduction"] = {"pretraining_graph_sha256": digest, "original": ORIGINAL_SHA}
    check("reproduction", "pretraining_data_hash", digest == ORIGINAL_SHA,
          f"regenerated {digest[:12]} vs original {ORIGINAL_SHA[:12]}")

    if not COMMITTED_AUDIT.exists():
        info("reproduction", "committed_audit", "dataset_audit.json not found; skipped")
        return
    committed = json.loads(COMMITTED_AUDIT.read_text())
    for split in ("validation", "test"):
        sess = sessions[split]
        turns = [t for s in sess for t in s.turns]
        ops = Counter()
        for turn in turns:
            for edit in turn.gold_edits or (turn.gold_edit,):
                ops[edit.operation.value] += 1
        answers = Counter(t.gold_answer for t in turns)
        reference = committed[split]
        check("reproduction", f"{split}_operation_distribution",
              dict(ops) == reference["operation_distribution"], f"{dict(ops)}")
        check("reproduction", f"{split}_answer_distribution",
              dict(answers) == reference["answer_distribution"], "")
        changed = sum(t.gold_answer != t.stale_answer for t in turns) / len(turns)
        check("reproduction", f"{split}_answer_changing_fraction",
              abs(changed - reference["labels"]["_answer_changing_fraction"]) < 1e-12, f"{changed:.4f}")
    for split in ("train", "validation", "test"):
        key = f"static_{split}"
        labels = V3.label_audit(static[split])
        same = all(
            [list(x) for x in labels[task]["top_answers"]] == committed[key]["labels"][task]["top_answers"]
            for task in labels if not task.startswith("_")
        )
        check("reproduction", f"{key}_label_counts", same and len(static[split]) == committed[key]["tuples"],
              f"{len(static[split])} tuples")


# ------------------------------------------------------------------------------ 2 labels
def section_labels(static: dict[str, list], sessions: dict[str, list]) -> None:
    print("\n[2] Labels re-solved independently")
    report: dict[str, Any] = {}
    for split, items in static.items():
        wrong: list[str] = []
        by_task = Counter()
        library_wrong = 0
        for item in items:
            state = State(item.graph)
            got = solve(state, item.query)
            by_task[item.query.reasoning_type.value] += 1
            if got != item.answer:
                wrong.append(f"{item.tuple_id}:{item.query.reasoning_type.value}:{item.answer}!={got}")
            if answer_query(item.graph, item.query) != item.answer:
                library_wrong += 1
            if item.query.reasoning_type is RT.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS and not mode_is_unique(state, item.query):
                wrong.append(f"{item.tuple_id}:tied_mode")
        per_task = Counter(w.split(":")[1] for w in wrong if ":" in w)
        report[split] = {"checked": len(items), "mismatches": len(wrong), "examples": wrong[:5], "by_task": dict(per_task)}
        check("labels", f"static_{split}", not wrong, f"{len(items)} items, {len(wrong)} mismatches {wrong[:3]}")
        check("labels", f"static_{split}_library_consistency", library_wrong == 0, f"{library_wrong} library mismatches")
    for split, sess in sessions.items():
        mismatches: list[str] = []
        stale_wrong = 0
        fp_wrong = 0
        turns = 0
        anomalies: Counter[str] = Counter()
        for session in sess:
            state = State(session.initial_graph)
            for turn in session.turns:
                turns += 1
                edits = turn.gold_edits or (turn.gold_edit,)
                after, bad = apply_indep(state, edits)
                anomalies.update(bad)
                if solve(after, turn.query) != turn.gold_answer:
                    mismatches.append(f"{session.session_id}/{turn.turn_index}")
                if solve(state, turn.query) != turn.stale_answer:
                    stale_wrong += 1
                if turn.query.reasoning_type is RT.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS and not mode_is_unique(after, turn.query):
                    mismatches.append(f"{session.session_id}/{turn.turn_index}:tied_mode")
                state = after
            # the independent replay must land on the same graph as the library replay
            from graph_modi.graph.executor import apply_edit_program
            lib = session.initial_graph
            for turn in session.turns:
                edits = turn.gold_edits or (turn.gold_edit,)
                lib = apply_edit_program(lib, edits).graph
            ref = State(lib)
            if ref.attrs != state.attrs or ref.edges != state.edges:
                fp_wrong += 1
        report[f"sessions_{split}"] = {"turns": turns, "label_mismatches": len(mismatches),
                                       "stale_mismatches": stale_wrong, "replay_divergence": fp_wrong,
                                       "edit_anomalies": dict(anomalies)}
        check("labels", f"sessions_{split}_gold", not mismatches, f"{turns} turns, {len(mismatches)} mismatches")
        check("labels", f"sessions_{split}_stale", stale_wrong == 0, f"{stale_wrong} stale mismatches")
        check("labels", f"sessions_{split}_replay", fp_wrong == 0, f"{fp_wrong} sessions diverge from library replay")
        check("labels", f"sessions_{split}_edit_anomalies", not anomalies, f"{dict(anomalies)}")
    RESULTS["labels"] = report


def section_cycle_labels(static: dict[str, list]) -> None:
    """Characterise the cycle_membership disagreement between the library and the independent solver."""
    print("\n[2b] cycle_membership label disagreement")
    out: dict[str, Any] = {}
    for split, items in static.items():
        wrong_by_deg: Counter[int] = Counter(); total_by_deg: Counter[int] = Counter()
        lib_yes = true_yes = n = 0
        for item in items:
            if item.query.reasoning_type is not RT.CYCLE_MEMBERSHIP:
                continue
            state = State(item.graph)
            truth = solve(state, item.query)
            deg = len(state.adj()[item.query.source])
            total_by_deg[deg] += 1; n += 1
            lib_yes += item.answer == "yes"; true_yes += truth == "yes"
            wrong_by_deg[deg] += truth != item.answer
        wrong = sum(wrong_by_deg.values())
        ge3 = sum(v for d, v in total_by_deg.items() if d >= 3)
        out[split] = {"items": n, "wrong": wrong, "wrong_rate": wrong / n, "wrong_among_degree_ge3": wrong / ge3,
                      "stored_yes_rate": lib_yes / n, "true_yes_rate": true_yes / n,
                      "wrong_by_degree": dict(sorted(wrong_by_deg.items()))}
        info("labels", f"cycle_{split}", f"{wrong}/{n} stored labels wrong ({100 * wrong / n:.1f}%); {100 * wrong / ge3:.1f}% of degree>=3 queries; "
             f"stored yes-rate {lib_yes / n:.3f} vs true {true_yes / n:.3f}; wrong by degree {dict(sorted(wrong_by_deg.items()))}")
    RESULTS["cycle_labels"] = out


# ------------------------------------------------------------------------------ 3 structure
MEAN_DEGREE = {k.value: v for k, v in V3.MEAN_DEGREE.items()}
MAX_DIAMETER = {k.value: v for k, v in V3.MAX_DIAMETER.items()}
DOMAINS = {
    "architecture": set(V3.ARCHITECTURES), "cleanliness": set(V3.CLEANLINESS),
    "music": set(V3.MUSIC), "size": set(V3.SIZES), "line": set(V3.LINES),
    "status": {"open", "closed"}, "disabled_access": {True, False}, "has_rail": {True, False},
}


def graph_issues(graph: Any, *, closed_range: tuple[int, int]) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    ids = [n.id for n in graph.nodes]
    if len(set(ids)) != len(ids):
        issues.append("duplicate_ids")
    if len({n.label for n in graph.nodes}) != len(ids):
        issues.append("duplicate_names")
    pairs = Counter(frozenset((e.source, e.target)) for e in graph.edges)
    if any(c > 1 for c in pairs.values()):
        issues.append("parallel_edges")
    if any(len(p) < 2 for p in pairs):
        issues.append("self_loop")
    state = State(graph)
    adj = state.adj()
    dist0 = bfs(adj, ids[0])
    if len(dist0) != len(ids):
        issues.append("disconnected")
        return issues, {}
    ecc = [max(bfs(adj, n).values()) for n in ids]
    meta = graph.metadata
    density = str(meta["density"])
    n = len(ids)
    mean_degree = 2 * len(pairs) / n
    if max(ecc) > MAX_DIAMETER[density]:
        issues.append("diameter_cap")
    if meta.get("diameter") != max(ecc) or meta.get("radius") != min(ecc):
        issues.append("metadata_diameter")
    if abs(meta.get("mean_degree", -1) - round(mean_degree, 3)) > 1e-9:
        issues.append("metadata_mean_degree")
    if abs(mean_degree - MEAN_DEGREE[density]) > 0.35:
        issues.append("density_off_target")
    if meta.get("node_count") != n:
        issues.append("metadata_node_count")
    terminals = sum(1 for node in ids if len(adj[node]) == 1)
    if terminals < (2 if n < 17 else 3):
        issues.append("too_few_terminals")
    for node in graph.nodes:
        for key, domain in DOMAINS.items():
            if node.attributes.get(key) not in domain:
                issues.append(f"bad_attr:{key}")
                break
    closed = sum(not state.is_open(i) for i in ids)
    if not closed_range[0] <= closed <= closed_range[1]:
        issues.append("closed_count_range")
    for line in set(a["line"] for a in state.attrs.values()):
        members = {i for i, a in state.attrs.items() if a["line"] == line}
        start = next(iter(members))
        if set(bfs(adj, start, allowed=lambda x, m=members: x in m)) != members:
            issues.append("line_not_connected")
    line_count = len({a["line"] for a in state.attrs.values()})
    if line_count != meta.get("line_count"):
        issues.append("line_count")
    for edge in graph.edges:
        same = state.attrs[edge.source]["line"] == state.attrs[edge.target]["line"]
        if edge.attributes and (edge.relation == "track") != same:
            issues.append("relation_vs_line")
        if edge.weight != 1.0:
            issues.append("weight_not_one")
    return issues, {"n": n, "density": density, "mean_degree": mean_degree, "diameter": max(ecc),
                    "terminals": terminals, "closed": closed}


def section_structure(static: dict[str, list], sessions: dict[str, list]) -> dict[str, list[dict[str, Any]]]:
    print("\n[3] Graph structure")
    profiles: dict[str, list[dict[str, Any]]] = {}
    for split, items in static.items():
        seen: dict[str, Any] = {}
        for item in items:
            seen.setdefault(item.graph.graph_id, item.graph)
        counter: Counter[str] = Counter()
        rows = []
        for graph in seen.values():
            if "-cf" in graph.graph_id:
                continue  # edited counterfactual graphs are checked in section 8
            problems, profile = graph_issues(graph, closed_range=(1, 3))
            counter.update(problems)
            if profile:
                rows.append(profile)
        profiles[f"static_{split}"] = rows
        check("structure", f"static_{split}", not counter, f"{len(rows)} graphs, issues {dict(counter)}")
    for split, sess in sessions.items():
        counter = Counter()
        rows = []
        for session in sess:
            problems, profile = graph_issues(session.initial_graph, closed_range=(0, 2))
            counter.update(problems)
            if profile:
                rows.append(profile)
        profiles[f"sessions_{split}"] = rows
        check("structure", f"sessions_{split}_initial", not counter, f"{len(rows)} graphs, issues {dict(counter)}")
    # density bins must separate (mean degree ranges must not overlap across bins)
    rows = [r for key, v in profiles.items() if key.startswith("static_") for r in v]
    by_density: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_density[r["density"]].append(r["mean_degree"])
    ranges = {d: (min(v), max(v)) for d, v in by_density.items()}
    ordered = ["sparse", "medium", "dense"]
    gaps = [ranges[a][1] < ranges[b][0] for a, b in zip(ordered, ordered[1:])]
    info("structure", "density_ranges", f"{ {d: (round(a, 2), round(b, 2)) for d, (a, b) in ranges.items()} }")
    check("structure", "density_bins_separate", all(gaps), "mean-degree ranges overlap across bins" if not all(gaps) else "", "warn")
    cells = Counter((r["n"], r["density"]) for r in profiles["static_test"])
    check("structure", "test_grid_exact", len(cells) == 33 and len(set(cells.values())) == 1, f"{len(cells)} cells, {set(cells.values())} per cell")
    RESULTS["structure"] = {"density_ranges": ranges}
    return profiles


# ------------------------------------------------------------------------------ 4 split hygiene
def wl_hash(graph: Any, *, use_attrs: bool, rounds: int = 4) -> str:
    state = State(graph)
    adj = state.adj()
    color = {
        i: hashlib.md5(json.dumps(sorted(state.attrs[i].items()) if use_attrs else [], default=str).encode()).hexdigest()
        for i in adj
    }
    for _ in range(rounds):
        color = {
            i: hashlib.md5((color[i] + "|" + ",".join(sorted(color[j] for j in adj[i]))).encode()).hexdigest()
            for i in adj
        }
    return hashlib.md5(",".join(sorted(color.values())).encode()).hexdigest()


def section_splits(static: dict[str, list], sessions: dict[str, list]) -> None:
    print("\n[4] Split hygiene")
    graphs: dict[str, dict[str, Any]] = defaultdict(dict)
    for split, items in static.items():
        for item in items:
            graphs[f"static_{split}"].setdefault(item.graph.graph_id, item.graph)
    for split, sess in sessions.items():
        for session in sess:
            graphs[f"sessions_{split}"][session.initial_graph.graph_id] = session.initial_graph
    # seeds used by the generator (seed + offset + index) must be unique across all families
    seeds: dict[str, set[int]] = {
        "static_train": {SEED + i for i in range(2000)},
        "static_validation": {SEED + 5_000_000 + i for i in range(198)},
        "static_test": {SEED + 6_000_000 + i for i in range(396)},
        "counterfactual": {SEED + 8_000_000 + i for i in range(1000)},
        "sessions_validation": {SEED + 1_000_000 + i for i in range(1188)},
        "sessions_test": {SEED + 2_000_000 + i for i in range(1188)},
    }
    collisions = {(a, b): len(seeds[a] & seeds[b]) for a in seeds for b in seeds if a < b and seeds[a] & seeds[b]}
    check("splits", "rng_seeds_unique", not collisions, f"{collisions}")
    ids = {k: set(v) for k, v in graphs.items()}
    names = sorted(ids)
    overlap = {(a, b): len(ids[a] & ids[b]) for i, a in enumerate(names) for b in names[i + 1:] if ids[a] & ids[b]}
    info("splits", "graph_id_name_reuse", f"{overlap} (ids are per-split counters like test-v3-graph-000000; names collide, graphs differ: see fingerprint checks)")
    # isomorphism-style overlap (attributed WL hash), train vs evaluation graphs
    hashes: dict[str, dict[str, str]] = {
        key: {gid: wl_hash(g, use_attrs=True) for gid, g in graph_map.items() if "-cf" not in gid}
        for key, graph_map in graphs.items()
    }
    train_hashes = set(hashes["static_train"].values())
    report: dict[str, Any] = {}
    for key in ("static_validation", "static_test", "sessions_validation", "sessions_test"):
        shared = sum(h in train_hashes for h in hashes[key].values())
        report[key] = shared
    check("splits", "attributed_graph_overlap_with_train", all(v == 0 for v in report.values()), f"{report}")
    eval_pairs = {
        "validation_vs_test": set(hashes["static_validation"].values()) & set(hashes["static_test"].values()),
        "sessions_val_vs_test": set(hashes["sessions_validation"].values()) & set(hashes["sessions_test"].values()),
        "static_test_vs_sessions_test": set(hashes["static_test"].values()) & set(hashes["sessions_test"].values()),
    }
    check("splits", "attributed_graph_overlap_between_eval_splits",
          all(not v for v in eval_pairs.values()), f"{ {k: len(v) for k, v in eval_pairs.items()} }")
    topo_train = {wl_hash(g, use_attrs=False) for gid, g in graphs["static_train"].items() if "-cf" not in gid}
    topo = {}
    for key in ("static_test", "sessions_test"):
        topo[key] = sum(wl_hash(g, use_attrs=False) in topo_train for g in graphs[key].values())
    info("splits", "topology_only_overlap_with_train", f"test graphs whose bare topology also occurs in train: {topo}")
    # identical questions / identical (graph-content, query) pairs across splits
    def questions(items: list) -> set[str]:
        return {item.query.question for item in items}
    train_q = questions(static["static_train"] if "static_train" in static else static["train"])
    exact = {s: len(train_q & questions(static[s])) for s in ("validation", "test")}
    info("splits", "identical_question_strings_with_train",
         f"{exact} (questions embed station names + attributes; a few collisions are expected from the 240-name pool)")
    fingerprints: dict[str, set[str]] = {s: {graph_fingerprint(i.graph) + i.query.question for i in static[s]} for s in static}
    pair_overlap = {s: len(fingerprints["train"] & fingerprints[s]) for s in ("validation", "test")}
    check("splits", "identical_graph_query_pairs_with_train", all(v == 0 for v in pair_overlap.values()), f"{pair_overlap}")
    # station-name pool is shared by design
    pool = {n.label for g in graphs["static_train"].values() for n in g.nodes}
    test_pool = {n.label for g in graphs["static_test"].values() for n in g.nodes}
    info("splits", "station_name_pool", f"train uses {len(pool)} names, test uses {len(test_pool)}, test-only names {len(test_pool - pool)} (names are exchangeable noise: shuffled per graph)")
    RESULTS["splits"] = {"overlap_with_train": report, "topology_overlap": topo}


# ------------------------------------------------------------------------------ 5 prompt text
def expected_prefix(graph: Any, query: Any) -> str:
    kind = query.reasoning_type.value
    hidden = {"status"} | ({query.attribute} if kind in FACTS and query.attribute else set())
    nodes = {n.id: n for n in graph.nodes}
    texts = []
    for node_id in (query.source, query.target):
        if not node_id:
            continue
        node = nodes[node_id]
        attrs = ", ".join(f"{k}={v}" for k, v in sorted(node.attributes.items()) if k not in hidden)
        texts.append(f"{node.label}: {attrs}.")
    return " ".join(dict.fromkeys(texts))


def section_prompt(static: dict[str, list]) -> None:
    print("\n[5] Prompt text")
    problems: Counter[str] = Counter()
    answer_in_prompt: Counter[str] = Counter()
    numeric_coincidence: Counter[str] = Counter()
    numeric_total: Counter[str] = Counter()
    checked = 0
    for split, items in static.items():
        for item in items:
            query, graph = item.query, item.graph
            kind = query.reasoning_type.value
            text = query.question
            checked += 1
            prefix = expected_prefix(graph, query)
            if not text.startswith(prefix + "\n"):
                problems["prefix_differs_from_spec"] += 1
                continue
            body = text[len(prefix) + 1:]
            labels = {n.id: n.label for n in graph.nodes}
            if labels[query.source] not in body:
                problems["source_missing"] += 1
            if query.target and labels[query.target] not in body:
                problems["target_missing"] += 1
            if query.attribute and str(query.attribute) not in body:
                problems["attribute_missing"] += 1
            if query.value is not None and str(query.value) not in body:
                problems["value_missing"] += 1
            low = text.lower()
            if "status" in low or "closed" in low:
                problems["status_word_in_prompt"] += 1
            if kind not in ("reachability", "shortest_path") and " open" in low:
                problems["open_word_in_prompt"] += 1
            tokens = set(low.replace("?", " ").replace(",", " ").replace(".", " ").replace(":", " ").replace("=", " ").split())
            answer = item.answer.lower()
            if kind not in YES_NO and not answer.isdigit() and answer != "unreachable" and answer in tokens:
                answer_in_prompt[kind] += 1
            if answer.isdigit():
                numeric_total[kind] += 1
                if answer in tokens:
                    numeric_coincidence[kind] += 1
    check("prompt", "prefix_matches_spec", problems["prefix_differs_from_spec"] == 0,
          f"{checked} prompts; W(f_i) = node attributes minus status (and the queried attribute on Facts tasks)")
    other = {k: v for k, v in problems.items() if k != "prefix_differs_from_spec"}
    check("prompt", "question_fields_present", not any(k.endswith("_missing") for k in other), f"{other}")
    check("prompt", "no_status_words", not other.get("status_word_in_prompt"), f"{other.get('status_word_in_prompt', 0)} prompts mention status/closed")
    check("prompt", "no_stray_open_wording", not other.get("open_word_in_prompt"), f"{other.get('open_word_in_prompt', 0)}", "warn")
    check("prompt", "categorical_answer_not_in_prompt", not answer_in_prompt, f"{dict(answer_in_prompt)}")
    info("prompt", "numeric_answer_equals_a_prompt_number",
         f"{ {k: f'{numeric_coincidence[k]}/{numeric_total[k]}' for k in sorted(numeric_total)} } (hop counts in the template; coincidence, not leakage)")
    leak = V3.text_leak_audit(static["train"], static["test"])
    flagged = [t for t, v in leak.items() if v["leak_flag"]]
    check("prompt", "text_only_rule_leak_audit", not flagged, f"flagged tasks: {flagged}")
    RESULTS["prompt"] = {"checked": checked, "problems": dict(problems), "leak_flags": flagged}


# ------------------------------------------------------------------------------ features
def features(state: State, query: Any) -> dict[str, Any]:
    adj = state.adj()
    s, t = query.source, query.target
    same = None if t is None else state.attrs[s]["line"] == state.attrs[t]["line"]
    return {
        "ds": len(adj[s]), "dt": len(adj[t]) if t else None,
        "ss": state.attrs[s]["status"], "st": state.attrs[t]["status"] if t else None,
        "same": same, "n": len(state.attrs),
        "nc": sum(a["status"] == "closed" for a in state.attrs.values()),
    }


LADDERS = {
    "majority": [()],
    "status_only": [("ss", "st"), ("ss",), ()],
    "degree_only": [("ds", "dt"), ("ds",), ()],
    "degree_status": [("ds", "dt", "ss", "st", "same"), ("ds", "dt", "ss", "st"), ("ds", "ss", "st"), ("ss", "st"), ()],
}


class Lookup:
    """Hierarchical lookup table: the finest feature cell with at least ``min_count`` examples."""

    def __init__(self, ladder: list[tuple[str, ...]], min_count: int = 3) -> None:
        self.ladder = ladder
        self.min_count = min_count
        self.tables: list[dict[tuple, Counter]] = [defaultdict(Counter) for _ in ladder]

    def fit(self, rows: list[tuple[dict[str, Any], str]]) -> Lookup:
        for feats, answer in rows:
            for level, table in zip(self.ladder, self.tables):
                table[tuple(feats[k] for k in level)][answer] += 1
        return self

    def predict(self, feats: dict[str, Any]) -> str:
        for level, table in zip(self.ladder, self.tables):
            counts = table.get(tuple(feats[k] for k in level))
            if counts and sum(counts.values()) >= self.min_count:
                return counts.most_common(1)[0][0]
        return self.tables[-1][()].most_common(1)[0][0]


def static_rows(items: list) -> dict[str, list[tuple[dict[str, Any], str]]]:
    rows: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for item in items:
        rows[item.query.reasoning_type.value].append((features(State(item.graph), item.query), item.answer))
    return rows


# ------------------------------------------------------------------------------ 6 balance
def section_balance(static: dict[str, list], rows: dict[str, dict[str, list]]) -> None:
    print("\n[6] Balance and answer structure")
    report: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        worst = 0.0
        for task in YES_NO:
            answers = [a for _, a in rows[split][task]]
            if split == "train":
                # counterfactual pairs add exactly one yes and one no per pair; still report overall
                pass
            p_yes = answers.count("yes") / len(answers)
            worst = max(worst, abs(p_yes - 0.5))
        check("balance", f"{split}_yes_no_overall", worst < 0.02, f"max |P(yes)-0.5| = {worst:.3f}")
    # TV distance between train and test answer marginals per task
    tv = {}
    for task in STATIC_TASKS:
        a = Counter(x for _, x in rows["train"][task]); b = Counter(x for _, x in rows["test"][task])
        na, nb = sum(a.values()), sum(b.values())
        tv[task] = round(0.5 * sum(abs(a[k] / na - b[k] / nb) for k in set(a) | set(b)), 3)
    info("balance", "train_vs_test_answer_marginal_TV", f"{tv}")
    report["tv"] = tv

    def rate(task: str, predicate: Any, positive: str = "yes", split: str = "train") -> tuple[float, int]:
        sel = [a for f, a in rows[split][task] if predicate(f)]
        return (sum(a == positive for a in sel) / len(sel) if sel else float("nan")), len(sel)

    out: dict[str, Any] = {}
    for split in ("train", "test"):
        reach_closed = rate("reachability", lambda f: "closed" in (f["ss"], f["st"]), "no", split)
        reach_open = rate("reachability", lambda f: "closed" not in (f["ss"], f["st"]), "no", split)
        no_total = sum(a == "no" for _, a in rows[split]["reachability"])
        no_closed = sum(a == "no" and "closed" in (f["ss"], f["st"]) for f, a in rows[split]["reachability"])
        sp_unreach = sum(a == "unreachable" for _, a in rows[split]["shortest_path"])
        sp_closed = sum(a == "unreachable" and "closed" in (f["ss"], f["st"]) for f, a in rows[split]["shortest_path"])
        cyc_no = sum(a == "no" for _, a in rows[split]["cycle_membership"])
        cyc_no_deg1 = sum(a == "no" and f["ds"] == 1 for f, a in rows[split]["cycle_membership"])
        cyc_deg1 = rate("cycle_membership", lambda f: f["ds"] == 1, "no", split)
        crc_closed = rate("constrained_reachability", lambda f: "closed" in (f["ss"], f["st"]), "yes", split)
        crc_open = rate("constrained_reachability", lambda f: "closed" not in (f["ss"], f["st"]), "yes", split)
        edge_term = rate("edge_exists", lambda f: min(f["ds"], f["dt"]) == 1, "yes", split)
        edge_other = rate("edge_exists", lambda f: min(f["ds"], f["dt"]) > 1, "yes", split)
        endpoint_closed_share = sum("closed" in (f["ss"], f["st"]) for f, _ in rows[split]["reachability"]) / len(rows[split]["reachability"])
        out[split] = {
            "P(reach=no | an endpoint closed)": reach_closed, "P(reach=no | both open)": reach_open,
            "share of reach=no with a closed endpoint": (no_closed, no_total),
            "share of shortest_path=unreachable with a closed endpoint": (sp_closed, sp_unreach),
            "share of cycle=no with degree 1": (cyc_no_deg1, cyc_no),
            "P(cycle=no | degree 1)": cyc_deg1,
            "P(constrained_reach=yes | endpoint closed)": crc_closed, "P(constrained_reach=yes | both open)": crc_open,
            "P(edge_exists=yes | min degree 1)": edge_term, "P(edge_exists=yes | min degree >1)": edge_other,
            "share of reachability queries with a closed endpoint": endpoint_closed_share,
        }
    for key, value in out["test"].items():
        info("balance", key, f"train={out['train'][key]} test={value}")
    report["structure"] = out
    t = out["test"]
    share_no_closed = t["share of reach=no with a closed endpoint"]
    check("balance", "reachability_label_not_decided_by_endpoint_status",
          share_no_closed[0] / share_no_closed[1] < 0.7,
          f"{share_no_closed[0]}/{share_no_closed[1]} of 'no' answers have a closed endpoint", "warn")
    cyc = t["share of cycle=no with degree 1"]
    check("balance", "cycle_label_not_decided_by_degree",
          cyc[0] / cyc[1] < 0.7, f"{cyc[0]}/{cyc[1]} of 'no' answers are degree-1 stations", "warn")
    crc_gap = abs(t["P(constrained_reach=yes | endpoint closed)"][0] - t["P(constrained_reach=yes | both open)"][0])
    check("balance", "constrained_reachability_independent_of_status", crc_gap < 0.1,
          f"status is irrelevant by definition; label gap {crc_gap:.3f}", "warn")
    RESULTS["balance"] = report


# ------------------------------------------------------------------------------ 7 sessions
def components(adj: dict[str, set[str]]) -> int:
    seen: set[str] = set()
    count = 0
    for node in adj:
        if node not in seen:
            count += 1
            seen |= set(bfs(adj, node))
    return count


def edit_kind(edits: Any) -> str:
    ops = [e for e in edits if e.operation is not EditOperation.NOOP]
    if not ops:
        return "noop"
    kinds = {"set" if e.target is EditTarget.NODE else "edge" for e in ops}
    return "set_only" if kinds == {"set"} else "edge_only" if kinds == {"edge"} else "mixed"


def section_sessions(sessions: dict[str, list], static_profiles: list[dict[str, Any]]) -> None:
    print("\n[7] Sessions")
    report: dict[str, Any] = {}
    train_closed = sum(p["closed"] / p["n"] for p in static_profiles) / len(static_profiles)
    for split, sess in sessions.items():
        by_task_kind: dict[tuple[str, str], list[bool]] = defaultdict(list)
        drift: dict[int, list[tuple[float, float, int, float]]] = defaultdict(list)
        drift_by_len: dict[int, list[tuple[float, float]]] = defaultdict(list)
        revert = net_noop = cancelling = repeated = non_noop = 0
        added_same_line = added_total = 0
        added_attr_free = 0
        weights = Counter()
        op_counts = Counter()
        noop_by_len: dict[int, list[int]] = defaultdict(list)
        task_constant = True
        stale_equals = Counter()
        for session in sess:
            state = State(session.initial_graph)
            seen_states = {(frozenset(i for i in state.attrs if not state.is_open(i)), frozenset(state.edges))}
            queries = Counter()
            tasks = {t.query.reasoning_type.value for t in session.turns}
            task_constant &= len(tasks) == 1
            length = len(session.turns)
            for turn in session.turns:
                edits = turn.gold_edits or (turn.gold_edit,)
                kind = edit_kind(edits)
                ops = [e for e in edits if e.operation is not EditOperation.NOOP]
                noop_by_len[length].append(kind == "noop")
                after, _ = apply_indep(state, edits)
                key = (frozenset(i for i in after.attrs if not after.is_open(i)), frozenset(after.edges))
                if ops:
                    non_noop += 1
                    if key == (frozenset(i for i in state.attrs if not state.is_open(i)), frozenset(state.edges)):
                        net_noop += 1
                    elif key in seen_states:
                        revert += 1
                    touched = Counter(
                        (e.node_id,) if e.target is EditTarget.NODE else (frozenset((e.source, e.destination)), "e")
                        for e in ops
                    )
                    if any(c > 1 for c in touched.values()):
                        cancelling += 1
                    for e in ops:
                        op_counts[(e.target.value, e.operation.value)] += 1
                        if e.target is EditTarget.EDGE and e.operation is EditOperation.ADD:
                            added_total += 1
                            added_same_line += state.attrs[e.source]["line"] == state.attrs[e.destination]["line"]
                            weights[e.weight] += 1
                            added_attr_free += 1  # executor creates added edges without attributes
                seen_states.add(key)
                q = turn.query
                qkey = (q.reasoning_type.value, q.source, q.target, q.attribute, str(q.value), q.hops)
                if queries[qkey]:
                    repeated += 1
                queries[qkey] += 1
                changed = turn.gold_answer != turn.stale_answer
                by_task_kind[(q.reasoning_type.value, kind)].append(changed)
                stale_equals[kind] += not changed
                adj = after.adj()
                closed = sum(not after.is_open(i) for i in after.attrs)
                drift[turn.turn_index].append((closed / len(after.attrs), 2 * len(after.edges) / len(after.attrs),
                                               components(adj), float(components(adj) == 1)))
                state = after
            last = drift[length - 1][-1] if False else None  # per-length summary below
            adj = state.adj()
            drift_by_len[length].append((sum(not state.is_open(i) for i in state.attrs) / len(state.attrs), float(components(adj) == 1)))
        total_turns = sum(len(s.turns) for s in sess)
        noop_rate = sum(sum(v) for v in noop_by_len.values()) / total_turns
        info("sessions", f"{split}_noop_rate", f"{noop_rate:.3f} overall; by session length "
             f"{ {k: round(sum(v) / len(v), 3) for k, v in sorted(noop_by_len.items())} } (docs say 15-20%)")
        check("sessions", f"{split}_one_task_per_session", task_constant, "every session asks one task for all of its turns", "info")
        check("sessions", f"{split}_net_noop_programs", net_noop == 0, f"{net_noop}/{non_noop} non-NOOP programs leave the graph unchanged")
        check("sessions", f"{split}_cancelling_within_program", cancelling == 0,
              f"{cancelling}/{non_noop} programs touch the same node/edge twice", "warn")
        check("sessions", f"{split}_reverts_to_earlier_state", revert == 0,
              f"{revert}/{non_noop} edit turns return the graph to a state it had before", "info")
        check("sessions", f"{split}_repeated_queries", repeated == 0, f"{repeated}/{total_turns} turns repeat an earlier query in the same session", "info")
        info("sessions", f"{split}_added_edges", f"{added_total} added; {added_same_line} join same-line stations but are labelled 'transfer' (original edges: track iff same line); "
             f"{added_attr_free} carry no edge attributes; weights {dict(weights)}")
        info("sessions", f"{split}_op_counts", f"{dict(op_counts)}")
        # answer-changing structure
        table: dict[str, dict[str, Any]] = {}
        for task in DYNAMIC_TASKS:
            row = {}
            for kind in ("noop", "set_only", "edge_only", "mixed"):
                v = by_task_kind.get((task, kind), [])
                row[kind] = (round(sum(v) / len(v), 3) if v else None, len(v))
            table[task] = row
        report[f"{split}_p_answer_changes"] = table
        irrelevant = {t: table[t]["set_only"] for t in DYNAMIC_TASKS if t not in ("reachability", "shortest_path")}
        check("sessions", f"{split}_status_edits_only_matter_for_reach_and_path",
              all(v[0] == 0.0 for v in irrelevant.values() if v[0] is not None),
              f"P(answer changes | set_only turn) for the 7 other tasks: { {k: v[0] for k, v in irrelevant.items()} }", "info")
        info("sessions", f"{split}_P(change|turn kind)",
             f"reachability {table['reachability']}, shortest_path {table['shortest_path']}, edge_exists {table['edge_exists']}")
        # drift of state by turn index
        drift_summary = {
            turn: {
                "closed_fraction": round(sum(x[0] for x in v) / len(v), 3),
                "mean_degree": round(sum(x[1] for x in v) / len(v), 2),
                "components": round(sum(x[2] for x in v) / len(v), 2),
                "connected": round(sum(x[3] for x in v) / len(v), 3),
                "n": len(v),
            }
            for turn, v in sorted(drift.items())
        }
        report[f"{split}_drift"] = drift_summary
        info("sessions", f"{split}_state_drift_by_turn", f"{drift_summary}")
        final_by_len = {
            L: {"closed_fraction": round(sum(x[0] for x in v) / len(v), 3), "connected": round(sum(x[1] for x in v) / len(v), 3)}
            for L, v in sorted(drift_by_len.items())
        }
        info("sessions", f"{split}_final_state_by_session_length", f"{final_by_len} (static train closed fraction {train_closed:.3f})")
        last_turn = max(drift)
        check("sessions", f"{split}_no_large_state_drift",
              abs(drift_summary[last_turn]["closed_fraction"] - train_closed) < 0.1 and drift_summary[last_turn]["connected"] > 0.9,
              f"turn {last_turn}: closed {drift_summary[last_turn]['closed_fraction']} vs train {train_closed:.3f}, "
              f"connected {drift_summary[last_turn]['connected']}", "warn")
        report[f"{split}_summary"] = {"noop_rate": noop_rate, "net_noop": net_noop, "cancelling": cancelling,
                                      "reverts": revert, "repeated_queries": repeated, "turns": total_turns,
                                      "added_edges": added_total, "added_same_line": added_same_line}
    RESULTS["sessions"] = report


# ------------------------------------------------------------------------------ 8 counterfactual
def section_counterfactual(static_train: list) -> None:
    print("\n[8] Counterfactual training pairs")
    pairs: dict[str, dict[str, Any]] = defaultdict(dict)
    for item in static_train:
        label = item.metadata.get("counterfactual")
        if label:
            pairs[item.tuple_id.rsplit("-", 1)[0]][label] = item
    bad_text = bad_answer = bad_label = same_graph = 0
    kinds: Counter[tuple[str, str]] = Counter()
    endpoint_touch: Counter[str] = Counter()
    per_task = Counter()
    issues: Counter[str] = Counter()
    for pair in pairs.values():
        before, after = pair["before"], pair["after"]
        task = before.query.reasoning_type.value
        per_task[task] += 1
        if before.query.question != after.query.question:
            bad_text += 1
        if before.answer == after.answer:
            bad_answer += 1
        sb, sa = State(before.graph), State(after.graph)
        if solve(sb, before.query) != before.answer or solve(sa, after.query) != after.answer:
            bad_label += 1
        if sb.attrs == sa.attrs and sb.edges == sa.edges:
            same_graph += 1
        status_diff = [i for i in sb.attrs if sb.attrs[i] != sa.attrs[i]]
        edge_diff = sb.edges ^ sa.edges
        kind = "set" if status_diff and not edge_diff else "edge" if edge_diff and not status_diff else "mixed"
        kinds[(task, kind)] += 1
        if status_diff and not edge_diff and task in ("reachability", "shortest_path"):
            endpoint_touch[task] += any(i in (before.query.source, before.query.target) for i in status_diff)
        problems, _ = graph_issues(after.graph, closed_range=(0, 22)) if False else ([], None)
        issues.update(problems)
    n = len(pairs)
    check("counterfactual", "pairs_share_question_text", bad_text == 0, f"{bad_text}/{n} pairs differ in prompt text")
    check("counterfactual", "answers_differ", bad_answer == 0, f"{bad_answer}/{n} pairs have equal answers")
    check("counterfactual", "labels_correct", bad_label == 0, f"{bad_label}/{n} pairs mislabeled by independent solver")
    check("counterfactual", "graphs_differ", same_graph == 0, f"{same_graph}/{n} pairs have identical graphs")
    by_task = {t: {k: kinds[(t, k)] for k in ("set", "edge", "mixed")} for t in sorted(per_task)}
    info("counterfactual", "pairs_by_task_and_edit_kind", f"{n} pairs; {by_task}")
    for task in ("reachability", "shortest_path"):
        sets = by_task.get(task, {}).get("set", 0)
        info("counterfactual", f"{task}_status_flip_pairs", f"{sets}/{per_task[task]} pairs are status-only edits; "
             f"{endpoint_touch[task]} of them flip a queried endpoint")
    RESULTS["counterfactual"] = {"pairs": n, "by_task": by_task}


# ------------------------------------------------------------------------------ 9 shortcuts
def section_shortcuts(rows: dict[str, dict[str, list]], sessions: dict[str, list]) -> None:
    print("\n[9] Structural shortcut ceilings (lookup tables fitted on train, scored on test)")
    models: dict[str, dict[str, Lookup]] = {
        name: {task: Lookup(ladder).fit(rows["train"][task]) for task in STATIC_TASKS}
        for name, ladder in LADDERS.items()
    }
    table: dict[str, dict[str, float]] = {name: {} for name in LADDERS}
    for name in LADDERS:
        for task in STATIC_TASKS:
            test = rows["test"][task]
            table[name][task] = sum(models[name][task].predict(f) == a for f, a in test) / len(test)
    header = f"{'task':36}" + "".join(f"{n:>15}" for n in LADDERS)
    print("  " + header)
    for task in STATIC_TASKS:
        print("  " + f"{task:36}" + "".join(f"{100 * table[n][task]:15.1f}" for n in LADDERS))
    summary = {}
    for name in LADDERS:
        summary[name] = {
            "all_11_tasks": sum(table[name][t] for t in STATIC_TASKS) / len(STATIC_TASKS),
            "eight_tasks": sum(table[name][t] for t in EIGHT_TASKS) / len(EIGHT_TASKS),
        }
        print(f"  {name:14} all-11 {100 * summary[name]['all_11_tasks']:.1f}   8-task {100 * summary[name]['eight_tasks']:.1f}")
    committed = json.loads(Path("documents/experiments/results/audit-20261016/shortcut_audit.json").read_text())
    reference = committed["static"]["shortcut"]["task_mean"]
    info("shortcuts", "static_8_task_ceiling",
         f"lookup {100 * summary['degree_status']['eight_tasks']:.1f}% vs gradient-boosting {100 * reference:.1f}% in shortcut_audit.json; "
         f"best trained model (warm start) 66.5%")
    check("shortcuts", "shortcut_ceiling_reproduced", abs(summary["degree_status"]["eight_tasks"] - reference) < 0.04,
          f"|lookup - committed gradient boosting| = {abs(summary['degree_status']['eight_tasks'] - reference):.3f}", "warn")
    # multi-turn tracker on updated vs stale graph
    mt: dict[str, Any] = {}
    sess = sessions["test"]
    counts = Counter()
    for session in sess:
        state = State(session.initial_graph)
        for turn in session.turns:
            task = turn.query.reasoning_type.value
            if task not in EIGHT_TASKS:
                after, _ = apply_indep(state, turn.gold_edits or (turn.gold_edit,))
                state = after
                continue
            after, _ = apply_indep(state, turn.gold_edits or (turn.gold_edit,))
            model = models["degree_status"][task]
            f_after, f_before = features(after, turn.query), features(state, turn.query)
            changed = turn.gold_answer != turn.stale_answer
            counts["n"] += 1
            counts["stale_answer"] += turn.stale_answer == turn.gold_answer
            counts["tracker_updated"] += model.predict(f_after) == turn.gold_answer
            counts["tracker_stale"] += model.predict(f_before) == turn.gold_answer
            counts["status_only_updated"] += models["status_only"][task].predict(f_after) == turn.gold_answer
            counts["degree_only_updated"] += models["degree_only"][task].predict(f_after) == turn.gold_answer
            counts["changed"] += changed
            counts["tracker_updated_on_changed"] += changed and model.predict(f_after) == turn.gold_answer
            counts["tracker_updated_on_unchanged"] += (not changed) and model.predict(f_after) == turn.gold_answer
            counts["tracker_stale_on_changed"] += changed and model.predict(f_before) == turn.gold_answer
            state = after
    n = counts["n"]
    mt = {k: v / n for k, v in counts.items() if k not in ("n", "changed") and "changed" not in k and "unchanged" not in k}
    mt["on_changed"] = {"updated": counts["tracker_updated_on_changed"] / counts["changed"],
                        "stale": counts["tracker_stale_on_changed"] / counts["changed"]}
    mt["on_unchanged"] = {"updated": counts["tracker_updated_on_unchanged"] / (n - counts["changed"])}
    mt["turns"] = n
    mt["changed_fraction"] = counts["changed"] / n
    info("shortcuts", "multi_turn_8_tasks",
         f"{n} turns: stale answer {100 * mt['stale_answer']:.1f}, tracker on updated graph {100 * mt['tracker_updated']:.1f}, "
         f"tracker on stale graph {100 * mt['tracker_stale']:.1f}, status-only {100 * mt['status_only_updated']:.1f}, "
         f"degree-only {100 * mt['degree_only_updated']:.1f} (committed gradient boosting: 64.6 updated / 45.5 stale)")
    RESULTS["shortcuts"] = {"static_table": table, "static_summary": summary, "multi_turn": mt}


# ------------------------------------------------------------------------------ 10 latent hazards
def section_hazards(static: dict[str, list]) -> None:
    print("\n[10] Latent hazards")
    last_terminal_deg1 = recall_num = recall_den = 0
    graphs = 0
    corr_acc = 0.0
    seen: set[str] = set()
    for item in static["test"]:
        g = item.graph
        if g.graph_id in seen:
            continue
        seen.add(g.graph_id)
        graphs += 1
        state = State(g)
        adj = state.adj()
        order = [n.id for n in g.nodes]
        k = 2 if len(order) < 17 else 3
        last = order[-k:]
        last_terminal_deg1 += all(len(adj[i]) == 1 for i in last)
        deg1 = [i for i in order if len(adj[i]) == 1]
        recall_den += len(deg1)
        recall_num += sum(i in last for i in deg1)
        dist = bfs(adj, order[0])
        xs = list(range(len(order))); ys = [dist[i] for i in order]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = math.sqrt(sum((x - mx) ** 2 for x in xs)); vy = math.sqrt(sum((y - my) ** 2 for y in ys))
        corr_acc += cov / (vx * vy) if vx and vy else 0.0
    check("hazards", "node_order_hides_terminals", last_terminal_deg1 == 0,
          f"in {last_terminal_deg1}/{graphs} test graphs the last {2}-{3} entries of graph.nodes are exactly degree-1 stations; "
          f"they hold {recall_num}/{recall_den} of all degree-1 stations", "warn")
    check("hazards", "node_order_hides_depth", abs(corr_acc / graphs) < 0.2,
          f"mean corr(position in graph.nodes, BFS distance from first node) = {corr_acc / graphs:.2f}", "warn")
    ids_sorted = sum(
        [int(n.id[1:]) for n in item.graph.nodes] == sorted(int(n.id[1:]) for n in item.graph.nodes)
        for item in static["test"][:11 * 40:11]
    )
    info("hazards", "ids_permuted", f"{ids_sorted}/40 sampled test graphs list nodes in id order (ids are permuted, node order is not)")
    first_edges = [
        [(e.source, e.target) for e in item.graph.edges][:3]
        for item in static["test"][:11 * 3:11]
    ]
    info("hazards", "edge_order", f"edges are stored sorted by generation position; example first edges {first_edges[0]}")


# ------------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/tmp/graphmodi_bench")
    parser.add_argument("--out", default="documents/experiments/results/benchmark-audit-20261021")
    parser.add_argument("--degree-balance", action="store_true",
                        help="audit the degree-balanced variant (data.degree_balance: true, metro_v3_bal)")
    args = parser.parse_args()
    work, out = Path(args.work), Path(args.out)
    sys.setrecursionlimit(10000)

    print("Generating (or reusing) metro_v3 at seed 42 ...")
    paths = generate(work, args.degree_balance)
    RESULTS['variant'] = {'degree_balance': args.degree_balance}
    static = {s: load_static_tuples(paths[f"static_{s}"]) for s in ("train", "validation", "test")}
    sessions = {s: load_sessions(paths[f"sessions_{s}"]) for s in ("validation", "test")}
    print(f"loaded: static {{{', '.join(f'{k}: {len(v)}' for k, v in static.items())}}}, "
          f"sessions {{{', '.join(f'{k}: {len(v)}' for k, v in sessions.items())}}}")

    section_reproduction(paths, static, sessions)
    section_labels(static, sessions)
    section_cycle_labels(static)
    profiles = section_structure(static, sessions)
    section_splits(static, sessions)
    section_prompt(static)
    rows = {split: static_rows(items) for split, items in static.items()}
    section_balance(static, rows)
    section_sessions(sessions, profiles["static_train"])
    section_counterfactual(static["train"])
    section_shortcuts(rows, sessions)
    section_hazards(static)

    summary = Counter(c["status"] for c in CHECKS)
    print(f"\nSUMMARY  pass {summary['pass']}  fail {summary['fail']}  warn {summary['warn']}  info {summary['info']}")
    for c in CHECKS:
        if c["status"] in ("fail", "warn"):
            print(f"  [{c['status'].upper()}] {c['section']}/{c['name']}  {c['detail']}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"summary": dict(summary), "checks": CHECKS, "sections": RESULTS}, indent=1, default=str))
    (out / "manifest.json").write_text(json.dumps(
        {"seed": SEED, "generator": "graph_modi.data.v3 (configs/v3_tea.yaml)",
         "files": {name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for name, p in paths.items()},
         "pretraining_graph_sha256": RESULTS["reproduction"]["pretraining_graph_sha256"]}, indent=1))
    print(f"wrote {out}/results.json and manifest.json")


if __name__ == "__main__":
    main()
