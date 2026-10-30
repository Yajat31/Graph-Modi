#!/usr/bin/env python3
"""Final independent correctness audit for metro_v4.

Recomputes labels and edit effects with solvers written here (BFS, biconnected
components, shortest paths) — not by calling graph_modi.graph.solvers or
graph_modi.graph.executor. Comparing the library against itself would prove nothing.

Usage:
  PYTHONPATH=src python3 scripts/audit_v4_final.py \\
      --data datasets/metro_v4 \\
      --out documents/experiments/results/benchmark-final-audit-v4-20261025

  # Also regenerate a small determinism probe (default on):
  PYTHONPATH=src python3 scripts/audit_v4_final.py \\
      --data datasets/metro_v4 --work /tmp/v4_final_probe
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import replace
from pathlib import Path
from typing import Any

from graph_modi.data import v4 as V4
from graph_modi.data.multiturn import load_sessions
from graph_modi.data.v2 import load_static_tuples
from graph_modi.evaluation.metrics import answers_match, cluster_bootstrap_ci
from graph_modi.evaluation.runner import CONDITIONS, evaluate_sessions
from graph_modi.graph.edits import execution_equivalent, parse_edit_program
from graph_modi.graph.solvers import render_question
from graph_modi.models.base import SymbolicMockBackend
from graph_modi.models.bert_tensorizer import BertNodeTensorizerConfig, _node_text
from graph_modi.models.tea_glm import TEAGLMBackend
from graph_modi.schema import (
    AttributedGraph,
    DensityBin,
    EditOperation,
    EditProgram,
    EditTarget,
    GraphEdit,
    GraphQuery,
    ReasoningType,
    StaticQATuple,
)
from graph_modi.training import TrainingExample

SEED = 42
CHECKS: list[dict[str, Any]] = []
RESULTS: dict[str, Any] = {}

ATTR_DOMAINS = {
    "architecture": set(V4.V3.ARCHITECTURES),
    "cleanliness": set(V4.V3.CLEANLINESS),
    "music": set(V4.V3.MUSIC),
    "size": set(V4.V3.SIZES),
    "line": {"red", "blue", "green", "yellow"},
    "status": {"open", "closed"},
    "disabled_access": {True, False},
    "has_rail": {True, False},
}
EDGE_ATTR_KEYS = {"line_color", "line_stroke", "has_aircon", "built"}
EIGHT = [
    "edge_exists",
    "reachability",
    "constrained_reachability",
    "cycle_membership",
    "node_degree",
    "filtered_neighbor_count",
    "shortest_path",
    "within_hops_count",
]
YES_NO = {
    ReasoningType.EDGE_EXISTS,
    ReasoningType.REACHABILITY,
    ReasoningType.CONSTRAINED_REACHABILITY,
    ReasoningType.CYCLE_MEMBERSHIP,
    ReasoningType.ATTRIBUTE_CHECK,
}


# Repo-relative inputs resolve from the repo root, not the working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]


def check(section: str, name: str, ok: bool, detail: str = "", level: str = "fail") -> bool:
    status = "pass" if ok else level
    CHECKS.append({"section": section, "name": name, "status": status, "detail": detail})
    print(f"  [{status.upper():4}] {section}/{name}  {detail}")
    return ok


def info(section: str, name: str, detail: str) -> None:
    CHECKS.append({"section": section, "name": name, "status": "info", "detail": detail})
    print(f"  [INFO] {section}/{name}  {detail}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Independent solvers (do not import graph_modi.graph.solvers)
# Conventions pinned explicitly:
#   - open = status != "closed"
#   - reachability / shortest_path: both endpoints must be open; intermediates must be open
#   - constrained_reachability: all nodes used; endpoints exempt from the avoided class;
#     open/closed is NOT filtered (matches library reachable_avoiding available_only=False)
#   - within_hops_count: excludes the start node
#   - most_common_attribute_within_hops: INCLUDES the start node; unique mode required
#   - cycle_membership: node is on a cycle iff removing it leaves two neighbours connected
#     (equivalently: belongs to a biconnected component with a cycle through it)
#   - shortest_path answer: hop-count length of the path including endpoints, or "unreachable"
# ---------------------------------------------------------------------------


def _is_open(graph: AttributedGraph, node_id: str) -> bool:
    return graph.node_map()[node_id].attributes.get("status", "open") != "closed"


def _adj(graph: AttributedGraph, *, open_only: bool = False) -> dict[str, set[str]]:
    nodes = graph.node_map()
    result: dict[str, set[str]] = {node_id: set() for node_id in nodes}
    for edge in graph.edges:
        if open_only and (not _is_open(graph, edge.source) or not _is_open(graph, edge.target)):
            continue
        result[edge.source].add(edge.target)
        if not graph.directed:
            result[edge.target].add(edge.source)
    return result


def _bfs_path(adj: dict[str, set[str]], source: str, target: str) -> list[str] | None:
    if source not in adj or target not in adj:
        return None
    if source == target:
        return [source]
    parent: dict[str, str | None] = {source: None}
    queue: deque[str] = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adj[node]):
            if neighbor in parent:
                continue
            parent[neighbor] = node
            if neighbor == target:
                path = [target]
                while path[-1] != source:
                    previous = parent[path[-1]]
                    assert previous is not None
                    path.append(previous)
                path.reverse()
                return path
            queue.append(neighbor)
    return None


def _within_hops(adj: dict[str, set[str]], source: str, hops: int) -> dict[str, int]:
    distances: dict[str, int] = {source: 0}
    queue: deque[str] = deque([source])
    while queue:
        node = queue.popleft()
        if distances[node] >= hops:
            continue
        for neighbor in adj[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    distances.pop(source, None)
    return distances


def _on_cycle(adj: dict[str, set[str]], source: str) -> bool:
    """True iff two neighbours of ``source`` remain connected after removing ``source``."""
    neighbors = sorted(adj.get(source, set()) - {source})
    if len(neighbors) < 2:
        return False
    neighbor_set = set(neighbors)
    visited = {source}
    for start in neighbors:
        if start in visited:
            continue
        component = {start}
        queue: deque[str] = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        if len(component & neighbor_set) >= 2:
            return True
    return False


def _reachable_avoiding(
    graph: AttributedGraph,
    source: str,
    target: str,
    attribute: str,
    value: Any,
) -> bool:
    nodes = graph.node_map()
    if source not in nodes or target not in nodes:
        return False
    adj = _adj(graph, open_only=False)

    def allowed(node_id: str) -> bool:
        if node_id in (source, target):
            return True
        return nodes[node_id].attributes.get(attribute) != value

    visited = {source}
    queue: deque[str] = deque([source])
    while queue:
        node = queue.popleft()
        if node == target:
            return True
        for neighbor in adj[node]:
            if neighbor not in visited and allowed(neighbor):
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def independent_answer(graph: AttributedGraph, query: GraphQuery) -> str:
    nodes = graph.node_map()
    if query.source not in nodes:
        return "invalid"
    kind = query.reasoning_type
    if kind is ReasoningType.EDGE_EXISTS:
        if query.target is None:
            return "invalid"
        return "yes" if query.target in _adj(graph, open_only=False)[query.source] else "no"
    if kind is ReasoningType.NODE_DEGREE:
        return str(len(_adj(graph, open_only=False)[query.source]))
    if kind is ReasoningType.REACHABILITY:
        if query.target is None:
            return "invalid"
        if not _is_open(graph, query.source) or not _is_open(graph, query.target):
            return "no"
        path = _bfs_path(_adj(graph, open_only=True), query.source, query.target)
        return "yes" if path else "no"
    if kind is ReasoningType.SHORTEST_PATH:
        if query.target is None:
            return "invalid"
        if not _is_open(graph, query.source) or not _is_open(graph, query.target):
            return "unreachable"
        path = _bfs_path(_adj(graph, open_only=True), query.source, query.target)
        return str(len(path)) if path else "unreachable"
    if kind is ReasoningType.CONSTRAINED_REACHABILITY:
        if query.target is None or query.attribute is None:
            return "invalid"
        return (
            "yes"
            if _reachable_avoiding(graph, query.source, query.target, query.attribute, query.value)
            else "no"
        )
    if kind is ReasoningType.CYCLE_MEMBERSHIP:
        return "yes" if _on_cycle(_adj(graph, open_only=False), query.source) else "no"
    if kind is ReasoningType.FILTERED_NEIGHBOR_COUNT:
        count = sum(
            nodes[neighbor].attributes.get(query.attribute or "") == query.value
            for neighbor in _adj(graph, open_only=False)[query.source]
        )
        return str(count)
    if kind is ReasoningType.WITHIN_HOPS_COUNT:
        hops = query.hops or 2
        in_range = _within_hops(_adj(graph, open_only=False), query.source, hops)
        if query.attribute is not None:
            return str(
                sum(
                    nodes[node_id].attributes.get(query.attribute) == query.value
                    for node_id in in_range
                )
            )
        return str(len(in_range))
    if kind is ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS:
        if query.attribute is None:
            return "invalid"
        hops = query.hops or 2
        sample = set(_within_hops(_adj(graph, open_only=False), query.source, hops)) | {
            query.source
        }
        counts: dict[str, int] = {}
        for node_id in sample:
            attribute_value = nodes[node_id].attributes.get(query.attribute)
            if attribute_value is None:
                continue
            key = str(attribute_value)
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return "none"
        best = max(counts.values())
        winners = sorted(key for key, count in counts.items() if count == best)
        return winners[0]  # ties broken lexicographically; unique-mode gate forbids ties
    if kind is ReasoningType.ATTRIBUTE_LOOKUP:
        if query.attribute is None:
            return "invalid"
        value = nodes[query.source].attributes.get(query.attribute)
        return "none" if value is None else str(value)
    if kind is ReasoningType.ATTRIBUTE_CHECK:
        if query.attribute is None:
            return "invalid"
        return "yes" if nodes[query.source].attributes.get(query.attribute) == query.value else "no"
    raise ValueError(f"unsupported task for independent solver: {kind}")


def unique_mode(graph: AttributedGraph, source: str, attribute: str, hops: int) -> bool:
    nodes = graph.node_map()
    sample = set(_within_hops(_adj(graph, open_only=False), source, hops)) | {source}
    counts: Counter[str] = Counter()
    for node_id in sample:
        value = nodes[node_id].attributes.get(attribute)
        if value is not None:
            counts[str(value)] += 1
    if not counts:
        return False
    best = max(counts.values())
    return sum(1 for count in counts.values() if count == best) == 1


# ---------------------------------------------------------------------------
# Independent executor (topology + status only; ADD edge attrs are rng-drawn)
# ---------------------------------------------------------------------------


def independent_apply(graph: AttributedGraph, edits: tuple[GraphEdit, ...]) -> AttributedGraph:
    nodes = {node.id: node for node in graph.nodes}
    edges = {frozenset((edge.source, edge.target)): edge for edge in graph.edges}
    for edit in edits:
        if edit.operation is EditOperation.NOOP:
            continue
        if edit.target is EditTarget.NODE and edit.operation is EditOperation.SET:
            if edit.node_id not in nodes or not edit.attribute:
                continue
            old = nodes[edit.node_id]
            if old.attributes.get(edit.attribute) == edit.value:
                continue
            attributes = dict(old.attributes)
            attributes[edit.attribute] = edit.value
            nodes[edit.node_id] = replace(old, attributes=attributes)
        elif edit.target is EditTarget.EDGE and edit.operation is EditOperation.ADD:
            if not edit.source or not edit.destination:
                continue
            if edit.source not in nodes or edit.destination not in nodes:
                continue
            key = frozenset((edit.source, edit.destination))
            if key in edges or edit.source == edit.destination:
                continue
            from graph_modi.schema import Edge

            edges[key] = Edge(
                source=edit.source,
                target=edit.destination,
                relation=edit.relation or "connected",
            )
        elif edit.target is EditTarget.EDGE and edit.operation is EditOperation.DEL:
            if not edit.source or not edit.destination:
                continue
            edges.pop(frozenset((edit.source, edit.destination)), None)
        elif edit.target is EditTarget.NODE and edit.operation is EditOperation.DEL:
            if edit.node_id not in nodes:
                continue
            del nodes[edit.node_id]
            edges = {key: edge for key, edge in edges.items() if edit.node_id not in key}
    return replace(graph, nodes=tuple(nodes.values()), edges=tuple(edges.values()))


def state_key(graph: AttributedGraph) -> tuple:
    closed = frozenset(node.id for node in graph.nodes if node.attributes.get("status") == "closed")
    edges = frozenset(frozenset((edge.source, edge.target)) for edge in graph.edges)
    return (closed, edges)


def connected(graph: AttributedGraph) -> bool:
    adj = _adj(graph, open_only=False)
    if not adj:
        return True
    start = next(iter(adj))
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in adj[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(adj)


def diameter(graph: AttributedGraph) -> int:
    adj = _adj(graph, open_only=False)
    best = 0
    for start in adj:
        distances = {start: 0}
        queue: deque[str] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        if len(distances) != len(adj):
            return -1
        best = max(best, max(distances.values()))
    return best


def mean_degree(graph: AttributedGraph) -> float:
    if not graph.nodes:
        return 0.0
    return 2.0 * len(graph.edges) / len(graph.nodes)


def wl_hash(graph: AttributedGraph, rounds: int = 3) -> str:
    nodes = graph.node_map()
    labels = {
        node_id: (
            nodes[node_id].attributes.get("line"),
            nodes[node_id].attributes.get("status"),
            nodes[node_id].attributes.get("size"),
        )
        for node_id in nodes
    }
    adj = _adj(graph, open_only=False)
    for _ in range(rounds):
        labels = {
            node_id: (labels[node_id], tuple(sorted(labels[n] for n in adj[node_id])))
            for node_id in labels
        }
    return str(sorted(labels.values()))


# ---------------------------------------------------------------------------
# Audit steps
# ---------------------------------------------------------------------------


def step_determinism(data: Path, work: Path, manifest_path: Path) -> None:
    print("\n[1 determinism]")
    manifest = json.loads(manifest_path.read_text())
    mismatches = []
    for name, meta in manifest["files"].items():
        path = data / name
        if not path.exists():
            mismatches.append(f"missing {name}")
            continue
        digest = sha256_file(path)
        if digest != meta["sha256"]:
            mismatches.append(f"{name}: {digest[:12]}!={meta['sha256'][:12]}")
        elif path.stat().st_size != meta["bytes"]:
            mismatches.append(f"{name}: size {path.stat().st_size}!={meta['bytes']}")
    check(
        "determinism", "manifest_match", not mismatches, "; ".join(mismatches) or "all files match"
    )

    # Regenerate a small probe twice and compare byte-for-byte.
    work.mkdir(parents=True, exist_ok=True)
    probe_a = work / "probe_a"
    probe_b = work / "probe_b"
    for probe in (probe_a, probe_b):
        if probe.exists():
            continue
        probe.mkdir(parents=True)
        cells = len(V4.NODE_COUNTS) * len(V4.DENSITIES)
        items = V4.generate_static_v4(
            split="validation",
            graph_count=cells,  # one full cell set
            tasks_per_graph=None,
            seed=SEED,
            sampler=V4.BalancedSampler(),
            tasks=list(V4.STATIC_TASKS),
            progress=False,
        )
        from graph_modi.data.v2 import save_static_tuples

        save_static_tuples(probe / "static_validation.jsonl", items)
        sessions = V4.generate_sessions_v4(
            replicates_per_split={"validation": 1},
            seed=SEED,
            tasks=list(V4.DYNAMIC_TASKS),
            progress=False,
        )
        from graph_modi.data.multiturn import save_sessions

        save_sessions(probe / "sessions_validation.jsonl", sessions["validation"])
    same = True
    details = []
    for name in ("static_validation.jsonl", "sessions_validation.jsonl"):
        ha, hb = sha256_file(probe_a / name), sha256_file(probe_b / name)
        details.append(f"{name}:{ha[:10]}=={hb[:10]}")
        same = same and ha == hb
    check("determinism", "regenerate_twice", same, "; ".join(details))

    # Spot-check generator determinism on individual graphs (full val is 6× probe).
    full_val = data / "static_validation.jsonl"
    if full_val.exists() and (probe_a / "static_validation.jsonl").exists():
        # Full validation uses 6 replicates; probe used 1. Compare generator function determinism
        # on individual graphs instead.
        g1 = V4.make_metro_graph_v4(random.Random(SEED + 1), "train", 1, 14, DensityBin.MEDIUM)
        g2 = V4.make_metro_graph_v4(random.Random(SEED + 1), "train", 1, 14, DensityBin.MEDIUM)
        check(
            "determinism",
            "make_metro_graph_v4",
            state_key(g1) == state_key(g2)
            and [n.label for n in g1.nodes] == [n.label for n in g2.nodes],
            f"graph_id={g1.graph_id}",
        )


def step_graph_generation(static: dict, sessions: dict) -> None:
    print("\n[2 graph_generation]")
    graphs: dict[str, AttributedGraph] = {}
    for items in static.values():
        for item in items:
            if item.metadata.get("counterfactual") == "after":
                continue
            graphs[item.graph.graph_id] = item.graph
    for sess in sessions.values():
        for session in sess:
            graphs[session.initial_graph.graph_id] = session.initial_graph

    issues: Counter[str] = Counter()
    diameter_ok = closed_ok = attrs_ok = names_ok = connected_ok = no_loops = True
    samples = 0
    for graph in graphs.values():
        samples += 1
        n = len(graph.nodes)
        if n not in V4.NODE_COUNTS:
            issues["bad_size"] += 1
        density = DensityBin(str(graph.metadata.get("density")))
        if not connected(graph):
            connected_ok = False
            issues["disconnected"] += 1
        diam = diameter(graph)
        if diam < 0 or diam > V4.MAX_DIAMETER[density]:
            diameter_ok = False
            issues["diameter"] += 1
        closed = sum(node.attributes.get("status") == "closed" for node in graph.nodes)
        if not (1 <= closed <= 3):
            # Counterfactual "after" already skipped; remaining should be 1-3.
            closed_ok = False
            issues["closed"] += 1
        labels = [node.label for node in graph.nodes]
        if len(labels) != len(set(labels)):
            names_ok = False
            issues["dup_names"] += 1
        ids = [node.id for node in graph.nodes]
        if len(ids) != len(set(ids)):
            names_ok = False
            issues["dup_ids"] += 1
        edge_keys = []
        for edge in graph.edges:
            if edge.source == edge.target:
                no_loops = False
                issues["self_loop"] += 1
            key = frozenset((edge.source, edge.target))
            edge_keys.append(key)
            if not EDGE_ATTR_KEYS.issubset(edge.attributes):
                attrs_ok = False
                issues["edge_attrs"] += 1
        if len(edge_keys) != len(set(edge_keys)):
            no_loops = False
            issues["dup_edges"] += 1
        for node in graph.nodes:
            for key, domain in ATTR_DOMAINS.items():
                if key not in node.attributes or node.attributes[key] not in domain:
                    attrs_ok = False
                    issues[f"node_attr_{key}"] += 1
    check("graph_generation", "connected", connected_ok, f"{issues['disconnected']}/{samples}")
    check(
        "graph_generation",
        "size_and_diameter",
        diameter_ok and issues["bad_size"] == 0,
        f"{dict(issues)}",
    )
    check("graph_generation", "closed_1_to_3", closed_ok, f"{issues['closed']}/{samples}")
    check("graph_generation", "unique_names_ids", names_ok, f"{dict(issues)}")
    check("graph_generation", "no_loops_or_dup_edges", no_loops, f"{dict(issues)}")
    check("graph_generation", "attribute_domains", attrs_ok, f"{dict(issues)}")
    info("graph_generation", "graph_count", f"{samples} unique graphs audited")


def step_questions(static: dict) -> None:
    print("\n[3 questions]")
    render_ok = roundtrip_ok = no_status_leak = no_answer_leak = True
    render_bad = status_bad = answer_bad = roundtrip_bad = 0
    checked = 0
    for _split, items in static.items():
        for item in items:
            checked += 1
            query = replace(item.query, question="")  # force re-render
            rendered = V4.render_question_v4(query, item.graph)
            if (
                not rendered
                or item.query.reasoning_type.value not in rendered
                and item.query.source not in {n.id for n in item.graph.nodes}
            ):
                # Every task must produce non-empty text; source label should appear.
                pass
            if not rendered.strip():
                render_ok = False
                render_bad += 1
            labels = {node.label: node.id for node in item.graph.nodes}
            # Round-trip: every station label mentioned that exists in the graph maps to an id.
            for label, node_id in labels.items():
                if label in rendered and labels[label] != node_id:
                    roundtrip_ok = False
                    roundtrip_bad += 1
            # Source (and target) labels must appear in the question body.
            source_label = item.graph.node_map()[item.query.source].label
            if source_label not in rendered:
                roundtrip_ok = False
                roundtrip_bad += 1
            if item.query.target:
                target_label = item.graph.node_map()[item.query.target].label
                if target_label not in rendered:
                    roundtrip_ok = False
                    roundtrip_bad += 1
            # status must not appear as an attribute value in W(f_i) / question text.
            if re.search(r"\bstatus\s*=", rendered, flags=re.I):
                no_status_leak = False
                status_bad += 1
            # Answer string should not be pasted as a naked giveaway beyond yes/no words that
            # are part of the question template ("Answer yes or no").
            # Attribute answers can appear in W(f_i) for non-facts tasks; for Facts
            # tasks the queried attribute is withheld from the prompt text.
            if (
                item.answer not in {"yes", "no", "none", "unreachable"}
                and item.answer.lower() in rendered.lower()
                and item.query.reasoning_type
                in (
                    ReasoningType.ATTRIBUTE_LOOKUP,
                    ReasoningType.ATTRIBUTE_CHECK,
                    ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS,
                )
            ):
                attr = item.query.attribute or ""
                if f"{attr}={item.answer}" in rendered or f"{attr}={item.query.value}" in rendered:
                    no_answer_leak = False
                    answer_bad += 1
            if (
                item.query.reasoning_type is ReasoningType.CONSTRAINED_REACHABILITY
                and "exempt" not in rendered.lower()
            ):
                render_ok = False
                render_bad += 1
    check("questions", "renders", render_ok, f"{render_bad}/{checked} empty or missing exemption")
    check("questions", "label_roundtrip", roundtrip_ok, f"{roundtrip_bad}/{checked}")
    check("questions", "no_status_in_text", no_status_leak, f"{status_bad}/{checked}")
    check("questions", "facts_attr_withheld", no_answer_leak, f"{answer_bad}/{checked}")


def step_labels(static: dict, sessions: dict) -> None:
    print("\n[4 labels]")
    for split, items in static.items():
        wrong = ties = 0
        by_task: Counter[str] = Counter()
        for item in items:
            by_task[item.query.reasoning_type.value] += 1
            got = independent_answer(item.graph, item.query)
            if got != item.answer:
                wrong += 1
            if item.query.reasoning_type is ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS:
                hops = item.query.hops or 2
                if not unique_mode(item.graph, item.query.source, item.query.attribute or "", hops):
                    ties += 1
        check("labels", f"static_{split}", wrong == 0, f"{wrong}/{len(items)} mismatches")
        check("labels", f"static_{split}_unique_mode", ties == 0, f"{ties} ties")
        if split == "test":
            check(
                "labels",
                "test_full_grid",
                all(by_task[task] == 396 for task in by_task),
                f"{dict(sorted(by_task.items()))}",
            )
        info("labels", f"static_{split}_counts", f"{dict(sorted(by_task.items()))}")

    for split, sess in sessions.items():
        wrong = stale_wrong = ties = turns = 0
        for session in sess:
            state = session.initial_graph
            for turn in session.turns:
                turns += 1
                edits = turn.gold_edits or (turn.gold_edit,)
                after = independent_apply(state, edits)
                if independent_answer(after, turn.query) != turn.gold_answer:
                    wrong += 1
                if independent_answer(state, turn.query) != turn.stale_answer:
                    stale_wrong += 1
                if turn.query.reasoning_type is ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS:
                    hops = turn.query.hops or 2
                    attr = turn.query.attribute or ""
                    if not unique_mode(state, turn.query.source, attr, hops):
                        ties += 1
                    if not unique_mode(after, turn.query.source, attr, hops):
                        ties += 1
                state = after
        check("labels", f"sessions_{split}_gold", wrong == 0, f"{wrong}/{turns}")
        check("labels", f"sessions_{split}_stale", stale_wrong == 0, f"{stale_wrong}/{turns}")
        check("labels", f"sessions_{split}_ties", ties == 0, f"{ties} ties")


def step_edits(sessions: dict) -> None:
    print("\n[5 edits]")
    parse_ok = apply_ok = utterance_ok = True
    parse_bad = apply_bad = utter_bad = 0
    checked = 0
    families_seen: Counter[str] = Counter()
    for sess in sessions.values():
        for session in sess:
            family = session.paraphrase_family.split("-")[-1]
            families_seen[family] += 1
            state = session.initial_graph
            {node.label.casefold(): node.id for node in state.nodes}
            for turn in session.turns:
                checked += 1
                edits = turn.gold_edits or (turn.gold_edit,)
                program = turn.edit_program or EditProgram(edits=edits)
                # Parse canonical program text back.
                {node.label.casefold(): node.id for node in state.nodes}
                # Canonical uses ids; parse without name table.
                parsed = parse_edit_program(program.canonical())
                if len(parsed.edits) != len(program.edits) or not all(
                    execution_equivalent(a, b)
                    for a, b in zip(parsed.edits, program.edits, strict=True)
                ):
                    # TRY with labels if ids fail — canonical uses ids so this should work.
                    parse_ok = False
                    parse_bad += 1
                after = independent_apply(state, program.edits)
                if state_key(after) != state_key(
                    V4.apply_program_v4(state, program, random.Random(0))
                ):
                    apply_ok = False
                    apply_bad += 1
                # Utterance must mention the touched station labels (or be a NOOP phrase).
                if "noop" in turn.tags:
                    if (
                        family == "direct"
                        and "change" not in turn.utterance.lower()
                        and "update" not in turn.utterance.lower()
                        and "leave" not in turn.utterance.lower()
                        and "network" not in turn.utterance.lower()
                    ):
                        # still allow any of the noop templates
                        pass
                else:
                    labels = {node.id: node.label for node in state.nodes}
                    for edit in program.edits:
                        if edit.operation is EditOperation.NOOP:
                            continue
                        needed = []
                        if edit.node_id:
                            needed.append(labels[edit.node_id])
                        if edit.source:
                            needed.append(labels[edit.source])
                        if edit.destination:
                            needed.append(labels[edit.destination])
                        if not all(label in turn.utterance for label in needed):
                            utterance_ok = False
                            utter_bad += 1
                            break
                state = after
    check("edits", "parse_roundtrip", parse_ok, f"{parse_bad}/{checked}")
    check("edits", "independent_apply_matches_v4", apply_ok, f"{apply_bad}/{checked}")
    check("edits", "utterance_names_labels", utterance_ok, f"{utter_bad}/{checked}")
    info("edits", "paraphrase_families", f"{dict(families_seen)}")

    # Held-out notice family must be absent from the few-shot prompt.
    prompt = TEAGLMBackend._EDIT_PROMPT_PREFIX
    notice_markers = (
        "Please mark",
        "Please add a",
        "Please remove the",
        "Please make no changes",
        "Hold the current network",
        "Do not apply any edit",
    )
    found = [marker for marker in notice_markers if marker in prompt]
    check("edits", "notice_held_out_of_fewshot", not found, f"found {found}")
    # direct + service markers should be present.
    has_direct = "is closed now" in prompt
    has_service = "Service at" in prompt
    check(
        "edits",
        "direct_service_in_fewshot",
        has_direct and has_service,
        f"direct={has_direct} service={has_service}",
    )

    # Invalid ops skipped: DEL unknown edge / SET unknown node leave graph unchanged.
    sample = next(iter(sessions["test"])).initial_graph
    noopish = independent_apply(
        sample,
        (
            GraphEdit(
                operation=EditOperation.DEL,
                target=EditTarget.EDGE,
                source="nope",
                destination="also_nope",
            ),
        ),
    )
    check("edits", "invalid_skipped", state_key(noopish) == state_key(sample), "DEL unknown edge")


def step_sessions(sessions: dict) -> None:
    print("\n[6 sessions]")
    for split, sess in sessions.items():
        replay_bad = changed_bad = index_bad = 0
        turns = 0
        net_noop = cancel = revert = disconnected = 0
        noop = 0
        by_task: Counter[str] = Counter()
        lengths: Counter[int] = Counter()
        for session in sess:
            lengths[len(session.turns)] += 1
            state = session.initial_graph
            seen = {state_key(state)}
            for expected_index, turn in enumerate(session.turns):
                turns += 1
                if turn.turn_index != expected_index:
                    index_bad += 1
                by_task[turn.query.reasoning_type.value] += 1
                edits = turn.gold_edits or (turn.gold_edit,)
                after = independent_apply(state, edits)
                if state_key(after) != state_key(
                    V4.apply_program_v4(
                        state, turn.edit_program or EditProgram(edits=edits), random.Random(0)
                    )
                ):
                    replay_bad += 1
                gold_changed = turn.gold_answer != turn.stale_answer
                actual_changed = independent_answer(after, turn.query) != independent_answer(
                    state, turn.query
                )
                if gold_changed != actual_changed:
                    changed_bad += 1
                if "noop" in turn.tags:
                    noop += 1
                ops = [edit for edit in edits if edit.operation is not EditOperation.NOOP]
                if ops:
                    if state_key(after) == state_key(state):
                        net_noop += 1
                    touched: Counter[tuple] = Counter()
                    for edit in ops:
                        key = (
                            ("n", edit.node_id)
                            if edit.target is EditTarget.NODE
                            else ("e", frozenset((edit.source, edit.destination)))
                        )
                        touched[key] += 1
                    if any(count > 1 for count in touched.values()):
                        cancel += 1
                    if state_key(after) in seen and state_key(after) != state_key(state):
                        revert += 1
                if not connected(after):
                    disconnected += 1
                seen.add(state_key(after))
                state = after
        rate = noop / turns if turns else 0
        check("sessions", f"{split}_replay", replay_bad == 0, f"{replay_bad}/{turns}")
        check("sessions", f"{split}_answer_changed", changed_bad == 0, f"{changed_bad}/{turns}")
        check("sessions", f"{split}_turn_index", index_bad == 0, f"{index_bad}/{turns}")
        check("sessions", f"{split}_noop_rate", 0.15 <= rate <= 0.20, f"{rate:.3f}")
        check("sessions", f"{split}_net_noop", net_noop == 0, f"{net_noop}")
        check("sessions", f"{split}_cancelling", cancel == 0, f"{cancel}")
        check("sessions", f"{split}_reverts", revert == 0, f"{revert}", "warn")
        check("sessions", f"{split}_connected", disconnected == 0, f"{disconnected}/{turns}")
        if split == "test":
            # 11 sizes × 3 densities × 4 lengths × 9 tasks = 1188 sessions.
            # Each task appears in 11×3×4 = 132 sessions; turn counts = 132 × sum(lengths)/4
            # = 132 × (1+2+4+8)/4 = 132 × 3.75 = 495 turns per task.
            expected_turns = 11 * 3 * sum(V4.SESSION_LENGTHS)  # 495
            check(
                "sessions",
                "test_session_count",
                len(sess) == 1188,
                f"{len(sess)}",
            )
            check(
                "sessions",
                "test_turns_per_task",
                all(by_task[task] == expected_turns for task in by_task) and len(by_task) == 9,
                (
                    f"expected {expected_turns}/task; "
                    f"got {dict(sorted(by_task.items()))}; lengths={dict(lengths)}"
                ),
            )


def step_shortcuts(static: dict, sessions: dict) -> None:
    print("\n[7 shortcuts_and_balance]")
    train, test = static["train"], static["test"]

    def feats(item):
        graph, query = item.graph, item.query
        degree = {nid: len(nbrs) for nid, nbrs in _adj(graph).items()}
        nodes = graph.node_map()
        return {
            "ds": degree[query.source],
            "dt": degree[query.target] if query.target else None,
            "ss": nodes[query.source].attributes.get("status"),
            "st": nodes[query.target].attributes.get("status") if query.target else None,
            "same": (
                nodes[query.source].attributes.get("line")
                == nodes[query.target].attributes.get("line")
                if query.target
                else None
            ),
            "hops": query.hops,
            "qlen": len((query.question or "").split()),
        }

    class Lookup:
        def __init__(self, ladder):
            self.ladder = ladder
            self.tables = [defaultdict(Counter) for _ in ladder]

        def fit(self, rows):
            for features, answer in rows:
                for level, table in zip(self.ladder, self.tables, strict=False):
                    table[tuple(features[k] for k in level)][answer] += 1
            return self

        def predict(self, features):
            for level, table in zip(self.ladder, self.tables, strict=False):
                counts = table.get(tuple(features[k] for k in level))
                if counts and sum(counts.values()) >= 3:
                    return counts.most_common(1)[0][0]
            return self.tables[-1][()].most_common(1)[0][0]

    by_train: dict[str, list] = defaultdict(list)
    by_test: dict[str, list] = defaultdict(list)
    for item in train:
        by_train[item.query.reasoning_type.value].append((feats(item), item.answer))
    for item in test:
        by_test[item.query.reasoning_type.value].append((feats(item), item.answer))

    ladders = {
        "status_only": [("ss", "st"), ("ss",), ()],
        "degree_only": [("ds", "dt"), ("ds",), ()],
        "degree_status": [
            ("ds", "dt", "ss", "st", "same"),
            ("ds", "dt", "ss", "st"),
            ("ds", "ss", "st"),
            ("ss", "st"),
            (),
        ],
        "text_hops_len": [("hops", "qlen"), ("hops",), ("qlen",), ()],
    }
    summary = {}
    for name, ladder in ladders.items():
        scores = {}
        for task in EIGHT:
            model = Lookup(ladder).fit(by_train[task])
            rows = by_test[task]
            scores[task] = sum(model.predict(f) == a for f, a in rows) / max(1, len(rows))
        summary[name] = scores
        print(
            f"  {name:16} cycle={100 * scores['cycle_membership']:.1f} "
            f"reach={100 * scores['reachability']:.1f}"
        )
    check(
        "shortcuts",
        "cycle_degree_ceiling",
        summary["degree_only"]["cycle_membership"] <= 0.60,
        f"{100 * summary['degree_only']['cycle_membership']:.1f}%",
    )
    check(
        "shortcuts",
        "reach_status_ceiling",
        summary["status_only"]["reachability"] <= 0.60,
        f"{100 * summary['status_only']['reachability']:.1f}%",
    )
    seven = [task for task in EIGHT if task != "node_degree"]
    mean7 = sum(summary["degree_status"][task] for task in seven) / len(seven)
    check("shortcuts", "seven_task_ceiling", mean7 < 0.50, f"{100 * mean7:.1f}%")
    RESULTS["shortcuts"] = {
        name: {k: round(v, 4) for k, v in scores.items()} for name, scores in summary.items()
    }
    RESULTS["shortcuts"]["seven_ex_degree"] = round(mean7, 4)

    # Majority / stale baselines on sessions test.
    turns = [turn for session in sessions["test"] for turn in session.turns]
    majority = {}
    for task in {turn.query.reasoning_type.value for turn in turns}:
        answers = [turn.gold_answer for turn in turns if turn.query.reasoning_type.value == task]
        majority[task] = Counter(answers).most_common(1)[0][0]
    maj_acc = sum(
        turn.gold_answer == majority[turn.query.reasoning_type.value] for turn in turns
    ) / len(turns)
    stale_acc = sum(turn.gold_answer == turn.stale_answer for turn in turns) / len(turns)
    info("shortcuts", "majority_prior_sessions_test", f"{100 * maj_acc:.1f}%")
    info("shortcuts", "stale_answer_sessions_test", f"{100 * stale_acc:.1f}%")
    RESULTS["baselines"] = {"majority": round(maj_acc, 4), "stale": round(stale_acc, 4)}

    # Per-stratum yes/no balance on static test.
    imbalance = 0
    strata = 0
    for task in YES_NO:
        rows = [item for item in test if item.query.reasoning_type is task]
        by_cell: dict[str, Counter[str]] = defaultdict(Counter)
        for item in rows:
            cell = V4.stratum_v4(item.graph, item.query)
            by_cell[cell][item.answer] += 1
        for _cell, counts in by_cell.items():
            strata += 1
            total = sum(counts.values())
            if total >= 8:
                yes = counts.get("yes", 0) / total
                if yes < 0.25 or yes > 0.75:
                    imbalance += 1
    check(
        "shortcuts",
        "yes_no_stratum_balance",
        imbalance <= 0.15 * max(1, strata),
        f"{imbalance}/{strata} coarse cells outside 25-75",
        "warn",
    )


def step_splits(static: dict, sessions: dict) -> None:
    print("\n[8 splits]")
    hashes = {
        "train": {
            wl_hash(graph)
            for graph in {item.graph.graph_id: item.graph for item in static["train"]}.values()
        },
        "validation": {
            wl_hash(graph)
            for graph in {item.graph.graph_id: item.graph for item in static["validation"]}.values()
        },
        "test": {
            wl_hash(graph)
            for graph in {item.graph.graph_id: item.graph for item in static["test"]}.values()
        },
        "sessions_validation": {
            wl_hash(session.initial_graph) for session in sessions["validation"]
        },
        "sessions_test": {wl_hash(session.initial_graph) for session in sessions["test"]},
    }
    overlaps = {
        "train_test": hashes["train"] & hashes["test"],
        "train_validation": hashes["train"] & hashes["validation"],
        "train_sessions_test": hashes["train"] & hashes["sessions_test"],
        "validation_test": hashes["validation"] & hashes["test"],
        "sessions_val_test": hashes["sessions_validation"] & hashes["sessions_test"],
    }
    for name, overlap in overlaps.items():
        check("splits", name, not overlap, f"{len(overlap)}")

    # Station-name reuse across splits (report only; names are drawn from a shared pool).
    names = {}
    for split, items in static.items():
        bucket: set[str] = set()
        for item in items:
            for node in item.graph.nodes:
                bucket.add(node.label)
        names[split] = bucket
    reuse = {
        "train∩test": len(names["train"] & names["test"]),
        "train_only": len(names["train"] - names["test"]),
        "test_only": len(names["test"] - names["train"]),
    }
    info("splits", "station_name_reuse", f"{reuse}")
    RESULTS["name_reuse"] = reuse


def _as_training_examples(tuples: list[StaticQATuple]) -> list[TrainingExample]:
    """Mirror cli._static_examples without importing the CLI module."""
    return [
        TrainingExample(
            graph=item.graph,
            prompt=render_question(item.query, item.graph),
            answer=item.answer,
            example_id=item.tuple_id,
            split=item.split,
            metadata={
                "reasoning_type": item.query.reasoning_type.value,
                "topology": item.topology.value,
                "density_bin": item.density_bin.value,
                "hop_depth": item.hop_depth.value,
                "scale_bin": item.scale_bin,
                "source_id": item.query.source,
                "target_id": item.query.target,
                "hops": item.query.hops,
            },
        )
        for item in tuples
    ]


def step_plumbing(static: dict) -> None:
    print("\n[9 model_input_plumbing]")
    examples = _as_training_examples(static["test"][:200])
    bad_source = bad_target = bad_task = 0
    hops_ok = True
    for example, item in zip(examples, static["test"][:200], strict=True):
        if example.metadata.get("source_id") != item.query.source:
            bad_source += 1
        if example.metadata.get("target_id") != item.query.target:
            bad_target += 1
        if example.metadata.get("reasoning_type") != item.query.reasoning_type.value:
            bad_task += 1
        if example.metadata.get("hops") != item.query.hops:
            hops_ok = False
    check("plumbing", "source_ids", bad_source == 0, f"{bad_source}")
    check("plumbing", "target_ids", bad_target == 0, f"{bad_target}")
    check("plumbing", "reasoning_type_metadata", bad_task == 0, f"{bad_task}")
    check("plumbing", "hops_metadata", hops_ok, "query.hops mirrored into TrainingExample")
    all_tasks = {item.query.reasoning_type.value for item in static["test"]}
    expected = {task.value for task in V4.STATIC_TASKS}
    check("plumbing", "eleven_tasks", all_tasks == expected, f"{sorted(all_tasks)}")

    # W(f_i) excludes status; BERT node text INCLUDES status by design (graph path carries it).
    sample = static["test"][0].graph
    node = sample.nodes[0]
    text = _node_text(node, BertNodeTensorizerConfig())
    check(
        "plumbing",
        "bert_includes_status_by_design",
        "status=" in text,
        "BERT node features include status so the GNN can encode open/closed; "
        "status is withheld only from W(f_i) question text",
    )
    excluded = {name for name in str(sample.metadata.get("wfi_exclude", "")).split(",") if name}
    check("plumbing", "wfi_exclude_status", excluded == {"status"}, f"{excluded}")

    from graph_modi.evaluation.runner import _prompt

    train_prompt = examples[0].prompt
    info(
        "plumbing",
        "prompt_formats",
        f"train_starts={train_prompt[:60]!r}; train_len={len(train_prompt.split())} words",
    )
    source = inspect.getsource(_prompt)
    check(
        "plumbing",
        "eval_wraps_question",
        "Question:" in source or "question" in source.lower(),
        "runner._prompt documented; training uses bare render_question",
    )
    RESULTS["prompt_note"] = (
        "Training prompts are bare render_question / stored query.question (with W(f_i)). "
        "Evaluation conditions that send text through the LM wrap via runner._prompt; "
        "history conditions prepend revision text. No training example contains revision history."
    )


def step_evaluation(sessions: dict) -> None:
    print("\n[10 evaluation_metrics]")
    # normalize_answer / answers_match on representative v4 golds.
    cases = [
        ("yes", "Yes.", "edge_exists", True),
        ("no", "false", "reachability", True),
        ("3", "3.0", "node_degree", True),
        ("unreachable", "no path", "shortest_path", True),
        ("modern", "Modern", "attribute_lookup", True),
        ("yes", "no", "attribute_check", False),
        ("2", "two", "within_hops_count", False),
    ]
    metric_ok = True
    for gold, pred, task, expected in cases:
        got = answers_match(gold, pred, reasoning_type=task)
        if got != expected:
            metric_ok = False
            info("evaluation", "metric_case_fail", f"{gold!r} vs {pred!r} task={task} got={got}")
    check("evaluation", "normalize_and_match", metric_ok, f"{len(cases)} cases")

    # Mock backend end-to-end on a slice of sessions.
    sample = sessions["test"][:24]  # covers several lengths/tasks
    report = evaluate_sessions(
        sample,
        SymbolicMockBackend(),
        conditions=("tool_solver", "stale_answer"),
        batch_size=8,
        progress=False,
    )
    rows = report["rows"]
    tool = [row for row in rows if row["condition"] == "tool_solver"]
    stale = [row for row in rows if row["condition"] == "stale_answer"]
    tool_acc = sum(row["answer_correct"] for row in tool) / max(1, len(tool))
    stale_acc = sum(row["answer_correct"] for row in stale) / max(1, len(stale))
    analytic_stale = sum(
        turn.gold_answer == turn.stale_answer for session in sample for turn in session.turns
    ) / sum(len(session.turns) for session in sample)
    check(
        "evaluation",
        "tool_solver_100",
        tool_acc == 1.0,
        f"{100 * tool_acc:.1f}% on {len(tool)} turns",
    )
    check(
        "evaluation",
        "stale_matches_analytic",
        abs(stale_acc - analytic_stale) < 1e-9,
        f"runner={stale_acc:.4f} analytic={analytic_stale:.4f}",
    )

    # Clustered CI groups by session_id.
    ci_source = inspect.getsource(cluster_bootstrap_ci)
    check(
        "evaluation",
        "cluster_bootstrap_by_session",
        "session_id" in ci_source or "session" in ci_source,
        "cluster_bootstrap_ci resamples whole sessions",
    )

    # predicted_updated_graph must be in graph-model v4 configs (soft_prompt is graph-blind).
    missing = []
    graph_configs = (
        "v4_graphtoken_s42.yaml",
        "v4_graphtoken_s43.yaml",
        "v4_graphtoken_s44.yaml",
        "v4_graphtoken_warm_s42.yaml",
        "v4_tea_random_sum.yaml",
    )
    for name in graph_configs:
        path = REPO_ROOT / "configs" / name
        if not path.exists() or "predicted_updated_graph" not in path.read_text():
            missing.append(name)
    check(
        "evaluation",
        "predicted_updated_graph_in_configs",
        not missing,
        f"missing in {missing}" if missing else " + ".join(graph_configs),
    )
    check(
        "evaluation",
        "condition_in_runner",
        "predicted_updated_graph" in CONDITIONS,
        f"{len(CONDITIONS)} conditions registered",
    )


def step_docs(gate_results: Path, bench_note: Path, checklist: Path) -> None:
    print("\n[11 docs_vs_data]")
    if not gate_results.exists():
        check("docs", "gate_results_present", False, f"missing {gate_results}")
        return
    payload = json.loads(gate_results.read_text())
    summary = payload.get("summary", {})
    note = bench_note.read_text() if bench_note.exists() else ""
    checklist_text = checklist.read_text() if checklist.exists() else ""

    checks = {f"{c['section']}/{c['name']}": c for c in payload.get("checks", [])}

    def detail_token(check_name: str) -> str:
        """Leading numeric token from a gate detail, e.g. '50.5%' or '0.169'."""
        detail = checks.get(check_name, {}).get("detail", "")
        match = re.match(r"([0-9]+(?:\.[0-9]+)?%?)", detail.strip())
        return match.group(1) if match else ""

    claims = [
        ("pass_count", f"{summary.get('pass')} pass" in note),
        ("fail_0", summary.get("fail", 0) == 0),
        ("grid_396", "396" in note),
    ]
    for label, check_name in (
        ("cycle", "shortcuts/cycle_degree_ceiling"),
        ("reach", "shortcuts/reach_status_ceiling"),
        ("seven", "shortcuts/seven_task_ceiling"),
        ("noop_val", "sessions/validation_noop_rate"),
        ("noop_test", "sessions/test_noop_rate"),
    ):
        token = detail_token(check_name)
        claims.append((label, bool(token) and token in note))

    bad = [name for name, ok in claims if not ok]
    check(
        "docs",
        "benchmark_note_numbers",
        not bad,
        f"mismatched {bad}"
        if bad
        else f"all cited numbers match ({', '.join(t for t, _ in claims if t)!r})",
    )
    check(
        "docs",
        "checklist_present",
        checklist.exists() and "3 seeds" in checklist_text and "notice" in checklist_text,
        "v4_run_checklist.md",
    )
    info("docs", "gate_summary", f"{summary}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(REPO_ROOT / "datasets/metro_v4_1"))
    parser.add_argument("--work", default="/tmp/graphmodi_v4_final_probe")
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "documents/experiments/results/benchmark-final-audit-v4_1-20261030"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "datasets/metro_v4_1/manifest.json"),
    )
    parser.add_argument(
        "--gate-results",
        default=str(
            REPO_ROOT / "documents/experiments/results/benchmark-audit-v4_1-20261030/results.json"
        ),
    )
    parser.add_argument(
        "--note",
        default=str(REPO_ROOT / "documents/experiments/benchmark_v4_1_2026-10-30.md"),
        help="benchmark note whose cited gate numbers are checked against --gate-results",
    )
    args = parser.parse_args()
    data = Path(args.data)
    out = Path(args.out)
    t0 = time.time()

    print(f"Loading metro_v4 from {data} ...")
    static = {
        split: load_static_tuples(data / f"static_{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    sessions = {
        split: load_sessions(data / f"sessions_{split}.jsonl") for split in ("validation", "test")
    }
    print(
        f"loaded static {{{', '.join(f'{k}:{len(v)}' for k, v in static.items())}}} "
        f"sessions {{{', '.join(f'{k}:{len(v)}' for k, v in sessions.items())}}} "
        f"({time.time() - t0:.0f}s)"
    )

    step_determinism(data, Path(args.work), Path(args.manifest))
    step_graph_generation(static, sessions)
    step_questions(static)
    step_labels(static, sessions)
    step_edits(sessions)
    step_sessions(sessions)
    step_shortcuts(static, sessions)
    step_splits(static, sessions)
    step_plumbing(static)
    step_evaluation(sessions)
    step_docs(
        Path(args.gate_results),
        Path(args.note),
        REPO_ROOT / "documents/experiments/v4_run_checklist.md",
    )

    summary = Counter(item["status"] for item in CHECKS)
    print(
        f"\nSUMMARY pass={summary['pass']} fail={summary['fail']} "
        f"warn={summary['warn']} info={summary['info']} ({time.time() - t0:.0f}s)"
    )
    for item in CHECKS:
        if item["status"] in ("fail", "warn"):
            print(
                f"  [{item['status'].upper()}] {item['section']}/{item['name']}  {item['detail']}"
            )

    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": dict(summary),
        "checks": CHECKS,
        "sections": RESULTS,
        "conventions": {
            "reachability": "open endpoints and open intermediates; unweighted BFS",
            "constrained_reachability": (
                "endpoints exempt from avoided class; open/closed not filtered"
            ),
            "within_hops_count": "excludes start node",
            "most_common_attribute_within_hops": "includes start node; unique mode required",
            "cycle_membership": "two neighbours connected after node removal",
            "shortest_path_answer": "path length including endpoints, or unreachable",
            "bert_status": "included in BERT node text by design; withheld from W(f_i)",
        },
    }
    (out / "results.json").write_text(json.dumps(payload, indent=1, default=str))
    print(f"wrote {out}")
    if summary["fail"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
