"""Dataset v4: corrected metro benchmark (audit findings from 2026-10).

Builds on v3 but does not modify it. Changes vs v3:
- pendant chains (length 1-3) so non-leaf stations off a cycle exist;
- node and edge order shuffled (no position leak);
- closed count always 1-3 (sessions match static);
- cycle queries only on degree-2 stations (degree alone cannot decide);
- reachability / shortest_path only over open endpoints (status shortcut broken);
- constrained reachability question states the endpoint exemption;
- unique mode required on both before and after for most_common;
- hop_depth varies with actual distance / hop radius;
- edit programs: realized NOOP ~17%, no net-noop / cancel / revert, non-bridge DEL,
  closed-count cap, ADD edges carry track/transfer + attributes;
- session queries balanced on mention-of-edit so revision text does not decide the answer;
- write-path uses paraphrase families and natural-language edge templates.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from graph_modi.data import v3 as V3
from graph_modi.graph.executor import apply_edit, graph_fingerprint
from graph_modi.graph.solvers import (
    _prepend_wfi,
    answer_query,
    shortest_path,
    wfi_excluded_attributes,
)
from graph_modi.schema import (
    AttributedGraph,
    DensityBin,
    Edge,
    EditOperation,
    EditProgram,
    EditTarget,
    GraphEdit,
    GraphQuery,
    HopDepth,
    Node,
    ReasoningType,
    Session,
    StaticQATuple,
    TopologyFamily,
    Turn,
)
from graph_modi.utils.progress import ProgressTracker

DISTRIBUTION_V4 = "metro_v4"
NODE_COUNTS = V3.NODE_COUNTS
DENSITIES = V3.DENSITIES
MEAN_DEGREE = V3.MEAN_DEGREE
MAX_DIAMETER = V3.MAX_DIAMETER
SESSION_LENGTHS = V3.SESSION_LENGTHS
NOOP_RATE = 0.17
MAX_EDITS_PER_TURN = 3
MAX_CLOSED = 3
STATIC_TASKS = V3.STATIC_TASKS
DYNAMIC_TASKS = V3.DYNAMIC_TASKS
YES_NO_TASKS = V3.YES_NO_TASKS
BalancedSampler = V3.BalancedSampler
scale_bin = V3.scale_bin
label_audit = V3.label_audit
session_label_audit = V3.session_label_audit
graph_audit = V3.graph_audit
text_leak_audit = V3.text_leak_audit

PARAPHRASE_FAMILIES = ("direct", "service", "notice")
# Held out of the few-shot examples in tea_glm so write-path eval is not a copy of the prompt.
HELD_OUT_FAMILY = "notice"


# --------------------------------------------------------------------------- topology
def _bridges(adjacency: list[set[int]]) -> set[frozenset[int]]:
    """Tarjan bridges on an undirected adjacency list keyed by integer indices."""
    n = len(adjacency)
    disc = [-1] * n
    low = [-1] * n
    parent = [-1] * n
    bridges: set[frozenset[int]] = set()
    time = 0

    def dfs(u: int) -> None:
        nonlocal time
        disc[u] = low[u] = time
        time += 1
        for v in adjacency[u]:
            if disc[v] == -1:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.add(frozenset((u, v)))
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if disc[i] == -1:
            dfs(i)
    return bridges


def _bridges_named(graph: AttributedGraph) -> set[frozenset[str]]:
    ids = [n.id for n in graph.nodes]
    index = {nid: i for i, nid in enumerate(ids)}
    adj: list[set[int]] = [set() for _ in ids]
    for edge in graph.edges:
        adj[index[edge.source]].add(index[edge.target])
        adj[index[edge.target]].add(index[edge.source])
    return {frozenset((ids[a], ids[b])) for a, b in _bridges(adj)}


def _base_topology_chains(
    rng: random.Random,
    node_count: int,
    mean_degree: float,
    *,
    max_chain: int = 4,
) -> set[tuple[int, int]]:
    """Ring core + hub chords + pendant chains (deg-2 yes on the ring, deg-2 no on chains)."""
    n_chains = 3 if node_count < 17 else 4
    budget = node_count - 5  # keep a core of at least 5
    lengths: list[int] = []
    for _ in range(n_chains):
        remaining_slots = n_chains - len(lengths)
        remaining_budget = budget - sum(lengths)
        if remaining_budget <= 0:
            break
        lo, hi = 1, min(max_chain, remaining_budget - (remaining_slots - 1))
        if hi < lo:
            break
        lengths.append(rng.randint(lo, hi))
    if not lengths:
        lengths = [1, 1]
    chain_nodes = sum(lengths)
    core = node_count - chain_nodes
    assert core >= 5
    edges: set[tuple[int, int]] = set()
    if core >= 3:
        for index in range(core):
            left, right = sorted((index, (index + 1) % core))
            edges.add((left, right))
    else:
        for index in range(1, core):
            edges.add((0, index))
    hub = list(range(max(3, (core * 2 + 2) // 3)))  # leave ~1/3 of the ring at deg 2
    target = round(mean_degree * node_count / 2) - chain_nodes
    guard = 0
    while len(edges) < target and guard < 2000:
        guard += 1
        left, right = sorted(rng.sample(hub, 2))
        edges.add((left, right))
    next_id = core
    for length in lengths:
        attach = rng.choice(hub)
        prev = attach
        for _ in range(length):
            edges.add((prev, next_id))
            prev = next_id
            next_id += 1
    assert next_id == node_count
    return edges


def _edge_record(
    left_id: str,
    right_id: str,
    lines: dict[str, str],
    rng: random.Random | None = None,
) -> Edge:
    """Build a track/transfer edge with deterministic attributes.

    See ``executor.metro_edge_attributes``. ``rng`` is accepted for call-site
    compatibility but unused: replay via ``apply_edit`` must match generation,
    which forbids stream-dependent attributes.
    """
    del rng  # deterministic; see metro_edge_attributes
    from graph_modi.graph.executor import metro_edge_attributes

    same = lines[left_id] == lines[right_id]
    return Edge(
        source=left_id,
        target=right_id,
        relation="track" if same else "transfer",
        weight=1.0,
        attributes=dict(metro_edge_attributes(left_id, right_id, lines)),
    )


def make_metro_graph_v4(
    rng: random.Random,
    split: str,
    index: int,
    node_count: int,
    density: DensityBin,
    *,
    closed_count: int | None = None,
) -> AttributedGraph:
    # Dense / large graphs need shorter chains to stay under the diameter cap.
    max_chain = (
        2
        if density is DensityBin.DENSE or node_count >= 20
        else 3
        if density is DensityBin.MEDIUM
        else 4
    )
    for _ in range(2000):
        edges = _base_topology_chains(rng, node_count, MEAN_DEGREE[density], max_chain=max_chain)
        adjacency = V3._adjacency(node_count, edges)
        ecc = V3._eccentricities(adjacency)
        if ecc is not None and max(ecc) <= MAX_DIAMETER[density]:
            break
    else:
        raise RuntimeError(f"no connected {density.value} graph with n={node_count}")

    line_count = 3 if node_count <= 14 else 4
    lines_by_pos = V3._color_lines(rng, adjacency, line_count)
    # Permute both ids and list order independently so neither leaks structure.
    id_perm = list(range(node_count))
    rng.shuffle(id_perm)
    order = list(range(node_count))
    rng.shuffle(order)
    names = [f"{a} {b}" for a in V3._NAME_A for b in V3._NAME_B]
    rng.shuffle(names)
    if closed_count is None:
        closed_count = rng.choice((1, 2, 3))
    closed_positions = set(rng.sample(range(node_count), min(closed_count, node_count)))

    nodes = []
    for list_pos, gen_pos in enumerate(order):
        nodes.append(
            Node(
                id=f"n{id_perm[gen_pos]}",
                label=names[list_pos],
                attributes={
                    "disabled_access": bool(rng.getrandbits(1)),
                    "has_rail": bool(rng.getrandbits(1)),
                    "architecture": rng.choice(V3.ARCHITECTURES),
                    "cleanliness": rng.choice(V3.CLEANLINESS),
                    "music": rng.choice(V3.MUSIC),
                    "size": rng.choice(V3.SIZES),
                    "line": lines_by_pos[gen_pos],
                    "status": "closed" if gen_pos in closed_positions else "open",
                },
            )
        )
    id_of = {gen_pos: f"n{id_perm[gen_pos]}" for gen_pos in range(node_count)}
    lines = {id_of[p]: lines_by_pos[p] for p in range(node_count)}
    edge_pairs = [(id_of[a], id_of[b]) for a, b in edges]
    rng.shuffle(edge_pairs)
    edge_records = [_edge_record(a, b, lines, rng) for a, b in edge_pairs]
    return AttributedGraph(
        graph_id=f"{split}-v4-graph-{index:06d}",
        nodes=tuple(nodes),
        edges=tuple(edge_records),
        metadata={
            "distribution": DISTRIBUTION_V4,
            "split": split,
            "node_count": node_count,
            "density": density.value,
            "mean_degree": round(2 * len(edges) / node_count, 3),
            "diameter": max(ecc),
            "radius": min(ecc),
            "wfi_exclude": V3.WFI_EXCLUDE,
            "line_count": line_count,
        },
    )


# --------------------------------------------------------------------------- questions
def render_question_v4(query: GraphQuery, graph: AttributedGraph) -> str:
    """Like render_question, but constrained reachability states the endpoint exemption."""
    if query.question:
        return query.question
    labels = {node.id: node.label for node in graph.nodes}
    source = labels.get(query.source, query.source)
    target = labels.get(query.target or "", query.target or "")
    if query.reasoning_type is ReasoningType.CONSTRAINED_REACHABILITY:
        body = (
            f"Can {target} be reached from {source} without passing through any intermediate "
            f"station where {query.attribute} is {query.value}? "
            f"(The start and end stations are exempt.) Answer yes or no."
        )
        return _prepend_wfi(
            body, graph, query.source, query.target, wfi_excluded_attributes(query, graph)
        )
    from graph_modi.graph.solvers import render_question

    return render_question(query, graph)


def _hop_v4(query: GraphQuery, graph: AttributedGraph) -> HopDepth:
    kind = query.reasoning_type
    if kind in (
        ReasoningType.ATTRIBUTE_LOOKUP,
        ReasoningType.ATTRIBUTE_CHECK,
        ReasoningType.NODE_DEGREE,
        ReasoningType.FILTERED_NEIGHBOR_COUNT,
        ReasoningType.EDGE_EXISTS,
    ):
        return HopDepth.LOCAL
    if kind in (ReasoningType.WITHIN_HOPS_COUNT, ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS):
        hops = query.hops or 2
        if hops <= 1:
            return HopDepth.LOCAL
        if hops == 2:
            return HopDepth.MID
        return HopDepth.GLOBAL
    if kind is ReasoningType.SHORTEST_PATH and query.target:
        path = shortest_path(graph, query.source, query.target)
        if not path:
            return HopDepth.GLOBAL
        hops = len(path)
        if hops <= 2:
            return HopDepth.LOCAL
        if hops <= 4:
            return HopDepth.MID
        return HopDepth.GLOBAL
    if kind is ReasoningType.REACHABILITY and query.target:
        path = shortest_path(graph, query.source, query.target)
        if not path:
            return HopDepth.GLOBAL
        hops = len(path)
        if hops <= 2:
            return HopDepth.LOCAL
        if hops <= 4:
            return HopDepth.MID
        return HopDepth.GLOBAL
    if kind is ReasoningType.CONSTRAINED_REACHABILITY:
        return HopDepth.GLOBAL
    if kind is ReasoningType.CYCLE_MEMBERSHIP:
        return HopDepth.MID if answer_query(graph, query) == "yes" else HopDepth.LOCAL
    return HopDepth.MID


def _is_open(graph: AttributedGraph, node_id: str) -> bool:
    return graph.node_map()[node_id].attributes.get("status", "open") != "closed"


def candidate_queries_v4(
    graph: AttributedGraph, task: ReasoningType, rng: random.Random
) -> list[GraphQuery]:
    """v3 candidates plus endpoint-open filter, degree>=2 cycles, and varied hop radii."""
    if task in (ReasoningType.REACHABILITY, ReasoningType.SHORTEST_PATH):
        # Prefer open endpoints so closed-endpoint → "no" cannot dominate. Only add
        # closed-endpoint pairs when the open pool has no negative answer.
        open_pairs, closed_pairs = [], []
        for a, b in V3._pairs(graph):
            q = GraphQuery(task, a, b)
            if _is_open(graph, a) and _is_open(graph, b):
                open_pairs.append(q)
            else:
                closed_pairs.append(q)
        # Open endpoints only: a closed endpoint makes the label trivial ("no" /
        # "unreachable"), and the sampler would over-pick those to fill the "no" quota.
        return V3._cap(rng, open_pairs)
    if task is ReasoningType.CYCLE_MEMBERSHIP:
        # Only degree-2 stations: both cycle and bridge-chain answers occur here, so
        # degree alone cannot predict the label (degree >= 3 is almost always "yes").
        degree = V3._degrees(graph)
        return V3._cap(rng, [GraphQuery(task, s) for s in degree if degree[s] == 2])
    if task is ReasoningType.WITHIN_HOPS_COUNT:
        filters: list[tuple[str, Any]] = [(attr, True) for attr in V3._BOOL_ATTRS]
        filters += [("line", value) for value in V3._values_present(graph, "line")]
        for attribute in ("architecture", "cleanliness"):
            filters += [(attribute, value) for value in V3._values_present(graph, attribute)]
        ids = [n.id for n in graph.nodes]
        return V3._cap(
            rng,
            [
                GraphQuery(task, s, attribute=a, value=v, hops=h)
                for s in ids
                for a, v in filters
                for h in (1, 2, 3)
            ],
        )
    if task is ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS:
        attributes = ("line", "architecture", "cleanliness", "music", "size")
        ids = [n.id for n in graph.nodes]
        found = [
            GraphQuery(task, s, attribute=a, hops=h)
            for s in ids
            for a in attributes
            for h in (1, 2, 3)
            if V3._unique_mode(graph, s, a, h)
        ]
        return V3._cap(rng, found)
    return V3.candidate_queries(graph, task, rng)


def stratum_v4(graph: AttributedGraph, query: GraphQuery) -> str:
    """Always include degree bins for yes/no structural tasks (anti-shortcut)."""
    nodes = graph.node_map()
    parts: list[str] = []
    if query.target is not None and query.target in nodes:
        same = nodes[query.source].attributes.get("line") == nodes[query.target].attributes.get(
            "line"
        )
        parts.append("same_line" if same else "diff_line")
    if query.attribute:
        parts.append(str(query.attribute))
    if query.hops is not None:
        parts.append(f"h{query.hops}")
    if (
        query.reasoning_type
        in (
            ReasoningType.REACHABILITY,
            ReasoningType.SHORTEST_PATH,
        )
        and query.target is not None
    ):
        # Balance closed-endpoint vs both-open so status cannot dominate.
        closed_end = (not _is_open(graph, query.source)) or (not _is_open(graph, query.target))
        parts.append("end_closed" if closed_end else "both_open")
    # Degree bins for edge_exists / constrained (where degree is a known shortcut). For
    # cycle and reachability the answer quota alone must drive 50/50; degree bins would
    # let the sampler fill "yes" from high-degree cells and leave cycle unbalanced.
    if query.reasoning_type in (
        ReasoningType.EDGE_EXISTS,
        ReasoningType.CONSTRAINED_REACHABILITY,
    ):
        degree = V3._degrees(graph)
        bins = [f"deg{min(degree[query.source], 3 if query.target else 4)}"]
        if query.target is not None and query.target in degree:
            bins.append(f"deg{min(degree[query.target], 3)}")
        parts.append("-".join(bins))
    # Cycle queries only name the source; balance yes/no inside prompt-visible source
    # attribute cells so W(f_i) text cannot pick up a spurious cleanliness/rail correlation.
    if query.reasoning_type is ReasoningType.CYCLE_MEMBERSHIP:
        src = nodes[query.source]
        parts.append(f"cln={src.attributes.get('cleanliness')}")
        parts.append(f"rail={src.attributes.get('has_rail')}")
    return "|".join(parts)


def _two_sided_v4(
    task: ReasoningType, answered: Sequence[tuple[Any, Any, str]]
) -> list[tuple[Any, Any, str]]:
    """Keep strata that offer both answers.

    Degree bins are stripped from the stratum key first: with open endpoints, a fine
    (deg_s, deg_t) cell is often all-yes, and requiring both answers inside it would
    empty the pool. Coarse strata (same_line / attribute) still block a single
    prompt-visible feature from deciding the label.
    """
    if task not in (
        ReasoningType.EDGE_EXISTS,
        ReasoningType.REACHABILITY,
        ReasoningType.CONSTRAINED_REACHABILITY,
        ReasoningType.CYCLE_MEMBERSHIP,
    ):
        return list(answered)

    def coarse(cell: str) -> str:
        return "|".join(part for part in cell.split("|") if not part.startswith("deg"))

    answers_by_stratum: dict[str, set[Any]] = defaultdict(set)
    for _, answer, cell in answered:
        answers_by_stratum[coarse(cell)].add(answer)
    return [item for item in answered if len(answers_by_stratum[coarse(item[2])]) >= 2]


def _static_tuple(
    split: str,
    graph: AttributedGraph,
    query: GraphQuery,
    answer: str,
    tuple_id: str,
    density: DensityBin,
    task: ReasoningType,
    extra: dict[str, Any] | None = None,
) -> StaticQATuple:
    query = replace(query, question=render_question_v4(query, graph))
    return StaticQATuple(
        tuple_id=tuple_id,
        split=split,
        graph=graph,
        query=query,
        answer=answer,
        topology=TopologyFamily.METRO,
        density_bin=density,
        hop_depth=_hop_v4(query, graph),
        scale_bin=scale_bin(len(graph.nodes)),
        metadata={"reasoning_type": task.value, "distribution": DISTRIBUTION_V4, **(extra or {})},
    )


def _cells() -> list[tuple[int, DensityBin]]:
    return [(n, d) for n in NODE_COUNTS for d in DENSITIES]


# --------------------------------------------------------------------------- edits
def _closed_count(graph: AttributedGraph) -> int:
    return sum(n.attributes.get("status") == "closed" for n in graph.nodes)


def _candidate_edit_v4(
    graph: AttributedGraph, operation: EditOperation, rng: random.Random
) -> GraphEdit | None:
    node_ids = [node.id for node in graph.nodes]
    if operation is EditOperation.SET:
        open_nodes = [n for n in graph.nodes if n.attributes.get("status", "open") == "open"]
        closed_nodes = [n for n in graph.nodes if n.attributes.get("status") == "closed"]
        # Prefer closing only while under the cap; always allow reopening.
        choices: list[tuple[Node, str]] = [(n, "open") for n in closed_nodes]
        if _closed_count(graph) < MAX_CLOSED:
            choices += [(n, "closed") for n in open_nodes]
        if not choices:
            return None
        node, value = rng.choice(choices)
        return GraphEdit(
            operation=operation,
            target=EditTarget.NODE,
            node_id=node.id,
            attribute="status",
            value=value,
        )
    if operation is EditOperation.ADD:
        existing = {frozenset((e.source, e.target)) for e in graph.edges}
        candidates = [
            (left, right)
            for i, left in enumerate(node_ids)
            for right in node_ids[i + 1 :]
            if frozenset((left, right)) not in existing
        ]
        if not candidates:
            return None
        source, destination = rng.choice(candidates)
        lines = {n.id: n.attributes["line"] for n in graph.nodes}
        same = lines[source] == lines[destination]
        return GraphEdit(
            operation=operation,
            target=EditTarget.EDGE,
            source=source,
            destination=destination,
            relation="track" if same else "transfer",
        )
    if operation is EditOperation.DEL and graph.edges:
        bridges = _bridges_named(graph)
        removable = [e for e in graph.edges if frozenset((e.source, e.target)) not in bridges]
        if not removable:
            return None
        edge = rng.choice(removable)
        return GraphEdit(
            operation=operation,
            target=EditTarget.EDGE,
            source=edge.source,
            destination=edge.target,
            relation=edge.relation,
        )
    return None


def _state_key(graph: AttributedGraph) -> tuple[frozenset[str], frozenset[frozenset[str]]]:
    closed = frozenset(n.id for n in graph.nodes if n.attributes.get("status") == "closed")
    edges = frozenset(frozenset((e.source, e.target)) for e in graph.edges)
    return (closed, edges)


def _touches(edit: GraphEdit) -> tuple[str, str | frozenset[str] | None]:
    if edit.target is EditTarget.NODE:
        return ("n", edit.node_id)
    return ("e", frozenset((edit.source or "", edit.destination or "")))


def apply_program_v4(
    graph: AttributedGraph, program: EditProgram, rng: random.Random
) -> AttributedGraph:
    """Apply edits. ADD edges get CLEGR-style attributes inside ``apply_edit`` (deterministic)."""
    del rng  # attributes no longer depend on the session RNG stream
    current = graph
    for edit in program.edits:
        if edit.operation is EditOperation.NOOP:
            continue
        result = apply_edit(current, edit)
        if result.applied:
            current = result.graph
    return current


def sample_edit_program_v4(
    graph: AttributedGraph,
    rng: random.Random,
    *,
    force_noop: bool = False,
    seen_states: set[tuple[frozenset[str], frozenset[frozenset[str]]]] | None = None,
) -> EditProgram:
    if force_noop:
        return EditProgram(edits=(GraphEdit(operation=EditOperation.NOOP, reason="explicit"),))
    edit_count = rng.randint(1, MAX_EDITS_PER_TURN)  # never 0: NOOP is only via force_noop
    operations = (EditOperation.SET, EditOperation.ADD, EditOperation.DEL)
    for _attempt in range(48):
        edits: list[GraphEdit] = []
        current = graph
        touched: set[tuple[str, str | frozenset[str] | None]] = set()
        for _ in range(edit_count):
            placed = False
            for _try in range(32):
                operation = rng.choice(operations)
                edit = _candidate_edit_v4(current, operation, rng)
                if edit is None:
                    continue
                key = _touches(edit)
                if key in touched:
                    continue
                trial = apply_program_v4(current, EditProgram(edits=(edit,)), rng)
                if _state_key(trial) == _state_key(current):
                    continue
                edits.append(edit)
                touched.add(key)
                current = trial
                placed = True
                break
            if not placed:
                break
        if not edits:
            continue
        final = apply_program_v4(graph, EditProgram(edits=tuple(edits)), rng)
        if _state_key(final) == _state_key(graph):
            continue  # net no-op
        if seen_states is not None and _state_key(final) in seen_states:
            continue  # revert
        return EditProgram(edits=tuple(edits))
    return EditProgram(edits=(GraphEdit(operation=EditOperation.NOOP, reason="fallback"),))


# --------------------------------------------------------------------------- utterances
def _utterance_v4(edit: GraphEdit, graph: AttributedGraph, family: str) -> str:
    labels = {node.id: node.label for node in graph.nodes}
    if edit.operation is EditOperation.SET and edit.node_id:
        label = labels[edit.node_id]
        if edit.value == "closed":
            return {
                "direct": f"{label} is closed now.",
                "service": f"Service at {label} has been suspended.",
                "notice": f"Please mark {label} as closed.",
            }[family]
        return {
            "direct": f"{label} has reopened.",
            "service": f"Service at {label} has resumed.",
            "notice": f"Please mark {label} as open.",
        }[family]
    if edit.target is EditTarget.EDGE and edit.source and edit.destination:
        a, b = labels[edit.source], labels[edit.destination]
        if edit.operation is EditOperation.ADD:
            return {
                "direct": f"A new {edit.relation} link now connects {a} and {b}.",
                "service": f"A {edit.relation} connection between {a} and {b} has opened.",
                "notice": f"Please add a {edit.relation} edge between {a} and {b}.",
            }[family]
        return {
            "direct": f"The {edit.relation} link between {a} and {b} is gone.",
            "service": f"The {edit.relation} connection between {a} and {b} has closed.",
            "notice": f"Please remove the {edit.relation} edge between {a} and {b}.",
        }[family]
    return edit.canonical()


def _noop_utterance(rng: random.Random, family: str) -> str:
    options = {
        "direct": (
            "No change this turn.",
            "Nothing to update on the graph.",
            "Leave the network as it is.",
        ),
        "service": (
            "Operations report: no network changes.",
            "Service bulletin: status quo maintained.",
            "No suspension or reopening this turn.",
        ),
        "notice": (
            "Please make no changes to the graph.",
            "Hold the current network state.",
            "Do not apply any edit this turn.",
        ),
    }
    return rng.choice(options[family])


def _mentions_edit(query: GraphQuery, edits: Sequence[GraphEdit]) -> bool:
    touched: set[str] = set()
    for edit in edits:
        if edit.operation is EditOperation.NOOP:
            continue
        if edit.target is EditTarget.NODE and edit.node_id:
            touched.add(edit.node_id)
        if edit.target is EditTarget.EDGE:
            if edit.source:
                touched.add(edit.source)
            if edit.destination:
                touched.add(edit.destination)
    if not touched:
        return False
    named = {query.source} | ({query.target} if query.target else set())
    return bool(named & touched)


# -------------------------------------------------------- static / counterfactual / sessions
def generate_static_v4(
    *,
    split: str,
    graph_count: int,
    tasks_per_graph: int | None,
    seed: int,
    sampler: BalancedSampler,
    tasks: Sequence[ReasoningType] = STATIC_TASKS,
    progress: bool = True,
) -> list[StaticQATuple]:
    offsets = {"train": 0, "validation": 5_000_000, "test": 6_000_000}
    cells = _cells()
    tracker = ProgressTracker(
        "[generate]", graph_count, phase=f"static-v4-{split}", enabled=progress
    )
    tracker.banner(graphs=graph_count)
    result: list[StaticQATuple] = []
    for index in range(graph_count):
        rng = random.Random(seed + offsets[split] + index)
        node_count, density = cells[index % len(cells)]
        graph = make_metro_graph_v4(rng, split, index, node_count, density)
        chosen = (
            list(tasks)
            if tasks_per_graph is None
            else [tasks[(index + j) % len(tasks)] for j in range(tasks_per_graph)]
        )
        for j, task in enumerate(chosen):
            answered = [
                (query, answer_query(graph, query), stratum_v4(graph, query))
                for query in candidate_queries_v4(graph, task, rng)
            ]
            if not answered and task is ReasoningType.CYCLE_MEMBERSHIP:
                # Rare graph with no degree-2 station: allow degree >= 2 rather than
                # falling back to leaves (which reintroduce the degree shortcut).
                degree = V3._degrees(graph)
                answered = [
                    (q, answer_query(graph, q), stratum_v4(graph, q))
                    for q in (GraphQuery(task, s) for s in degree if degree[s] >= 2)
                ]
            if not answered:
                # Last resort for other tasks (e.g. empty unique-mode pool).
                answered = [
                    (query, answer_query(graph, query), stratum_v4(graph, query))
                    for query in V3.candidate_queries(graph, task, rng)
                ]
            if not answered:
                continue
            # Prefer two-sided strata, but never drop the task (eval needs a full grid).
            answered = _two_sided_v4(task, answered) or answered
            query, answer = sampler.choose((split, task.value, density.value), answered, rng)
            result.append(
                _static_tuple(
                    split,
                    graph,
                    query,
                    answer,
                    f"{split}-static-{index:06d}-{j:02d}",
                    density,
                    task,
                )
            )
        tracker.tick(graph_i=f"{index + 1}/{graph_count}")
    tracker.end(tuples=len(result))
    return result


def generate_counterfactual_pairs_v4(
    *,
    graph_count: int,
    pairs_per_graph: int,
    seed: int,
    sampler: BalancedSampler,
    tasks: Sequence[ReasoningType] = DYNAMIC_TASKS,
    progress: bool = True,
) -> list[StaticQATuple]:
    cells = _cells()
    split = "train"
    tracker = ProgressTracker(
        "[generate]", graph_count, phase="counterfactual-v4", enabled=progress
    )
    tracker.banner(graphs=graph_count)
    result: list[StaticQATuple] = []
    for index in range(graph_count):
        rng = random.Random(seed + 8_000_000 + index)
        node_count, density = cells[index % len(cells)]
        graph = make_metro_graph_v4(rng, split, 100_000 + index, node_count, density)
        made = 0
        for attempt in range(pairs_per_graph * 12):
            if made >= pairs_per_graph:
                break
            task = tasks[(index + made + attempt) % len(tasks)]
            program = sample_edit_program_v4(graph, rng, force_noop=False)
            if program.edits[0].operation is EditOperation.NOOP:
                continue
            after = apply_program_v4(graph, program, rng)
            changed = []
            for query in candidate_queries_v4(after, task, rng):
                before_answer = answer_query(graph, query)
                after_answer = answer_query(after, query)
                if before_answer == after_answer:
                    continue
                if task is ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS:
                    hops = query.hops or 2
                    if not (
                        V3._unique_mode(graph, query.source, query.attribute or "", hops)
                        and V3._unique_mode(after, query.source, query.attribute or "", hops)
                    ):
                        continue
                changed.append((query, after_answer, before_answer))
            if not changed:
                continue
            key = (split, task.value, density.value, "cf")
            scored = [(q, a, stratum_v4(after, q)) for q, a, _ in changed]
            query, after_answer = sampler.choose(key, _two_sided_v4(task, scored) or scored, rng)
            before_answer = next(b for q, a, b in changed if q == query and a == after_answer)
            edited = replace(after, graph_id=f"{graph.graph_id}-cf{made}")
            for label, item_graph, answer in (
                ("before", graph, before_answer),
                ("after", edited, after_answer),
            ):
                result.append(
                    _static_tuple(
                        split,
                        item_graph,
                        query,
                        answer,
                        f"train-cf-{index:06d}-{made:02d}-{label}",
                        density,
                        task,
                        {"counterfactual": label},
                    )
                )
            made += 1
        tracker.tick(graph_i=f"{index + 1}/{graph_count}")
    tracker.end(tuples=len(result))
    return result


def generate_session_v4(
    *,
    split: str,
    index: int,
    seed: int,
    node_count: int,
    density: DensityBin,
    turn_count: int,
    task: ReasoningType,
    sampler: BalancedSampler,
) -> Session:
    rng = random.Random(seed)
    initial = make_metro_graph_v4(
        rng, split, index, node_count, density, closed_count=rng.choice((1, 2, 3))
    )
    current = initial
    turns: list[Turn] = []
    family = PARAPHRASE_FAMILIES[index % len(PARAPHRASE_FAMILIES)]
    seen_states: set[tuple[frozenset[str], frozenset[frozenset[str]]]] = {_state_key(initial)}
    for turn_index in range(turn_count):
        force_noop = rng.random() < NOOP_RATE
        program = sample_edit_program_v4(
            current, rng, force_noop=force_noop, seen_states=seen_states
        )
        before = current
        updated = apply_program_v4(before, program, rng)
        is_noop = program.edits[0].operation is EditOperation.NOOP
        candidates = []
        for query in candidate_queries_v4(updated, task, rng):
            gold = answer_query(updated, query)
            stale = answer_query(before, query)
            if task is ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS:
                hops = query.hops or 2
                if not (
                    V3._unique_mode(before, query.source, query.attribute or "", hops)
                    and V3._unique_mode(updated, query.source, query.attribute or "", hops)
                ):
                    continue
            mention = _mentions_edit(query, program.edits)
            # Stratum includes mention so the sampler balances mention x answer.
            cell = f"{stratum_v4(updated, query)}|{'m' if mention else 'u'}"
            candidates.append((query, gold, cell, stale, mention))
        if not candidates:
            # Fall back without the unique-mode / open-endpoint filters so the session can finish.
            for query in V3.candidate_queries(updated, task, rng)[:40]:
                gold = answer_query(updated, query)
                stale = answer_query(before, query)
                mention = _mentions_edit(query, program.edits)
                cell = f"{stratum_v4(updated, query)}|{'m' if mention else 'u'}"
                candidates.append((query, gold, cell, stale, mention))
        scored = [(q, a, c) for q, a, c, _, _ in candidates]
        query, answer = sampler.choose(
            (split, task.value, density.value, "dyn"),
            _two_sided_v4(task, scored) or scored,
            rng,
        )
        meta = next((m, s) for q, a, c, s, m in candidates if q == query and a == answer)
        mention, stale_answer = meta
        query = replace(query, question=render_question_v4(query, updated))
        utterance = (
            _noop_utterance(rng, family)
            if is_noop
            else " ; ".join(_utterance_v4(edit, before, family) for edit in program.edits)
        )
        turns.append(
            Turn(
                turn_index=turn_index,
                utterance=utterance,
                gold_edit=program.edits[0],
                gold_edits=program.edits,
                edit_program=program,
                query=query,
                gold_answer=answer,
                stale_answer=stale_answer,
                before_fingerprint=graph_fingerprint(before),
                after_fingerprint=graph_fingerprint(updated),
                tags=(
                    str(len(program.edits)),
                    query.reasoning_type.value,
                    TopologyFamily.METRO.value,
                    "noop" if is_noop else "edit",
                    "mention" if mention else "no_mention",
                    family,
                ),
                complexity={
                    "topology": TopologyFamily.METRO.value,
                    "density": density.value,
                    "target_density": density.value,
                    "scale_bin": scale_bin(node_count),
                    "node_count": node_count,
                    "hop_depth": _hop_v4(query, updated).value,
                    "edit_count": len(program.edits),
                    "ood": False,
                    "session_length": turn_count,
                    "mentions_edit": mention,
                    "paraphrase_family": family,
                    "distribution": DISTRIBUTION_V4,
                },
            )
        )
        seen_states.add(_state_key(updated))
        current = updated
    return Session(
        session_id=f"{split}-v4-{index:06d}",
        split=split,
        initial_graph=initial,
        turns=tuple(turns),
        seed=seed,
        paraphrase_family=f"{split}-{family}",
    )


def generate_sessions_v4(
    *,
    replicates_per_split: dict[str, int],
    seed: int,
    tasks: Sequence[ReasoningType] = DYNAMIC_TASKS,
    progress: bool = True,
) -> dict[str, list[Session]]:
    offsets = {"validation": 1_000_000, "test": 2_000_000, "train": 0}
    cells = [
        (n, d, length, task)
        for n in NODE_COUNTS
        for d in DENSITIES
        for length in SESSION_LENGTHS
        for task in tasks
    ]
    total = sum(count * len(cells) for count in replicates_per_split.values())
    tracker = ProgressTracker("[generate]", total, phase="sessions-v4", enabled=progress)
    tracker.banner(sessions=total, cells=len(cells))
    result: dict[str, list[Session]] = {}
    for split, replicates in replicates_per_split.items():
        sampler = BalancedSampler()
        sessions: list[Session] = []
        index = 0
        for _ in range(replicates):
            for node_count, density, length, task in cells:
                sessions.append(
                    generate_session_v4(
                        split=split,
                        index=index,
                        seed=seed + offsets[split] + index,
                        node_count=node_count,
                        density=density,
                        turn_count=length,
                        task=task,
                        sampler=sampler,
                    )
                )
                index += 1
                tracker.tick(split=split)
        result[split] = sessions
    tracker.end(sessions=total)
    return result
