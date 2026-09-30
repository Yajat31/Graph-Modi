"""Dataset v3: compact CLEGR-style subway graphs with label-balanced tasks.

Differences from v2 (see documents/experiments/status_report_fixed_final.md for why):
- 12-22 stations; density defined by mean degree, connectivity and diameter enforced, and a few
  terminal (degree-1) stations per graph so every task has both answer classes available;
- CLEGR-like node semantics (name, disabled_access, has_rail, architecture, cleanliness, music,
  size, line) and edge semantics (line_color, line_stroke, has_aircon, built);
- ``status`` (open/closed) stays in the graph but is withheld from the W(f_i) prompt text;
- queries are drawn with per-task answer quotas, so yes/no tasks are 50/50 and the other tasks
  have flat answer distributions;
- only tasks a 3-layer GraphSAGE over node features can plausibly encode (no edge weights, no
  node-name outputs, no global counts).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any

from graph_modi.data.multiturn import _utterance
from graph_modi.data.v2 import _hop_depth, _sample_edit_program
from graph_modi.graph.executor import apply_edit_program, graph_fingerprint
from graph_modi.graph.solvers import answer_query, render_question
from graph_modi.schema import (
    AttributedGraph,
    DensityBin,
    Edge,
    EditOperation,
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

DISTRIBUTION_V3 = "metro_v3"
NODE_COUNTS = tuple(range(12, 23))
DENSITIES = (DensityBin.SPARSE, DensityBin.MEDIUM, DensityBin.DENSE)
MEAN_DEGREE = {DensityBin.SPARSE: 2.8, DensityBin.MEDIUM: 3.8, DensityBin.DENSE: 5.0}
MAX_DIAMETER = {DensityBin.SPARSE: 7, DensityBin.MEDIUM: 6, DensityBin.DENSE: 5}
SESSION_LENGTHS = (1, 2, 4, 8)
CANDIDATE_CAP = 80
NOOP_RATE = 0.17
WFI_EXCLUDE = "status"

LINES = ("red", "blue", "green", "yellow")
LINE_STROKES = {"red": "solid", "blue": "solid", "green": "dashed", "yellow": "dotted"}
ARCHITECTURES = ("modern", "art_deco", "brutalist", "victorian")
CLEANLINESS = ("clean", "average", "dirty")
MUSIC = ("none", "jazz", "classical", "pop")
SIZES = ("small", "medium", "large")
_NAME_A = (
    "Astor", "Baywood", "Cedar", "Dunmore", "Elm", "Fairhaven", "Grove", "Harlow", "Ironwood",
    "Juniper", "Kestrel", "Linden", "Marlow", "Norwood", "Oakley", "Pemberton", "Quill",
    "Rowan", "Sable", "Thornbury", "Upland", "Vale", "Willow", "Yarrow", "Zephyr", "Ashby",
    "Brook", "Crest", "Delmar", "Everly",
)
_NAME_B = ("Square", "Junction", "Park", "Cross", "Gate", "Hill", "Court", "Market")

STATIC_TASKS = (
    ReasoningType.EDGE_EXISTS,
    ReasoningType.REACHABILITY,
    ReasoningType.CONSTRAINED_REACHABILITY,
    ReasoningType.CYCLE_MEMBERSHIP,
    ReasoningType.NODE_DEGREE,
    ReasoningType.FILTERED_NEIGHBOR_COUNT,
    ReasoningType.SHORTEST_PATH,
    ReasoningType.WITHIN_HOPS_COUNT,
    ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS,
    ReasoningType.ATTRIBUTE_LOOKUP,
    ReasoningType.ATTRIBUTE_CHECK,
)
DYNAMIC_TASKS = STATIC_TASKS[:9]
YES_NO_TASKS = frozenset(
    {
        ReasoningType.EDGE_EXISTS,
        ReasoningType.REACHABILITY,
        ReasoningType.CONSTRAINED_REACHABILITY,
        ReasoningType.CYCLE_MEMBERSHIP,
        ReasoningType.ATTRIBUTE_CHECK,
    }
)
_BOOL_ATTRS = ("disabled_access", "has_rail")
_CATEGORICAL = {
    "architecture": ARCHITECTURES,
    "cleanliness": CLEANLINESS,
    "music": MUSIC,
    "size": SIZES,
}


def scale_bin(node_count: int) -> str:
    if node_count <= 15:
        return "scale_small"
    if node_count <= 19:
        return "scale_medium"
    return "scale_large"


# --------------------------------------------------------------------------- graphs
def _adjacency(node_count: int, edges: Iterable[tuple[int, int]]) -> list[set[int]]:
    result: list[set[int]] = [set() for _ in range(node_count)]
    for left, right in edges:
        result[left].add(right)
        result[right].add(left)
    return result


def _eccentricities(adjacency: list[set[int]]) -> list[int] | None:
    values: list[int] = []
    for start in range(len(adjacency)):
        distance = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in distance:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
        if len(distance) != len(adjacency):
            return None
        values.append(max(distance.values()))
    return values


def _base_topology(
    rng: random.Random, node_count: int, mean_degree: float, terminals: int
) -> set[tuple[int, int]]:
    core = node_count - terminals
    edges: set[tuple[int, int]] = set()
    for index in range(1, core):
        edges.add((rng.randrange(max(0, index - 3), index), index))
    target = round(mean_degree * node_count / 2) - terminals
    guard = 0
    while len(edges) < target and guard < 2000:
        guard += 1
        left, right = sorted(rng.sample(range(core), 2))
        edges.add((left, right))
    for offset in range(terminals):
        edges.add((rng.randrange(core), core + offset))
    return edges


def _color_lines(
    rng: random.Random, adjacency: list[set[int]], line_count: int
) -> list[str]:
    seeds = rng.sample(range(len(adjacency)), line_count)
    color: dict[int, str] = {seed: LINES[i] for i, seed in enumerate(seeds)}
    queue = deque(seeds)
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adjacency[node]):
            if neighbor not in color:
                color[neighbor] = color[node]
                queue.append(neighbor)
    return [color[i] for i in range(len(adjacency))]


def make_metro_graph(
    rng: random.Random,
    split: str,
    index: int,
    node_count: int,
    density: DensityBin,
    *,
    closed_count: int | None = None,
) -> AttributedGraph:
    terminals = 2 if node_count < 17 else 3
    for _ in range(400):
        edges = _base_topology(rng, node_count, MEAN_DEGREE[density], terminals)
        adjacency = _adjacency(node_count, edges)
        ecc = _eccentricities(adjacency)
        if ecc is not None and max(ecc) <= MAX_DIAMETER[density]:
            break
    else:
        raise RuntimeError(f"no connected {density.value} graph with n={node_count}")

    line_count = 3 if node_count <= 14 else 4
    lines = _color_lines(rng, adjacency, line_count)
    permutation = list(range(node_count))
    rng.shuffle(permutation)  # ids are permuted so adjacency cannot be read off id arithmetic
    names = [f"{a} {b}" for a in _NAME_A for b in _NAME_B]
    rng.shuffle(names)
    if closed_count is None:
        closed_count = rng.choice((1, 2, 3))
    closed = set(rng.sample(range(node_count), min(closed_count, node_count)))
    nodes = []
    for position in range(node_count):
        nodes.append(
            Node(
                id=f"n{permutation[position]}",
                label=names[position],
                attributes={
                    "disabled_access": bool(rng.getrandbits(1)),
                    "has_rail": bool(rng.getrandbits(1)),
                    "architecture": rng.choice(ARCHITECTURES),
                    "cleanliness": rng.choice(CLEANLINESS),
                    "music": rng.choice(MUSIC),
                    "size": rng.choice(SIZES),
                    "line": lines[position],
                    "status": "closed" if position in closed else "open",
                },
            )
        )
    edge_records = []
    for left, right in sorted(edges):
        same = lines[left] == lines[right]
        edge_records.append(
            Edge(
                source=f"n{permutation[left]}",
                target=f"n{permutation[right]}",
                relation="track" if same else "transfer",
                weight=1.0,
                attributes={
                    "line_color": lines[left] if same else "transfer",
                    "line_stroke": LINE_STROKES[lines[left]] if same else "solid",
                    "has_aircon": bool(rng.getrandbits(1)),
                    "built": rng.randint(1950, 2020),
                },
            )
        )
    return AttributedGraph(
        graph_id=f"{split}-v3-graph-{index:06d}",
        nodes=tuple(nodes),
        edges=tuple(edge_records),
        metadata={
            "distribution": DISTRIBUTION_V3,
            "split": split,
            "node_count": node_count,
            "density": density.value,
            "mean_degree": round(2 * len(edges) / node_count, 3),
            "diameter": max(ecc),
            "radius": min(ecc),
            "wfi_exclude": WFI_EXCLUDE,
            "line_count": line_count,
        },
    )


# --------------------------------------------------------------------------- queries
def _pairs(graph: AttributedGraph) -> list[tuple[str, str]]:
    ids = [node.id for node in graph.nodes]
    return [(a, b) for i, a in enumerate(ids) for b in ids[i + 1 :]]


def _values_present(graph: AttributedGraph, attribute: str) -> list[Any]:
    return sorted({node.attributes[attribute] for node in graph.nodes}, key=str)


def _unique_mode(graph: AttributedGraph, source: str, attribute: str, hops: int) -> bool:
    from graph_modi.graph.solvers import nodes_within_hops

    ids = set(nodes_within_hops(graph, source, hops)) | {source}
    nodes = graph.node_map()
    counts = Counter(str(nodes[i].attributes.get(attribute)) for i in ids)
    top = counts.most_common(2)
    return len(top) == 1 or top[0][1] > top[1][1]


def _cap(rng: random.Random, items: list[Any], limit: int = CANDIDATE_CAP) -> list[Any]:
    rng.shuffle(items)
    return items[:limit]


def _edge_exists_candidates(graph: AttributedGraph, rng: random.Random) -> list[GraphQuery]:
    edge_set = {frozenset((e.source, e.target)) for e in graph.edges}
    adjacency: dict[str, set[str]] = {n.id: set() for n in graph.nodes}
    for edge in graph.edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
    positives, near, far = [], [], []
    for a, b in _pairs(graph):
        query = GraphQuery(ReasoningType.EDGE_EXISTS, a, b)
        if frozenset((a, b)) in edge_set:
            positives.append(query)
        elif adjacency[a] & adjacency[b]:
            near.append(query)
        else:
            far.append(query)
    rng.shuffle(near)
    rng.shuffle(far)
    keep = min(len(near), len(far)) if near and far else 0
    negatives = near[:keep] + far[:keep] if keep else near + far
    return _cap(rng, positives + negatives)


def candidate_queries(
    graph: AttributedGraph, task: ReasoningType, rng: random.Random
) -> list[GraphQuery]:
    ids = [node.id for node in graph.nodes]
    if task is ReasoningType.EDGE_EXISTS:
        return _edge_exists_candidates(graph, rng)
    if task in (ReasoningType.REACHABILITY, ReasoningType.SHORTEST_PATH):
        return _cap(rng, [GraphQuery(task, a, b) for a, b in _pairs(graph)])
    if task is ReasoningType.CONSTRAINED_REACHABILITY:
        avoid = [("disabled_access", False), ("has_rail", False), ("cleanliness", "dirty")]
        return _cap(
            rng,
            [
                GraphQuery(task, a, b, attribute=attr, value=value)
                for a, b in _pairs(graph)
                for attr, value in avoid
            ],
        )
    if task in (ReasoningType.CYCLE_MEMBERSHIP, ReasoningType.NODE_DEGREE):
        return _cap(rng, [GraphQuery(task, source) for source in ids])
    filters: list[tuple[str, Any]] = [(attr, True) for attr in _BOOL_ATTRS]
    filters += [("line", value) for value in _values_present(graph, "line")]
    for attribute in ("architecture", "cleanliness"):
        filters += [(attribute, value) for value in _values_present(graph, attribute)]
    if task is ReasoningType.FILTERED_NEIGHBOR_COUNT:
        return _cap(
            rng, [GraphQuery(task, s, attribute=a, value=v) for s in ids for a, v in filters]
        )
    if task is ReasoningType.WITHIN_HOPS_COUNT:
        return _cap(
            rng,
            [GraphQuery(task, s, attribute=a, value=v, hops=2) for s in ids for a, v in filters],
        )
    if task is ReasoningType.MOST_COMMON_ATTRIBUTE_WITHIN_HOPS:
        attributes = ("line", "architecture", "cleanliness", "music", "size")
        found = [
            GraphQuery(task, s, attribute=a, hops=2)
            for s in ids
            for a in attributes
            if _unique_mode(graph, s, a, 2)
        ]
        return _cap(rng, found)
    if task is ReasoningType.ATTRIBUTE_LOOKUP:
        attributes = ("architecture", "cleanliness", "music", "size", "line")
        return _cap(rng, [GraphQuery(task, s, attribute=a) for s in ids for a in attributes])
    if task is ReasoningType.ATTRIBUTE_CHECK:
        checks: list[tuple[str, Any]] = [(a, v) for a in _BOOL_ATTRS for v in (True, False)]
        checks += [(a, v) for a, values in _CATEGORICAL.items() for v in values]
        checks += [("line", v) for v in _values_present(graph, "line")]
        return _cap(rng, [GraphQuery(task, s, attribute=a, value=v) for s in ids for a, v in checks])
    raise ValueError(f"unsupported v3 task {task}")


def stratum(graph: AttributedGraph, query: GraphQuery) -> str:
    """Prompt-visible features an answer must not correlate with: whether the two named stations
    share a line, and which attribute the question is about."""
    nodes = graph.node_map()
    parts: list[str] = []
    if query.target is not None and query.target in nodes:
        same = nodes[query.source].attributes.get("line") == nodes[query.target].attributes.get("line")
        parts.append("same_line" if same else "diff_line")
    if query.attribute:
        parts.append(str(query.attribute))
    return "|".join(parts)


class BalancedSampler:
    """Greedy quota sampler over (stratum, answer) cells: always draw from the rarest available
    cell, so answers are balanced within every prompt-visible stratum, not only overall."""

    def __init__(self) -> None:
        self.counts: dict[tuple[str, ...], Counter[tuple[str, str]]] = defaultdict(Counter)

    def choose(
        self,
        key: tuple[str, ...],
        candidates: Sequence[tuple[GraphQuery, str, str]],
        rng: random.Random,
    ) -> tuple[GraphQuery, str]:
        by_cell: dict[tuple[str, str], list[tuple[GraphQuery, str]]] = defaultdict(list)
        for query, answer, cell_stratum in candidates:
            by_cell[(cell_stratum, answer)].append((query, answer))
        seen = self.counts[key]
        rarest = min(seen[cell] for cell in by_cell)
        pool = sorted(cell for cell in by_cell if seen[cell] == rarest)
        cell = rng.choice(pool)
        seen[cell] += 1
        return rng.choice(by_cell[cell])


def _hop(query: GraphQuery, graph: AttributedGraph) -> HopDepth:
    if query.reasoning_type in (ReasoningType.ATTRIBUTE_LOOKUP, ReasoningType.ATTRIBUTE_CHECK):
        return HopDepth.LOCAL
    return _hop_depth(query, graph)


# --------------------------------------------------------------------------- static
def _cells() -> list[tuple[int, DensityBin]]:
    return [(n, d) for n in NODE_COUNTS for d in DENSITIES]


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
    query = replace(query, question=render_question(query, graph))
    return StaticQATuple(
        tuple_id=tuple_id,
        split=split,
        graph=graph,
        query=query,
        answer=answer,
        topology=TopologyFamily.METRO,
        density_bin=density,
        hop_depth=_hop(query, graph),
        scale_bin=scale_bin(len(graph.nodes)),
        metadata={"reasoning_type": task.value, **(extra or {})},
    )


def generate_static_v3(
    *,
    split: str,
    graph_count: int,
    tasks_per_graph: int | None,
    seed: int,
    sampler: BalancedSampler,
    tasks: Sequence[ReasoningType] = STATIC_TASKS,
    progress: bool = True,
) -> list[StaticQATuple]:
    """Independent (graph, query, answer) tuples over the exact (size x density) grid.

    ``tasks_per_graph=None`` asks every task once per graph (evaluation); otherwise tasks are
    cycled so each graph gets that many (training)."""
    offsets = {"train": 0, "validation": 5_000_000, "test": 6_000_000}
    cells = _cells()
    tracker = ProgressTracker("[generate]", graph_count, phase=f"static-v3-{split}", enabled=progress)
    tracker.banner(graphs=graph_count)
    result: list[StaticQATuple] = []
    for index in range(graph_count):
        rng = random.Random(seed + offsets[split] + index)
        node_count, density = cells[index % len(cells)]
        graph = make_metro_graph(rng, split, index, node_count, density)
        chosen = (
            list(tasks)
            if tasks_per_graph is None
            else [tasks[(index + j) % len(tasks)] for j in range(tasks_per_graph)]
        )
        for j, task in enumerate(chosen):
            answered = [
                (query, answer_query(graph, query), stratum(graph, query))
                for query in candidate_queries(graph, task, rng)
            ]
            if not answered:
                continue
            query, answer = sampler.choose((split, task.value, density.value), answered, rng)
            result.append(
                _static_tuple(split, graph, query, answer, f"{split}-static-{index:06d}-{j:02d}", density, task)
            )
        tracker.tick(graph_i=f"{index + 1}/{graph_count}")
    tracker.end(tuples=len(result))
    return result


def generate_counterfactual_pairs_v3(
    *,
    graph_count: int,
    pairs_per_graph: int,
    seed: int,
    sampler: BalancedSampler,
    tasks: Sequence[ReasoningType] = DYNAMIC_TASKS,
    progress: bool = True,
) -> list[StaticQATuple]:
    """Before/after-one-edit training pairs: same question text, different graph, changed answer.

    Because ``status`` is withheld from the prompt text, a pair is only distinguishable through
    the graph itself, which is what trains the graph channel to track state."""
    cells = _cells()
    split = "train"
    tracker = ProgressTracker("[generate]", graph_count, phase="counterfactual-v3", enabled=progress)
    tracker.banner(graphs=graph_count)
    result: list[StaticQATuple] = []
    for index in range(graph_count):
        rng = random.Random(seed + 8_000_000 + index)
        node_count, density = cells[index % len(cells)]
        graph = make_metro_graph(rng, split, 100_000 + index, node_count, density)
        made = 0
        for attempt in range(pairs_per_graph * 6):
            if made >= pairs_per_graph:
                break
            task = tasks[(index + made + attempt) % len(tasks)]
            program = _sample_edit_program(graph, rng, force_noop=False)
            if program.edits[0].operation is EditOperation.NOOP:
                continue
            after = apply_edit_program(graph, program).graph
            changed = []
            for query in candidate_queries(after, task, rng):
                before_answer = answer_query(graph, query)
                after_answer = answer_query(after, query)
                if before_answer != after_answer:
                    changed.append((query, after_answer, before_answer))
            if not changed:
                continue
            key = (split, task.value, density.value, "cf")
            query, after_answer = sampler.choose(
                key, [(q, a, stratum(after, q)) for q, a, _ in changed], rng
            )
            before_answer = next(b for q, a, b in changed if q == query and a == after_answer)
            edited = replace(after, graph_id=f"{graph.graph_id}-cf{made}")
            for label, item_graph, answer in (
                ("before", graph, before_answer),
                ("after", edited, after_answer),
            ):
                result.append(
                    _static_tuple(
                        split, item_graph, query, answer,
                        f"train-cf-{index:06d}-{made:02d}-{label}", density, task,
                        {"counterfactual": label},
                    )
                )
            made += 1
        tracker.tick(graph_i=f"{index + 1}/{graph_count}")
    tracker.end(tuples=len(result))
    return result


# --------------------------------------------------------------------------- sessions
def generate_session_v3(
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
    initial = make_metro_graph(rng, split, index, node_count, density, closed_count=rng.choice((0, 1, 2)))
    current = initial
    turns: list[Turn] = []
    for turn_index in range(turn_count):
        force_noop = rng.random() < NOOP_RATE
        program = _sample_edit_program(current, rng, force_noop=force_noop)
        before = current
        updated = apply_edit_program(before, program).graph
        candidates = []
        for query in candidate_queries(updated, task, rng):
            candidates.append((query, answer_query(updated, query), answer_query(before, query)))
        changing = [c for c in candidates if c[1] != c[2]]
        pool = changing or candidates
        key = (split, task.value, density.value, "dyn")
        query, answer = sampler.choose(key, [(q, a, stratum(updated, q)) for q, a, _ in pool], rng)
        query = replace(query, question=render_question(query, updated))
        stale_answer = answer_query(before, query)
        is_noop = program.edits[0].operation is EditOperation.NOOP
        utterance = (
            "NOOP: no graph update this turn."
            if is_noop
            else " ; ".join(_utterance(edit, before, "direct") for edit in program.edits)
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
                ),
                complexity={
                    "topology": TopologyFamily.METRO.value,
                    "density": density.value,
                    "target_density": density.value,
                    "scale_bin": scale_bin(node_count),
                    "node_count": node_count,
                    "hop_depth": _hop(query, updated).value,
                    "edit_count": len(program.edits),
                    "ood": False,
                    "session_length": turn_count,
                },
            )
        )
        current = updated
    return Session(
        session_id=f"{split}-v3-{index:06d}",
        split=split,
        initial_graph=initial,
        turns=tuple(turns),
        seed=seed,
        paraphrase_family=f"{split}-{TopologyFamily.METRO.value}",
    )


def generate_sessions_v3(
    *,
    replicates_per_split: dict[str, int],
    seed: int,
    tasks: Sequence[ReasoningType] = DYNAMIC_TASKS,
    progress: bool = True,
) -> dict[str, list[Session]]:
    """Exact grid: (node count x density x session length x task), ``replicates`` passes each."""
    offsets = {"validation": 1_000_000, "test": 2_000_000, "train": 0}
    cells = [
        (n, d, length, task)
        for n in NODE_COUNTS
        for d in DENSITIES
        for length in SESSION_LENGTHS
        for task in tasks
    ]
    total = sum(count * len(cells) for count in replicates_per_split.values())
    tracker = ProgressTracker("[generate]", total, phase="sessions-v3", enabled=progress)
    tracker.banner(sessions=total, cells=len(cells))
    result: dict[str, list[Session]] = {}
    for split, replicates in replicates_per_split.items():
        sampler = BalancedSampler()
        sessions: list[Session] = []
        index = 0
        for _ in range(replicates):
            for node_count, density, length, task in cells:
                sessions.append(
                    generate_session_v3(
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


# --------------------------------------------------------------------------- audits
def label_audit(tuples: Sequence[StaticQATuple]) -> dict[str, Any]:
    """Per-task answer distribution and majority-class accuracy."""
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for item in tuples:
        by_task[item.query.reasoning_type.value][item.answer] += 1
    report: dict[str, Any] = {}
    total = majority_total = 0
    for task, counts in sorted(by_task.items()):
        n = sum(counts.values())
        top = counts.most_common(1)[0][1]
        report[task] = {
            "n": n,
            "majority_accuracy": top / n,
            "top_answers": counts.most_common(6),
        }
        total += n
        majority_total += top
    report["_overall_majority_accuracy"] = majority_total / max(1, total)
    return report


def session_label_audit(sessions: Sequence[Session]) -> dict[str, Any]:
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    changed = total = 0
    for session in sessions:
        for turn in session.turns:
            by_task[turn.query.reasoning_type.value][turn.gold_answer] += 1
            total += 1
            changed += turn.gold_answer != turn.stale_answer
    report: dict[str, Any] = {}
    n_all = maj_all = 0
    for task, counts in sorted(by_task.items()):
        n = sum(counts.values())
        top = counts.most_common(1)[0][1]
        report[task] = {"n": n, "majority_accuracy": top / n, "top_answers": counts.most_common(6)}
        n_all += n
        maj_all += top
    report["_overall_majority_accuracy"] = maj_all / max(1, n_all)
    report["_answer_changing_fraction"] = changed / max(1, total)
    return report


def graph_audit(graphs: Iterable[AttributedGraph]) -> dict[str, Any]:
    sizes: Counter[int] = Counter()
    diameters: dict[str, list[int]] = defaultdict(list)
    seen: set[str] = set()
    for graph in graphs:
        if graph.graph_id in seen:
            continue
        seen.add(graph.graph_id)
        sizes[len(graph.nodes)] += 1
        diameters[str(graph.metadata.get("density"))].append(int(graph.metadata.get("diameter", -1)))
    return {
        "graphs": len(seen),
        "node_count_histogram": dict(sorted(sizes.items())),
        "diameter_by_density": {
            k: {"mean": sum(v) / len(v), "max": max(v)} for k, v in diameters.items()
        },
    }


def _text_features(item: StaticQATuple) -> dict[str, Any]:
    """Everything a text-only model can see in the W(f_i) prompt, as categorical features."""
    from graph_modi.graph.solvers import wfi_excluded_attributes

    graph, query = item.graph, item.query
    hidden = wfi_excluded_attributes(query, graph)
    nodes = graph.node_map()
    features: dict[str, Any] = {
        "q.attribute": query.attribute,
        "q.value": query.value,
        "q.hops": query.hops,
    }
    for role, node_id in (("src", query.source), ("tgt", query.target)):
        if node_id and node_id in nodes:
            for key, value in nodes[node_id].attributes.items():
                if key not in hidden:
                    features[f"{role}.{key}"] = value
    if "src.line" in features and "tgt.line" in features:
        features["same_line"] = features["src.line"] == features["tgt.line"]
    return features


def _rule_accuracy(
    fit: Sequence[tuple[dict[str, Any], str]],
    score: Sequence[tuple[dict[str, Any], str]],
    combo: tuple[str, ...],
) -> float:
    table: dict[tuple, Counter[str]] = defaultdict(Counter)
    for feats, answer in fit:
        table[tuple(feats.get(name) for name in combo)][answer] += 1
    rule = {key: counts.most_common(1)[0][0] for key, counts in table.items()}
    hits = sum(rule.get(tuple(feats.get(name) for name in combo)) == answer for feats, answer in score)
    return hits / max(1, len(score))


def text_leak_audit(
    train: Sequence[StaticQATuple], evaluate: Sequence[StaticQATuple]
) -> dict[str, Any]:
    """Can the prompt text alone predict the answer beyond what the question type gives away?

    Per task: the best rule over prompt-visible feature combinations is chosen by 2-fold
    cross-validation on ``train`` (no peeking at ``evaluate``) and then scored once on
    ``evaluate``. It is compared with the best rule that uses only the question's own fields
    (attribute / value / hops), i.e. the prior a model gets from reading the question."""
    def group(items: Sequence[StaticQATuple]) -> dict[str, list[tuple[dict[str, Any], str]]]:
        out: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
        for item in items:
            out[item.query.reasoning_type.value].append((_text_features(item), item.answer))
        return out

    train_by, eval_by = group(train), group(evaluate)
    report: dict[str, Any] = {}
    for task, rows in sorted(eval_by.items()):
        fit_rows = train_by.get(task, [])
        names = sorted({name for feats, _ in fit_rows for name in feats})
        question_names = [n for n in names if n.startswith("q.")]
        node_names = [n for n in names if not n.startswith("q.")]
        half = len(fit_rows) // 2
        folds = ((fit_rows[:half], fit_rows[half:]), (fit_rows[half:], fit_rows[:half]))

        def best_combo(combos: list[tuple[str, ...]]) -> tuple[str, ...]:
            def cv(combo: tuple[str, ...]) -> float:
                return sum(_rule_accuracy(a, b, combo) for a, b in folds) / 2

            return max(combos, key=cv)

        question_combos = [(), *[(n,) for n in question_names]]
        if len(question_names) > 1:
            question_combos.append(tuple(question_names))
        node_combos = [tuple(question_names) + (n,) for n in node_names] + [
            tuple(question_names) + (a, b) for i, a in enumerate(node_names) for b in node_names[i + 1 :]
        ]
        q_combo = best_combo(question_combos)
        n_combo = best_combo(node_combos) if node_combos else q_combo
        q_acc = _rule_accuracy(fit_rows, rows, q_combo)
        n_acc = _rule_accuracy(fit_rows, rows, n_combo)
        majority = Counter(answer for _, answer in rows).most_common(1)[0][1] / len(rows)
        report[task] = {
            "majority_accuracy": majority,
            "question_only_rule_accuracy": q_acc,
            "with_node_text_rule_accuracy": n_acc,
            "node_text_rule_features": "+".join(n_combo),
            "leak_flag": n_acc > q_acc + 0.05,
        }
    return report
