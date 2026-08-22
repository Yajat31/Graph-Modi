"""Solver-verified, in-distribution cumulative metro sessions."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

from graph_modi.graph.executor import apply_edit, graph_fingerprint
from graph_modi.graph.solvers import answer_query, render_question
from graph_modi.schema import (
    AttributedGraph,
    Edge,
    EditOperation,
    EditTarget,
    GraphEdit,
    GraphQuery,
    Node,
    ReasoningType,
    Session,
    Turn,
)

_LINES = ("red", "blue", "green", "yellow")

DISTRIBUTION = "watts_strogatz_metro_v1"
DEFAULT_GRAPH_DEGREE = 4
DEFAULT_REWIRE_PROBABILITY = 0.15
DEFAULT_LINE_COUNT = 4
TRACK_RELATION = "track"
TRANSFER_RELATION = "transfer"
_CONNECTIVITY_ATTEMPTS = 32


def _pair(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _watts_strogatz_positions(
    rng: random.Random,
    node_count: int,
    degree: int,
    rewire_probability: float,
) -> dict[tuple[int, int], bool]:
    """Build a rewired ring lattice over ring positions.

    Returns each position pair mapped to whether it was rewired, so callers can
    distinguish surviving ring track from rewired transfer links. Edge count is
    always ``node_count * degree // 2``: a rewire removes and adds one edge.
    """
    edges: dict[tuple[int, int], bool] = {
        _pair(position, (position + offset) % node_count): False
        for position in range(node_count)
        for offset in range(1, degree // 2 + 1)
    }
    neighbors: dict[int, set[int]] = {position: set() for position in range(node_count)}
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    for left, right in list(edges):
        if rng.random() >= rewire_probability:
            continue
        options = [
            candidate
            for candidate in range(node_count)
            if candidate != left and candidate not in neighbors[left]
        ]
        if not options:
            continue
        target = rng.choice(options)
        del edges[_pair(left, right)]
        neighbors[left].discard(right)
        neighbors[right].discard(left)
        edges[_pair(left, target)] = True
        neighbors[left].add(target)
        neighbors[target].add(left)
    return edges


def _is_connected(node_count: int, position_edges: Iterable[tuple[int, int]]) -> bool:
    adjacency: dict[int, list[int]] = {position: [] for position in range(node_count)}
    for left, right in position_edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    seen = {0}
    frontier = [0]
    while frontier:
        position = frontier.pop()
        for neighbor in adjacency[position]:
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return len(seen) == node_count


def _make_graph(
    rng: random.Random,
    split: str,
    index: int,
    node_count: int,
    *,
    degree: int = DEFAULT_GRAPH_DEGREE,
    rewire_probability: float = DEFAULT_REWIRE_PROBABILITY,
    line_count: int = DEFAULT_LINE_COUNT,
) -> AttributedGraph:
    if degree < 2 or degree % 2:
        raise ValueError("graph_degree must be an even number of at least 2")
    if degree >= node_count:
        raise ValueError(f"graph_degree {degree} requires more than {degree} nodes")
    if not 0.0 <= rewire_probability <= 1.0:
        raise ValueError("rewire_probability must be in [0, 1]")
    if not 1 <= line_count <= len(_LINES):
        raise ValueError(f"line_count must be between 1 and {len(_LINES)}")

    for _ in range(_CONNECTIVITY_ATTEMPTS):
        position_edges = _watts_strogatz_positions(rng, node_count, degree, rewire_probability)
        if _is_connected(node_count, position_edges):
            break
    else:
        raise RuntimeError(
            f"Could not draw a connected graph for {split}/{index} after "
            f"{_CONNECTIVITY_ATTEMPTS} attempts"
        )

    # Ring positions are permuted onto node identifiers so that adjacency cannot
    # be recovered from node id or label arithmetic.
    identifiers = list(range(node_count))
    rng.shuffle(identifiers)
    lines = {
        identifiers[position]: _LINES[(position * line_count) // node_count]
        for position in range(node_count)
    }

    nodes = tuple(
        Node(
            id=f"n{node_index}",
            label=f"{split.title()}Station{index:06d}_{node_index:02d}",
            attributes={
                "status": "open",
                "accessible": bool(rng.getrandbits(1)),
                "line": lines[node_index],
            },
        )
        for node_index in range(node_count)
    )
    edges = tuple(
        Edge(
            source=f"n{identifiers[left]}",
            target=f"n{identifiers[right]}",
            relation=TRANSFER_RELATION if rewired else TRACK_RELATION,
        )
        for (left, right), rewired in sorted(position_edges.items())
    )
    return AttributedGraph(
        graph_id=f"{split}-graph-{index:06d}",
        nodes=nodes,
        edges=edges,
        metadata={
            "distribution": DISTRIBUTION,
            "split": split,
            "degree": degree,
            "rewire_probability": rewire_probability,
            "line_count": line_count,
        },
    )


def _candidate_edit(
    graph: AttributedGraph,
    operation: EditOperation,
    rng: random.Random,
) -> GraphEdit | None:
    node_ids = [node.id for node in graph.nodes]
    if operation is EditOperation.SET:
        node = rng.choice(graph.nodes)
        current = node.attributes.get("status", "open")
        return GraphEdit(
            operation=operation,
            target=EditTarget.NODE,
            node_id=node.id,
            attribute="status",
            value="closed" if current == "open" else "open",
        )
    if operation is EditOperation.ADD:
        existing = {frozenset((edge.source, edge.target)) for edge in graph.edges}
        candidates = [
            (left, right)
            for left_index, left in enumerate(node_ids)
            for right in node_ids[left_index + 1 :]
            if frozenset((left, right)) not in existing
        ]
        if not candidates:
            return None
        source, destination = rng.choice(candidates)
        return GraphEdit(
            operation=operation,
            target=EditTarget.EDGE,
            source=source,
            destination=destination,
            relation=TRANSFER_RELATION,
        )
    if operation is EditOperation.DEL and graph.edges:
        edge = rng.choice(graph.edges)
        return GraphEdit(
            operation=operation,
            target=EditTarget.EDGE,
            source=edge.source,
            destination=edge.target,
            relation=edge.relation,
        )
    return None


def _queries(graph: AttributedGraph, reasoning_type: ReasoningType) -> Iterable[GraphQuery]:
    nodes = [node.id for node in graph.nodes]
    if reasoning_type in {
        ReasoningType.SHORTEST_PATH,
        ReasoningType.REACHABILITY,
        ReasoningType.FILTERED_PATH_COUNT,
        ReasoningType.EDGE_EXISTS,
    }:
        for left_index, source in enumerate(nodes):
            for target in nodes[left_index + 1 :]:
                if reasoning_type is ReasoningType.FILTERED_PATH_COUNT:
                    yield GraphQuery(
                        reasoning_type,
                        source,
                        target,
                        attribute="accessible",
                        value=True,
                    )
                else:
                    yield GraphQuery(reasoning_type, source, target)
        return
    if reasoning_type is ReasoningType.FILTERED_NEIGHBOR_COUNT:
        lines = sorted(
            {str(node.attributes["line"]) for node in graph.nodes if "line" in node.attributes}
        )
        for source in nodes:
            yield GraphQuery(
                reasoning_type,
                source,
                attribute="status",
                value="open",
            )
            yield GraphQuery(
                reasoning_type,
                source,
                attribute="accessible",
                value=True,
            )
            for line in lines:
                yield GraphQuery(
                    reasoning_type,
                    source,
                    attribute="line",
                    value=line,
                )
        return
    if reasoning_type is ReasoningType.CYCLE_MEMBERSHIP:
        for source in nodes:
            yield GraphQuery(reasoning_type, source)


_DEFAULT_CHANGED_QUERY_FALLBACK = (
    ReasoningType.SHORTEST_PATH,
    ReasoningType.REACHABILITY,
    ReasoningType.FILTERED_NEIGHBOR_COUNT,
    ReasoningType.FILTERED_PATH_COUNT,
    ReasoningType.CYCLE_MEMBERSHIP,
)


def _changed_query(
    before: AttributedGraph,
    after: AttributedGraph,
    preferred: ReasoningType,
    rng: random.Random,
    *,
    fallback_order: Sequence[ReasoningType] | None = None,
) -> GraphQuery | None:
    """Find an answer-changing query, trying ``preferred`` first.

    ``fallback_order`` lets a caller restrict which reasoning types this may
    fall back to when ``preferred`` has no answer-changing candidate; None
    preserves the original fixed fallback list unchanged.
    """
    order = fallback_order if fallback_order is not None else _DEFAULT_CHANGED_QUERY_FALLBACK
    reasoning_order = [preferred, *order]
    seen: set[ReasoningType] = set()
    for reasoning_type in reasoning_order:
        if reasoning_type in seen:
            continue
        seen.add(reasoning_type)
        candidates = list(_queries(before, reasoning_type))
        rng.shuffle(candidates)
        for query in candidates:
            if answer_query(before, query) != answer_query(after, query):
                return query
    return None


def _utterance(edit: GraphEdit, graph: AttributedGraph, family: str) -> str:
    labels = {node.id: node.label for node in graph.nodes}
    if edit.operation is EditOperation.SET and edit.node_id:
        label = labels[edit.node_id]
        if edit.value == "closed":
            variants = {
                "direct": f"{label} is closed now.",
                "service": f"Service at {label} has been suspended.",
                "notice": f"Please mark {label} as closed.",
            }
        else:
            variants = {
                "direct": f"{label} has reopened.",
                "service": f"Service at {label} has resumed.",
                "notice": f"Please mark {label} as open.",
            }
        return variants[family]
    if edit.target is EditTarget.EDGE and edit.source and edit.destination:
        action = "ADD" if edit.operation is EditOperation.ADD else "DEL"
        return f"{action} EDGE {labels[edit.source]} {labels[edit.destination]} {edit.relation}"
    return edit.canonical()


def generate_session(
    *,
    split: str,
    index: int,
    seed: int,
    node_count: int,
    turn_count: int,
    reasoning_types: Sequence[ReasoningType],
    degree: int = DEFAULT_GRAPH_DEGREE,
    rewire_probability: float = DEFAULT_REWIRE_PROBABILITY,
    line_count: int = DEFAULT_LINE_COUNT,
) -> Session:
    rng = random.Random(seed)
    initial = _make_graph(
        rng,
        split,
        index,
        node_count,
        degree=degree,
        rewire_probability=rewire_probability,
        line_count=line_count,
    )
    current = initial
    turns: list[Turn] = []
    families = ("direct", "service", "notice")
    family = families[index % len(families)]
    operations = (EditOperation.SET, EditOperation.ADD, EditOperation.DEL)
    for turn_index in range(turn_count):
        preferred = reasoning_types[turn_index % len(reasoning_types)]
        chosen: tuple[GraphEdit, AttributedGraph, GraphQuery] | None = None
        for attempt in range(250):
            operation = operations[(index + turn_index + attempt) % len(operations)]
            edit = _candidate_edit(current, operation, rng)
            if edit is None:
                continue
            result = apply_edit(current, edit)
            if not result.applied:
                continue
            query = _changed_query(current, result.graph, preferred, rng)
            if query is not None:
                chosen = edit, result.graph, query
                break
        if chosen is None:
            raise RuntimeError(
                f"Could not generate changed-answer turn {turn_index} for {split}/{index}"
            )
        edit, updated, query = chosen
        query = replace(query, question=render_question(query, updated))
        stale_answer = answer_query(current, query)
        answer = answer_query(updated, query)
        utterance = _utterance(edit, current, family)
        utterance_tokens = set(re.findall(r"\b[\w-]+\b", utterance.casefold()))
        if answer.casefold() in utterance_tokens:
            raise AssertionError("Revision text leaked the answer")
        turns.append(
            Turn(
                turn_index=turn_index,
                utterance=utterance,
                gold_edit=edit,
                query=query,
                gold_answer=answer,
                stale_answer=stale_answer,
                before_fingerprint=graph_fingerprint(current),
                after_fingerprint=graph_fingerprint(updated),
                tags=(edit.operation.value.lower(), query.reasoning_type.value),
            )
        )
        current = updated
    return Session(
        session_id=f"{split}-session-{index:06d}",
        split=split,
        initial_graph=initial,
        turns=tuple(turns),
        seed=seed,
        paraphrase_family=f"{split}-{family}",
    )


def generate_dataset(
    *,
    counts: dict[str, int],
    seed: int,
    node_count_min: int,
    node_count_max: int,
    turns: Sequence[int],
    reasoning_types: Sequence[str],
    distribution: str = DISTRIBUTION,
    degree: int = DEFAULT_GRAPH_DEGREE,
    rewire_probability: float = DEFAULT_REWIRE_PROBABILITY,
    line_count: int = DEFAULT_LINE_COUNT,
) -> dict[str, list[Session]]:
    if distribution != DISTRIBUTION:
        raise ValueError(
            f"Unsupported graph distribution {distribution!r}; this generator samples "
            f"{DISTRIBUTION!r}"
        )
    parsed_types = tuple(ReasoningType(value) for value in reasoning_types)
    if not parsed_types:
        raise ValueError("At least one reasoning type is required")
    if not turns or min(turns) < 1:
        raise ValueError("Turn counts must be positive")
    if node_count_min <= degree:
        raise ValueError(f"node_count_min {node_count_min} must exceed graph_degree {degree}")
    result: dict[str, list[Session]] = {}
    split_offsets = {"train": 0, "validation": 1_000_000, "test": 2_000_000}
    for split, count in counts.items():
        offset = split_offsets.get(split, 3_000_000)
        sessions = []
        for index in range(count):
            session_seed = seed + offset + index
            rng = random.Random(session_seed)
            sessions.append(
                generate_session(
                    split=split,
                    index=index,
                    seed=session_seed,
                    node_count=rng.randint(node_count_min, node_count_max),
                    turn_count=turns[index % len(turns)],
                    reasoning_types=parsed_types,
                    degree=degree,
                    rewire_probability=rewire_probability,
                    line_count=line_count,
                )
            )
        result[split] = sessions
    assert_disjoint_splits(result)
    return result


def assert_disjoint_splits(dataset: dict[str, list[Session]]) -> None:
    seen_graphs: set[str] = set()
    seen_sessions: set[str] = set()
    seen_labels: set[str] = set()
    seen_families: set[str] = set()
    for split, sessions in dataset.items():
        graph_ids = {session.initial_graph.graph_id for session in sessions}
        session_ids = {session.session_id for session in sessions}
        labels = {node.label for session in sessions for node in session.initial_graph.nodes}
        families = {session.paraphrase_family for session in sessions}
        if seen_graphs & graph_ids or seen_sessions & session_ids or seen_labels & labels:
            raise ValueError(f"Data leakage detected in split {split}")
        if seen_families & families:
            raise ValueError(f"Paraphrase-family leakage detected in split {split}")
        seen_graphs.update(graph_ids)
        seen_sessions.update(session_ids)
        seen_labels.update(labels)
        seen_families.update(families)


def save_sessions(path: str | Path, sessions: Sequence[Session]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for session in sessions:
            handle.write(json.dumps(session.to_dict(), sort_keys=True) + "\n")


def load_sessions(path: str | Path) -> list[Session]:
    with Path(path).open(encoding="utf-8") as handle:
        return [Session.from_dict(json.loads(line)) for line in handle if line.strip()]
