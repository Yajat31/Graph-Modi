"""Preliminary dataset v2: multi-topology, multi-edit, NOOP, static QA corpus."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

from graph_modi.data.multiturn import (
    DEFAULT_GRAPH_DEGREE,
    DEFAULT_LINE_COUNT,
    DEFAULT_REWIRE_PROBABILITY,
    TRACK_RELATION,
    TRANSFER_RELATION,
    _candidate_edit,
    _changed_query,
    _is_connected,
    _pair,
    _queries,
    _utterance,
    assert_disjoint_splits,
)
from graph_modi.data.multiturn import (
    _make_graph as _make_ws_graph,
)
from graph_modi.graph.executor import apply_edit, apply_edit_program, graph_fingerprint
from graph_modi.graph.solvers import answer_query, render_question
from graph_modi.schema import (
    AttributedGraph,
    DensityBin,
    Edge,
    EditOperation,
    EditProgram,
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

DISTRIBUTION_V2 = "metro_v2_preliminary"
NOOP_RATE = 0.17
MAX_EDITS_PER_TURN = 3

_ID_SCALE = ((16, 24), (25, 32))
_OOD_SCALE = (40, 48)
_ID_TURNS = (1, 2, 4)
_OOD_TURNS = (8,)

_STATIC_TASKS = (
    ReasoningType.EDGE_EXISTS,
    ReasoningType.NODE_DEGREE,
    ReasoningType.REACHABILITY,
    ReasoningType.CYCLE_MEMBERSHIP,
    ReasoningType.FILTERED_NEIGHBOR_COUNT,
    ReasoningType.FILTERED_PATH_COUNT,
    ReasoningType.SHORTEST_PATH,
    ReasoningType.PATH_COST,
)

_DYNAMIC_TASKS = (
    ReasoningType.SHORTEST_PATH,
    ReasoningType.REACHABILITY,
    ReasoningType.FILTERED_NEIGHBOR_COUNT,
    ReasoningType.EDGE_EXISTS,
    ReasoningType.PATH_COST,
)


def _density_bin(node_count: int, edge_count: int) -> DensityBin:
    max_edges = node_count * (node_count - 1) // 2
    ratio = edge_count / max(1, max_edges)
    if ratio < 0.08:
        return DensityBin.SPARSE
    if ratio < 0.18:
        return DensityBin.MEDIUM
    return DensityBin.DENSE


def _scale_bin(node_count: int, *, ood: bool) -> str:
    if ood:
        return "scale_ood"
    if node_count <= 24:
        return "scale_small"
    return "scale_medium"


def _hop_depth(query: GraphQuery, graph: AttributedGraph) -> HopDepth:
    if query.reasoning_type in {
        ReasoningType.NODE_DEGREE,
        ReasoningType.FILTERED_NEIGHBOR_COUNT,
        ReasoningType.EDGE_EXISTS,
    }:
        return HopDepth.LOCAL
    if query.reasoning_type in {ReasoningType.SHORTEST_PATH, ReasoningType.PATH_COST}:
        if query.target is None:
            return HopDepth.MID
        path = answer_query(graph, query)
        if path == "unreachable":
            return HopDepth.GLOBAL
        try:
            hops = int(path) if query.reasoning_type is ReasoningType.SHORTEST_PATH else 3
        except ValueError:
            hops = 3
        if hops <= 2:
            return HopDepth.LOCAL
        if hops <= 4:
            return HopDepth.MID
        return HopDepth.GLOBAL
    if query.reasoning_type is ReasoningType.CYCLE_MEMBERSHIP:
        return HopDepth.GLOBAL
    return HopDepth.MID


def _sbm_graph(
    rng: random.Random,
    split: str,
    index: int,
    node_count: int,
    *,
    blocks: int = 3,
    p_in: float = 0.35,
    p_out: float = 0.08,
    line_count: int = DEFAULT_LINE_COUNT,
) -> AttributedGraph:
    from graph_modi.data.multiturn import _LINES

    block_size = node_count // blocks
    remainder = node_count % blocks
    sizes = [block_size + (1 if block < remainder else 0) for block in range(blocks)]
    labels = [block for block, size in enumerate(sizes) for _ in range(size)]
    rng.shuffle(labels)
    position_edges: dict[tuple[int, int], bool] = {}
    for left in range(node_count):
        for right in range(left + 1, node_count):
            probability = p_in if labels[left] == labels[right] else p_out
            if rng.random() < probability:
                position_edges[_pair(left, right)] = labels[left] != labels[right]
    if not _is_connected(node_count, position_edges):
        for offset in range(1, node_count):
            position_edges.setdefault(_pair(0, offset), False)
    identifiers = list(range(node_count))
    rng.shuffle(identifiers)
    lines = {
        identifiers[position]: _LINES[(position * line_count) // node_count]
        for position in range(node_count)
    }
    nodes = tuple(
        Node(
            id=f"n{node_index}",
            label=f"{split.title()}Hub{index:06d}_{node_index:02d}",
            attributes={
                "status": "open",
                "accessible": bool(rng.getrandbits(1)),
                "line": lines[node_index],
                "block": int(labels[node_index]),
            },
        )
        for node_index in range(node_count)
    )
    edges = tuple(
        Edge(
            source=f"n{identifiers[left]}",
            target=f"n{identifiers[right]}",
            relation=TRANSFER_RELATION if rewired else TRACK_RELATION,
            weight=float(rng.randint(1, 5)),
        )
        for (left, right), rewired in sorted(position_edges.items())
    )
    return AttributedGraph(
        graph_id=f"{split}-sbm-{index:06d}",
        nodes=nodes,
        edges=edges,
        metadata={
            "distribution": DISTRIBUTION_V2,
            "topology": TopologyFamily.SBM.value,
            "split": split,
            "blocks": blocks,
            "p_in": p_in,
            "p_out": p_out,
        },
    )


def _erdos_renyi_graph(
    rng: random.Random,
    split: str,
    index: int,
    node_count: int,
    *,
    edge_probability: float = 0.12,
    line_count: int = DEFAULT_LINE_COUNT,
) -> AttributedGraph:
    from graph_modi.data.multiturn import _LINES

    position_edges: dict[tuple[int, int], bool] = {}
    for left in range(node_count):
        for right in range(left + 1, node_count):
            if rng.random() < edge_probability:
                position_edges[_pair(left, right)] = True
    if not _is_connected(node_count, position_edges):
        for offset in range(1, node_count):
            position_edges.setdefault(_pair(0, offset), False)
    identifiers = list(range(node_count))
    rng.shuffle(identifiers)
    lines = {
        identifiers[position]: _LINES[(position * line_count) // node_count]
        for position in range(node_count)
    }
    nodes = tuple(
        Node(
            id=f"n{node_index}",
            label=f"{split.title()}Rand{index:06d}_{node_index:02d}",
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
            weight=float(rng.randint(1, 5)),
        )
        for (left, right), rewired in sorted(position_edges.items())
    )
    return AttributedGraph(
        graph_id=f"{split}-er-{index:06d}",
        nodes=nodes,
        edges=edges,
        metadata={
            "distribution": DISTRIBUTION_V2,
            "topology": TopologyFamily.ERDOS_RENYI.value,
            "split": split,
            "edge_probability": edge_probability,
        },
    )


def make_graph(
    rng: random.Random,
    split: str,
    index: int,
    node_count: int,
    topology: TopologyFamily,
    *,
    degree: int = DEFAULT_GRAPH_DEGREE,
    rewire_probability: float = DEFAULT_REWIRE_PROBABILITY,
    line_count: int = DEFAULT_LINE_COUNT,
) -> AttributedGraph:
    if topology is TopologyFamily.WATTS_STROGATZ:
        graph = _make_ws_graph(
            rng,
            split,
            index,
            node_count,
            degree=degree,
            rewire_probability=rewire_probability,
            line_count=line_count,
        )
        return replace(
            graph,
            metadata={
                **graph.metadata,
                "distribution": DISTRIBUTION_V2,
                "topology": topology.value,
            },
        )
    if topology is TopologyFamily.SBM:
        density = _density_bin(node_count, node_count * degree // 2)
        p_in = {"sparse": 0.25, "medium": 0.35, "dense": 0.5}[density.value]
        p_out = {"sparse": 0.04, "medium": 0.08, "dense": 0.12}[density.value]
        return _sbm_graph(
            rng, split, index, node_count, p_in=p_in, p_out=p_out, line_count=line_count
        )
    if topology is TopologyFamily.ERDOS_RENYI:
        return _erdos_renyi_graph(rng, split, index, node_count, line_count=line_count)
    raise ValueError(f"Unsupported topology: {topology}")


def _noop_utterance() -> str:
    return "NOOP: no graph update this turn."


def _sample_edit_program(
    graph: AttributedGraph,
    rng: random.Random,
    *,
    force_noop: bool = False,
) -> EditProgram:
    if force_noop:
        return EditProgram(edits=(GraphEdit(operation=EditOperation.NOOP, reason="explicit"),))
    edit_count = rng.randint(0, MAX_EDITS_PER_TURN)
    if edit_count == 0:
        return EditProgram(edits=(GraphEdit(operation=EditOperation.NOOP, reason="zero_ops"),))
    operations = (EditOperation.SET, EditOperation.ADD, EditOperation.DEL)
    edits: list[GraphEdit] = []
    current = graph
    for _ in range(edit_count):
        for _attempt in range(32):
            operation = rng.choice(operations)
            edit = _candidate_edit(current, operation, rng)
            if edit is None:
                continue
            result = apply_edit(current, edit)
            if result.applied:
                edits.append(edit)
                current = result.graph
                break
        else:
            break
    if not edits:
        return EditProgram(edits=(GraphEdit(operation=EditOperation.NOOP, reason="fallback"),))
    return EditProgram(edits=tuple(edits))


def generate_session_v2(
    *,
    split: str,
    index: int,
    seed: int,
    node_count: int,
    turn_count: int,
    topology: TopologyFamily,
    reasoning_types: Sequence[ReasoningType],
    ood: bool = False,
    degree: int = DEFAULT_GRAPH_DEGREE,
    rewire_probability: float = DEFAULT_REWIRE_PROBABILITY,
    line_count: int = DEFAULT_LINE_COUNT,
    fallback_tasks: Sequence[ReasoningType] | None = None,
) -> Session:
    rng = random.Random(seed)
    initial = make_graph(
        rng,
        split,
        index,
        node_count,
        topology,
        degree=degree,
        rewire_probability=rewire_probability,
        line_count=line_count,
    )
    current = initial
    turns: list[Turn] = []
    family = "direct"
    density = _density_bin(len(initial.nodes), len(initial.edges))
    scale = _scale_bin(node_count, ood=ood)
    for turn_index in range(turn_count):
        preferred = reasoning_types[turn_index % len(reasoning_types)]
        force_noop = rng.random() < NOOP_RATE
        program = _sample_edit_program(current, rng, force_noop=force_noop)
        before = current
        result = apply_edit_program(current, program)
        updated = result.graph
        query = _changed_query(before, updated, preferred, rng, fallback_order=fallback_tasks)
        if query is None:
            candidates = list(_queries(updated, preferred))
            rng.shuffle(candidates)
            query = (
                candidates[0]
                if candidates
                else GraphQuery(ReasoningType.NODE_COUNT, source=updated.nodes[0].id)
            )
        query = replace(query, question=render_question(query, updated))
        stale_answer = answer_query(before, query)
        answer = answer_query(updated, query)
        if program.edits and program.edits[0].operation is EditOperation.NOOP:
            utterance = _noop_utterance()
            primary_edit = program.edits[0]
        else:
            utterance = " ; ".join(_utterance(edit, before, family) for edit in program.edits)
            primary_edit = program.edits[0]
        hop = _hop_depth(query, updated)
        turns.append(
            Turn(
                turn_index=turn_index,
                utterance=utterance,
                gold_edit=primary_edit,
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
                    topology.value,
                    "noop" if primary_edit.operation is EditOperation.NOOP else "edit",
                ),
                complexity={
                    "topology": topology.value,
                    "density": density.value,
                    "scale_bin": scale,
                    "hop_depth": hop.value,
                    "edit_count": len(program.edits),
                    "ood": ood,
                },
            )
        )
        current = updated
    return Session(
        session_id=f"{split}-v2-{index:06d}",
        split=split,
        initial_graph=initial,
        turns=tuple(turns),
        seed=seed,
        paraphrase_family=f"{split}-{topology.value}",
    )


def _static_queries(graph: AttributedGraph, reasoning_type: ReasoningType) -> Iterable[GraphQuery]:
    yield from _queries(graph, reasoning_type)
    if reasoning_type is ReasoningType.NODE_COUNT:
        yield GraphQuery(reasoning_type, source=graph.nodes[0].id)
    if reasoning_type is ReasoningType.PATH_COST:
        nodes = [node.id for node in graph.nodes]
        for left_index, source in enumerate(nodes):
            for target in nodes[left_index + 1 :]:
                yield GraphQuery(reasoning_type, source, target)


def generate_static_corpus(
    *,
    graph_counts: dict[str, int],
    tuples_per_graph: int,
    seed: int,
    node_count_ranges: dict[str, tuple[int, int]],
    topologies: Sequence[TopologyFamily],
    tasks: Sequence[ReasoningType] | None = None,
    progress: bool = True,
) -> dict[str, list[StaticQATuple]]:
    static_tasks = tuple(tasks) if tasks else _STATIC_TASKS
    total_graphs = sum(graph_counts.values())
    tracker = ProgressTracker("[generate]", total_graphs, phase="static", enabled=progress)
    tracker.banner(graphs=total_graphs, tuples_per_graph=tuples_per_graph)
    result: dict[str, list[StaticQATuple]] = {split: [] for split in graph_counts}
    graph_index = 0
    split_offsets = {"train": 0, "validation": 5_000_000, "test": 6_000_000}
    for split, count in graph_counts.items():
        offset = split_offsets.get(split, 7_000_000)
        low, high = node_count_ranges.get(split, (16, 32))
        for index in range(count):
            graph_seed = seed + offset + index
            rng = random.Random(graph_seed)
            node_count = rng.randint(low, high)
            topology = topologies[index % len(topologies)]
            graph = make_graph(rng, split, index, node_count, topology)
            density = _density_bin(len(graph.nodes), len(graph.edges))
            scale = _scale_bin(node_count, ood=node_count >= 40)
            for tuple_index in range(tuples_per_graph):
                reasoning_type = static_tasks[(index + tuple_index) % len(static_tasks)]
                candidates = list(_static_queries(graph, reasoning_type))
                rng.shuffle(candidates)
                if not candidates:
                    continue
                query = candidates[tuple_index % len(candidates)]
                answer = answer_query(graph, query)
                query = replace(query, question=render_question(query, graph))
                hop = _hop_depth(query, graph)
                result[split].append(
                    StaticQATuple(
                        tuple_id=f"{split}-static-{index:06d}-{tuple_index:02d}",
                        split=split,
                        graph=graph,
                        query=query,
                        answer=answer,
                        topology=topology,
                        density_bin=density,
                        hop_depth=hop,
                        scale_bin=scale,
                        metadata={"reasoning_type": reasoning_type.value},
                    )
                )
            graph_index += 1
            tracker.tick(graph_i=f"{graph_index}/{total_graphs}")
    tracker.end(tuples=sum(len(items) for items in result.values()))
    return result


def generate_counterfactual_static_pairs(
    *,
    graph_count: int,
    pairs_per_graph: int,
    seed: int,
    node_count_range: tuple[int, int],
    topologies: Sequence[TopologyFamily],
    tasks: Sequence[ReasoningType],
    split: str,
    progress: bool = True,
) -> list[StaticQATuple]:
    """Counterfactual (before/after-one-edit) static training pairs.

    generate_static_corpus draws every tuple from an independent, freshly
    sampled graph, so the projector never sees "same entities, graph state
    changed, answer must track the change" -- exactly the CLEGR-style signal
    README section 3 calls for. This reuses the dynamic-session edit/query
    machinery (_sample_edit_program, apply_edit_program, _changed_query) to
    build that signal directly into static training: for each base graph,
    sample one edit and find an answer-changing query, then emit a before
    tuple (stale answer) and an after tuple (updated answer) sharing a
    pair_id, so both states of the same entities appear in training.
    """
    tracker = ProgressTracker("[generate]", graph_count, phase="counterfactual-static", enabled=progress)
    tracker.banner(graphs=graph_count, pairs_per_graph=pairs_per_graph)
    tuples: list[StaticQATuple] = []
    offset = 8_000_000
    low, high = node_count_range
    for index in range(graph_count):
        graph_seed = seed + offset + index
        rng = random.Random(graph_seed)
        node_count = rng.randint(low, high)
        topology = topologies[index % len(topologies)]
        before = make_graph(rng, split, index, node_count, topology)
        scale = _scale_bin(node_count, ood=node_count >= 40)
        for pair_index in range(pairs_per_graph):
            preferred = tasks[(index + pair_index) % len(tasks)]
            found: tuple[AttributedGraph, GraphQuery] | None = None
            for _attempt in range(5):
                program = _sample_edit_program(before, rng, force_noop=False)
                result = apply_edit_program(before, program)
                if not result.applied:
                    continue
                after = result.graph
                query = _changed_query(before, after, preferred, rng, fallback_order=tasks)
                if query is not None:
                    found = (after, query)
                    break
            if found is None:
                continue
            after, query = found
            before_query = replace(query, question=render_question(query, before))
            after_query = replace(query, question=render_question(query, after))
            stale_answer = answer_query(before, before_query)
            answer = answer_query(after, after_query)
            pair_id = f"{split}-cf-{index:06d}-{pair_index:02d}"
            tuples.append(
                StaticQATuple(
                    tuple_id=f"{pair_id}-before",
                    split=split,
                    graph=before,
                    query=before_query,
                    answer=stale_answer,
                    topology=topology,
                    density_bin=_density_bin(len(before.nodes), len(before.edges)),
                    hop_depth=_hop_depth(before_query, before),
                    scale_bin=scale,
                    metadata={
                        "reasoning_type": query.reasoning_type.value,
                        "counterfactual_role": "before",
                        "pair_id": pair_id,
                    },
                )
            )
            tuples.append(
                StaticQATuple(
                    tuple_id=f"{pair_id}-after",
                    split=split,
                    graph=after,
                    query=after_query,
                    answer=answer,
                    topology=topology,
                    density_bin=_density_bin(len(after.nodes), len(after.edges)),
                    hop_depth=_hop_depth(after_query, after),
                    scale_bin=scale,
                    metadata={
                        "reasoning_type": query.reasoning_type.value,
                        "counterfactual_role": "after",
                        "pair_id": pair_id,
                    },
                )
            )
        tracker.tick(graph_i=f"{index + 1}/{graph_count}")
    tracker.end(tuples=len(tuples))
    return tuples


def generate_dataset_v2(
    *,
    counts: dict[str, int],
    seed: int,
    static_graph_counts: dict[str, int] | None = None,
    static_tuples_per_graph: int = 12,
    static_tasks: Sequence[ReasoningType] | None = None,
    dynamic_tasks: Sequence[ReasoningType] | None = None,
    counterfactual_static_train: bool = False,
    progress: bool = True,
) -> tuple[dict[str, list[Session]], dict[str, list[StaticQATuple]]]:
    session_tasks = tuple(dynamic_tasks) if dynamic_tasks else _DYNAMIC_TASKS
    total_sessions = sum(counts.values())
    tracker = ProgressTracker("[generate]", total_sessions, phase="dynamic", enabled=progress)
    tracker.banner(sessions=total_sessions)
    sessions: dict[str, list[Session]] = {}
    split_offsets = {"train": 0, "validation": 1_000_000, "test": 2_000_000, "ood": 3_000_000}
    done = 0
    for split, count in counts.items():
        offset = split_offsets.get(split, 4_000_000)
        split_sessions: list[Session] = []
        for index in range(count):
            session_seed = seed + offset + index
            rng = random.Random(session_seed)
            ood = split == "ood" or split.endswith("_ood")
            if ood:
                node_count = rng.randint(*_OOD_SCALE)
                turn_count = _OOD_TURNS[index % len(_OOD_TURNS)]
            else:
                node_range = _ID_SCALE[index % len(_ID_SCALE)]
                node_count = rng.randint(*node_range)
                turn_count = _ID_TURNS[index % len(_ID_TURNS)]
            topology = (
                TopologyFamily.ERDOS_RENYI
                if ood and index % 3 == 0
                else (TopologyFamily.SBM if index % 2 else TopologyFamily.WATTS_STROGATZ)
            )
            split_sessions.append(
                generate_session_v2(
                    split=split,
                    index=index,
                    seed=session_seed,
                    node_count=node_count,
                    turn_count=turn_count,
                    topology=topology,
                    reasoning_types=session_tasks,
                    ood=ood,
                    fallback_tasks=tuple(dynamic_tasks) if dynamic_tasks else None,
                )
            )
            done += 1
            tracker.tick(sessions=f"{done}/{total_sessions}")
        sessions[split] = split_sessions
    tracker.end()
    assert_disjoint_splits({key: value for key, value in sessions.items() if key != "ood"})
    static_counts = static_graph_counts or {
        "train": 80,
        "validation": 10,
        "test": 20,
    }
    if counterfactual_static_train and static_counts.get("train", 0) > 0:
        eval_counts = {key: value for key, value in static_counts.items() if key != "train"}
        static = (
            generate_static_corpus(
                graph_counts=eval_counts,
                tuples_per_graph=static_tuples_per_graph,
                seed=seed,
                node_count_ranges={"validation": (16, 32), "test": (16, 48)},
                topologies=(TopologyFamily.WATTS_STROGATZ, TopologyFamily.SBM),
                tasks=static_tasks,
                progress=progress,
            )
            if eval_counts
            else {}
        )
        static["train"] = generate_counterfactual_static_pairs(
            graph_count=static_counts["train"],
            pairs_per_graph=max(1, static_tuples_per_graph // 2),
            seed=seed,
            node_count_range=(16, 32),
            topologies=(TopologyFamily.WATTS_STROGATZ, TopologyFamily.SBM),
            tasks=tuple(static_tasks) if static_tasks else _STATIC_TASKS,
            split="train",
            progress=progress,
        )
    else:
        static = generate_static_corpus(
            graph_counts=static_counts,
            tuples_per_graph=static_tuples_per_graph,
            seed=seed,
            node_count_ranges={
                "train": (16, 32),
                "validation": (16, 32),
                "test": (16, 48),
            },
            topologies=(TopologyFamily.WATTS_STROGATZ, TopologyFamily.SBM),
            tasks=static_tasks,
            progress=progress,
        )
    return sessions, static


def save_static_tuples(path: str | Path, tuples: Sequence[StaticQATuple]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in tuples:
            handle.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")


def load_static_tuples(path: str | Path) -> list[StaticQATuple]:
    with Path(path).open(encoding="utf-8") as handle:
        return [StaticQATuple.from_dict(json.loads(line)) for line in handle if line.strip()]
