"""Typed, JSON-serializable experiment records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

Scalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class Node:
    id: str
    label: str
    attributes: dict[str, Scalar] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Edge:
    source: str
    target: str
    relation: str = "connected"
    weight: float = 1.0
    attributes: dict[str, Scalar] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AttributedGraph:
    graph_id: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    directed: bool = False
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def node_map(self) -> dict[str, Node]:
        return {node.id: node for node in self.nodes}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> AttributedGraph:
        return cls(
            graph_id=str(record["graph_id"]),
            nodes=tuple(
                Node(
                    id=str(node["id"]),
                    label=str(node.get("label", node["id"])),
                    attributes=dict(node.get("attributes", {})),
                )
                for node in record["nodes"]
            ),
            edges=tuple(
                Edge(
                    source=str(edge["source"]),
                    target=str(edge["target"]),
                    relation=str(edge.get("relation", "connected")),
                    weight=float(edge.get("weight", 1.0)),
                    attributes=dict(edge.get("attributes", {})),
                )
                for edge in record["edges"]
            ),
            directed=bool(record.get("directed", False)),
            metadata=dict(record.get("metadata", {})),
        )


class EditOperation(str, Enum):
    SET = "SET"
    ADD = "ADD"
    DEL = "DEL"
    NOOP = "NOOP"


class EditTarget(str, Enum):
    NODE = "NODE"
    EDGE = "EDGE"


@dataclass(frozen=True, slots=True)
class GraphEdit:
    operation: EditOperation
    target: EditTarget | None = None
    node_id: str | None = None
    source: str | None = None
    destination: str | None = None
    attribute: str | None = None
    value: Scalar = None
    label: str | None = None
    relation: str = "connected"
    weight: float = 1.0
    reason: str | None = None

    def canonical(self) -> str:
        if self.operation is EditOperation.NOOP:
            return "NOOP" if not self.reason else f"NOOP {self.reason}"
        if self.target is EditTarget.NODE:
            if self.operation is EditOperation.SET:
                return f"SET NODE {self.node_id} {self.attribute} {self.value}"
            suffix = f" {self.label}" if self.label else ""
            return f"{self.operation.value} NODE {self.node_id}{suffix}"
        if self.target is EditTarget.EDGE:
            suffix = "" if self.relation == "connected" else f" {self.relation}"
            return f"{self.operation.value} EDGE {self.source} {self.destination}{suffix}"
        return str(self.operation.value)

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["operation"] = self.operation.value
        record["target"] = self.target.value if self.target else None
        return record

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> GraphEdit:
        return cls(
            operation=EditOperation(str(record["operation"]).upper()),
            target=(EditTarget(str(record["target"]).upper()) if record.get("target") else None),
            node_id=record.get("node_id"),
            source=record.get("source"),
            destination=record.get("destination"),
            attribute=record.get("attribute"),
            value=record.get("value"),
            label=record.get("label"),
            relation=str(record.get("relation", "connected")),
            weight=float(record.get("weight", 1.0)),
            reason=record.get("reason"),
        )


class ReasoningType(str, Enum):
    SHORTEST_PATH = "shortest_path"
    REACHABILITY = "reachability"
    FILTERED_NEIGHBOR_COUNT = "filtered_neighbor_count"
    FILTERED_PATH_COUNT = "filtered_path_count"
    CYCLE_MEMBERSHIP = "cycle_membership"
    EDGE_EXISTS = "edge_exists"
    NODE_DEGREE = "node_degree"
    NODE_COUNT = "node_count"
    PATH_COST = "path_cost"
    LINK_PREDICTION = "link_prediction"


class TopologyFamily(str, Enum):
    WATTS_STROGATZ = "watts_strogatz"
    SBM = "sbm"
    ERDOS_RENYI = "erdos_renyi"


class DensityBin(str, Enum):
    SPARSE = "sparse"
    MEDIUM = "medium"
    DENSE = "dense"


class HopDepth(str, Enum):
    LOCAL = "local"
    MID = "mid"
    GLOBAL = "global"


@dataclass(frozen=True, slots=True)
class EditProgram:
    """Ordered edit sequence terminated by END in canonical serialization."""

    edits: tuple[GraphEdit, ...]

    def canonical(self) -> str:
        if not self.edits:
            return "NOOP ; END"
        return " ; ".join(edit.canonical() for edit in self.edits) + " ; END"

    def to_dict(self) -> dict[str, Any]:
        return {"edits": [edit.to_dict() for edit in self.edits]}

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> EditProgram:
        return cls(edits=tuple(GraphEdit.from_dict(item) for item in record.get("edits", [])))

    @classmethod
    def single(cls, edit: GraphEdit) -> EditProgram:
        return cls(edits=(edit,))


@dataclass(frozen=True, slots=True)
class StaticQATuple:
    """Independent static (G, Q, A) sample for GLM pretraining."""

    tuple_id: str
    split: str
    graph: AttributedGraph
    query: GraphQuery
    answer: str
    topology: TopologyFamily
    density_bin: DensityBin
    hop_depth: HopDepth
    scale_bin: str
    metadata: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tuple_id": self.tuple_id,
            "split": self.split,
            "graph": self.graph.to_dict(),
            "query": self.query.to_dict(),
            "answer": self.answer,
            "topology": self.topology.value,
            "density_bin": self.density_bin.value,
            "hop_depth": self.hop_depth.value,
            "scale_bin": self.scale_bin,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> StaticQATuple:
        return cls(
            tuple_id=str(record["tuple_id"]),
            split=str(record["split"]),
            graph=AttributedGraph.from_dict(record["graph"]),
            query=GraphQuery.from_dict(record["query"]),
            answer=str(record["answer"]),
            topology=TopologyFamily(record["topology"]),
            density_bin=DensityBin(record["density_bin"]),
            hop_depth=HopDepth(record["hop_depth"]),
            scale_bin=str(record["scale_bin"]),
            metadata=dict(record.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class GraphQuery:
    reasoning_type: ReasoningType
    source: str
    target: str | None = None
    attribute: str | None = None
    value: Scalar = None
    question: str = ""

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["reasoning_type"] = self.reasoning_type.value
        return record

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> GraphQuery:
        return cls(
            reasoning_type=ReasoningType(record["reasoning_type"]),
            source=str(record["source"]),
            target=record.get("target"),
            attribute=record.get("attribute"),
            value=record.get("value"),
            question=str(record.get("question", "")),
        )


@dataclass(frozen=True, slots=True)
class Turn:
    turn_index: int
    utterance: str
    gold_edit: GraphEdit
    query: GraphQuery
    gold_answer: str
    stale_answer: str
    before_fingerprint: str
    after_fingerprint: str
    tags: tuple[str, ...] = ()
    gold_edits: tuple[GraphEdit, ...] = ()
    edit_program: EditProgram | None = None
    complexity: dict[str, Scalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "turn_index": self.turn_index,
            "utterance": self.utterance,
            "gold_edit": self.gold_edit.to_dict(),
            "query": self.query.to_dict(),
            "gold_answer": self.gold_answer,
            "stale_answer": self.stale_answer,
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "tags": list(self.tags),
        }
        if self.gold_edits:
            record["gold_edits"] = [edit.to_dict() for edit in self.gold_edits]
        if self.edit_program is not None:
            record["edit_program"] = self.edit_program.to_dict()
        if self.complexity:
            record["complexity"] = dict(self.complexity)
        return record

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Turn:
        gold_edits = tuple(
            GraphEdit.from_dict(item) for item in record.get("gold_edits", [])
        )
        edit_program = (
            EditProgram.from_dict(record["edit_program"])
            if record.get("edit_program")
            else (EditProgram(edits=gold_edits) if gold_edits else None)
        )
        return cls(
            turn_index=int(record["turn_index"]),
            utterance=str(record["utterance"]),
            gold_edit=GraphEdit.from_dict(record["gold_edit"]),
            query=GraphQuery.from_dict(record["query"]),
            gold_answer=str(record["gold_answer"]),
            stale_answer=str(record["stale_answer"]),
            before_fingerprint=str(record["before_fingerprint"]),
            after_fingerprint=str(record["after_fingerprint"]),
            tags=tuple(record.get("tags", [])),
            gold_edits=gold_edits,
            edit_program=edit_program,
            complexity=dict(record.get("complexity", {})),
        )


@dataclass(frozen=True, slots=True)
class Session:
    session_id: str
    split: str
    initial_graph: AttributedGraph
    turns: tuple[Turn, ...]
    seed: int
    paraphrase_family: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "split": self.split,
            "initial_graph": self.initial_graph.to_dict(),
            "turns": [turn.to_dict() for turn in self.turns],
            "seed": self.seed,
            "paraphrase_family": self.paraphrase_family,
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Session:
        return cls(
            session_id=str(record["session_id"]),
            split=str(record["split"]),
            initial_graph=AttributedGraph.from_dict(record["initial_graph"]),
            turns=tuple(Turn.from_dict(turn) for turn in record["turns"]),
            seed=int(record["seed"]),
            paraphrase_family=str(record["paraphrase_family"]),
        )


@dataclass(frozen=True, slots=True)
class TurnOutput:
    turn_index: int
    predicted_edit: GraphEdit | None
    edit_applied: bool
    predicted_answer: str | None
    graph_fingerprint: str
    encode_calls: int
    latency_seconds: float
    error: str | None = None
