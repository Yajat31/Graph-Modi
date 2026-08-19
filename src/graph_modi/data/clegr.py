"""Minimal adapters for CLEGR-style PyTorch graph records.

CLEGR itself is not vendored. These adapters accept common ``x``/``edge_index``
records and preserve the original tensors in metadata only through explicit
scalar summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graph_modi.schema import AttributedGraph, Edge, Node


def graph_from_record(
    record: dict[str, Any],
    *,
    graph_id: str,
    id_to_name: dict[int, str] | None = None,
) -> AttributedGraph:
    features = record.get("x")
    edge_index = record.get("edge_index")
    if features is None or edge_index is None:
        raise ValueError("CLEGR record must contain x and edge_index")
    feature_rows = features.tolist() if hasattr(features, "tolist") else features
    edges_raw = edge_index.tolist() if hasattr(edge_index, "tolist") else edge_index
    if len(edges_raw) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    nodes = tuple(
        Node(
            id=f"n{index}",
            label=(id_to_name or {}).get(index, f"entity_{index}"),
            attributes={
                f"feature_{feature_index}": float(value) for feature_index, value in enumerate(row)
            },
        )
        for index, row in enumerate(feature_rows)
    )
    seen: set[tuple[int, int]] = set()
    edges: list[Edge] = []
    for source, target in zip(edges_raw[0], edges_raw[1], strict=True):
        source_int, target_int = int(source), int(target)
        if source_int == target_int:
            continue
        key = (source_int, target_int) if source_int < target_int else (target_int, source_int)
        if key in seen:
            continue
        seen.add(key)
        edges.append(Edge(f"n{source_int}", f"n{target_int}"))
    return AttributedGraph(
        graph_id=graph_id,
        nodes=nodes,
        edges=tuple(edges),
        metadata={"source": "clegr"},
    )


def load_clegr_pt(
    path: str | Path,
    *,
    mapper_path: str | Path | None = None,
) -> list[AttributedGraph]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install graph-modi[tea] to read CLEGR .pt files") from error
    id_to_name: dict[int, str] | None = None
    if mapper_path is not None:
        mapper = json.loads(Path(mapper_path).read_text(encoding="utf-8"))
        id_to_name = {int(index): str(name) for index, name in mapper.items()}
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    records = payload if isinstance(payload, list) else [payload]
    normalized: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record, dict):
            normalized.append(record)
        elif hasattr(record, "x") and hasattr(record, "edge_index"):
            normalized.append({"x": record.x, "edge_index": record.edge_index})
        else:
            raise ValueError(f"Unsupported CLEGR graph record: {type(record)!r}")
    return [
        graph_from_record(record, graph_id=f"clegr-{index}", id_to_name=id_to_name)
        for index, record in enumerate(normalized)
    ]


def export_graphs(path: str | Path, graphs: list[AttributedGraph]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for graph in graphs:
            handle.write(json.dumps(graph.to_dict(), sort_keys=True) + "\n")
