"""Canonical edit parsing and execution-equivalence checks."""

from __future__ import annotations

import json
import re

from graph_modi.schema import EditOperation, EditTarget, GraphEdit, Scalar

_SPACE = re.compile(r"\s+")


def _parse_scalar(text: str) -> Scalar:
    normalized = text.strip()
    aliases: dict[str, Scalar] = {
        "open": "open",
        "closed": "closed",
        "true": True,
        "false": False,
        "null": None,
    }
    if normalized.lower() in aliases:
        return aliases[normalized.lower()]
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError:
        return normalized
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError("Edit values must be scalar JSON values")


def parse_edit(text: str, name_to_id: dict[str, str] | None = None) -> GraphEdit:
    """Parse the deliberately small, auditable GraphModi edit language."""
    cleaned = _SPACE.sub(" ", text.strip())
    if not cleaned:
        raise ValueError("Empty edit")
    parts = cleaned.split(" ")
    operation = EditOperation(parts[0].upper())
    if operation is EditOperation.NOOP:
        return GraphEdit(operation=operation, reason=" ".join(parts[1:]) or None)
    if len(parts) < 3:
        raise ValueError(f"Incomplete edit: {text!r}")
    target = EditTarget(parts[1].upper())

    def resolve(value: str) -> str:
        if name_to_id is None:
            return value
        return name_to_id.get(value.casefold(), value)

    if target is EditTarget.NODE:
        node_id = resolve(parts[2])
        if operation is EditOperation.SET:
            if len(parts) < 5:
                raise ValueError("SET NODE requires an attribute and value")
            return GraphEdit(
                operation=operation,
                target=target,
                node_id=node_id,
                attribute=parts[3],
                value=_parse_scalar(" ".join(parts[4:])),
            )
        if operation in {EditOperation.ADD, EditOperation.DEL}:
            return GraphEdit(
                operation=operation,
                target=target,
                node_id=node_id,
                label=" ".join(parts[3:]) or None,
            )
    if target is EditTarget.EDGE:
        if operation is EditOperation.SET:
            raise ValueError("SET EDGE is not supported; delete and re-add the edge")
        if len(parts) < 4:
            raise ValueError(f"{operation.value} EDGE requires two node IDs")
        return GraphEdit(
            operation=operation,
            target=target,
            source=resolve(parts[2]),
            destination=resolve(parts[3]),
            relation=parts[4] if len(parts) > 4 else "connected",
        )
    raise ValueError(f"Unsupported edit: {text!r}")


def execution_equivalent(left: GraphEdit | None, right: GraphEdit | None) -> bool:
    if left is None or right is None:
        return left is right
    if left.operation is EditOperation.NOOP and right.operation is EditOperation.NOOP:
        return True
    return (
        left.operation == right.operation
        and left.target == right.target
        and left.node_id == right.node_id
        and left.source == right.source
        and left.destination == right.destination
        and left.attribute == right.attribute
        and left.value == right.value
        and left.relation == right.relation
    )
