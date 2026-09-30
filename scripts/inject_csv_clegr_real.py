#!/usr/bin/env python3
"""Prepend the full-graph CSV textualization CLEGR's own paper actually uses
(arXiv:2508.20583, Appendix E.1.7 -- see clegr.md section 8.2 in this repo)
to every prompt, for real CLEGR.

This supersedes scripts/inject_wfi_clegr_real.py's narrower approximation
(source/target node attributes only). The paper's own Eq. 2 term `f` is
"the concatenated textual features of the WHOLE graph" -- confirmed by the
exact prompt shell in Appendix E.1.7, which dumps a CSV of every node and
every edge into every prompt, for every architecture including soft-prompt
(the paper's soft-prompt is not graph-blind at all; it only lacks the 10
GNN-derived tokens the GLMs additionally get). Also corrects a
mischaracterization from earlier this session: CLEGR always uses graph-level
pooling (Eq. 2), never the node-indexed Eq. 1 -- so this script does not
depend on source_id/target_id at all, unlike inject_wfi_clegr_real.py.

Exact format (clegr.md 8.2):
  --- Nodes ---
  "id", "name", "disabled_access", "has_rail", "architecture", "cleanliness", "music", "size"
  <one row per node>
  --- Edges ---
  "source_id", "target_id", "line_color", "line_stroke", "has_aircon", "built"
  <one row per edge>
  Above is the representation of a synthetic subway network. All
  stations and lines are completely fictional. Keep in mind that the
  subway network is not real. All information necessary to answer the
  question is present in the above representation. The question is:
  {existing prompt, which already ends in the correct answer-format suffix}

Usage:
  .venv/bin/python scripts/inject_csv_clegr_real.py \
    --input-dir datasets/clegr_real_fixed --output-dir datasets/clegr_real_csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graph_modi.schema import AttributedGraph  # noqa: E402
from graph_modi.training import TrainingExample  # noqa: E402

_NODE_HEADER = '"id", "name", "disabled_access", "has_rail", "architecture", "cleanliness", "music", "size"'
_EDGE_HEADER = '"source_id", "target_id", "line_color", "line_stroke", "has_aircon", "built"'
_PREAMBLE = (
    "Above is the representation of a synthetic subway network. All\n"
    "stations and lines are completely fictional. Keep in mind that the\n"
    "subway network is not real. All information necessary to answer the\n"
    "question is present in the above representation. The question is:\n"
)


def _load_examples(path: Path) -> list[TrainingExample]:
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            examples.append(
                TrainingExample(
                    graph=AttributedGraph.from_dict(record["graph"]),
                    prompt=record["prompt"],
                    answer=record["answer"],
                    example_id=record["example_id"],
                    split=record["split"],
                    metadata=record["metadata"],
                )
            )
    return examples


def _csv_field(value: object) -> str:
    text = str(value)
    if "," in text or '"' in text or "\n" in text:
        text = '"' + text.replace('"', '""') + '"'
    return text


def _serialize_graph_csv(graph: AttributedGraph) -> str:
    # Real node IDs are 36-char UUIDs; every node row and every edge's
    # source_id/target_id column references one, and with ~26 nodes + ~35
    # edges per graph that's ~96 UUID mentions -- extremely token-inefficient
    # under BPE (measured: pushes median prompt length to ~2761 tokens,
    # truncating 89% of examples at a 2048-token budget). Remap to short,
    # graph-local sequential IDs (n0, n1, ...) instead; this changes nothing
    # about the information content, only how cheaply it's spelled in tokens.
    short_id = {node.id: f"n{i}" for i, node in enumerate(graph.nodes)}

    node_lines = [_NODE_HEADER]
    for node in graph.nodes:
        a = node.attributes
        row = [
            short_id[node.id],
            node.label,
            a.get("disabled_access"),
            a.get("has_rail"),
            a.get("architecture"),
            a.get("cleanliness"),
            a.get("music"),
            a.get("size"),
        ]
        node_lines.append(",".join(_csv_field(v) for v in row))

    edge_lines = [_EDGE_HEADER]
    for edge in graph.edges:
        a = edge.attributes
        row = [
            short_id.get(edge.source, edge.source),
            short_id.get(edge.target, edge.target),
            a.get("line_color"),
            a.get("line_stroke"),
            a.get("has_aircon"),
            a.get("built"),
        ]
        edge_lines.append(",".join(_csv_field(v) for v in row))

    return (
        "--- Nodes ---\n"
        + "\n".join(node_lines)
        + "\n--- Edges ---\n"
        + "\n".join(edge_lines)
        + "\n"
        + _PREAMBLE
    )


def _inject(example: TrainingExample) -> TrainingExample:
    csv_block = _serialize_graph_csv(example.graph)
    return replace(example, prompt=f"{csv_block}{example.prompt}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/clegr_real_fixed")
    parser.add_argument("--output-dir", default="datasets/clegr_real_csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_estimate_chars = 0
    total_examples = 0
    for split in ("train", "validation", "test"):
        examples = _load_examples(input_dir / f"{split}.jsonl")
        out_path = output_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for example in examples:
                new_example = _inject(example)
                token_estimate_chars += len(new_example.prompt)
                total_examples += 1
                handle.write(json.dumps(new_example.record(), default=str) + "\n")
        print(f"[inject-csv] {split}: wrote {len(examples)} examples to {out_path}", flush=True)

    avg_chars = token_estimate_chars / max(total_examples, 1)
    print(
        f"[inject-csv] average prompt length: {avg_chars:.0f} chars "
        f"(~{avg_chars / 4:.0f} tokens at ~4 chars/token)",
        flush=True,
    )


if __name__ == "__main__":
    main()
