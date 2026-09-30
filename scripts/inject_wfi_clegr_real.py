#!/usr/bin/env python3
"""Add the CLEGR paper's missing W(f_i) term to the real-CLEGR dataset: the
target entity's own textual attributes, serialized and prepended to the
question, so the LLM can read them directly as ordinary tokens -- exactly
what arXiv:2508.20583's node-level formula concatenates alongside the GNN
prefix (see documents/experiments/results/v2_clegr_extended-20260919/
report.md, the discussion of equation (1): "M_l(M_P(M_g(G,n_i)) || W(f_i)
|| W(q))"). This project's TEAGLM/SoftPromptGLM forward/generate_batch never
had this term -- only the graph-encoded prefix and the bare question text.

This script does NOT touch model code at all: it rewrites the `prompt`
field of each existing TrainingExample (loaded from the metadata-fixed
dataset in datasets/clegr_real_fixed/, which already carries correct
source_id/target_id from the earlier pooling-fix work) by prepending the
serialized attributes of the source/target node(s), then writes a new
dataset to datasets/clegr_real_wfi/. Because it is a pure text change
applied identically regardless of which model consumes it afterward, the
same output feeds TEA, GraphToken, and soft_prompt without any
architecture-specific handling -- and because the underlying graphs are
untouched, the existing GNN pretrain checkpoint
(outputs/clegr_real_pooling_fix/gnn) remains valid and does not need to be
regenerated for TEA/GraphToken.

Usage:
  .venv/bin/python scripts/inject_wfi_clegr_real.py \
    --input-dir datasets/clegr_real_fixed --output-dir datasets/clegr_real_wfi
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


def _serialize_node(graph: AttributedGraph, node_id: str | None) -> str | None:
    if not node_id:
        return None
    node = next((n for n in graph.nodes if n.id == node_id), None)
    if node is None:
        return None
    attrs = ", ".join(f"{key}={value}" for key, value in node.attributes.items())
    return f"{node.label}: {attrs}."


def _inject(example: TrainingExample) -> TrainingExample:
    source_id = example.metadata.get("source_id")
    target_id = example.metadata.get("target_id")
    texts = [t for t in (_serialize_node(example.graph, source_id), _serialize_node(example.graph, target_id)) if t]
    if not texts:
        return example
    prefix = " ".join(dict.fromkeys(texts))  # de-dupe if source==target text happens to coincide
    return replace(example, prompt=f"{prefix}\n{example.prompt}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/clegr_real_fixed")
    parser.add_argument("--output-dir", default="datasets/clegr_real_wfi")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_injected = 0
    total_examples = 0
    for split in ("train", "validation", "test"):
        examples = _load_examples(input_dir / f"{split}.jsonl")
        out_path = output_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for example in examples:
                new_example = _inject(example)
                if new_example.prompt != example.prompt:
                    total_injected += 1
                total_examples += 1
                handle.write(json.dumps(new_example.record(), default=str) + "\n")
        print(f"[inject-wfi] {split}: wrote {len(examples)} examples to {out_path}", flush=True)

    print(f"[inject-wfi] injected W(f_i) text into {total_injected}/{total_examples} examples", flush=True)


if __name__ == "__main__":
    main()
