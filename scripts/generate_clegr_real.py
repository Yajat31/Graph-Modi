#!/usr/bin/env python3
"""Generate REAL CLEGR graphs + questions + answers (via the vendored
generator in third_party/clegr/, copied from
https://github.com/rethinking-graph-language-evals/CLEGR, Apache-2.0 --
see third_party/clegr/NOTICE) and convert them into this project's own
AttributedGraph + TrainingExample schema, so they can be fed through the
existing pretrain_graph_encoder / train_projector / train_graph_token /
evaluate_static_oracle pipeline unchanged.

This does NOT use CLEGR's own PyG/BERT dataset pipeline (src/data/clegr.py
upstream) -- that requires torch_geometric's compiled extensions, BERT
embedding of every node/edge sentence, and PyTorch-Lightning, none of which
this project needs (we tensorize nodes/edges ourselves, the same way we do
for the synthetic metro benchmark). We only use their pure-Python graph
generator (GraphGenerator) and question DSL (QuestionForm/question_forms) --
the part that actually defines "the CLEGR dataset" -- to get 100% real
CLEGR graphs, questions, and gold answers.

One dependency patch: third_party/clegr/generate_graph.py has `import
bezier` removed and its one call site replaced with an equivalent pure-numpy
cubic Bezier evaluation (`bezier` ships no wheels for Python 3.13+ and fails
to build from source) -- see the docstring on
`_cubic_bezier_evaluate_multi` in that file for the exact substitution.
Nothing else in the vendored files was changed.

Splits by graph_id (not by example), 60/20/20, matching CLEGR's own
`split_by_graph` convention (src/data/clegr.py upstream) so no graph's
questions leak across splits.

Usage:
  .venv/bin/python scripts/generate_clegr_real.py \
    --graphs 500 --questions-per-graph 2 --size small \
    --seed 0 --output-dir datasets/clegr_real
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from third_party.clegr.generate_graph import GraphGenerator  # noqa: E402
from third_party.clegr.questions import question_forms  # noqa: E402
from third_party.clegr.types_ import GraphSpec  # noqa: E402

from graph_modi.schema import AttributedGraph, Edge, Node  # noqa: E402
from graph_modi.training import TrainingExample  # noqa: E402


def _to_attributed_graph(spec: GraphSpec) -> AttributedGraph:
    nodes = tuple(
        Node(
            id=node.id,
            label=node.name,
            attributes={
                "architecture": node.architecture,
                "cleanliness": node.cleanliness,
                "disabled_access": bool(node.disabled_access),
                "has_rail": bool(node.has_rail),
                "music": node.music,
                "size": node.size,
            },
        )
        for node in spec.nodes.values()
    )
    edges = tuple(
        Edge(
            source=edge.station1,
            target=edge.station2,
            relation="line",
            attributes={
                "line_id": edge.line_id,
                "line_name": edge.line_name,
                "line_color": edge.line_color,
                "line_stroke": edge.line_stroke,
                "has_aircon": bool(spec.lines[edge.line_id].has_aircon) if edge.line_id in spec.lines else None,
                "built": spec.lines[edge.line_id].built if edge.line_id in spec.lines else None,
            },
        )
        for edge in spec.edges
    )
    return AttributedGraph(graph_id=spec.id, nodes=nodes, edges=edges, directed=False)


def _generate_graph(args: argparse.Namespace) -> GraphSpec:
    gen_args = argparse.Namespace(
        small=args.size == "small",
        medium=args.size == "medium",
        mixed=args.size == "mixed",
        int_names=False,
        disconnected=False,
    )
    return GraphGenerator(gen_args).generate().graph_spec


def _extract_node_ids(functional: object, found: list[str]) -> None:
    """Recursively walk a FunctionalOperator.to_dict() tree (q_spec.functional)
    collecting every referenced station id, in encounter order.

    third_party/clegr/functional.py's FunctionalOperator._serialize_arg already
    tags a NodeSpec argument as the leaf dict {"Node": <id>} (a plain string
    value, never a list); every operator's own to_dict() instead produces
    {OperatorName: [...]} (the value is always a list -- see to_dict()'s
    definition: `serialized_args` is built via a list comprehension). That
    shape difference is what distinguishes a leaf node reference from an
    operator to recurse into, with no need to enumerate every operator type
    or touch the vendored generator: the identity info CLEGR's own generator
    already computes (it must, to build and answer each question) just was
    not being extracted into this project's TrainingExample metadata before,
    leaving TEA/GraphToken's source_row/target_row node-addressing (see
    graph_modi.models.tea_glm.TEAGLM.encode_graphs) permanently zero-filled
    for every real-CLEGR example.
    """
    if isinstance(functional, dict):
        if set(functional) == {"Node"} and isinstance(functional["Node"], str):
            found.append(functional["Node"])
            return
        for value in functional.values():
            _extract_node_ids(value, found)
    elif isinstance(functional, (list, tuple)):
        for item in functional:
            _extract_node_ids(item, found)


def _format_answer(answer: object) -> str:
    """Render the solver's raw answer the way the question's own suffix asks
    for it (LIST_SUFFIX says "comma-separated list", BOOL_SUFFIX says "True"
    or "False") -- plain str(answer) on a Python list/tuple would instead
    train the model to imitate a Python repr like "['A', 'B']", not the
    comma-separated text the prompt actually requests."""
    if isinstance(answer, bool):
        return "True" if answer else "False"
    if isinstance(answer, (list, tuple, set, frozenset)):
        return ", ".join(str(item) for item in answer)
    return str(answer)


def _generate_examples_for_graph(
    spec: GraphSpec, graph: AttributedGraph, questions_per_form: int, max_attempts: int, split: str
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for form in question_forms:
        seen_english: set[str] = set()
        produced = 0
        attempts = 0
        while produced < questions_per_form and attempts < max_attempts:
            attempts += 1
            result = form.generate(spec, None)
            if not result or result[0] is None:
                continue
            q_spec, answer = result[0], result[1]
            if q_spec.english in seen_english:
                continue  # avoid duplicate (station-pair reuse) within the same graph
            seen_english.add(q_spec.english)
            node_ids: list[str] = []
            _extract_node_ids(q_spec.functional, node_ids)
            source_id = node_ids[0] if len(node_ids) >= 1 else None
            target_id = node_ids[1] if len(node_ids) >= 2 else None
            examples.append(
                TrainingExample(
                    graph=graph,
                    prompt=q_spec.english,
                    answer=_format_answer(answer),
                    example_id=str(uuid.uuid4()),
                    split=split,
                    metadata={
                        "clegr_type": q_spec.type_string,
                        "clegr_group": q_spec.group,
                        "clegr_subgroup": q_spec.subgroup,
                        "source_id": source_id,
                        "target_id": target_id,
                        "hops": None,
                        "reasoning_type": None,
                    },
                )
            )
            produced += 1
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=int, default=500, help="total graphs, matching CLEGR's own default")
    parser.add_argument("--questions-per-graph", type=int, default=2, help="instances per question template per graph")
    parser.add_argument("--max-attempts", type=int, default=15, help="retries per question-template instance")
    parser.add_argument("--size", choices=["small", "medium", "mixed"], default="small",
                        help="small empirically matches the paper's ~26.5-node base CLeGR average; "
                             "medium/mixed produce much larger graphs than either CLeGR or CLeGR-Large in practice")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="datasets/clegr_real")
    parser.add_argument("--split-ratio", type=float, nargs=3, default=(0.6, 0.2, 0.2))
    args = parser.parse_args()

    random.seed(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
    except ImportError:
        pass

    graph_ids = list(range(args.graphs))
    rng = random.Random(args.seed)
    shuffled = graph_ids[:]
    rng.shuffle(shuffled)
    train_ratio, val_ratio, _test_ratio = args.split_ratio
    n_train = int(args.graphs * train_ratio)
    n_val = int(args.graphs * val_ratio)
    split_by_index = {}
    for i, idx in enumerate(shuffled):
        if i < n_train:
            split_by_index[idx] = "train"
        elif i < n_train + n_val:
            split_by_index[idx] = "validation"
        else:
            split_by_index[idx] = "test"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "validation", "test")
    }
    counts = {"train": 0, "validation": 0, "test": 0}

    try:
        for i in range(args.graphs):
            split = split_by_index[i]
            spec = _generate_graph(args)
            graph = _to_attributed_graph(spec)
            examples = _generate_examples_for_graph(
                spec, graph, args.questions_per_graph, args.max_attempts, split
            )
            for example in examples:
                handles[split].write(json.dumps(example.record(), default=str) + "\n")
            counts[split] += len(examples)
            if (i + 1) % 50 == 0 or i + 1 == args.graphs:
                print(f"[clegr-real] {i + 1}/{args.graphs} graphs done, counts={counts}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    print(f"[clegr-real] wrote {output_dir} -- final counts={counts}")


if __name__ == "__main__":
    main()
