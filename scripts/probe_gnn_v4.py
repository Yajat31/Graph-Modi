"""Linear probes on frozen node embeddings of the v4.1 GNNs (GPU recommended, CPU works).

  PYTHONPATH=src python3 scripts/probe_gnn_v4.py --outputs outputs \
      --out documents/experiments/results/v4_1-analysis/gnn_probe.json

For each encoder (BERT input features without a GNN, the frozen random GNN, the TEA-GLM-pretrained
GNN, and the GNNs trained jointly by GraphToken s42 and the warm start), node embeddings of static
graphs are computed once. A multinomial logistic-regression probe (full-batch L-BFGS on
standardized features) fit on train-split graphs and scored on test-split graphs measures how linearly decodable node degree, station status and the number of
stations within two hops are. Embedding collapse is measured by the mean cosine similarity between
nodes of the same graph and the effective rank (participation ratio of the covariance spectrum).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from graph_modi.cli import _neural_components  # noqa: E402
from graph_modi.config import load_config  # noqa: E402
from graph_modi.data.v2 import load_static_tuples  # noqa: E402

ENCODERS = {
    "bert_input": None,
    "random_frozen": "v4_tea_random_sum/projector/checkpoint-best/gnn.pt",
    "tea_glm_pretrained": "v4_pretrain_sum/gnn/gnn.pt",
    "graphtoken_s42": "v4_graphtoken_s42/projector/checkpoint-best/gnn.pt",
    "warm_start_s42": "v4_graphtoken_warm_s42/projector/checkpoint-best/gnn.pt",
}


def unique_graphs(path: Path, limit: int) -> list:
    graphs = {item.graph.graph_id: item.graph for item in load_static_tuples(path)}
    return list(graphs.values())[:limit]


def node_targets(graph) -> dict[str, list]:
    neighbours: dict[str, set[str]] = {node.id: set() for node in graph.nodes}
    for edge in graph.edges:
        neighbours[edge.source].add(edge.target)
        neighbours[edge.target].add(edge.source)
    ordered = sorted(graph.nodes, key=lambda n: n.id)
    two_hop = []
    for node in ordered:
        reach = set(neighbours[node.id])
        for other in neighbours[node.id]:
            reach |= neighbours[other]
        reach.discard(node.id)
        two_hop.append(min(len(reach), 8))
    return {
        "degree": [min(len(neighbours[n.id]), 6) for n in ordered],
        "status": [str(n.attributes.get("status", "?")) for n in ordered],
        "within_two_hops": two_hop,
    }


def embed(graphs, tensorizer, gnn, device) -> np.ndarray:
    import torch

    rows = []
    with torch.no_grad():
        for graph in graphs:
            tensor = tensorizer(graph).to(device)
            rows.append((tensor.x if gnn is None else gnn(tensor)).float().cpu().numpy())
    return np.concatenate(rows)


def probe_accuracy(x_train, y_train, x_test, y_test, device, steps: int = 500) -> float:
    import torch

    classes = sorted(set(y_train) | set(y_test), key=str)
    index = {c: i for i, c in enumerate(classes)}
    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    xe = torch.tensor(x_test, dtype=torch.float32, device=device)
    mean, std = xt.mean(0), xt.std(0).clamp_min(1e-6)
    xt, xe = (xt - mean) / std, (xe - mean) / std
    yt = torch.tensor([index[c] for c in y_train], device=device)
    ye = torch.tensor([index[c] for c in y_test], device=device)
    torch.manual_seed(0)
    layer = torch.nn.Linear(xt.shape[1], len(classes)).to(device)
    optimizer = torch.optim.LBFGS(layer.parameters(), max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(layer(xt), yt)
        loss = loss + 1e-4 * layer.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return float((layer(xe).argmax(1) == ye).float().mean())


def collapse_stats(embeddings: np.ndarray, sizes: list[int]) -> dict[str, float]:
    cosines, offset = [], 0
    for size in sizes:
        block = embeddings[offset : offset + size]
        block = block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)
        sim = block @ block.T
        cosines.append(float((sim.sum() - size) / max(1, size * (size - 1))))
        offset += size
    eigenvalues = np.clip(np.linalg.eigvalsh(np.cov(embeddings, rowvar=False)), 0, None)
    return {
        "mean_within_graph_cosine": float(np.mean(cosines)),
        "effective_rank": float(eigenvalues.sum() ** 2 / max((eigenvalues**2).sum(), 1e-12)),
        "dim": int(embeddings.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v4_tea_s42.yaml")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--train-graphs", type=int, default=600)
    parser.add_argument("--test-graphs", type=int, default=396)
    parser.add_argument(
        "--out", default="documents/experiments/results/v4_1-analysis/gnn_probe.json"
    )
    args = parser.parse_args()

    import torch

    config = load_config(args.config)
    data = Path(config.data_dir)
    train = unique_graphs(data / "static_train.jsonl", args.train_graphs)
    test = unique_graphs(data / "static_test.jsonl", args.test_graphs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, tensorizer = _neural_components(config)
    targets = {
        split: {k: sum((node_targets(g)[k] for g in graphs), []) for k in node_targets(graphs[0])}
        for split, graphs in (("train", train), ("test", test))
    }
    majority = {
        key: max(
            np.mean(np.array(targets["test"][key]) == value) for value in set(targets["test"][key])
        )
        for key in targets["test"]
    }
    report: dict = {"graphs": {"train": len(train), "test": len(test)}, "majority": majority}
    for name, rel in ENCODERS.items():
        gnn = None
        if rel is not None:
            path = Path(args.outputs) / rel
            if not path.exists():
                print(f"skip {name}: {path} missing")
                continue
            _, gnn, _ = _neural_components(config)
            gnn.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            gnn.to(device).eval()
        x_train = embed(train, tensorizer, gnn, device)
        x_test = embed(test, tensorizer, gnn, device)
        entry = {"collapse": collapse_stats(x_test, [len(g.nodes) for g in test]), "probe": {}}
        for key in targets["train"]:
            entry["probe"][key] = probe_accuracy(
                x_train, targets["train"][key], x_test, targets["test"][key], device
            )
        report[name] = entry
        print(name, json.dumps(entry))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
