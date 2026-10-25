"""Probe what a (pre)trained GNN's node embeddings contain -- no LLM forward pass needed.

For each GNN checkpoint (and for the raw BERT node features, as the "no GNN" baseline) this fits
closed-form ridge probes on embeddings of *training* graphs and scores them on *held-out*
validation graphs:

* retrieval: ridge-map an embedding to the LLM-input-embedding of the node's own text and check
  whether the nearest target is the node's own. NOT informative: the map is refit on training
  graphs and the raw BERT features (which contain the text) already score 1.000. It says nothing
  about a pretraining run's own alignment head (see that run's metadata.json for that);
* status / neighbour status ("some neighbour is closed", which needs message passing): ridge
  probes scored by AUC (threshold-free; chance 0.5) as well as accuracy next to the majority rate;
* flip sensitivity: change one node's status and measure how far its own and its neighbours'
  embeddings move, relative to the distance between two different nodes of the same graph.

    python scripts/probe_gnn.py --config configs/v3_tea.yaml --out probe.json \
        tea=outputs/v3_tea/gnn/gnn.pt deep=outputs/v3x_deep/gnn/gnn.pt random=random:0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from graph_modi.cli import (  # noqa: E402
    _embedding_only_lm,
    _neural_components,
    _static_path,
)
from graph_modi.config import load_config  # noqa: E402
from graph_modi.data.v2 import load_static_tuples  # noqa: E402
from graph_modi.schema import AttributedGraph  # noqa: E402


def distinct_graphs(path: Path, limit: int) -> list[AttributedGraph]:
    graphs = {item.graph.graph_id: item.graph for item in load_static_tuples(path)}
    return list(graphs.values())[:limit]


def node_text(node) -> str:
    attrs = ", ".join(f"{key}={value}" for key, value in sorted(node.attributes.items()))
    return f"{node.label}: {attrs}" if attrs else node.label


def embed_graphs(gnn, tensorizer, graphs, device, *, raw=False):
    """[N, D] embeddings for every node (sorted by id within each graph) plus per-graph sizes."""
    rows, sizes = [], []
    with torch.no_grad():
        for graph in graphs:
            tensor = tensorizer(graph).to(device)
            out = tensor.x.float() if raw else gnn(tensor).float()
            order = sorted(range(len(tensor.node_ids)), key=lambda i: tensor.node_ids[i])
            rows.append(out[order])
            sizes.append(len(order))
    return torch.cat(rows), sizes


def ridge(x, y, lam):
    mean_x, mean_y = x.mean(0, keepdim=True), y.mean(0, keepdim=True)
    xc, yc = x - mean_x, y - mean_y
    gram = xc.T @ xc
    gram += lam * gram.diagonal().mean() * torch.eye(gram.shape[0], device=x.device)
    weights = torch.linalg.solve(gram, xc.T @ yc)
    return lambda z: (z - mean_x) @ weights + mean_y


def labels(graphs, kind):
    out = []
    for graph in graphs:
        closed = {n.id for n in graph.nodes if n.attributes.get("status") == "closed"}
        neighbours = {n.id: set() for n in graph.nodes}
        for edge in graph.edges:
            neighbours[edge.source].add(edge.target)
            neighbours[edge.target].add(edge.source)
        for node in sorted(graph.nodes, key=lambda n: n.id):
            if kind == "status":
                out.append(float(node.id in closed))
            else:
                out.append(float(bool(neighbours[node.id] & closed)))
    return torch.tensor(out)


def probe(train_x, val_x, train_graphs, val_graphs, val_sizes, train_y, val_y, device):
    result = {}
    fit = ridge(train_x, train_y, 1e-2)
    pred = torch.nn.functional.normalize(fit(val_x), dim=-1)
    target = torch.nn.functional.normalize(val_y, dim=-1)
    scores = pred @ target.T
    within = in_batch = 0
    offset = 0
    batch_starts = list(range(0, len(val_sizes), 8))
    for graph_index, size in enumerate(val_sizes):
        block = scores[offset : offset + size, offset : offset + size]
        within += int((block.argmax(1) == torch.arange(size, device=device)).sum())
        offset += size
    for start in batch_starts:
        lo = sum(val_sizes[:start])
        hi = sum(val_sizes[: start + 8])
        block = scores[lo:hi, lo:hi]
        in_batch += int((block.argmax(1) == torch.arange(hi - lo, device=device)).sum())
    total = sum(val_sizes)
    result["retrieval_within_graph"] = within / total
    result["chance_within_graph"] = sum(1 / s for s in val_sizes for _ in range(s)) / total
    result["retrieval_in_batch_of_8"] = in_batch / total
    result["chance_in_batch_of_8"] = sum(
        (hi_ - lo_) * (1 / (hi_ - lo_))
        for lo_, hi_ in (
            (sum(val_sizes[:s]), sum(val_sizes[: s + 8])) for s in batch_starts
        )
    ) / total
    for kind in ("status", "neighbour_status"):
        ytr = labels(train_graphs, "status" if kind == "status" else "nbr").to(device)
        yva = labels(val_graphs, "status" if kind == "status" else "nbr").to(device)
        fit_label = ridge(train_x, ytr[:, None], 1e-2)
        raw_scores = fit_label(val_x)[:, 0]
        predicted = (raw_scores > ytr.mean()).float()
        majority = max(yva.mean().item(), 1 - yva.mean().item())
        # Accuracy of a mean-thresholded ridge output is a poor metric when features are
        # uninformative (it can fall below the majority rate); the AUC is the one to read.
        result[f"{kind}_probe_accuracy"] = (predicted == yva).float().mean().item()
        result[f"{kind}_majority"] = majority
        result[f"{kind}_probe_auc"] = auc(raw_scores, yva)
    return result


def auc(scores, truth):
    """Rank-based ROC AUC of ``scores`` for binary ``truth`` (0.5 = uninformative)."""
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, device=scores.device, dtype=torch.float32)
    positives = truth.bool()
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def flip_graph(graph: AttributedGraph, node_id: str) -> AttributedGraph:
    record = graph.to_dict()
    for node in record["nodes"]:
        if node["id"] == node_id:
            current = node["attributes"].get("status", "open")
            node["attributes"]["status"] = "open" if current == "closed" else "closed"
    return AttributedGraph.from_dict(record)


def flip_sensitivity(gnn, tensorizer, graphs, device, *, raw=False, per_graph=3, seed=0):
    rng = random.Random(seed)
    own, neighbour, between = [], [], []
    for graph in graphs:
        base, _ = embed_graphs(gnn, tensorizer, [graph], device, raw=raw)
        ids = sorted(n.id for n in graph.nodes)
        adjacency = {i: set() for i in ids}
        for edge in graph.edges:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
        for node_id in rng.sample(ids, min(per_graph, len(ids))):
            flipped, _ = embed_graphs(gnn, tensorizer, [flip_graph(graph, node_id)], device, raw=raw)
            dist = 1 - torch.nn.functional.cosine_similarity(base, flipped, dim=-1)
            index = {n: i for i, n in enumerate(ids)}
            own.append(dist[index[node_id]].item())
            if adjacency[node_id]:
                neighbour.append(sum(dist[index[n]].item() for n in adjacency[node_id]) / len(adjacency[node_id]))
            other = rng.choice([n for n in ids if n != node_id])
            between.append(1 - torch.nn.functional.cosine_similarity(
                base[index[node_id]], base[index[other]], dim=0).item())
    mean = lambda values: sum(values) / max(1, len(values))
    return {
        "flip_own_cosine_shift": mean(own),
        "flip_neighbour_cosine_shift": mean(neighbour),
        "between_two_nodes_cosine_distance": mean(between),
        "flip_own_shift_over_between": mean(own) / max(1e-9, mean(between)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train-graphs", type=int, default=600)
    parser.add_argument("--val-graphs", type=int, default=200)
    parser.add_argument("--flip-graphs", type=int, default=60)
    parser.add_argument("--random-seeds", type=int, default=3,
                        help="extra random-GNN references (seeds 100..) to show seed-to-seed spread")
    parser.add_argument("gnns", nargs="+", help="label=path/to/gnn.pt | label=random:SEED | label=raw")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_graphs = distinct_graphs(_static_path(config, "train"), args.train_graphs)
    val_graphs = distinct_graphs(_static_path(config, "validation"), args.val_graphs)
    embedding_lm, tokenizer = _embedding_only_lm(str(config.section("model")["lm_name"]))
    table = embedding_lm.get_input_embeddings()

    def target_matrix(graphs):
        vectors = []
        for graph in graphs:
            for node in sorted(graph.nodes, key=lambda n: n.id):
                ids = tokenizer.encode(node_text(node), add_special_tokens=False)
                vectors.append(table(torch.tensor(ids, device=device)).float().mean(0))
        return torch.stack(vectors)

    train_y, val_y = target_matrix(train_graphs), target_matrix(val_graphs)
    report = {"train_graphs": len(train_graphs), "val_graphs": len(val_graphs)}
    specs = list(args.gnns) + [f"random_s{k}=random:{100 + k}" for k in range(args.random_seeds)]
    for spec in specs:
        label, _, source = spec.partition("=")
        raw = source == "raw"
        state = None
        # A private copy per GNN: the checkpoint's depth must not leak into the next spec.
        gnn_config = copy.deepcopy(config)
        if not raw and not source.startswith("random:"):
            state = torch.load(source, map_location="cpu", weights_only=True)
            # The checkpoint decides the depth (e.g. the 6-layer variants).
            gnn_config.section("model")["graph_layers"] = 1 + max(
                int(key.split(".")[1]) for key in state if key.startswith("self_layers.")
            )
        if source.startswith("random:"):
            torch.manual_seed(int(source.split(":")[1]))
        graph_config, gnn, tensorizer = _neural_components(gnn_config)
        if state is not None:
            gnn.load_state_dict(state)
        gnn.to(device).eval()
        tx, _ = embed_graphs(gnn, tensorizer, train_graphs, device, raw=raw)
        vx, sizes = embed_graphs(gnn, tensorizer, val_graphs, device, raw=raw)
        entry = probe(tx, vx, train_graphs, val_graphs, sizes, train_y, val_y, device)
        entry.update(flip_sensitivity(gnn, tensorizer, val_graphs[: args.flip_graphs], device, raw=raw))
        report[label] = entry
        print(f"\n{label}")
        for key, value in entry.items():
            print(f"  {key:36s} {value:.3f}")
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
