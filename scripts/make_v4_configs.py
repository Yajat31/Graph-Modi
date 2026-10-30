"""Write the frozen v4.1 run configs (configs/v4_*.yaml) from one shared recipe.

Usage:
  python3 scripts/make_v4_configs.py

The graph recipe: sum aggregation, no GNN dropout, joint GNN lr 1e-4, cosine schedule with warmup, and validation
every 100 steps with best-checkpoint selection. Every graph run trains for 1 epoch, so the warm
start differs from GraphToken s42 only in the pretrained GNN initialisation. Re-running this script is idempotent.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DATA_DIR = "datasets/metro_v4_1"
LM_NAME = "/data/lmw/models/Meta-Llama-3.1-8B-Instruct"

STATIC_TASKS = [
    "edge_exists",
    "reachability",
    "constrained_reachability",
    "cycle_membership",
    "node_degree",
    "filtered_neighbor_count",
    "shortest_path",
    "within_hops_count",
    "most_common_attribute_within_hops",
    "attribute_lookup",
    "attribute_check",
]
DYNAMIC_TASKS = STATIC_TASKS[:9]

DATA = {
    "distribution": "metro_v4",
    "static_train_graphs": 2000,
    "static_tuples_per_graph": 9,
    "counterfactual_graphs": 1000,
    "counterfactual_pairs_per_graph": 3,
    "static_validation_replicates": 6,
    "static_test_replicates": 12,
    "sessions": {"validation_replicates": 1, "test_replicates": 1},
    "static_tasks": STATIC_TASKS,
    "dynamic_tasks": DYNAMIC_TASKS,
}

GRAPH_MODEL = {
    "lm_name": LM_NAME,
    "tensorizer": "bert",
    "bert_model_name": "bert-base-uncased",
    "node_feature_size": 768,
    "graph_hidden_size": 1024,
    "graph_layers": 3,
    "graph_dropout": 0.0,
    "graph_aggregation": "sum",
    "prefix_tokens": 10,
    "projector_num_layers": 1,
    "trust_remote_code": False,
    "torch_dtype": "bfloat16",
    "gradient_checkpointing": True,
}

TRAINING = {
    "epochs": 1,
    "batch_size": 8,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "gradient_accumulation_steps": 3,
    "mixed_precision": "bf16",
    "save_every_steps": 500,
    "resume_from": None,
    "lr_schedule": "cosine",
    "warmup_steps": 200,
    "min_lr_ratio": 0.05,
    "eval_every_steps": 100,
    "val_max_examples": 1000,
    # Train on the same "Question: {q}\nAnswer:" prompt template the evaluator sends.
    "wrap_prompts": True,
}

GRAPH_CONDITIONS = [
    "oracle_updated_graph",
    "predicted_updated_graph",
    "structure_only",
    "frozen_graph_history",
    "shuffled_graph",
    "question_only",
    "stale_answer",
    "majority_answer",
    "tool_solver",
]

# Multi-turn sessions run on test only; validation is used for checkpoint selection, which the
# static validation split already covers.
EVALUATION = {
    "splits": ["test"],
    "batch_size": 32,
    "max_new_tokens": 32,
    "checkpoint": "best",
    "encoder_autocast": "match_training",
}


def _base(output_dir: str, seed: int, header: str, *, gpu: bool = True) -> dict:
    config = {
        "_header": header,
        "seed": seed,
        "output_dir": output_dir,
        "data_dir": DATA_DIR,
        "gpu_memory_fraction": 0.97,
        "data": copy.deepcopy(DATA),
    }
    if not gpu:  # data generation is CPU-only and must not need torch
        del config["gpu_memory_fraction"]
    return config


def graph_token(seed: int, *, warm: bool = False) -> dict:
    name = f"v4_graphtoken_warm_s{seed}" if warm else f"v4_graphtoken_s{seed}"
    config = _base(
        f"outputs/{name}",
        seed,
        "GraphToken (joint GNN + projector), sum aggregation"
        + (", GNN warm-started from v4_pretrain_sum" if warm else ""),
    )
    model = {"backend": "tea", "architecture": "graph_token", **GRAPH_MODEL}
    if warm:
        model["gnn_checkpoint"] = "outputs/v4_pretrain_sum/gnn/gnn.pt"
    config["model"] = model
    config["training"] = {
        "seed": seed,
        **TRAINING,
        "gnn_learning_rate_joint": 1e-4,
    }
    config["evaluation"] = {**EVALUATION, "conditions": list(GRAPH_CONDITIONS)}
    return config


def pretrain_sum() -> dict:
    config = _base(
        "outputs/v4_pretrain_sum",
        42,
        "Contrastive GNN pretraining only (pretrain-gnn); feeds v4_graphtoken_warm_s42",
    )
    config["model"] = {"backend": "tea", "architecture": "tea", **GRAPH_MODEL}
    config["training"] = {
        "seed": 42,
        "pretrain_objective": "contrastive",
        "gnn_epochs": 10,
        "gnn_learning_rate": 1e-3,
        "gnn_graphs_per_batch": 8,
        "contrastive_temperature": 0.07,
        "pretrain_dropout": 0.0,
    }
    return config


def tea_random_sum() -> dict:
    config = _base(
        "outputs/v4_tea_random_sum",
        42,
        "Frozen random sum GNN + trained projector: no-GNN-training control for GraphToken",
    )
    config["model"] = {
        "backend": "tea",
        "architecture": "tea",
        **GRAPH_MODEL,
        "gnn_init": "random",
    }
    config["training"] = {"seed": 42, **TRAINING}
    config["evaluation"] = {**EVALUATION, "conditions": list(GRAPH_CONDITIONS)}
    return config


def soft_prompt() -> dict:
    config = _base("outputs/v4_soft_prompt", 42, "Graph-blind soft-prompt control")
    config["model"] = {
        "backend": "soft_prompt",
        "lm_name": LM_NAME,
        "prefix_tokens": 10,
        "trust_remote_code": False,
        "torch_dtype": "bfloat16",
        "gradient_checkpointing": True,
    }
    training = {"seed": 42, **TRAINING}
    for key in (
        "lr_schedule",
        "warmup_steps",
        "min_lr_ratio",
        "eval_every_steps",
        "val_max_examples",
    ):
        training.pop(key)
    config["training"] = training
    config["evaluation"] = {
        **EVALUATION,
        "checkpoint": "final",
        "batch_size": 64,
        "conditions": ["soft_prompt", "soft_prompt_no_history"],
    }
    return config


def data_only() -> dict:
    return _base(
        "outputs/v4_data", 42, "Generate metro_v4 (generator v4.1) data and nothing else", gpu=False
    )


def write(name: str, config: dict) -> None:
    header = config.pop("_header")
    text = (
        f"# {header}\n# Generated by scripts/make_v4_configs.py; edit the script, not this file.\n"
    )
    text += yaml.safe_dump(config, sort_keys=False, default_flow_style=False, width=100)
    (CONFIGS / f"{name}.yaml").write_text(text, encoding="utf-8")
    print(f"wrote configs/{name}.yaml")


def main() -> None:
    write("v4_data", data_only())
    write("v4_soft_prompt", soft_prompt())
    for seed in (42, 43, 44):
        write(f"v4_graphtoken_s{seed}", graph_token(seed))
    write("v4_pretrain_sum", pretrain_sum())
    write("v4_graphtoken_warm_s42", graph_token(42, warm=True))
    write("v4_tea_random_sum", tea_random_sum())


if __name__ == "__main__":
    main()
