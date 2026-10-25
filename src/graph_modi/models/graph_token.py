"""GraphToken-style GLM (Perozzi et al., 2024): a GNN trained from random initialisation
jointly with the projector on static QA, with no alignment pretraining and a frozen LLM."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from graph_modi.models.tea_glm import TEAGLM, TEAGLMBackend
from graph_modi.training import ProjectorTrainingConfig, TrainingExample, train_projector


def train_graph_token(
    model: TEAGLM,
    examples: list[TrainingExample],
    config: ProjectorTrainingConfig,
    *,
    data_split: dict[str, Any] | None = None,
    val_examples: Any = None,
) -> Any:
    return train_projector(
        model,
        examples,
        replace(config, train_gnn=True),
        data_split=data_split,
        val_examples=val_examples,
    )


class GraphTokenBackend(TEAGLMBackend):
    """Inference backend identical to TEA-GLM."""

    pass
