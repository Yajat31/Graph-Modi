"""GraphToken-style GLM: end-to-end GNN + projector training on static QA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graph_modi.models.tea_glm import TEAGLM, TEAGLMBackend, require_tea_dependencies
from graph_modi.training import ProjectorTrainingConfig, TrainingExample, train_projector

GRAPH_TOKEN_CHECKPOINT_FORMAT = "graph-modi-graph-token-v1"


@dataclass(frozen=True, slots=True)
class GraphTokenConfig:
    prefix_tokens: int = 10
    train_gnn: bool = True


def configure_graph_token(model: TEAGLM, *, train_gnn: bool = True) -> TEAGLM:
    """Enable joint GNN + projector training (no TEA alignment phase)."""
    require_tea_dependencies()
    model.set_gnn_frozen(not train_gnn)
    for parameter in model.language_model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.projector.parameters():
        parameter.requires_grad_(True)
    if train_gnn:
        for parameter in model.gnn.parameters():
            parameter.requires_grad_(True)
    return model


def train_graph_token(
    model: TEAGLM,
    examples: list[TrainingExample],
    config: ProjectorTrainingConfig,
    *,
    data_split: dict[str, Any] | None = None,
) -> Any:
    configure_graph_token(model, train_gnn=True)
    return train_projector(model, examples, config, data_split=data_split)


class GraphTokenBackend(TEAGLMBackend):
    """Inference backend identical to TEA-GLM."""

    pass
