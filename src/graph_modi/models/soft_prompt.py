"""Learned soft-prompt baseline: text/history only, no GNN, no topology channel.

README section 5: "Add a learned 10-vector soft-prompt baseline: one shared
10 x d_model parameter matrix trained over all training examples, with no GNN
or projector." This mirrors TEAGLM's tokenization/generation machinery
(models/tea_glm.py) but replaces the graph-conditioned prefix with one
constant learned embedding sequence shared across every example.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from graph_modi.models.base import ModelInput
from graph_modi.models.tea_glm import (
    _object_name,
    _sha256_json,
    require_tea_dependencies,
    save_checkpoint_metadata,
    tokenizer_identity,
)
from graph_modi.schema import AttributedGraph, GraphEdit

_TEA_IMPORT_ERROR: ImportError | None = None
try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on optional environment
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TEA_IMPORT_ERROR = exc

_Module = nn.Module if nn is not None else object

SOFT_PROMPT_CHECKPOINT_FORMAT = "graph-modi-soft-prompt-v1"


@dataclass(frozen=True, slots=True)
class SoftPromptConfig:
    prefix_tokens: int = 10
    max_sequence_length: int = 512
    prompt_template: str = "Question: {question}\nAnswer:"
    add_bos_token: bool = True
    add_eos_token: bool = True

    def __post_init__(self) -> None:
        if self.prefix_tokens <= 0:
            raise ValueError("prefix_tokens must be positive")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if "{question}" not in self.prompt_template:
            raise ValueError("prompt_template must contain {question}")


def _tensor_sha256(tensor: Any) -> str:
    import hashlib

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(list(value.shape)).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class SoftPromptGLM(_Module):
    """Frozen causal LM conditioned on one learned, graph-independent prefix."""

    def __init__(self, *, language_model: Any, tokenizer: Any, config: SoftPromptConfig | None = None) -> None:
        require_tea_dependencies()
        super().__init__()
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.config = config or SoftPromptConfig()
        lm_hidden_dim = int(language_model.get_input_embeddings().embedding_dim)
        self.soft_prompt = nn.Parameter(torch.empty(self.config.prefix_tokens, lm_hidden_dim))
        nn.init.normal_(self.soft_prompt, std=0.02)
        for parameter in self.language_model.parameters():
            parameter.requires_grad_(False)
        self.language_model.eval()

    @classmethod
    def from_pretrained(
        cls,
        lm_name_or_path: str,
        *,
        prefix_tokens: int = 10,
        config: SoftPromptConfig | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        torch_dtype: Any = None,
    ) -> SoftPromptGLM:
        require_tea_dependencies()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Soft-prompt baseline requires Transformers. Install `graph-modi[tea]`."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            lm_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        language_model = AutoModelForCausalLM.from_pretrained(
            lm_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define either pad_token_id or eos_token_id")
            tokenizer.pad_token = tokenizer.eos_token
        return cls(
            language_model=language_model,
            tokenizer=tokenizer,
            config=config or SoftPromptConfig(prefix_tokens=prefix_tokens),
        )

    def train(self, mode: bool = True) -> SoftPromptGLM:
        super().train(mode)
        self.language_model.eval()
        return self

    @property
    def device(self) -> Any:
        return self.soft_prompt.device

    def _lm_autocast(self) -> Any:
        lm_dtype = next(self.language_model.parameters()).dtype
        if self.device.type == "cuda" and lm_dtype in (torch.bfloat16, torch.float16):
            return torch.autocast(device_type="cuda", dtype=lm_dtype)
        return contextlib.nullcontext()

    def _prefix(self) -> Any:
        embedding_dtype = self.language_model.get_input_embeddings().weight.dtype
        return self.soft_prompt.to(dtype=embedding_dtype)

    def _encode_text(self, text: str, *, answer: bool) -> list[int]:
        token_ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        if not answer and self.config.add_bos_token and self.tokenizer.bos_token_id is not None:
            token_ids.insert(0, int(self.tokenizer.bos_token_id))
        if answer and self.config.add_eos_token and self.tokenizer.eos_token_id is not None:
            token_ids.append(int(self.tokenizer.eos_token_id))
        return token_ids

    def _truncate(self, prompt_ids: list[int], answer_ids: list[int]) -> tuple[list[int], list[int]]:
        available = self.config.max_sequence_length - self.config.prefix_tokens
        if available <= 0:
            raise ValueError("max_sequence_length must exceed prefix_tokens")
        if len(answer_ids) >= available:
            return [], answer_ids[:available]
        prompt_budget = available - len(answer_ids)
        return prompt_ids[-prompt_budget:], answer_ids

    def forward(self, *, prompts: Sequence[str], answers: Sequence[str]) -> Any:
        """Compute causal loss over answer tokens, masking prompt + soft-prompt prefix."""
        if not prompts or len(prompts) != len(answers):
            raise ValueError("prompts and answers must be non-empty and equally sized")
        embedding = self.language_model.get_input_embeddings()
        prefix = self._prefix()
        sequences: list[Any] = []
        label_rows: list[Any] = []
        for prompt, answer in zip(prompts, answers, strict=True):
            prompt_ids, answer_ids = self._truncate(
                self._encode_text(prompt, answer=False),
                self._encode_text(answer, answer=True),
            )
            if not answer_ids:
                raise ValueError("An answer produced no tokens")
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
            answer_tensor = torch.tensor(answer_ids, dtype=torch.long, device=self.device)
            parts = []
            if prompt_ids:
                parts.append(embedding(prompt_tensor))
            parts.extend([prefix, embedding(answer_tensor)])
            sequences.append(torch.cat(parts, dim=0))
            masked = len(prompt_ids) + self.config.prefix_tokens
            label_rows.append(
                torch.cat(
                    [
                        torch.full((masked,), -100, dtype=torch.long, device=self.device),
                        answer_tensor,
                    ]
                )
            )
        max_length = max(sequence.shape[0] for sequence in sequences)
        hidden_dim = sequences[0].shape[-1]
        inputs_embeds = torch.zeros(
            (len(sequences), max_length, hidden_dim), dtype=sequences[0].dtype, device=self.device
        )
        attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long, device=self.device)
        labels = torch.full((len(sequences), max_length), -100, dtype=torch.long, device=self.device)
        for index, (sequence, row_labels) in enumerate(zip(sequences, label_rows, strict=True)):
            length = sequence.shape[0]
            inputs_embeds[index, :length] = sequence
            attention_mask[index, :length] = 1
            labels[index, :length] = row_labels
        with self._lm_autocast():
            return self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )

    @torch.no_grad() if torch is not None else (lambda function: function)
    def generate(self, prompt: str, *, max_new_tokens: int = 64, **generation_kwargs: Any) -> str:
        prompt_ids, _ = self._truncate(self._encode_text(prompt, answer=False), [])
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        prompt_embeds = self.language_model.get_input_embeddings()(prompt_tensor)
        inputs_embeds = torch.cat([prompt_embeds, self._prefix()], dim=0).unsqueeze(0)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)
        with self._lm_autocast():
            generated = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs,
            )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    @staticmethod
    def _left_pad_embeds(sequences: list[Any], device: Any) -> tuple[Any, Any]:
        max_length = max(sequence.shape[0] for sequence in sequences)
        hidden_dim = sequences[0].shape[-1]
        batch = torch.zeros((len(sequences), max_length, hidden_dim), dtype=sequences[0].dtype, device=device)
        mask = torch.zeros((len(sequences), max_length), dtype=torch.long, device=device)
        for index, sequence in enumerate(sequences):
            length = sequence.shape[0]
            batch[index, max_length - length :] = sequence
            mask[index, max_length - length :] = 1
        return batch, mask

    @torch.no_grad() if torch is not None else (lambda function: function)
    def generate_batch(
        self, prompts: Sequence[str], *, max_new_tokens: int = 64, **generation_kwargs: Any
    ) -> list[str]:
        if not prompts:
            raise ValueError("prompts must be non-empty")
        embedding = self.language_model.get_input_embeddings()
        prefix = self._prefix()
        sequences = []
        for prompt in prompts:
            prompt_ids, _ = self._truncate(self._encode_text(prompt, answer=False), [])
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
            sequences.append(torch.cat([embedding(prompt_tensor), prefix], dim=0))
        inputs_embeds, attention_mask = self._left_pad_embeds(sequences, self.device)
        with self._lm_autocast():
            generated = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs,
            )
        return [self.tokenizer.decode(row, skip_special_tokens=True).strip() for row in generated]


def soft_prompt_checkpoint_metadata(
    model: SoftPromptGLM,
    *,
    training_config: dict[str, Any],
    data_split: dict[str, Any],
    data_sha256: str,
    step: int,
) -> dict[str, Any]:
    lm_config = model.language_model.config.to_dict()
    return {
        "format": SOFT_PROMPT_CHECKPOINT_FORMAT,
        "step": step,
        "soft_prompt": {
            "class": _object_name(model),
            "config": asdict(model.config),
            "state_sha256": _tensor_sha256(model.soft_prompt),
        },
        "language_model": {
            "class": _object_name(model.language_model),
            "name_or_path": str(getattr(model.language_model.config, "_name_or_path", "")),
            "config_sha256": _sha256_json(lm_config),
        },
        "tokenizer": tokenizer_identity(model.tokenizer),
        "training_config": dict(training_config),
        "data_split": dict(data_split),
        "data_sha256": data_sha256,
    }


def save_soft_prompt_checkpoint(checkpoint_dir: str | Path, model: SoftPromptGLM, metadata: dict[str, Any]) -> None:
    destination = Path(checkpoint_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(model.soft_prompt.detach().cpu(), destination / "soft_prompt.pt")
    save_checkpoint_metadata(destination / "metadata.json", metadata)


def load_soft_prompt_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format") != SOFT_PROMPT_CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported soft-prompt checkpoint metadata: {path}")
    return value


def load_soft_prompt_checkpoint(model: SoftPromptGLM, *, checkpoint_dir: str | Path) -> None:
    """Load a trained soft-prompt matrix, rejecting shape/backbone mismatches."""
    checkpoint_dir = Path(checkpoint_dir)
    metadata = load_soft_prompt_checkpoint_metadata(checkpoint_dir / "metadata.json")
    expected = soft_prompt_checkpoint_metadata(
        model, training_config={}, data_split={}, data_sha256="", step=0
    )
    if metadata.get("soft_prompt", {}).get("config") != expected["soft_prompt"]["config"]:
        raise ValueError("Checkpoint soft-prompt config does not match the active model")
    if metadata.get("language_model") != expected["language_model"]:
        raise ValueError("Checkpoint language model does not match the active model")
    state = torch.load(checkpoint_dir / "soft_prompt.pt", map_location="cpu", weights_only=True)
    with torch.no_grad():
        model.soft_prompt.copy_(state.to(model.soft_prompt.device))
    expected_hash = metadata["soft_prompt"].get("state_sha256")
    if not expected_hash or _tensor_sha256(model.soft_prompt) != expected_hash:
        raise ValueError("soft-prompt weights do not match checkpoint metadata")


class SoftPromptBackend:
    """GraphBackend adapter for a trained soft-prompt model.

    Never reads ``model_input.current_graph``/``encoded_graph`` in ``answer`` -
    that is the whole point of this baseline (no topology channel).
    """

    def __init__(self, model: SoftPromptGLM, *, max_new_tokens: int = 64) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.encode_calls = 0

    def encode(self, graph: AttributedGraph) -> str:
        from graph_modi.graph.executor import graph_fingerprint

        self.encode_calls += 1
        return graph_fingerprint(graph)

    def predict_edit(self, utterance: str, graph: AttributedGraph) -> GraphEdit | None:
        from graph_modi.models.base import SymbolicMockBackend

        return SymbolicMockBackend().predict_edit(utterance, graph)

    def predict_edit_batch(
        self, items: Sequence[tuple[str, AttributedGraph]]
    ) -> list[GraphEdit | None]:
        return [self.predict_edit(utterance, graph) for utterance, graph in items]

    def answer(self, model_input: ModelInput) -> str | None:
        if model_input.condition in {"structure_only", "question_only"}:
            return None
        prompt = self.model.config.prompt_template.format(question=model_input.question)
        return self.model.generate(prompt, max_new_tokens=self.max_new_tokens)

    def answer_batch(self, model_inputs: Sequence[ModelInput]) -> list[str | None]:
        results: list[str | None] = [None] * len(model_inputs)
        indices: list[int] = []
        prompts: list[str] = []
        for index, model_input in enumerate(model_inputs):
            if model_input.condition in {"structure_only", "question_only"}:
                continue
            indices.append(index)
            prompts.append(self.model.config.prompt_template.format(question=model_input.question))
        if prompts:
            generated = self.model.generate_batch(prompts, max_new_tokens=self.max_new_tokens)
            for index, output in zip(indices, generated, strict=True):
                results[index] = output
        return results
