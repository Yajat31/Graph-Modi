"""Configuration loading with small, explicit validation rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    path: Path
    values: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.values.get("seed", 42))

    @property
    def output_dir(self) -> Path:
        return Path(self.values.get("output_dir", "outputs/default"))

    @property
    def data_dir(self) -> Path:
        """Shared session JSONL directory; defaults to ``output_dir / "data"``."""
        if "data_dir" in self.values:
            return Path(self.values["data_dir"])
        return self.output_dir / "data"

    def section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"Configuration section {name!r} must be a mapping")
        return value


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {config_path}")
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Top-level configuration must be a mapping")
    return ExperimentConfig(path=config_path, values=values)
