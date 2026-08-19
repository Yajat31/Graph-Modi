PYTHON ?= .venv/bin/python
UV ?= uv

.PHONY: setup setup-tea test lint typecheck smoke check

setup:
	$(UV) sync --extra dev

setup-tea:
	$(UV) sync --extra dev --extra tea

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

typecheck:
	$(UV) run mypy src/graph_modi

smoke:
	$(UV) run graph-modi generate --config configs/smoke.yaml
	$(UV) run graph-modi pretrain-gnn --config configs/smoke.yaml
	$(UV) run graph-modi train-projector --config configs/smoke.yaml
	$(UV) run graph-modi evaluate --config configs/smoke.yaml

check: lint typecheck test smoke
