#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

EXTRAS=(--extra dev)
if [[ "${1:-}" == "--tea" ]]; then
  EXTRAS+=(--extra tea)
fi

uv sync "${EXTRAS[@]}"
echo "Environment ready. Run: uv run graph-modi --help"
