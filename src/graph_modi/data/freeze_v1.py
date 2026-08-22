"""Freeze v1 dataset, checkpoints, and reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha(repo: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def freeze_v1_manifest(
    repo_root: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)
    datasets = root / "datasets" / "watts_strogatz_metro_v1"
    results = root / "documents" / "experiments" / "results"
    manifest: dict[str, Any] = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(root),
        "branch": "prelim",
        "datasets": {},
        "results": {},
        "seeds": {"default": 42},
    }
    if datasets.is_dir():
        for path in sorted(datasets.glob("*.jsonl")):
            manifest["datasets"][path.name] = {
                "path": str(path.relative_to(root)),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        audit = datasets / "audit.json"
        if audit.is_file():
            manifest["datasets"]["audit.json"] = {
                "path": str(audit.relative_to(root)),
                "sha256": _sha256_file(audit),
            }
    if results.is_dir():
        for path in sorted(results.rglob("*.json")):
            relative = str(path.relative_to(root))
            manifest["results"][relative] = {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
    destination = Path(output_path or root / "documents" / "experiments" / "v1_freeze_manifest.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
