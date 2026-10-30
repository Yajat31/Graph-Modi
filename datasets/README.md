# Shared datasets

Versioned session JSONL that the team trains and evaluates on. Unlike `outputs/`
(gitignored checkpoints and scratch), the **recipes** for datasets live in git;
the JSONL itself is gitignored because it is hundreds of megabytes.

```text
datasets/
  README.md
  <dataset-id>/
    README.md          provenance (optional)
    manifest.json      sha256 + byte sizes of each JSONL (small; keep in sync)
    audit.json         validity and label distributions (optional)
    train.jsonl        gitignored — regenerate
    validation.jsonl
    test.jsonl
    sessions_*.jsonl
    static_*.jsonl
```

## Current datasets

| Dataset ID | Config | Status |
|---|---|---|
| `metro_v4_1` | [`configs/v4_data.yaml`](../configs/v4_data.yaml) | **Current, frozen** benchmark (`data/v4.py`, generator v4.1); manifest in `metro_v4_1/manifest.json`. Gate: `scripts/audit_v4_gates.py`. Final audit: `scripts/audit_v4_final.py`. |
| `metro_v4` | — | Generator v4.0, superseded by v4.1; manifest archived at `archive/datasets/metro_v4/`. |
| `metro_v3` | [`archive/configs/v3/v3_tea.yaml`](../archive/configs/v3/v3_tea.yaml) | Frozen historical benchmark. Not comparable with v4. |
| `metro_v3_big` | [`archive/configs/v3x_data.yaml`](../archive/configs/v3x_data.yaml) | 3× training data for the archived `v3x_data` variant. |

## Regenerate

```bash
# v4 (preferred)
uv run graph-modi generate --config configs/v4_data.yaml
# or generate + run acceptance gates into a work dir, then copy:
PYTHONPATH=src python3 scripts/audit_v4_gates.py --work datasets/metro_v4_1
git checkout -- datasets/metro_v4_1/manifest.json   # the gate script rewrites it
# verify sha256 against datasets/metro_v4_1/manifest.json (scripts/run_v4_ladder.sh does this)

# v3 (historical only)
uv run graph-modi generate --config archive/configs/v3/v3_tea.yaml
```

## Layout rules

- **`datasets/`** — released session files teammates load. JSONL gitignored; manifests tracked.
- **`outputs/`** — local GNN/projector checkpoints, evaluation dumps, smoke scratch. Gitignored.
- Configs set both `data_dir: datasets/<dataset-id>` and `output_dir: outputs/<run-name>`.

Regenerate only when the generator or locked parameters change, then update the
manifest and re-audit. Do not silently overwrite a shared folder mid-experiment.
