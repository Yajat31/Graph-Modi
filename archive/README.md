# Archive

Superseded configs and one-off scripts kept so old `documents/experiments/log.md` rows still point at something. Nothing here is part of the current (metro_v4) workflow.

| Old path | New path |
|---|---|
| `configs/audit/*` | [`configs/audit/`](configs/audit/) |
| `configs/v3x_*.yaml` | [`configs/`](configs/) |
| `configs/v3_tea_b2.yaml` | [`configs/v3_tea_b2.yaml`](configs/v3_tea_b2.yaml) |
| `configs/v3_graphtoken_b1.yaml` | [`configs/v3_graphtoken_b1.yaml`](configs/v3_graphtoken_b1.yaml) |
| `configs/v3_graphtoken_b2.yaml` | [`configs/v3_graphtoken_b2.yaml`](configs/v3_graphtoken_b2.yaml) |
| `configs/v3_exact2x_graphtoken_b32.yaml` | [`configs/v3_exact2x_graphtoken_b32.yaml`](configs/v3_exact2x_graphtoken_b32.yaml) |
| `scripts/probe_*.py` | [`scripts/`](scripts/) |
| `scripts/diagnose_reachability.py` | [`scripts/diagnose_reachability.py`](scripts/diagnose_reachability.py) |
| `scripts/select_pretraining.py` | [`scripts/select_pretraining.py`](scripts/select_pretraining.py) |
| `scripts/audit_second_pass.py` | [`scripts/audit_second_pass.py`](scripts/audit_second_pass.py) |
| `scripts/summarize_v3_exact2x.py` | [`scripts/summarize_v3_exact2x.py`](scripts/summarize_v3_exact2x.py) |
| `scripts/make_audit_configs.py` | [`scripts/make_audit_configs.py`](scripts/make_audit_configs.py) |

`log.md` is append-only and keeps the old paths in historical rows. To re-run an archived config:

```bash
uv run graph-modi evaluate --config archive/configs/v3x_deep.yaml
PYTHONPATH=src python3 archive/scripts/probe_gnn.py ...
```
