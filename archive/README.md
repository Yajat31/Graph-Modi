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
| `configs/v3_*.yaml` (v3 headline runs) | [`configs/v3/`](configs/v3/) |
| `scripts/{analyze_exact2x,build_report_v3,make_figures_v3,make_report_figures,stats_v3,shortcut_audit}.py` | [`scripts/v3/`](scripts/v3/) |
| `scripts/audit_benchmark{,_deep,_extra}.py` (v3 label audits) | [`scripts/v3/`](scripts/v3/) |
| `datasets/metro_v4/manifest.json` (generator v4.0) | [`datasets/metro_v4/`](datasets/metro_v4/) |
| `documents/presentation/` (pre-audit, 2026-10-19) | [`documents/presentation/`](documents/presentation/) |

Archived on 2026-10-30 for the v4.1 code freeze. The `scripts/v3/` scripts resolve the repo root
themselves, so they still run from here, e.g.
`python3 archive/scripts/v3/make_report_figures.py`.

`log.md` is append-only and keeps the old paths in historical rows. To re-run an archived config:

```bash
uv run graph-modi evaluate --config archive/configs/v3x_deep.yaml
PYTHONPATH=src python3 archive/scripts/probe_gnn.py ...
```
