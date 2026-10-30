# Documents

Durable record of the project, kept separate from `outputs/` (gitignored).

```text
documents/
  README.md
  interim_proposal.pdf
  mid-report/                LaTeX source of the report (main.tex, custom.bib, figures/)
  experiments/
    README.md
    log.md                   one row per run whose result was used
    benchmark_v4_1_2026-10-30.md   benchmark design, gates, local ceilings
    v4_run_checklist.md      GPU ladder, go/no-go, pre-registered slices
    results/<run-id>/        small summary JSON per run
```

Sibling trees: [`datasets/`](../datasets/) (regenerated JSONL + manifest), [`configs/`](../configs/),
`outputs/` (gitignored).

## Index

| Document | One line |
|---|---|
| [benchmark_v4_1_2026-10-30.md](experiments/benchmark_v4_1_2026-10-30.md) | metro_v4 (generator v4.1) design, acceptance gates, learned local ceilings. |
| [v4_run_checklist.md](experiments/v4_run_checklist.md) | GPU run ladder, go/no-go, required reporting, pre-registered slices. |
| [log.md](experiments/log.md) | Run log with key results. |
| [mid-report/main.tex](mid-report/main.tex) | Report source. |
