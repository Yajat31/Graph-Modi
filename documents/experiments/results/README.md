One subfolder per run, named `<run-id>` matching a row in
[`../log.md`](../log.md). Each subfolder holds only small summary JSON —
audit summaries, evaluation summaries, checkpoint metadata — never full
session JSONL (those live under [`datasets/`](../../../datasets/)) and never
weight files from gitignored `outputs/`. See [`../README.md`](../README.md).
