# Decision records

This folder holds short, permanent records of choices that constrain later
work and would be expensive to silently change: locking a graph
distribution, freezing a dataset scale, changing the set of evaluation
conditions, picking a GNN/projector size, changing the go/no-go bar, and
similar. It is the durable counterpart to the throwaway planning
conversations that led to each choice.

Write one when a choice:

- fixes a parameter or design that later experiments will assume is stable
  (e.g. the graph distribution and its parameters), or
- reverses or narrows something the [README.md](../../README.md) or an
  earlier decision stated, or
- would otherwise force someone to re-read a whole chat transcript to find
  out "why is it like this."

Do not write one for routine code changes, bug fixes, or anything already
obvious from reading the code and tests.

## Format

One file per decision: `NNNN-short-title.md`, numbered sequentially
starting at `0001`, using the [template](template.md):

```text
documents/decisions/
  README.md
  template.md
  0001-graph-distribution.md
  0002-...
```

Each record has:

- **Status** — `proposed`, `accepted`, or `superseded by NNNN`.
- **Context** — the problem and the options that were on the table (link to
  [`documents/distributions/`](../distributions/) or another supporting
  document instead of repeating it).
- **Decision** — the exact choice, including locked parameters/config
  values, stated precisely enough to diff against later.
- **Consequences** — what this rules out, what it leaves open, and what
  would have to happen to revisit it.

Never edit the Decision section of an accepted record to change the
choice — write a new record and mark the old one superseded. Editing a
record's prose for clarity is fine; editing what was actually decided is
not.

## Relationship to the experiment log

A decision record says what was chosen and why. The
[experiment log](../experiments/log.md) says what was run and what came out
of it. A decision often cites the experiment run(s) that justified it; an
experiment log entry often cites the decision it was testing.
