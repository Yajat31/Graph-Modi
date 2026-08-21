# 0001. Lock the graph distribution to a Watts-Strogatz metro ring

- **Status:** accepted
- **Date:** 2026-08-21
- **Author(s):** Team Hansraj

## Context

The first experiment is an in-distribution feasibility test: does applying a
typed edit and re-encoding the graph beat keeping the initial graph plus a
short text history? Every eval condition in
[`src/graph_modi/evaluation/runner.py`](../../src/graph_modi/evaluation/runner.py)
compares against the same sessions, so the graph distribution decides whether
those comparisons carry any signal.

Ten candidate distributions were written up with pros, cons, and generation
sketches in
[`documents/distributions/graph_distributions.md`](../distributions/graph_distributions.md).

The previous generator (a random recursive tree plus independent `p=0.12`
chords) had a measured defect: because density was fixed rather than
normalized by node count, it was a different graph family at each scale.
Mean degree was 2.5 at smoke scale (`n=8-10`) and 4.3 at full scale
(`n=15-30`), and closing a station on the shortest path made the pair
unreachable 52% of the time at smoke scale but only 6% at full scale. A
dataset where closures usually sever the map rewards a degenerate shortcut
("the user said closed, answer `unreachable`") that mimics graph reasoning
without performing it, and the stale-answer metric cannot distinguish the
two.

The deciding argument for Watts-Strogatz is that the property the experiment
depends on -- a graded, non-degenerate counterfactual -- becomes a structural
guarantee rather than a lucky draw. A `k=4` ring is 2-connected, so closing a
station leaves a longer specific route instead of destroying reachability.
Degree is also decoupled from `n`, so the CPU smoke set and the full run are
draws from the same family. This is a claim about methodological control, not
about realism: real transit networks are closer to planar line unions, and
that limitation is recorded below.

## Decision

Sample initial graphs from a rewired Watts-Strogatz ring lattice, named
`watts_strogatz_metro_v1`, implemented in `_make_graph` in
[`src/graph_modi/data/multiturn.py`](../../src/graph_modi/data/multiturn.py).

```yaml
data:
  distribution: watts_strogatz_metro_v1
  graph_degree: 4          # k, exact mean degree at every node count
  rewire_probability: 0.15 # p, per ring-lattice edge
  line_count: 4            # contiguous ring arcs, one per line name
  node_count_min: 15       # 8 in configs/smoke.yaml
  node_count_max: 30       # 10 in configs/smoke.yaml
```

Construction, in order:

1. Build a ring lattice over ring *positions*: position `i` connects to
   `i + d (mod n)` for `d` in `1..k/2`.
2. Rewire each lattice edge with probability `p`, rejecting self-loops and
   duplicate pairs. Edge count stays exactly `n * k / 2`.
3. Redraw (bounded retries) if the result is disconnected.
4. Permute ring positions onto node identifiers, so adjacency is **not**
   recoverable from node id or label arithmetic. Without this, `n3`-`n4`
   would almost always be an edge and a model could infer the graph from id
   arithmetic instead of reading the encoded graph.
5. Attributes: `status="open"` for all nodes at `t=0`; `accessible` drawn
   `Bern(0.5)`; `line` assigned from the contiguous ring arc a node's
   position falls in. The independently-sampled `zone` attribute from the
   previous generator is removed.
6. Edge relations: `track` for surviving lattice edges, `transfer` for
   rewired chords. `ADD EDGE` edits produced by the generator are explicitly
   labelled `transfer`, because `parse_edit` defaults an unspecified relation
   to `connected` and `_edge_key` includes relation in the identity of an
   edge.

`line` is now genuinely queried: `_queries` yields line-valued
`filtered_neighbor_count` questions. This needed no solver change, since
`answer_query` compares `attributes.get(query.attribute) == query.value`
generically and `render_question` already had a generic template.
Configs reject any `distribution` value other than the locked name, so a
stale config fails loudly instead of silently sampling something else.

## Measured characterization

From `scripts/characterize_distribution.py`, 200 graphs per scale, seed 42
(`unreachable`/`longer`/`same_length` is the outcome mix of closing an
interior station on a shortest path):

Full scale, `n=15-30`:
- mean degree exactly 4 (min 4, max 4); edges 30-60
- diameter mean 4.9; clustering mean 0.35
- bridges mean 0.02, articulation points mean 0.02
- 99.9% of pairs retain an alternate path
- closures: 0.1% unreachable, 56.3% longer, 43.5% same length
- shortest paths 2-8 stations, mean 3.5

Smoke scale, `n=8-10`:
- mean degree exactly 4 (min 4, max 4); edges 16-20
- diameter mean 2.5; clustering mean 0.48
- bridges mean 0.015, articulation points mean 0.015
- 99.6% of pairs retain an alternate path
- closures: 0.5% unreachable, 33.1% longer, 66.4% same length

The two scales are now the same family: degree is identical and the
degenerate-`unreachable` regime is gone at both (52% to 0.5% at smoke scale).

Generation stays valid end to end. `audit_sessions` reports `valid: true` for
all splits at both scales, with majority-answer accuracy 0.24-0.33 (not
degenerate). The mock smoke evaluation reproduces the expected sanity
pattern: `oracle_updated_graph` answer accuracy 1.00 with stale rate 0.00
against `cached_no_reencode` 0.00 with stale rate 1.00, `shuffled_graph` near
chance at 0.17, and execution-equivalent edit accuracy 1.00 for
`predicted_updated_graph`, which confirms the new `track`/`transfer`
relations survive the utterance-to-`parse_edit` round trip.

## Consequences

- **Reachability and cycle-membership counterfactuals become rare.** This is
  the direct flip side of choosing a 2-connected family. On a `k=4` ring
  every node lies on a cycle, so `cycle_membership` answers "yes" and never
  changes; a full-scale sample produced 71% `shortest_path`, 20%
  `filtered_neighbor_count`, 9% `reachability`, and no `cycle_membership`
  turns, because `_changed_query` falls back when the preferred type does not
  change. Listing those types in a config no longer guarantees coverage. Any
  future need for reachability coverage requires a lower degree, an edit type
  that deliberately cuts the graph, or a different family.
- **A substantial share of closures leave shortest-path length unchanged**
  (43.5% full scale, 66.4% smoke). Those turns are simply not selected as
  counterfactuals, so the rejection-sampling accept rate is lower for
  shortest-path questions than the raw closure rate suggests. Generation is
  still fast: 30 full-scale sessions in 0.4s.
- **Lines are equal-length arcs on a ring backbone**, a structural prior a
  model could exploit. A real line-union generator (option 9 in the
  candidates document) is the natural upgrade if multi-line reasoning becomes
  the research question.
- **`relation` is structural-only in v1.** No reasoning type reads it, so
  `track`/`transfer` shapes the graph and the hashed node/edge features but
  enters no question. Making it queryable requires solver work and would be
  a new decision.
- **`zone` is gone.** Any future zone question needs a distribution where
  zones are real communities (option 8, stochastic block model).
- **The out-of-distribution follow-up is now a single-knob sweep** on the
  same named family: `rewire_probability=0` for a pure ring, `graph_degree=2`
  for a near-cycle, or larger `node_count` ranges. That is a stronger claim
  than swapping generators between experiments.
- Revisiting this requires a new decision record superseding this one; do not
  edit the Decision section above.

## Related

- Candidate options: [`documents/distributions/graph_distributions.md`](../distributions/graph_distributions.md)
- Implementation: [`src/graph_modi/data/multiturn.py`](../../src/graph_modi/data/multiturn.py),
  [`src/graph_modi/cli.py`](../../src/graph_modi/cli.py), `scripts/characterize_distribution.py`
- Tests: [`tests/test_distribution.py`](../../tests/test_distribution.py)
- Released sessions: [`datasets/watts_strogatz_metro_v1/`](../../datasets/watts_strogatz_metro_v1/)
- Code state when accepted: branch `graphmodi-experiments`, parent commit `43cc523`
- Experiment log entries:
  [`qwen8b_projector-20260821-data`](../experiments/log.md)
- Superseded by: not superseded
