# Mutable State: Executable Representations for Graph Language Model Reasoning

## The one-sentence idea

A graph language model should not only *read* a graph — it should **edit its own representation as intermediate reasoning**, where each schema-checked edit becomes the next computational state after re-encoding, and the trajectory of states *is* the chain of thought.

---

## The key shift

Today, an LLM reasons roughly like:

```text
input → tokens → hidden states → answer
```

Even chain-of-thought is fundamentally:

```text
thought → text → thought → text → answer
```

GraphModi opened a third path:

```text
utterance → structured edit → new graph → re-encode → answer
```

This extension asks what happens when the model **owns that loop** — not just reacting to user revisions, but performing multiple state transitions *before* answering:

```text
S₀  ──Δ₁──>  S₁  ──Δ₂──>  S₂  ──Δ₃──>  S₃  →  answer
```

The state `S` is not a tool result retrieved from outside. It is the model's **persistent computational workspace** — part of the forward pass, rewritten during inference.

---

## What S is

Start simple. GraphModi is the first instantiation:

```text
S = G
```

where `G` is the attributed input graph the GNN encodes.

Then generalize:

```text
S = (G, D, R, C, ...)
```

| Component | Meaning | Example |
|-----------|---------|---------|
| `G` | Graph structure + node/edge attributes | Metro topology, open/closed stations |
| `D` | Pairwise distances | Shortest-path matrix, recomputed after each edit |
| `R` | Reachability / components | Which stations can still reach each other |
| `C` | Centrality or importance | Which station is most strategically critical |
| `P` | Candidate plans or hypotheses | Annotated search frontier |

The model performs operations like:

```text
SET_ATTRIBUTE    ADD_EDGE    DELETE_EDGE
BRANCH           CHECKOUT    DISCARD
MERGE            COMPARE
```

Reasoning becomes search over representational states, not search over token sequences.

---

## How this unifies the project arc

```text
GraphModi          S = G, trunk edits           "GLMs need a write path"
      ↓
Branching Worlds   branch / rollback / compare  "the write path is a simulator"
      ↓
Mutable State      multi-step S₀→S₁→…→Sₙ       "the state trajectory IS the reasoning"
      ↓
S = (G, D, …)      synchronized derived views   "the substrate generalizes"
```

GraphModi is **Experiment 1** in this paper, not the paper itself.

| Phase | What it proves |
|-------|----------------|
| **GraphModi** (interim) | Frozen graph fails; oracle edit + re-encode succeeds; self-edit gap isolates the write bottleneck |
| **Branching Worlds** | Same loop on branches answers counterfactuals, retrospectives, and planning rollouts |
| **Mutable State** | Multi-step state transitions before answering beat matched-compute NL CoT as horizon grows |
| **S = (G, D)** | Derived views recomputed post-edit show the substrate generalizes beyond raw adjacency |

See [branching-worlds.md](branching-worlds.md) for the branching/simulation layer in detail.

---

## How this differs from Branching Worlds

Both extensions use the same edit-and-re-encode loop. They differ in **scope** and **what question they ask**.

| | **Branching Worlds** | **Mutable State** |
|---|---------------------|-------------------|
| **Core question** | Can the write path *simulate* alternative worlds? | Can the write path *be* the model's reasoning? |
| **Who drives edits** | Mostly the user ("what if B closed?") | The model, as intermediate steps before answering |
| **Where edits land** | Trunk (real) vs branch (speculative) | A trajectory `S₀ → S₁ → … → Sₙ` the model builds itself |
| **Typical session** | One question → one branch edit → answer | Many state transitions → compare/rollback → answer |
| **Main capability** | Counterfactuals, time travel, parallel rollouts | Multi-step search over states; inspectable reasoning trajectories |
| **State substrate** | `S = G` only | Starts at `S = G`, generalizes to `S = (G, D, R, …)` |
| **What it adds architecturally** | Version tree (`BRANCH`, `CHECKOUT`, `DISCARD`) | Multi-step edit loop owned by the model, not just reactive updates |

**Branching Worlds** answers: once a GLM can edit its graph, can it use *copies* of that graph to reason about hypotheticals, the past, and planning candidates?

**Mutable State** answers: can a GLM reason by *editing its representation* the way chain-of-thought reasons by generating text — with the state trajectory as the trace?

Branching Worlds is a **mechanism** inside the Mutable State picture. You need branching to simulate and compare candidate worlds; Mutable State is the broader claim that structured state edits are a viable form of reasoning at all.

---

## Three ways of reasoning (and why ours is different)

### Chain-of-thought

```text
LLM → text thought → LLM → text thought → answer
```

Intermediate state lives in natural language. Opaque, unverifiable, degrades with depth.

### Tool use

```text
LLM → external tool → result text → LLM → answer
```

State is external and returned as text. The model does not *become* the new state.

### Mutable state (this work)

```text
LLM → state edit → representation update → re-encode → LLM → … → answer
```

The edit is applied to the representation the GNN encodes. The next forward pass literally reads the model's own intermediate world. The trajectory is inspectable:

```text
Step 0:  G₀
Step 1:  G₁ = G₀ − edge(B,C)        [branch: close B-C]
Step 2:  G₂ = G₁ + edge(A,D)        [branch: add bypass]
Step 3:  checkout(G₀)                [rollback]
Step 4:  compare(G₁, G₂) → answer
```

---

## The killer example

> "Which station should we close to minimize disruption while keeping every neighborhood connected?"

A conventional LLM must juggle connectivity, alternative paths, distances, affected neighborhoods, and multiple candidate interventions — all inside hidden activations.

A mutable-state GLM does:

```text
                    Current state S₀
                          │
                  identify candidates
                          │
           ┌──────────────┼──────────────┐
           ↓              ↓              ↓
        close A        close B        close C
           ↓              ↓              ↓
        S_A'           S_B'           S_C'
     re-encode       re-encode       re-encode
           ↓              ↓              ↓
      evaluate        evaluate        evaluate
           └──────────────┼──────────────┘
                          ↓
                       compare
                          ↓
                        answer
```

This is **search over representational states**. The model isn't generating a reasoning trace in text. It is **executing reasoning on a mutable state representation**.

---

## The central experiment

Build a task where **textual reasoning becomes difficult** but structured state makes it easy.

Give the model a graph with ~50–100 nodes and ask:

> "Find a sequence of interventions satisfying all these constraints."

Or: compositional hypotheticals requiring 2–4 stacked edits before the answer is knowable.

Then compare:

| Condition | Mechanism |
|-----------|-----------|
| **A. Chain-of-thought** | Model writes reasoning in text |
| **B. CoT + GraphRAG** | Retrieve/serialize graph information in text |
| **C. GraphModi** | One edit → updated graph → answer |
| **D. Mutable state** | Branch, edit, inspect, rollback, compare — multiple state transitions |

**Hypothesis:**

```text
structured state reasoning  >  text-only reasoning
```

…especially as reasoning horizon `d` increases:

```text
reasoning depth d:     1    2    4    8    16

CoT accuracy:         ████████████░░░░░░░░░░░░
mutable-state:        ████████████████████████
```

The interesting result is not a single-edit win on a small metro graph — it is a systematic gap that widens as reasoning depth increases.

---

## Mechanistic evaluation (not just final accuracy)

Mutable state gives us something CoT cannot: **auditable reasoning trajectories**.

| Metric | What it measures |
|--------|------------------|
| `edit_accuracy` | Parsed Δ matches gold edit |
| `state_consistency` | Applied state fingerprint matches oracle |
| `branch_validity` | Speculative branches don't contaminate trunk |
| `rollback_recovery` | After a bad edit + checkout, answer recovers |
| `trajectory_length` | State transitions before answer |
| `horizon_degradation_slope` | Δaccuracy / Δdepth per condition |

We can publish state trajectories `S₀, S₁, S₂, …` alongside answers. That is mechanistic evaluation of reasoning, not answer-only benchmarking.

---

## Grounding: what exists, what doesn't (2024–2026)

**Verdict: adjacent with a gap.** The broad thesis ("writable state beats NL CoT") is crowded. The narrow mechanism is not.

| Neighbor | What they do | What they don't do |
|----------|--------------|-------------------|
| **CoEvoT** (2026) | Soft residual rewriting of graph tokens per thought | No discrete world-graph mutation; no full GNN re-encode of edited G |
| **Map-of-Actions** (ACL 2026) | Schema-checked ADD/EDIT/DELETE/ROLLBACK on reasoning DAGs | Text-side substrate; no GLM input-graph re-encode |
| **Agentic Graph Token** (2026) | Per-step GNN re-encode of graph views | Read-only; no write path |
| **Coconut** (2024) | Writable continuous latent thoughts | No external structured substrate |
| **GTEval** (2026) | — | Shows graph-token LLMs may be structure-blind — direct threat |

**Nobody** does: schema-checked discrete edits to the GLM's **input world graph**, full GNN re-encode as next state, plus branch/rollback/compare over a version tree for counterfactual/planning QA.

**Do not pitch:** *"LLMs benefit from writable structured state."* Coconut, MoA, and CoEvoT will cite.

**Pitch instead:**

> A graph language model reasons more reliably over counterfactual, retrospective, and planning questions when it emits schema-checked edits to its own input graph, re-encodes each resulting world through the graph encoder, and searches over versioned branches — and this substrate-update loop degrades more slowly than matched-compute NL chain-of-thought as the number of sequential reasoning steps grows.

---

## Mandatory baselines (the paper lives or dies here)

1. **Frozen G + hypothetical in text** — in-context imagination
2. **Apply edit, serialize edited graph as text, no GNN re-encode** — is the encoder channel necessary?
3. **Oracle edit + re-encode** — reading upper bound
4. **Symbolic replay / solver** — non-neural ceiling
5. **Matched-compute NL CoT** — same token/step budget as edit loop
6. **Shuffled / noop / invalid-edit controls** — is gain content-dependent?

If (2) matches (3), the GNN path isn't earning its keep. If the solver solves everything, we need language-heavy or preference questions mixed in.

---

## Kill switch (run before building everything)

Before investing in branching infrastructure:

| Probe | Pass | Fail |
|-------|------|------|
| Oracle `DEL edge` on branch + re-encode | QA answer changes correctly | GTEval structure-blindness wins → pivot away from GLM-centric claim |
| Shuffled-edge oracle edit + re-encode | QA does *not* change | Gain is not content-dependent → mechanism is fake |
| Textualized edited graph (no re-encode) | Strictly worse than re-encode on compositional items | Re-encode channel adds no value |

If the kill switch fails, the honest pivot is symbolic planning over edits (Map-of-Actions territory), not a GLM paper.

---

## Experiment ladder

### Phase 0 — GraphModi interim (S = G, trunk)

Finish the existing benchmark. Confirm: frozen graph fails, oracle re-encode succeeds, self-edit gap isolates write bottleneck.

*Deliverable: interim result that motivates everything below.*

### Phase 1 — Kill switch

~50 counterfactual items, mock backend first, then real GLM. Three conditions from the table above.

### Phase 2 — Branching Worlds (S = G, version tree)

Extend the edit loop with `BRANCH`, `CHECKOUT`, `DISCARD` alongside `SET/ADD/DEL`.

- **Rung 1:** Counterfactual QA
- **Rung 2:** Retrospective QA (time travel via checkout)
- **Rung 3:** Compositional hypotheticals (2–3 stacked edits)

### Phase 3 — Mutable state reasoning loop

Allow multiple state transitions before answering. Branch-per-candidate, re-encode, evaluate, compare.

Task: planning over edit space ("which closure minimizes disruption?").

### Phase 4 — Horizon scaling

Items with controlled reasoning depth `d ∈ {1, 2, 4, 8}`:

- `d` stacked hypotheticals, or
- `d` candidate interventions to compare, or
- `d` sequential constraint checks

Plot accuracy vs `d` for CoT vs mutable-state.

### Phase 5 — S = (G, D) (generalization)

Recompute distance view `D` after each edit. Tests whether mutable state generalizes to **multiple synchronized views** — connects to the v1 supplementary-state idea, but only after Phases 1–4 hold.

---

## What would kill this (be honest)

- Oracle edit + re-encode doesn't beat textualized edit on compositional items → re-encode channel isn't earning its keep
- Horizon scaling shows CoT and mutable-state degrade equally → no fundamental advantage
- Self-edit accuracy is so low the oracle upper bound never translates → paper becomes "how to teach GLMs to emit edits" (narrower, still publishable)
- Symbolic solver solves 100% of eval items → need mixed language-heavy questions

---

## Paper structure (one arc)

| Section | Content |
|---------|---------|
| **1. Intro** | LLM reasoning = language + state manipulation + re-encoding |
| **2. Related** | CoT vs tools vs MoA vs CoEvoT vs AGT — position the gap |
| **3. Method** | Mutable state loop: `S_t → Δ → apply → re-encode → S_{t+1}` |
| **4. Exp 1: World tracking** | GraphModi (S = G, trunk) |
| **5. Exp 2: Simulation** | Branching Worlds (counterfactual, time travel) |
| **6. Exp 3: Planning** | Branch/search over edit space |
| **7. Exp 4: Scaling** | Horizon curve — CoT degrades, mutable state doesn't |
| **8. Analysis** | State trajectories, kill-switch, structure sensitivity |
| **9. Discussion** | S = (G, D, …), non-graph substrates as future work |

**Working titles:** *Executable Representations for Graph Language Model Reasoning* · *Mutable World State for Graph Language Models*

---

## The narrowest defensible claim

> The first system in which a graph language model answers counterfactual, retrospective, and planning questions by emitting schema-checked edits to its own input graph (and later derived views S = (G, D)), re-encoding each resulting world through the graph encoder, and searching over versioned branches — with evidence that this substrate-update loop beats both in-context imagination and textualized-edited-graph prompting on compositional multi-edit horizons, and degrades more slowly than matched-compute NL chain-of-thought as reasoning depth increases.

Everything above that sentence is framing. That sentence is what has to survive review.

---

## Relationship to other extensions

- **[branching-worlds.md](branching-worlds.md)** — The version-tree mechanism: branch, checkout, discard. Handles counterfactuals, time travel, and parallel planning rollouts. Mutable State builds on this; see "How this differs from Branching Worlds" above.
- **v1 (task-specific supplementary state)** — Phase 5. Derived views `D, R, C` synchronized with `G` across edits and branches.
- **GraphModi interim** — Phase 0 / Experiment 1. Proves the write path matters before we claim the write path *is* reasoning.

---

## The bigger picture (why this could matter beyond graphs)

If the scaling result holds, the claim is not "GraphModi improves metro QA." It is:

```text
LLM reasoning ≠ just generating tokens

LLM reasoning = language + state manipulation + re-encoding
```

Graph is the first substrate because GraphModi already has the edit loop, the evaluator, and the synthetic benchmark. But the thesis is substrate-agnostic: any structured representation that can be schema-checked, mutated, and re-encoded into the model's forward pass is a candidate workspace.

Scheduling tables, constraint networks, proof states, spatial maps — the mechanism is the same. GraphModi is the existence proof. Mutable state is the generalization.
