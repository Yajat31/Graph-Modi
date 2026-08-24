# Branching Worlds: The Write Path as a Simulator

*(extension proposal — successor slot after v2, given its own name to avoid collision)*

## The one-sentence idea

GraphModi built a write path so the graph can track reality; this extension asks what happens when the model uses that same write path to **imagine** — answering "what if?" questions by actually editing a *branch* of the graph, re-encoding it, and reasoning over the simulated world instead of a remembered or imagined one.

---

## Where this comes from

GraphModi's whole argument is:

```text
telling the model about a change (text)   →  fails
actually applying the change and re-encoding  →  works
```

Now notice that a counterfactual question is *structurally identical* to a revision — except the change isn't real:

```text
Revision:        "Station B is closed."          → edit G, keep it
Counterfactual:  "What if station B closed?"     → edit a COPY of G, discard it
```

If our interim result holds — that GLMs can't simulate a graph change in-context and need the change applied to the tensors the GNN encodes — then the exact same failure should appear for hypotheticals, and the exact same fix should work. Nothing new architecturally. Same model, same edit grammar, same loop. The only new thing is that some edits are applied to branches instead of to the trunk.

That is the leap: **the edit loop is not just a state-synchronization mechanism, it is a world simulator the model can query.**

---

## The mechanism: a version tree of worlds

Treat the graph the way git treats a repository:

```text
G0 ──edit──> G1 ──edit──> G2        ← trunk: reality, as in GraphModi
                    │
                    ├──what-if──> G2' (B closed)      ← branch, discarded after answering
                    │
                    └──what-if──> G2'' (new line A-E) ← another branch
```

- **Trunk commits** = real revisions, exactly what GraphModi already does.
- **Branches** = speculative edits. The model emits the same `SET / ADD / DEL` operations, but tagged as hypothetical. We apply them to a copy, re-encode the copy, answer the question grounded in the branch, and throw the branch away. Reality is never contaminated.
- **Checkouts** = time travel. "Before the closure, how long was the commute?" is answered by re-encoding `G1`, not by hoping the model remembers what the world used to look like.

Three question families fall out of one mechanism:

### 1. Counterfactual QA

> "If the B–C line were closed, could I still get from A to E?"

Model emits `DEL edge(B,C)` as a branch edit → branch re-encoded → QA prompt is just `(G', q)` — the hypothetical utterance is *not* passed to QA, mirroring the interim design. If QA succeeds, the hypothetical world is genuinely in the graph tokens.

### 2. Retrospective QA (time travel)

> "How many stops was the trip before yesterday's closure?"

Checkout `G1`, re-encode, answer. The version tree replaces memory. No model in our related-work set can do this: they either see only the current graph or only the transcript.

### 3. Planning as search over edit space

> "We must close one station for maintenance. Which closure disrupts the fewest commutes?"

Now edits are **actions** and branches are **rollouts**:

```text
           G2
   ┌───────┼───────┐
   ↓       ↓       ↓
 close A  close B  close D
   ↓       ↓       ↓
re-encode re-encode re-encode
   ↓       ↓       ↓
 evaluate  evaluate evaluate  →  compare  →  answer
```

The GLM proposes candidate edits, each spawns a branch, each branch is scored by asking the GLM the evaluation question on it, and the answers are compared. This is model-based planning (in the RAP / tree-search-over-world-states sense) where the world state is, for the first time, *the re-encoded input graph of a GLM* rather than text or a game engine.

---

## Grounding: what exists, what doesn't (checked against 2024–2026 literature)

We ran a literature sweep on each pillar. Honest summary:

**Counterfactual QA over graphs — adjacent, gap is real.**
- *CFKGR / COULDD* (EACL 2024) does counterfactual reasoning over KGs with hypothetical edits — but symbolically/textually, no graph encoder in the loop.
- The knowledge-editing line (*MQuAKE*, *GMeLLo*) established that explicitly applying an edit beats asking the model to imagine it — which **supports our premise** but in parametric/text settings, not GLM soft-prompt settings.
- Nobody applies a speculative edit to a branch of a GLM's input graph and re-encodes it through the GNN. That mechanism is ours.

**Branching / versioned world state — the concept is saturated, the substrate is not.**
- Git-like context and world-state management for LLM agents already exists (*Git-Context-Controller* and several 2025–2026 event-graph systems). Fork-based counterfactual querying is done symbolically.
- No prior work versions the *input graph of a GNN-fused GLM* and re-encodes alternative versions through the encoder. The novelty claim must be stated at the encoder level, not the "git for worlds" level.
- Validity support is strong: RAP-style explicit rollouts beat in-context imagination; GPT-4-as-world-simulator studies document exactly the state-tracking failures we exploit.

**The mandatory baselines (these decide whether the paper stands):**
1. **Textualized-edited-graph baseline.** Apply the hypothetical edit, but serialize the edited graph as text instead of re-encoding. If this matches branch-re-encoding, the GNN path isn't earning its keep. The premise is known to hold for multi-hop questions on larger graphs and *not* necessarily for single edits on tiny in-context graphs — so the dataset must include compositional hypotheticals ("what if B closed *and* a new A–E line opened?") and graphs big enough that serialization degrades.
2. **Symbolic replay baseline.** A non-neural system that applies the edit and runs the solver will get 100% on solver-computable questions. So the evaluation must include questions that need the *language* side (attribute-laden, fuzzy, or preference questions over the branch) where a pure solver has no answer — otherwise a reviewer correctly asks why a GLM is involved at all.

**Verdict across all three sweeps: adjacent with a gap** — nothing here is already done, but each pillar has close neighbors that dictate exactly how narrowly we must aim.

---

## Why this is the right extension for *this* project

1. **It reuses the proven loop.** Emit edit → validate → apply → re-encode → answer. We change *where* the edit lands (branch vs trunk), nothing else. No new architecture, no fine-tuning required to start.
2. **The interim result is the motivation.** If frozen-graph + textual revision fails while oracle-edit + re-encode succeeds (our existing benchmark), the counterfactual version of the same comparison is the obvious and unclaimed next experiment.
3. **The dataset generator extends naturally.** The same synthetic metro generator produces counterfactual items: sample a hypothetical edit, verbalize it as a "what if" question, compute gold on the branched graph with the solver, and apply the same filters (discard if answerable without the graph, or if the un-branched graph already gives the same answer).
4. **The evaluation stays exact-match against a solver.** Same rigor, no human annotation.
5. **It upgrades the story.** GraphModi: "GLMs need a write path to stay grounded." Branching Worlds: "once a GLM has a write path, it has a simulator — and simulation, memory, and planning are all the same loop pointed at different versions of the graph."

---

## Experiment ladder (no code yet — this is the plan)

**Rung 1 — Counterfactual grounding (the core claim).**
Conditions on identical items: (a) frozen graph + hypothetical stated in text, (b) frozen graph + textualized edited graph, (c) self-emitted branch edit + re-encode, (d) oracle branch edit + re-encode. Metrics: answer exact match, edit correctness. Prediction from interim: (a) fails, (d) succeeds; the paper lives in the (b) vs (c) gap, and (c)→(d) isolates whether writing the edit is the bottleneck.

**Rung 2 — Time travel.**
Multi-turn sessions with real edits, then retrospective questions. Conditions: transcript-only (GraphRAG-style memory), current-graph-only, version-checkout + re-encode. This tests whether the version tree beats both memory-in-text and memory-in-weights.

**Rung 3 — Compositional hypotheticals.**
Two-to-three stacked hypothetical edits per question. This is where the textualized baseline is expected to break down (multi-hop over an edited serialized graph) and where the mechanism should shine.

**Rung 4 (stretch) — Planning over edit space.**
Small action sets (choose 1 of k closures), branch-per-candidate, GLM-evaluated rollouts. Compare against asking the model to plan in-context in one shot. Only attempt after rungs 1–3 hold.

---

## A flagged frontier (explicitly not the core): the graph as working memory

There is a seductive further step: let the model write *annotations* into a scratch branch as it reasons — mark nodes visited, eliminated, candidate — so chain-of-thought becomes a trajectory in graph-edit space that the GNN re-encodes each step.

We ground-checked this too, and the honest finding is that it is **under direct empirical threat**: 2026 work (*CoEvoT* — latent graph-token rewriting per thought; *Agentic Graph Token reasoning* — per-step re-encoding, read-only; *Map-of-Actions* — schema-verified graph-edit reasoning, text-side) has surrounded the idea, and a systematic evaluation (arXiv:2605.03514) found graph-token LLMs nearly **structure-blind** — TEA-GLM's accuracy moved 0.02 points under adversarial edge rewiring. If the model can barely see adversarial topology changes, it may not be able to read back its own annotations through the graph-token channel.

So we keep it as a frontier section with a kill-switch experiment: write gold annotations into the graph (oracle), re-encode, and test whether QA improves *at all*, with a shuffled-annotation control to verify the content (not just the extra compute) matters. If the oracle probe fails, the frontier dies cheaply and the core is untouched. Notably, rung 1 already produces the evidence this needs: if branch re-encoding works for `DEL edge` hypotheticals, structure-blindness is not absolute in our regime.

---

## The narrowest defensible claim (for the eventual paper)

> The first system in which a graph language model answers counterfactual, retrospective, and planning questions by maintaining a versioned tree of its own input graph — applying self-emitted, schema-checked edits to speculative branches and re-encoding each branch through the graph encoder — with evidence that this simulation-by-editing beats both in-context imagination and textualized-edited-graph prompting on compositional hypotheticals.

Everything above that sentence is framing; that sentence is what has to survive review.

---

## Relationship to the other extensions

- **v1 (task-specific supplementary state)** gives the GLM *derived views* of one world. Branching Worlds gives it *many worlds*. They compose: a branch can carry its own recomputed supplementary state (distances on the hypothetical graph), which is exactly the delivery-network scenario from v1 turned counterfactual ("if road A–B closed, which warehouse *would* serve X?").
- **v2** (in progress) — to be reconciled once drafted; the version-tree mechanism is agnostic to what lives at each node of the tree, so it should sit cleanly alongside.
