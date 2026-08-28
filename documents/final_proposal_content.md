# Final Proposal — Content Organization

Working document. Unconstrained length. Everything here is a *proposal*: what we will build, generate, and measure. No results, no run logs, no numbers from any study.

The final PDF is 2 pages excluding references, so this doc is the superset from which that is cut. Section numbering here does not have to match the paper.

---

## 1. Problem

A Graph Language Model answers a question from an input graph `G`. The standard setup is a single pair `(G, q)`. GLMs exist precisely so that answers are *grounded in `G`*.

Conversation breaks that. A user corrects a fact in language. The language model sees the correction; the graph encoder still reads the original `G`. Later answers are then grounded in a stale world.

Existing work treats `G` as fixed:

- Plenz and Frank (2024) introduced GLMs so structure and text are encoded together.
- He et al. (2024) describe a conversational interface ("chat with your graph"), but GraphQA is still single-shot. Their ablations show both the GNN path and the textualized graph matter — so if conversation revises a fact and `G` does not, retrieval from the old graph grounds the model in the wrong world.
- TEA-GLM (Wang et al., 2024), GraphToken (Perozzi et al., 2024) and related models treat `G` as a fixed dataset.
- CLEGR (Petkar et al., 2026) shows many GLM benchmarks can be solved from one modality alone.
- Agentic Graph Token reasoning (Peng and Yang, 2026) lets a model re-choose a *view* of a still-fixed graph.

None of these test a user correction that makes `G` stale.

Conversational KG and GraphRAG systems keep state in an external store, or retrieve serialized triples for an LLM. That is the right design if `G` is a database. It is not the GLM setting: graph tokens remain part of the forward pass and are not overwritten by extra retrieved text. Auditors such as SKG-Eval (Shil and Samui, 2026) score dialogue against a parallel KG; they do not update the GLM's input graph. Tool-using LLMs call graph algorithms; G-Retriever's retrieval is pipeline code. None of these expose a mutation of the tensors the GNN encodes.

The claim is simple. GLMs exist so answers are grounded in `G`. Chat includes corrections. There is currently no write path that refreshes the graph the encoder reads. Two turns of inconsistent input are enough. We are not adding a memory model or native tool calling, and we are not proposing a new architecture.

---

## 2. The write path (foundation)

A GLM forward pass cannot rewrite `G` inside the GNN. We put an outer loop around an existing model (for example G-Retriever, GraphToken, or TEA-GLM).

On each user turn `u_t`:

1. Prompt the GLM with the current graph and `u_t` to emit an edit `Δ`.
2. Parse `Δ`. If it is illegal, treat it as a no-op.
3. Otherwise apply `Δ` to `G` with a small schema-checked function.
4. Re-encode `G` and answer the next question `q_t`.

```text
u_t -> emit Δ -> parse -> schema-check -> apply -> G_t -> re-encode -> answer q_t
```

The model has no tools. It already emits text. We restrict that text to four operations:

| Operation | Meaning |
|-----------|---------|
| `SET` | set a node attribute |
| `ADD` | add an edge |
| `DEL` | delete an edge |
| `NOOP` | no change |

Two calls, same weights: emit `Δ`, then answer on the updated `G`. The QA prompt is only `(G, q_t)` — **the revision utterance is not passed to QA**. If QA succeeds, the fact is in the graph.

We start with prompting and parsing. Constrained decoding or light LoRA on the same LLM head is a fallback only if writing the edit turns out to be the bottleneck. No second model.

If we skip re-encoding, this is just GraphRAG. It is a GLM method only if the apply step always precedes the next GNN forward pass.

---

## 3. Why multi-turn escalates the problem

Edits are cumulative. Turn three applies to `G_2`, never to the original graph. That is what makes the setting realistic, and it is also what introduces a failure the write path alone does not address.

If every revision can be applied on top of the current graph, the loop is enough. But real revision dialogue routinely produces a later utterance that *cannot* sit on top of an earlier one:

```text
Turn 2:  "Close station B for maintenance."
Turn 5:  "The B–C line is now running through-service again."
```

Turn 5 is not a no-op and not a clean reversal. Its implied state is incompatible with the closure still sitting in the log. A write path with no conflict notion has exactly three options, and it takes the first silently:

```text
last-write-wins     -> reopen B, discard the closure, never mention it
refuse the new edit -> keep B closed, never mention the conflict
ask                 -> "Turn 2 closed B. Keep the closure, or take through-service?"
```

So the proposal is not just "give the GLM a write path." It is: **give the GLM a write path that knows when a new revision contradicts an old one, can say which old one, and asks instead of guessing.**

---

## 4. Defining contradiction

"Contradiction" is overloaded across three literatures. We adopt the notion from belief revision and ontology debugging, where a set of sentences is *inconsistent* when it has no model, and we ground it on the graph.

Let each revision `r` denote a set of constraints `I(r)` over attributed graphs — the implied state of the edit, not the wording of the utterance. Let `Inv` be the stated invariants: the schema plus any task invariants we choose to enforce (connectivity, line topology, attribute domains, mutual exclusions). Let `R_active` be the revisions whose constraints have not been superseded.

```text
T = ( ⋃_{r ∈ R_active} I(r) )  ∪  Inv

Conflict(T)     ⇔  T has no model — no attributed graph satisfies it
Culprit set     ⇔  a minimal unsatisfiable subset of T
Contradiction   ⇔  Conflict(T) and the user has not retired a member of the culprit set
Revision        ⇔  Conflict(T ∪ I(r_new)) and the user intends r_new to supersede
                   some r_j in the culprit set
```

So, stated plainly:

> Two revisions **contradict** iff there is no single attributed graph satisfying both implied states together with the invariants.

This is the DECODE task (Nie et al., 2021) with the informal "last utterance versus history" judgment replaced by a decision procedure, and it is Xu et al.'s (2024) *inter-context conflict* except that the two contexts are committed edits rather than retrieved passages. Attribution corresponds to DECODE's evidence indices, computed as a minimal unsatisfiable subset in the tradition of ontology debugging (Shchekotykhin et al., 2012).

### 4.1 What must not be conflated

| Term | Tradition | What it is | What it is not |
|------|-----------|------------|----------------|
| Inconsistency | Belief revision, DL/OWL debugging | The theory has no model | Not a speech act; carries no notion of intent |
| Dialogue contradiction | DECODE, VISTA, SKG-Eval | Last utterance conflicts with still-active prior information, labelled by humans or an NLI/KG engine | Not a formal satisfiability check |
| Knowledge conflict | Xu et al. (2024) | Two *sources* disagree (context vs. parameters, or two documents) | Not one agent's own accumulated commitments; no edit log |
| Revision / supersession | Belief revision; temporal KG stores | The later statement is meant to *replace* the earlier one | Not a contradiction to be flagged |

The last row is the crux. Unsatisfiability is the *conflict*. Whether it is a contradiction to surface or a revision to absorb depends on **which constraints are still active** — and that is not derivable from `G`, from the edit log, or from recency. It is exactly what we propose to ask.

---

## 5. Taxonomy of conflicts in this setting

Three cases, ordered by what is needed to decide them.

| Case | Example | Decided by |
|------|---------|------------|
| **Invariant violation** | The new edit cannot apply at all: unknown node, duplicate edge, dangling endpoint, self-loop | Schema check inside the apply step |
| **Silent incompatibility** | Each edit applies alone, but together they violate a stated task invariant (the ring stays connected; a line remains a cycle; two attributes are mutually exclusive) | A solver over the edit log, given `Inv` |
| **Intent conflict** | Both apply, the graph stays valid, but the two revisions cannot both be what the user meant | Not derivable — needs a keep-decision from the user |

Cases 1 and 2 are the **same formal object**: unsatisfiability relative to `Inv`. They differ only in whether the invariant already lives in the schema checker or is stated as part of the task. That is an implementation boundary, not a conceptual one.

Case 3 is, under our definition, **not a contradiction** unless an invariant makes the two implied states jointly unsatisfiable. "Close B" and "through-service on B–C" contradict only if `Inv` says a closed station cannot carry through-service. Absent such an invariant, this is an *unresolved conflict*: a pragmatic clash that stays open until the keep-decision settles it. We will keep this distinction explicit rather than calling everything a contradiction.

---

## 6. Proposed mechanism

Four stages inserted around the existing write path.

```text
incoming revision u_k
        |
        v
   parse typed edit
        |
        v
   conflict gate ---consistent---> apply, re-encode, answer   (Section 2 loop)
        |
     conflict
        |
        v
   attribute: replay the edit log, isolate the culprit set
        |
        v
   ask one discriminating question naming the earlier edit
        |
        v
   repair: retire the loser, replay the surviving suffix, re-encode
```

**Detect.** A conflict gate runs before commit. For invariant violations it is the existing schema check. For silent incompatibilities we apply tentatively and test `Inv` on the result together with the active constraints.

**Attribute.** Given a conflict at turn `k`, replay `G_0 … G_{k-1}` from the initial graph and the stored edits, and isolate the minimal earlier edit set whose presence makes `u_k` unsatisfiable — the latest edit whose retirement restores satisfiability, or a minimal hitting set when several are implicated. This is a decision procedure over a typed log, not a learned component, and we will say so rather than claim it as a result.

**Ask.** One discriminating question, binary in the common case, that *names the earlier edit*:

> Turn 2 closed station B. Turn 5 puts through-service on B–C. Which should stand?

**Repair.** The answer names a survivor. Retire the loser, replay every later edit that still applies, drop those that no longer do, re-encode, continue. The graph the user actually wanted is the target.

---

## 7. Why asking is the contribution: recency is wrong

Every conflict-resolution strategy we are aware of resolves by recency — last-write-wins, supersession, temporal invalidation — and instruction-tuned models are already trained toward preferring the latest instruction, which IHEval (2025) documents directly. ConInstruct (2026) further reports that models largely *detect* conflicting instructions yet rarely disclose them or ask.

So detection alone is not the open problem, and neither is attribution over a typed log. Asking the user has value on exactly one class of items: those where **recency is wrong** — where the user wants the *earlier* edit preserved and the later utterance rejected or amended, because it was a slip, a stale fact, or a local change that should not overwrite the earlier commitment.

No benchmark we are aware of constructs those items, because they all assume the newest statement is the truth. That is the hole this proposal aims at, and it is why the keep-decision is the one label a solver cannot supply.

---

## 8. Dataset

We will not scrape a dialogue corpus. Items are generated from a synthetic attributed graph in the style of CLEGR: a fictional metro with invented station names, lines, and attributes, so parametric memory cannot skip `G`.

### 8.1 Base generator (from the interim plan)

- Sample a small graph `G_0`, roughly 15–30 nodes, with attributes such as `open` or `closed`.
- Require at least two paths between the query pair, so closing a node changes the answer.
- Fix a query executable on the graph — for example, the number of stations on the shortest path using only open stations.
- Sample a short sequence of `SET`, `ADD`, `DEL` operations that change that answer, and verbalize each as a short user utterance.
- Compute gold answers with a solver on the graph, never by hand.

Discard an item if:

- the question can be answered without the graph;
- the utterance plus the question recover the answer with no graph;
- the original `G_0` already yields the new answer.

A few hundred sessions of two to five turns is enough for the comparisons.

### 8.2 Contradiction sessions (new)

To produce genuine conflicts, the generator must stop drawing every edit from the current graph. For a planted conflict at turn `k` against turn `j < k`, sample the later edit from an **earlier ancestor state** `G_{j-1}`, so it is coherent on its own but incompatible with what the log has since committed. Where the conflict is a silent incompatibility, state the invariant explicitly so the solver has something to check.

Each contradiction item carries:

| Label | Purpose |
|-------|---------|
| `contradicts_turn_j` | Which earlier turn the new revision conflicts with |
| `culprit_edits` | Gold minimal culprit set, for attribution scoring |
| `keep_decision` ∈ {earlier, later, amend} | Gold resolution. **The one label not computable from `G` or the log.** |
| `intended_graph` | What the graph should be once the decision is applied |
| `invariant` | The stated invariant, for silent incompatibilities |

Crucially, gold answers stay solver-computable *after* the keep-decision is applied: replay the repaired log and run the solver on the intended graph. Only the keep-decision itself is planted rather than derived. The **recency-wrong subset** is the set of items where `keep_decision ≠ later`; if that subset cannot be constructed, the rest of the benchmark is not worth generating.

Simulated user answers come from `keep_decision`, so no human is in the loop at evaluation time.

---

## 9. Evaluation

The point of the dataset is to ask whether the GLM has to update `G`, or whether text is already enough — and then whether it has to *ask*, or whether recency is already enough. Same sessions, same decoder, across conditions.

### 9.1 Conditions for the write path

| Condition | What it tests |
|-----------|---------------|
| Frozen graph | Should keep the old answer after a revision. If it does not, the item is too easy or the model is not using `G`. |
| Oracle edit + re-encode, no revision text | Whether the GLM can read an updated graph at all. |
| Revision prompted in text, `G_0` frozen | If this already matches the oracle, prompting suffices and rewriting `G` is unnecessary. |
| GraphRAG-style transcript, no re-encode | Keeps history, never re-encodes. |
| Self-edit loop | The method: model writes `Δ`, we apply, we ask `q_t` on the new graph, without passing `u_t` into QA. |

If oracle updates restore accuracy and the self-edit loop does not, the bottleneck is writing `Δ`, not reading `G`.

### 9.2 Baselines for contradiction handling

These decide whether the contradiction half stands, and we will report them even where they are expected to win:

1. **Symbolic solver over the edit log.** Replay, check invariants, enumerate minimal culprit sets. Expected to be near-exact on invariant violations and silent incompatibilities. We include it to make clear we are *not* claiming detection or attribution as results.
2. **Transcript-only LLM.** Frozen graph, full revision history in context, asked to flag and localize the conflict. Also a variant with the serialized edited graph.
3. **Last-write-wins.** Always commit the newest edit, never attribute, never ask. This is the product default and it fails by construction on the recency-wrong subset.
4. **Oracle attribution.** Hand the system the true culprit set for free. If end-task accuracy does not move, attribution is not the bottleneck and should not be in the claim.

### 9.3 Metrics

| Metric | Measures |
|--------|----------|
| Answer exact match | Against the solver, as in the base benchmark |
| Edit correctness | Whether the parsed `Δ` matches the gold edit |
| Detection precision / recall | Conflict gate at the turn where the conflict is planted |
| Culprit localization | Exact and set-level match against `culprit_edits` |
| Questions asked | Whether the system asked, and how many times |
| Post-repair state exactness | Repaired graph against `intended_graph` |
| Recency-wrong accuracy | Post-repair exactness restricted to `keep_decision ≠ later` |

The last row is the number the proposal lives or dies on, because last-write-wins scores zero on it by construction.

---

## 10. Related work and positioning

**GLMs treat the graph as fixed.** Plenz and Frank (2024); He et al. (2024); Wang et al. (2024); Perozzi et al. (2024); Petkar et al. (2026); Peng and Yang (2026). None expose a write path to the tensors the GNN encodes.

**Dialogue contradiction is text-side.** DECODE (Nie et al., 2021) defines the detection task with evidence indices, over utterances rather than state mutations. VISTA (2026) verifies atomic claims against dialogue history. SKG-Eval (Shil and Samui, 2026) maintains an incremental semantic KG and separates intentional update from accidental inconsistency — but as an *evaluator*, with no encoder in the loop and no repair.

**Conflict taxonomies concern sources, not commitments.** Xu et al. (2024) organize context-memory, inter-context, and intra-memory conflict. ConInstruct (2026) shows detection is largely achievable while disclosure is not. IHEval (2025) shows recency-wins is already learned.

**Diagnosis and interactive repair are old and strong.** Ontology debugging computes minimal unsatisfiable subsets and asks a user oracle discriminating questions (Shchekotykhin et al., 2012). Failure attribution over agent traces is a named task (Who and When, 2025). We therefore position detection and attribution as *machinery*, and the recency-wrong resolution target as the contribution.

Positioning sentence for the paper: the novelty is not the detect-attribute-ask-repair pipeline, which exists on other substrates; it is doing it over the input graph a GLM actually encodes, and evaluating it on items where the correct resolution is not the most recent one.

---

## 11. Risks

- **Structure-blindness.** Reported evaluations suggest graph-token LLMs can be weakly sensitive to structural perturbation. If a contradiction cannot be perceived through the graph-token channel, detection will fall entirely to the symbolic gate and the GLM's role shrinks to phrasing the question. We will probe this early with an oracle-injected violation before building the full loop.
- **A solver plus recency may close the gap.** If a transcript-reading LLM prompted with "the user may not mean the latest revision" matches the full mechanism on the recency-wrong subset, the attribution and clarification apparatus is unjustified.
- **Intent conflicts may resist gold labelling.** If they cannot be planted without human annotation, the remaining items are solver-decidable and the contribution narrows to the benchmark itself.

---

## 12. Cut list for the 2-page version

Keep, in priority order: the staleness argument, the write path with the four operations and the withheld utterance, the formal definition block, the conflict gate through repair, the dataset labels including `keep_decision`, the conditions, and the recency-wrong metric.

Cut first if over length: the figure, then the taxonomy prose in Section 5 (compress to one sentence naming the three cases), then the detail in Section 10 (fold into citations), then Section 11 down to two sentences. Never cut the definition or the `keep_decision` label.
