# CLEGR Paper: Exact Implementation Details

> Source: `clegr.pdf` — *A Graph Talks, But Who's Listening? Rethinking Evaluations for Graph-Language Models* (arXiv:2508.20583v1).
> Focus: TEA-GLM, GraphToken, and Soft-Prompt baseline. Majority of fine-grained details from Appendix A; also §§2, 3.1, 5.1, D, E.

---

## 0. Shared GLM Formalism (applies to TEA-GLM & GraphToken)

From §2.1, a GLM is a triple:

\[
\text{GLM} = (M_l,\ M_g,\ M_P)
\]

| Component | Role |
|-----------|------|
| \(M_l\) | Frozen LLM |
| \(M_g\) | Graph encoder (GNN) |
| \(M_P\) | **Linear projector** aligning graph ↔ text spaces |

Graph: \(G = (V, E, F^V, F^E)\) (text-attributed).

**Node-level QA** (Eq. 1):

\[
\hat{y} = M_l\!\big(M_P(M_g(G, n_i))\ \oplus\ W(f_i)\ \oplus\ W(q)\big)
\]

**Graph-level QA** (Eq. 2; used for CLEGR):

\[
\hat{y} = M_l\!\big(M_P(\mathrm{Pool}(M_g(G)))\ \oplus\ W(f)\ \oplus\ W(q)\big)
\]

- \(W \in \mathbb{R}^{|L|\times d}\): LLM word embedding matrix
- \(\oplus\): token-sequence concatenation
- \(\mathrm{Pool}(\cdot)\): aggregates node embeddings \(M_g(G,0),\ldots,M_g(G,N-1)\) into one graph representation
- \(f\): concatenated textual features of the whole graph

Figure 1 marks: **graph encoder 🔥, projector 🔥, LLM ❄️**.

---

## 1. Soft-Prompt Baseline

### 1.1 Conceptual role (§2.2)

Isolates whether gains need a graph encoder. A GLM is treated as “a soft-prompt where a graph encoder is learnt instead of token embeddings.” Soft-prompt has **no graph structure access** (no \(M_g\)).

### 1.2 Components

\[
(M_l,\ s)
\]

- \(M_l\): same LLM family as the GLMs
- \(s\): trainable soft-prompt

§2.2 writes \(s \in \mathbb{R}^d\) (one vector) and Eq. 3:

\[
\hat{y}_{\text{soft}} = M_l\!\big(s \oplus W(f_i) \oplus W(q)\big)
\]

**Appendix A.6 overrides the count:** soft-prompt uses the **same 10 tokens** as GLM graph tokens (see §4 below). Figure 1 also shows multiple `SOFT` slots. Treat **10 soft tokens** as the implemented baseline.

### 1.3 What is trained

- Only soft-prompt embeddings \(s\) (and not a GNN/projector)
- Same objective and training procedure as GLMs
- Encodes task-relevant info **without** graph structure

### 1.4 LLM backbones used with soft-prompt

- Node classification (§3.1): **Llama3-8B**, **Phi3-3.5B**
- CLEGR (§5.1.1): those plus **Phi4-14B**

Named in Table 1 as `Llama3-8B-SPT`, `Phi3-3.5B-SPT`.

### 1.5 Input still includes text (and for CLEGR, a textualized graph)

Soft-prompt does **not** get GNN tokens, but CLEGR prompts still include the CSV textualization of nodes/edges (Appendix E.1.7)—so structure can enter **as text**, not as GNN embeddings.

---

## 2. TEA-GLM

### 2.1 High-level method (Appendix A.1)

**TEA-GLM** = Token Embedding-Aligned Graph Language Model [27] (Wang et al., “LLMs as zero-shot graph learners…”).

**Two stages:**

1. **GNN pretraining**
   - Enhanced self-supervised learning
   - **Feature-wise contrastive learning**
   - Align GNN **node representations** with **LLM token embeddings** so the GNN can use LLM pretrained knowledge

2. **Linear projector training**
   - Maps GNN reps → a **fixed number of graph token embeddings**
   - Tokens go into a **unified instruction** for graph tasks
   - **LLM is not tuned** (frozen)

§3.1’s one-line summary (“encoding graph structure through textual descriptions”) is coarser; A.1 is the precise pipeline.

### 2.2 Implementation choices in this paper (A.6)

| Detail | Value |
|--------|--------|
| GNN backbone | **GraphSAGE** |
| Pretraining features | **1000 PCA-projected features** extracted from **each LLM** |
| Protocol | “following the protocol described in [27]” |
| Projector | Linear → **10 graph tokens** |
| LLM | Frozen; Llama3-8B / Phi3-3.5B (NC); + Phi4-14B on CLEGR |

**Not expanded in `clegr.pdf`:** contrastive loss form, temperature, PCA whitening details, number of pretrain epochs, negative sampling—those are deferred to [27].

### 2.3 Task modes

- **Node classification:** node-level Eq. 1
- **CLEGR:** graph-level Eq. 2 with **Pooling** over all nodes (§5.1.2)—needed because questions can target any nodes/edges/subgraphs

---

## 3. GraphToken (G-Token)

### 3.1 Method (Appendix A.3)

From Perozzi et al. [24] (“Let your graph do the talking…”).

Pipeline:

1. GNN encodes structure → node/graph representations via neighborhood aggregation
2. **Trained linear projector** → token embeddings
3. Tokens are **prepended** to the prompt of a **frozen** LLM

Two encoder variants:

| Name | Encoder |
|------|---------|
| **G-Token (GSAGE)** | GraphSAGE (mean aggregation; Eq. 6 in A.4.2) |
| **G-Token (GAT)** | GAT (attention; Eqs. 7–8 in A.4.3) |

§3.1: both TEA-GLM and GraphToken use GraphSAGE by default; CLEGR also evaluates GraphToken+GAT (§5.1.1).

### 3.2 GraphSAGE update used (A.4.2)

\[
h_v^{(l)} = \sigma\!\Big(h_v^{(l-1)} W_1^{(l)} + \mathrm{mean}_{u\in N(v)}\big(h_u^{(l-1)}\big)\,W_2^{(l)}\Big)
\]

### 3.3 GAT update used when applicable (A.4.3)

Attention:

\[
\alpha_{vu}^{(l)} = \frac{\exp\!\big(\mathrm{LeakyReLU}(a^\top[W h_v^{(l-1)}\,\|\,W h_u^{(l-1)}])\big)}{\sum_{r\in N(v)}\exp\!\big(\mathrm{LeakyReLU}(a^\top[W h_v^{(l-1)}\,\|\,W h_r^{(l-1)}])\big)}
\]

Update:

\[
h_v^{(l)} = \sigma\!\Big(\sum_{u\in N(v)}\alpha_{vu}^{(l)}\,W h_u^{(l-1)}\Big)
\]

### 3.4 What differs from TEA-GLM (as stated in this paper)

| | TEA-GLM | GraphToken |
|--|---------|------------|
| Pretrain GNN vs LLM tokens | Yes (contrastive + 1000 PCA LLM features) | **Not mentioned** |
| Projector → tokens | Yes | Yes |
| Tokens prepended / inserted | Yes (into unified instruction) | Yes (prepended) |
| LLM | Frozen | Frozen |
| Default backbone here | GraphSAGE | GraphSAGE (+ GAT ablation) |

---

## 4. Shared Training Recipe for All Three (Appendix A.6)

Applies to **TEA-GLM, GraphToken, soft-prompt** (and G-Retriever where listed):

| Hyperparameter | Exact value |
|----------------|-------------|
| Random seeds | **0, 42, 1918, 2004, 2024** (5 runs; mean ± std) |
| # prompt / graph tokens | **10** for GLMs **and** soft-prompt |
| Optimizer | **AdamW** |
| Learning rate | **0.001 constant** (no schedule stated) |
| Epochs | **1** (single epoch) |
| Batch size | **1** (train **and** eval) |
| Tasks covered by this recipe | Node classification **and** CLEGR |

### 4.1 Exact LLM input template (A.6)

```
[<BOS> + 10×<G> + Context + Question + <EOS>]
```

- `<G>` = GNN-projected **graph tokens** (TEA-GLM / GraphToken), or **soft-prompt tokens** (baseline)
- Context / Question come from the Appendix E prompt templates

### 4.2 Decoding (§5.1.2)

- **Greedy decoding** for generation

### 4.3 Hardware (A.9)

- **NVIDIA A100 80GB**

---

## 5. GNN / Projector Hyperparameters (Tables 3 & 4)

Caption for Table 3: “G-Retriever & G-Token (GSAGE)”. Table 4: GAT. A.6 says all GLMs use GraphSAGE except G-Token(GAT), and points to Tables 3 & 4—**same dimensional recipe** is the GLM stack used in practice.

### 5.1 GraphSAGE / G-Token(GSAGE) / shared GLM dims (Table 3)

| Dataset | Backbone | Task | In Dim | Hidden | Out | Proj | Layers | Dropout |
|---------|----------|------|--------|--------|-----|------|--------|---------|
| Cora | GraphSAGE | Node | **500** | 1024 | 1024 | 1024 | **3** | **0.5** |
| CiteSeer | GraphSAGE | Node | **500** | 1024 | 1024 | 1024 | 3 | 0.5 |
| Arxiv | GraphSAGE | Node | **128** | 1024 | 1024 | 1024 | 3 | 0.5 |
| Computers | GraphSAGE | Node | **768** | 1024 | 1024 | 1024 | 3 | 0.5 |
| History | GraphSAGE | Node | **768** | 1024 | 1024 | 1024 | 3 | 0.5 |
| Photo | GraphSAGE | Node | **768** | 1024 | 1024 | 1024 | 3 | 0.5 |
| CLEGR-Facts | GraphSAGE | **Graph** | **768** | 1024 | 1024 | 1024 | 3 | 0.5 |
| CLEGR-Reasoning | GraphSAGE | **Graph** | **768** | 1024 | 1024 | 1024 | 3 | 0.5 |

### 5.2 GAT variant of GraphToken (Table 4)

Same grid, but backbone = **GAT**; In Dim / layers / dims / dropout identical to Table 3.

**Soft-prompt:** no `In Dim` / GNN layers; only the **10×d** soft embeddings + frozen LLM (same train recipe).

---

## 6. Textual GNN Equations (A.4)

Only needed if you rebuild GraphSAGE/GAT exactly as written; soft-prompt does not use them. GCN is given for **pure GNN baselines**, not for TEA-GLM/GraphToken in this paper’s GLM runs.

### 6.1 GCN (A.4.1) — GNN baseline only

\[
h_v^{(l)} = \sigma\!\Big(\sum_{u\in N(v)\cup\{v\}} \frac{1}{\sqrt{\hat{d}_v\hat{d}_u}} h_u^{(l-1)} W^{(l)}\Big)
\]

### 6.2 GraphSAGE (A.4.2) — used by TEA-GLM & G-Token(GSAGE)

See §3.2 above.

### 6.3 GAT (A.4.3) — used by G-Token(GAT)

See §3.3 above.

---

## 7. Evaluation Protocols Tied to These Models

### 7.1 Node classification (A.8)

- GNNs: accuracy on fixed splits ([21])
- GLMs + soft-prompt: **string matching**
  - Dataset-specific candidate class names
  - `match_prediction`: correct if output **begins with** the class name (e.g. `"Asia"` matches `"Asia in the 20th century"`)
  - Unmatched → special `None` class index

### 7.2 CLEGR (Appendix D)

Answer-type-specific scoring:

- **Categorical** (station names, architecture types): exact match after normalization (lowercasing and punctuation removal)
- **Boolean** (yes/no): map `"yes"/"no"/"true"/"false"` to binary; report accuracy, F1, MCC
- **Numeric** (distances, years): `numpy.isclose`; if parsing fails, extract numbers via regex; report accuracy, MAE, RMSE
- **Set-valued** (lists of stations): set-based precision, recall, F1

Primary reported metric: **overall accuracy**.

### 7.3 CLEGR adaptation shared by GLMs (§5.1)

- Graphs included in **textualized** form (G-Retriever-style), for consistent prompts across models
- Full-graph **Pool** into projector (Eq. 2)

---

## 8. Exact Prompt Shells (Appendix E)

What sits after the 10 tokens.

### 8.1 Node classification (pattern)

Instruction-style:

```
<s>[INST] {node textual fields}
Answer the following question: Which ... ?
Please only output the most likely answer from the following ... and nothing else: {class list}.
Answer: [/INST] {Label} [/s]
```

Dataset-specific fields:

- **Arxiv:** Title + Abstract; subcategory list
- **Cora:** Title + Abstract; classes: `theory, reinforcement learning, genetic algorithms, neural networks, probabilistic methods, case based, rule learning`
- **CiteSeer:** Text; classes: `Agents, AI, DB, IR, ML, HCI`
- **Computers:** Context; computer product subcategory list
- **Photo:** Context; photography subcategory list
- **History:** Context; history subcategory list

### 8.2 CLEGR Facts & Reasoning (E.1.7)

```
<s>[INST]
--- Nodes ---
{CSV of all nodes}
--- Edges ---
{CSV of all edges}
Above is the representation of a synthetic subway network. All
stations and lines are completely fictional. Keep in mind that the
subway network is not real. All information necessary to answer the
question is present in the above representation. The question is:
{Question}

{Answer Format Suffix}
[/INST] {Label} [/s]
```

**Node CSV header:**

```
"id", "name", "disabled_access", "has_rail", "architecture", "cleanliness", "music", "size"
```

**Edge CSV header:**

```
"source_id", "target_id", "line_color", "line_stroke", "has_aircon", "built"
```

**Answer format suffixes:**

| Type | Suffix |
|------|--------|
| String | `Answer directly:` |
| Bool | `Answer with 'True' or 'False':\n\nAnswer:` |
| List | `Output a comma-separated list:` |
| Count | `Answer with a number:\n\nAnswer:` |
| Cycle | `Answer with 'True' if it is in a cycle, otherwise 'False':\n\nAnswer:` |

Soft-prompt vs GLM difference on CLEGR: **same text prompt**; GLM also injects **10 GNN tokens** as `<G>`, soft-prompt injects **10 learned soft tokens**.

---

## 9. Linear Probing of Graph Tokens (related; TEA-GLM & G-Token GSAGE)

§3.3 / A.5 — not a baseline method, but specifies how they treat projector outputs:

1. Freeze trained GLM
2. Run \(M_g\) → \(M_P\) → graph tokens
3. Train linear classifier on `flatten(M_P(M_g(G)))` (Eq. 4 / 9):

\[
\hat{y} = \arg\max_i \big(W \cdot \mathrm{flatten}(M_P(M_g(G))) + b\big)_i
\]

where \(W \in \mathbb{R}^{c\times d'}\), \(b \in \mathbb{R}^{c}\), \(c\) = # classes, \(d'\) = flattened embedding dim.

4. Done for TEA-GLM and G-Token(GSAGE) on Cora/CiteSeer

Shows projector outputs are linearly separable for structure-heavy sets.

---

## 10. Pure GNN Baseline Hyperparameters (Table 2; for completeness)

Used for GAT / GCN / GraphSAGE **unimodal** baselines (not the GLM stack). From [21].

| Model | Dataset | ResNet | Norm | Dropout | #Layers | Hidden Dim | LR | Epochs |
|-------|---------|--------|------|---------|---------|------------|-----|--------|
| GCN | Cora | False | None | 0.7 | 3 | 512 | 0.001 | 500 |
| GCN | CiteSeer | False | None | 0.5 | 2 | 512 | 0.001 | 500 |
| GCN | Computer | False | LN | 0.5 | 3 | 512 | 0.001 | 1000 |
| GCN | Photo | True | LN | 0.5 | 6 | 256 | 0.001 | 1000 |
| GCN | History | True | LN | 0.5 | 6 | 256 | 0.001 | 1000 |
| GCN | Arxiv | True | BN | 0.5 | 5 | 512 | 0.0005 | 2000 |
| GAT | Cora | True | None | 0.2 | 3 | 512 | 0.001 | 500 |
| GAT | CiteSeer | True | None | 0.5 | 3 | 256 | 0.001 | 500 |
| GAT | Computer | False | LN | 0.5 | 2 | 64 | 0.001 | 1000 |
| GAT | Photo | True | LN | 0.5 | 3 | 64 | 0.001 | 1000 |
| GAT | History | True | LN | 0.5 | 3 | 64 | 0.001 | 1000 |
| GAT | Arxiv | True | BN | 0.5 | 5 | 256 | 0.0005 | 2000 |
| GraphSAGE | Cora | False | None | 0.7 | 3 | 256 | 0.001 | 500 |
| GraphSAGE | CiteSeer | False | None | 0.2 | 3 | 512 | 0.001 | 500 |
| GraphSAGE | Computer | False | LN | 0.3 | 4 | 64 | 0.001 | 1000 |
| GraphSAGE | Photo | True | LN | 0.2 | 6 | 64 | 0.001 | 1000 |
| GraphSAGE | History | True | LN | 0.2 | 6 | 64 | 0.001 | 1000 |
| GraphSAGE | Arxiv | True | BN | 0.5 | 4 | 256 | 0.0005 | 2000 |

---

## 11. What `clegr.pdf` Does **Not** Specify

| Gap | Status in paper |
|-----|-----------------|
| TEA-GLM contrastive loss, temperature, pretrain schedule | Only “feature-wise contrastive” + “follow [27]” + 1000 PCA features |
| Soft-prompt init (random / embedding copy) | Not stated |
| Soft-prompt dim \(d\) | Implied = LLM hidden size |
| Whether projector is one linear layer or multi-layer MLP | Called **“linear projector”**; Out Dim = Proj Dim = 1024 |
| How 10 tokens are produced from one pooled vector (tile / reshape / multi-head linear) | Not stated—only “fixed number of graph token embeddings” / “10 graph tokens” |
| Loss for generative training | Not named (standard next-token CE is implied by instruction format with `{Label}`) |
| Weight decay / AdamW β / grad clip | Not stated |
| Soft-prompt Eq. 3 vs “10 tokens” | Notation inconsistency; **A.6 is authoritative** |

---

## 12. Compact Comparison (Implementation View)

| | Soft-prompt | TEA-GLM | GraphToken |
|--|-------------|---------|------------|
| Structure path | None (text only) | GraphSAGE (+ pretrain align) | GraphSAGE or GAT |
| Token source | 10 learnable embeddings | 10 tokens from \(M_P(M_g(\cdot))\) | Same |
| LLM | Frozen | Frozen | Frozen |
| Extra stage | — | Contrastive pretrain + 1000 PCA LLM feats | — |
| Train | 1 epoch, AdamW 1e-3, bs=1 | Same | Same |
| Seeds | 0, 42, 1918, 2004, 2024 | Same | Same |
| CLEGR | Soft tokens + CSV text | Pool→project→tokens + CSV text | Same |

---

## References (as cited in the paper)

- **[24]** Bryan Perozzi et al. *Let your graph do the talking: Encoding structured data for LLMs*, 2024. https://arxiv.org/abs/2402.05862 (GraphToken)
- **[27]** Duo Wang, Yuan Zuo, Fengzhi Li, and Junjie Wu. *LLMs as zero-shot graph learners: Alignment of GNN representations with LLM token embeddings*, 2024. https://arxiv.org/abs/2408.14512 (TEA-GLM)
- **[16]** Soft prompting citation in paper (Lester et al. / soft-prompt literature as used in §2.2)
- **[9]** G-Retriever (He et al.) — textualization pipeline reused for CLEGR prompts
- **[6]** GraphSAGE (Hamilton et al.)
- **[26]** GAT (Veličković et al.)
