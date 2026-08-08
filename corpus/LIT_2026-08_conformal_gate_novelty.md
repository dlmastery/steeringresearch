# LIT 2026-08 — The conformal gate: novelty check, and an audit of the §18.2 "scooped" citations

**Scanned:** 2026-08-08. **Window:** Apr–Aug 2026 for the novelty sweep (arXiv `2604`–`2608`);
the citation audit covers whatever dates the eight audited IDs carry (Sep 2025 – Jun 2026).

**Verification status:** every arXiv ID below was **WebFetch-verified** — real title + real
author list + submission date pulled from the abstract page. None cited from memory.
`2603.14623` was additionally read in **full text** (HTML v1). Papers characterised from a
search snippet only are marked `[UNVERIFIED]` inline.

**Why this scan exists.** `CLAUDE.md` §18.2 asserts that two of the program's three planned
contributions were *"scooped while the program sat idle"*, citing eight arXiv IDs. Those IDs
had **never been fetched**. Under R17.4 (never cite from memory; every ID WebFetch-verified)
and §18.8 (inherited numbers are `[NEEDS VERIFICATION]` until reproduced), an unverified
citation block that retired two contributions is exactly the artifact this program exists to
distrust. This file checks it.

---

## 0. The one-sentence synthesis

> **The scoop was overstated by a factor of three, one citation is a chest-X-ray paper, one
> is an adversary misfiled as a scoop — and the surviving contribution is not "a better gate"
> but the observation that gate feasibility is a *local* ROC property while the program
> reports *global* AUC everywhere.**

---

## 1. Verification table — the eight IDs cited in §18.2

| ID | filed under | real? | actual title | supports §18.2? |
|---|---|---|---|---|
| `2602.02712` | displacement budget | REAL | Towards Understanding Steering Strength | **SUPPORTS** (strong theory neighbour) |
| `2606.06735` | displacement budget | REAL | A Geometric Account of Activation Steering through Angle-Norm Decomposition | **SUPPORTS** — the one genuine scoop in Group A |
| `2604.09839` | displacement budget | REAL | Steered LLM Activations are Non-Surjective | **PARTIAL** — off-manifold ≠ budget |
| `2601.19375` | displacement budget | REAL | Selective Steering: Norm-Preserving Control Through Discriminative Layer Selection | **NO — argues the OPPOSITE of HC-1** |
| `2509.22067` | direction fungibility | REAL | The Rogue Scalpel: Activation Steering Compromises LLM Safety | **PARTIAL** — already the §10 foundation, not a discovery |
| `2602.06801` | direction fungibility | REAL | On the Non-Identifiability of Steering Vectors in LLMs | **SUPPORTS** — the one genuine scoop in Group B |
| `2606.20852` | direction fungibility | **WRONG PAPER** | Activation Steering for Pneumonia Classification on Chest X-rays | **FALSE CITATION** |
| `2603.14623` | conformal gate | REAL | Proactive Routing to Interpretable Surrogates with Distribution-Free Safety Guarantees | **SUPPORTS** — AUC-feasibility confirmed at theorem level |

**8/8 resolve. 7/8 are real papers correctly identified. 1 is a false citation.**

---

## 2. The false citation

### `2606.20852` — cited as scooping direction fungibility; is a radiology paper

Source: <https://arxiv.org/abs/2606.20852>

> *Translating Inference-Time Control to Radiology Vision-Language Models: Activation
> Steering for Pneumonia Classification on Chest X-rays* — Farina, Esmeraldo, Matsuoka,
> Kuriki, Kitamura. Submitted 18 Jun 2026, cs.CV.

Tests whether Contrastive Activation Addition improves pneumonia classification in three
frozen chest-radiograph VLMs on the Kermany dataset. It has **nothing** to do with direction
fungibility, equivalence classes of steering vectors, or non-identifiability. Its sole
connection to the claim it was cited for is the phrase "activation steering."

**This is the §18.8 pattern exactly.** It failed *silently and plausibly*: correctly
formatted, sitting in the constitution between two real papers, and it survives any check
that only asks "does the ID resolve?" It fails the instant anyone reads the title.

Whether some *other* paper was intended is unknowable from here and no guess is offered.
The finding is not "could not find the intended paper" — it is **"this ID points at
radiology."**

**Action: strike `2606.20852` from §18.2.**

---

## 3. The adversary misfiled as a scoop

### `2601.19375` — *Selective Steering: Norm-Preserving Control Through Discriminative Layer Selection*

Source: <https://arxiv.org/abs/2601.19375> — Quy-Anh Dang, Chris Ngo. Submitted 27 Jan 2026.

Its two claimed innovations are (1) a **norm-preserving rotation** formulation "that maintains
activation distribution integrity" and (2) discriminative layer selection. It reports zero
perplexity violations and ~100% capability retention, and states explicitly that Angular
Steering's implementation "violates norm preservation, causing distribution shift and
generation collapse, **particularly in models below 7B parameters**."

**HC-1 measured the opposite, in that same sub-7B regime.** On Gemma-3-1B: norm preservation
(r=0) is the **worst** allocation, and at f=0.10 pure rotation costs **4×** the perplexity of
pure addition.

So HC-1 is not pre-empted by this paper — it **contradicts** it. That is a sharper
contribution than the one §18.2 mourns as lost, *provided HC-1 is re-run to this paper's
protocol*. Its layer-selection half also overlaps **HC-3** (layer 12 → 11) and belongs in
that write-up's citations.

**Action: move `2601.19375` out of the "scooped us" list and into an adversary/contradiction
line.** Filing an adversary as a scoop is how a live disagreement gets silently retired.

---

## 4. What the genuine scoops actually cost

**`2606.06735`** — *A Geometric Account of Activation Steering through Angle-Norm
Decomposition* (Aparin, Gaintseva; 4 Jun 2026). Source: <https://arxiv.org/abs/2606.06735>.
Concludes steering "should be parameterized by interpretable angular and radial components
of the intervention, rather than by a single additive coefficient that entangles these two
effects" — **HC-1's own angle/radius-at-fixed-chord parameterisation, arrived at
independently, across seven models to HC-1's one.**

HC-1 is therefore **not novel as a parameterisation**. What may survive is its specific
ordering result (r=0 worst; pure rotation 4× the PPL cost of pure addition at f=0.10) — this
paper says norm "remains important for stability," which is directionally compatible but is
not the same measured claim. **Any HC-1 write-up must be positioned as a result *within* this
framework, not as the framework.**

**`2602.06801`** — *On the Non-Identifiability of Steering Vectors in LLMs* (Venkatesh,
Kurapath; v1 6 Feb 2026, v4 1 Apr 2026, code released). Source:
<https://arxiv.org/abs/2602.06801>. "Large equivalence classes of behaviorally
indistinguishable interventions" is the fungibility thesis stated outright.

**But note the measurement mismatch.** That paper claims *behavioural* indistinguishability.
**HC-S measured a perplexity cost** between two directions at cos = 0.966. Those are not the
same measurement, and HC-S may be a **refinement** of non-identifiability rather than a
casualty of it: *behaviourally equivalent, coherence-wise not*. That is a publishable gap.

### The framing correction

Both Group-B papers that are real **were already known to the program** — `2509.22067`
underpins §10 (v1 dated **26 Sep 2025**), and `2602.06801` already has a `non_identifiability`
lesson built against it with an **unrun alpha sweep** (§18.6 item 3). Neither was discovered
by a Jun–Jul 2026 scan.

> The accurate account is **not** "scooped while the program sat idle." It is **"built lessons
> against published results and did not run them."** That is a different problem, and a more
> fixable one.

---

## 5. The conformal gate — full-text check of `2603.14623`

Source: <https://arxiv.org/abs/2603.14623>, full text <https://arxiv.org/html/2603.14623v1>.
Uddin, Khider, Bauer. Submitted 15 Mar 2026.

### The AUC-feasibility reading is CONFIRMED, at theorem level

> **Proposition 2 (Feasibility condition), §5.1:** "The constraint V(t)≤α is satisfiable with
> positive coverage if and only if there exists threshold t such that:
> TPR(t)/FPR(t) ≥ C(π,α) ≜ (1−π)(1−α)/(πα)."

> **Theorem 1 (Sufficient AUC for feasibility), §5.2:** "If AUC ≥ Φc(π,α), then there exists a
> routing threshold t satisfying V(t)≤α with positive coverage," where Φc(π,α) = min(1, C(π,α)/2).

The paper's central result **is** a feasibility condition with an explicit AUC threshold.
§18.2's characterisation is accurate. It also does **not** scoop a steering-domain conformal
gate: it routes **which model runs** on 35 OpenML tabular datasets and states it "does not
involve intervening on model internals."

### But the inference §18.2 draws from it is over-claimed

> **§5.2, Remark 2:** "The residual gap (observed AUC 0.57 < 0.76, yet routing succeeds)
> reflects a fundamental limit of scalar summaries: feasibility is a **local** ROC property
> (Corollary 1), while AUC is **global**."

> **§6.3:** "Moderate AUC can suffice due to local ROC slope" — "the conformal search can
> select a threshold that isolates a small, high-purity region even when global AUC is modest."

Theorem 1's bound is **sufficient, not necessary**, and the authors demonstrate successful
routing at **AUC 0.57**, well under their own 0.76 threshold. A higher-AUC probe therefore
does **not** unlock a regime the paper leaves closed.

### Two consequences, both good news

1. **The program's judge calibration at AUC 0.7508 is not the wound §18.2 treats it as.**
   Against this framework, 0.7508 is comfortably inside the regime where conformal routing is
   *demonstrated* to work. The 0.85 gate is a **self-imposed bar, not a feasibility bar**.
2. **The contribution to claim is not "we got a better gate."** It is that feasibility is a
   **local ROC property** — so the right thing to report for a steering gate is the **local
   slope in the operating region**, not the global AUC. The program currently reports global
   AUC everywhere. That is a concrete, cheap, **judge-free** change with a published theorem
   behind it.

**Action: narrow the §18.2 phrase "our probe result *strengthens*".** The paper is not waiting
on a better gate. The opening is the local-ROC framing.

---

## 6. Novelty sweep — is the conformal gate still open?

**The idea.** Apply conformal prediction / distribution-free risk control to the **gate** of a
conditionally-applied activation steering vector (CAST-style), so that *when* the steering
fires carries a calibrated, finite-sample guarantee.

### Verdict: **STILL OPEN**

**Epistemic status: this is "I could not find prior work," not "there is no prior work."** The
strength rests on exhaustive arXiv full-text conjunctions, which enumerate the corpus rather
than rank-and-truncate like a web search:

| exhaustive arXiv conjunction | total results | on-topic |
|---|---|---|
| `all:"conformal" AND all:"activation steering"` | 2 | **0** (protein folding; rare-event control) |
| `all:"conformal" AND all:"steering vector"` | 2 | **0** (beamforming; conformer ASR) |
| `all:"conformal" AND all:"residual stream"` | **0** | **0** |
| `all:"risk control" AND all:"steering vector"` | **0** | **0** |

**Residual risk, stated honestly:** the arXiv API matches phrases, so a paper using different
vocabulary ("calibrated context gate", "certified representation control", "PAC-Bayes
trigger") would be missed. Non-arXiv venues and 2026 workshop tracks are outside this sweep —
two are live in exactly this space (ICML 2026 *Statistical Frameworks for Uncertainty in
Agentic Systems*; TrustNLP@ACL 2026) and were **not** enumerated paper-by-paper.

### The three nearest published works

1. **`2606.12299`** — *Learning What to Say to Your VLA: Mostly Harmless Vision Language Action
   Model Steering*. Jeong, Swamy, Bajcsy. **10 Jun 2026.** <https://arxiv.org/abs/2606.12299>
   *Nearest by structure.* Conformalizes an "improvement head" predicting when steering will
   help, explicitly "to prevent harmful steering interventions" — but steers a frozen VLA robot
   policy via **language feedback**, controlling **task degradation**, not activation editing.
2. **`2603.14623`** — *Proactive Routing…* (§5 above). *Nearest by machinery* —
   Clopper-Pearson calibration of a gate threshold with an AUC feasibility condition — but
   gates **which model runs**, on tabular data.
3. **`2604.19775`** — *From Actions to Understanding: Conformal Interpretability of Temporal
   Concepts in LLM Agents*. Padhi et al. **27 Mar 2026** (rev 1 Jul 2026).
   <https://arxiv.org/abs/2604.19775> *Nearest by substrate* — the only paper found doing
   conformal prediction **and** activation-space steering in one work. But conformal is used
   **upstream, to label representations for probe training**; the steering is unconditional and
   reported as "preliminary."

**Runner-up worth watching:** `2605.14746` *Selective Safety Steering via Value-Filtered
Decoding* (Einbinder, Davidov, Teh, Gal, Romano; 14 May 2026, rev 12 Jul 2026) names the exact
problem — "existing decoding-time steering methods often intervene unnecessarily" — and gives
"an explicit bound on the probability of false interventions," but at the **decoding/token**
layer and (verified) **without** conformal prediction. Given the author list is the Romano
conformal group, **a conformalized sequel on this line is the most likely route by which this
idea gets scooped.**

### The precise gap that remains open

Every conformal-gates-an-LLM-intervention paper found gates at the **output / action / token /
prompt** layer: abstain, re-look, re-ask, execute-or-not, say-this-instead. Every
conditional-activation-steering paper found sets its firing threshold by **grid search or
empirical tuning** with no finite-sample validity — including **CAST itself** (`2409.05907`,
cosine-similarity threshold tuned by grid search) and ASA (`2602.04935`) `[UNVERIFIED]`.

> **Nobody has put a distribution-free, finite-sample guarantee on the decision to WRITE to the
> residual stream.**

Specifically open:

- **Conformal calibration of the CAST condition-vector threshold** — converting "cosine > τ,
  τ found by grid search" into "the false-fire rate on benign prompts is ≤ α with probability
  1−δ."
- **The two-sided version** — controlling **over-refusal** (fires on benign) and **miss**
  (silent on harmful) simultaneously. This is precisely the **selectivity axis** of the §6
  composite.
- **`2603.14623`'s feasibility condition instantiated on a *steering* gate** — what probe AUC
  is required, for a given (α, base harmful rate), to admit a feasible threshold at all. **This
  is the point at which the program's judge-free probe-AUC measurements become the input to a
  novel result rather than a repetition of one.**

---

## 7. What this means for the program

| # | consequence | cost |
|---|---|---|
| 1 | **§18.2 needs three corrections**: strike `2606.20852` (radiology); re-file `2601.19375` as an adversary, not a scoop; narrow "our probe result *strengthens*" to the local-ROC framing. | doc edit |
| 2 | **HC-1 survives, re-scoped.** Not novel as a parameterisation (`2606.06735` got there first, 7 models). Novel as a **contradiction of `2601.19375`** in the sub-7B regime — but only if re-run to that paper's protocol. | one sweep |
| 3 | **HC-S survives, re-scoped.** `2602.06801` claims *behavioural* non-identifiability; HC-S measured a *perplexity* cost at cos 0.966. Behaviourally equivalent, coherence-wise not — a gap, not a casualty. | write-up |
| 4 | **Report local ROC slope, not global AUC**, for every gate in the program. Published theorem behind it, judge-free, cheap. | code change |
| 5 | **The 0.85 judge gate is self-imposed.** At AUC 0.7508 the program is inside the regime `2603.14623` demonstrates works (they route at 0.57). | reframing |
| 6 | **The conformal gate is still the live contribution — and it is time-limited.** The Romano group (`2605.14746`) is one paper away from it. | prioritise |
| 7 | **`2602.02712` (ICML 2026) reports *non-monotonic* strength effects; HC-1's headline is *monotone*.** Different quantities (strength vs angle/radius at fixed chord), so this is a tension to resolve, not a contradiction — but it must be read before HC-1 ships. | read |

---

## 8. Method note

All eight §18.2 IDs were fetched from their own arXiv abstract pages; `2603.14623` was read in
HTML full text. The novelty sweep ran **22 distinct queries** — 10 WebSearch, 6 exhaustive
arXiv-API conjunctions, 6 primary-abstract WebFetch verifications — and logged negative queries
as well as positive ones, so the coverage is auditable rather than asserted. No secondary
sources (blogs, threads, summaries) were used as evidence for what any paper claims. No fetch
failed; nothing in §§1–5 is marked `[UNVERIFIED]`.

Raw agent working notes: `part1_ids.md` (citation audit) and `part2_novelty.md` (sweep log with
all 22 queries), in the session scratchpad.
