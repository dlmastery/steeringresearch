# Part 1 — Primary-source verification of the 8 arXiv IDs cited in CLAUDE.md §18.2

Task: fetch https://arxiv.org/abs/<ID> for each. Record REAL / NOT REAL, exact
title, authors, date, venue, what it claims (from its own abstract), and whether
it supports the §18.2 characterisation.

Status: **COMPLETE** — all 8 fetched. 8/8 IDs resolve; 1 is a false citation
(2606.20852 is a radiology paper). See SUMMARY at the end.

| ID | group | REAL? | title | verdict vs §18.2 |
|---|---|---|---|---|
| 2602.02712 | A (displacement budget) | **REAL** | Towards Understanding Steering Strength | SUPPORTS (strongly) |
| 2606.06735 | A | **REAL** | A Geometric Account of Activation Steering through Angle-Norm Decomposition | SUPPORTS (strongest scoop) |
| 2604.09839 | A | **REAL** | Steered LLM Activations are Non-Surjective | PARTIAL — different claim (off-manifold, not budget) |
| 2601.19375 | A | **REAL** | Selective Steering: Norm-Preserving Control Through Discriminative Layer Selection | DOES NOT SCOOP — it asserts the OPPOSITE of HC-1 |
| 2509.22067 | B (direction fungibility) | **REAL** | The Rogue Scalpel: Activation Steering Compromises LLM Safety | PARTIAL — real, but it is the §10 red-team paper, double-booked |
| 2602.06801 | B | **REAL** | On the Non-Identifiability of Steering Vectors in Large Language Models | SUPPORTS (a true scoop) |
| 2606.20852 | B | **WRONG PAPER** | Activation Steering for Pneumonia Classification on Chest X-rays | **FALSE CITATION — radiology, nothing to do with fungibility** |
| 2603.14623 | C (conformal gate) | **REAL** | Proactive Routing to Interpretable Surrogates with Distribution-Free Safety Guarantees | SUPPORTS — feasibility-conditioned on gate AUC CONFIRMED |

---

## Per-ID findings

### GROUP A — cited as scooping the "displacement budget" contribution

---

#### 2602.02712 — **REAL**
Source: https://arxiv.org/abs/2602.02712

- **Title:** *Towards Understanding Steering Strength*
- **Authors:** Magamed Taimeskhanov, Samuel Vaiter, Damien Garreau
- **Submitted:** 2 Feb 2026 (v1); last revised 8 Jul 2026 (v2)
- **Venue (comments field):** "Accepted for publication at ICML 2026 (50 pages)"

**Claims (its own abstract):** "the first theoretical analysis of steering
strength." Characterises the effect of the steering magnitude on next-token
probability, concept presence, and cross-entropy, "deriving precise qualitative
laws"; reports "surprising behaviors, including **non-monotonic** effects of
steering strength"; validated on eleven language models.

**Verdict vs §18.2: SUPPORTS, and is the most serious of the four.** This is a
theory paper about exactly the quantity the displacement-budget contribution was
about — how far you may move, and what it costs. Note the direct collision with
HC-1: this paper reports **non-monotonic** effects of strength, whereas HC-1's
headline is that angle/radius at fixed chord is **monotone, not U-shaped**. The
two are not measuring the identical quantity (strength vs angle/radius
allocation at fixed chord), so this is a tension to resolve, not an automatic
contradiction — but it must be read before HC-1 is written up.

---

#### 2606.06735 — **REAL**
Source: https://arxiv.org/abs/2606.06735

- **Title:** *A Geometric Account of Activation Steering through Angle-Norm Decomposition*
- **Authors:** Georgii Aparin, Tatiana Gaintseva
- **Submitted:** 4 Jun 2026 (v1); revised 8 Jun 2026 (v2)
- **Venue:** none stated (cs.AI listing only)

**Claims (its own abstract):** Revisits the assumption that hidden-state norm
carries no concept-relevant information via "a controlled empirical study
designed to disentangle the roles of angular and radial components." Finds
across seven models that "concepts are represented primarily in angular
structure ... but that norm remains important for the stability and downstream
effects of steering," and concludes steering "should be parameterized by
interpretable angular and radial components of the intervention, rather than by
a single additive coefficient that entangles these two effects."

**Verdict vs §18.2: SUPPORTS — this is the strongest scoop of the eight.** It is
not merely adjacent: *angular and radial components* is HC-1's own
angle/radius-at-fixed-chord parameterisation, arrived at independently, published
4 Jun 2026, with seven models to HC-1's one. HC-1 is not novel as a
*parameterisation*. What may survive is HC-1's specific ordering result (r=0 is
the WORST allocation; pure rotation costs 4x the perplexity of pure addition at
f=0.10) — this paper says norm "remains important for stability," which is
directionally compatible but is not the same measured claim. Any HC-1 write-up
must now be positioned as a result *within* this paper's framework, not as the
framework.

---

#### 2604.09839 — **REAL** (but mischaracterised)
Source: https://arxiv.org/abs/2604.09839

- **Title:** *Steered LLM Activations are Non-Surjective*
- **Authors:** Aayush Mishra, Daniel Khashabi, Anqi Liu
- **Submitted:** 10 Apr 2026 (v1); last revised 7 May 2026 (v2)
- **Venue (comments):** ICLR 2026 Workshops (Sci4DL, Re-Align)

**Claims (its own abstract):** Casts as a surjectivity question whether steered
behaviour is reachable by any prompt. Proves under stated assumptions that
"activation steering pushes the residual stream off the manifold of states
reachable from discrete prompts" and that "almost surely, no prompt can reproduce
the same internal behavior induced by steering." Concludes with a "formal
separation between white-box steerability and black-box prompting."

**Verdict vs §18.2: PARTIAL / MISCHARACTERISED.** This paper is about
**off-manifold-ness as a binary, formal property** — that steered states have no
prompt preimage. It is *not* about a displacement budget: it does not quantify
how much displacement is affordable, does not trade displacement against
perplexity, and offers no allocation rule. It is a strong prior for the
program's `offshell_displacement` geometry axis (§3) and it is highly relevant
to `prompt_activation_duality`, but calling it a scoop of the displacement-budget
contribution overstates it. Listing it alongside 2606.06735 inflates the
apparent weight of the scoop.

---

#### 2601.19375 — **REAL** (and it does NOT support the characterisation)
Source: https://arxiv.org/abs/2601.19375

- **Title:** *Selective Steering: Norm-Preserving Control Through Discriminative Layer Selection*
- **Authors:** Quy-Anh Dang, Chris Ngo
- **Submitted:** 27 Jan 2026
- **Venue:** none stated

**Claims (its own abstract):** Proposes Selective Steering with two innovations:
"(1) a mathematically rigorous **norm-preserving rotation** formulation that
maintains activation distribution integrity, and (2) discriminative layer
selection that applies steering only where feature representations exhibit
opposite-signed class alignment." Reports 5.5x higher attack success rates than
prior methods "while maintaining zero perplexity violations and approximately
100% capability retention." Explicitly says Angular Steering's implementation
"violates norm preservation, causing distribution shift and generation collapse,
particularly in models below 7B parameters."

**Verdict vs §18.2: DOES NOT SCOOP — it is a DIRECT ADVERSARY, which is a
better fact than a scoop.** This paper's entire pitch is that **norm preservation
(pure rotation, r=0 in HC-1's coordinates) is the GOOD regime**, and that
violating it is what collapses small models. HC-1 measured the opposite on
Gemma-3-1B: r=0 is the **worst** allocation, and at f=0.10 pure rotation costs
**4x** the perplexity of pure addition. Both claims concern sub-7B models, which
is precisely where this paper stakes its ground. So HC-1 is not pre-empted by
2601.19375; it **contradicts** it, and that is a sharper contribution than the
one §18.2 mourns as lost — provided HC-1 is re-run to this paper's protocol.
Note also that its second innovation (discriminative layer selection) overlaps
HC-3 (layer 12 -> 11) and should be cited there.

**Correction needed in CLAUDE.md §18.2:** 2601.19375 must be moved out of the
"scooped us" list. Filing an adversary as a scoop is how a live disagreement gets
silently retired.

---

### GROUP B — cited as scooping the "direction fungibility" contribution

---

#### 2509.22067 — **REAL**, but it is not a new citation
Source: https://arxiv.org/abs/2509.22067

- **Title:** *The Rogue Scalpel: Activation Steering Compromises LLM Safety*
- **Authors:** Anton Korznikov, Andrey Galichin, Alexey Dontsov, Oleg Y. Rogov,
  Ivan Oseledets, Elena Tutubalina
- **Submitted:** 26 Sep 2025 (v1); revised 15 Feb 2026 (v2)
- **Venue:** none stated (cs.LG listing only)

**Claims (its own abstract):** Activation steering "systematically breaks model
alignment safeguards, making it comply with harmful requests." Random directional
steering raises harmful compliance from 0% to 1-13%; steering *benign* SAE
features has "comparable jailbreak potential"; combining 20 randomly sampled
attack vectors yields a universal attack effective on unseen prompts. Concludes
that "precise control over model internals does not guarantee precise control
over model behavior."

**Verdict vs §18.2: PARTIAL, and the framing is misleading.** Two problems.

(1) *It is not new to this program.* This is the paper CLAUDE.md §10 is built
around — the "Rogue Scalpel mandate," the 20-vector universal attack that §10
requires be reproduced as a Rung-4 red-team probe. §18.2 presents it as part of a
Jun-Jul 2026 literature refresh that discovered work scooping us. The v1 is
**26 Sep 2025**, and the program has been citing it as a design input since §10
was written. Nothing was discovered here.

(2) *It only glances off fungibility.* The relevant content is that **random**
directions and **benign** SAE features jailbreak about as well as purpose-built
ones — which is a fungibility-flavoured result, but confined to the safety
axis, and offered as evidence that steering is dangerous rather than as a claim
about direction equivalence classes. It does not establish that steering
directions are interchangeable in general.

Note the tension with HC-S, which is the program's live counter-evidence:
cos(diffmean, pca) = 0.966 is above the usual 0.95 "same direction" bar and
**still** costs +3.33 PPL. HC-S says directions are *not* freely
interchangeable. That is a finding worth defending, not conceding.

---

#### 2602.06801 — **REAL**, and this one genuinely scoops
Source: https://arxiv.org/abs/2602.06801

- **Title:** *On the Non-Identifiability of Steering Vectors in Large Language Models*
- **Authors:** Sohan Venkatesh, Ashish Mahendran Kurapath
- **Submitted:** 6 Feb 2026 (v1); v2 16 Feb 2026; v3 5 Mar 2026; v4 1 Apr 2026
- **Comments field:** "Code available at https://github.com/sohv/non-identifiability"

**Claims (its own abstract):** Steering vectors are "fundamentally
non-identifiable due to large equivalence classes of behaviorally
indistinguishable interventions" under white-box single-layer access. Orthogonal
perturbations achieve comparable performance with minimal effect sizes across
models. SVD analysis of activation covariance matrices establishes that
non-identifiability persists across operational steering ranges and prompt
distributions. Calls for "structural constraints beyond behavioral testing."

**Verdict vs §18.2: SUPPORTS. This is the one real scoop in Group B.**
"Large equivalence classes of behaviorally indistinguishable interventions" is
the direction-fungibility thesis stated outright, with code released.

**But note:** the program already knows this paper. §18.6's pending list names a
`non_identifiability` lesson with an "unrun alpha sweep." So this was not
discovered by a Jun-Jul 2026 scan either — a lesson was already built against it.
The honest statement is "we built a lesson on a published result and did not run
it," which is a very different failure from "we were scooped while idle."

Also worth stating plainly: this paper claims *behavioural* indistinguishability.
HC-S measured a **perplexity** cost between two directions at cos 0.966. Those
are not the same measurement, and HC-S may well be compatible with — even a
refinement of — non-identifiability: behaviourally equivalent, coherence-wise
not. That is a publishable gap, not a dead contribution.

---

#### 2606.20852 — **REAL ARXIV ID, COMPLETELY WRONG PAPER**
Source: https://arxiv.org/abs/2606.20852

- **Title:** *Translating Inference-Time Control to Radiology Vision-Language
  Models: Activation Steering for Pneumonia Classification on Chest X-rays*
- **Authors:** Eduardo Moreno Judice de Mattos Farina, Mateus A. Esmeraldo,
  Felipe Akio Matsuoka, Paulo Eduardo de Aguiar Kuriki, Felipe Campos Kitamura
- **Submitted:** 18 Jun 2026
- **Listing:** arXiv:2606.20852 [cs.CV]

**Claims (its own abstract):** Tests whether Contrastive Activation Addition can
improve pneumonia classification in three frozen chest-radiograph
vision-language models on the Kermany dataset. "Fixed-threshold F1 improvements
were frequently observed but did not consistently indicate improved diagnostic"
outcomes; one model improved calibrated F1 from 0.7692 to 0.8727 with
image-conditioned steering.

**Verdict vs §18.2: THE CITATION IS FALSE. FLAG LOUDLY.**
This is an applied **medical imaging** paper. It is a chest-X-ray pneumonia
classification study. It has **nothing whatsoever** to do with direction
fungibility, equivalence classes of steering vectors, or non-identifiability. Its
only connection to the cited claim is the phrase "activation steering."

Per the task's own rule — an ID resolving to a paper on an unrelated subject is
NOT a valid citation — **2606.20852 must be struck from §18.2.** The ID exists,
so this would survive any check that only asks "does the ID resolve?"; it fails
the moment anyone reads the title. This is exactly the R17.4 failure mode
(citing from memory, never fetching), and exactly the §18.8 pattern: it failed
**silently and plausibly**, formatted correctly, sitting in the constitution
next to two real papers.

The likely mechanism: an ID pattern-matched from memory to fill out a group of
three. Whether some *other* paper was intended is unknowable from here — I make
no guess. **"Could not find the intended paper" is not the finding; "this ID
points at radiology" is.**

---

### GROUP C — the conformal gate paper

---

#### 2603.14623 — **REAL**
Source: https://arxiv.org/abs/2603.14623

- **Title:** *Proactive Routing to Interpretable Surrogates with Distribution-Free
  Safety Guarantees*
- **Authors:** Iqtedar Uddin, Mazin Khider, André Bauer
- **Submitted:** 15 Mar 2026
- **Venue:** none stated

**Claims (its own abstract):** Studies **proactive (input-based) routing**, where
"a lightweight gate selects the model before either runs, enabling
distribution-free control of the fraction of routed inputs whose degradation
exceeds a tolerance τ." The gate distinguishes safe from unsafe inputs; the
routing threshold is set by "Clopper-Pearson conformal calibration on a held-out
set, guaranteeing that the routed-set violation rate is at most α with
probability 1-δ." Evaluated on 35 OpenML datasets.

**Verdict vs §18.2: SUPPORTS — and the "feasibility-conditioned on gate AUC"
reading is CONFIRMED, from the abstract alone.** See the dedicated section below.

---

## 2603.14623 FULL-TEXT CHECK — is it "feasibility-conditioned on gate AUC"?

Source: https://arxiv.org/html/2603.14623v1 (HTML v1 retrieved successfully)

### VERDICT: the reading is CONFIRMED — but with a caveat that cuts against the
### inference §18.2 draws from it.

**Part 1 — the feasibility-conditioning is real, explicit, and theorem-level.**

> **Proposition 2 (Feasibility condition), §5.1:** "The constraint V(t)≤α is
> satisfiable with positive coverage if and only if there exists threshold t such
> that: TPR(t)/FPR(t) ≥ C(π,α) ≜ (1−π)(1−α)/(πα)."

> **Theorem 1 (Sufficient AUC for feasibility), §5.2:** "If AUC ≥ Φc(π,α), then
> there exists a routing threshold t satisfying V(t)≤α with positive coverage,"
> where Φc(π,α) = min(1, C(π,α)/2).

> **Theorem 2 (Tight AUC under concavity), §5.2:** For concave ROC curves: "If
> AUC ≥ Φc*, feasible routing exists."

That is as literal a confirmation as one could ask for. The paper's central
result is a **feasibility condition**, and it publishes an explicit **AUC
threshold** above which safe routing is guaranteed to exist. §18.2's
characterisation of the paper is accurate.

It also follows that the paper does **not** scoop a conformal-gate contribution
in the steering domain. Its evaluation is "35 OpenML datasets and multiple
black-box model families" — tabular model routing to interpretable surrogates.
It is not about LLMs, activations, or safety gating. The §18.2 conclusion that
the conformal gate "survives as novel" stands.

**Part 2 — the caveat: the paper itself says a strong gate is NOT the binding
constraint, which weakens the "our probe strengthens it" inference.**

§18.2 reasons that because the paper is feasibility-conditioned on gate AUC, "our
probe result *strengthens* rather than scoops it." The first half is right; the
second half is over-claimed, and the paper says so directly:

> **§5.2, Remark 2:** "The residual gap (observed AUC 0.57 < 0.76, yet routing
> succeeds) reflects a fundamental limit of scalar summaries: feasibility is a
> local ROC property (Corollary 1), while AUC is global."

> **§6.3:** "Moderate AUC can suffice due to local ROC slope" because "the
> conformal search can select a threshold that isolates a small, high-purity
> region even when global AUC is modest."

So: Theorem 1's AUC bound is **sufficient, not necessary**, and the authors
demonstrate successful routing at **AUC 0.57** — well under their own 0.76
threshold. A higher-AUC probe therefore does not unlock a regime the paper leaves
closed; the paper has already shown the regime is open without one.

**What this actually means for the program — and it is good news, twice over:**

1. The program's judge calibration reached **AUC 0.7508**, under its own 0.85
   gate, and §18.2 treats this as a wound. Against this paper's framework, 0.7508
   is *comfortably* in the regime where conformal routing is demonstrated to
   work. The 0.85 gate is a self-imposed bar, not a feasibility bar.
2. The contribution to claim is **not** "we got a better gate." It is that
   feasibility is a **local ROC property** (Corollary 1) — which means the right
   thing to report for a steering gate is the *local slope in the operating
   region*, not the global AUC. The program currently reports global AUC
   everywhere. That is a concrete, cheap, judge-free change with a published
   theorem behind it.

**Correction needed in CLAUDE.md §18.2:** the phrase "our probe result
*strengthens*" should be narrowed. The paper is not waiting on a better gate.
The opening is the local-ROC framing, not the AUC number.

---

## SUMMARY

**8 of 8 IDs resolve to real arXiv papers. 7 of 8 are correctly cited.
1 is a false citation.**

| verdict | IDs |
|---|---|
| Real + correctly characterised | 2602.02712, 2606.06735, 2602.06801, 2603.14623 |
| Real but mischaracterised / overstated | 2604.09839 (off-manifold ≠ budget), 2509.22067 (already the §10 paper, not a discovery) |
| Real but **argues the OPPOSITE** of what it is filed under | 2601.19375 (norm preservation is its *pitch*; HC-1 contradicts it) |
| **FALSE CITATION — unrelated subject** | **2606.20852 (chest-X-ray pneumonia classification)** |

**The scoop is smaller than §18.2 says.** Of the four Group-A "displacement
budget" citations, exactly one (2606.06735) is a genuine scoop of the
parameterisation, one is a strong theory neighbour (2602.02712), one is a
different claim (2604.09839), and one is an adversary misfiled as a scoop
(2601.19375). Of the three Group-B "fungibility" citations, exactly one
(2602.06801) genuinely scoops, one was already the program's own §10 foundation
(2509.22067), and one does not exist as cited (2606.20852).

**Both Group-B papers that are real were already known to the program** —
2509.22067 underpins §10, and 2602.06801 has a `non_identifiability` lesson built
against it (with an unrun alpha sweep, per §18.6). Neither was discovered by a
Jun-Jul 2026 scan. The accurate account is not "scooped while the program sat
idle" but "built lessons against published results and did not run them," which
points at a different and more fixable problem.

**No claim above rests on anything but the papers' own arXiv pages** (abstract
pages for all eight; HTML full text for 2603.14623). No secondary sources were
used. No fetch failed, so nothing is marked [UNVERIFIED].



