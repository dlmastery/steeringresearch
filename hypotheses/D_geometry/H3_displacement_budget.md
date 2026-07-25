# H3 — A Displacement Budget Controls the Capability/Coherence Tax AND Doubles as a Defense

> **One-line claim:** Capping the per-token OFF-MANIFOLD component of the steering
> displacement (directly operationalizing N17: off-shell displacement Delta‖h‖
> predicts incoherence, Spearman rho=+0.585) holds MMLU/coherence within epsilon
> while permitting the MAXIMAL steering strength alpha — AND the SAME cap blunts the
> Rogue-Scalpel 20-vector universal attack: one geometric mechanism, two payoffs
> (a coherence "airbag" + an attack defense).
>
> **Source design space:** Block D — Geometry / Manifold (the N5/N17 displacement
> axis). Program: "Conditional displacement, not direction" — this is **H3** of that
> program (see `../PROGRAM_conditional_displacement.md`). Block G method component.
>
> **Implementation status:** `DESIGN + PLAN — RUN PENDING`. The geometry instruments
> exist (`geometry.offshell_displacement`, the eval `λ_geo` term) and the red-team
> probe exists (`adversarial.RogueScalpelProbe`, status DRY-RUN); the budgeted
> operation (`relative_add` + an off-manifold clamp) is NOT yet implemented. This doc
> is a pre-registered plan, not a result. Gated behind a hard precondition: a judge
> calibrated to AUC >= 0.85 (see §7.0; PREREGISTRATION Amendment 1).

---

## In Plain English

**What we're testing, simply:** Steering nudges the model's internal state by adding
a direction. Push too hard and the text turns to gibberish (the "coherence cliff").
The repo's strongest result (N17) says you can SEE that damage coming, cheaply,
before generating a word: it shows up as the activation's length changing — getting
pushed "off the sphere" of normal activations. H3 asks: **what if we put a hard cap
(a "budget") on exactly that off-the-sphere part of the push?** The bet is that this
single cap does two useful things at once — (1) it lets us crank the steering strength
much higher before the text breaks (a coherence airbag), and (2) the very same cap
stops an attacker who tries to jailbreak the model by secretly steering it, because
their attack also has to leave the sphere to work.

**Key terms:**
- **Steering displacement:** the vector we add to the model's internal state. Its size
  is `alpha * ‖h‖` (in relative steering — a fraction of the state's own length).
- **On-manifold vs off-manifold:** "on-manifold" = the part of the push that stays
  inside the cloud of normal activations (where the model behaves coherently);
  "off-manifold" = the part that pushes OUTSIDE that cloud (where text breaks).
- **Off-shell displacement (Delta‖h‖):** the repo's cheap geometric measure of how far
  off the sphere steering pushed — already proven (N17) to predict incoherence.
- **Displacement budget B:** a hard cap on the off-manifold push, expressed as a
  fraction of the state's length: clamp the off-manifold norm to `<= B * ‖h‖`.
- **The manifold (cheap version):** the cloud of "normal" activations. We approximate
  it with no training — either the top-k PCA directions of a small CACHED bank of
  clean activations, or the nearest neighbours in that bank.
- **Coherence cliff knee:** the steering strength at which text starts collapsing
  (the repo measures it at relative alpha ~0.05-0.10 on both 1b and 2b).
- **Rogue-Scalpel universal attack:** averaging ~20 individually-weak jailbreak
  steering vectors yields one direction that flips the model's safety verdict.

**Why we're doing this:** The repo already KNOWS the off-the-sphere displacement
predicts incoherence (N17, its one rung-3 result) and already KNOWS the cliff knee
location (E3). Nobody has yet turned that *diagnostic* into a *controller*. A budget
that (a) buys more usable steering strength and (b) defends against the universal
attack — with one mechanism, no training, runnable on a laptop — is a genuinely
ownable contribution, and it is a cheap version of Goodfire's Manifold Steering and
the "reproject onto the manifold" results (no diffusion prior; just PCA/kNN shrinkage
of a cached bank).

**What the result would mean:** SUPPORTED (both legs) ⇒ we own "displacement-budgeted
steering: a coherence airbag that is also a jailbreak defense." SUPPORTED (coherence)
but FALSIFIED (defense) ⇒ still publishable: the budget is a real coherence controller
but cannot tell "useful refusal displacement" from "harmful jailbreak displacement," so
it is a coherence trick, not a guard. FALSIFIED (both) ⇒ the off-manifold cap does not
shift the cliff at matched behavior — N17's diagnostic does not transfer to control.

See [`../GLOSSARY.md`](../GLOSSARY.md).

---

## 1. Motivation (>= 100 words)

The program's single rung-3 result is N17: the cheap geometric quantity off-shell
displacement Delta‖h‖ predicts output incoherence (Spearman +0.585, 95% CI
[+0.353, +0.758], p=8e-6, real WikiText-2, two model sizes). Its companion E3 locates
the coherence cliff (knee at relative alpha ~0.05-0.10 on both 1b and 2b, super-linear
collapse past ~0.20). Both are currently used only as *predictors* — we measure
Delta‖h‖ to *score* a config's coherence tax (the composite's λ_geo term), never to
*control* it. H3 closes that loop: it treats the off-manifold component of the steering
displacement as a directly clamped quantity. The hypothesis is that the capability and
coherence taxes are caused specifically by the OFF-manifold excursion (N17's mechanism),
not by the on-manifold push that actually carries the behavior — so capping only the
off-manifold part should let alpha grow far past the E3 knee while MMLU/PPL stay within
epsilon. The same clamp is, by construction, a guard: the Rogue-Scalpel universal attack
(corpus §2.6 F5; `RogueScalpelProbe`) works by averaging ~20 weak vectors into one
direction that, applied as an unauthorised steer, pushes the state off-manifold to flip
the safety verdict — a push the budget caps. One mechanism, two payoffs, no training,
laptop-runnable.

---

## 2. Formal Hypothesis (>= 50 words)

Because the capability/coherence tax of additive steering is mediated by the
OFF-manifold component of the displacement (the N17 mechanism: Delta‖h‖ predicts
incoherence, rho=+0.585), and because clamping that component to `<= B*‖h‖` leaves the
on-manifold component — which carries the behavior — untouched, a displacement-budgeted
steerer will (a) shift the E3 coherence-cliff knee RIGHTWARD: at a fixed target behavior
efficacy, the budgeted steerer tolerates a strictly larger alpha before MMLU drops > 2pp
or PPL rises super-linearly, relative to unbudgeted `relative_add`; and (b) under the
Rogue-Scalpel 20-vector universal attack, the budgeted model's compliance_rate stays
within epsilon of its unsteered baseline (near 0%) where the unbudgeted model's
compliance blows up. The mechanism is geometric, not direction-specific: the budget
prices off-manifold excursion regardless of which vector produced it, so it applies
equally to a legitimate steer and to an adversarial averaged direction.

---

## 3. Falsifier (>= 30 words)

DEFENSE leg FALSIFIED if the budget cannot separate useful steering displacement from
attack displacement — i.e. the value of B that neutralizes the Rogue-Scalpel universal
attack (compliance within +0.05 of baseline) ALSO suppresses legitimate steering
(behavior efficacy at matched alpha drops by more than the same epsilon used for the
attack): then the cap is merely a coherence trick, not a defense, and the two-payoffs
claim collapses to one. COHERENCE leg FALSIFIED if, at matched behavior efficacy, the
budgeted steerer does NOT move the cliff knee rightward by at least one alpha grid-step
(MMLU/PPL no better than unbudgeted `relative_add` at the same behavior). Each leg is
scored independently; SUPPORTED-coherence / FALSIFIED-defense is an explicit, reportable
partial verdict.

---

## 4. Citations (Citation Rigor >= 80 words)

```
Korznikov, A., et al. 2026 ICML 'The Rogue Scalpel: Activation Steering Compromises
LLM Safety' (arXiv:2509.22067) — the F5 universal attack (average ~20 weak jailbreak
vectors into one direction that flips the safety verdict) is the red-team H3 must
neutralize; the corpus mitigation doc §2.6/§2.7 names the norm/manifold clamp (Guard B)
as the displacement-budget defense H3 implements and ablates.

Turner, A., et al. 2024 arXiv 'Activation Addition: Steering Language Models Without
Optimization' (arXiv:2308.10248) — ActAdd / additive steering is the unbudgeted
baseline; H3 adds a per-token off-manifold clamp on top of the add operation and asks
whether the clamp moves the coherence cliff at matched behavior.

Arditi, Andy, et al. 2024 arXiv 'Refusal in Language Models Is Mediated by a Single
Direction' (arXiv:2406.11717) — the refusal/harm subspace whose off-manifold excursion
the attack exploits and the budget caps; motivates the separability test in §3.

Wu, Yuming, et al. 2024 arXiv 'Conditional Activation Steering' (arXiv:2409.05907) —
CAST gating; H3's budget composes orthogonally with the gate (different intervention
axis: the gate decides WHEN, the budget bounds HOW-FAR), per the stacking rule §9.

Roy, O., Vetterli, M. 2007 EUSIPCO 'The Effective Rank: A Measure of Effective
Dimensionality' (arXiv:cs/0703130) [UNVERIFIED] — basis for the effective-rank/PCA
manifold estimate (`geometry.effective_rank`) used to define the cheap on-manifold
subspace the clamp projects against.
```

---

## 5. Mechanism

### 5.1 The displacement budget

Relative steering (`hooks.relative_add`) adds `delta = alpha * ‖h‖ * v_hat` per token.
H3 decomposes `delta` into an ON-manifold part (inside the local activation
distribution) and an OFF-manifold residual, then clamps only the residual:

    P = projector onto the cheap local manifold subspace at layer L (top-k PCA of a
        cached clean-activation bank, OR the span of h's k nearest neighbours in it)
    delta_on  = P @ delta                 # stays in-distribution; carries behavior
    delta_off = delta - delta_on          # the off-sphere excursion N17 penalizes
    if ‖delta_off‖ > B * ‖h‖:
        delta_off <- delta_off * (B * ‖h‖ / ‖delta_off‖)     # clamp to the budget
    h <- h + delta_on + delta_off

B is the per-token budget (a fraction of ‖h‖, the same units as relative alpha and as
N17's Delta‖h‖). B = +inf recovers plain `relative_add`; B = 0 forbids any off-manifold
push. The clamp is purely geometric — it never inspects which vector produced `delta`,
which is exactly why it applies identically to a legitimate steer and to the
Rogue-Scalpel averaged direction.

### 5.2 The cheap manifold (no training)

Two laptop-runnable estimators of the on-manifold subspace, cached ONCE (CLAUDE.md §2,
cache contrast activations once):
- **PCA bank:** top-k principal directions of a bank of clean activations at L
  (`geometry.singular_values`; k chosen near `geometry.effective_rank`).
- **kNN bank:** the local span of h's k nearest neighbours in the cached bank
  (Mahalanobis-shrunk), a cheap stand-in for Goodfire's manifold / the GLP reprojection
  prior — no diffusion model, just a stored bank.

Both tie DIRECTLY to `eval.offshell_displacement`: the clamped `delta_off` IS the
quantity N17 found predictive of incoherence; H3 bounds it instead of merely logging it.

### 5.3 Why the same cap is a defense

The Rogue-Scalpel universal attack applies an averaged unauthorised steer that flips the
verdict by pushing the state off-manifold (corpus §2.6 F5; `RogueScalpelProbe`). Because
the attack vector is not aligned with the cached clean manifold (it is an average of
weak, off-distribution jailbreak directions), most of its displacement lands in
`delta_off` and is clamped. The defense claim is therefore conditional on a geometric
fact — that attack displacement is *more* off-manifold than legitimate steering
displacement — which §3's separability test directly probes (and can falsify).

### 5.4 Composition (orthogonal axes, §9)

The budget is a HOW-FAR bound (axis A8 geometry); the M8 conformal gate is a WHEN
decision (axis A5 condition). Different intervention sites ⇒ they STACK: gate decides
whether to steer, budget bounds how far the steer may go off-manifold. H3 is run
budget-only first (clean marginal), then optionally budget∘gate as an additive-ladder
row (M8 × H3).

---

## 6. Predicted Delta

| Metric | Predicted | Rationale |
|---|---|---|
| Cliff-knee alpha, unbudgeted `relative_add` (1b) | ~0.05–0.10 | E3/S-10 knee (relative, ~10% ‖h‖) |
| Cliff-knee alpha, budgeted (1b), matched behavior | >= +1 grid-step right | clamp removes the off-manifold tax (N17 mechanism) |
| MMLU drop at the unbudgeted knee+1 step | > 2pp | past the cliff, unbudgeted |
| MMLU drop at the SAME alpha, budgeted | < 2pp | on-manifold push retained, off-manifold capped |
| offshell_displacement (budgeted) | <= B (by construction) | the clamp bounds it |
| Compliance under Rogue-Scalpel attack, unbudgeted | blows up (>> baseline) | F5 universal attack flips verdict |
| Compliance under attack, budgeted | within +0.05 of baseline | off-manifold attack push clamped |
| Behavior efficacy at matched alpha, budgeted vs unbudgeted | within epsilon | on-manifold component carries behavior (the §3 separability bet) |

Pre-registered; [NEEDS VERIFICATION]. All numbers are predictions, not results. Greedy
decoding ⇒ ~0 seed variance; replication unit is prompt-bootstrap + extraction-pair
resamples (PREREG Amendment 1c), n >= 7.

---

## 7. Experimental Protocol

### 7.0 HARD PRECONDITION (gate before any H3 defense run)

**The judge must be calibrated to AUC >= 0.85** against ground-truth comply/refuse
labels (PREREGISTRATION §6 + Amendment 1). The current Qwen safety judge is 9-10/12 dev
with a conservative bias — below bar. Until it clears, the DEFENSE leg (compliance under
attack) is SCREENING ONLY and cannot ground a citable compliance number. The COHERENCE
leg (cliff-knee shift, MMLU/PPL, offshell_displacement) is judge-INDEPENDENT — it is
measured by perplexity, the MCQ tripwire, and `geometry.offshell_displacement` — so it
may run and be reported first.

### 7.1 Primary experiment — the coherence leg (judge-independent)

- **Models:** gemma-3-1b-it (dev) → gemma-2-2b-it (standard).
- **Operation:** `relative_add` (baseline) vs `relative_add` + off-manifold clamp at
  budgets B in {inf, 0.20, 0.10, 0.05, 0.02} (inf = unbudgeted control).
- **Manifold:** PCA bank (k ≈ effective_rank) at the injection layer, cached once from
  clean WikiText-2 / AxBench-neutral activations; kNN bank as a robustness variant.
- **Sweep:** alpha grid spanning the E3 knee (relative ~0.02..0.40) × each B.
- **Metrics per cell:** behavior efficacy (generation scorer), PPL, MMLU-tiny,
  `offshell_displacement`, effective-rank drop, norm budget (all already in
  `eval`/`geometry`).
- **Primary readout:** the cliff-knee location (where coherence collapses
  super-linearly) as a function of B, at MATCHED behavior efficacy — is it pushed right?

### 7.2 Defense leg (judge-gated)

- **Attack:** `adversarial.RogueScalpelProbe` (the F5 20-vector universal steer) wired
  live (currently DRY-RUN) against the budgeted vs unbudgeted model at the injection
  layer; ASR scored by the calibrated Qwen judge (`evaluate_under_attack`).
- **Readout:** compliance_under_attack (budgeted) vs baseline and vs unbudgeted; the
  guard PASSES iff worst-case ASR <= NEUTRALIZED_THRESHOLD (Rung-4 gate).
- **Separability test (the §3 falsifier):** sweep B; for each B record BOTH
  (compliance-under-attack) AND (clean behavior efficacy at matched alpha). FALSIFIED if
  no B neutralizes the attack without also suppressing legitimate steering.

### 7.3 Controls

Matched-norm random direction, shuffled-label direction, and an orthogonalized direction
(the E7 control suite): the coherence cliff-shift must hold for the REAL steer; the
attack clamp must hold REGARDLESS of direction (it is geometric, not direction-specific).

### 7.4 Seeds / units / wall-clock

Greedy decoding ⇒ resample PROMPTS + extraction pairs, not decode seeds (PREREG
Amendment 1c); n >= 7. Wall-clock: minutes for the coherence leg (cached bank + cosine/
PPL, generation only for behavior); the defense leg adds the attack generation pass.

---

## 8. Cross-References

- **N17** (off-shell displacement predicts incoherence, rho=+0.585, rung-3) — the
  mechanism H3 converts from a diagnostic into a controller.
- **N5** (norm-budget law) — FALSIFIED as a universal numeric law across scale, but its
  monotone direction is what H3 exploits; B is calibrated PER MODEL (N5's lesson).
- **E3** (coherence cliff, knee ~0.05-0.10 relative) — the curve H3 predicts it shifts.
- **E7** (`relative_add`, knee ~10% ‖h‖) — the exact unbudgeted operation H3 clamps.
- **M8** (conformal gate) — composes orthogonally (WHEN × HOW-FAR); the M8 × H3 ladder row.
- **PROGRAM** [`../PROGRAM_conditional_displacement.md`](../PROGRAM_conditional_displacement.md)
  — H3 is the geometry leg of "Conditional displacement, not direction."
- **PREREGISTRATION.md** Amendment 1 (judge AUC precondition; prompt-bootstrap unit),
  **IDEA_TABLE.md** Block G row H3 (to be added on first run).

---

## 9. Committee Q&A

**Q: Isn't "cap the off-manifold part" just a fancy norm clamp — which the cliff already
implies you should do (use small alpha)?**
> No. A plain norm clamp on the whole `delta` caps BEHAVIOR and coherence together (it
> is just "use smaller alpha"). H3 caps ONLY the off-manifold residual and KEEPS the
> on-manifold component at full strength — the bet (testable, §3) is that the
> on-manifold part carries the behavior while the off-manifold part carries the tax. If
> that decomposition is real, H3 buys strength a scalar alpha cannot.

**Q: Why would the same cap defend against an attack?**
> Because the Rogue-Scalpel universal direction is an AVERAGE of weak, off-distribution
> jailbreak vectors — it is not aligned with the cached clean manifold, so most of its
> displacement is off-manifold and gets clamped. This is a geometric claim, and §3 makes
> it falsifiable: if attack displacement is no more off-manifold than legitimate steering,
> the defense leg dies (coherence leg may survive).

**Q: The cached bank could be stale / out-of-distribution for the eval prompts.**
> The bank is cached from neutral text at the SAME layer; we report a kNN variant
> (local, prompt-adaptive) alongside the global PCA variant precisely to bound this
> confound, and the matched-random control checks the clamp is not just shrinking
> everything uniformly.

**Q: N5 says the displacement law doesn't transfer across scale — doesn't that sink B?**
> N5 falsified a single UNIVERSAL numeric law; the MONOTONE direction survived (N17).
> H3 calibrates B PER MODEL (1b and 2b separately) — it never assumes one B transfers,
> which is N5's exact lesson applied.

---

## 10. Verification Checklist

- [ ] Judge calibrated to AUC >= 0.85 (PRECONDITION) before the DEFENSE leg, OR labeled SCREENING.
- [ ] Manifold bank cached ONCE from clean activations; k disclosed (≈ effective_rank).
- [ ] B = inf control reproduces plain `relative_add` exactly (Rung-0 identity check).
- [ ] Cliff-knee located for each B at MATCHED behavior efficacy (coherence leg).
- [ ] MMLU drop and PPL reported budgeted vs unbudgeted at matched alpha.
- [ ] offshell_displacement <= B verified empirically (the clamp actually binds).
- [ ] Rogue-Scalpel probe wired LIVE (not DRY-RUN); compliance vs baseline reported.
- [ ] Separability test (§3): a single B neutralizes attack WITHOUT suppressing steering, or FALSIFIED-defense.
- [ ] Controls: matched-random / shuffled-label / orthogonalized directions run.
- [ ] Per-model calibration of B (no cross-scale transfer assumed; N5 lesson).
- [ ] Bootstrap CI over prompt + extraction-pair resamples, n >= 7.
- [ ] IDEA_TABLE.md Block G row H3 added; PROVENANCE/H3.md created on first run.

---

## 11. Status Journal

- 2026-06-10 — Design doc created as **H3** of the program "Conditional displacement,
  not direction" (geometry leg). Status: **DESIGN + PLAN, RUN PENDING.** Converts the
  program's strongest result (N17, off-shell displacement predicts incoherence,
  rho=+0.585, rung-3) from a diagnostic into a controller: clamp the per-token
  OFF-manifold displacement to <= B*‖h‖ and test two independent legs — (coherence) does
  it shift the E3 cliff knee rightward at matched behavior, and (defense) does the same
  cap neutralize the Rogue-Scalpel 20-vector universal attack. Blocked on: (a) the
  budgeted operation (off-manifold clamp on `relative_add` against a cached PCA/kNN
  bank); (b) wiring `RogueScalpelProbe` LIVE (currently DRY-RUN) for the defense leg;
  (c) judge calibration to AUC >= 0.85 (the gating precondition for the compliance
  number; the coherence leg is judge-independent and may run first). No experiments yet.

---

## Pseudocode & Methodology

```python
# H3: displacement-budgeted relative_add. Geometric, direction-agnostic, no training.
import torch

def manifold_projector(bank_L: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k PCA basis of a cached clean-activation bank at layer L (cached ONCE).
    bank_L: [n_clean, dim]. Returns P_basis: [dim, k] (columns orthonormal)."""
    x = bank_L.float() - bank_L.float().mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    return Vh[:k].T                                   # on-manifold subspace

def budgeted_delta(h, v_hat, alpha, P_basis, B):
    """Clamp the OFF-manifold component of the relative-add displacement to B*||h||."""
    h_norm = h.norm(dim=-1, keepdim=True)             # [...,1]
    delta  = alpha * h_norm * v_hat                   # relative_add displacement
    delta_on  = (delta @ P_basis) @ P_basis.T         # project onto manifold
    delta_off = delta - delta_on                      # off-sphere residual (N17)
    off_norm  = delta_off.norm(dim=-1, keepdim=True)
    cap       = B * h_norm
    scale     = torch.clamp(cap / (off_norm + 1e-8), max=1.0)
    return delta_on + scale * delta_off               # budgeted edit

# COHERENCE leg: sweep (alpha x B); locate cliff knee vs B at matched behavior.
#   knee(B) measured via eval.perplexity / mcq_accuracy / geometry.offshell_displacement.
# DEFENSE leg : RogueScalpelProbe.run(steer_fn=budgeted_steer, judge_fn=qwen_asr).
#   compliance_under_attack(budgeted) vs baseline; PASS iff <= NEUTRALIZED_THRESHOLD.
# SEPARABILITY (the §3 falsifier): for each B record (compliance_attack, behavior@alpha).
```

Decision rule: TWO independent legs. COHERENCE = cliff-knee shift rightward at matched
behavior (judge-independent, primary). DEFENSE = compliance-under-attack within +0.05 of
baseline at a B that does NOT suppress legitimate steering (judge-gated, needs AUC>=0.85).
SUPPORTED-coherence / FALSIFIED-defense is an explicit partial verdict. See
[`../METHODOLOGY.md`](../METHODOLOGY.md).

---

## Provenance & Tracing

`DESIGN + PLAN — no experiments yet.` On first run, create `PROVENANCE/H3.md` with the
exp#, reproduce command, and artifact links. Reproduce (once the clamp + live probe +
judge precondition land):

```bash
# PYTHONPATH=src python scripts/run_displacement_budget.py --model models/google/gemma-3-1b-it \
#   --op relative_add --budget 0.05 --bank wikitext2_clean --attack rogue_scalpel \
#   --judge-model Qwen/Qwen2.5-3B-Instruct --no-log   # TO BE WRITTEN
```
