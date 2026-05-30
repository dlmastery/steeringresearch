# Steering, From First Principles — An ML Engineer's Walkthrough
### Part 1: the geometry (ASCII) | Part 1.5: the SEVEN axes of intervention (incl. PSR) | Part 2: Rogue Scalpel + principled mitigation

> Audience: an ML engineer who knows transformers and linear algebra but hasn't internalized *why* steering works and *why it quietly breaks alignment*. Everything below is mechanism-first. The mitigation in Part 2 is grounded in **The Rogue Scalpel: Activation Steering Compromises LLM Safety** (Korznikov et al., arXiv:2509.22067, ICML), read in full including its Appendix E.

==============================================================

# PART 1 — FIRST-PRINCIPLES FINDINGS (step by step)

## Step 0 — The one object that matters: the residual stream

A decoder transformer is, mechanically, a sequence of additive writes into a shared vector h (the residual stream). Every block READS h, computes something, and ADDS its output back:

    h_{l+1} = h_l + Attn_l(h_l) + MLP_l(h_l)

    token --> [embed] --> h0
               |
        +------v------+   reads h, writes delta
        |  Block 0    |---+
        +-------------+   |  h1 = h0 + delta0
        +------v------+   |
        |  Block 1    |---+  h2 = h1 + delta1
        +-------------+   |
             ...          v
        +-------------+
        |  Block L-1  |---> h_final --> [unembed] --> logits
        +-------------+

KEY INSIGHT 1: because every contribution is ADDED, the residual stream is a *linear highway*. If a behavior is written as a direction v somewhere on this highway, I can imitate that write by simply adding alpha*v myself. That is literally all activation steering is:

    h  <-  h + alpha * v

This is why steering needs no training and no weights — it borrows the model's own additive bus.

---

## Step 1 — Concepts live as DIRECTIONS, not neurons (the Linear Representation Hypothesis)

Empirically, high-level concepts (truthfulness, refusal, sentiment, a language, a persona) are encoded as approximately linear directions in h. 'How much refusal' is roughly a dot product:

    refusal_amount(h)  ~  <h, v_refusal>

    activation space (cartoon, 2 of ~2304 dims for Gemma-2-2B)

        ^ v_refusal
        |        . safe prompts cluster here (high projection)
        |     .  .
        |   .  .
        | .  .            . harmful-but-complied (low projection)
        +-------------------------> v_topic

KEY INSIGHT 2: to extract v for a behavior, take the DIFFERENCE OF MEANS between activations on contrastive examples (e.g. mean(harmful) - mean(harmless)). The noise cancels; the shared concept direction survives. That difference vector IS the steering vector.

---

## Step 2 — Why there is a COHERENCE CLIFF (the manifold)

Real activations don't fill the whole space. They lie on a thin, curved MANIFOLD (a crumpled sheet) inside the huge ambient space. The model only behaves coherently for h *on* that sheet.

    ambient space (everything)
    ........................................
    ..   _____the data manifold_____      ..
    ..  /  (where h is coherent)   \      ..
    .. |   * h        --alpha-->  X  |     ..   X = pushed OFF the sheet
    ..  \__________________________/      ..       => gibberish / collapse
    ........................................

A small alpha slides h ALONG the sheet (behavior changes, text stays fluent). A large alpha launches h OFF the sheet into a region the downstream layers never trained on -> perplexity explodes, MMLU craters.

KEY INSIGHT 3: there is a behavior-specific alpha 'cliff'. Below it: clean control. Above it: incoherence. The cliff exists because the manifold CURVES away from the straight steering line. (This is the first-principles reason rotation-based methods preserve coherence on small models: a rotation keeps ||h|| fixed and hugs the sphere instead of shooting off it.)

---

## Step 3 — Why some methods STACK and others COMPETE (it's just linear algebra)

Two steering vectors added together:

    h <- h + a1*v1 + a2*v2

Decompose v2 into the part along v1 and the part orthogonal to it:

    v2 = (v2.v1) v1   +   v2_perp
          \_______/       \_____/
         INTERFERENCE      CLEAN, independent behavior

    v1 ^
       |      v2
       |     /
       |    /  } overlap (v2.v1) = they fight for the same axis (COMPETE)
       |   /
       +--/-----> v2_perp  (orthogonal part = free to stack)

KEY INSIGHT 4: stackability is governed by the GRAM MATRIX of the active vectors. Off-diagonal mass = interference. Near-orthogonal vectors (|cos|~0) stack cleanly; aligned/anti-aligned vectors compete. AND the SUM must still respect the manifold budget from Step 2.

    Decision rule, derived not memorized:
      same axis  + same op (add+add)      -> COMPETE for that axis
      same axis  + diff op (add vs rotate)-> COMPETE (double-count, off-manifold)
      orthogonal axes                      -> STACK (until norm budget spent)
      disjoint SITES (resid vs KV vs logits)-> STACK (different writes entirely)

---

## Step 4 — Conditioning is a META-layer, not a peer method

CAST-style conditional steering separates DETECTING (should I act?) from ACTING (the behavior write). The detector is a read-only probe; the actuator is the additive write:

    h_early --> <h, v_condition> > theta ? --YES--> apply alpha*v_behavior at h_late
                     (READ / gate)            --NO--> leave h untouched

    prompt
      |
      v
   [layer k]  read:  s = cos(h, v_cond)   (no modification)
      |                       |
      |                 s>theta?  --no--> normal answer
      v                       yes
   [layer m]  write: h += alpha * v_behavior  --> gated behavior

KEY INSIGHT 5: because the gate is a READ on a DIFFERENT axis at a DIFFERENT layer than the WRITE, conditioning is orthogonal to almost every behavior method -> it stacks on top of all of them. This is also the seed of the safety fix in Part 2.

==============================================================

# PART 1.5 — THE AXES OF INTERVENTION (now including PSR)

Earlier we said steering = add alpha*v to the residual stream. But that hides a richer truth: a steering METHOD is a choice along SEVERAL INDEPENDENT AXES. PSR ('Steer Like the LLM', arXiv:2605.03907) forces us to make one of these axes explicit — the COEFFICIENT axis — because PSR's whole contribution is to stop using a single global coefficient and instead use a token-specific one.

## How many axes do we have now? SEVEN.

A steering intervention is fully specified by answering seven orthogonal questions. Any method = one setting per axis.

    AXIS 1  WHERE (site)        which part of the forward pass do we touch?
    AXIS 2  WHAT (direction)    which vector / subspace do we use?
    AXIS 3  HOW MUCH (coeff)    how strong is the edit?            <-- PSR lives here
    AXIS 4  HOW (operation)     add? rotate? project? clamp? flow?
    AXIS 5  WHEN (condition)    always, or gated on input?
    AXIS 6  WHICH TOKENS (span) all positions, or selected ones?   <-- PSR also lives here
    AXIS 7  HOW DERIVED (source) diff-of-means? SAE? hypernet? learned?

### AXIS 1 — WHERE (intervention SITE)
  decoding/logits (DoLa) | residual stream (CAA/ActAdd) | attention Q/V (DISCO) |
  attention scores (PASTA/SpotLight) | KV cache (KV-steering/MAGS) | weights low-rank (ReFT)
  ASCII:
    embed -> [resid] -> [attn Q/K/V] -> [attn scores] -> [resid] -> ... -> [logits]
               ^             ^              ^                                  ^
             CAA          DISCO          PASTA                               DoLa

### AXIS 2 — WHAT (the direction / subspace)
  single direction (refusal) | low-rank subspace (ReFT) | SAE feature | conceptor ellipsoid | persona vector

### AXIS 3 — HOW MUCH (the COEFFICIENT) — the PSR axis, part 1
  This is the axis PSR redefines. Three settings seen in the literature:
    (a) GLOBAL scalar alpha        — classic (ActAdd/CAA/ITI). One number for the whole run.
    (b) ADAPTIVE-per-unit          — AUSteer/Adaptive Activation Steering. Different strength per atomic unit.
    (c) TOKEN-SPECIFIC g(h_t)      — PSR. The coefficient is a LEARNED FUNCTION of each token's activation,
                                     trained to imitate what real prompting does.
  ASCII (the axis-3 spectrum):
    global a:      a a a a a a    (flat)            <- non-PSR
    adaptive:      a b a c b a    (per-unit)        
    PSR g(h_t):    .1 1.8 .2 0 2.1 .3 (per-token, prompt-faithful, peaky)  <- PSR

### AXIS 4 — HOW (the OPERATION)
  add (CAA) | rotate (Angular/Spherical/Selective) | project-out (refusal ablation, guard) |
  clamp-to-manifold (norm budget) | flow/transport (FLAS multi-step)
  Note: AXIS 4 is WHY additive vs rotational COMPETE on the same plane — same site (1), same direction (2),
  but incompatible operation (4).

### AXIS 5 — WHEN (the CONDITION / gate)
  unconditional (always on) | conditional (CAST: fire only if <h,v_cond> > theta) |
  energy-ratio gate (FineSteer-SCS) | discriminative-layer gate (Selective)
  This is the META-axis: a read that decides whether axes 1-4 even apply.

### AXIS 6 — WHICH TOKENS (the SPAN) — the PSR axis, part 2
  all tokens (classic) | prompt-only vs generation-only (some refusal work) |
  TOKEN-SELECTIVE (PSR: strong on the few tokens that carry the behavior, ~0 elsewhere)
  PSR couples AXIS 3 and AXIS 6: its token-specific gain g(h_t) IS a soft token-selection. Prompting
  naturally hits some tokens hard and others not at all; PSR reproduces that spatial profile, which is
  why it beats flat steering at matched coherence.

### AXIS 7 — HOW DERIVED (the SOURCE of the vector)
  difference-of-means | PCA | SAE decoder column | hypernetwork (HyperSteer: vector generated from a
  natural-language prompt) | learned/distilled (PSR, FLAS, BiPO) | ICL/task-vector (function vectors)

## Putting PSR on the map (one row per axis)

    AXIS               classic CAA            PSR (Prompt Steering Replacement)
    1 site             residual stream        residual stream (same)
    2 direction        diff-of-means v        a direction v (same idea)
    3 coefficient      GLOBAL alpha           TOKEN-SPECIFIC g(h_t)   <== the change
    4 operation        add                    add (same)
    5 condition        usually none           composable with a gate
    6 span             ALL tokens             TOKEN-SELECTIVE         <== the change
    7 source           diff-of-means          DISTILLED from prompting

So PSR is not a new SITE or a new OPERATION — it is a principled move along axes 3 and 6: 'stop steering every token equally; steer the tokens prompting would have steered, by the amount prompting would have used.'

## Why the 7-axis view matters for first principles
  - Most 'new methods' in the arXiv firehose are just a NEW POINT in this 7-axis space, not a new dimension. Recognizing the axes prevents re-discovering the same method under a new name.
  - STACKABILITY is now precise: two methods stack iff they differ on AXIS 1 (site) OR are orthogonal on AXIS 2 (direction); they compete iff same site + same direction + different AXIS 4 operation, or if they jointly blow the norm budget.
  - The SAFETY guard from Part 2 is itself just settings: AXIS 4 = project-out + clamp, AXIS 5 = gate, AXIS 6 = veto specific tokens. PSR's token-selectivity (axis 6) is a free safety ally: fewer over-steered tokens = less off-manifold displacement = less Rogue-Scalpel risk.
  - Novel hypothesis link: N2 ('behavior = direction, where-to-act = a field') is exactly the claim that AXIS 2 and AXIS 3/6 are separable — PSR is the first method to learn the AXIS-3/6 field explicitly. The capstone N12 unifies ALL SEVEN axes into one operator: h <- h + gate(h) * g_t(h) * Proj_tangent( Op( v ) ), capped at budget B.

## TL;DR — how many axes of intervention now?
SEVEN: (1) site, (2) direction, (3) coefficient, (4) operation, (5) condition, (6) token-span, (7) source. PSR's contribution is to make axes 3 (coefficient) and 6 (token-span) first-class, learned, and faithful to prompting — closing the steering-vs-prompting gap without changing the site or the basic additive operation.

==============================================================

# PART 2 — THE ROGUE SCALPEL PROBLEM, AND HOW TO PREVENT IT

## 2.1 What the paper actually shows (read in full, incl. Appendix E)

Paper: The Rogue Scalpel: Activation Steering Compromises LLM Safety (Korznikov, Galichin, Dontsov, Rogov, Oseledets, Tutubalina; arXiv:2509.22067, v2 Feb 2026).
Setup: JailbreakBench (100 harmful prompts, 10 categories), LLM-as-judge (Qwen3-8B, 94% precision on the harmful class), models Llama3.1-8B/70B, Qwen2.5-7B/32B, Falcon3-7B. Baseline compliance WITHOUT steering = 0%.

Findings (their numbers):
  F1. Steering in a RANDOM direction raises harmful compliance from 0% to 1-13%. Just noise on the bus breaks refusal.
  F2. Steering BENIGN SAE features is as dangerous or worse: +1-4% over random. 817/1000 benign features jailbreak >=1 prompt; the strongest 'master key' feature was the concept 'modal verbs' / 'brand identity'.
  F3. Effect peaks in EARLY-MIDDLE layers (e.g. Llama layer 16), not late layers.
  F4. Poor cross-prompt generalization -> you CANNOT pre-screen 'dangerous features'; monitoring is infeasible by enumeration.
  F5. WEAPONIZATION: average just 20 random vectors that each jailbreak ONE prompt -> a UNIVERSAL attack vector, ~4x compliance on unseen prompts (Falcon3-7B 5.7% -> 63.4%). Zero gradients, zero weights, zero harmful training data.
  F6 (Appendix E, the crucial one): harmful features have NEAR-ZERO cosine with the Arditi refusal direction (mean 0.027 +/- 0.021). So steering does NOT work by cancelling the refusal vector.

## 2.2 First-principles diagnosis: WHY does benign steering break safety?

Tie F6 back to Part 1. If breaking safety were just 'subtract the refusal direction', harmful features would align with v_refusal. They DON'T. So the mechanism is NOT linear suppression of one axis. Instead:

  Step 2 said: refusal is the model RE-ENTERING a safe region of the manifold as abstract 'this is harmful' features assemble in the early-middle layers (F3). That assembly is NON-LINEAR and FRAGILE.

    normal harmful prompt:
        h --build 'harmful' concept--> [refusal circuit fires] --> REFUSE

    with ANY sizeable steering nudge at the fragile mid-layer:
        h + alpha*v --perturbed--> [refusal circuit fails to assemble] --> COMPLY

    manifold view:
        SAFE basin            CLIFF            COMPLY basin
        \__ refuse __/      __||__         \__ comply __/
            * h  --any nudge over the ridge-->  * h'

KEY DIAGNOSIS: the refusal mechanism sits on a NARROW RIDGE in mid-layers. It is a high-curvature, low-margin region (Part 1, Step 2/3). ALMOST ANY off-manifold perturbation of sufficient size — random, benign, whatever — knocks the trajectory off the ridge into the comply basin. The direction barely matters; the DISPLACEMENT MAGNITUDE at the FRAGILE LAYER does. That is why random == benign-SAE in damage, and why direction-based screening (F4) is hopeless.

Restated as a principle:
  >> Steering damages alignment as a SIDE EFFECT of moving h off the data manifold near the refusal ridge, NOT by aligning with an attack direction. <<

## 2.3 Why existing 'fixes' are insufficient

  - Adversarial training (the paper's own suggestion): retrains weights, expensive, and chases an infinite direction set (F4) — you can't enumerate benign-looking attack vectors.
  - Direction blacklists / feature screening: dead on arrival because of F4/F6 (no stable dangerous-direction set; benign concepts are the attackers).
  - Just use small alpha: the 'sweet spot' (their Fig 3, alpha<=0.75) still yields 4-5% compliance. Small alpha reduces but does not remove the leak, and the UNIVERSAL attack (F5) is built precisely from small, individually-weak vectors.

## 2.4 The mitigation: a MANIFOLD-CONSTRAINED, REFUSAL-AWARE STEERING GUARD

The diagnosis dictates the cure. If damage = (off-manifold displacement) x (at the fragile refusal layer), then we must (a) keep edits ON the manifold, (b) protect the refusal ridge specifically, and (c) verify the safety verdict didn't move. Five composable layers, cheap enough for a 4090, none requiring weight updates:

GUARD LAYER A — Refusal-subspace PROJECTION LOCK (protect the ridge).
  Compute the safety subspace S once (the refusal direction v_refusal + a few PCs of the harmful-vs-harmless contrast, the very thing benign steering perturbs). Before applying ANY steering vector v, project OUT its component inside S:
      v_safe = v - P_S v        (P_S = projector onto safety subspace)
  Now steering can change topic/style/persona but is forbidden from writing into the dimensions that carry the harmfulness verdict.
      v ----remove S-component----> v_safe   (behavior kept, safety axis frozen)
  NOTE: F6 says attacks are near-ORTHOGONAL to v_refusal, so projecting out only v_refusal is not enough — S must be the LOCAL refusal-FORMATION subspace at the fragile mid-layers (estimated from how the harmful concept assembles), not just the single late refusal direction. This is the key upgrade over naive 'orthogonalize to refusal'.

GUARD LAYER B — Manifold / NORM-BUDGET clamp (don't leave the sheet).
  Estimate the local activation shell at the injection layer (mean activation norm mu(l), and a kNN/PCA patch of natural h). Cap the edit so the steered state stays in-distribution:
      if ||alpha*v_safe|| > beta*mu(l):  rescale to beta*mu(l)        (beta ~ 0.5-0.75)
      optionally: h_steered <- project_to_tangent(h + alpha*v_safe)   (slide along sheet, not off it)
  This directly removes the 'displacement magnitude' factor that the diagnosis blamed.

GUARD LAYER C — Avoid the FRAGILE layers (move the write).
  The paper: damage peaks in early-middle layers (F3). Legitimate style/topic steering usually works at later layers too. POLICY: forbid (or heavily down-weight) steering at the empirically-fragile band; prefer the latest layer that still achieves the intended behavior. Pick the injection layer by max behavior-effect-per-unit-safety-damage, not max effect.

GUARD LAYER D — DUAL-FORWARD safety verdict check (verify, don't trust).
  Run a tiny read-only refusal probe r(h) = <h, v_refusal_late> on BOTH the unsteered and steered states. If steering pushed the safety verdict across the refuse->comply boundary on a prompt the model WOULD have refused, ABORT the steer for that prompt (fall back to unsteered).
      s_clean = probe(h_unsteered);  s_steer = probe(h_steered)
      if s_clean = REFUSE and s_steer = COMPLY:  reject edit, emit clean refusal
  This is O(one extra probe), catches the residual leak that A-C miss, and—crucially—defeats the UNIVERSAL attack (F5): the attack's whole purpose is to flip this verdict, so we watch the verdict itself.

GUARD LAYER E — CONDITIONAL gating wraps all of the above (spend risk only when needed).
  Using Part 1 Step 4: only ALLOW behavior steering to fire when a condition probe says the input is in the intended benign domain; for inputs that look harmful, the gate withholds steering entirely so there is no perturbation to knock the refusal ridge over.
      if input looks harmful (condition probe):  DO NOT STEER (let native refusal run)
      else:                                      apply guarded steer (A+B+C), then verify (D)

## 2.5 The guarded operator (one expression)

  Unsteered behavior write becomes:

      v_safe   = (I - P_S) v                         # A: freeze safety axes
      delta    = clamp_norm( alpha * v_safe, beta*mu(l) )   # B: stay on manifold
      delta    = Proj_tangent(h, delta)              # B: slide along sheet
      apply at layer l in the NON-fragile band       # C
      h'       = h + delta   only if gate_benign(h)  # E
      commit h' only if probe(h') keeps the refuse/comply verdict of h   # D, else rollback

## 2.6 Why this prevents the Rogue Scalpel (point-by-point)

  F1 random nudge breaks refusal      -> B caps displacement + D vetoes any verdict flip => random noise can't cross the ridge.
  F2 benign SAE features jailbreak     -> A removes their safety-subspace component; D catches the rest. Benign-looking == irrelevant; we gate on the VERDICT, not the concept label.
  F3 mid-layer fragility               -> C refuses to write into the fragile band.
  F4 can't screen directions           -> we DON'T screen directions (hopeless); we constrain GEOMETRY (A,B) and check OUTCOME (D). Direction-agnostic by design.
  F5 universal 20-vector attack        -> built to flip the safety verdict; D measures exactly that verdict and rolls back. E also blocks it on harmful-looking inputs.
  F6 attacks orthogonal to v_refusal   -> this is WHY A uses the local refusal-FORMATION subspace S (mid-layer), not just the single late refusal vector, and why D is needed as the outcome backstop.

## 2.7 How to validate the mitigation (drop-in experiments for the harness)

  V1. Reproduce the leak: random + benign-SAE steering on Gemma-2-2B, measure Compliance Rate on JailbreakBench (expect non-zero, like the paper).
  V2. Turn on Guard A+B+C+D+E one at a time; report Compliance Rate after each (ablation). Target: drive CR back to ~baseline (0%) while keeping the INTENDED benign behavior >=90% of unguarded efficacy and MMLU drop <2pt.
  V3. Re-run the universal-attack construction (average 20 jailbreaking vectors) AGAINST the guarded model; show D's verdict-rollback neutralizes it.
  V4. Stress A: confirm projecting out only the LATE refusal direction is insufficient (consistent with F6), but projecting the MID-LAYER formation subspace S is. This isolates the paper's core mechanism.
  V5. Cost: measure the latency of the dual-forward probe (D); it should be a single extra read, not a full second generation.

==============================================================

## TL;DR for the engineer

Steering works because the residual stream is an additive bus and concepts are linear directions on a curved manifold. It STACKS when edits are orthogonal and on-manifold, COMPETES when they share an axis or blow the norm budget. It DAMAGES ALIGNMENT not by finding an 'attack direction' (attacks are orthogonal to refusal — Rogue Scalpel Appendix E) but by knocking the fragile mid-layer refusal ridge OFF the manifold; random and benign edits do this equally, and 20 weak ones average into a universal jailbreak. The fix is therefore GEOMETRIC and OUTCOME-BASED, not direction-screening: freeze the safety subspace (A), clamp to the manifold (B), avoid the fragile layers (C), verify the refusal verdict survived with a cheap dual-forward probe (D), and only steer benign-looking inputs in the first place (E).

*Grounded in arXiv:2509.22067 (read in full, incl. Appendix E). All quantitative targets are pre-registered predictions for the harness, not established results.*
