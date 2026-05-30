# PSR vs non-PSR Steering: Prompt Steering Replacement Explained
### Grounded in 'Steer Like the LLM: Activation Steering that Mimics Prompting' (Heyman & Vandeputte, ICML 2026, arXiv:2605.03907)

> One-line: PSR (Prompt Steering Replacement) is a class of activation-steering methods that imitate what PROMPTING actually does to activations - applying STRONG, TOKEN-SPECIFIC interventions - instead of the classic one-size-fits-all uniform vector. This closes the long-standing gap where activation steering underperforms prompting.

---

## 1. The problem PSR was built to solve

Two ways to steer an LLM at inference: prompting (put instructions in context) and activation steering (add a vector to hidden states). Empirically, **prompting usually beats activation steering** on the same goal (this is the AxBench verdict, 2501.17148). PSR asks: WHY, and can we make steering faithful to prompting?

The paper's diagnostic finding: popular activation steering methods are **not faithful to the mechanics of prompt steering**. A prompt does NOT push every token equally - it 'applies strong interventions on some tokens while barely affecting others.' Classic steering ignores this entirely.

---

## 2. non-PSR (classic) activation steering - the uniform write

non-PSR methods (ActAdd, CAA, RepE, ITI, refusal-direction, most SAE-feature steering) share one assumption: a SINGLE fixed vector v, added with a SINGLE scalar alpha, to ALL token positions, at a fixed layer.

    classic / non-PSR:   h_t  <-  h_t + alpha * v     for EVERY token t

    token:   t1   t2   t3   t4   t5   t6
    alpha:   a    a    a    a    a    a      <- same coefficient everywhere (flat)
             |    |    |    |    |    |
             +v   +v   +v   +v   +v   +v

Properties:
  - position-invariant (every token gets the same nudge)
  - single-step (one additive shift, no trajectory)
  - cheap, train-free, interpretable direction
  - BUT: over-steers tokens that didn't need it (coherence cost) and under-steers the few tokens that carry the behavior -> the gap vs prompting, and part of why over-steering trips the Rogue-Scalpel safety ridge.

---

## 3. PSR - the token-specific, prompt-imitating write

PSR keeps the direction idea but makes the COEFFICIENT a function of the token's own activation, trained to reproduce what a real prompt does:

    PSR:   h_t  <-  h_t + g(h_t) * v        g(.) = learned, token-specific gain

    token:   t1    t2    t3    t4    t5    t6
    g(h_t):  0.1   1.8   0.2   0.0   2.1   0.3   <- STRONG on some, near-zero on others
             |     ||    |           ||    |
             +     ++    +           ++    +

How PSR is built (per the paper):
  1. Run real PROMPT steering, record the per-token activation interventions it induces (the 'teacher').
  2. Train a SIMPLE, interpretable model (the PSR model) to ESTIMATE token-specific steering coefficients g(h_t) from the activations themselves.
  3. At inference, apply v with the learned per-token gains - distilling prompt behavior into a lightweight, promptless intervention.

---

## 4. PSR vs non-PSR - the differences that matter

    PROPERTY              non-PSR (classic)            PSR (Prompt Steering Replacement)
    coefficient           single global alpha          token-specific g(h_t)
    spatial profile       flat (all tokens equal)      sparse/peaky (mimics prompting)
    fidelity to prompting  low (the gap)               high (trained to imitate)
    training              none (diff-of-means)         lightweight distillation from prompt teacher
    coherence at fixed    worse (over-steers)          better (esp. high-coherence regime)
      behavior level
    interpretability      direction only               direction + a readable gain model
    cost                  cheapest                     small extra (the gain model)

Reported results (paper): PSR models outperform existing activation steering methods - ESPECIALLY when controlling for high-coherence completions - and compare favorably to PROMPTING on AxBench and persona steering, on Gemma-2-2B-IT and Gemma-2-9B-IT.

---

## 5. Where PSR sits in the broader taxonomy

PSR is best seen as a 'faithfulness upgrade' to additive steering, and it's a sibling of two other dynamic-coefficient ideas:
  - **SpotLight (2505.12025)** - dynamic, only-when-needed attention-score bias (dynamic, but on attention not residual).
  - **Adaptive Activation Steering / AUSteer (2602.04428)** - adaptive per-unit strengths.
  - **FLAS (2605.05892)** - learns a curved velocity field v_t(h,t,c); PSR keeps a fixed direction but learns the per-token GAIN, whereas FLAS learns the whole trajectory. PSR = scalar gain field; FLAS = vector flow field.

Composability: PSR is still an additive residual write, so on the stack/compete axis it behaves like CAA (stacks with disjoint-site methods like KV/attention/DoLa; competes with rotational edits on the same plane). Its token-sparsity is actually a SAFETY ADVANTAGE: by NOT over-steering the tokens that don't need it, PSR keeps more of the trajectory on-manifold, which (per the Rogue Scalpel diagnosis) is exactly what protects the fragile mid-layer refusal ridge.

---

## 6. Why an autoresearch harness should care
  - PSR is the current strongest evidence that the right control variable is the TOKEN-SPECIFIC coefficient, not the direction. (Connects to novel hypothesis N2: behavior = direction, where-to-act = a field; PSR learns that field.)
  - It gives a coherence-controlled comparison protocol (compare methods at MATCHED coherence, not matched alpha) - adopt this everywhere.
  - PSR's per-token gain is a natural place to BOLT ON the safety guard: clamp/zero g(h_t) on tokens whose steered state would flip the refusal verdict.

*Grounded in arXiv:2605.03907 (abstract read directly; Gemma-2-2B/9B usage stated in the paper). Internal-detail extrapolations marked by context; treat specific numbers as [NEEDS VERIFICATION] against the PDF.*
