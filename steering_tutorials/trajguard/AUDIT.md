# AUDIT — trajguard

**Auditor role:** independent paper verifier. Scope: does the cited paper exist,
does the code implement what the lesson claims, are the claims/results honest.
No git, no code/README edits were made.

## Paper existence (the critical check)

| field | finding |
|---|---|
| arXiv id | **2604.07727 — [UNVERIFIED]** (the lead will WebFetch `arxiv.org/abs/2604.07727` to confirm the id resolves, the title, and the author list). |
| claimed title | *TrajGuard: Training-free Streaming Detection of Jailbroken Generations via Decoding-time Hidden-state Trajectories* — as cited in `config.py` and `README.md`. [UNVERIFIED] |
| claimed authors | **Liu, Liu, Li, Xin, Ding** (per the lesson's citation). [UNVERIFIED] |
| claimed venue / date | ACL 2026 Findings; arXiv YYMM prefix `2604` = 2026-04, consistent with a 2026 ACL Findings paper. Plausible but **not yet confirmed**. |
| method in abstract | Claimed: a training-free, streaming, decoding-time defence that monitors a sliding window of per-token hidden states and flags a jailbroken generation before the harmful content is fully emitted. Consistency with the actual abstract is **pending the WebFetch**. |
| sibling citation | DeepContext (arXiv:2602.16935) is cited as a context-aware multi-turn sibling — also **[UNVERIFIED]**. |

## Findings

| check | verdict | evidence |
|---|---|---|
| Paper id verified | **PENDING** | id 2604.07727 marked `[UNVERIFIED]` in both `config.py` and `README.md`; verification deferred to the lead's WebFetch. The hedge is present and correct until then. |
| Citation format | **PASS** | Full author-list + year + venue + arXiv id given in `config.py` and `README.md`; `[UNVERIFIED]` tags applied per corpus discipline. |
| Inspired-by, not a reproduction | **PASS (disclosed)** | `README.md` §10 and the citation footer state plainly that the lesson is an **inspired-by reconstruction** — a sliding-window harm-projection detector plus **reused** `multiturn_jailbreak` sequence classifiers — **not** a faithful reimplementation of TrajGuard's exact architecture. The paper's precise scoring function / detector internals are not claimed to be reproduced. |
| Method fidelity — decoding-time trajectory | **PASS** | `trajectory.generate_and_capture` captures the layer-12 hidden state at each **generated** token position (one forward pass, `output_hidden_states=True`); `harm_direction` / `token_scores` / `sliding_window_risk` implement the training-free sliding-window projection; `ThresholdDetector.predict_proba_earlyK` provides the streaming early-detection readout. This operationalizes the paper's stated decoding-time-trajectory idea. |
| Reuse claim honest | **PASS** | The learned classifiers (`per_turn_max`, `trajectory_mlp`, `seq_gru`) are imported unchanged from `multiturn_jailbreak.models`; `README.md` §7 documents the turn-level vs. token-level sibling relationship rather than implying new modelling. |
| Label semantics disclosed | **PASS** | `README.md` §6 and §10 state plainly that the label is the **prompt's class**, not a judged harm rating of the completion, and that the abliterated model's compliance is both the enabling assumption and a confound. |
| Results honesty | **PASS** | The results section carries a **[PENDING GPU RUN]** banner; screening tier disclosed (1B model, one layer, one seed, few-hundred completions); a pre-registered falsifier (both `threshold_freeform` and `seq_gru` ≤ 0.60 AUC ⇒ thesis false) is stated before the run; `"judge": null` documented (detection, not generation). |
| Train/test hygiene | **PASS** | `harm_direction` and `tau` fit on train folds only; no completion graded on the direction it defined — stated in §6 and §10. |

## Overall verdict

**CONDITIONAL PASS — pending paper verification.** The lesson is an honestly
framed **inspired-by reconstruction** of TrajGuard's decoding-time-trajectory
defence (sliding-window harm projection + reused sequence classifiers), not a
reproduction of the paper's exact architecture, and it says so plainly. Method
fidelity to the stated idea, label semantics, train/test hygiene, and the
pending-run results framing all pass. The one open item is external:
**WebFetch arXiv:2604.07727** (and the DeepContext sibling 2602.16935) to confirm
the id resolves and the title/authors/venue match; drop the `[UNVERIFIED]` tags
only once confirmed, and only together with any author-name correction the fetch
surfaces.

*Internal QA pass — independent external review pending (auditor shares a model
family with the author).*
