# AUDIT — trajguard

**Auditor role:** independent paper verifier. Scope: does the cited paper exist,
does the code implement what the lesson claims, are the claims/results honest.
No git, no code/README edits were made.

## Paper existence (the critical check)

| field | finding |
|---|---|
| arXiv id | **2604.07727 — VERIFIED** (WebFetch of `arxiv.org/abs/2604.07727` resolves). |
| actual title | *TrajGuard: Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense* — the lesson's earlier cited title was slightly off and has been corrected in `README.md`/`config.py` to this verified form. |
| actual authors | **Cheng Liu, Xiaolei Liu, Xingyu Li, Bangzhou Xin, Kangyi Ding.** |
| venue / date | ACL 2026 Findings; submitted 2026-04-09 (arXiv YYMM `2604`). Confirmed. |
| method in abstract | Confirmed: a training-free, streaming, decoding-time defence that aggregates hidden-state trajectories via a sliding window to quantify risk in real time; jailbreak-attempted tokens progressively shift toward high-risk latent regions — matches the lesson's framing. |
| sibling citation | DeepContext (arXiv:2602.16935, Albrethsen et al.) — **VERIFIED** (checked in the `multiturn_jailbreak` audit). |

## Findings

| check | verdict | evidence |
|---|---|---|
| Paper id verified | **PASS** | 2604.07727 WebFetch-resolves; title corrected to the verified form ("...Streaming Hidden-state Trajectory Detection for Decoding-time Jailbreak Defense"), full author list (Cheng Liu, Xiaolei Liu, Xingyu Li, Bangzhou Xin, Kangyi Ding), ACL 2026 Findings, 2026-04-09. `[UNVERIFIED]` tags dropped. |
| Citation format | **PASS** | Full author-list + year + venue + verified arXiv id in `config.py` and `README.md`. |
| Inspired-by, not a reproduction | **PASS (disclosed)** | `README.md` §10 and the citation footer state plainly that the lesson is an **inspired-by reconstruction** — a sliding-window harm-projection detector plus **reused** `multiturn_jailbreak` sequence classifiers — **not** a faithful reimplementation of TrajGuard's exact architecture. The paper's precise scoring function / detector internals are not claimed to be reproduced. |
| Method fidelity — decoding-time trajectory | **PASS** | `trajectory.generate_and_capture` captures the layer-12 hidden state at each **generated** token position (one forward pass, `output_hidden_states=True`); `harm_direction` / `token_scores` / `sliding_window_risk` implement the training-free sliding-window projection; `ThresholdDetector.predict_proba_earlyK` provides the streaming early-detection readout. This operationalizes the paper's stated decoding-time-trajectory idea. |
| Reuse claim honest | **PASS** | The learned classifiers (`per_turn_max`, `trajectory_mlp`, `seq_gru`) are imported unchanged from `multiturn_jailbreak.models`; `README.md` §7 documents the turn-level vs. token-level sibling relationship rather than implying new modelling. |
| Label semantics disclosed | **PASS** | `README.md` §6 and §10 state plainly that the label is the **prompt's class**, not a judged harm rating of the completion, and that the abliterated model's compliance is both the enabling assumption and a confound. |
| Results honesty | **PASS** | Results now report the first run (n=80/class, 32 tokens, 5-fold): the falsifier is NOT triggered (all methods clear chance; learned models 0.92-0.98). Crucially the README reports the *complication* honestly — `per_turn_max` (0.977) beats the sequence models and the paper's training-free projection is the WEAKEST (0.665), because the abliterated model complies immediately so individual tokens are already separable; the cross-lesson caveat (trajectory matters when chunks look benign, less under active generation) and the immediate-compliance confound are both named. `"judge": null` (detection). |
| Train/test hygiene | **PASS** | `harm_direction` and `tau` fit on train folds only; no completion graded on the direction it defined — stated in §6 and §10. |

## Overall verdict

**PASS.** arXiv:2604.07727 is verified (id resolves; title corrected to the
verified form; authors Cheng Liu, Xiaolei Liu, Xingyu Li, Bangzhou Xin, Kangyi
Ding; ACL 2026 Findings), and the DeepContext sibling 2602.16935 is verified. The
lesson is an honestly framed **inspired-by reconstruction** of TrajGuard's
decoding-time-trajectory defence (sliding-window harm projection + reused
`multiturn_jailbreak` sequence classifiers), not a reproduction of the paper's
exact architecture, and says so plainly. Method fidelity to the stated idea, label
semantics, train/test hygiene, and — now that the run has landed — results honesty
all pass, including the honest complication that on this immediate-compliance setup
the per-token baseline beats the trajectory models and the paper's training-free
projection is the weakest.

*Internal QA pass — independent external review pending (auditor shares a model
family with the author).*
