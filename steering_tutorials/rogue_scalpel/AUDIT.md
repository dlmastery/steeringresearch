# AUDIT — `rogue_scalpel` (Lesson 10): steering compromises safety, and a layered guard

Independent paper-auditor pass. Scope: verify the lesson against `attack.py`,
`guard.py`, README, and `artifacts/results.json`. Primary citation:
**"The Rogue Scalpel: Activation Steering Compromises LLM Safety"
(arXiv:2509.22067)**. Secondary: Arditi refusal-direction (arXiv:2406.11717),
Rimsky CAA (arXiv:2312.06681).

## Paper verification (WebFetch, arxiv.org/abs/2509.22067)

**Real — confirmed.** Title exact: *"The Rogue Scalpel: Activation Steering
Compromises LLM Safety."* Authors: Korznikov, Galichin, Dontsov, Rogov,
Oseledets, Tutubalina. Abstract claim matches the lesson's use: "steering
systematically breaks model alignment safeguards, making it comply with harmful
requests"; random directions boost harmful compliance 0→1–13%; **combining 20
randomly sampled jailbreaking vectors creates a transferable attack** — exactly
the "20-vector universal attack" the README names. arXiv:2406.11717 also verified
(Arditi et al., *Refusal ... Single Direction*; "erasing this direction ...
prevents refusal, adding ... elicits refusal").

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Paper real + attribution correct | **CONCERN (year only)** | The paper is real and its claim is faithfully represented (see above). BUT the lesson cites it as **"Korznikov et al. 2026"** (`README.md:299`, `guard.py:32`). arXiv ID `2509.*` = submitted **September 2025**, so the year should be **2025**, not 2026. Content attribution is otherwise correct: the README is explicit that the paper's *20-vector* universal attack is "far stronger than the single-direction attack shown here," so it does not overclaim that its toy reproduces the paper. Fix the year. |
| 2 | Method fidelity (code vs claim) | **PASS** | The attack is a faithful *pedagogical* version, honestly distinguished from the paper's universal attack. `suppress_refusal_delta` implements both `project_out` (`h − (h·û)û`, Arditi directional ablation — correctly attributed to 2406.11717) and `negative_add` (`h − α‖h‖û`, lesson-2 relative-add sign-flipped). Self-test asserts project-out zeroes the refusal component and negative-add is anti-parallel at the right magnitude. Guards implement CLAUDE.md's A/B/D: `norm_clamp` bounds ‖Δh‖≤budget·‖h‖, `projection_lock` re-adds refusal to a floor, `dual_forward_verdict` is the second-forward judge. Hooks restore the model exactly on exit (self-tested). |
| 3 | Claim accuracy in Results table | **PASS** | README §7 transcribes `results.json` faithfully: refusal 0.95(baseline)→0.45(attacked)→0.95(+clamp)→0.35(+lock)→0.35(+dual); gibberish 0.05→0.55→0.05→0.65→0.65; ASR 0.0 at every rung; benign collateral 0.25→0.15. The headline is correctly reframed: because ASR is pinned at 0.0 (attack yields *gibberish*, not compliance, on an aligned 1B), the readable axis is refusal rate, not ASR — stated plainly. |
| 4 | Results honesty vs artifacts | **PASS — GOOD honesty** | Two strong honest negatives, both matching `results.json` and NOT smoothed: (a) the attack "breaks the model rather than jailbreaking it" (refusal→gibberish, ASR stays 0.0); (b) the +lock and +dual rungs are *worse* than +clamp alone (refusal 0.35 vs 0.95) — the README states "not every guard rung earns its place ... the lock over-writes and degrades," the opposite of the additive-ladder's hope. **Flag:** aligned 1B self-graded, Guard D uses the same Gemma as judge (README caveat acknowledges this); per the FLAS audit the 1B self-judge inflates refusal. Re-run with the off-family Qwen judge before any non-screening claim. The README also correctly warns `+dual ≈ baseline` must not be read as "solved" given the real 20-vector attack. |

## Overall verdict: **PASS** (one minor citation-year fix)

The cited paper is real and faithfully represented; the attack and guards are
correct and honestly framed as a 1B toy distinct from the paper's stronger
universal attack; the Results are transcribed accurately and the two honest
negatives (attack→gibberish; guards A/D degrade rather than help) are stated
prominently. Only defect: the citation year "2026" should be "2025"
(arXiv:2509.22067 = Sept 2025). Standing caveat: 1B self-judge screening tier —
re-run with the off-family Qwen judge.

Internal QA pass — independent external review pending (auditor shares a model family with the author).
