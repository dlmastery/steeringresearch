# AUDIT — `realignment` (Lesson 11): transplanting the refusal direction into an abliterated model

Independent paper-auditor pass. Scope: verify the lesson against
`extract_refusal.py`, `run_realignment.py`, README, and `artifacts/results.json`.
Primary citation: **Arditi et al. 2024, "Refusal in LLMs is Mediated by a Single
Direction" (arXiv:2406.11717)**.

## Paper verification (WebFetch, arxiv.org/abs/2406.11717)

**Real — confirmed.** Title exact: *"Refusal in Language Models Is Mediated by a
Single Direction."* Authors: Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee,
Nanda. Findings match the lesson's use precisely: refusal is a one-dimensional
subspace across 13 models; "erasing this direction ... prevents refusing harmful
instructions, while adding this direction elicits refusal on even harmless
instructions." The transplant method (read diff-of-means refusal direction on the
aligned model; add it via relative-add to restore refusal in the abliterated one)
is a faithful constructive use of the paper's bidirectional result.

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Paper real + attribution correct | **PASS (over-cautious tag)** | Paper verified real and correctly attributed; the diff-of-means formula `r = unit(mean_lasttok(harmful) − mean_lasttok(benign))` in `extract_refusal.py:11` and README §1 matches Arditi's method. The lesson marks it **`[UNVERIFIED]`** (`README.md:77,241,258`; `extract_refusal.py:9`) per corpus discipline. That tag is now *stale* — this audit verified it via WebFetch — but flagging conservatively is honest, not a defect. Recommend upgrading `[UNVERIFIED]`→verified. JailbreakBench (arXiv:2404.01318) also correctly cited. |
| 2 | Method fidelity (code vs claim) | **PASS** | Two-process design is faithful and well-motivated (one model per process, RAM constraint). Phase 1 loads only the aligned base, computes the normalized diff-of-means at layer 12, saves `refusal_dir.pt`. Phase 2 loads only the abliterated model, transplants via `generate(..., operation="relative_add")`, sweeps α. Both phases call `load_harmful_benign(N_PER_CLASS, SEED)` with the same seed so extract/eval splits are byte-identical without data passing — verified in code (`extract_refusal.py:65`, `run_realignment.py:175`). `choose_best_alpha` correctly requires over_refusal ≤ tol AND coherence ≥ floor AND α>0, returning `None` honestly when nothing qualifies. |
| 3 | Claim accuracy in Results table | **PASS** | README transcribes `results.json` correctly: ASR 0.375(α=0)→0.25(0.15)→0.25(0.25); coherence 0.929→0.801→0.564; over_refusal 0.25→0.50→0.125; `best=null`. Verdicts are appropriately hedged ("Directionally supported", "Not cleared"). |
| 4 | Results honesty vs artifacts | **PASS — GOOD honesty, one minor data-record note** | The headline negative is stated prominently and matches the artifact: `best=null`, "no clean operating point at this scale" — α=0.25 gets closest (over_refusal 0.125, coherence 0.564) but misses the over_refusal gate. This honest "coherence cliff beat the restoration" outcome is treated as a legitimate result, not a bug (README §5, `_summary_table` prints the NEGATIVE verdict explicitly). **Minor CONCERN:** the top-level `alphas` field in `results.json` lists 5 values `[0.0,0.1,0.15,0.2,0.25]` (the full config) but the `sweep` actually ran only 3 (0.0, 0.15, 0.25) via the `REALIGN_ALPHAS` RAM cap. The README correctly reports "α ∈ {0.0, 0.15, 0.25}", so the prose is honest, but the `alphas` key in the JSON overstates what was run — worth reconciling. **Flag:** phase-2 judge is the abliterated model self-grading (README §7 acknowledges this and points to `scripts/run_realign_abliterated.py` with an off-family Qwen judge); per the FLAS audit the 1B self-judge inflates refusal. Re-run with Qwen before any non-screening claim. |

## Overall verdict: **PASS**

The cited paper is real and the transplant method faithfully applies its
bidirectional single-direction result; the two-process design is sound; the
Results are transcribed accurately and the honest negative (`best=null`, no clean
operating point) is stated as a first-class outcome — good scientific honesty.
Minor fixes: upgrade the stale `[UNVERIFIED]` tag on arXiv:2406.11717, and
reconcile the `results.json` `alphas` field (5 listed, 3 run). Standing caveat:
abliterated-1B self-judge screening tier — re-run with the off-family Qwen judge.

Internal QA pass — independent external review pending (auditor shares a model family with the author).
