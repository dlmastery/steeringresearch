# AUDIT — `multi_intent` (Lesson 9): compositional steering, orthogonalization + norm budget

Independent paper-auditor pass. Scope: verify the lesson's claims against its own
code (`multi_intent.py`, `run_multi_intent.py`), README, and
`artifacts/results.json`. No external single paper is claimed for the core idea —
the claim is the project's OWN (Gram-Schmidt orthogonalization cuts interference;
the N5 norm budget bounds stacking). This audit checks that framing is honest.

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Attribution: internal claim honestly framed (not attributed to a paper that doesn't say it) | **PASS** | README §5 titles the block "The hypothesis we TEST, not assume." The norm budget is labelled "this project's own leading-indicator (see CLAUDE.md §3)" in the module docstring (`multi_intent.py:36`). The only external citations — Rimsky CAA (arXiv:2312.06681, diff-of-means per concept) and Arditi (arXiv:2406.11717, concept = one linear direction) — are used for the *component mechanics*, not for the interference/budget claim. Both arXiv IDs are real and correctly characterized. Gram-Schmidt is standard linear algebra, attributed to no paper. No nonexistent citation is implied. |
| 2 | Method fidelity (code vs claim) | **PASS** | `gram_schmidt()` is a correct *modified* Gram-Schmidt (subtracts against running `w`, `multi_intent.py:138-149`), emits a zero axis on degeneracy rather than NaN. `norm_budget()` returns `sqrt(Σαᵢ²)`, the orthonormal-case quadrature (`:175-194`), matching the README formula. `_combined_vector` folds α-weighted unit directions for one `relative_add` hook. The CPU self-test asserts orthonormality (Gram off-diag < 1e-5), unit norm, degenerate→0, budget = √(Σα²), and per-axis alpha recovery from the mixture — all the load-bearing math is checked. Shared benign baseline gives the K directions a common origin so cosine measures concept overlap (`:82-84`), as claimed. |
| 3 | Claim accuracy in Results table | **PASS** | The README Results table matches `results.json` cell-for-cell: raw success 0.40→0.10→0.13→0.15 and gibberish 0.60→0.90→0.87→0.85 (ladder rungs k=1..4); ortho success 0.40/0.30/0.00/0.40; budget 0.060→0.085→0.104→0.120 (matches the `budget` fields 0.06, 0.0849, 0.1039, 0.12). The verdicts are appropriately hedged: raw-sum interference "Supported", orthogonalization "Directionally supported", budget "Supported". |
| 4 | Results honesty vs artifacts | **PASS (with judge caveat)** | The README does not cherry-pick: it names the K=3 ortho arm collapsing to 0.00 (below raw) as "the clearest sign" of noise, and states "the separation is neither clean nor monotone at 1B." It does *not* hide that ortho's K=2 cross-talk (0.6) is actually *worse* than raw's (0.0) in `results.json` — though the README could call this out more directly than the general "neither clean nor monotone" line. GOOD honesty overall. **Flag:** every rate is from an abliterated 1B target self-graded by the same 1B — gibberish dominates (0.6–1.0 even at K=1, α=0.06), so success/refusal rates are noisy; per the FLAS audit the 1B self-judge inflates refusal. This should be re-run with the off-family Qwen judge and larger eval sets before any non-smoke claim. README §7 already states this caveat. |

## Overall verdict: **PASS**

The lesson's central claim is honestly framed as the project's own testable
hypothesis, the math is correct and unit-tested, the Results table faithfully
transcribes `results.json`, and the honest negatives (K=3 ortho collapse,
non-monotone separation) are stated prominently — good scientific hygiene.
Standing caveat: 1B abliterated-self-judge screening tier; re-run with the
off-family Qwen judge before promoting any number past smoke-grade.

Internal QA pass — independent external review pending (auditor shares a model family with the author).
