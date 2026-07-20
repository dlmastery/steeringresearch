# AUDIT — multiturn_jailbreak

**Auditor role:** independent paper verifier. Scope: do the cited papers exist,
does the code implement what the lesson claims, is the data provenance and the
"inspired-by vs. reproduction" framing honest. No git, no code/README edits were
made. The lead will independently WebFetch-verify the arXiv ids below.

## Paper existence (the critical check)

| citation | id | status (lead WebFetch-verified 2026-07) | role in the lesson |
|---|---|---|---|
| DeepContext: Stateful Real-Time Detection of Multi-Turn Adversarial Intent Drift in LLMs | **arXiv:2602.16935** | **VERIFIED** — resolves; Albrethsen, Datta, Kumar, Rajasekar; 2026-02-18. Abstract = an RNN over sequenced turn-level embeddings that captures gradual intent drift (F1 0.84) — matches the lesson's stateful-sequence framing exactly. | motivates: read the *conversation context* / trajectory, not the isolated turn. |
| Scalable Hierarchical Attention Transformers for Multi-Turn Jailbreak Detection in Long Conversations | **arXiv:2606.21082** | **VERIFIED** — resolves; Hu, Salih, Guha, Srinivasan; 2026-06-19. Conversation-level classification via a hierarchical (per-turn encode -> attention-over-turns) detector — matches the `HierAttn` method. | motivates: attention-pool over turns; expose which turn mattered. |
| ActorAttack/ActorBreaker (Ren et al.) — "LLMs know their vulnerabilities: Uncover Safety Gaps through Natural Distribution Shifts" (v1: "Derail Yourself") | **arXiv:2410.10700** | **VERIFIED** — resolves; Ren, Li, Liu, Xie, Lu, Qiao, Sha, Yan, Ma, Shao; 2024-10-14. Introduces ActorBreaker + the multi-turn safety dataset that `Attack_600` derives from. (Note: retitled after v1 — README uses the current title.) | provides the POSITIVES (the attack the detector must catch). |
| SafeMTData (`SafeMTData/SafeMTData`, `Attack_600`) | HF dataset | **VERIFIED** — loaded locally; 600 rows exposing `multi_turn_queries` (4-5 escalating user turns), `category` (6 harm classes), `query_id`, `actor_name`. | positives loader in `data.py`. |
| UltraChat 200k (`HuggingFaceH4/ultrachat_200k`) | HF dataset | **VERIFIED** — loaded locally (streaming); `messages` = `[{role,content}]`, `train_sft` split; multi-turn benign. | negatives loader in `data.py`. |

All five citations are **verified** (three arXiv abstracts WebFetch-confirmed with
matching titles/authors; two HF datasets loaded and their schemas inspected). The
`[UNVERIFIED]` hedges have been dropped in the README accordingly.

## Findings

| check | verdict | evidence |
|---|---|---|
| Primary-method papers cited with real, verified ids | **PASS** | DeepContext 2602.16935, Hierarchical Attention 2606.21082, ActorAttack 2410.10700 all WebFetch-verified with matching titles/authors/dates; ActorAttack title corrected to its current form. |
| Data provenance stated accurately | **PASS** | Positives = SafeMTData `Attack_600` `multi_turn_queries` grouped by `query_id` (`data.py:_load_positives`); negatives = streamed UltraChat user turns, topic-matched via `CATEGORY_KEYWORDS` + turn-count biased (`data.py:_load_negatives`). Matches the README's dataset table. |
| Method fidelity — stateless baseline vs. stateful models | **PASS** | `models.py` (per contract) exposes `PerTurnMaxProbe` (logreg per turn, max-pool — the stateless baseline), `TrajectoryMLP`, `SeqGRU` (GRU -> last hidden, plus `risk_trajectory`), `HierAttn` (attention over turns, `attention_weights`). The four-method ladder from stateless to stateful directly operationalizes the "trajectory, not turn" thesis. |
| Reproduction honesty — inspired-by, not a paper clone | **PASS** | README §9 states plainly this is an "inspired-by" reconstruction: a per-turn embedding + sequence classifier that operationalizes the *shared idea* of the cited multi-turn defenses, **not** a faithful reimplementation of any one paper's exact architecture. No over-claim of reproducing DeepContext or the Hierarchical-Attention model. |
| Confound honesty | **PASS** | `data.length_confound_report` measures turn-count and total-char AUC vs. label; README §6/§8 instruct the reader to read the confound BEFORE the method table and to discount a length-separable headline. The trap of a lazy negative set is named and mitigated (topic-matching + turn-count biasing). |
| Leakage discipline (CV) | **PASS** | Group-aware CV by `query_id` (3 attack paths share one) with benign groups `10_000_000+i` disjoint from attack ids; README §9 flags grouped CV as load-bearing against near-duplicate leakage. |
| Results honesty (measured) | **PASS** | §8 now reports the first run (MiniLM, n=200/class, 5-fold, bootstrap CIs) as TWO conditions: EASY (per_turn_max already 0.99 — a cautionary trivial benchmark) and HARD (per_turn_max collapses to 0.57, sequence models hold 0.83-0.85). The pre-registered falsifier `AUC(seq_gru) <= AUC(per_turn_max)` is cleared on HARD (0.83 vs 0.57, non-overlapping CIs). `judge: null` (detection, no LLM judge). Screening-tier disclosed. |
| Residual confound reported, not buried | **PASS** | HARD is length-matched (`turncount_auc=0.50`), but `totalchar_auc=0.75` (payload turns wordier) is quoted right beside the 0.85 result, and the lesson claims only the margin over the length-only 0.75 baseline, not the full gap over per-turn. EASY confound (`totalchar_auc=0.11` => length ~0.89) is named as the reason that benchmark is trivial. |
| Hard negatives are leakage-free + length-matched | **PASS** | HARD positive = an attack's last-4 turns; negative = a DIFFERENT attack's first-4 turns (disjoint `query_id`s => zero shared conversations; both exactly 4 turns => turn-count confound designed out). `per_turn_max` is the length-invariant control. Still not per-attack *twins* (disclosed as future work). |

## Overall verdict

**PASS.** All three cited arXiv papers are verified real with matching
titles/authors/dates (the ActorAttack title was corrected to its current form),
and both HF datasets were loaded and their schemas confirmed. The data
provenance, the stateless-to-stateful method ladder, the two-condition
(EASY/HARD) design, the leakage-free + length-matched hard-negative construction,
the confound audit, and the measured results (falsifier cleared on HARD) are all
sound. The lesson is framed correctly as an **inspired-by reconstruction**
(per-turn embedding + sequence classifier) rather than a reproduction of any one
paper's exact architecture. Residual `totalchar_auc=0.75` on HARD is disclosed
and priced into the claim; per-attack benign *twins* remain the one stronger
control left to future work.

*Internal QA pass — independent external review pending (auditor shares a model
family with the author).*
