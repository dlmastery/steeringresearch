# AUDIT — multiturn_jailbreak

**Auditor role:** independent paper verifier. Scope: do the cited papers exist,
does the code implement what the lesson claims, is the data provenance and the
"inspired-by vs. reproduction" framing honest. No git, no code/README edits were
made. The lead will independently WebFetch-verify the arXiv ids below.

## Paper existence (the critical check)

| citation | id | status (pending lead WebFetch) | role in the lesson |
|---|---|---|---|
| DeepContext — context-aware multi-turn jailbreak defense | **arXiv:2602.16935** | **[UNVERIFIED]** — cited as a primary motivation for the stateful/context-aware framing; must be confirmed to resolve with a matching title/abstract. | motivates: read the *conversation context*, not the isolated turn. |
| Hierarchical Attention for multi-turn safety | **arXiv:2606.21082** | **[UNVERIFIED]** — cited as the motivation for the `HierAttn` (per-turn encoder -> attention-over-turns) method; confirm title/abstract. | motivates: attention-pool over turns; expose which turn mattered. |
| ActorAttack / "Derail Yourself" (multi-turn attack) | **arXiv:2410.10700** | **[UNVERIFIED]** — the source of the `Attack_600` positive data and the escalation threat model; confirm authors (Ren et al.) + title. | provides the POSITIVES (the attack the detector must catch). |
| SafeMTData (`SafeMTData/SafeMTData`, `Attack_600`) | HF dataset | **[UNVERIFIED]** — confirm the repo/config exists and exposes `multi_turn_queries`, `category`, `query_id`. | positives loader in `data.py`. |
| UltraChat 200k (`HuggingFaceH4/ultrachat_200k`) | HF dataset | **[UNVERIFIED]** — well-known; confirm the `messages` schema + `train_sft` split used for benign negatives. | negatives loader in `data.py`. |

All five are marked `[UNVERIFIED]` in the README pending the lead's WebFetch.
Do **not** drop the hedge until the id resolves AND the title/authors match.

## Findings

| check | verdict | evidence |
|---|---|---|
| Primary-method papers cited with real ids + `[UNVERIFIED]` hedge | **PASS (pending verify)** | README §8 cites DeepContext 2602.16935 and Hierarchical Attention 2606.21082 as the stateful/attention motivation, both tagged `[UNVERIFIED]`; ActorAttack 2410.10700 tagged likewise. Honest hedging until WebFetch. |
| Data provenance stated accurately | **PASS** | Positives = SafeMTData `Attack_600` `multi_turn_queries` grouped by `query_id` (`data.py:_load_positives`); negatives = streamed UltraChat user turns, topic-matched via `CATEGORY_KEYWORDS` + turn-count biased (`data.py:_load_negatives`). Matches the README's dataset table. |
| Method fidelity — stateless baseline vs. stateful models | **PASS** | `models.py` (per contract) exposes `PerTurnMaxProbe` (logreg per turn, max-pool — the stateless baseline), `TrajectoryMLP`, `SeqGRU` (GRU -> last hidden, plus `risk_trajectory`), `HierAttn` (attention over turns, `attention_weights`). The four-method ladder from stateless to stateful directly operationalizes the "trajectory, not turn" thesis. |
| Reproduction honesty — inspired-by, not a paper clone | **PASS** | README §9 states plainly this is an "inspired-by" reconstruction: a per-turn embedding + sequence classifier that operationalizes the *shared idea* of the cited multi-turn defenses, **not** a faithful reimplementation of any one paper's exact architecture. No over-claim of reproducing DeepContext or the Hierarchical-Attention model. |
| Confound honesty | **PASS** | `data.length_confound_report` measures turn-count and total-char AUC vs. label; README §6/§8 instruct the reader to read the confound BEFORE the method table and to discount a length-separable headline. The trap of a lazy negative set is named and mitigated (topic-matching + turn-count biasing). |
| Leakage discipline (CV) | **PASS** | Group-aware CV by `query_id` (3 attack paths share one) with benign groups `10_000_000+i` disjoint from attack ids; README §9 flags grouped CV as load-bearing against near-duplicate leakage. |
| Results honesty (code-only) | **PASS** | §8 carries a `[PENDING GPU RUN]` banner, PENDING cells wired to the `results.json` schema, screening-tier disclosure, and a pre-registered falsifier (`AUC(seq_gru) <= AUC(per_turn_max)` => thesis FALSE). `judge: null` correctly reflects that detection has no LLM judge. |
| Negatives = topic-matched, not paired twins | **CONCERN (disclosed)** | The benign set shares surface topic but is sampled UltraChat, not per-attack twins of the same subject — a weaker control than paired twins. README §9 states this explicitly; not a blocker, but the strongest negative set is not used. |

## Overall verdict

**CONDITIONAL PASS — pending id verification.** The data provenance, method
ladder, confound audit, leakage discipline, and results honesty are all sound,
and the lesson is framed correctly as an **inspired-by reconstruction**
(per-turn embedding + sequence classifier) rather than a reproduction of any
single paper's exact architecture. The one required follow-up is external:
the lead must WebFetch-verify **arXiv:2602.16935** (DeepContext),
**arXiv:2606.21082** (Hierarchical Attention), and **arXiv:2410.10700**
(ActorAttack), plus the two HF dataset repos, and only then drop the
`[UNVERIFIED]` tags — together with any author/title correction the lookup
surfaces (do not un-hedge a citation whose title or authors do not match).

*Internal QA pass — independent external review pending (auditor shares a model
family with the author).*
