# Confound discipline — never report a detection number without its trivial baseline

> **The rule.** A detection result is not a result until you have reported the
> strongest **trivial-confound baseline** on the same data. You may claim only the
> **margin above** that baseline — never the raw score, and never the gap over a
> weaker *method*.

This is a course-wide standard. It binds every lesson that produces a
classification/detection number: `hello_world`, `gavel`, `multiturn_jailbreak`,
`trajguard`, `cross_trajectory`, `meerkat`, `biencoder_guard`, and anything added
to the guardrail-detection family later.

---

## 1. The motivating case — a model that hears nothing

A sibling research program in this repo audits **voice-based disease detection**:
classify pathological vs. healthy speech from an audio recording. The published
benchmarks on these corpora report strong ROC-AUCs, and the implied story is that
the model hears something clinically real in the voice.

The audit computed one trivial baseline first: **patient age alone**.

> **Age alone reached ROC-AUC 0.871** on a voice-pathology corpus — matching the
> published benchmark that uses the audio.

A single scalar from the metadata sheet, no audio at all, no model, reproduces the
headline. That does not prove the published models are worthless — but it does mean
**their reported numbers cannot distinguish "the model hears pathology" from "sick
patients in this corpus are older."** Every claim built on the raw AUC is
unsupported until the age baseline is subtracted.

*(Reported by the sibling voice program; this tutorial course has not re-run it.
It is quoted here as the motivating case, not as a result of this course.)*

The generalisation is what binds this course:

| domain | the confound that already "solves" the task |
|---|---|
| voice pathology | patient age (0.871) |
| prompt-harm detection | raw character length |
| multi-turn jailbreak detection | turn count, total characters |
| trace/agent detection | number of trajectories, total text |
| any corpus assembled from two sources | *the rendering of the two sources* |

The last row is the one that bites in practice, and this course has already been
bitten by it: in `biencoder_guard` the benign pool was originally drawn
**prompt-only** while positives were rendered **prompt+response**, giving
`length_auc = 0.72` — a large artefact created purely by how the two halves were
assembled. Redrawing the benigns from the same source rendering brought it to
**0.52** (see `CLAUDE.md`, rigor rubric item 7). No modelling change; the "signal"
was in the plumbing.

---

## 2. What the rule requires, concretely

1. **Compute a confound baseline before you look at the method score.** Every
   trivial feature you can name: length, token/turn/trajectory count, source
   corpus id, any metadata column, any rendering difference between the class
   pools.
2. **Report it beside the headline**, in the results table — not in a caveats
   section, not only in `results.json`.
3. **Claim only `headline − max(confound, non-trivial baseline)`.** If a
   length-only rule gets 0.75 and your model gets 0.85, the claim is "+0.10 over
   length", never "0.85 detection" and never "+0.28 over the per-turn baseline".
4. **Use the directionless AUC.** An AUC of **0.11 is not clean** — it is a **0.89
   confound with the sign flipped** (the *negatives* are the long ones). Always
   report `max(auc, 1 − auc)`. This is the single most common way a confound audit
   is misread.
5. **Prefer designing the confound out to discounting it after.**
   Length-match the negatives, render both pools identically, fix the turn count.
   A measured 0.50 you engineered beats a measured 0.75 you apologise for.
6. **When you cannot design it out, stratify.** Bin the data by the confound and
   re-score *within* bins (the `hello_world` recipe, §4). A method that survives
   inside matched bins has earned its margin.
7. **Pre-register the falsifier in terms of the margin**, e.g. "if
   `AUC(method) ≤ max(length_auc, 1 − length_auc)` the claim is FALSE" — so a
   failed audit cannot be reinterpreted after the fact.

---

## 3. Compliance inventory

Measured from each lesson's `artifacts/results.json` and `README.md` as of 2026-07-25,
**revised 2026-08-08** per the independent four-lesson audit
(`AUDIT_2026-08_four_detection_lessons.md`) which re-ran the loaders and re-read the artifacts.
Confound AUCs are quoted **as recorded**; the directionless value `max(auc, 1−auc)` is given
where it differs, because that is the number the claim must clear.

### Fully compliant

| lesson | confound baseline(s) measured | value(s) | headline | margin claimed |
|---|---|---|---|---|
| **`hello_world`** | length-only probe, TF-IDF bag-of-words, label-shuffle control, **and** within-length-bin stratification | length-only **acc 0.643 / AUC 0.728**; TF-IDF acc 0.741 / AUC 0.857; shuffled 0.580 (≈chance) | probe **acc 0.875 / AUC 0.965** (Toxic-Chat, n=748) | **+0.232** over length-only, **+0.134** over TF-IDF; in overlap-length bins probe **0.839** vs length-only **0.518** = **+0.321** |
| **`multiturn_jailbreak`** | `turncount_auc`, `totalchar_auc`, both conditions | **hard:** turncount **0.500** (designed out), totalchar **0.752**. **easy:** turncount 0.723, totalchar 0.113 → **0.887 directionless** | hard/Gemma `trajectory_mlp` **AUC 0.956** | claims only the margin over **0.75**; explicitly flags `seq_gru` (0.725) as *below* the length baseline and therefore unconvincing |
| **`trajguard`** | `confound_report()` (charlen bar), computed and **folded correctly**, run **before** the CV/method block | charlen **AUC 0.7354** directionless | best methods `trajectory_mlp` **0.944** / `seq_gru` **0.945** | margins computed against `max{method baseline, confound}`; the paper's own `threshold_freeform` (0.638) is **reported landing below** the 0.7354 bar rather than buried — the best confound *reporting* discipline in the set |

`hello_world` is the reference implementation of this rule — it is the only lesson
that runs *all four* controls (shuffle, length-only, TF-IDF, matched-bin
stratification) and reports the matched-bin number as the load-bearing one.
`multiturn_jailbreak` is the reference for **reporting** it: it names its own
weakest cell (`seq_gru` at 0.725 vs a 0.752 length baseline) rather than burying it.
`trajguard` is the reference for **sequencing** it: the bar is computed and asserted
*before* any method score is looked at, and a method that lands *below* its own bar
(`threshold_freeform`) is reported that way, not omitted. (`trajguard` still fails the
course's ≥500/class floor at 300/class — see `DATA_SUFFICIENCY.md` — and lacks a
label-shuffle / matched-bin control; compliance here is about the confound bar
specifically, not the whole rigor rubric.)

### Measured but not reported — the audit ran, the README did not catch up

| lesson | value in `results.json` | status |
|---|---|---|
| **`biencoder_guard`** | `length_auc` **0.5170** (pos mean 447.9 chars vs neg 409.3) — the *post-fix* number; the original pool gave 0.72 | Clean audit, **best confound number in the course**, but §10 "Results" is still `_pending_` placeholders while `results.json` holds real numbers (held-out zero-shot `bi_encoder` macro-AP **0.408**). The number must be pulled into the results table. |
| **`meerkat`** | `length_auc` **0.3246** (pos mean 104.5 chars vs neg **121.7** — *benign* traces are longer) → **0.675 directionless** | Audit ran; README §9 still says `length_auc` *pending*. And 0.675 is **not** "≈0.5" — the sparse-regime `per_trace` AP **0.818** must be claimed against a 0.675 length tell, and `kmeans_enrich` (**0.568**) sits *below* it. This is a live discount, currently unstated. |
| **`cross_trajectory`** | `easy` condition: `totalchar_auc = 0.10975` raw → **0.890 directionless** | True "fully compliant" only for the **`hard`** condition. **`easy` is not**: the README prints AUCs of **0.991–0.998** for `easy` against no bar at all, so a true margin of roughly **+0.10** is presented as **+0.99**. The irony: this exact **0.890** figure was already recorded in this document's §3 (the table above, pre-2026-08-08 revision) — the lesson never picked it up. `hard` (kcount 0.500, totalchar **0.704**, `mean_agg` AUC 0.936, margin **+0.232** stated) remains genuinely compliant and is unaffected. |

### Non-compliant — a detection claim with no confound baseline

| lesson | what is claimed | what is missing |
|---|---|---|
| **`gavel`** | block rate harmful **0.135** vs benign false-block **0.085** (broad baseline 0.115 / 0.055) | Length is matched **by construction** — median char length harmful **165** vs benign **166**, inherited from `common.data`'s length-matched sampler — which is the *right* fix (rule 5). But no `length_auc` is computed, so the residual tell is unmeasured. |

**Summary (revised 2026-08-08):** 3 of 7 fully compliant (`hello_world`, `multiturn_jailbreak`,
`trajguard`), 3 measured-but-unreported (`biencoder_guard`, `meerkat`, and now `cross_trajectory`'s
`easy` condition specifically — its `hard` condition is compliant), 1 (`gavel`) with the confound
designed out but not verified. **`trajguard` moved from non-compliant to compliant** — its
`confound_report()` now exists, runs before the CV block, folds correctly, and prices its own
weak method honestly. See `AUDIT_2026-08_four_detection_lessons.md` and each lesson's
`AUDIT_2026-08.md` for the full evidence trail.

---

## 4. The shared instrument — `common/confound.py`

The canonical implementation lives in **`steering_tutorials/common/confound.py`** — not in any
individual lesson's `data.py`. (An earlier version of this section claimed
`cross_trajectory/data.py`'s `confound_report()` was "the canonical form." **It was not** — that
function returned the raw AUC and never folded it. This document described code that had never
existed, and it was the document every future lesson would have copied from. See the fold example
below for the live cost of that error.)

The 2026-08-08 four-lesson audit found each detection lesson had grown a different partial
reimplementation:

| lesson | folds `max(auc,1-auc)` | count/turn bar | content (TF-IDF) bar | shuffle control |
|---|---|---|---|---|
| `trajguard` | yes | n/a | no | no |
| `multiturn_jailbreak` | yes | yes | no | no |
| `cross_trajectory` | **no — returns raw AUC** | yes | no | no |
| `biencoder_guard` | **no** | **absent** | no | no |

`common/confound.py` is the single course-wide instrument that replaces all four. It runs **four
bars**, not one — a length-only audit is not sufficient:

1. **`length_bar(texts, labels)`** — can raw character count separate the classes?
2. **`count_bar(units, labels)`** — can the number of units (turns / trajectories / tokens)
   separate them? (`units` is a list of sequences; a lesson that fixes the count by construction
   should land at exactly 0.5 — a PASS worth recording, not a reason to skip the measurement.)
3. **`content_bar(texts, labels, seed=0, n_folds=5, ...)`** — can a bag-of-words / TF-IDF
   centroid-cosine classifier, fit strictly **train-fold-only** under K-fold CV, separate the
   classes? This is the bar that actually matters and no lesson had it before this file: a
   "trajectory" / "bi-encoder" detector that cannot beat unigrams is not reading what it claims to.
4. **`shuffle_control(texts, labels, ...)`** — re-run the content bar with labels permuted; should
   land near 0.5. **This is a LEAKAGE DIAGNOSTIC, not a bar to clear.** `confound_report()`
   deliberately excludes `shuffle` from `worst_auc`/`worst_name` (the binding bar a method must
   beat) — a shuffle AUC far from 0.5 means the *pipeline* is leaking (duplicate texts straddling
   folds, a group spanning the split), not that the data legitimately carries the signal a method
   is allowed to use.

`confound_report(texts, labels, units=None, seed=0, n_folds=5)` runs the applicable bars and
returns the worst (largest directionless) as `report["worst_auc"]` / `report["worst_name"]`.
`margin_over_bar(method_auc, report, baseline_auc=None)` is the only quantity a lesson may
headline: `method_auc − max(confound_bar, method_baseline)`.

**Why the fold matters, concretely — a real incident, not a hypothetical:**
`cross_trajectory`'s `easy` condition measured `totalchar_auc = 0.10975`. Read raw, that looks
*cleaner than chance*. Folded — `directionless(auc) = max(auc, 1 - auc)` — it is **0.890**: a
near-perfect length tell with the sign flipped (the negatives are the long ones). **A raw 0.110 IS
a 0.890 confound.** The lesson's README printed `easy`-condition AUCs of 0.991–0.998 against no bar
at all, so a true margin of roughly +0.10 was presented as +0.99 (see §3).

Import it directly — no per-lesson reimplementation:

```python
from steering_tutorials.common.confound import confound_report, margin_over_bar, format_report

report = confound_report(texts, labels, units=turns_per_conversation)  # units optional
print(format_report(report))                        # ASCII-only, cp1252-safe
m = margin_over_bar(headline_auc, report)
# CLAIMED: m["margin"] over m["binding_bar_name"] = m["binding_bar"].
# NEVER the raw headline_auc, and NEVER a margin over "shuffle" (it isn't a bar).
```

**The matched-bin check** (the `hello_world` recipe) — for when the confound cannot
be designed out and you need to show the method works *at fixed confound*:

```python
def matched_bin_scores(scores, labels, confound, n_bins=4):
    """Re-score inside quantile bins of the confound.

    Inside a bin the confound is ~constant, so a method that still separates
    the classes there is not riding the confound. Report the MIDDLE bins --
    the extreme bins are where the classes barely overlap and mean least.
    """
    edges = np.quantile(confound, np.linspace(0, 1, n_bins + 1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confound >= lo) & (confound <= hi)
        if len(np.unique(labels[m])) < 2:      # bin is single-class: skip
            continue
        out.append({
            "range": (float(lo), float(hi)),
            "n": int(m.sum()),
            "method_auc":   roc_auc_score(labels[m], scores[m]),
            "confound_auc": confound_auc(confound[m], labels[m]),
        })
    return out
```

**Reporting template** — paste this into the results section, filled in:

```
Confound audit: worst trivial baseline = <name> at AUC <x.xxx>
                (directionless; raw <y.yyy>, <n_pos>/<n_neg> means <a>/<b>).
Headline <method> AUC <z.zzz>.
CLAIMED: +<z-x> over the strongest trivial baseline. We do NOT claim <z.zzz>.
```

---

## 5. Actions this inventory implies

1. ~~**`trajguard`** — add `confound_report()`...~~ **DONE as of 2026-08-08.** `confound_report()`
   now runs before the CV block and folds correctly; see §3. Remaining gap: no label-shuffle or
   matched-bin control (the `hello_world`-recipe controls), and it still fails the ≥500/class floor
   at 300/class (`DATA_SUFFICIENCY.md`).
2. **`cross_trajectory`** — price the `easy` condition against its own **0.890** `totalchar_auc`
   bar (currently printed with no bar at all); state the margin as **~+0.10**, not the raw
   0.991–0.998. `hard` needs no change.
3. **`meerkat`** — fill §9 from `results.json` and state the length tell as
   **0.675 directionless**, not "≈0.5". Re-state `per_trace` (0.818) and
   `kmeans_enrich` (0.568) as margins over 0.675 — `kmeans_enrich` currently sits
   *below* the confound and must be reported that way.
4. **`biencoder_guard`** — fill §10 from `results.json`; quote `length_auc = 0.517`
   in the table. The audit is clean, the reporting is not.
5. **`gavel`** — compute a `length_auc` to verify the by-construction match, so the
   claim rests on a measurement rather than on the sampler's promise. Still the sole
   non-compliant lesson.

---

## 6. See also

- [`README.md`](README.md) — the course map and the standards table.
- [`DATA_SUFFICIENCY.md`](DATA_SUFFICIENCY.md) — the companion rule on the
  ≥500/class data floor. Confound discipline and data sufficiency fail together:
  a small pool is also the pool most likely to be accidentally length-separable.
- [`hello_world/artifacts/audit_large.md`](hello_world/artifacts/audit_large.md) —
  the worked reference audit (shuffle + length-only + TF-IDF + matched bins).
- [`AUDIT_2026-08_four_detection_lessons.md`](AUDIT_2026-08_four_detection_lessons.md) —
  the source of the 2026-08-08 revision above: the cross-lesson confound-instrument audit,
  the `cross_trajectory` `easy`-condition finding, and the `trajguard` compliance re-grade.
  Each lesson's own `AUDIT_2026-08.md` (`biencoder_guard/`, `cross_trajectory/`,
  `multiturn_jailbreak/`, `trajguard/`) carries the per-lesson evidence.
- `CLAUDE.md`, hard rigor rubric item 7 — the project-level statement of this rule
  and the `biencoder_guard` 0.72 → 0.52 incident that produced it.
