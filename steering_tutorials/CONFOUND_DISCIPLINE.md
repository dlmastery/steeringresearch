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

Measured from each lesson's `artifacts/results.json` and `README.md` as of
2026-07-25. Confound AUCs are quoted **as recorded**; the directionless value
`max(auc, 1−auc)` is given where it differs, because that is the number the claim
must clear.

### Fully compliant

| lesson | confound baseline(s) measured | value(s) | headline | margin claimed |
|---|---|---|---|---|
| **`hello_world`** | length-only probe, TF-IDF bag-of-words, label-shuffle control, **and** within-length-bin stratification | length-only **acc 0.643 / AUC 0.728**; TF-IDF acc 0.741 / AUC 0.857; shuffled 0.580 (≈chance) | probe **acc 0.875 / AUC 0.965** (Toxic-Chat, n=748) | **+0.232** over length-only, **+0.134** over TF-IDF; in overlap-length bins probe **0.839** vs length-only **0.518** = **+0.321** |
| **`multiturn_jailbreak`** | `turncount_auc`, `totalchar_auc`, both conditions | **hard:** turncount **0.500** (designed out), totalchar **0.752**. **easy:** turncount 0.723, totalchar 0.113 → **0.887 directionless** | hard/Gemma `trajectory_mlp` **AUC 0.956** | claims only the margin over **0.75**; explicitly flags `seq_gru` (0.725) as *below* the length baseline and therefore unconvincing |
| **`cross_trajectory`** | `kcount_auc`, `totalchar_auc`, both conditions | **hard:** kcount **0.500**, totalchar **0.704**. **easy:** kcount 0.500, totalchar 0.110 → **0.890 directionless** | hard `mean_agg` **AUC 0.936** (per-traj baseline collapses to 0.607) | claims the margin over **0.704**; the course README states the clearance explicitly |

`hello_world` is the reference implementation of this rule — it is the only lesson
that runs *all four* controls (shuffle, length-only, TF-IDF, matched-bin
stratification) and reports the matched-bin number as the load-bearing one.
`multiturn_jailbreak` is the reference for **reporting** it: it names its own
weakest cell (`seq_gru` at 0.725 vs a 0.752 length baseline) rather than burying it.

### Measured but not reported — the audit ran, the README did not catch up

| lesson | value in `results.json` | status |
|---|---|---|
| **`biencoder_guard`** | `length_auc` **0.5170** (pos mean 447.9 chars vs neg 409.3) — the *post-fix* number; the original pool gave 0.72 | Clean audit, **best confound number in the course**, but §10 "Results" is still `_pending_` placeholders while `results.json` holds real numbers (held-out zero-shot `bi_encoder` macro-AP **0.408**). The number must be pulled into the results table. |
| **`meerkat`** | `length_auc` **0.3246** (pos mean 104.5 chars vs neg **121.7** — *benign* traces are longer) → **0.675 directionless** | Audit ran; README §9 still says `length_auc` *pending*. And 0.675 is **not** "≈0.5" — the sparse-regime `per_trace` AP **0.818** must be claimed against a 0.675 length tell, and `kmeans_enrich` (**0.568**) sits *below* it. This is a live discount, currently unstated. |

### Non-compliant — a detection claim with no confound baseline

| lesson | what is claimed | what is missing |
|---|---|---|
| **`trajguard`** | streaming detection at **AUC 0.944** (`trajectory_mlp`) / **0.945** (`seq_gru`) / 0.931 (`per_turn_max`), n=300/class | **No trivial-confound baseline anywhere** — no `confound_report`, no `confound` key in `results.json`, no length-only AUC in the README. The only length quantity recorded is mean trajectory length (harmful **38.78** vs benign **37.96** tokens), which is near-matched *by the 40-token generation cap* — a design-time control, not a measured baseline. `threshold_freeform` (0.638) is a weak *method*, not a confound. **The margin the lesson claims is currently unpriced.** |
| **`gavel`** | block rate harmful **0.135** vs benign false-block **0.085** (broad baseline 0.115 / 0.055) | Length is matched **by construction** — median char length harmful **165** vs benign **166**, inherited from `common.data`'s length-matched sampler — which is the *right* fix (rule 5). But no `length_auc` is computed, so the residual tell is unmeasured. Weaker gap than `trajguard`: the confound is designed out, just not verified. |

**Summary:** 3 of 7 fully compliant, 2 measured-but-unreported, 1 designed-out-but-unverified,
**1 (`trajguard`) making a detection claim with no confound baseline at all.**

---

## 4. Code recipe

Drop-in, no dependencies beyond numpy + scikit-learn. The pattern already exists as
`confound_report()` / `length_confound_report()` in `multiturn_jailbreak/data.py`,
`cross_trajectory/data.py`, `meerkat/data.py`, and `biencoder_guard/data.py` —
this is the canonical form.

```python
import numpy as np
from sklearn.metrics import roc_auc_score


def confound_auc(feature, labels):
    """Directionless AUC of ONE trivial feature against the label.

    Returns a value in [0.5, 1.0].
      0.5  -> the feature carries no class information (clean)
      1.0  -> the feature ALONE separates the classes perfectly

    Critically: an AUC of 0.11 is NOT clean. It is a 0.89 confound with the
    sign flipped (the NEGATIVES are the long ones). Always fold with 1 - auc.
    """
    auc = roc_auc_score(labels, np.asarray(feature, dtype=float))
    return float(max(auc, 1.0 - auc))


def confound_report(texts, labels, extra=None):
    """Every trivial baseline for a detection dataset, in one dict.

    ``extra`` carries the domain's own metadata columns -- turn count,
    #trajectories, source-corpus id, patient age. Add every scalar you have;
    the cheap ones are exactly the ones that embarrass you later.
    """
    feats = {
        "char_len":   [len(t) for t in texts],
        "word_count": [len(t.split()) for t in texts],
    }
    feats.update(extra or {})
    report = {name: confound_auc(v, labels) for name, v in feats.items()}
    report["worst"] = max(report.values())          # the number to clear
    return report


def claimable_margin(headline_auc, report, other_baselines=()):
    """The ONLY quantity a detection lesson may headline."""
    floor = max([report["worst"], *other_baselines])
    return float(headline_auc - floor), floor
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

1. **`trajguard`** — add `confound_report()` over the generated completions (token
   count, total characters, prompt length) and quote the worst value beside the
   0.944 headline. Until then its margin is unpriced. *(Blocked on nothing — the
   trajectories are cached; this is a CPU-only recompute.)*
2. **`meerkat`** — fill §9 from `results.json` and state the length tell as
   **0.675 directionless**, not "≈0.5". Re-state `per_trace` (0.818) and
   `kmeans_enrich` (0.568) as margins over 0.675 — `kmeans_enrich` currently sits
   *below* the confound and must be reported that way.
3. **`biencoder_guard`** — fill §10 from `results.json`; quote `length_auc = 0.517`
   in the table. The audit is clean, the reporting is not.
4. **`gavel`** — compute a `length_auc` to verify the by-construction match, so the
   claim rests on a measurement rather than on the sampler's promise.

---

## 6. See also

- [`README.md`](README.md) — the course map and the standards table.
- [`DATA_SUFFICIENCY.md`](DATA_SUFFICIENCY.md) — the companion rule on the
  ≥500/class data floor. Confound discipline and data sufficiency fail together:
  a small pool is also the pool most likely to be accidentally length-separable.
- [`hello_world/artifacts/audit_large.md`](hello_world/artifacts/audit_large.md) —
  the worked reference audit (shuffle + length-only + TF-IDF + matched bins).
- `CLAUDE.md`, hard rigor rubric item 7 — the project-level statement of this rule
  and the `biencoder_guard` 0.72 → 0.52 incident that produced it.
