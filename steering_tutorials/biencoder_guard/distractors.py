"""distractors.py -- deterministic SYNTHETIC DISTRACTOR policies for the
LARGE-LABEL-SCALE ACCURACY experiment.

------------------------------------------------------------------------------
WHY THIS FILE EXISTS
------------------------------------------------------------------------------
GLiNER-bi-Encoder ("The Million-Label NER", Stepanov et al., arXiv:2602.18487)
makes TWO claims as the label bank grows toward ~1000 labels:

  (1) LATENCY stays flat for a bi-encoder while a cross/uni-encoder grows
      linearly   -> already tested by `encoders.scaling_latency` (EXP-D).
  (2) ACCURACY is MAINTAINED at that scale                -> tested HERE.

To test (2) honestly you need labels that a real system would plausibly hold
but that are NOT the labels you are scoring. Our taxonomy has only 16 real
policies, so we PAD the policy bank with synthetic DISTRACTOR policies up to
K in {16, 64, 256, 900} and then measure the 16 real columns *in the presence
of* those competitors.

------------------------------------------------------------------------------
THE CORRECTNESS PROBLEM, AND HOW IT IS SOLVED
------------------------------------------------------------------------------
A distractor that is secretly a PARAPHRASE of a real policy is poison. If the
bank contains "material that encourages assault" alongside the real `Violence`
column, a violent test row will legitimately rank the distractor at or above
the true column -- and we would score that as accuracy LOST at scale when it is
really a LABELLING ERROR of our own making (an unlabelled true positive, i.e.
a false negative by construction).

Four independent, auditable guards keep the distractor bank disjoint from the
real taxonomy. All are deterministic; none depends on the model under test:

  G1 CONSTRUCTION BY DISJOINT DOMAIN. Distractors are composed from an
     ARTIFACT x DEFECT grid drawn entirely from NON-HARM content-operations
     moderation (duplicate cross-posting, undisclosed sponsorship, keyword
     stuffing, missing alt text, dead links, lorem-ipsum filler, ...). These
     are real policy families in a large taxonomy -- Opir (arXiv:2605.29659)
     runs a three-level 996-category label space -- but they share no harm
     semantics with BeaverTails/toxic-chat/wildguard categories.
  G2 HARM-STEM BLOCKLIST. Every candidate phrasing is regex-checked against a
     word-prefix blocklist of harm stems (violen|weapon|drug|child|sexual|hate|
     suicid|terror|fraud|stalk|jailbreak|toxic|...). Any hit -> the candidate is
     DROPPED, not repaired.
  G3 LEXICAL OVERLAP GATE. Jaccard similarity on CONTENT words (moderation
     boilerplate such as "content"/"material"/"text"/"posts" is stoplisted so
     the gate measures substantive overlap, not template scaffolding) against
     EVERY real description AND every real paraphrase. Above
     C.DISTRACTOR_MAX_JACCARD -> DROPPED.
  G4 EMBEDDING AUDIT (report-only, never a filter). After the bank is built we
     report each distractor's maximum cosine to the 16 real policy vectors.
     This is REPORTED rather than filtered on purpose: filtering by the very
     encoder under test would delete exactly the confusable distractors and
     make the scaling test trivially easy -- a self-serving selection effect.
     The reader gets the number and can judge.

ASSERT-YOUR-ANCHORS: if the filters ever leave fewer distractors than the grid
demands, `build_distractors` RAISES. It never silently returns a short bank
(a short bank would quietly turn "K=900" into a smaller, easier K).

Determinism: the candidate grid is a fixed nested loop (no set iteration), and
the only randomness is one `np.random.default_rng(seed)` permutation, so the
same seed always yields the same bank in the same order. Because the order is
fixed, the K=64 bank is a strict PREFIX of the K=900 bank -- the scales are
nested, so a change between two K values cannot be an artifact of a different
distractor sample.

ASCII-only stdout (Windows cp1252).
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from . import config as C

# ---------------------------------------------------------------------------
# G1: the ARTIFACT x DEFECT grid -- ordinary content-operations moderation.
# Nothing here touches violence, sex, drugs, weapons, hate, self-harm, crime,
# privacy, terrorism, jailbreaking or toxicity: the 16 real policy families.
# 45 artifacts x 24 defects = 1080 candidate policies, comfortably more than the
# 884 needed for K=900 even after G2/G3 drop some.
# ---------------------------------------------------------------------------
_ARTIFACTS = [
    "product listings", "support tickets", "wiki edits", "forum replies",
    "profile biographies", "event announcements", "job postings",
    "recipe submissions", "photo captions", "code snippets",
    "changelog entries", "poll questions", "marketplace offers",
    "podcast descriptions", "study-group invitations", "translation contributions",
    "hardware reviews", "travel itineraries", "meeting notes", "newsletter issues",
    "quiz answers", "flashcard decks", "gardening logs", "sports fixtures",
    "book summaries", "restaurant ratings", "bug reports", "feature requests",
    "classroom handouts", "workshop agendas", "volunteer sign-up sheets",
    "furniture assembly guides", "knitting patterns", "birdwatching checklists",
    "chess puzzle posts", "language-exchange adverts", "fan-fiction chapter notes",
    "conference abstracts", "library catalogue entries", "museum exhibit labels",
    "weather station readings", "transit schedule updates",
    "charity fundraiser pages", "hobby-project build logs", "municipal notice posts",
]

_DEFECTS = [
    "that are duplicated verbatim across many boards",
    "that omit a required sponsorship disclosure",
    "padded with keyword stuffing to game search ranking",
    "written entirely in unformatted capital letters",
    "that link only to expired or broken destinations",
    "auto-generated by a bot without any disclosure",
    "filed under a clearly mismatched category",
    "that reuse another member's wording without attribution",
    "containing placeholder lorem-ipsum filler",
    "that solicit off-platform payment",
    "that inflate engagement through coordinated voting",
    "missing the accessibility descriptions the platform requires",
    "that mix incompatible measurement units without conversion",
    "left abandoned in an unfinished draft state",
    "that bury the actual question under unrelated anecdotes",
    "machine-translated so poorly the meaning is lost",
    "that repost the same announcement every single day",
    "whose title promises something the body never delivers",
    "that quote an entire prior thread without trimming it",
    "uploaded at a resolution too low to be legible",
    "that reference a version of the product retired years ago",
    "tagged with dozens of irrelevant topic labels",
    "that consist only of a bare external link with no context",
    "submitted repeatedly from newly created duplicate accounts",
]

# Five phrasings per distractor: index 0 is the DESCRIPTION, 1..4 are the
# PARAPHRASES, mirroring the real taxonomy's 1 + C.POLICY_PARAPHRASES layout so
# the policy tower is built identically for real and synthetic policies.
_TEMPLATES = [
    "Content that consists of {a} {d}.",
    "Submissions made up of {a} {d}.",
    "Material flagged because it involves {a} {d}.",
    "A community-standards breach in which a member posts {a} {d}.",
    "Text presenting {a} {d}.",
]

# ---------------------------------------------------------------------------
# G2: harm-stem blocklist. Word-PREFIX matched (\bstem) so "skill" does not trip
# "kill" and "stale" does not trip "steal". A candidate containing ANY of these
# is dropped outright -- we never try to rewrite it.
# ---------------------------------------------------------------------------
_HARM_STEMS = [
    "violen", "weapon", "firearm", "explosiv", "attack", "assault", "kill",
    "murder", "threat", "incit", "abet", "harm", "hurt", "injur", "suicid",
    "drug", "narcot", "substance", "child", "minor", "groom", "sexual", "sex",
    "porn", "nsfw", "explicit", "hate", "hateful", "slur", "racis", "discrimin",
    "stereotyp", "prejudic", "bigot", "terror", "extremis", "crime", "criminal",
    "fraud", "scam", "launder", "theft", "steal", "embezzl", "forger",
    "stalk", "dox", "privacy", "surveil", "track", "jailbreak", "guardrail",
    "bypass", "override", "toxic", "abus", "harass", "insult", "offensive",
    "misinform", "disinform", "deceit", "deceptive", "unethical", "cheat",
    "animal", "cruel", "torture", "neglect", "exploit", "endanger", "propagand",
    "partisan", "divisive", "political", "politics", "illegal", "illicit",
    "banned", "adult",
]
_HARM_RE = re.compile(r"\b(" + "|".join(_HARM_STEMS) + r")", re.IGNORECASE)

# ---------------------------------------------------------------------------
# G3: lexical-overlap gate. The stoplist removes (a) English function words and
# (b) MODERATION BOILERPLATE that both the real descriptions and our templates
# necessarily share ("content", "material", "text", "posts", ...). Leaving the
# boilerplate in would make every pair look ~20% similar for purely structural
# reasons and the gate would measure nothing.
# ---------------------------------------------------------------------------
_STOPWORDS = set("""
a an the and or but if of to in on at by for with without from into over under
that this those these it its is are was were be been being as not no any some
all each other another such than then so very can could may might must will
would should do does did done has have had having up out off about against
""".split()) | set("""
content contents material materials text texts post posts posting postings
submission submissions request requests message messages user users member
members community standards standard platform policy policies rule rules
involves involve consists consist consisting presenting presents present made
make makes flagged flag breach violation violating someone something anyone
""".split())

_WORD_RE = re.compile(r"[a-z][a-z'-]*")


def _content_words(s: str) -> set:
    """Lowercase content-word set of a phrasing (stopwords + boilerplate removed)."""
    return {w for w in _WORD_RE.findall(str(s).lower()) if w not in _STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def real_policy_phrasings(policies) -> list:
    """Every phrasing the REAL taxonomy uses: each description + each paraphrase.

    The overlap gate must run against paraphrases too, not just descriptions --
    a distractor that duplicates a paraphrase is just as poisonous as one that
    duplicates the description, because the policy tower averages all of them
    into the real policy's prototype vector.
    """
    out = []
    for pol in policies:
        out.append(str(pol.get("description", "")))
        out.extend(str(p) for p in pol.get("paraphrases", []))
    return [s for s in out if s]


def _candidate_grid():
    """The deterministic (artifact, defect) -> 5 phrasings candidate stream.

    A plain nested loop (never a set/dict iteration) so the candidate ORDER is
    reproducible on every interpreter and platform.
    """
    for a in _ARTIFACTS:
        for d in _DEFECTS:
            yield a, d, [t.format(a=a, d=d) for t in _TEMPLATES]


def build_distractors(policies, n_needed: int, seed: int = None,
                      max_jaccard: float = None, verbose: bool = True) -> dict:
    """Build `n_needed` synthetic distractor policies disjoint from `policies`.

    Returns {"policies": [...], "audit": {...}}. Each distractor is a Policy dict
    with the SAME keys the real taxonomy uses ({id,name,description,paraphrases,
    group}) so `encoders.build_policy_bank` consumes it unchanged.

    RAISES if the filters leave fewer than `n_needed` survivors -- a short bank
    would silently shrink the "K=900" condition into an easier one.
    """
    seed = C.DISTRACTOR_SEED if seed is None else int(seed)
    max_jaccard = C.DISTRACTOR_MAX_JACCARD if max_jaccard is None else float(max_jaccard)

    real_phrasings = real_policy_phrasings(policies)
    real_sets = [_content_words(s) for s in real_phrasings]
    real_exact = {str(s).strip().lower() for s in real_phrasings}

    kept, seen_desc = [], set()
    n_total = n_harm = n_jac = n_dup = 0
    worst_jac = 0.0
    for a, d, phrasings in _candidate_grid():
        n_total += 1
        # G2: harm-stem blocklist over EVERY phrasing (description + paraphrases).
        if any(_HARM_RE.search(p) for p in phrasings):
            n_harm += 1
            continue
        # exact-duplicate guard (belt and braces; should never fire).
        if any(p.strip().lower() in real_exact for p in phrasings):
            n_dup += 1
            continue
        # G3: lexical overlap vs every real phrasing, worst case over our phrasings.
        j = 0.0
        for p in phrasings:
            ws = _content_words(p)
            for rs in real_sets:
                v = _jaccard(ws, rs)
                if v > j:
                    j = v
        if j > max_jaccard:
            n_jac += 1
            continue
        worst_jac = max(worst_jac, j)
        desc = phrasings[0]
        if desc in seen_desc:
            n_dup += 1
            continue
        seen_desc.add(desc)
        kept.append((a, d, phrasings, j))

    if len(kept) < n_needed:
        raise RuntimeError(
            "distractor pool exhausted: need %d, only %d survived the filters "
            "(harm-stem drops=%d, jaccard drops=%d, dup drops=%d of %d candidates). "
            "Widen _ARTIFACTS/_DEFECTS rather than relaxing a filter."
            % (n_needed, len(kept), n_harm, n_jac, n_dup, n_total))

    # ONE seeded permutation -> a fixed order, so smaller K banks are PREFIXES of
    # larger ones (nested scales; a K-to-K change cannot be a resampling artifact).
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(kept))[:n_needed]

    out = []
    for rank, idx in enumerate(order):
        a, d, phrasings, j = kept[int(idx)]
        short = " ".join(str(d).split()[:4])
        out.append({
            "id": "distractor_%04d" % rank,
            "name": "%s / %s" % (a, short),
            "description": phrasings[0],
            "paraphrases": list(phrasings[1:]),
            "group": "distractor",
            "max_jaccard_to_real": float(j),
        })

    audit = {
        "seed": int(seed),
        "n_candidates": int(n_total),
        "n_survived_filters": int(len(kept)),
        "n_used": int(len(out)),
        "dropped_harm_stem": int(n_harm),
        "dropped_jaccard": int(n_jac),
        "dropped_duplicate": int(n_dup),
        "max_jaccard_gate": float(max_jaccard),
        "max_jaccard_observed": float(max(d["max_jaccard_to_real"] for d in out)) if out else 0.0,
        "mean_jaccard_observed": float(np.mean([d["max_jaccard_to_real"] for d in out])) if out else 0.0,
        "fingerprint": fingerprint(out),
        "examples": [d["description"] for d in out[:3]],
    }
    if verbose:
        print("[distract] %d/%d candidates survived (harm-stem drop=%d, jaccard drop=%d); "
              "using %d  max_jaccard=%.3f  fp=%s"
              % (len(kept), n_total, n_harm, n_jac, len(out),
                 audit["max_jaccard_observed"], audit["fingerprint"]))
    return {"policies": out, "audit": audit}


def fingerprint(distractors) -> str:
    """Stamp the exact bank: sha256 over the ordered description list (12 hex).

    STAMP YOUR INPUTS -- a cached distractor bank that cannot be tied back to the
    generator that produced it is not evidence. The runner stores this beside the
    cached embedding matrix and refuses a cache whose fingerprint disagrees.
    """
    h = hashlib.sha256()
    for d in distractors:
        h.update(str(d["description"]).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def embedding_audit(distractor_bank: np.ndarray, real_bank: np.ndarray) -> dict:
    """G4 (report-only): how close does each distractor sit to a REAL policy?

    Returns the distribution of "max cosine to any real policy" over distractors.
    Deliberately NOT used as a filter: pruning by the encoder under test would
    remove precisely the hard competitors and inflate the scaling result.
    """
    D = np.asarray(distractor_bank, dtype=np.float32)
    R = np.asarray(real_bank, dtype=np.float32)
    if D.size == 0 or R.size == 0:
        return {"error": "empty bank"}
    D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    R = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    m = (D @ R.T).max(axis=1)
    return {
        "mean_max_cos_to_real": float(np.mean(m)),
        "p95_max_cos_to_real": float(np.percentile(m, 95)),
        "max_max_cos_to_real": float(np.max(m)),
        "n_above_0.90": int((m >= 0.90).sum()),
        "n_above_0.95": int((m >= 0.95).sum()),
    }


# ---------------------------------------------------------------------------
# CPU self-test: no model, no data.
# ---------------------------------------------------------------------------
def _self_test():
    from . import data

    policies = data.build_taxonomy()
    need = 900 - len(policies)
    built = build_distractors(policies, need)
    ds = built["policies"]
    assert len(ds) == need, "wrong count"

    # determinism: same seed -> identical bank.
    again = build_distractors(policies, need, verbose=False)
    assert fingerprint(ds) == fingerprint(again["policies"]), "generator not deterministic"

    # nesting: a smaller request is a strict PREFIX of the larger one.
    small = build_distractors(policies, 48, verbose=False)["policies"]
    assert [d["description"] for d in small] == [d["description"] for d in ds[:48]], \
        "smaller K is not a prefix of larger K"

    # disjointness: no distractor phrasing equals a real phrasing, and no harm stem.
    real = {s.strip().lower() for s in real_policy_phrasings(policies)}
    for d in ds:
        for p in [d["description"]] + list(d["paraphrases"]):
            assert p.strip().lower() not in real, "distractor duplicates a real phrasing"
            assert _HARM_RE.search(p) is None, "harm stem leaked into %r" % p
        assert d["max_jaccard_to_real"] <= C.DISTRACTOR_MAX_JACCARD
        assert len(d["paraphrases"]) >= C.POLICY_PARAPHRASES - 1

    # unique descriptions
    assert len({d["description"] for d in ds}) == len(ds), "duplicate distractors"

    # NEGATIVE CONTROL. G2/G3 drop ZERO of our 1080 candidates -- construction (G1)
    # already made the grid disjoint. A filter that never fires is untested code and
    # must not be reported as a guarantee, so we prove here that both gates DO fire
    # on deliberately poisoned input: a covert paraphrase of the real Violence and
    # Self Harm policies must be rejected, by the harm-stem rule and (with the stems
    # scrubbed) by the lexical-overlap rule.
    poison_harm = "Content that consists of threats or incitement of physical violence."
    assert _HARM_RE.search(poison_harm) is not None, "G2 failed to flag a harm paraphrase"
    # a paraphrase whose harm words are spelled around the blocklist still trips G3.
    real_sets = [_content_words(s) for s in real_policy_phrasings(policies)]
    poison_lex = "Content that consists of instructions or methods for self-injury or suicide."
    j = max(_jaccard(_content_words(poison_lex), rs) for rs in real_sets)
    assert j > C.DISTRACTOR_MAX_JACCARD, \
        "G3 failed to flag a near-duplicate paraphrase (jaccard %.3f)" % j
    # ...and a genuine distractor must NOT trip either gate (no false alarms).
    clean = ds[0]["description"]
    assert _HARM_RE.search(clean) is None
    assert max(_jaccard(_content_words(clean), rs) for rs in real_sets) <= C.DISTRACTOR_MAX_JACCARD
    print("negative control OK: G2 rejects a harm paraphrase; G3 rejects a lexical "
          "near-duplicate (jaccard %.3f > %.2f); neither fires on a real distractor"
          % (j, C.DISTRACTOR_MAX_JACCARD))

    print("distractors.py self-test OK: %d deterministic, prefix-nested, harm-free, "
          "max lexical Jaccard to any real phrasing = %.3f (gate %.2f)"
          % (len(ds), max(d["max_jaccard_to_real"] for d in ds), C.DISTRACTOR_MAX_JACCARD))


if __name__ == "__main__":
    _self_test()
