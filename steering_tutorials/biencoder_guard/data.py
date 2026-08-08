"""data.py -- the HARD, MULTI-DATASET, MANY-LABEL safety corpus for the
BI-ENCODER GUARDRAIL lesson.

WHAT A NEWCOMER NEEDS TO KNOW FIRST
-----------------------------------
This lesson moderates a piece of text against a whole TAXONOMY of safety
POLICIES at once. A *policy* is one named rule ("no instructions for building
weapons", "no self-harm encouragement", ...). Real moderation systems carry
dozens to thousands of these. The bi-encoder trick (see __init__.py / config.py)
is to embed the text ONCE and match it against a CACHED bank of policy-description
vectors -- so cost stays flat as the policy list grows, and a brand-new policy is
added zero-shot from its written description alone.

To teach that honestly we need a corpus that is (a) MANY-LABEL (one text can
violate several policies), (b) genuinely HARD (benign look-alikes that a length
or keyword shortcut cannot separate from violations), and (c) REAL (in-the-wild
data, not toy strings). No single public dataset gives all three, so we POOL
complementary ones into ONE unified multi-label corpus over a shared taxonomy:

  * BeaverTails (PKU-Alignment)  -- 14 fine-grained, MULTI-LABEL harm categories on
    prompt+response pairs. This is the CORE of our label space, and the safe rows
    give topically-adjacent benign hard-negatives.
  * Aegis 2.0 (nvidia)           -- a SECOND, independent safety taxonomy (12 core
    + 9 fine-grained categories) annotated by a DIFFERENT regime, over prompt and
    response. Pooled to break the single-dataset dependence and to fill the
    columns BeaverTails' 30k split starves.
  * toxic-chat (lmsys)           -- REAL user prompts hand-labelled for `toxicity`
    and `jailbreaking`. Supplies in-the-wild toxic + adversarial positives and
    benign hard-negatives that look adversarial but are not.
  * wildguardmix (allenai)       -- GATED. It has NEVER loaded on this host: HTTP
    403, no HF token, ZERO rows contributed to every corpus this lesson has built.
    The call is kept so a tokened host gets it; no result here rests on it.

WHERE THE ROWS ACTUALLY CAME FROM
---------------------------------
Read `results.json -> source_distribution` and `results.json -> achieved`, never a
docstring (this one included). The 2026-08 audit found the previous version of this
file describing a three-way pool that measured 93.5% BeaverTails / 6.5% toxic-chat /
0% wildguardmix, with `results.json` recording only the REQUESTED config. Every
corpus now stamps what it achieved, per column, per source, plus a
`pool_fingerprint` over the sampled texts.

WHY BENIGN LOOK-ALIKES ARE *HARD* NEGATIVES
-------------------------------------------
A cheap classifier "wins" by reading surface features -- length, a banned keyword.
The interesting failures are benign texts that sit RIGHT NEXT TO a policy in
meaning ("how do chemists dispose of reactive waste safely?" vs a weapons
request). We deliberately draw benign rows from the SAME datasets' safe slices so
they are topically adjacent, and we run a length-confound audit
(:func:`confound_report`) to prove length alone cannot separate the classes.

CPU-only. Downloads datasets via HuggingFace `datasets`. Loads NO model. Stdout is
ASCII-only (Windows cp1252 -- we never print unicode; write "cos"/"F1"/">=").
"""
from __future__ import annotations

try:  # SSL-intercepting middlebox on this host: use the OS trust store.
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import sys
import hashlib
import random
from collections import Counter, defaultdict

import numpy as np

from steering_tutorials.biencoder_guard import config as C
from steering_tutorials.common.confound import (
    confound_report as _shared_confound_report,
    format_report as format_confound_report,
    margin_over_bar,
)


# --- ASCII-safe logging ------------------------------------------------------
def _eprint(*args) -> None:
    """ASCII-only stderr print (Windows cp1252 consoles choke on unicode)."""
    msg = " ".join(str(a) for a in args)
    try:
        print(msg, file=sys.stderr)
    except Exception:
        print(msg.encode("ascii", "replace").decode("ascii"), file=sys.stderr)


def _clean(text) -> str:
    """Collapse whitespace, drop the U+FFFD replacement char, trim. One readable line."""
    return " ".join(str(text).replace("�", " ").split()).strip()


def _norm_key(text: str) -> str:
    """Aggressive normalization for exact-dup detection: lowercase, alnum tokens only.

    Two texts differing only by case/spacing/punctuation share this key, so trivial
    surface duplicates collapse to one row and cannot straddle a train/test split.
    """
    import re
    s = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    return re.sub(r"\s+", " ", s).strip()


def _hash_int(key: str) -> int:
    """Stable positive 31-bit int id from a string (for reproducible group ids)."""
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)


# --- THE TAXONOMY ------------------------------------------------------------
# The 14 BeaverTails category KEYS (long, comma-joined) mapped to our short, stable
# policy ids. This is the single source of truth for the CORE label columns; the
# jailbreak + toxicity policies (from toxic-chat / wildguard) are appended after.
_BEAVERTAILS_KEY_TO_ID = {
    "animal_abuse": "animal_abuse",
    "child_abuse": "child_abuse",
    "controversial_topics,politics": "controversial_topics",
    "discrimination,stereotype,injustice": "discrimination_stereotype",
    "drug_abuse,weapons,banned_substance": "drug_weapon",
    "financial_crime,property_crime,theft": "financial_crime",
    "hate_speech,offensive_language": "hate_speech",
    "misinformation_regarding_ethics,laws_and_safety": "misinformation",
    "non_violent_unethical_behavior": "non_violent_unethical",
    "privacy_violation": "privacy_violation",
    "self_harm": "self_harm",
    "sexually_explicit,adult_content": "sexually_explicit",
    "terrorism,organized_crime": "terrorism",
    "violence,aiding_and_abetting,incitement": "violence",
}

# One clear sentence + several diverse paraphrases per policy. The paraphrases are
# the "synthetic schema expansion" teaching point: many natural-language phrasings
# of the SAME rule, which the policy tower averages into a robust prototype vector
# (encoders.build_policy_bank). Handwritten here (no GPT-4.1) -- honest and auditable.
_POLICY_SPEC = {
    "animal_abuse": (
        "Content that encourages, describes, or facilitates cruelty, neglect, or harm to animals.",
        ["Text promoting the mistreatment or torture of animals.",
         "Requests that help someone hurt, abuse, or neglect an animal.",
         "Material glorifying cruelty toward pets, livestock, or wildlife.",
         "Instructions or encouragement for harming living creatures."]),
    "child_abuse": (
        "Content that sexualizes, exploits, endangers, or abuses children.",
        ["Material that exploits or endangers minors.",
         "Requests involving the sexual abuse or grooming of children.",
         "Text that facilitates harm to a child.",
         "Any content abusing, endangering, or exploiting a minor."]),
    "controversial_topics": (
        "Inflammatory or one-sided content on divisive political or social topics meant to provoke.",
        ["Politically charged content designed to inflame division.",
         "One-sided propaganda on hot-button social issues.",
         "Text that pushes a partisan agenda in a provocative way.",
         "Divisive commentary intended to stir political conflict."]),
    "discrimination_stereotype": (
        "Content that demeans or stereotypes people based on a protected attribute.",
        ["Text that stereotypes or belittles a group by race, gender, religion, etc.",
         "Prejudiced content promoting unfair treatment of a protected class.",
         "Material reinforcing harmful stereotypes about a group of people.",
         "Discriminatory statements targeting someone's identity."]),
    "drug_weapon": (
        "Content that facilitates the illicit manufacture, acquisition, or use of drugs or weapons.",
        ["Instructions for making or obtaining illegal drugs or weapons.",
         "Help acquiring banned substances, firearms, or explosives.",
         "Text explaining how to synthesize narcotics or build a weapon.",
         "Requests that facilitate drug abuse or weapon trafficking."]),
    "financial_crime": (
        "Content that facilitates fraud, theft, money laundering, or other financial crime.",
        ["Help committing fraud, scams, or theft of money or property.",
         "Instructions for laundering money or running a financial scam.",
         "Text that facilitates embezzlement, forgery, or credit-card fraud.",
         "Requests enabling property crime or financial deception."]),
    "hate_speech": (
        "Hateful, dehumanizing, or offensive language directed at a person or group.",
        ["Slurs or dehumanizing language aimed at a group.",
         "Offensive content expressing hatred toward people.",
         "Text that attacks or vilifies someone with hateful language.",
         "Abusive, demeaning speech targeting an individual or community."]),
    "misinformation": (
        "False or misleading claims about ethics, law, health, or safety presented as fact.",
        ["Fabricated claims about laws, health, or safety.",
         "Misleading information that could cause real-world harm.",
         "False assertions presented as established fact.",
         "Deceptive content spreading dangerous misinformation."]),
    "non_violent_unethical": (
        "Non-violent but clearly unethical behavior such as lying, cheating, or manipulation.",
        ["Help with deceitful, dishonest, or manipulative conduct.",
         "Text encouraging cheating, lying, or betraying trust.",
         "Requests to enable unethical (though non-violent) actions.",
         "Content facilitating dishonest or exploitative behavior."]),
    "privacy_violation": (
        "Content that exposes private information or helps surveil or stalk a person.",
        ["Requests to reveal someone's private personal information.",
         "Help doxxing, tracking, or surveilling an individual.",
         "Text that violates a person's privacy or confidentiality.",
         "Instructions for stalking or unmasking someone's identity."]),
    "self_harm": (
        "Content that encourages, instructs, or glorifies self-harm or suicide.",
        ["Text encouraging someone to hurt or kill themselves.",
         "Instructions or methods for self-injury or suicide.",
         "Content that glorifies or normalizes self-harm.",
         "Requests seeking help to end one's own life."]),
    "sexually_explicit": (
        "Explicit sexual or adult content.",
        ["Graphic sexual or pornographic material.",
         "Explicit adult content describing sexual acts.",
         "Text of an overtly sexual, NSFW nature.",
         "Pornographic or sexually explicit requests."]),
    "terrorism": (
        "Content that supports, plans, or facilitates terrorism or organized crime.",
        ["Support or planning for terrorist acts.",
         "Help facilitating organized crime or a violent extremist plot.",
         "Text promoting or coordinating terrorism.",
         "Requests aiding a terrorist or criminal organization."]),
    "violence": (
        "Content that incites, threatens, or provides help for physical violence against people.",
        ["Threats or incitement of physical violence.",
         "Help planning or carrying out an attack on a person.",
         "Text encouraging assault, murder, or bodily harm.",
         "Requests that aid and abet violent acts."]),
    # --- appended (adversarial / toxicity) ---
    "jailbreak": (
        "An adversarial prompt that tries to bypass, override, or trick the model's safety rules.",
        ["A prompt engineered to evade the assistant's safety guardrails.",
         "An attempt to jailbreak or override the model's restrictions.",
         "Manipulative instructions designed to unlock forbidden behavior.",
         "An adversarial attack that tricks the model into ignoring its policies."]),
    "toxicity": (
        "Toxic, abusive, or harassing language in a user message.",
        ["Rude, abusive, or harassing user input.",
         "A message with toxic or hostile language.",
         "Insulting or aggressive content aimed at someone.",
         "Harassing, offensive, or hateful phrasing in a request."]),
}

# --- Aegis 2.0 columns that have NO clean home in the 16 above ---------------
# Rule (task brief): where an Aegis category has no clean mapping, ADD a column
# rather than force it into a neighbour. Five do. They are appended AFTER the
# original 16 so every existing column index is unchanged, which is what lets
# EXP-G/EXP-H keep comparing against prior runs column-for-column.
_AEGIS_EXTRA_SPEC = {
    "criminal_planning": (
        "Content that plans, assists, or confesses to a crime not covered by a more specific policy.",
        ["Help planning or carrying out a criminal act.",
         "A confession to, or coordination of, illegal activity.",
         "Requests that assist someone in breaking the law.",
         "Text that facilitates general criminal conduct."]),
    "unauthorized_advice": (
        "Authoritative medical, legal, or financial advice given without qualification or caveat.",
        ["Unqualified medical, legal, or financial guidance presented as expert advice.",
         "Professional-sounding advice a licensed expert should give.",
         "Definitive health, legal, or investment instructions given to an individual.",
         "Specific advice that requires professional credentials to give safely."]),
    "malware": (
        "Content that creates, distributes, or deploys malicious software or cyber intrusions.",
        ["Help writing viruses, ransomware, or other malicious code.",
         "Instructions for hacking into a system or network.",
         "Requests to build or spread malware.",
         "Text that facilitates a cyber attack or unauthorized system access."]),
    "intellectual_property": (
        "Content that infringes copyright or trademark, or that plagiarizes protected work.",
        ["Reproducing copyrighted work without permission.",
         "Help plagiarizing someone else's writing.",
         "Requests to infringe a trademark or pirate protected material.",
         "Text that copies protected content as if it were original."]),
    "high_risk_gov_decisions": (
        "Content that steers high-stakes government or public-policy decisions such as "
        "law enforcement, military, or immigration determinations.",
        ["Guidance driving consequential government or public-policy determinations.",
         "Advice on law-enforcement, military, or immigration decisions about people.",
         "Recommendations for high-stakes state decisions affecting individuals' rights.",
         "Input intended to decide a governmental action with serious consequences."]),
}

# THE AEGIS TAXONOMY CROSSWALK (mirrored in README section 6 as a table).
# Aegis category string (as it appears in `violated_categories`, ", "-joined)
#   -> the list of OUR policy ids it activates. A list, because this corpus is
# multi-label and one Aegis category can legitimately span two of our policies.
# Keys are matched after `_norm_key` normalisation, so casing, punctuation and
# "Sexual (minor)" vs "Sexual Minor" all resolve to the same entry.
_AEGIS_CROSSWALK_RAW = {
    # --- 12 core categories ---
    "Hate/Identity Hate": ["hate_speech", "discrimination_stereotype"],
    "Sexual": ["sexually_explicit"],
    "Suicide and Self Harm": ["self_harm"],
    "Violence": ["violence"],
    "Guns/Illegal Weapons": ["drug_weapon"],
    "Threat": ["violence"],
    "PII/Privacy": ["privacy_violation"],
    "Sexual (minor)": ["child_abuse", "sexually_explicit"],
    "Sexual Minor": ["child_abuse", "sexually_explicit"],
    "Criminal Planning/Confessions": ["criminal_planning"],
    "Harassment": ["toxicity"],
    "Controlled/Regulated Substances": ["drug_weapon"],
    "Profanity": ["toxicity"],
    # --- 9 fine-grained subcategories ---
    "Illegal Activity": ["criminal_planning"],
    "Immoral/Unethical": ["non_violent_unethical"],
    "Unauthorized Advice": ["unauthorized_advice"],
    "Political/Misinformation/Conspiracy": ["misinformation", "controversial_topics"],
    "Fraud/Deception": ["financial_crime", "non_violent_unethical"],
    "Copyright/Trademark/Plagiarism": ["intellectual_property"],
    "High Risk Gov. Decision Making": ["high_risk_gov_decisions"],
    "Malware": ["malware"],
    "Manipulation": ["non_violent_unethical"],
    # --- top-level markers that are NOT harm categories ---
    # Aegis' top level is Safe / Needs Caution; "Needs Caution" can appear here and
    # names no policy. It maps to NOTHING on purpose: a row whose only category is
    # this is dropped and counted, never relabelled benign (see _load_aegis).
    "Needs Caution": [],
    "None": [],
    "Safe": [],
}
# The five ids above that exist ONLY because of Aegis. Dropping AEGIS_EXTRA_COLUMNS
# removes these columns and every category routed to them becomes unmapped.
_AEGIS_ONLY_IDS = list(_AEGIS_EXTRA_SPEC.keys())


def _aegis_crosswalk(extra_columns: bool) -> dict:
    """Normalised {aegis_category_key -> [our policy ids]} for the active taxonomy.

    Its ONE job: resolve Aegis' category strings to our columns without silent
    forcing. When `extra_columns` is False the five Aegis-only ids are stripped from
    every target list, so a category that mapped ONLY to them becomes unmapped and
    its rows are dropped-and-counted rather than pushed into a neighbouring policy.
    """
    out = {}
    for raw, ids in _AEGIS_CROSSWALK_RAW.items():
        kept = [i for i in ids if extra_columns or i not in _AEGIS_ONLY_IDS]
        out[_norm_key(raw)] = kept
    return out


# Fixed column order: the 14 BeaverTails core policies, then jailbreak, then
# toxicity, then (optionally) the five Aegis-only columns. APPEND-ONLY -- indices
# 0..15 are frozen so results stay comparable across runs.
_CORE_IDS = list(_BEAVERTAILS_KEY_TO_ID.values())
_BASE_POLICY_ORDER = _CORE_IDS + ["jailbreak", "toxicity"]
_POLICY_GROUP = {**{i: "beavertails" for i in _CORE_IDS},
                 "jailbreak": "adversarial", "toxicity": "toxicity",
                 **{i: "aegis" for i in _AEGIS_ONLY_IDS}}


def policy_order(extra_columns=None) -> list:
    """The active column order (its ONE job: one place decides how many columns exist).

    16 base columns, plus the five Aegis-only columns when `C.AEGIS_EXTRA_COLUMNS`
    is on. APPEND-ONLY: index 0..15 never move, so a run with the extra columns is
    still column-comparable to one without for the first 16.
    """
    if extra_columns is None:
        extra_columns = bool(C.AEGIS_EXTRA_COLUMNS)
    return list(_BASE_POLICY_ORDER) + (list(_AEGIS_ONLY_IDS) if extra_columns else [])


def build_taxonomy() -> list:
    """Return the unified many-label policy taxonomy (its ONE job: define the label
    columns). Each entry is a Policy dict {id,name,description,paraphrases,group}.
    The column ORDER here is the column order of every Y matrix in the corpus, so it
    must stay stable. WHY paraphrases: the policy tower averages a description plus
    its paraphrases into a robust multi-prototype vector (the schema-expansion idea).
    """
    spec = {**_POLICY_SPEC, **_AEGIS_EXTRA_SPEC}
    policies = []
    for pid in policy_order():
        desc, paras = spec[pid]
        # Guarantee at least C.POLICY_PARAPHRASES paraphrases (pad by reusing the
        # description if a spec is ever short) so build_policy_bank never underflows.
        paras = list(paras)
        while len(paras) < C.POLICY_PARAPHRASES:
            paras.append(desc)
        policies.append({
            "id": pid,
            "name": pid.replace("_", " ").title(),
            "description": desc,
            "paraphrases": paras,
            "group": _POLICY_GROUP[pid],
        })
    return policies


# --- source loaders ----------------------------------------------------------
def _add_row(store, text, label_ids, source, group_key, col_of, P):
    """Append one corpus row into `store` (dedup by normalized text).

    `store` = {"seen":set, "texts":[], "Y":[], "sources":[], "gkeys":[]}. A row is a
    text, its multi-hot label vector over the P policy columns, its source tag, and a
    GROUP KEY (the underlying prompt) so a group-aware split keeps near-dup rows on
    one side. Returns True if the row was added (new), False if it was a duplicate.
    """
    text = _clean(text)
    if len(text) < 10:                      # skip empty / near-empty rows
        return False
    nk = _norm_key(text)
    if not nk or nk in store["seen"]:       # exact / surface duplicate -> skip
        return False
    store["seen"].add(nk)
    vec = np.zeros(P, dtype=np.float32)
    for lid in label_ids:                   # set the columns this text violates
        vec[col_of[lid]] = 1.0
    store["texts"].append(text)
    store["Y"].append(vec)
    store["sources"].append(source)
    store["gkeys"].append(group_key or nk)  # fall back to the text key if no prompt
    return True


def _load_beavertails(store, n_per_class, n_benign, col_of, P, rng):
    """Pool BeaverTails 30k_train into the corpus (its ONE job: the CORE 14 columns).

    text = prompt + "\\n" + response; row['category'] is a dict of 14 bools -> the 14
    core policy columns; row['is_safe']==True with no active category -> a benign
    hard-negative. We STREAM (RAM-friendly) and stop once every core column has
    n_per_class positives and we have n_benign benign rows. Rare categories may not
    fill -- we take what the pool offers and report it honestly.

    BeaverTails is the PRIORITY benign source: its safe rows are rendered
    prompt+response, so they share the LENGTH distribution of the BeaverTails
    positives -- filling the benign quota from here first drives the length-confound
    AUC toward ~0.5 (a guard must read intent, not length). Returns the number of
    benign rows actually collected so the caller can backfill any deficit from the
    (shorter, prompt-only) toxic-chat / wildguard benign slices.
    """
    from datasets import load_dataset as hf_load

    col_counts = defaultdict(int)           # positives collected per core policy id
    n_benign_have = 0
    scanned = 0
    # Bound the scan so a low-cap smoke does not walk all 30k rows chasing a rare class.
    max_scan = max(30000, n_per_class * 200)
    try:
        stream = hf_load(C.BEAVERTAILS_DATASET, split=C.BEAVERTAILS_TRAIN_SPLIT,
                         streaming=True)
    except Exception as e:
        _eprint("[data] BeaverTails load FAILED (%s) -- continuing without it" % e)
        return 0
    for row in stream:
        scanned += 1
        if scanned > max_scan:
            break
        cats = row.get("category") or {}
        # Active core policies = BeaverTails category keys flagged True.
        active = [_BEAVERTAILS_KEY_TO_ID[k] for k, v in cats.items()
                  if v and k in _BEAVERTAILS_KEY_TO_ID]
        text = _clean(row.get("prompt", "")) + "\n" + _clean(row.get("response", ""))
        gkey = _norm_key(row.get("prompt", ""))  # group by PROMPT (dup prompts share a split side)
        if active:                          # a harmful (positive) row
            # keep only while it still helps an under-filled column (avoids runaway)
            if any(col_counts[a] < n_per_class for a in active):
                if _add_row(store, text, active, "beavertails", gkey, col_of, P):
                    for a in active:
                        col_counts[a] += 1
        elif row.get("is_safe", False):     # a benign hard-negative (all-zero label)
            if n_benign_have < n_benign:
                if _add_row(store, text, [], "beavertails_benign", gkey, col_of, P):
                    n_benign_have += 1
        # stop early once the CORE columns and the benign quota are satisfied
        if n_benign_have >= n_benign and all(col_counts[c] >= n_per_class for c in _CORE_IDS):
            break
    _eprint("[data] BeaverTails: scanned=%d benign=%d per-core-col=%s"
            % (scanned, n_benign_have,
               {c: col_counts[c] for c in _CORE_IDS}))
    return n_benign_have


def _parse_aegis_categories(raw) -> list:
    """Split Aegis' `violated_categories` string into raw category names.

    The field is a ", "-joined string ("Sexual (minor), Criminal Planning/Confessions").
    Forward slashes are INSIDE names, not separators, so we split on commas only.
    """
    if raw is None:
        return []
    s = str(raw).strip()
    if not s or s.lower() in ("none", "null", "nan"):
        return []
    return [part.strip() for part in s.split(",") if part.strip()]


def _load_aegis(store, n_per_class, n_benign, col_of, P, rng, split=None):
    """Pool Aegis 2.0 into the corpus (its ONE job: a SECOND annotation regime).

    WHY this dataset and not more BeaverTails. The 2026-08 audit measured the corpus
    at 93.5% BeaverTails -- a "three-dataset pool" that was really one dataset, so
    every number inherited one annotation regime's idiosyncrasies with no way to see
    it. Aegis 2.0 is 33,416 rows over an INDEPENDENT 12-core + 9-fine-grained
    taxonomy, human- and LLM-labelled in separate columns, and it is ungated. It
    fills the six columns BeaverTails' 30k split starves, supplies thousands of extra
    safe rows for the benign side, and its held-out test split becomes a genuine
    CROSS-ANNOTATOR transfer arm (see load_transfer_arms).

    text = prompt + "\\n" + response (response may be empty -> prompt only), matching
    how the BeaverTails rows are rendered so the two sources share a length
    distribution and the length-confound bar stays near 0.5.

    Labels come from `violated_categories` through `_AEGIS_CROSSWALK`. A row flagged
    unsafe whose categories ALL fail to map is DROPPED and counted in
    `unmapped_categories` -- it is NOT relabelled benign, which is exactly the silent,
    plausible-looking failure CLAUDE.md section 18.8 is about.

    Returns {"n_pos","n_benign","n_dropped_unmapped","unmapped_categories",
             "per_col"} so the caller can report what actually landed.
    """
    from datasets import load_dataset as hf_load

    split = split or C.AEGIS_TRAIN_SPLIT
    cross = _aegis_crosswalk(bool(C.AEGIS_EXTRA_COLUMNS))
    stats = {"n_pos": 0, "n_benign": 0, "n_dropped_unmapped": 0,
             "unmapped_categories": {}, "per_col": {}}
    try:
        ds = hf_load(C.AEGIS_DATASET, split=split)
    except Exception as e:
        _eprint("[data] Aegis 2.0 (%s) load FAILED (%s) -- continuing without it"
                % (split, e))
        stats["error"] = str(e)
        return stats

    col_counts = defaultdict(int)
    rows = list(ds)
    rng.shuffle(rows)                       # deterministic order for reproducibility
    for row in rows:
        prompt = _clean(row.get("prompt", ""))
        resp = _clean(row.get("response") or "")
        text = (prompt + "\n" + resp) if resp else prompt
        gkey = _norm_key(prompt)            # group by PROMPT: an Aegis prompt appears
                                            # with several responses; they must not
                                            # straddle the train/test split.
        p_lab = str(row.get("prompt_label", "") or "").strip().lower()
        r_lab = str(row.get("response_label", "") or "").strip().lower()
        unsafe = (p_lab == "unsafe") or (r_lab == "unsafe")

        if unsafe:
            raw_cats = _parse_aegis_categories(row.get("violated_categories"))
            active, unmapped = [], []
            for cat in raw_cats:
                key = _norm_key(cat)
                if key in cross:
                    active.extend(cross[key])
                else:
                    unmapped.append(cat)
            active = sorted(set(active))
            for cat in unmapped:            # count EVERY unrecognised string
                stats["unmapped_categories"][cat] = \
                    stats["unmapped_categories"].get(cat, 0) + 1
            if not active:
                # unsafe but nothing to put it in -> DROP and count. Never a benign.
                stats["n_dropped_unmapped"] += 1
                continue
            if any(col_counts[a] < n_per_class for a in active):
                if _add_row(store, text, active, "aegis", gkey, col_of, P):
                    stats["n_pos"] += 1
                    for a in active:
                        col_counts[a] += 1
        elif p_lab == "safe" and r_lab in ("safe", ""):
            if stats["n_benign"] < n_benign:
                if _add_row(store, text, [], "aegis_benign", gkey, col_of, P):
                    stats["n_benign"] += 1

    stats["per_col"] = {k: int(v) for k, v in sorted(col_counts.items())}
    _eprint("[data] Aegis 2.0 (%s): pos=%d benign=%d dropped_unmapped=%d "
            "unmapped_category_strings=%d"
            % (split, stats["n_pos"], stats["n_benign"], stats["n_dropped_unmapped"],
               len(stats["unmapped_categories"])))
    if stats["unmapped_categories"]:
        _eprint("[data]   UNMAPPED Aegis categories (rows dropped, not relabelled): "
                + ", ".join("%s=%d" % kv
                            for kv in sorted(stats["unmapped_categories"].items())))
    _eprint("[data]   Aegis per-col: %s" % stats["per_col"])
    return stats


def _load_toxicchat(store, n_per_class, n_benign, col_of, P, rng):
    """Pool toxic-chat into the corpus (its ONE job: the jailbreak + toxicity columns).

    text = user_input; toxicity==1 -> the `toxicity` column; jailbreaking==1 -> the
    `jailbreak` column (a row can be both). Non-toxic, non-jailbreak prompts are REAL
    in-the-wild benign hard-negatives, but they are PROMPT-ONLY (shorter than the
    prompt+response BeaverTails rows), so we only take enough of them to BACKFILL a
    benign deficit `n_benign` the caller could not fill from BeaverTails (n_benign==0
    means the positives are still added but no benign is drawn). Returns the number of
    benign rows added. Small dataset -> loaded whole (not streamed).
    """
    from datasets import load_dataset as hf_load

    n_jb = n_tox = n_ben = 0
    try:
        ds = hf_load(C.TOXICCHAT_DATASET, C.TOXICCHAT_CONFIG, split="train")
    except Exception as e:
        _eprint("[data] toxic-chat load FAILED (%s) -- continuing without it" % e)
        return 0
    rows = list(ds)
    rng.shuffle(rows)                       # deterministic order for reproducibility
    for row in rows:
        text = row.get("user_input", "")
        gkey = _norm_key(text)
        toxic = int(row.get("toxicity", 0) or 0) == 1
        jail = int(row.get("jailbreaking", 0) or 0) == 1
        labels = []
        if toxic:
            labels.append("toxicity")       # toxicity==1 -> toxicity policy
        if jail:
            labels.append("jailbreak")      # jailbreaking==1 -> jailbreak policy
        if labels:                          # a positive (toxic and/or jailbreak)
            need = (("toxicity" in labels and n_tox < n_per_class) or
                    ("jailbreak" in labels and n_jb < n_per_class))
            if need and _add_row(store, text, labels, "toxicchat", gkey, col_of, P):
                n_tox += "toxicity" in labels
                n_jb += "jailbreak" in labels
        else:                               # benign in-the-wild hard-negative
            if n_ben < n_benign:
                if _add_row(store, text, [], "toxicchat_benign", gkey, col_of, P):
                    n_ben += 1
    _eprint("[data] toxic-chat: jailbreak=%d toxicity=%d benign(backfill)=%d"
            % (n_jb, n_tox, n_ben))
    return n_ben


def _load_wildguard(store, n_per_class, n_benign, col_of, P, rng):
    """OPTIONALLY pool wildguardmix (its ONE job: extra adversarial hard cases).

    prompt_harm_label=='harmful' -> the jailbreak column if the row is adversarial,
    else the toxicity column (a generic harm signal -- wildguard gives no fine
    categories). Benign adversarial rows are drawn only as a LAST-RESORT benign
    backfill (`n_benign` = the deficit still unfilled after BeaverTails + toxic-chat).
    wildguardmix is GATED; if it fails to load we SKIP it silently (BeaverTails +
    toxic-chat suffice). Returns the number of benign rows added.
    """
    from datasets import load_dataset as hf_load

    try:
        ds = hf_load(C.WILDGUARD_DATASET, C.WILDGUARD_CONFIG, split="train")
    except Exception as e:
        _eprint("[data] wildguardmix SKIPPED (gated/unavailable: %s)" % e)
        return 0
    n_pos = n_ben = 0
    rows = list(ds)
    rng.shuffle(rows)
    for row in rows:
        text = row.get("prompt", "")
        resp = row.get("response") or ""
        if resp:
            text = _clean(text) + "\n" + _clean(resp)
        gkey = _norm_key(row.get("prompt", ""))
        harmful = str(row.get("prompt_harm_label", "")).lower() == "harmful"
        adversarial = bool(row.get("adversarial", False))
        if harmful:
            labels = ["jailbreak"] if adversarial else ["toxicity"]
            if n_pos < n_per_class and _add_row(store, text, labels, "wildguard", gkey, col_of, P):
                n_pos += 1
        else:                               # benign (possibly adversarial) hard-negative
            if n_ben < n_benign:
                if _add_row(store, text, [], "wildguard_benign", gkey, col_of, P):
                    n_ben += 1
    _eprint("[data] wildguardmix: harmful=%d benign(backfill)=%d" % (n_pos, n_ben))
    return n_ben


def _finalize(store, policies, rng) -> dict:
    """Turn the row store into the corpus dict (fixed-seed shuffle, int group ids)."""
    n = len(store["texts"])
    order = list(range(n))
    rng.shuffle(order)                      # deterministic shuffle of ALL rows
    texts = [store["texts"][i] for i in order]
    Y = np.stack([store["Y"][i] for i in order]).astype(np.float32) if n else \
        np.zeros((0, len(policies)), np.float32)
    sources = [store["sources"][i] for i in order]
    gkeys = [store["gkeys"][i] for i in order]
    groups = [_hash_int(g) for g in gkeys]  # stable int group id per underlying prompt
    is_harmful = (Y.sum(axis=1) > 0).astype(np.int64) if n else np.zeros(0, np.int64)
    return {"texts": texts, "Y": Y, "policies": policies, "sources": sources,
            "groups": groups, "is_harmful": is_harmful}


def load_corpus(n_per_class=C.N_PER_CLASS, n_benign=C.N_BENIGN, seed=C.SEED) -> dict:
    """Pool the three datasets into ONE multi-label safety corpus (the lesson's data).

    Its ONE job: return the shared dataset dict {texts, Y[n,P] multi-hot float32,
    policies, sources, groups, is_harmful}. Positives come from BeaverTails (14 core
    columns) + toxic-chat/wildguard (jailbreak + toxicity); benign hard-negatives
    come from the same datasets' safe slices. Exact/surface duplicates are dropped and
    the whole set is shuffled with `seed`. Per-column positive + benign counts are
    printed to stderr (rubric: aim >= n_per_class positives per column and >= n_benign
    benign; pool-limited columns are reported honestly).
    """
    rng = random.Random(seed)
    policies = build_taxonomy()
    P = len(policies)
    col_of = {p["id"]: i for i, p in enumerate(policies)}
    store = {"seen": set(), "texts": [], "Y": [], "sources": [], "gkeys": []}

    # BENIGN = LENGTH-MATCHED. BeaverTails and Aegis safe rows are rendered
    # prompt+response like the positives, so they share the positives' length
    # distribution; we fill the benign quota from those FIRST and only backfill any
    # deficit from the shorter, prompt-only toxic-chat / wildguard benigns -- this
    # keeps length from leaking the label (length_auc ~0.5). Every source still
    # contributes its POSITIVES regardless of the benign cap.
    bt_benign = _load_beavertails(store, n_per_class, n_benign, col_of, P, rng)
    deficit = max(0, n_benign - bt_benign)
    aegis_stats = {}
    if C.AEGIS_ON:
        aegis_stats = _load_aegis(store, n_per_class, deficit, col_of, P, rng)
        deficit = max(0, deficit - int(aegis_stats.get("n_benign", 0)))
    tc_benign = _load_toxicchat(store, n_per_class, deficit, col_of, P, rng)
    deficit = max(0, deficit - tc_benign)
    _load_wildguard(store, n_per_class, deficit, col_of, P, rng)

    corpus = _finalize(store, policies, rng)
    corpus["aegis"] = aegis_stats
    corpus["requested"] = {"n_per_class": int(n_per_class),
                           "n_benign": int(n_benign), "seed": int(seed)}
    corpus["achieved"] = corpus_provenance(corpus, n_per_class, n_benign)

    # --- provenance: per-column positive counts + benign + mean chars (stderr) ---
    Y = corpus["Y"]
    n = len(corpus["texts"])
    col_pos = Y.sum(axis=0).astype(int) if n else np.zeros(P, int)
    n_benign_have = int((corpus["is_harmful"] == 0).sum())
    mean_chars = float(np.mean([len(t) for t in corpus["texts"]])) if n else 0.0
    _eprint("[data] corpus N=%d  policies=%d  benign=%d  mean_chars=%.1f"
            % (n, P, n_benign_have, mean_chars))
    for i, p in enumerate(policies):
        _eprint("[data]   col %2d %-24s pos=%d" % (i, p["id"], int(col_pos[i])))
    _eprint("[data] source dist: "
            + ", ".join("%s=%d" % (s, c) for s, c in sorted(Counter(corpus["sources"]).items())))
    _eprint(format_shortfall(corpus["achieved"]))
    return corpus


# --- ACHIEVED-vs-REQUESTED provenance (never echo the config back) -----------
def pool_fingerprint(texts) -> str:
    """SHA-256 over the SORTED normalised texts (its ONE job: identify this pool).

    Sorted, so the fingerprint is invariant to row order but changes the instant the
    membership changes. This is the anchor that tells a reader whether a cached
    embedding matrix belongs to the corpus sitting beside it -- the `meerkat` defect
    (an artifact that cannot be regenerated from the code beside it) made concrete.
    """
    h = hashlib.sha256()
    for key in sorted(_norm_key(t) for t in texts):
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def corpus_provenance(corpus, requested_n_per_class, requested_n_benign,
                      test_idx=None) -> dict:
    """What the corpus ACHIEVED, per column and per source (its ONE job).

    `results.json` used to record `n_per_class: 500` read straight off the config --
    the REQUESTED value, while six of sixteen columns held 109-374 positives. A
    reader could not detect the rubric failure from the artifact at all. This returns
    the achieved side so the runner can persist it:

      per_column_positives   corpus counts, per policy id
      per_column_positives_test  the same on the test split (pass `test_idx`)
      n_benign_achieved / n_harmful / class_balance
      source_distribution    measured row counts per source tag
      requested_vs_achieved  both numbers plus a per-column `shortfall` bool
      pool_fingerprint       sha256 of the sorted normalised texts
    """
    Y = np.asarray(corpus["Y"])
    texts = list(corpus["texts"])
    policies = corpus["policies"]
    n = len(texts)
    col_pos = Y.sum(axis=0).astype(int) if n else np.zeros(len(policies), int)
    n_benign = int((np.asarray(corpus["is_harmful"]) == 0).sum()) if n else 0
    n_harmful = n - n_benign

    per_col = {p["id"]: int(col_pos[i]) for i, p in enumerate(policies)}
    req = int(requested_n_per_class)
    rva = {pid: {"requested": req, "achieved": got, "shortfall": bool(got < req)}
           for pid, got in per_col.items()}

    out = {
        "n_rows": int(n),
        "n_policies": int(len(policies)),
        "per_column_positives": per_col,
        "n_harmful": int(n_harmful),
        "n_benign_achieved": int(n_benign),
        "n_benign_requested": int(requested_n_benign),
        "benign_shortfall": bool(n_benign < int(requested_n_benign)),
        "class_balance_harmful_frac": float(n_harmful / n) if n else 0.0,
        "source_distribution": {s: int(c) for s, c in
                                sorted(Counter(corpus["sources"]).items())},
        "requested_vs_achieved": rva,
        "n_columns_short": int(sum(1 for v in rva.values() if v["shortfall"])),
        "pool_fingerprint": pool_fingerprint(texts),
    }
    if test_idx is not None:
        ti = np.asarray(test_idx, dtype=np.int64)
        Yt = Y[ti] if len(ti) else Y[:0]
        tcol = Yt.sum(axis=0).astype(int) if len(ti) else np.zeros(len(policies), int)
        out["per_column_positives_test"] = {p["id"]: int(tcol[i])
                                            for i, p in enumerate(policies)}
        out["n_test_rows"] = int(len(ti))
        out["n_benign_test"] = int((np.asarray(corpus["is_harmful"])[ti] == 0).sum()) \
            if len(ti) else 0
    return out


def format_shortfall(achieved: dict) -> str:
    """LOUD ASCII warning when requested != achieved (its ONE job). Never raises.

    A warning, not a failure: a pool-limited column is legitimate under CLAUDE.md
    section 17 rule 2 -- what is NOT legitimate is failing to say so. So we print it
    every run, at a size that cannot be skimmed past.
    """
    short = [(pid, v) for pid, v in sorted(achieved.get("requested_vs_achieved", {}).items())
             if v["shortfall"]]
    lines = []
    if short or achieved.get("benign_shortfall"):
        lines.append("=" * 72)
        lines.append("!! DATA SHORTFALL -- requested != achieved (rubric rule 1) !!")
        for pid, v in short:
            lines.append("!!   %-26s requested %5d  achieved %5d"
                         % (pid, v["requested"], v["achieved"]))
        if achieved.get("benign_shortfall"):
            lines.append("!!   %-26s requested %5d  achieved %5d"
                         % ("(benign negatives)", achieved.get("n_benign_requested", 0),
                            achieved.get("n_benign_achieved", 0)))
        lines.append("!! %d of %d policy columns are under the requested floor."
                     % (achieved.get("n_columns_short", 0), achieved.get("n_policies", 0)))
        lines.append("!! results.json records the ACHIEVED counts under 'achieved'.")
        lines.append("=" * 72)
    else:
        lines.append("[data] requested == achieved on every column and on the benign side.")
    lines.append("[data] class balance: harmful=%d benign=%d (harmful frac %.3f)"
                 % (achieved.get("n_harmful", 0), achieved.get("n_benign_achieved", 0),
                    achieved.get("class_balance_harmful_frac", 0.0)))
    lines.append("[data] pool_fingerprint=%s" % achieved.get("pool_fingerprint", "?")[:16])
    return "\n".join(lines)


# --- seen / held-out policy split (the zero-shot test) -----------------------
def split_seen_heldout(corpus, n_heldout=C.N_HELDOUT_POLICIES, seed=C.SEED) -> dict:
    """Choose n_heldout policy COLUMNS to withhold from training (its ONE job).

    The held-out policies are NEVER trained on; the bi/uni encoders must detect them
    zero-shot from the description alone -- the lesson's headline. We only pick
    held-out columns that still have >= ~200 positives so the zero-shot test is REAL
    (a starved column would give a meaningless AP). Returns
    {"seen_cols":[int], "heldout_cols":[int]}.
    """
    rng = random.Random(seed + 1)
    Y = corpus["Y"]
    P = Y.shape[1]
    col_pos = Y.sum(axis=0).astype(int)
    # candidates = well-populated columns (>=200 positives, or the top ones if data is thin)
    eligible = [c for c in range(P) if col_pos[c] >= 200]
    if len(eligible) < n_heldout:           # thin-data fallback: take the most populated
        eligible = sorted(range(P), key=lambda c: -col_pos[c])[:max(n_heldout, len(eligible))]
    rng.shuffle(eligible)
    heldout = sorted(eligible[:n_heldout])
    seen = [c for c in range(P) if c not in heldout]
    names = [corpus["policies"][c]["id"] for c in heldout]
    _eprint("[data] held-out (zero-shot) policies: %s  (pos=%s)"
            % (names, [int(col_pos[c]) for c in heldout]))
    return {"seen_cols": seen, "heldout_cols": heldout}


# --- group-aware train/test split --------------------------------------------
def group_train_test(corpus, test_frac=0.3, seed=C.SEED):
    """Group-aware row split on `groups` (its ONE job: no text leakage across split).

    All rows sharing a group id (same underlying prompt / near-dup) go to the SAME
    side, so a prompt cannot appear in both train and test. Returns (train_idx,
    test_idx) as int numpy arrays of ROW indices.
    """
    rng = random.Random(seed + 2)
    groups = corpus["groups"]
    uniq = sorted(set(groups))
    rng.shuffle(uniq)
    n_test = int(round(len(uniq) * test_frac))
    test_groups = set(uniq[:n_test])
    train_idx = [i for i, g in enumerate(groups) if g not in test_groups]
    test_idx = [i for i, g in enumerate(groups) if g in test_groups]
    _eprint("[data] group split: %d train rows / %d test rows (%d/%d groups)"
            % (len(train_idx), len(test_idx), len(uniq) - n_test, n_test))
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


# --- transfer arms: three DIFFERENT things "OOD" could mean -------------------
# AUDIT_2026-08.md section A4: what this lesson called OOD was BeaverTails/30k_test --
# the same dataset, annotators, taxonomy and prompt+response rendering as 93.5% of
# train. Only the ROWS changed. It is kept (it is a real held-out check) but is now
# named `heldout_split`, and two arms where something actually shifts sit beside it.
def load_heldout_split(seed=C.SEED) -> dict:
    """BeaverTails 30k_test: a HELD-OUT SPLIT of the training dataset. NOT OOD.

    Same annotators, same 14-way taxonomy, same prompt+response rendering as the
    BeaverTails rows in train -- only the rows differ. It measures split transfer,
    which is worth measuring and is not distribution transfer. jailbreak/toxicity
    stay 0 here (BeaverTails has no such labels), so it is scored over the core
    columns.
    """
    from datasets import load_dataset as hf_load

    rng = random.Random(seed + 3)
    policies = build_taxonomy()
    P = len(policies)
    col_of = {p["id"]: i for i, p in enumerate(policies)}
    store = {"seen": set(), "texts": [], "Y": [], "sources": [], "gkeys": []}
    n_pos = n_ben = 0
    cap = 1500                              # bounded slice (screening tier)
    src = "BeaverTails/%s" % C.BEAVERTAILS_TEST_SPLIT
    try:
        ds = hf_load(C.BEAVERTAILS_DATASET, split=C.BEAVERTAILS_TEST_SPLIT,
                     streaming=True)
    except Exception as e:
        _eprint("[transfer] %s load FAILED (%s)" % (src, e))
        return _finalize(store, policies, rng) | {
            "source": "none", "n": 0, "arm": "heldout_split", "error": str(e)}
    for row in ds:
        cats = row.get("category") or {}
        active = [_BEAVERTAILS_KEY_TO_ID[k] for k, v in cats.items()
                  if v and k in _BEAVERTAILS_KEY_TO_ID]
        text = _clean(row.get("prompt", "")) + "\n" + _clean(row.get("response", ""))
        gkey = _norm_key(row.get("prompt", ""))
        if active and n_pos < cap:
            if _add_row(store, text, active, "beavertails_test", gkey, col_of, P):
                n_pos += 1
        elif row.get("is_safe", False) and n_ben < cap:
            if _add_row(store, text, [], "beavertails_test_benign", gkey, col_of, P):
                n_ben += 1
        if n_pos >= cap and n_ben >= cap:
            break
    out = _finalize(store, policies, rng)
    out["source"] = src
    out["arm"] = "heldout_split"
    out["shift"] = "rows only -- same dataset, annotators, taxonomy and rendering"
    out["n"] = len(out["texts"])
    _eprint("[transfer/heldout_split] %s: N=%d harmful=%d benign=%d"
            % (out["source"], out["n"], n_pos, n_ben))
    return out


def load_cross_annotator(seed=C.SEED) -> dict:
    """Aegis 2.0's HELD-OUT TEST split: a different annotation regime, same rendering.

    Genuinely cross-annotator: Aegis' labels come from a different taxonomy and a
    different labelling process than BeaverTails', so agreement here is a real
    transfer question rather than a resampling question. Rows whose Aegis categories
    do not map are dropped and counted, exactly as in training (see _load_aegis).
    """
    rng = random.Random(seed + 4)
    policies = build_taxonomy()
    P = len(policies)
    col_of = {p["id"]: i for i, p in enumerate(policies)}
    store = {"seen": set(), "texts": [], "Y": [], "sources": [], "gkeys": []}
    cap = 1500
    stats = _load_aegis(store, cap, cap, col_of, P, rng, split=C.AEGIS_TEST_SPLIT)
    out = _finalize(store, policies, rng)
    out["source"] = "%s/%s" % (C.AEGIS_DATASET, C.AEGIS_TEST_SPLIT)
    out["arm"] = "cross_annotator"
    out["shift"] = "annotators + taxonomy -- same prompt+response rendering"
    out["n"] = len(out["texts"])
    out["aegis"] = stats
    if "error" in stats:
        out["error"] = stats["error"]
    _eprint("[transfer/cross_annotator] %s: N=%d" % (out["source"], out["n"]))
    return out


def _cstm_session_text(session) -> str:
    """Flatten one CSTM-Bench session dict into one readable string."""
    if not isinstance(session, dict):
        return _clean(session)
    parts = []
    anchor = session.get("identity_anchor")
    if isinstance(anchor, str) and anchor.strip():
        parts.append(_clean(anchor))
    for key in ("messages", "content", "text", "turns", "conversation"):
        val = session.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(_clean(val))
        elif isinstance(val, list):
            for m in val:
                if isinstance(m, str) and m.strip():
                    parts.append(_clean(m))
                elif isinstance(m, dict):
                    for mk in ("content", "text", "message"):
                        mv = m.get(mk)
                        if isinstance(mv, str) and mv.strip():
                            parts.append(_clean(mv))
    return " ".join(parts).strip()


def load_ood_benchmark(seed=C.SEED) -> dict:
    """CSTM-Bench (intrinsec-ai/cstm-bench): a RELEASED external benchmark as OOD.

    The benchmark CLAUDE.md section 17 rule 8 names for this lesson family, and the
    only arm here where the task shape itself changes: rows are multi-session attack
    scenarios, not single moderated messages.

    Mapping, stated plainly because it is a CHOICE and not a given:
      * one row per SCENARIO, text = its sessions concatenated, capped at
        C.CSTM_MAX_CHARS so one scenario cannot dominate the length distribution;
      * scenario_class == 'attack'  -> the `jailbreak` column (the scenarios are
        prompt-injection / role-override attacks, which is what that policy names);
        anything else -> all-zero (benign).
    THREE HONEST LIMITS. (1) The label is SCENARIO-level, so a benign session inside
    an attack scenario is folded into a positive. (2) Only the `jailbreak` column and
    the binary harmful/benign score are meaningful; the other columns are all-zero
    by construction and must not be read as "the model missed them". (3) n is ~108,
    which is a screening-tier read on a genuinely external distribution, not an
    evaluation-tier number.
    """
    from datasets import load_dataset as hf_load
    import json as _json

    rng = random.Random(seed + 5)
    policies = build_taxonomy()
    P = len(policies)
    col_of = {p["id"]: i for i, p in enumerate(policies)}
    store = {"seen": set(), "texts": [], "Y": [], "sources": [], "gkeys": []}
    n_att = n_ben = 0
    errors = []
    for split in C.CSTM_SPLITS:
        try:
            ds = hf_load(C.CSTM_DATASET, "default", split=split)
        except Exception as e:
            _eprint("[transfer/ood_benchmark] split %s load FAILED (%s)" % (split, e))
            errors.append("%s: %s" % (split, e))
            continue
        for row in ds:
            raw = row.get("sessions_json")
            try:
                sessions = _json.loads(raw) if isinstance(raw, str) else (raw or [])
            except Exception:
                sessions = []
            parts = [t for t in (_cstm_session_text(s) for s in sessions) if t]
            if not parts:
                continue
            text = " ".join(parts)[:int(C.CSTM_MAX_CHARS)]
            attack = str(row.get("scenario_class", "")) == "attack"
            labels = ["jailbreak"] if attack else []
            gkey = _norm_key(str(row.get("scenario_id", "")) or text[:200])
            if _add_row(store, text, labels, "cstm_%s" % split, gkey, col_of, P):
                n_att += int(attack)
                n_ben += int(not attack)
    out = _finalize(store, policies, rng)
    out["source"] = C.CSTM_DATASET
    out["arm"] = "ood_benchmark"
    out["shift"] = "external benchmark -- different corpus, task shape and label source"
    out["n"] = len(out["texts"])
    out["scored_columns"] = ["jailbreak"]
    out["label_granularity"] = "scenario-level"
    if errors:
        out["error"] = "; ".join(errors)
    _eprint("[transfer/ood_benchmark] %s: N=%d attack=%d benign=%d"
            % (out["source"], out["n"], n_att, n_ben))
    return out


_TRANSFER_LOADERS = {
    "heldout_split": load_heldout_split,
    "cross_annotator": load_cross_annotator,
    "ood_benchmark": load_ood_benchmark,
}


def load_transfer_arms(arms=None, seed=C.SEED) -> dict:
    """Load every configured transfer arm (its ONE job: name each shift honestly).

    Returns {arm_name: corpus-shaped dict}. A failing arm records its error and does
    not take the others down with it.
    """
    names = list(arms if arms is not None else C.TRANSFER_ARMS)
    out = {}
    for name in names:
        fn = _TRANSFER_LOADERS.get(name)
        if fn is None:
            _eprint("[transfer] unknown arm '%s' -- skipped" % name)
            continue
        try:
            out[name] = fn(seed=seed)
        except Exception as exc:
            _eprint("[transfer] arm %s FAILED: %s" % (name, exc))
            out[name] = {"arm": name, "source": "?", "n": 0, "error": str(exc),
                         "texts": [], "Y": np.zeros((0, len(build_taxonomy())), np.float32),
                         "is_harmful": np.zeros(0, np.int64)}
    return out


# Back-compat: `load_ood` was the BeaverTails held-out split all along. The alias
# keeps older callers working while the honest name is what new code uses.
load_ood = load_heldout_split


# --- confound audit (delegated to the ONE shared implementation) --------------
# There used to be a local `confound_report` here. It ran ONE bar (character
# length), returned the RAW auc without folding it about 0.5, and had no count bar,
# no content bar and no shuffle control -- one of four partial reimplementations
# across the detection lessons. `steering_tutorials/common/confound.py` is now the
# single implementation; this module only adapts the call and keeps the three legacy
# keys so an older reader of `results.json` still finds what it expects.
def confound_report(texts, is_harmful, units=None, seed=C.SEED, n_folds=5,
                    run_content=True, run_shuffle=True) -> dict:
    """Run the SHARED four-bar confound audit over this corpus (its ONE job).

    Four bars, because a length-only audit is not sufficient (CLAUDE.md section 17
    rule 7): character LENGTH, unit COUNT (word count here, so a token-count tell
    cannot hide behind a clean char-count), a TF-IDF CONTENT bar under 5-fold CV
    (a policy-matching guard that cannot beat unigrams is not matching policies),
    and a label-SHUFFLE control (a leakage diagnostic, never a bar to clear).

    Every bar is DIRECTIONLESS -- `max(auc, 1-auc)` -- because a feature that
    predicts the benign class perfectly is exactly as damning as one that predicts
    the harmful class perfectly.

    Returns the shared report (`length`/`count`/`content`/`shuffle`/`worst_auc`/
    `worst_name`) PLUS the three legacy keys `length_auc`, `len_pos_mean`,
    `len_neg_mean`. `length_auc` is the FOLDED value; the unfolded one is at
    `length.auc_raw`.
    """
    labels = [int(y) for y in is_harmful]
    if units is None:                       # word count: a second, non-char length tell
        units = [str(t).split() for t in texts]
    rep = _shared_confound_report(texts, labels, units=units, seed=int(seed),
                                  n_folds=int(n_folds), run_content=bool(run_content),
                                  run_shuffle=bool(run_shuffle))
    length = rep.get("length", {})
    rep["length_auc"] = float(length.get("auc", 0.5))          # folded (legacy key)
    rep["length_auc_raw"] = float(length.get("auc_raw", 0.5))  # unfolded
    rep["len_pos_mean"] = float(length.get("mean_pos", 0.0))
    rep["len_neg_mean"] = float(length.get("mean_neg", 0.0))
    return rep


# --- CPU smoke: python -m steering_tutorials.biencoder_guard.data ------------
if __name__ == "__main__":
    # SMALL caps so this runs on CPU without loading any model. ASCII stdout.
    pols = build_taxonomy()
    print("TAXONOMY: P=%d policies" % len(pols))
    ex = pols[0]
    print("policy[0] id=%s name=%s" % (ex["id"], ex["name"]))
    print("  description: %s" % ex["description"])
    print("  paraphrases(%d): %s" % (len(ex["paraphrases"]), ex["paraphrases"][0]))
    print("  ids:", ", ".join(p["id"] for p in pols))

    print("\n[smoke] load_corpus(n_per_class=40, n_benign=40) ...")
    corpus = load_corpus(n_per_class=40, n_benign=40)
    Y = corpus["Y"]
    print("CORPUS: N=%d  P=%d  harmful=%d  benign=%d"
          % (len(corpus["texts"]), Y.shape[1] if Y.size else len(pols),
             int(corpus["is_harmful"].sum()),
             int((corpus["is_harmful"] == 0).sum())))

    split = split_seen_heldout(corpus)
    print("SEEN cols=%d  HELDOUT cols=%s" % (len(split["seen_cols"]), split["heldout_cols"]))

    tr, te = group_train_test(corpus)
    print("SPLIT: train=%d test=%d" % (len(tr), len(te)))

    print("\n" + format_shortfall(corpus["achieved"]))
    print("source_distribution: %s" % corpus["achieved"]["source_distribution"])

    conf = confound_report(corpus["texts"], corpus["is_harmful"])
    print("\n" + format_confound_report(conf))
    print("legacy keys: length_auc=%.4f (raw %.4f) len_pos_mean=%.1f len_neg_mean=%.1f"
          % (conf["length_auc"], conf["length_auc_raw"],
             conf["len_pos_mean"], conf["len_neg_mean"]))

    if corpus["texts"]:
        i = 0
        print("sample[0] harmful=%d cols=%s text=%s"
              % (int(corpus["is_harmful"][i]),
                 list(np.nonzero(Y[i])[0]),
                 corpus["texts"][i][:90]))
