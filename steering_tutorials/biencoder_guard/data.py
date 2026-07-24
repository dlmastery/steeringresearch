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
THREE complementary ones into ONE unified multi-label corpus over a shared
taxonomy:

  * BeaverTails (PKU-Alignment)  -- 14 fine-grained, MULTI-LABEL harm categories on
    prompt+response pairs. This is the CORE of our label space: thousands of
    positives per common category, and the safe rows give topically-adjacent
    benign hard-negatives.
  * toxic-chat (lmsys)           -- REAL user prompts hand-labelled for `toxicity`
    and `jailbreaking`. Supplies in-the-wild toxic + adversarial positives and
    benign hard-negatives that look adversarial but are not.
  * wildguardmix (allenai)       -- adversarial prompt-harm labels (gated; we SKIP
    it gracefully if it will not load -- BeaverTails + toxic-chat already clear
    the rubric).

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

# Fixed column order: the 14 BeaverTails core policies, then jailbreak, then toxicity.
_CORE_IDS = list(_BEAVERTAILS_KEY_TO_ID.values())
_POLICY_ORDER = _CORE_IDS + ["jailbreak", "toxicity"]
_POLICY_GROUP = {**{i: "beavertails" for i in _CORE_IDS},
                 "jailbreak": "adversarial", "toxicity": "toxicity"}


def build_taxonomy() -> list:
    """Return the unified many-label policy taxonomy (its ONE job: define the label
    columns). Each entry is a Policy dict {id,name,description,paraphrases,group}.
    The column ORDER here is the column order of every Y matrix in the corpus, so it
    must stay stable. WHY paraphrases: the policy tower averages a description plus
    its paraphrases into a robust multi-prototype vector (the schema-expansion idea).
    """
    policies = []
    for pid in _POLICY_ORDER:
        desc, paras = _POLICY_SPEC[pid]
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
        stream = hf_load(C.BEAVERTAILS_DATASET, split="30k_train", streaming=True)
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

    # BENIGN = LENGTH-MATCHED. BeaverTails safe rows (prompt+response) share the
    # positives' length distribution, so we fill the benign quota from BeaverTails
    # FIRST and only backfill any deficit from the shorter, prompt-only toxic-chat /
    # wildguard benigns -- this keeps length from leaking the label (length_auc ~0.5).
    # All three sources still contribute their POSITIVES regardless of the benign cap.
    bt_benign = _load_beavertails(store, n_per_class, n_benign, col_of, P, rng)
    deficit = max(0, n_benign - bt_benign)
    tc_benign = _load_toxicchat(store, n_per_class, deficit, col_of, P, rng)
    deficit = max(0, deficit - tc_benign)
    _load_wildguard(store, n_per_class, deficit, col_of, P, rng)

    corpus = _finalize(store, policies, rng)

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
    return corpus


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


# --- OOD slice ---------------------------------------------------------------
def load_ood(seed=C.SEED) -> dict:
    """Load a DISJOINT OOD slice in the same corpus shape (the real generalization check).

    Source = BeaverTails '30k_test' (a split never seen in load_corpus's '30k_train'),
    mapped over the SAME taxonomy. jailbreak/toxicity columns stay 0 here (BeaverTails
    has no such labels) -- OOD is scored over the CORE/seen columns. Returns the same
    dict shape; notes source + counts.
    """
    from datasets import load_dataset as hf_load

    rng = random.Random(seed + 3)
    policies = build_taxonomy()
    P = len(policies)
    col_of = {p["id"]: i for i, p in enumerate(policies)}
    store = {"seen": set(), "texts": [], "Y": [], "sources": [], "gkeys": []}
    n_pos = n_ben = 0
    cap = 1500                              # bounded OOD slice (screening tier)
    try:
        ds = hf_load(C.BEAVERTAILS_DATASET, split="30k_test", streaming=True)
    except Exception as e:
        _eprint("[ood] BeaverTails 30k_test load FAILED (%s)" % e)
        return _finalize(store, policies, rng) | {"source": "none", "n": 0}
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
    ood = _finalize(store, policies, rng)
    ood["source"] = "BeaverTails/30k_test"
    ood["n"] = len(ood["texts"])
    _eprint("[ood] %s: N=%d harmful=%d benign=%d"
            % (ood["source"], ood["n"], n_pos, n_ben))
    return ood


# --- confound audit ----------------------------------------------------------
def _auc(scores, labels) -> float:
    """Threshold-free AUC = P(score(pos) > score(neg)); sklearn with a rank fallback."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        ranks = [0.0] * len(scores)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for m in range(i, j + 1):
                ranks[order[m]] = avg
            i = j + 1
        n_p, n_n = len(pos), len(neg)
        sum_pos = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
        u = sum_pos - n_p * (n_p + 1) / 2.0
        return float(u / (n_p * n_n))


def confound_report(texts, is_harmful) -> dict:
    """Can raw CHAR LENGTH separate harmful vs benign? (its ONE job: a length-tell audit).

    A guard must read INTENT, not length. length_auc ~0.5 => no trivial tell; far from
    0.5 => the classes are length-separable and a "win" might be a length artifact.
    Returns {"length_auc","len_pos_mean","len_neg_mean"}.
    """
    lens = [len(t) for t in texts]
    labels = [int(y) for y in is_harmful]
    pos = [n for n, y in zip(lens, labels) if y == 1]
    neg = [n for n, y in zip(lens, labels) if y == 0]
    return {
        "length_auc": _auc(lens, labels),
        "len_pos_mean": float(sum(pos) / len(pos)) if pos else 0.0,
        "len_neg_mean": float(sum(neg) / len(neg)) if neg else 0.0,
    }


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

    conf = confound_report(corpus["texts"], corpus["is_harmful"])
    print("CONFOUND: length_auc=%.4f  len_pos_mean=%.1f  len_neg_mean=%.1f"
          % (conf["length_auc"], conf["len_pos_mean"], conf["len_neg_mean"]))

    if corpus["texts"]:
        i = 0
        print("sample[0] harmful=%d cols=%s text=%s"
              % (int(corpus["is_harmful"][i]),
                 list(np.nonzero(Y[i])[0]),
                 corpus["texts"][i][:90]))
