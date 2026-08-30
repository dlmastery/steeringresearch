"""data.py -- an instruction-vs-data corpus whose ROLE label is a CONSTRUCTION
decision, never a property read out of the text.

WHY ROLE CANNOT COME FROM THE TEXT
-----------------------------------
aside.py's `rotate_embeddings` takes an explicit boolean mask precisely because
`arXiv:2606.27567` (`inseparability.py`) proves an attacker-visible role channel
is no channel at all: if role were inferred from a delimiter or a register cue
inside the string, an attacker who controls the string forges the cue. The only
sound role channel is one the APPLICATION decides out of band -- e.g. "this text
arrived as the direct chat instruction" vs "this text arrived embedded in a
retrieved document / tool output". This module manufactures exactly that
decision in the loader, not in the strings:

  INSTRUCTION role (is_data=False) -- HuggingFaceH4/ultrachat_200k user turns.
      A real user's direct chat request: the textbook TRUSTED channel.
  DATA role (is_data=True) -- lmsys/toxic-chat + JailbreakBench/JBB-Behaviors
      prompts, assigned wholesale to the untrusted/retrieved channel. Crucially
      this pool is BOTH harmful and benign content (`common.data.load_harmful_benign`
      draws roughly half and half): the data channel is not a synonym for
      "harmful" -- a retrieved document or tool output is usually innocuous, and
      conflating "data role" with "harmful content" would make the two axes
      (channel vs. harm) look like one, which they are not.

THE HONEST CONFOUND
--------------------
Both pools are chat-register short-to-medium English text, but they are NOT the
same distribution: ultrachat openers are long, detailed, instructive requests;
toxic-chat/JBB prompts are shorter, blunter, imperative "Goal" sentences. Any
separability this lesson measures downstream is real separability of THESE two
concrete pools, not a proof that role-in-general is separable from surface form
alone -- `inseparability.py`'s same-distribution floor exists exactly to price
that gap, and `separability.py` reports it beside every AUC. Do not read a high
separability number here as "instructions and data always look different" -- see
README.md Section on limitations.

CPU-only. Downloads datasets via the `datasets` library (network + local HF
cache only). Loads NO model, NO tokenizer. ASCII stdout (Windows cp1252).
"""
from __future__ import annotations

import sys
from pathlib import Path

try:  # SSL middlebox on this host (truststore lets requests use the OS trust store)
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

from steering_tutorials.common import dataset_export as DE
from steering_tutorials.common.data import load_harmful_benign
from steering_tutorials.control_data_split import config as C

__all__ = ["load_role_corpus", "corpus_manifest"]

_HERE = Path(__file__).resolve().parent


def _eprint(*args) -> None:
    """ASCII-only stderr print (Windows cp1252 console safe)."""
    msg = " ".join(str(a) for a in args)
    try:
        print(msg, file=sys.stderr)
    except Exception:
        print(msg.encode("ascii", "replace").decode("ascii"), file=sys.stderr)


def _clean(text) -> str:
    return " ".join(str(text).split()).strip()


# --- INSTRUCTION role: ultrachat_200k user turns ------------------------------
def _load_instruction_texts(n_needed: int, seed: int) -> list:
    """Stream UltraChat; each conversation's FIRST user turn = one instruction.

    One turn per conversation (not every turn in it) so the pool is not
    dominated by a few long multi-turn dialogues, and so `group_id` below is
    naturally 1:1 with `text` -- no within-conversation near-duplicate risk.
    """
    from datasets import load_dataset as hf_load

    rows = []
    max_scan = max(5000, n_needed * 6)  # cap so a run cannot stream all 200k rows
    stream = hf_load(C.HF_INSTRUCTION_SOURCE, split="train_sft", streaming=True)
    for i, row in enumerate(stream):
        if i >= max_scan or len(rows) >= n_needed:
            break
        msgs = row.get("messages") or []
        first_user = next((m.get("content", "") for m in msgs
                           if m.get("role") == "user"), "")
        text = _clean(first_user)
        if len(text) < 10:
            continue
        rows.append({"text": text, "group": "ultrachat_%d" % i,
                     "_source": C.HF_INSTRUCTION_SOURCE, "_source_tag": "ultrachat"})
    return rows


# --- DATA role: harmful+benign pool from the shared toxic-chat/JBB loader ----
def _load_data_texts(n_per_role_side: int, seed: int) -> list:
    """Reuse `common.data.load_harmful_benign` (lmsys/toxic-chat + JBB top-up,
    dedup + length-matched, already vetted redistributable). BOTH its classes --
    harmful and benign -- are assigned to the DATA role: the channel decision is
    orthogonal to whether the content happens to be harmful.
    """
    rec = load_harmful_benign(n_per_class=n_per_role_side, seed=seed)
    rows = []
    for i, text in enumerate(rec["harmful"]):
        rows.append({"text": _clean(text), "group": "data_harmful_%d" % i,
                     "_source": "lmsys/toxic-chat+JailbreakBench/JBB-Behaviors",
                     "_source_tag": "data_harmful"})
    for i, text in enumerate(rec["benign"]):
        rows.append({"text": _clean(text), "group": "data_benign_%d" % i,
                     "_source": "lmsys/toxic-chat+JailbreakBench/JBB-Behaviors",
                     "_source_tag": "data_benign"})
    return rows


# --- public API ----------------------------------------------------------------
def load_role_corpus(n_per_role: int = C.N_PER_ROLE, seed: int = C.SEED) -> dict:
    """Build the instruction-vs-data corpus. Role is assigned HERE, by which
    loader a row came from -- never parsed out of `text`.

    Returns
    -------
    dict with:
      texts       : list[str]                (len = 2 * n_per_role, roughly)
      is_data     : list[bool]                (the role label; True = data channel)
      groups      : list[str]                 (dedup/leakage-safe unit id)
      sources     : list[str]                 (which HF dataset the row came from)
      source_tags : list[str]                 ("ultrachat" / "data_harmful" / "data_benign")
      n_instruction, n_data : int
      requested_n_per_role : int
    """
    instr_rows = _load_instruction_texts(n_per_role, seed)
    # split n_per_role roughly in half between harmful/benign so the DATA role
    # totals ~n_per_role, matching the INSTRUCTION role's count.
    half = max(1, n_per_role // 2)
    data_rows = _load_data_texts(half, seed)

    texts, is_data, groups, sources, source_tags = [], [], [], [], []
    for r in instr_rows:
        texts.append(r["text"]); is_data.append(False); groups.append(r["group"])
        sources.append(r["_source"]); source_tags.append(r["_source_tag"])
    for r in data_rows:
        texts.append(r["text"]); is_data.append(True); groups.append(r["group"])
        sources.append(r["_source"]); source_tags.append(r["_source_tag"])

    n_instruction = sum(1 for d in is_data if not d)
    n_data = sum(1 for d in is_data if d)
    if n_instruction < n_per_role:
        _eprint("[data] WARNING: instruction role got %d < requested %d"
                % (n_instruction, n_per_role))
    if n_data < n_per_role:
        _eprint("[data] WARNING: data role got %d < requested %d (half-split of %d)"
                % (n_data, n_per_role, n_per_role))

    return {
        "texts": texts, "is_data": is_data, "groups": groups,
        "sources": sources, "source_tags": source_tags,
        "n_instruction": n_instruction, "n_data": n_data,
        "requested_n_per_role": n_per_role, "seed": seed,
    }


def corpus_manifest(corpus: dict) -> dict:
    """Licence + fingerprint for the corpus, WITHOUT writing any file.

    Every source here is already in `dataset_export.REDISTRIBUTABLE`
    (asserted below, not assumed), so this never raises for our own sources --
    it raises loudly if a future edit adds an unvetted one.
    """
    rows = [{"text": t, "is_data": bool(d), "group": g, "source": s, "source_tag": st}
            for t, d, g, s, st in zip(corpus["texts"], corpus["is_data"],
                                      corpus["groups"], corpus["sources"],
                                      corpus["source_tags"])]
    licences = {}
    for src in (C.HF_INSTRUCTION_SOURCE,) + C.HF_DATA_SOURCES:
        licences[src] = DE.assert_redistributable(src)
    fp = DE.slice_fingerprint(rows)
    return {
        "n_rows": len(rows), "n_instruction": corpus["n_instruction"],
        "n_data": corpus["n_data"], "seed": corpus["seed"],
        "requested_n_per_role": corpus["requested_n_per_role"],
        "licences": licences, "slice_fingerprint": fp,
    }


def _self_test() -> None:
    """SYNTHETIC self-test (CDS_SELFTEST=1): no network, no HF, no model.

    Checks the CONSTRUCTION property this module exists to guarantee: role is a
    function of WHICH LOADER a row came from, and is recoverable even after the
    two pools are shuffled together -- i.e. it behaves exactly like the boolean
    mask `aside.rotate_embeddings` expects, and like the trusted/untrusted split
    `inseparability.estimate_provenance_bound` measures.
    """
    import random

    rng = random.Random(0)
    fake_instr = [{"text": "Please write a %d-word summary of X." % i,
                  "group": "instr_%d" % i, "_source": "fake-instr",
                  "_source_tag": "ultrachat"} for i in range(20)]
    fake_data = [{"text": "Document %d contains some retrieved content." % i,
                 "group": "data_%d" % i, "_source": "fake-data",
                 "_source_tag": "data_benign"} for i in range(20)]

    texts, is_data, groups = [], [], []
    for r in fake_instr:
        texts.append(r["text"]); is_data.append(False); groups.append(r["group"])
    for r in fake_data:
        texts.append(r["text"]); is_data.append(True); groups.append(r["group"])

    order = list(range(len(texts)))
    rng.shuffle(order)
    texts = [texts[i] for i in order]
    is_data = [is_data[i] for i in order]
    groups = [groups[i] for i in order]

    assert sum(is_data) == 20 and sum(not d for d in is_data) == 20
    print("OK  role counts survive a shuffle (20 instruction / 20 data)")

    # role is recoverable from GROUP, not from any substring of TEXT ("Document",
    # "summary" etc. are illustrative register cues here, not what the label is
    # keyed on) -- the group prefix stands in for the out-of-band channel decision.
    recovered = ["data_" in g for g in groups]
    assert recovered == is_data
    print("OK  is_data matches the GROUP-derived (out-of-band) assignment exactly")

    rows = [{"text": t, "is_data": bool(d), "group": g}
            for t, d, g in zip(texts, is_data, groups)]
    fp1 = DE.slice_fingerprint(rows)
    fp2 = DE.slice_fingerprint(list(reversed(rows)))
    assert fp1 == fp2
    print("OK  corpus fingerprint is order-independent (%s)" % fp1)

    fp3 = DE.slice_fingerprint(rows[:-1])
    assert fp3 != fp1
    print("OK  fingerprint is content-sensitive (dropping one row changes it)")

    for src in ("HuggingFaceH4/ultrachat_200k", "lmsys/toxic-chat",
                "JailbreakBench/JBB-Behaviors"):
        lic = DE.assert_redistributable(src)
        assert lic != "UNKNOWN"
    print("OK  all three real sources this lesson uses are in REDISTRIBUTABLE "
          "(instruction=ultrachat_200k; data=toxic-chat+JBB-Behaviors)")

    try:
        DE.assert_redistributable("walledai/HarmBench")
    except SystemExit as exc:
        assert "REFUSING" in str(exc)
        print("OK  a GATED source (not used by this lesson) is still refused by "
              "the shared gate -- confirms the gate applies here too")
    else:
        raise AssertionError("gated source was not refused")

    print("")
    print("OK -- data.py: role is a construction-time, out-of-band label; "
          "recoverable after shuffling, fingerprinted, and licence-gated.")


if __name__ == "__main__":
    import os

    if os.environ.get("CDS_SELFTEST") == "1":
        _self_test()
    else:
        c = load_role_corpus()
        print("[data] instruction=%d data=%d (requested %d/role)"
              % (c["n_instruction"], c["n_data"], c["requested_n_per_role"]))
        man = corpus_manifest(c)
        print("[data] fingerprint=%s licences=%s"
              % (man["slice_fingerprint"], man["licences"]))
