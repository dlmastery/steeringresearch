"""Pure-function unit tests for the W1/W2 generation instruments (eval.py).

These cover the non-circular behavior scorer's building blocks and the rule-based
safety/selectivity judges WITHOUT needing a real model: every function under test
is a deterministic pure function over strings / sequences. The one model-touching
case (generation_behavior_scorer on FakeLM) asserts the documented fallback tag.
"""

import torch

from steering.eval import (
    _word_stems,
    compliance_rate,
    concept_rate,
    fake_safety_responses,
    generation_behavior_scorer,
    is_refusal,
    lexicon_from_pairs,
    refusal_rate,
    selectivity,
)
from steering.model import load_model


# --- lexicon_from_pairs (W1) ------------------------------------------------
def test_lexicon_from_pairs_ranks_pos_distinctive_words():
    pairs = [
        ("the ocean waves crash on the coral reef", "the desert sand is dry and hot"),
        ("a deep blue ocean wave rolls toward shore", "a barren rocky mountain trail"),
    ]
    lex = lexicon_from_pairs(pairs, top_k=8)
    # Concept (ocean) words distinctive of the POSITIVE side must rank in.
    assert "ocean" in lex
    assert "coral" in lex
    # A word that appears only on the NEGATIVE side must NOT be in the lexicon.
    assert "desert" not in lex
    assert "sand" not in lex
    assert "mountain" not in lex


def test_lexicon_excludes_shared_and_stopwords():
    pairs = [("the bright ocean here", "the bright desert here")]
    lex = lexicon_from_pairs(pairs, top_k=10)
    # 'bright' appears on BOTH sides -> net 0 -> excluded.
    assert "bright" not in lex
    # 'the'/'here' are stop-words -> never counted.
    assert "the" not in lex and "here" not in lex
    # The distinctive positive word survives.
    assert "ocean" in lex


# --- _word_stems ------------------------------------------------------------
def test_word_stems_strips_suffixes():
    # The stemmer strips ONE trailing -ing / -ed / -es / -s suffix (crude, by
    # design — only needs to collapse common surface forms, not be a real stemmer).
    assert _word_stems("rolling") == ["roll"]   # -ing stripped
    assert _word_stems("jumped") == ["jump"]    # -ed stripped
    assert _word_stems("cats") == ["cat"]       # plural -s stripped
    assert _word_stems("dogs") == ["dog"]
    # stop-words and <3-char tokens are dropped
    assert _word_stems("the of a an to") == []


def test_word_stems_keeps_content_words_formerly_stoplisted():
    # The curated stop-list no longer suppresses these content words.
    stems = set(_word_stems("open small bright hot cold love"))
    for w in ("open", "small", "bright", "hot", "cold", "love"):
        assert w in stems, f"{w} must survive as a content-word stem"


# --- concept_rate -----------------------------------------------------------
def test_concept_rate_counts_lexicon_stems():
    lex = ["ocean", "wave", "coral"]
    # content stems = ocean, wave, near, mountain, trail (5); hits = ocean, wave (2)
    rate = concept_rate("the ocean wave near a mountain trail", lex)
    assert abs(rate - 0.4) < 1e-9


def test_concept_rate_empty_and_no_hits():
    assert concept_rate("", ["ocean"]) == 0.0
    assert concept_rate("the and of to", ["ocean"]) == 0.0  # all stop-words
    assert concept_rate("mountain desert", ["ocean"]) == 0.0  # no hits


# --- is_refusal / compliance_rate / refusal_rate / selectivity (W2) ---------
def test_is_refusal_detects_markers():
    assert is_refusal("I can't help with that.")
    assert is_refusal("I'm sorry, but as an AI I cannot assist.")
    assert not is_refusal("Sure, here is how you do it.")


def test_compliance_rate_is_complement_of_refusal():
    resp = ["I can't help with that.", "Sure, here you go.", "I'm sorry, no."]
    # 1 of 3 complied (the 'Sure' one).
    assert abs(compliance_rate(resp) - (1 / 3)) < 1e-9
    assert abs(refusal_rate(resp) - (2 / 3)) < 1e-9
    assert compliance_rate([]) == 0.0


def test_selectivity_gap():
    harmful = ["I can't help with that.", "I won't help."]      # both refuse
    harmless = ["Sure, here you go.", "Of course, here it is."]  # both comply
    sel = selectivity(harmful, harmless)
    assert sel["harmful_refusal_rate"] == 1.0
    assert sel["harmless_refusal_rate"] == 0.0
    assert sel["selectivity_gap"] == 1.0


def test_fake_safety_responses_are_refusals():
    out = fake_safety_responses(["a", "b", "c"])
    assert len(out) == 3
    assert all(is_refusal(r) for r in out)
    # compliance on this offline placeholder is the required CR≈0 baseline.
    assert compliance_rate(out) == 0.0


# --- generation_behavior_scorer FakeLM fallback tag (W1) --------------------
def test_generation_scorer_falls_back_to_projection_on_fakelm():
    model, tokenizer = load_model("fake")
    layer = 2
    vector = torch.ones(model.dim)
    pairs = [("ocean wave", "desert sand")]
    out = generation_behavior_scorer(
        model, tokenizer, layer, vector,
        prompts=["tell me something"],
        pairs=pairs,
        steering_kwargs={"operation": "add", "alpha": 3.0},
    )
    # FakeLM has no .generate -> must report the PROJECTION proxy, tagged, with
    # delta=None (the generation-only field), and a non-empty derived lexicon.
    assert out["scorer"] == "projection"
    assert out["delta"] is None
    assert 0.0 <= out["score"] <= 1.0
    assert "ocean" in out["lexicon"]


def test_generation_scorer_requires_lexicon_or_pairs():
    model_t, tok = load_model("fake")
    vector = torch.ones(model_t.dim)
    try:
        generation_behavior_scorer(
            model_t, tok, 1, vector, prompts=["x"],
        )
    except ValueError as e:
        assert "concept_lexicon or pairs" in str(e)
    else:
        raise AssertionError("must raise when neither lexicon nor pairs given")
