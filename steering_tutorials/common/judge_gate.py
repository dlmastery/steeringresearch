"""judge_gate.py — refuse to produce, or to publish, a self-judged number.

CLAUDE.md sec. 17 rubric item 3 is unconditional: every REPORTED GENERATION
number needs an OFF-FAMILY judge (``STEER_JUDGE_MODEL=Qwen/Qwen2.5-3B-Instruct``).
A 1B model grading its own output INFLATES refusal, so a self-judged rate is
inadmissible — not "weaker evidence", inadmissible.

``Judge`` used to fall back to self-judging SILENTLY when ``STEER_JUDGE_MODEL``
was unset. That is the sec. 18.8 failure mode exactly: it never crashed, it
produced confident well-formed numbers, and eight lessons shipped with
unverifiable judge provenance. ``Judge.stamp()`` now records the provenance on
disk, but a stamp is only a diagnosis after the fact. This module is the
prophylaxis, in two places:

  * :func:`require_off_family_judge` — called at the TOP of a runner's ``main``,
    before the model loads and before a single token is generated. A run that
    could only ever yield an inadmissible number must refuse to START, not
    discover it after hours of GPU.
  * :func:`assert_publishable` — called immediately before ``results.json`` is
    written. The write path is the one place a check cannot be forgotten
    (sec. 18.8: "a check that must be remembered will eventually be skipped").

The semantics are lifted from ``steering_tutorials/reft_r1/run_reft.py``, which
is the reference implementation; this is that guard factored out, not a new
design. Both helpers raise :class:`SystemExit` — a loud, non-catchable-by-accident
stop, matching reft_r1.

Import cost: nothing heavy at module scope, so ``import judge_gate`` never drags
in torch and the self-test below runs on a CPU-only box with no model present::

    python -m steering_tutorials.common.judge_gate
"""
from __future__ import annotations

import os
import re
from typing import Any, Iterable, Sequence

# Global opt-out, mirroring reft_r1's REFT_ALLOW_SELF_JUDGE. Every lesson also
# honours a per-lesson <LESSON>_ALLOW_SELF_JUDGE. Setting either is a DELIBERATE
# choice to produce a smoke-tier, non-reportable number.
GLOBAL_ALLOW_ENV = "STEER_ALLOW_SELF_JUDGE"

#: The judge every reported generation number must be graded by (sec. 17 item 3).
OFF_FAMILY_JUDGE = "Qwen/Qwen2.5-3B-Instruct"

#: Word-parts that mark a key as carrying a JUDGE VERDICT. Matched against the
#: key's underscore-separated tokens, not the raw string — see :data:`_GEOMETRY`.
_VERDICT = ("refusal", "refusals", "compliance", "compliant", "noncompliant",
            "complied", "gibberish", "asr", "jailbroken", "jailbreak", "leak")

#: Word-parts that mark a key as GEOMETRY or BOOKKEEPING even though it also
#: names a verdict. Two families, both learned from real artifacts:
#:
#:   * geometry — ``cos_refusal_shift_vs_vector`` and ``refusal_shift_norm`` are
#:     prompt_activation_duality's judge-FREE headline: cosines between
#:     activation vectors, computed with no judge in the room.
#:   * counts — ``n_pool_jailbreak_unique`` (trajguard) is a dataset size.
#:
#: A gate that demanded judge provenance for these would be wrong in the
#: direction that matters most: it would make the judge-free lessons
#: unpublishable, and a gate that cries wolf is a gate somebody switches off.
_NOT_A_RATE = ("cos", "cosine", "norm", "shift", "vector", "direction", "dim",
               "rank", "layer", "threshold", "path", "png", "instruction",
               "idx", "seed", "sep",
               "n", "count", "total", "pool", "unique", "cap", "length")

#: Judge-derived keys whose names carry none of the verdict words above. Lessons
#: that invented a local vocabulary land here.
GENERATION_RATE_KEYS = frozenset({
    "success",       # multi_intent: mean judged refusal over the active concepts
    "crosstalk",     # multi_intent: judged refusal on an INACTIVE concept
})


def _is_rate_key(key: str) -> bool:
    """True when ``key`` names a number that only a judge could have produced."""
    if key in GENERATION_RATE_KEYS:
        return True
    tokens = set(re.split(r"[^a-z0-9]+", key.lower()))
    return bool(tokens & set(_VERDICT)) and not (tokens & set(_NOT_A_RATE))


def _is_rate_value(value: Any) -> bool:
    """True when ``value`` could be a rate: a real number in [-1, 1].

    The bound is [-1, 1] rather than [0, 1] because stacking reports signed
    DELTAS between judged rates (``overstack_gibberish_delta``). It excludes
    ``median_char_length.jailbreak = 732`` — a character count that merely
    happens to be filed under a verdict word.
    """
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and -1.0 <= float(value) <= 1.0)


# --------------------------------------------------------------------------- #
# 1. The pre-flight gate.
# --------------------------------------------------------------------------- #
def require_off_family_judge(
    lesson: str,
    judge: Any = None,
    *,
    judge_free_metrics: Sequence[str] = (),
    allow_env: str | None = None,
) -> str:
    """Hard-fail unless the judge is off-family. Returns the resolved judge id.

    Parameters
    ----------
    lesson
        The lesson name, used in the message and to derive the per-lesson
        opt-out env var (``gavel`` -> ``GAVEL_ALLOW_SELF_JUDGE``).
    judge
        Optional live :class:`Judge`. Normally left ``None`` so this can be
        called BEFORE the model loads — with no judge the decision is read
        straight off ``STEER_JUDGE_MODEL`` via ``resolve_judge_id()``, which
        loads nothing. Pass a judge only to re-check an already-built one.
    judge_free_metrics
        Names of this lesson's metrics that do NOT depend on the judge. There is
        no way to compute them alone in a runner that has no judge-free arm, so
        the guard names them in the refusal message: a user who is blocked here
        should be told exactly what they are losing, and can then decide whether
        to set the opt-out for a judge-free-only look.
    allow_env
        Override the global opt-out variable name.

    Raises
    ------
    SystemExit
        When the target model would grade its own output and no opt-out is set.
    """
    # Imported here, not at module scope, so this module stays import-light.
    from steering_tutorials.hello_world_steering.judge import (
        SELF_JUDGE_ID, resolve_judge_id,
    )

    if judge is not None:
        stamp = judge.stamp()
        self_judged = bool(stamp["is_self_judge"])
        judge_id = str(stamp["judge_id"])
    else:
        judge_id = resolve_judge_id()
        self_judged = judge_id == SELF_JUDGE_ID

    if not self_judged:
        return judge_id

    for name in _opt_out_names(lesson, allow_env):
        if os.environ.get(name):
            print(
                f"[{lesson}][judge][WARN] {name} is set: running SELF-JUDGED. "
                f"Every generation rate this run produces is SMOKE-TIER and must "
                f"NOT be reported as a headline (CLAUDE.md sec.17 item 3).",
                flush=True,
            )
            return judge_id

    raise SystemExit(_refusal_message(lesson, judge_free_metrics, allow_env))


def _opt_out_names(lesson: str, allow_env: str | None) -> tuple[str, ...]:
    """The env vars that opt this lesson out, most specific first."""
    per_lesson = re.sub(r"[^A-Za-z0-9]+", "_", lesson).strip("_").upper()
    names = [f"{per_lesson}_ALLOW_SELF_JUDGE"] if per_lesson else []
    names.append(allow_env or GLOBAL_ALLOW_ENV)
    return tuple(dict.fromkeys(names))


def _refusal_message(
    lesson: str,
    judge_free_metrics: Sequence[str],
    allow_env: str | None,
) -> str:
    opts = _opt_out_names(lesson, allow_env)
    lines = [
        f"REFUSING TO RUN [{lesson}]: STEER_JUDGE_MODEL is unset, so the target "
        "model would grade its own output. CLAUDE.md sec.17 rubric item 3 "
        "requires an OFF-FAMILY judge for ALL reported generation numbers; a 1B "
        "model judging itself INFLATES refusal.",
        "",
        f"  Fix:  STEER_JUDGE_MODEL={OFF_FAMILY_JUDGE}  python -m "
        f"steering_tutorials.{lesson}...",
    ]
    if judge_free_metrics:
        lines += [
            "",
            f"  What you lose by being blocked here: this lesson has NO "
            f"judge-free-only code path, so the following JUDGE-FREE metrics "
            f"cannot be produced either without opting out --",
        ]
        lines += [f"      - {m}" for m in judge_free_metrics]
        lines += [
            "    Those are computable without any judge; only the generation "
            "rates in the same results.json are inadmissible.",
        ]
    lines += [
        "",
        f"  Deliberate smoke-tier run (NOT reportable): set {opts[0]}=1"
        + (f" or {opts[1]}=1" if len(opts) > 1 else "")
        + ".",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2. The publish gate.
# --------------------------------------------------------------------------- #
def assert_publishable(results: dict, lesson: str) -> dict:
    """Refuse to write ``results`` if it carries judged rates without provenance.

    A results dict is publishable when EITHER it contains no generation-rate key
    at all (a purely judge-free artifact), OR it carries an ``off_family`` stamp
    that is ``True``. An UNSTAMPED artifact fails just as hard as a self-judged
    one: absence of evidence is the defect (sec. 18.6), not a clean bill.

    Returns ``results`` unchanged so it can be used as a pass-through.

    Raises
    ------
    SystemExit
        When judged rates are present and the stamp is missing or self-judged.
    """
    rate_keys = sorted(set(_find_rate_keys(results)))
    if not rate_keys:
        return results

    stamps = list(_find_key(results, "off_family"))
    shown = ", ".join(rate_keys[:8]) + (" ..." if len(rate_keys) > 8 else "")

    if not stamps:
        raise SystemExit(
            f"REFUSING TO WRITE [{lesson}] results.json: it reports "
            f"judge-derived generation rates ({shown}) but carries NO "
            f"Judge.stamp() provenance. An artifact whose judge cannot be read "
            f"off the artifact is not evidence (CLAUDE.md sec.18.6/18.8). Add "
            f"`**judge.stamp()` to the results dict."
        )
    if not all(stamps):
        # HONOUR THE SAME OPT-OUT THE ENTRY GATE HONOURS. Added 2026-08-21 after a
        # readiness audit found the two gates disagreed: require_off_family_judge
        # lets <LESSON>_ALLOW_SELF_JUDGE through with a SMOKE-TIER warning, while
        # this one refused unconditionally. So a DELIBERATE smoke run would
        # generate for an hour and then be refused the write -- the escape hatch
        # was a trap that destroyed the work it claimed to permit.
        #
        # The point of the rule is that a self-judged rate must never be read as a
        # headline, not that it must never reach disk. So we WRITE and BRAND it,
        # which is strictly more informative than refusing: the artifact then
        # carries its own inadmissibility instead of not existing.
        if _self_judge_allowed(lesson):
            results["TIER"] = "SMOKE_SELF_JUDGED"
            results["publishable"] = False
            results["publishable_note"] = (
                "SELF-JUDGED. The target model graded its own output, which "
                "INFLATES refusal (CLAUDE.md sec.17 item 3). Written only because "
                "an explicit <LESSON>_ALLOW_SELF_JUDGE opt-out was set. NOT "
                "admissible as a headline; re-run with STEER_JUDGE_MODEL=%s."
                % OFF_FAMILY_JUDGE)
            return results
        raise SystemExit(
            f"REFUSING TO WRITE [{lesson}] results.json: it reports "
            f"judge-derived generation rates ({shown}) graded by the TARGET "
            f"MODEL ITSELF (off_family=False). Self-judged rates are "
            f"inadmissible under CLAUDE.md sec.17 item 3. Re-run with "
            f"STEER_JUDGE_MODEL={OFF_FAMILY_JUDGE}, or set an explicit "
            f"<LESSON>_ALLOW_SELF_JUDGE to write a branded SMOKE_SELF_JUDGED "
            f"artifact instead of losing the run."
        )
    return results


def _self_judge_allowed(lesson: str) -> bool:
    """Was an explicit self-judge opt-out set, per the entry gate's own rules?"""
    import os
    for name in _opt_out_names(lesson, None):
        if os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on"):
            return True
    return bool(os.environ.get(GLOBAL_ALLOW_ENV, "").strip().lower()
                in ("1", "true", "yes", "on"))


def _find_rate_keys(node: Any) -> Iterable[str]:
    """Yield every key in the tree whose name denotes a judge-derived rate."""
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and _is_rate_value(v) and _is_rate_key(k):
                yield k
            yield from _find_rate_keys(v)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _find_rate_keys(item)


def _find_key(node: Any, name: str) -> Iterable[Any]:
    """Yield every value stored under ``name`` anywhere in the tree."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == name:
                yield v
            yield from _find_key(v, name)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _find_key(item, name)


# --------------------------------------------------------------------------- #
# 3. Self-test — CPU-only, loads no model, touches no GPU.
# --------------------------------------------------------------------------- #
class _FakeJudge:
    """Minimal stand-in exposing only the ``stamp()`` surface the gate reads."""

    def __init__(self, off_family: bool, judge_id: str):
        self._off, self._id = off_family, judge_id

    def stamp(self) -> dict:
        return {"judge_id": self._id, "is_self_judge": not self._off,
                "judge_model_id": self._id, "off_family": self._off}


def _self_test() -> None:
    saved = {k: os.environ.get(k) for k in
             (GLOBAL_ALLOW_ENV, "STEER_JUDGE_MODEL", "GAVEL_ALLOW_SELF_JUDGE")}
    for k in saved:
        os.environ.pop(k, None)
    try:
        # --- require_off_family_judge -------------------------------------- #
        # (a) env unset => self-judge => MUST raise, before any model exists.
        try:
            require_off_family_judge("gavel", judge_free_metrics=["block_rate_harmful"])
        except SystemExit as e:
            msg = str(e)
            assert "REFUSING TO RUN" in msg and "gavel" in msg, msg
            assert OFF_FAMILY_JUDGE in msg, "message must name the off-family judge"
            assert "block_rate_harmful" in msg, "must name the judge-free metrics"
            assert "GAVEL_ALLOW_SELF_JUDGE" in msg, "must name the opt-out"
        else:
            raise AssertionError("unset STEER_JUDGE_MODEL must refuse to run")

        # (b) a live self-judging Judge is caught too, not just the env.
        try:
            require_off_family_judge("stacking", _FakeJudge(False, "self"))
        except SystemExit:
            pass
        else:
            raise AssertionError("a self-judging Judge must refuse to run")

        # (c) off-family env => passes and returns the id.
        os.environ["STEER_JUDGE_MODEL"] = OFF_FAMILY_JUDGE
        assert require_off_family_judge("stacking") == OFF_FAMILY_JUDGE
        assert require_off_family_judge(
            "stacking", _FakeJudge(True, OFF_FAMILY_JUDGE)) == OFF_FAMILY_JUDGE

        # (d) the opt-out is honoured, loudly.
        del os.environ["STEER_JUDGE_MODEL"]
        os.environ["GAVEL_ALLOW_SELF_JUDGE"] = "1"
        assert require_off_family_judge("gavel") == "self"
        del os.environ["GAVEL_ALLOW_SELF_JUDGE"]

        # --- assert_publishable -------------------------------------------- #
        judged = {"ladder": [{"rung": "attacked", "asr": 0.46,
                              "refusal_rate": 0.02, "n": 200}]}

        # (a) judged rates + self-judge stamp => raise.
        try:
            assert_publishable({**judged, **_FakeJudge(False, "self").stamp()},
                               "rogue_scalpel")
        except SystemExit as e:
            assert "off_family=False" in str(e), str(e)
        else:
            raise AssertionError("a self-judged results dict must not be written")

        # (b) judged rates + NO stamp => raise. Absence of evidence is the defect.
        try:
            assert_publishable(dict(judged), "rogue_scalpel")
        except SystemExit as e:
            assert "NO Judge.stamp()" in str(e), str(e)
        else:
            raise AssertionError("an unstamped results dict must not be written")

        # (c) judged rates + off-family stamp => passes through unchanged.
        ok = {**judged, **_FakeJudge(True, OFF_FAMILY_JUDGE).stamp()}
        assert assert_publishable(ok, "rogue_scalpel") is ok

        # (d) a purely judge-free artifact needs no stamp at all. This is the
        #     carve-out that keeps geometry/probe lessons publishable.
        geom = {"cos_refusal_shift_vs_vector": 0.41, "effective_rank": 88.0,
                "block_rate_harmful": 0.93, "n": 300}
        assert assert_publishable(geom, "prompt_activation_duality") is geom

        # (e) nested + list-borne rates are found, not just top-level ones.
        nested = {"by_rung": [{"deep": {"gibberish_rate": 0.9}}]}
        try:
            assert_publishable(nested, "stacking")
        except SystemExit:
            pass
        else:
            raise AssertionError("nested judged rates must be found")

        # (f) booleans are not rates (True is an int in Python -- guard it).
        assert list(_find_rate_keys({"is_refusal": True})) == []

        # (g) The classifier pinned against the REAL key names each guarded
        #     lesson writes. These are the two lists the gate lives or dies on,
        #     so they are asserted rather than trusted.
        for k in ("refusal_rate", "compliance_rate", "gibberish_rate", "asr",
                  "harmful_refusal_rate", "benign_over_refusal_rate",
                  "baseline_refusal_rate", "guarded_refusal_rate",
                  "harmful_passed_compliance_rate", "system_harmful_leak_rate",
                  "baseline_refusal", "prompting_refusal", "steer_proj_refusal",
                  "steer_shared_gibberish", "best_refusal", "refusal_spread",
                  "noncompliant_rate", "noncompliant_refusal_share",
                  "refusal_score", "overstack_gibberish_delta",
                  "success", "crosstalk"):
            assert _is_rate_key(k), f"{k} must be recognised as judge-derived"
        for k in ("cos_refusal_shift_vs_vector", "refusal_shift_norm",
                  "control_shift_norm", "block_rate_harmful",
                  "false_block_benign", "effective_threshold", "norm_budget",
                  "target_fpr", "n_shift", "tau", "alpha",
                  # trajguard dataset bookkeeping -- counts, not verdicts.
                  "n_pool_jailbreak_unique", "n_pool_jailbreak_after_length_cap"):
            assert not _is_rate_key(k), f"{k} is judge-free and must not gate"

        # (h) The VALUE filter, independently of the name filter. A char count
        #     filed under a verdict word (trajguard's median_char_length.jailbreak
        #     = 732) is not a rate; a signed delta between rates is.
        assert _is_rate_key("jailbreak"), "the NAME alone does look verdict-ish"
        assert not _is_rate_value(732), "a char count is not a rate"
        assert _is_rate_value(-0.13), "a signed rate delta is a rate"
        assert not _is_rate_value(True), "a bool is not a rate"
        assert list(_find_rate_keys({"median_char_length": {"jailbreak": 732}})) == []
        assert list(_find_rate_keys({"d": {"overstack_gibberish_delta": -0.13}})) \
            == ["overstack_gibberish_delta"]
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("judge_gate self-test OK (no model loaded, no GPU touched)")


if __name__ == "__main__":
    _self_test()
