"""data_floor.py — the >=500/class data floor, made explicit and machine-checked.

The course's hard rubric (CLAUDE.md sec.17) sets a floor of **>=500 positives and
>=500 negatives per class** for any headline number, and forbids tiny evaluation
slices outright. This lesson shipped a hill-climb whose benign arm was **20
prompts** — not a borderline call, a hard violation — and nothing in the code
said so. A floor that has to be remembered is a floor that will be skipped, so
this module makes it a build-time object: the split is *planned* against the real
pool, the achieved n is *stamped* into the results JSON, and a shortfall prints a
warning naming its cause.

The distinction this module exists to preserve
----------------------------------------------
There are two completely different reasons an arm can land below the floor, and
collapsing them is how a violation gets laundered into a caveat:

``capped_by = "pool"``
    the corpus does not contain enough clean prompts. Nothing the operator can do
    at run time fixes it; the number is honestly reported at the pool maximum.
``capped_by = "env"``
    the operator shrank the run with an env cap to fit a foreground window. The
    data exists; this run simply did not use it. The result is SCREENING and must
    never be quoted as a headline.

Measured pool (seed 0, ``common.data.build_harmful_benign``, 2026-08):
``harmful`` caps at **792** (693 unique toxic-chat + 99 length-windowed JBB
top-up) and the raw ``benign`` pool holds **8889** — but the loader returns a
BALANCED set, so both classes cap at **792**. With a 300/class extract slice the
disjoint eval slice therefore maxes out at **492**, eight short of the floor;
trimming the extract to 292 buys exactly 500. Both choices are legitimate and the
planner supports both — what is not legitimate is making the choice silently.

CPU-only. Run the self-test with
``python -m steering_tutorials.stacking.data_floor``.
"""
from __future__ import annotations

import sys

#: The rubric floor, per class, for a headline number (CLAUDE.md sec.17 item 1).
FLOOR_PER_CLASS = 500

#: Hard lower bound on the extract slice used to build a steering direction. The
#: planner may trim the preferred extract size to reach the eval floor, but never
#: below this — a direction built on fewer contrast pairs is a different defect.
MIN_EXTRACT = 250


def plan_split(pool_per_class: int, n_extract_pref: int, n_eval_target: int,
               min_extract: int = MIN_EXTRACT,
               preserve_extract: bool = False) -> dict:
    """Allocate disjoint extract / eval slices from one balanced pool.

    Parameters
    ----------
    pool_per_class : int
        Rows actually returned per class by the loader (NOT what was requested).
    n_extract_pref : int
        Preferred extract size — the value a pre-registration named, if any.
    n_eval_target : int
        Eval slice wanted per class, normally :data:`FLOOR_PER_CLASS`.
    min_extract : int
        The planner will not trim the extract below this to reach the target.
    preserve_extract : bool
        ``True`` freezes ``n_extract_pref`` even when trimming it would reach the
        eval floor. Use it when a pre-registration pinned the extract size: the
        eval arm then lands below the floor and is reported ``capped_by="pool"``
        with ``preserve_extract`` in the record, so the trade is visible.

    Returns
    -------
    dict with ``n_extract``, ``n_eval``, ``pool_per_class``, ``trimmed_extract``,
    ``meets_target``, and a human-readable ``note``.
    """
    pool = int(pool_per_class)
    pref = int(n_extract_pref)
    target = int(n_eval_target)

    n_extract = min(pref, max(0, pool))
    # The eval slice is what the caller ASKED for, or whatever the pool leaves —
    # never more than the request, so a big pool does not silently inflate the
    # run into a window it was capped to fit.
    n_eval = min(target, max(0, pool - n_extract))
    trimmed = False

    if n_eval < target and not preserve_extract:
        # Trim the extract slice (never below min_extract) to buy eval rows.
        room = pool - target
        if room >= min_extract:
            n_extract = room
            n_eval = target
            trimmed = n_extract != pref

    n_eval = min(target, max(0, pool - n_extract))
    if trimmed:
        note = (f"extract trimmed {pref} -> {n_extract} so the eval slice reaches "
                f"the {target}/class target from a {pool}/class pool")
    elif n_eval >= target:
        note = f"pool {pool}/class covers extract {n_extract} + eval {n_eval}"
    elif preserve_extract:
        note = (f"extract PINNED at {pref} (pre-registered); the {pool}/class pool "
                f"then leaves only {n_eval} for eval, below the {target} target")
    else:
        note = (f"pool {pool}/class cannot cover extract >= {min_extract} plus "
                f"eval {target}; eval capped at {n_eval}")
    return {"pool_per_class": pool, "n_extract": n_extract, "n_eval": n_eval,
            "n_extract_preferred": pref, "n_eval_target": target,
            "min_extract": min_extract, "preserve_extract": bool(preserve_extract),
            "trimmed_extract": trimmed, "meets_target": bool(n_eval >= target),
            "note": note}


def floor_report(n_harmful: int, n_benign: int, plan: dict,
                 requested_harmful: int, requested_benign: int,
                 pool_harmful_raw: "int | None" = None,
                 pool_benign_raw: "int | None" = None,
                 floor: int = FLOOR_PER_CLASS) -> dict:
    """The ``data_floor`` block that every results JSON from this lesson carries.

    ``n_harmful`` / ``n_benign`` are the slice sizes ACTUALLY evaluated;
    ``requested_*`` are what the run asked for after env caps. ``capped_by``
    separates a corpus limit from an operator's env cap (see the module
    docstring) — ``None`` when the floor is met.
    """
    def cap_cause(achieved: int, requested: int) -> "str | None":
        if achieved >= floor:
            return None
        return "env" if requested < floor else "pool"

    cap_h, cap_b = cap_cause(n_harmful, requested_harmful), cap_cause(n_benign,
                                                                     requested_benign)
    causes = [c for c in (cap_h, cap_b) if c]
    return {
        "floor_per_class": floor,
        "achieved_n_harmful": int(n_harmful),
        "achieved_n_benign": int(n_benign),
        "requested_n_harmful": int(requested_harmful),
        "requested_n_benign": int(requested_benign),
        "meets_floor_harmful": bool(n_harmful >= floor),
        "meets_floor_benign": bool(n_benign >= floor),
        "meets_floor": bool(n_harmful >= floor and n_benign >= floor),
        "pool_capped": bool("pool" in causes),
        "env_capped": bool("env" in causes),
        "capped_by": {"harmful": cap_h, "benign": cap_b},
        "pool_harmful_available": pool_harmful_raw,
        "pool_benign_available": pool_benign_raw,
        "split_plan": plan,
        "rubric": "CLAUDE.md sec.17 item 1: >=500 per class for a headline number; "
                  "below the floor this run is SCREENING and must be labelled so.",
    }


def warn_if_below_floor(report: dict, stream=sys.stderr) -> bool:
    """Print an unmissable ASCII warning when an arm is under the floor.

    Returns ``True`` when a warning was printed. Deliberately loud and
    deliberately not an exception: a capped screening run is a legitimate thing
    to do on this host — shipping its numbers as a headline is not.
    """
    if report.get("meets_floor"):
        return False
    bar = "!" * 78
    print(bar, file=stream)
    for cls in ("harmful", "benign"):
        n = report[f"achieved_n_{cls}"]
        if n >= report["floor_per_class"]:
            continue
        cause = report["capped_by"][cls]
        why = ("the corpus pool is exhausted (raise nothing; the number is at the "
               "pool maximum)" if cause == "pool" else
               "an ENV CAP shrank this run (the data exists; this run did not use it)")
        print(f"! DATA FLOOR NOT MET  {cls}: n={n} < {report['floor_per_class']} "
              f"-- {why}", file=stream)
    print("! This run is SCREENING tier. Do NOT quote it as a headline number.",
          file=stream)
    print(bar, file=stream)
    return True


# --------------------------------------------------------------------------- #
# CPU self-test — no model, no network.
# Run: python -m steering_tutorials.stacking.data_floor
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    # (a) a pool that comfortably covers both slices leaves the preference alone,
    #     and the eval slice stays at the REQUEST, not at everything left over.
    p = plan_split(1200, 300, 500)
    assert (p["n_extract"], p["n_eval"]) == (300, 500) and not p["trimmed_extract"]
    assert p["meets_target"]

    # (b) the real pool (792): trimming the extract by 8 buys the floor exactly.
    p = plan_split(792, 300, 500)
    assert (p["n_extract"], p["n_eval"]) == (292, 500), p
    assert p["trimmed_extract"] and p["meets_target"]

    # (c) preserve_extract honours a pre-registered extract and misses the floor.
    p = plan_split(792, 300, 500, preserve_extract=True)
    assert (p["n_extract"], p["n_eval"]) == (300, 492), p
    assert not p["trimmed_extract"] and not p["meets_target"]

    # (d) the extract is never trimmed below min_extract to chase the target.
    p = plan_split(600, 300, 500, min_extract=250)
    assert p["n_extract"] == 300 and p["n_eval"] == 300 and not p["meets_target"]

    # (e) capped_by separates a corpus limit from an operator's env cap.
    r_pool = floor_report(492, 492, plan_split(792, 300, 500, preserve_extract=True),
                          requested_harmful=500, requested_benign=500)
    assert r_pool["pool_capped"] and not r_pool["env_capped"]
    assert r_pool["capped_by"] == {"harmful": "pool", "benign": "pool"}
    r_env = floor_report(40, 20, plan_split(792, 300, 500),
                         requested_harmful=40, requested_benign=20)
    assert r_env["env_capped"] and not r_env["pool_capped"]
    assert not r_env["meets_floor"] and r_env["achieved_n_benign"] == 20
    r_ok = floor_report(500, 500, plan_split(792, 300, 500),
                        requested_harmful=500, requested_benign=500)
    assert r_ok["meets_floor"] and r_ok["capped_by"] == {"harmful": None, "benign": None}

    # (f) the warning fires exactly when the floor is missed.
    import io
    buf = io.StringIO()
    assert warn_if_below_floor(r_env, stream=buf) is True
    assert "DATA FLOOR NOT MET" in buf.getvalue() and "ENV CAP" in buf.getvalue()
    assert warn_if_below_floor(r_ok, stream=io.StringIO()) is False

    print("[self-test] OK - split planner reaches 500/class from the real 792 pool "
          "by trimming extract 300->292, honours a pinned extract, never trims below "
          "min_extract; floor_report separates pool-capped from env-capped; the "
          "warning fires only below the floor.")


if __name__ == "__main__":
    _self_test()
