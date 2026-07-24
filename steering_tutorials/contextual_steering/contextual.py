"""contextual.py — the input-adaptive steering schedule (the heart of the lesson).

Fixed-alpha steering (lesson 2) applies the SAME push to every prompt. That is
wasteful and risky: a benign prompt gets shoved just as hard as a harmful one, so
harmless requests get needlessly distorted (over-steering). Contextual steering
fixes this by making the push depend on the input.

The signal is free. The steering direction ``v`` was built as
``mean(harmful) - mean(benign)`` (lesson 2's diff-of-means), so a harmful
prompt's mean-pooled hidden state points ALONG ``v`` and a benign prompt's does
not. That alignment IS the "how harmful does this look?" gate — no separate probe
needed. We read it as a cosine and turn it into a per-prompt strength:

    proj_i  = cos(mean_pool(h_i @ layer), v_unit)              # in [-1, 1]
    frac_i  = clip( relu((proj_i - tau) / (ref - tau)), 0, cap)
    alpha_i = alpha_base * frac_i

Below ``tau`` (the benign floor) ``frac_i`` is 0 → the prompt is NOT steered.
At ``ref`` (the harmful anchor) ``frac_i`` is 1 → the prompt gets the full
``alpha_base``. In between it ramps linearly. This cosine ramp is our own
construction inspired by CLAS, NOT the paper's formula (CLAS learns a sensing
vector ``alpha = c . [h, 1]`` for the per-input strength); the run script
CALIBRATES ``tau`` and ``ref`` from the extract split (see
``calibrate_schedule``) so the ramp sits where the two prompt classes actually
separate.

  Contextual Linear Activation Steering — CLAS (Hsu, Beaglehole, Radhakrishnan
    & Belkin, arXiv:2604.24693, Apr 2026) — we borrow the per-input-adaptive
    strength IDEA; the ramp here is our own, not the paper's learned sensing
    vector.

Everything here is pure NumPy math except :class:`ContextualSteerer`, which wraps
lesson 2's ``mean_pool_activation`` (to read the projection) and ``generate``
(the relative-add steering hook). No new steering mechanism is implemented — we
only make lesson 2's ``alpha`` a function of the input.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Pure schedule math — no model, unit-tested on CPU below.
# --------------------------------------------------------------------------- #
def cosine_projection(
    pooled: np.ndarray, v_unit: np.ndarray, center: "np.ndarray | None" = None
) -> float:
    """cos(pooled − center, v_unit) — alignment of a mean-pooled state with ``v``.

    ``v_unit`` is already L2-normalized (it comes from ``extract_caa_vector``),
    so we only need to normalize ``pooled``. Returns a scalar in ``[-1, 1]``:
    positive ⇒ the input points TOWARD the harm/refusal direction, negative ⇒
    away from it. A zero-norm pooled vector (degenerate) yields 0.0.

    ``center`` (recommended: the extract **benign mean** pooled activation) is
    subtracted first. This matters on Gemma: the raw mean-pooled state is
    dominated by a huge, prompt-independent common component (high-norm
    attention-sink dimensions), so ``cos(pooled, v_unit)`` is nearly the SAME
    constant for every prompt (~−0.82 here) and the harmful/benign contrast is
    swamped. Centering by the benign mean removes that common component so the
    projection becomes a real "distance from benign toward harm" signal — the
    same trick ``gavel`` uses ((h − benign_mean)·direction). With ``center=None``
    the old raw cosine is preserved (kept for the CPU unit test).
    """
    pooled = np.asarray(pooled, dtype=np.float64).reshape(-1)
    v_unit = np.asarray(v_unit, dtype=np.float64).reshape(-1)
    if center is not None:
        pooled = pooled - np.asarray(center, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(pooled)
    if n == 0.0:
        return 0.0
    # v_unit is unit already, but divide defensively in case a raw v is passed.
    vn = np.linalg.norm(v_unit)
    if vn == 0.0:
        return 0.0
    return float(np.dot(pooled, v_unit) / (n * vn))


def contextual_alpha(
    proj: float,
    alpha_base: float,
    tau: float,
    ref: float = 1.0,
    cap: float = 1.0,
) -> float:
    """Map a projection to a per-prompt steering strength.

        frac = clip( relu((proj - tau) / (ref - tau)), 0, cap )
        alpha = alpha_base * frac

    - ``proj <= tau``  → ``frac = 0``    → no steering (the benign floor).
    - ``proj  = ref``  → ``frac = 1``    → the full ``alpha_base``.
    - ``proj  > ref``  → capped at ``cap`` so a very-aligned prompt is not
      over-steered past ``cap * alpha_base`` (default cap 1.0 ⇒ never exceeds).

    ``ref`` must be > ``tau``; if it is not (a degenerate calibration) we treat
    the schedule as a hard step at ``tau`` (frac 1 above, 0 below).
    """
    if ref <= tau:
        frac = 1.0 if proj > tau else 0.0
    else:
        frac = (proj - tau) / (ref - tau)
        if frac < 0.0:
            frac = 0.0          # relu
        if frac > cap:
            frac = cap          # ceiling
    return float(alpha_base * frac)


def calibrate_schedule(
    proj_harmful: "list[float] | np.ndarray",
    proj_benign: "list[float] | np.ndarray",
    benign_percentile: float = 90.0,
    harmful_percentile: "float | None" = None,
) -> dict:
    """Pick ``tau`` and ``ref`` from EXTRACT-split projections (never eval).

    - ``tau`` = a high percentile of the BENIGN projections. Steering only kicks
      in above where most benign prompts sit, so benign over-steering is cut.
      A high percentile (90) is deliberately conservative: it tolerates a few
      benign prompts leaking a little steering rather than clipping the harmful
      ramp too low.
    - ``ref`` = the MEAN harmful projection when ``harmful_percentile is None``
      (the honest central anchor: a typical harmful prompt receives ~``alpha_base``,
      more-aligned ones are capped, less-aligned ones ramp down). Pass a
      ``harmful_percentile`` in ``(0, 100]`` to instead anchor "full strength" at
      that percentile of the harmful projections — a gentler ramp that only
      saturates on high-confidence-harmful prompts.

    Whether the ramp actually OPERATES (rather than collapsing to the degenerate
    ``ref <= tau`` hard step) is entirely a property of the signal: a signal that
    separates the classes (the probe logit) puts the harmful projections above the
    benign ones so ``ref > tau``; one that does not (the diff-of-means cosine) does
    not. The caller checks ``ref > tau`` and warns if the ramp is degenerate.

    Returns ``{"tau", "ref", "benign_percentile", "harmful_percentile", ...}``
    diagnostics that the run stores so the schedule is fully transparent.
    """
    ph = np.asarray(proj_harmful, dtype=np.float64)
    pb = np.asarray(proj_benign, dtype=np.float64)
    tau = float(np.percentile(pb, benign_percentile)) if pb.size else 0.0
    if harmful_percentile is None:
        ref = float(ph.mean()) if ph.size else 1.0
    else:
        ref = float(np.percentile(ph, harmful_percentile)) if ph.size else 1.0
    return {
        "tau": tau,
        "ref": ref,
        "benign_percentile": float(benign_percentile),
        "harmful_percentile": (None if harmful_percentile is None
                               else float(harmful_percentile)),
        "harmful_proj_mean": float(ph.mean()) if ph.size else 0.0,
        "benign_proj_mean": float(pb.mean()) if pb.size else 0.0,
        "harmful_proj_std": float(ph.std()) if ph.size else 0.0,
        "benign_proj_std": float(pb.std()) if pb.size else 0.0,
    }


# --------------------------------------------------------------------------- #
# The PROBE gate — the CLAS fix: a LEARNED sensing signal that actually separates.
# --------------------------------------------------------------------------- #
class ProbeGate:
    """Per-prompt harm signal = the TRAINED lesson-1 probe's raw logit.

    This is the fix for the diff-of-means cosine gate's ~zero per-instance class
    separation. CLAS (Hsu et al., arXiv:2604.24693) drives its per-input strength
    from a LEARNED sensing vector; lesson-1's MLP probe IS such a learned sensing
    map (AUC ~0.87 at this layer), so its output separates harmful from benign
    per-instance where the diff-of-means cosine does not.

    We reuse the probe VERBATIM — exactly as the sibling ``flas`` lesson reuses it
    through ``HarmGate``: load the lesson-1 checkpoint (``load_probe`` gives the
    weights + the training scaler + metadata), read the residual at the probe's OWN
    layer (from ``meta``, so gate and probe can never disagree), standardize with
    the stored scaler. The one difference from ``HarmGate.is_harmful`` (which
    returns ``sigmoid(logit) >= threshold``) is that we read the raw PRE-sigmoid
    LOGIT: it is unbounded and monotone in confidence, which makes a cleaner linear
    ramp than the probability, whose 0/1 saturation would flatten the schedule.
    """

    def __init__(self, model: Any, tok: Any,
                 probe_path: "str | Path | None" = None,
                 layer: "int | None" = None) -> None:
        # Deferred import: keep ``import contextual`` torch-free. This mirrors how
        # flas's HarmGate pulls the lesson-1 probe.
        from steering_tutorials.hello_world.probe import load_probe

        self.model = model
        self.tok = tok
        if probe_path is None:  # resolve the lesson-1 artifact relative to this file
            probe_path = (Path(__file__).resolve().parent.parent
                          / "hello_world" / "artifacts" / "probe.pt")
        self.probe_path = Path(probe_path)
        # load_probe -> (probe, scaler, meta); meta carries the training layer.
        self.probe, self.scaler, self.meta = load_probe(str(self.probe_path),
                                                        device="cpu")
        self.layer = int(layer) if layer is not None else int(self.meta["layer"])

    def logit_from_pooled(self, pooled: np.ndarray) -> float:
        """The probe's raw logit for an ALREADY mean-pooled activation.

        Lets the run reuse the pooled extract activations it already computed for
        the diff-of-means direction (same layer) instead of forwarding twice.
        """
        import torch

        feats = np.asarray(pooled, dtype=np.float32).reshape(1, -1)
        x = torch.from_numpy(self.scaler.transform(feats))
        with torch.no_grad():
            return float(self.probe(x).reshape(-1)[0])

    def logit(self, prompt: str) -> float:
        """cos-free per-prompt signal: mean-pool at the probe's layer -> raw logit."""
        from steering_tutorials.hello_world_steering.model_utils import (
            mean_pool_activation,
        )

        pooled = mean_pool_activation(self.model, self.tok, prompt, self.layer)
        return self.logit_from_pooled(pooled)


# --------------------------------------------------------------------------- #
# The steerer — wraps lesson 2's read (mean_pool) + write (generate) helpers.
# --------------------------------------------------------------------------- #
class ContextualSteerer:
    """Generate with an input-adaptive OR a fixed steering strength.

    Holds the loaded model, the unit steering direction ``v_unit``, the layer,
    and the schedule parameters. The two generation methods share everything
    except how ``alpha`` is chosen, which is the whole comparison this lesson
    makes:

      - :meth:`fixed_generate`       — ``alpha = alpha_base`` for every prompt
                                        (the lesson-2-style baseline).
      - :meth:`contextual_generate`  — ``alpha = alpha_base * schedule(proj_i)``
                                        (this lesson's proposal).

    Both delegate the actual residual-stream edit to lesson 2's ``generate``
    (``operation="relative_add"``), so the ONLY thing that varies is the scalar
    ``alpha`` — a clean, single-variable comparison.

    The per-prompt strength SIGNAL is selectable via ``gate``:

      - ``"diffmean_cos"`` — ``proj = cos(mean_pool(h) - center, v_unit)`` (the
        diff-of-means cosine; the honest-negative baseline).
      - ``"probe"``        — ``proj =`` the trained lesson-1 probe's logit (the
        CLAS fix; requires a :class:`ProbeGate` in ``probe_gate``).

    The gate only changes WHICH signal scales ``alpha``. The steering DIRECTION
    written into the residual is always ``v_unit`` (the diff-of-means vector), so
    swapping gates is a clean single-variable change to the schedule alone.
    """

    def __init__(
        self,
        model: Any,
        tok: Any,
        v_unit: np.ndarray,
        layer: int,
        alpha_base: float,
        tau: float,
        ref: float = 1.0,
        cap: float = 1.0,
        center: "np.ndarray | None" = None,
        gate: str = "diffmean_cos",
        probe_gate: "ProbeGate | None" = None,
    ) -> None:
        self.model = model
        self.tok = tok
        self.v_unit = np.asarray(v_unit, dtype=np.float32).reshape(-1)
        self.layer = int(layer)
        self.alpha_base = float(alpha_base)
        self.tau = float(tau)
        self.ref = float(ref)
        self.cap = float(cap)
        # Benign-mean center for the projection (see cosine_projection). Must be
        # the SAME center used to calibrate tau/ref, or the schedule is meaningless.
        self.center = None if center is None else np.asarray(center, dtype=np.float32).reshape(-1)
        self.gate = str(gate)
        self.probe_gate = probe_gate
        if self.gate == "probe" and self.probe_gate is None:
            raise ValueError("gate='probe' requires a ProbeGate instance in probe_gate")

    # -- the per-prompt context signal ------------------------------------- #
    def projection(self, prompt: str) -> float:
        """The per-prompt harm signal for one prompt (one forward pass).

        ``"probe"`` -> the trained probe's logit; ``"diffmean_cos"`` ->
        ``cos(mean_pool(hidden @ layer) − center, v_unit)``.
        """
        if self.gate == "probe":
            return self.probe_gate.logit(prompt)

        from steering_tutorials.hello_world_steering.model_utils import (
            mean_pool_activation,
        )

        pooled = mean_pool_activation(self.model, self.tok, prompt, self.layer)
        return cosine_projection(pooled, self.v_unit, self.center)

    def alpha_for(self, proj: float) -> float:
        """The contextual strength for a given projection (the schedule)."""
        return contextual_alpha(proj, self.alpha_base, self.tau, self.ref, self.cap)

    # -- generation -------------------------------------------------------- #
    def _generate(self, prompt: str, alpha: float, max_new_tokens: int) -> str:
        from steering_tutorials.hello_world_steering.model_utils import generate

        return generate(
            self.model, self.tok, prompt,
            max_new_tokens=max_new_tokens,
            vector=self.v_unit, layer=self.layer, alpha=alpha,
            operation="relative_add",
        )

    def fixed_generate(self, prompt: str, max_new_tokens: int = 48) -> dict:
        """Baseline: steer with ``alpha_base`` regardless of the input."""
        alpha = self.alpha_base
        return {
            "prompt": prompt,
            "alpha": float(alpha),
            "proj": None,
            "response": self._generate(prompt, alpha, max_new_tokens),
        }

    def contextual_generate(self, prompt: str, max_new_tokens: int = 48) -> dict:
        """Proposal: read the projection, scale alpha by the schedule, steer.

        Returns the chosen ``alpha`` and the ``proj`` alongside the text so the
        run can show WHY a benign prompt was barely steered (proj below tau ⇒
        alpha ≈ 0) while a harmful one got the full push.
        """
        proj = self.projection(prompt)
        alpha = self.alpha_for(proj)
        return {
            "prompt": prompt,
            "alpha": float(alpha),
            "proj": float(proj),
            "response": self._generate(prompt, alpha, max_new_tokens),
        }


# --------------------------------------------------------------------------- #
# CPU self-test — NO model download. Exercises the pure schedule math only.
# Run: python -m steering_tutorials.contextual_steering.contextual
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    # (a) cosine_projection: aligned ⇒ +1, anti ⇒ -1, orthogonal ⇒ 0.
    v = np.array([0.0, 3.0, 0.0], dtype=np.float32)   # points along axis 1
    v_unit = v / np.linalg.norm(v)
    assert abs(cosine_projection(np.array([0.0, 5.0, 0.0]), v_unit) - 1.0) < 1e-6
    assert abs(cosine_projection(np.array([0.0, -2.0, 0.0]), v_unit) + 1.0) < 1e-6
    assert abs(cosine_projection(np.array([7.0, 0.0, 0.0]), v_unit)) < 1e-6
    assert cosine_projection(np.zeros(3), v_unit) == 0.0   # degenerate ⇒ 0

    # (b) contextual_alpha with the literal paper form (ref=1, tau=0.2, cap=1):
    #     below tau ⇒ 0; at ref ⇒ alpha_base; linear ramp; negative clipped.
    ab, tau = 0.10, 0.20
    assert contextual_alpha(0.10, ab, tau, ref=1.0) == 0.0          # below tau
    assert contextual_alpha(-0.5, ab, tau, ref=1.0) == 0.0          # negative ⇒ relu 0
    assert abs(contextual_alpha(1.0, ab, tau, ref=1.0) - ab) < 1e-9  # at ref ⇒ full
    mid = contextual_alpha(0.60, ab, tau, ref=1.0)                   # halfway in [tau,1]
    assert abs(mid - ab * ((0.60 - tau) / (1.0 - tau))) < 1e-9
    assert 0.0 < mid < ab

    # (c) the cap: a proj beyond ref does not exceed cap * alpha_base.
    assert abs(contextual_alpha(2.0, ab, tau, ref=0.8, cap=1.0) - ab) < 1e-9
    assert abs(contextual_alpha(2.0, ab, tau, ref=0.8, cap=1.5) - 1.5 * ab) < 1e-9

    # (d) degenerate calibration (ref <= tau) ⇒ hard step at tau.
    assert contextual_alpha(0.30, ab, tau=0.5, ref=0.4) == 0.0       # proj <= tau
    assert abs(contextual_alpha(0.60, ab, tau=0.5, ref=0.4) - ab) < 1e-9  # proj > tau

    # (e) calibrate_schedule: tau sits above benign, ref at harmful mean, and the
    #     resulting schedule steers a typical harmful prompt but not a benign one.
    rng = np.random.default_rng(0)
    proj_benign = rng.normal(0.0, 0.05, size=200)      # benign cluster near 0
    proj_harmful = rng.normal(0.4, 0.05, size=200)     # harmful cluster near 0.4
    sched = calibrate_schedule(proj_harmful, proj_benign, benign_percentile=90.0)
    assert sched["tau"] > sched["benign_proj_mean"]     # floor above benign centre
    assert abs(sched["ref"] - sched["harmful_proj_mean"]) < 1e-9
    a_benign = contextual_alpha(sched["benign_proj_mean"], ab, sched["tau"], sched["ref"])
    a_harmful = contextual_alpha(sched["harmful_proj_mean"], ab, sched["tau"], sched["ref"])
    assert a_benign == 0.0, "typical benign prompt should not be steered"
    assert a_harmful > 0.5 * ab, "typical harmful prompt should be steered hard"

    # (f) THE PROBE-GATE FIX on synthetic probe LOGITS. A well-separating signal
    #     (benign logits low, harmful logits high — what an AUC~0.87 probe gives)
    #     makes ref > tau so the ramp OPERATES (not the degenerate hard step), and
    #     the mean alpha over harmful prompts dominates the mean over benign
    #     (~0). This is the property the diff-of-means cosine gate fails to have.
    rng2 = np.random.default_rng(1)
    benign_logit = rng2.normal(-4.0, 1.0, size=300)    # probe: benign  -> low logit
    harmful_logit = rng2.normal(+4.0, 1.0, size=300)   # probe: harmful -> high logit
    sp = calibrate_schedule(harmful_logit, benign_logit, benign_percentile=90.0)
    assert sp["ref"] > sp["tau"], "separating probe logits must give ref > tau"
    a_harm = [contextual_alpha(x, ab, sp["tau"], sp["ref"], cap=1.0) for x in harmful_logit]
    a_ben = [contextual_alpha(x, ab, sp["tau"], sp["ref"], cap=1.0) for x in benign_logit]
    mah, mab = float(np.mean(a_harm)), float(np.mean(a_ben))
    assert mab < 0.02 * ab, "benign prompts must stay ~unsteered under the probe ramp"
    assert mah > 0.5 * ab, "typical harmful prompt must be steered hard"
    assert mah > 10.0 * max(mab, 1e-9), "harmful alpha must dominate benign alpha (>>)"

    # (g) the harmful-percentile ref option anchors full strength higher up.
    sp_pct = calibrate_schedule(harmful_logit, benign_logit, benign_percentile=90.0,
                                harmful_percentile=90.0)
    assert sp_pct["ref"] > sp["ref"], "90th-pct ref should sit above the mean ref"
    assert sp_pct["harmful_percentile"] == 90.0

    print("[self-test] OK - cosine, ramp, cap, degenerate step, calibration, AND "
          "the probe-logit gate (harm alpha >> benign alpha ~0) all behave.")


if __name__ == "__main__":
    _self_test()
