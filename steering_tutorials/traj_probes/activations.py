"""activations.py -- replay agent trajectories through a decoder and cache the
residual stream at EVERY turn boundary.

WHAT THIS PRODUCES
------------------
One `types.ActivationBundle` per (corpus, layer): a row for every turn of every
trajectory, carrying `step_index` (the turn's position inside its own
trajectory), `traj_uid` and `group_id`. `step_index` is not decoration -- CONTROL
1 of this whole series (step-index residualisation) is impossible without it, so
this module refuses to emit a bundle that lacks it.

THE PREFIX STRATEGY, AND WHY IT IS ONE PASS AND NOT n
-----------------------------------------------------
A probe must be evaluable at every step k, so we need the model's state after
turn 0, after turn 1, ... after turn n-1. The obvious implementation is n forward
passes over n growing prefixes, costing O(n^2) tokens per trajectory. On a corpus
with a median of ~130 steps that is roughly a 130x tax, and it is unnecessary.

We run ONE forward pass over the whole trajectory and read the residual at the
LAST TOKEN OF EACH TURN. In a CAUSAL decoder those are the same numbers: the
hidden state at position t is a function of positions <= t only, so the state at
the last token of turn k in the full-sequence pass is bit-identical to the state
it would have in a pass over the prefix through turn k. Cost drops from
O(n * total_tokens) to O(total_tokens).

That identity is a PROPERTY OF THE ATTENTION MASK, not of the code, and this repo
has already been burned once by assuming an attention mode instead of measuring
it: EmbeddingGemma ran strictly causal for weeks while its config field said
bidirectional (audits/AUDIT_2026-08-17_embeddinggemma_causal.md). Here the
polarity is reversed -- we REQUIRE causal, and a bidirectional (or
prefix-LM/sliding-window-with-lookahead) model would make the one-pass trick
silently wrong, producing well-formed activations that are not the states they
are labelled as.

So `verify_prefix_equivalence()` MEASURES it: one full pass and one truncated
pass, compared at the boundary positions themselves. It runs once per extract()
call, its result is a REQUIRED field of the cache's behaviour fingerprint, and a
non-causal model is refused rather than quietly re-interpreted. The self-test
plants a deliberately bidirectional toy block to prove the guard fires.

POOLING
-------
  "last"        (default) the last token of turn k. The decision-relevant
                position, and what reproductions A and B read.
  "mean_turn"   mean over turn k's own tokens.
  "mean_prefix" mean over every token up to the end of turn k. Available, and
                CONFOUNDED WITH STEP INDEX BY CONSTRUCTION: as k grows the pool
                grows, so the feature encodes k directly. Offered only so the
                confound can be demonstrated; do not headline it.

CACHING
-------
Keyed via `common.artifact_paths.keyed_path` (model, layer, pooling, stack hash),
so two arms cannot overwrite each other. Two fingerprints are stored and BOTH are
checked on read, each rejected loudly by name:
  * behaviour -- library versions + model id + the MEASURED prefix-equivalence
    bucket. A version string alone is what failed in August.
  * data      -- corpus pool fingerprint, uids, labels, turn counts, and the
    extraction settings. An artifact that cannot be attributed is not evidence.

RESUMABILITY
------------
This host reaps long background jobs (CLAUDE.md 18.5), so a reap must cost one
trajectory, not the run. Rows are appended to a raw `.rows.f32` journal, flushed
and fsynced, and only THEN is the trajectory's index line appended and fsynced.
On resume we accept only the CONTIGUOUS prefix of well-formed index lines whose
rows actually exist on disk, and truncate the rows file back to exactly that many
rows. A torn write therefore loses a trajectory; it can never shift activations
against labels, which is the failure that would be invisible.

CPU-only to import. Loads no model at import. ASCII stdout (Windows cp1252).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import steering_tutorials.traj_probes.config as _C

# --- sibling / cousin imports, tolerant of script-vs-package invocation ------
try:  # package form
    from .types import ActivationBundle
except ImportError:  # pragma: no cover - direct-script form
    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE.parent.parent))
from steering_tutorials.traj_probes.types import ActivationBundle

try:
    from ..common.artifact_paths import keyed_path
    from ..biencoder_guard import encoder_behaviour as _behaviour
except (ImportError, ValueError):  # pragma: no cover
    from steering_tutorials.common.artifact_paths import keyed_path
    from steering_tutorials.biencoder_guard import encoder_behaviour as _behaviour

__all__ = [
    "POOLINGS", "CAUSAL_TOL", "JOURNAL_FORMAT_VERSION",
    "ExtractSettings", "HFActivationExtractor", "RowJournal",
    "turn_char_ends", "boundary_token_indices", "verify_prefix_equivalence",
    "behaviour_block", "data_fingerprint", "bundle_cache_path",
    "save_bundle", "load_bundle",
]

POOLINGS = ("last", "mean_turn", "mean_prefix")

# RELATIVE difference at a boundary position between the full pass and the
# truncated pass that still counts as "the one-pass trick is exact". Relative,
# not absolute: Gemma's residual entries run to tens, so an absolute epsilon
# would be a different test at every layer. The causal value is exactly 0.0 in
# fp32; the tolerance only absorbs bf16 reduction jitter between two different
# sequence lengths (bf16 carries ~8 mantissa bits, so ~4e-3 relative per op).
# Real lookahead is an order-1 relative difference, so this separates by orders
# of magnitude rather than by a hair.
CAUSAL_TOL = 5e-2

JOURNAL_FORMAT_VERSION = 1


def _eprint(*args) -> None:
    """ASCII-only stderr print. This host's cp1252 console dies on unicode."""
    msg = " ".join(str(a) for a in args)
    try:
        print(msg, file=sys.stderr)
    except Exception:  # pragma: no cover
        print(msg.encode("ascii", "replace").decode("ascii"), file=sys.stderr)


# ---------------------------------------------------------------------------
# 1. Turn boundaries in the SAME string the content bar reads
# ---------------------------------------------------------------------------
def turn_char_ends(traj) -> tuple:
    """-> (text, tuple of per-turn END character offsets into that text).

    Rebuilds `AgentTrajectory.text` incrementally so the boundaries are exact,
    then ASSERTS the reconstruction is byte-identical to the property. That
    assert is the anchor: if `types.py` ever changes its rendering, this fails
    loudly instead of silently indexing the wrong tokens (CLAUDE.md 18.8 --
    "assert your anchors").

    The same string feeds the model AND `common.confound.content_bar`; if they
    diverged the confound comparison would not be a comparison.
    """
    parts, ends, cursor = [], [], 0
    for i, t in enumerate(traj.turns):
        seg = "%s: %s" % (t.role, t.content)
        if i:
            cursor += 1          # the "\n" join separator
            parts.append("\n")
        parts.append(seg)
        cursor += len(seg)
        ends.append(cursor)
    text = "".join(parts)
    if text != traj.text:
        raise AssertionError(
            "turn_char_ends rebuilt a string that differs from "
            "AgentTrajectory.text (len %d vs %d). The rendering in types.py "
            "changed; boundary offsets computed here would index the wrong "
            "tokens." % (len(text), len(traj.text)))
    return text, tuple(ends)


def boundary_token_indices(tok, text: str, char_ends, max_tokens: int):
    """-> (ids list, boundary index per turn or None, n_truncated_turns).

    Preferred path uses the fast tokenizer's `offset_mapping`, so the token
    sequence is the ordinary monolithic tokenization -- no per-segment merge
    artefacts. The boundary for turn k is the LAST token whose character span
    ends at or before that turn's end offset, ignoring special tokens (which
    carry a degenerate (0, 0) span).

    Fallback for a slow tokenizer: tokenize each turn segment separately and
    concatenate. Boundaries are exact by construction there too, at the cost of
    possibly different merges across a boundary. Which path ran is recorded in
    the bundle meta -- it is a property of the instrument, not a detail.
    """
    ids, offsets, path = None, None, "offsets"
    try:
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
        ids = list(enc["input_ids"])
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        if len(offsets) != len(ids):
            raise ValueError("offset_mapping length %d != ids %d"
                             % (len(offsets), len(ids)))
    except Exception:
        ids, offsets, path = None, None, "segments"

    if path == "offsets":
        bounds = []
        j = 0
        for end in char_ends:
            best = None
            while j < len(ids):
                a, b = offsets[j]
                if b > end:
                    break
                if b > a:                      # skip specials, span (0, 0)
                    best = j
                j += 1
            bounds.append(best)
        # A turn whose tokens all land past a previous boundary can legitimately
        # yield None only when the turn is empty; carry the previous boundary
        # forward so an empty turn does not silently drop a step.
        prev = None
        for i, b in enumerate(bounds):
            if b is None:
                bounds[i] = prev
            else:
                prev = bounds[i]
    else:
        ids, bounds = [], []
        segs, cursor = [], 0
        for k, end in enumerate(char_ends):
            segs.append(text[cursor:end])
            cursor = end + 1                   # skip the "\n"
        for k, seg in enumerate(segs):
            piece = tok(seg if k == 0 else "\n" + seg,
                        add_special_tokens=(k == 0))["input_ids"]
            ids.extend(list(piece))
            bounds.append(len(ids) - 1 if ids else None)

    n_trunc = 0
    if max_tokens and len(ids) > max_tokens:
        ids = ids[:max_tokens]
        cut = []
        for b in bounds:
            if b is not None and b < max_tokens:
                cut.append(b)
            else:
                cut.append(None)
                n_trunc += 1
        bounds = cut
    return ids, bounds, n_trunc


# ---------------------------------------------------------------------------
# 2. The causal guard -- measured, never assumed
# ---------------------------------------------------------------------------
def verify_prefix_equivalence(model, tok, text: str, boundaries,
                              layer: int, n_probe: int = 3) -> dict:
    """Measure whether reading turn-boundary states from ONE pass is exact.

    Runs the full token sequence, then the sequence truncated at each probed
    boundary, and compares the layer-`layer` residual AT that boundary position.
    Under causal attention the two are the same computation and the relative
    difference is ~0; under any lookahead it is order-1.

    SEVERAL boundaries are probed, not one, and the WORST is the verdict. A
    single position is easy to pass by accident: the first version of this
    module's own self-test compared at a position whose token happened to equal
    the sequence's last token, so a deliberately bidirectional toy scored
    EXACTLY 0.0 and cleared the guard. The check was well-formed, raised no
    error, and measured nothing -- the failure shape this repo keeps meeting
    (CLAUDE.md 18.8). Probing spread-out boundaries makes that coincidence have
    to hold at every one of them at once.

    -> dict with `max_abs`, `rel`, `causal` (bool) and the per-position detail.
    Never raises on a numerical result; the CALLER decides, so the measurement is
    always recorded even when it fails the gate.
    """
    ids_all, _b, _t = boundary_token_indices(tok, text, (len(text),), 0)
    if boundaries is None or isinstance(boundaries, (int, np.integer)):
        boundaries = [boundaries]
    cand = [int(b) for b in boundaries
            if b is not None and 1 <= int(b) < len(ids_all) - 1]
    if not cand:
        cand = [max(1, len(ids_all) // 2)]
    if len(cand) > n_probe:                       # spread, do not take the head
        step = len(cand) / float(n_probe)
        cand = [cand[min(len(cand) - 1, int(i * step))] for i in range(n_probe)]
    cand = sorted(set(cand))

    h_full = _forward_residual(model, ids_all, layer)
    per_pos, worst_rel, worst_abs = [], 0.0, 0.0
    for b in cand:
        h_pref = _forward_residual(model, ids_all[:b + 1], layer)
        a = h_full[b].astype(np.float64)
        c = h_pref[b].astype(np.float64)
        m = float(np.max(np.abs(a - c)))
        scale = float(np.max(np.abs(a))) or 1.0
        per_pos.append({"position": int(b), "max_abs": m, "rel": m / scale})
        worst_rel = max(worst_rel, m / scale)
        worst_abs = max(worst_abs, m)
    return {
        "max_abs": worst_abs,
        "rel": worst_rel,
        "position": int(max(per_pos, key=lambda d: d["rel"])["position"]),
        "positions": per_pos,
        "n_tokens_full": int(len(ids_all)),
        "causal": bool(worst_rel <= CAUSAL_TOL),
        "tol": CAUSAL_TOL,
    }


def _forward_residual(model, ids, layer: int, keep=None):
    """One forward pass; -> [seq, hidden] float32 residual after `layer`.

    With `keep` (an iterable of token positions), returns ONLY those rows plus a
    {position: row_index} map, and never materialises the full sequence in
    float32. That matters at long context: a 16k-token trajectory is
    16384 x 1152 floats = 75 MB per `.float()` copy on the GPU, on top of
    attention, and the "last" pooling needs about SIXTEEN of those rows. Keeping
    the whole thing was what made the layer sweep OOM at MAX_TOKENS=16384 while
    the same corpus ran fine at 4096.
    """
    import torch

    try:
        from ..hello_world_steering.model_utils import residual_layers
    except (ImportError, ValueError):  # pragma: no cover
        from steering_tutorials.hello_world_steering.model_utils import residual_layers

    device = next(model.parameters()).device
    layers = residual_layers(model)
    idx = max(0, min(int(layer), len(layers) - 1))
    cap = {}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        cap["h"] = h.detach()

    handle = layers[idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            t = torch.tensor([list(ids)], dtype=torch.long, device=device)
            model(t)
    finally:
        handle.remove()
    h = cap.pop("h")[0]
    if keep is None:
        out = h.float().cpu().numpy().astype(np.float32)
        del h
        return out
    pos = sorted({int(k) for k in keep if k is not None})
    sel = torch.as_tensor(pos, dtype=torch.long, device=h.device)
    sub = h.index_select(0, sel).float().cpu().numpy().astype(np.float32)
    del h, sel
    return sub, {p: i for i, p in enumerate(pos)}


def _forward_residual_pooled(model, ids, layer: int, bounds, pooling: str):
    """One forward pass; -> [n_turns, hidden] float32, pooled ON THE DEVICE.

    Returns one row PER TURN INDEX (aligned to `bounds`, so row k is turn k),
    with rows for turns whose boundary is None left as zeros -- callers skip
    those by the same `bounds[k] is None` test they already use.

    Exists so mean_turn / mean_prefix never materialise the whole sequence in
    float32 on the GPU. The arithmetic is done in float32 on the device, which
    matches what the CPU path computed, and only [n_turns, hidden] comes back.
    """
    import torch

    try:
        from ..hello_world_steering.model_utils import residual_layers
    except (ImportError, ValueError):  # pragma: no cover
        from steering_tutorials.hello_world_steering.model_utils import residual_layers

    device = next(model.parameters()).device
    layers = residual_layers(model)
    idx = max(0, min(int(layer), len(layers) - 1))
    cap = {}

    def hook(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = layers[idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            t = torch.tensor([list(ids)], dtype=torch.long, device=device)
            model(t)
    finally:
        handle.remove()
    h = cap.pop("h")[0]
    out = torch.zeros((len(bounds), h.shape[1]), dtype=torch.float32,
                      device=h.device)
    prev = None
    for k, b in enumerate(bounds):
        if b is None:
            continue
        b = int(b)
        if pooling == "mean_prefix":
            lo = 0
        else:                                   # mean_turn
            lo = 0 if prev is None else min(prev + 1, b)
        out[k] = h[lo:b + 1].float().mean(dim=0)
        prev = b
    res = out.cpu().numpy().astype(np.float32)
    del h, out
    return res


# ---------------------------------------------------------------------------
# 3. Fingerprints
# ---------------------------------------------------------------------------
def behaviour_block(model_id: str, layer: int, pooling: str,
                    equivalence: dict | None = None) -> dict:
    """Library versions + model id + the MEASURED prefix-equivalence bucket.

    `_behaviour.cheap_fingerprint` supplies the half that needs no model. The
    half that needs one -- whether the one-pass prefix trick is actually exact
    on THIS stack -- is the field a version string cannot stand in for.
    """
    block = _behaviour.cheap_fingerprint(
        model_id, extra={"layer": int(layer), "pooling": str(pooling),
                         "extractor_version": JOURNAL_FORMAT_VERSION})
    if equivalence is None:
        block["prefix_equivalence"] = _behaviour.NOT_PROBED
        block["prefix_equivalence_max_abs"] = None
    else:
        block["prefix_equivalence"] = ("causal" if equivalence["causal"]
                                       else _behaviour.delta_bucket(equivalence["max_abs"]))
        block["prefix_equivalence_max_abs"] = round(float(equivalence["max_abs"]), 8)
    return block


def compare_behaviour(want: dict, got: dict | None) -> list:
    """-> list of "field: expected X, cache has Y". Empty means compatible.

    Delegates the shared fields to `encoder_behaviour.compare` (so a transformers
    bump is caught by the same code that caught the August incident) and adds the
    equivalence field. `not-probed` on either side skips only that field: a cache
    READ must never have to load a model.
    """
    diffs = list(_behaviour.compare(want, got))
    if not got:
        return diffs
    w, g = want.get("prefix_equivalence"), got.get("prefix_equivalence")
    if _behaviour.NOT_PROBED not in (w, g) and str(w) != str(g):
        diffs.append(
            "prefix_equivalence: expected %s, cache has %s (whether one-pass "
            "turn-boundary reads are EXACT changed -- a non-causal model makes "
            "every cached row a state it is not labelled as)" % (w, g))
    for field in ("layer", "pooling", "extractor_version"):
        if str(want.get(field)) != str(got.get(field)):
            diffs.append("%s: expected %s, cache has %s"
                         % (field, want.get(field), got.get(field)))
    return diffs


def data_fingerprint(corpus, settings) -> str:
    """12 hex over everything that changes WHICH rows should exist.

    Includes each trajectory's uid, label, group and turn count, so a corpus
    reshuffle, a relabel or a turn-count change all invalidate the cache. The
    pool fingerprint alone would not: `meerkat` shipped a results file that could
    not be regenerated from the code beside it for exactly this reason.
    """
    h = hashlib.sha256()
    h.update(("corpus=%s|pool=%s|n=%d|"
              % (corpus.name, corpus.pool_fingerprint,
                 len(corpus.trajectories))).encode("utf-8"))
    h.update(json.dumps(settings.as_dict(), sort_keys=True).encode("utf-8"))
    for t in corpus.trajectories:
        h.update(("%s|%d|%s|%d|%d;" % (t.uid, int(t.label), t.group_id,
                                       t.n_turns, len(t.text))).encode("utf-8"))
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# 4. Settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtractSettings:
    """Everything that changes the NUMBERS, and therefore the cache identity."""

    pooling: str = "last"
    max_tokens: int = _C.MAX_TOKENS
    row_roles: tuple = ()        # () = a row for every turn
    layer: int = 12

    def __post_init__(self) -> None:
        if self.pooling not in POOLINGS:
            raise ValueError("pooling must be one of %s, got %r"
                             % (POOLINGS, self.pooling))

    def as_dict(self) -> dict:
        return {"pooling": self.pooling, "max_tokens": int(self.max_tokens),
                "row_roles": list(self.row_roles), "layer": int(self.layer)}


# ---------------------------------------------------------------------------
# 5. The resumable journal
# ---------------------------------------------------------------------------
class RowJournal:
    """Append-only row store + index, safe against a reap at any instant.

    Two files beside the cache:
      `<stem>.rows.f32`    raw float32, `hidden` values per row, no header.
      `<stem>.index.jsonl` line 0 is the header; then one line per COMPLETED
                           trajectory.

    ORDERING IS THE WHOLE POINT. Rows are written, flushed and fsynced BEFORE the
    index line that claims them. So the only reachable inconsistent state is
    "rows on disk that no index line claims", which resume truncates away. The
    reverse -- an index line whose rows were never written -- would slide every
    later row one place against its label, and is unreachable by construction.
    Resume still checks for it anyway and refuses, because "unreachable" is what
    was said about a config field that said bidirectional.
    """

    def __init__(self, stem, hidden: int, header: dict):
        self.stem = Path(stem)
        self.hidden = int(hidden)
        self.header = dict(header)
        self.rows_path = self.stem.with_suffix(self.stem.suffix + ".rows.f32")
        self.index_path = self.stem.with_suffix(self.stem.suffix + ".index.jsonl")
        self._row_bytes = self.hidden * 4
        self.entries: list = []
        self._rows_fh = None
        self._index_fh = None

    # -- resume ------------------------------------------------------------
    def open_or_resume(self) -> set:
        """-> set of trajectory uids already safely on disk. Truncates the tail."""
        self.stem.parent.mkdir(parents=True, exist_ok=True)
        entries, header_ok, refuse = [], False, ""
        if self.index_path.exists():
            n_rows_on_disk = (self.rows_path.stat().st_size // self._row_bytes
                              if self.rows_path.exists() else 0)
            cum = 0
            with open(self.index_path, "r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        break                      # a torn write: stop here
                    try:
                        rec = json.loads(line)
                    except Exception:
                        _eprint("[journal] index line %d is torn -- accepting only "
                                "the contiguous prefix before it" % lineno)
                        break
                    if lineno == 0:
                        stored = rec.get("__header__")
                        if stored is None:
                            _eprint("[journal] no header on line 0 -- discarding")
                            break
                        diffs = compare_behaviour(self.header.get("behaviour", {}),
                                                  stored.get("behaviour"))
                        if stored.get("data_fingerprint") != self.header.get("data_fingerprint"):
                            diffs.append("data_fingerprint: expected %s, journal has %s"
                                         % (self.header.get("data_fingerprint"),
                                            stored.get("data_fingerprint")))
                        if int(stored.get("hidden", -1)) != self.hidden:
                            diffs.append("hidden: expected %d, journal has %s"
                                         % (self.hidden, stored.get("hidden")))
                        if diffs:
                            # Quarantine AFTER the `with` closes -- Windows
                            # refuses to rename an open file (WinError 32) and
                            # the failure path would leave the stale journal in
                            # place, which is the state this refuses to allow.
                            refuse = "; ".join(diffs)
                            entries = []
                            break
                        header_ok = True
                        continue
                    need = cum + int(rec["n_rows"])
                    if need > n_rows_on_disk:
                        _eprint("[journal] index claims %d rows but only %d are on "
                                "disk -- refusing the claim and stopping at %d "
                                "trajectories" % (need, n_rows_on_disk, len(entries)))
                        break
                    cum = need
                    entries.append(rec)
            if refuse:
                _eprint("[journal] REFUSING to resume %s -- %s"
                        % (self.index_path.name, refuse))
                _behaviour.quarantine(self.index_path, tag="mismatch")
                _behaviour.quarantine(self.rows_path, tag="mismatch")
            if header_ok and self.rows_path.exists():
                keep = cum * self._row_bytes
                if self.rows_path.stat().st_size != keep:
                    _eprint("[journal] truncating %s from %d to %d bytes (rows with "
                            "no index line are unattributable)"
                            % (self.rows_path.name, self.rows_path.stat().st_size, keep))
                    os.truncate(str(self.rows_path), keep)
            if not header_ok:
                entries = []
                for p in (self.index_path, self.rows_path):
                    if p.exists():
                        p.unlink()

        self.entries = entries
        fresh = not self.index_path.exists() or not entries and not header_ok
        self._rows_fh = open(self.rows_path, "ab")
        self._index_fh = open(self.index_path, "a", encoding="utf-8")
        if fresh and self.index_path.stat().st_size == 0:
            self._write_index({"__header__": dict(self.header,
                                                  hidden=self.hidden,
                                                  journal_version=JOURNAL_FORMAT_VERSION)})
        if entries:
            _eprint("[journal] resuming with %d trajectories (%d rows) already done"
                    % (len(entries), cum))
        return set(e["uid"] for e in entries)

    # -- append ------------------------------------------------------------
    def append(self, uid: str, rows: np.ndarray, y: int, group_id: str,
               step_index, meta: dict | None = None) -> None:
        """Rows first (fsynced), THEN the index line (fsynced). Never reorder."""
        rows = np.ascontiguousarray(np.asarray(rows, dtype=np.float32))
        if rows.ndim != 2 or rows.shape[1] != self.hidden:
            raise ValueError("journal expects [n, %d] rows, got %s"
                             % (self.hidden, rows.shape))
        if len(step_index) != len(rows):
            raise ValueError("step_index has %d entries for %d rows"
                             % (len(step_index), len(rows)))
        self._rows_fh.write(rows.tobytes(order="C"))
        self._sync(self._rows_fh)
        rec = {"uid": str(uid), "n_rows": int(len(rows)), "y": int(y),
               "group_id": str(group_id),
               "step_index": [int(s) for s in step_index]}
        if meta:
            rec["meta"] = meta
        self._write_index(rec)
        self.entries.append(rec)

    def _write_index(self, rec: dict) -> None:
        self._index_fh.write(json.dumps(rec, sort_keys=True) + "\n")
        self._sync(self._index_fh)

    @staticmethod
    def _sync(fh) -> None:
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:  # pragma: no cover - some filesystems refuse
            pass

    # -- finish ------------------------------------------------------------
    def close(self) -> None:
        for fh in (self._rows_fh, self._index_fh):
            if fh is not None:
                try:
                    fh.close()
                except Exception:  # pragma: no cover
                    pass
        self._rows_fh = self._index_fh = None

    def materialise(self) -> dict:
        """-> dict of aligned arrays built from the journal on disk."""
        total = sum(int(e["n_rows"]) for e in self.entries)
        raw = np.fromfile(str(self.rows_path), dtype=np.float32,
                          count=total * self.hidden)
        if raw.size != total * self.hidden:
            raise RuntimeError(
                "journal rows file holds %d floats but the index claims %d. "
                "Refusing to reshape -- a short read here would silently offset "
                "every row against its label." % (raw.size, total * self.hidden))
        X = raw.reshape(total, self.hidden)
        y, step, uid, grp = [], [], [], []
        for e in self.entries:
            n = int(e["n_rows"])
            y.extend([int(e["y"])] * n)
            step.extend(int(s) for s in e["step_index"])
            uid.extend([e["uid"]] * n)
            grp.extend([e["group_id"]] * n)
        return {"X": X, "y": np.asarray(y, dtype=np.int64),
                "step_index": np.asarray(step, dtype=np.int64),
                "traj_uid": np.asarray(uid, dtype=object),
                "group_id": np.asarray(grp, dtype=object)}

    def discard(self) -> None:
        """Remove the journal after the .npz is safely written."""
        self.close()
        for p in (self.rows_path, self.index_path):
            try:
                if p.exists():
                    p.unlink()
            except OSError:  # pragma: no cover
                pass


# ---------------------------------------------------------------------------
# 6. Cache I/O
# ---------------------------------------------------------------------------
def bundle_cache_path(cache_dir, model_id: str, settings: ExtractSettings,
                      layer: int, data_fp: str) -> Path:
    """Per-arm cache path. `keyed_path` refuses an empty variant by design."""
    tag = str(model_id).rstrip("/").split("/")[-1]
    stack = _behaviour.key_component(model_id, extra={"pooling": settings.pooling,
                                                      "extractor": JOURNAL_FORMAT_VERSION})
    return keyed_path(cache_dir, "acts", ".npz", tag, "L%d" % int(layer),
                      settings.pooling, stack, data_fp)


def save_bundle(path, bundle: ActivationBundle, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path), X=bundle.X, y=bundle.y, step_index=bundle.step_index,
        traj_uid=np.asarray([str(u) for u in bundle.traj_uid]),
        group_id=np.asarray([str(g) for g in bundle.group_id]),
        meta=np.asarray(json.dumps(meta, sort_keys=True)))


def load_bundle(path, want_behaviour: dict, want_data_fp: str,
                quarantine_on_mismatch: bool = True):
    """-> (ActivationBundle, meta) or None. REFUSES a disagreeing cache, loudly.

    Rejection names the offending field. A cache with no fingerprint block at all
    predates this module and is unattributable, which is a rejection and not a
    pass -- the same rule `encoder_behaviour` applies.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        z = np.load(str(path), allow_pickle=False)
        meta = json.loads(str(z["meta"]))
    except Exception as exc:
        _eprint("[cache] %s is unreadable (%s) -- ignoring it" % (path.name, exc))
        return None

    diffs = compare_behaviour(want_behaviour, meta.get("behaviour"))
    if meta.get("data_fingerprint") != want_data_fp:
        diffs.append("data_fingerprint: expected %s, cache has %s (the corpus, "
                     "its labels, its turn counts or the extraction settings "
                     "changed)" % (want_data_fp, meta.get("data_fingerprint")))
    if diffs:
        _eprint("[cache] REFUSING %s -- %s" % (path.name, "; ".join(diffs)))
        if quarantine_on_mismatch:
            _behaviour.quarantine(path, tag="stale")
        return None

    bundle = ActivationBundle(
        X=z["X"], y=z["y"], step_index=z["step_index"],
        traj_uid=z["traj_uid"], group_id=z["group_id"],
        layer=int(meta["layer"]), model_id=str(meta["model_id"]),
        behaviour_fingerprint=str(meta["behaviour_key"]))
    return bundle, meta


# ---------------------------------------------------------------------------
# 7. The extractor
# ---------------------------------------------------------------------------
class HFActivationExtractor:
    """Implements `types.ActivationExtractor` against a HuggingFace decoder.

    The model is loaded LAZILY on the first `extract` that misses cache, so
    importing this module, building the object and answering a cache hit all cost
    zero GPU and zero RAM. A pre-loaded (model, tok) pair can be injected, which
    is what the self-test does with a toy module.
    """

    def __init__(self, model_id: str, settings: ExtractSettings | None = None,
                 cache_dir=None, model=None, tok=None, log_every: int = 25):
        self.model_id = str(model_id)
        self.settings = settings or ExtractSettings()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model, self._tok = model, tok
        self.log_every = int(log_every)
        self.last_meta: dict = {}

    # -- model -------------------------------------------------------------
    def _ensure_model(self):
        if self._model is None or self._tok is None:
            try:
                from ..hello_world_steering.model_utils import load_model
            except (ImportError, ValueError):  # pragma: no cover
                from steering_tutorials.hello_world_steering.model_utils import load_model
            self._model, self._tok = load_model(self.model_id)
        return self._model, self._tok

    # -- the row for one turn ---------------------------------------------
    def _pool(self, H: np.ndarray, bounds, k: int) -> np.ndarray:
        b = bounds[k]
        if self.settings.pooling == "last":
            return H[b]
        prev = None
        for j in range(k - 1, -1, -1):
            if bounds[j] is not None:
                prev = bounds[j]
                break
        start = 0 if prev is None else prev + 1
        if self.settings.pooling == "mean_turn":
            lo = min(start, b)
            return H[lo:b + 1].mean(axis=0)
        return H[:b + 1].mean(axis=0)          # mean_prefix

    # -- the protocol method ----------------------------------------------
    def extract(self, corpus, layer: int) -> ActivationBundle:
        settings = ExtractSettings(pooling=self.settings.pooling,
                                   max_tokens=self.settings.max_tokens,
                                   row_roles=self.settings.row_roles,
                                   layer=int(layer))
        data_fp = data_fingerprint(corpus, settings)
        want_cheap = behaviour_block(self.model_id, layer, settings.pooling, None)

        cache_path = None
        if self.cache_dir is not None:
            cache_path = bundle_cache_path(self.cache_dir, self.model_id,
                                           settings, layer, data_fp)
            hit = load_bundle(cache_path, want_cheap, data_fp)
            if hit is not None:
                bundle, meta = hit
                self.last_meta = meta
                _eprint("[acts] cache HIT %s (%d rows, %d trajectories)"
                        % (cache_path.name, len(bundle.X),
                           len(set(bundle.traj_uid.tolist()))))
                return bundle

        model, tok = self._ensure_model()
        try:
            from ..hello_world_steering.model_utils import hidden_size
        except (ImportError, ValueError):  # pragma: no cover
            from steering_tutorials.hello_world_steering.model_utils import hidden_size
        hidden = int(hidden_size(model))

        # --- the causal guard, measured once, before any row is written ----
        probe_traj = max(corpus.trajectories, key=lambda t: t.n_turns)
        p_text, p_ends = turn_char_ends(probe_traj)
        _pids, p_bounds, _pt = boundary_token_indices(tok, p_text, p_ends,
                                                      settings.max_tokens)
        equiv = verify_prefix_equivalence(model, tok, p_text, p_bounds, layer)
        if not equiv["causal"]:
            raise RuntimeError(
                "PREFIX EQUIVALENCE FAILED: reading turn-boundary states from a "
                "single full-sequence pass differs from the truncated pass by "
                "%.6e (rel %.3e, tolerance is on the RELATIVE figure) at "
                "position %d, tol %.0e. This model does "
                "not attend causally, so every one-pass row would be a state "
                "computed with LOOKAHEAD at information the agent had not "
                "produced yet -- well-formed activations that are not what they "
                "are labelled as. Refusing to extract."
                % (equiv["max_abs"], equiv["rel"], equiv["position"],
                   CAUSAL_TOL))
        _eprint("[acts] prefix equivalence OK: worst rel=%.3e (max|delta|=%.3e) "
                "over %d probed boundaries, rel tol %.0e -- one pass per "
                "trajectory is exact"
                % (equiv["rel"], equiv["max_abs"], len(equiv["positions"]),
                   CAUSAL_TOL))

        behaviour = behaviour_block(self.model_id, layer, settings.pooling, equiv)
        header = {"behaviour": behaviour, "data_fingerprint": data_fp,
                  "model_id": self.model_id, "layer": int(layer)}

        # WHERE THE JOURNAL LIVES. With a cache_dir it sits beside the .npz and
        # a reap costs one trajectory. WITHOUT one there is no durable artifact
        # to resume into, so it goes to a scratch directory that is deleted on
        # the way out.
        #
        # The first version put it in os.getcwd(). That wrote journal files into
        # the repo root AND made two unrelated cache_dir=None runs resume each
        # other -- the second run found a complete journal, skipped every
        # trajectory, ran zero forward passes and returned a full, correct-
        # looking bundle it had not computed. Caught by a forward-pass counter
        # going negative; nothing else about the run looked wrong.
        scratch = None
        if cache_path is not None:
            stem = cache_path
        else:
            import tempfile
            scratch = tempfile.mkdtemp(prefix="traj_probes_nocache_")
            stem = Path(scratch) / ("acts_%s_L%d.npz" % (data_fp, int(layer)))
            _eprint("[acts] cache_dir is None: this run is NOT resumable, and a "
                    "reap costs the whole run. Pass cache_dir to get the "
                    "journal.")
        journal = RowJournal(stem, hidden, header)
        done = journal.open_or_resume()

        wanted_roles = set(settings.row_roles) if settings.row_roles else None
        n_trunc_total, t0 = 0, time.time()
        try:
            for i, traj in enumerate(corpus.trajectories):
                if traj.uid in done:
                    continue
                text, ends = turn_char_ends(traj)
                ids, bounds, n_trunc = boundary_token_indices(
                    tok, text, ends, settings.max_tokens)
                n_trunc_total += n_trunc
                if not ids or all(b is None for b in bounds):
                    _eprint("[acts] skipping %s: no usable turn boundary" % traj.uid)
                    continue
                # "last" needs only the boundary rows, so do not drag the whole
                # sequence back in float32 (see _forward_residual).
                if self.settings.pooling == "last":
                    H, posmap = _forward_residual(
                        model, ids, layer,
                        keep=[b for b in bounds if b is not None])
                else:
                    # mean_turn / mean_prefix need spans, not points, so the
                    # boundary-only trick does not apply. Pool on the DEVICE in
                    # the model's own dtype and bring back only the pooled rows:
                    # a full float32 copy of a 16k-token sequence is what made
                    # this OOM at MAX_TOKENS=16384.
                    H, posmap = _forward_residual_pooled(
                        model, ids, layer, bounds, self.settings.pooling), "span"
                rows, steps = [], []
                for k, turn in enumerate(traj.turns):
                    if k >= len(bounds) or bounds[k] is None:
                        continue
                    if wanted_roles is not None and turn.role not in wanted_roles:
                        continue
                    if posmap == "span":
                        rows.append(H[k])
                    elif posmap is not None:
                        rows.append(H[posmap[int(bounds[k])]])
                    else:
                        rows.append(self._pool(H, bounds, k))
                    steps.append(turn.index)
                del H
                if not rows:
                    continue
                journal.append(traj.uid, np.stack(rows), traj.label,
                               traj.group_id, steps,
                               meta={"n_tokens": len(ids), "n_trunc": n_trunc})
                if self.log_every and (i + 1) % self.log_every == 0:
                    _eprint("[acts] %d/%d trajectories (%.1fs)"
                            % (i + 1, len(corpus.trajectories), time.time() - t0))
            arrays = journal.materialise()
        finally:
            journal.close()
            if scratch is not None:
                import shutil
                shutil.rmtree(scratch, ignore_errors=True)

        bundle = ActivationBundle(
            X=arrays["X"], y=arrays["y"], step_index=arrays["step_index"],
            traj_uid=arrays["traj_uid"], group_id=arrays["group_id"],
            layer=int(layer), model_id=self.model_id,
            behaviour_fingerprint=_behaviour.key_component(
                self.model_id, extra={"pooling": settings.pooling,
                                      "extractor": JOURNAL_FORMAT_VERSION}))

        meta = {
            "behaviour": behaviour,
            "behaviour_key": bundle.behaviour_fingerprint,
            "data_fingerprint": data_fp,
            "model_id": self.model_id,
            "layer": int(layer),
            "settings": settings.as_dict(),
            "corpus": {"name": corpus.name,
                       "pool_fingerprint": corpus.pool_fingerprint,
                       "licence": corpus.licence,
                       "label_provenance": corpus.label_provenance,
                       "n_trajectories": len(corpus.trajectories),
                       "turns": corpus.turn_count_summary()},
            "prefix_equivalence": equiv,
            "n_rows": int(len(bundle.X)),
            "n_turns_truncated": int(n_trunc_total),
            "elapsed_sec": round(time.time() - t0, 2),
        }
        self.last_meta = meta
        if cache_path is not None:
            save_bundle(cache_path, bundle, meta)
            journal.discard()
            _eprint("[acts] wrote %s (%d rows, %d trajectories, %d turns "
                    "truncated)" % (cache_path.name, len(bundle.X),
                                    len(corpus.trajectories), n_trunc_total))
        return bundle


# ---------------------------------------------------------------------------
# CPU self-test -- NO model download, NO GPU, NO network.
#   python -m steering_tutorials.traj_probes.activations
# ---------------------------------------------------------------------------
def _self_test() -> None:  # noqa: C901 - a test, read top to bottom
    import shutil
    import tempfile
    from types import SimpleNamespace

    import torch
    import torch.nn as nn

    try:
        from .types import AgentTrajectory, TrajCorpus, Turn
    except ImportError:  # pragma: no cover
        from steering_tutorials.traj_probes.types import (AgentTrajectory,
                                                          TrajCorpus, Turn)

    torch.manual_seed(0)

    # --- a toy tokenizer with real offsets --------------------------------
    class _Tok:
        """Whitespace tokenizer with an offset mapping and a BOS at (0, 0)."""

        def __call__(self, text, return_offsets_mapping=False,
                     add_special_tokens=True):
            ids, offs, i = [], [], 0
            if add_special_tokens:
                ids.append(1)
                offs.append((0, 0))
            while i < len(text):
                if text[i].isspace():
                    i += 1
                    continue
                j = i
                while j < len(text) and not text[j].isspace():
                    j += 1
                ids.append(2 + (hash(text[i:j]) % 60))
                offs.append((i, j))
                i = j
            out = {"input_ids": ids}
            if return_offsets_mapping:
                out["offset_mapping"] = offs
            return out

    # --- toy decoders: one causal, one deliberately NOT ---------------------
    class _CausalBlock(nn.Module):
        """h[t] = W . mean(x[:t+1]) -- depends on the prefix only."""

        def __init__(self, hidden):
            super().__init__()
            self.lin = nn.Linear(hidden, hidden)

        def forward(self, x):
            c = x.cumsum(dim=1) / torch.arange(
                1, x.shape[1] + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
            return self.lin(c)

    class _BidiBlock(nn.Module):
        """h[t] = W . (x[t] + x[-1]) -- every position reads the LAST token.

        Deliberately an order-1 lookahead, not a marginal one. A guard that only
        separates causal from non-causal by a hair is a guard that will be tuned
        away the first time it fires inconveniently.
        """

        def __init__(self, hidden):
            super().__init__()
            self.lin = nn.Linear(hidden, hidden)

        def forward(self, x):
            return self.lin(x + x[:, -1:, :])

    class _ToyLM(nn.Module):
        def __init__(self, block_cls, hidden=8, n=3, vocab=64):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([block_cls(hidden) for _ in range(n)])
            self.config = SimpleNamespace(hidden_size=hidden)

        def forward(self, ids, **_kw):
            x = self.embed(ids)
            for blk in self.layers:
                x = blk(x)
            return x

    # --- a toy corpus ------------------------------------------------------
    def _traj(uid, n_turns, label):
        turns = tuple(
            Turn(index=k, role=("assistant" if k % 2 else "user"),
                 # a DISTINCT trailing token per turn: with a shared last
                 # word the bidirectional toy's lookahead read the same value
                 # as the causal one and cleared the guard by coincidence.
                 content="turn %d of %s with some words end%d" % (k, uid, k))
            for k in range(n_turns))
        return AgentTrajectory(uid=uid, turns=turns, label=label,
                               group_id="g_" + uid, source="toy")

    trajs = [_traj("t%02d" % i, 3 + (i % 4), i % 2) for i in range(8)]
    corpus = TrajCorpus(name="toy", trajectories=trajs, requested_n_per_class=4,
                        pool_fingerprint="deadbeef", licence="unstated",
                        label_provenance="toy self-test")

    tok = _Tok()

    # (1) boundaries land on the LAST token of each turn, and the rebuilt text
    #     is byte-identical to AgentTrajectory.text.
    text, ends = turn_char_ends(trajs[0])
    assert text == trajs[0].text
    ids, bounds, n_trunc = boundary_token_indices(tok, text, ends, 0)
    assert n_trunc == 0 and len(bounds) == trajs[0].n_turns
    assert all(b is not None for b in bounds)
    assert bounds == sorted(bounds) and bounds[-1] == len(ids) - 1
    offs = tok(text, return_offsets_mapping=True)["offset_mapping"]
    for k, b in enumerate(bounds):
        assert offs[b][1] == ends[k], (
            "turn %d boundary token ends at char %d, turn ends at %d"
            % (k, offs[b][1], ends[k]))
    print("OK  boundaries: %d turns -> token idx %s, each ending exactly at its "
          "turn's last character" % (len(bounds), bounds))

    # (2) THE GUARD. On the causal toy the one-pass read is exact; on the
    #     bidirectional toy it is not, and extract() must REFUSE rather than
    #     emit well-formed rows that are not the states they claim to be.
    causal_lm = _ToyLM(_CausalBlock).eval()
    bidi_lm = _ToyLM(_BidiBlock).eval()
    eq_c = verify_prefix_equivalence(causal_lm, tok, text, bounds, 1)
    eq_b = verify_prefix_equivalence(bidi_lm, tok, text, bounds, 1)
    assert eq_c["causal"], "the causal toy failed its own equivalence check"
    assert not eq_b["causal"], "a BIDIRECTIONAL model passed the causal guard"
    assert len(eq_c["positions"]) >= 2, "the guard probed only one position"
    print("OK  prefix equivalence over %d positions: causal rel=%.2e (pass); "
          "bidirectional rel=%.2e (FAILS, as it must)"
          % (len(eq_c["positions"]), eq_c["rel"], eq_b["rel"]))

    tmp = Path(tempfile.mkdtemp(prefix="traj_probes_acts_"))
    try:
        ex_bad = HFActivationExtractor("toy/bidi", ExtractSettings(pooling="last"),
                                       cache_dir=tmp, model=bidi_lm, tok=tok,
                                       log_every=0)
        try:
            ex_bad.extract(corpus, layer=1)
        except RuntimeError as exc:
            assert "PREFIX EQUIVALENCE FAILED" in str(exc)
            print("OK  extract() REFUSES a non-causal model instead of caching "
                  "mislabelled states")
        else:
            raise AssertionError("a non-causal model was allowed to extract")

        # (3) a real end-to-end extract on the causal toy.
        ex = HFActivationExtractor("toy/causal", ExtractSettings(pooling="last"),
                                   cache_dir=tmp, model=causal_lm, tok=tok,
                                   log_every=0)
        bundle = ex.extract(corpus, layer=1)
        n_expected = sum(t.n_turns for t in trajs)
        assert len(bundle.X) == n_expected, (len(bundle.X), n_expected)
        assert bundle.step_index.min() == 0
        for t in trajs:
            m = bundle.traj_uid == t.uid
            assert bundle.step_index[m].tolist() == list(range(t.n_turns))
            assert set(bundle.y[m].tolist()) == {t.label}
        print("OK  extract(): %d rows over %d trajectories, step_index populated "
              "and each row's label matches its own trajectory"
              % (len(bundle.X), len(trajs)))

        # (4) ONE pass per trajectory, and its rows equal the n-pass rows.
        #     This is the claim the efficiency argument rests on, so it is
        #     measured, not asserted in prose.
        calls = {"n": 0}
        orig_fwd = _ToyLM.forward

        def counting(self, ids, **kw):
            calls["n"] += 1
            return orig_fwd(self, ids, **kw)

        _ToyLM.forward = counting
        try:
            ex2 = HFActivationExtractor("toy/causal", ExtractSettings(pooling="last"),
                                        cache_dir=None, model=causal_lm, tok=tok,
                                        log_every=0)
            b2 = ex2.extract(corpus, layer=1)
        finally:
            _ToyLM.forward = orig_fwd
        # The guard costs (1 full + up to n_probe truncated) passes; after
        # that it is exactly ONE pass per trajectory, which is the claim the
        # whole efficiency argument rests on -- so it is counted, not asserted
        # in prose.
        n_guard = calls["n"] - len(trajs)
        assert n_guard > 0, (
            "only %d forward passes for %d trajectories plus the guard. Fewer "
            "passes than trajectories means work was SKIPPED -- a stale journal "
            "resumed and the bundle was not computed by this run."
            % (calls["n"], len(trajs)))
        assert n_guard <= 4, "the equivalence guard cost %d passes" % n_guard
        naive = 0
        for t in trajs:
            txt, en = turn_char_ends(t)
            i_all, b_all, _ = boundary_token_indices(tok, txt, en, 0)
            for k, b in enumerate(b_all):
                h = _forward_residual(causal_lm, i_all[:b + 1], 1)
                naive += 1
                row = b2.X[(b2.traj_uid == t.uid)][k]
                assert np.allclose(h[-1], row, atol=1e-4), (
                    "one-pass row %d of %s differs from the prefix-pass row"
                    % (k, t.uid))
        print("OK  one pass per trajectory: %d forwards (%d of them the guard) "
              "vs %d for the naive prefix loop (%.1fx per trajectory), and "
              "every row is numerically identical"
              % (calls["n"], n_guard, naive, naive / float(len(trajs))))

        # (5) cache HIT costs no model; a changed CORPUS is refused by name.
        ex3 = HFActivationExtractor("toy/causal", ExtractSettings(pooling="last"),
                                    cache_dir=tmp, model=None, tok=None,
                                    log_every=0)
        b3 = ex3.extract(corpus, layer=1)     # would crash if it loaded a model
        assert np.array_equal(b3.X, bundle.X)
        print("OK  cache HIT served with model=None -- a read never loads a model")

        flipped = list(corpus.trajectories)
        flipped[0] = AgentTrajectory(uid=trajs[0].uid, turns=trajs[0].turns,
                                     label=1 - trajs[0].label,
                                     group_id=trajs[0].group_id, source="toy")
        corpus2 = TrajCorpus(name="toy", trajectories=flipped,
                             requested_n_per_class=4,
                             pool_fingerprint="deadbeef", licence="unstated",
                             label_provenance="toy self-test")
        fp1 = data_fingerprint(corpus, ExtractSettings(layer=1))
        fp2 = data_fingerprint(corpus2, ExtractSettings(layer=1))
        assert fp1 != fp2, "a RELABELLED corpus produced the same fingerprint"
        want = behaviour_block("toy/causal", 1, "last", eq_c)
        p = bundle_cache_path(tmp, "toy/causal", ExtractSettings(layer=1), 1, fp1)
        assert load_bundle(p, want, fp2, quarantine_on_mismatch=False) is None
        print("OK  a cache whose DATA fingerprint disagrees is refused by name")

        stale = dict(want)
        stale["prefix_equivalence"] = "1e+0"
        d = compare_behaviour(stale, want)
        assert d and "prefix_equivalence" in " ".join(d)
        d2 = compare_behaviour(behaviour_block("toy/causal", 1, "mean_prefix", eq_c),
                               want)
        assert d2 and "pooling" in " ".join(d2)
        print("OK  behaviour compare names the changed field (equivalence, pooling)")

        # (6) THE JOURNAL. A reap costs one trajectory, never an alignment.
        jstem = tmp / "journal_test.npz"
        hdr = {"behaviour": want, "data_fingerprint": fp1}
        j = RowJournal(jstem, hidden=4, header=hdr)
        j.open_or_resume()
        for i in range(3):
            j.append("u%d" % i, np.full((2, 4), float(i), dtype=np.float32),
                     y=i % 2, group_id="g%d" % i, step_index=[0, 1])
        j.close()

        # simulate a reap MID-TRAJECTORY: rows landed, the index line did not.
        with open(j.rows_path, "ab") as fh:
            fh.write(np.full((2, 4), 99.0, dtype=np.float32).tobytes())
        j2 = RowJournal(jstem, hidden=4, header=hdr)
        done2 = j2.open_or_resume()
        assert done2 == {"u0", "u1", "u2"}, done2
        arr = j2.materialise()
        assert arr["X"].shape == (6, 4) and 99.0 not in set(arr["X"].ravel().tolist())
        assert arr["y"].tolist() == [0, 0, 1, 1, 0, 0]
        j2.close()
        print("OK  journal: orphan rows from a mid-write reap are truncated; "
              "%d rows survive with labels still aligned" % len(arr["X"]))

        # simulate a reap MID-INDEX-LINE: a torn json line is not parsed.
        with open(j.index_path, "a", encoding="utf-8") as fh:
            fh.write('{"uid": "u3", "n_rows": 2, "y": 1, "group_i')
        j3 = RowJournal(jstem, hidden=4, header=hdr)
        assert j3.open_or_resume() == {"u0", "u1", "u2"}
        assert j3.materialise()["X"].shape == (6, 4)
        j3.close()
        print("OK  journal: a TORN index line is rejected, not half-parsed")

        # a journal written under a different behaviour is refused, not resumed.
        bad_hdr = {"behaviour": stale, "data_fingerprint": fp1}
        j4 = RowJournal(jstem, hidden=4, header=bad_hdr)
        assert j4.open_or_resume() == set(), \
            "a journal from a DIFFERENT stack was silently resumed"
        j4.close()
        print("OK  journal: a header from a different stack is refused, not resumed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("")
    print("OK -- activations.py: one causal pass per trajectory (verified equal "
          "to the n-pass reference), step_index always populated, both "
          "fingerprints binding, and a reap costs one trajectory.")


if __name__ == "__main__":
    _self_test()
