"""export_datasets.py -- STA's two used corpora are UNVETTED; nothing is exported.

CPU-only (loads from the local `artifacts/corpora/*.json` cache; no model, no GPU,
no fresh network fetch needed for either corpus already built there).

Both corpora this lesson actually ran (`results_shade.json`, `results_agentdojo.json`
both exist; `assebench`/`atbench` have no results files and were never run) declare
their OWN licence as the STRING "unstated" -- see `corpora/shade.py` and
`corpora/agentdojo.py`, `Corpus.licence` set literally to `"unstated"`, and each
module's docstring: "Neither repo declares a licence ... these community re-dumps
assert nothing." Per `common/dataset_export.py`'s own rule (an unvetted source is
refused exactly like a gated one -- "unchecked licence is not a permissive one"),
neither `adityaasinha28/control_arena_shade` + `aksh-n/control_arena_shade-monitor-
labels` nor `adityaasinha28/control_arena_agentdojo` + its labels repo is in
`REDISTRIBUTABLE` or `GATED`. This script does NOT add them there (out of scope --
common/ is owned by the licence-classification pass, not this lesson) and does NOT
export any rows. It only confirms the two corpora actually used, records their real
`Trajectory.uid`s in a refetch manifest (never the trajectory text/steps), and
prints the finding plainly.

Every per-step judge label on both corpora is Gemini-3.1-Pro OUTPUT, not human
ground truth, which is a second, independent reason not to treat these as a
citable public dataset even if the licence question were resolved.

Run: python -m steering_tutorials.streaming_trajectory_aggregation.export_datasets
"""
from __future__ import annotations

from pathlib import Path

from steering_tutorials.common import dataset_export as DE

from . import config as C

from .corpora import get_loaders

_HERE = Path(__file__).resolve().parent

_USED = ("shade", "agentdojo")   # the only two corpora with a results.json on disk


def main() -> None:
    loaders = get_loaders()
    for corpus_name in _USED:
        loader = loaders[corpus_name]
        corpus = loader.load(n_per_class=C.N_PER_CLASS, seed=C.SEED)
        uids = [t.uid for t in corpus.trajectories]
        print("[export] corpus=%-9s n=%d licence=%r pool_fingerprint=%s"
              % (corpus_name, len(uids), corpus.licence, corpus.pool_fingerprint[:16]))
        assert corpus.licence == "unstated", (
            "corpus %r no longer reports licence=='unstated' -- re-check whether "
            "dataset_export.py has since classified %r; if so this script should "
            "switch to export_slice." % (corpus_name, loader.hf_id))
        man = DE.write_refetch_manifest(
            _HERE, "%s_n%d_s%d" % (corpus_name, len(uids), C.SEED), loader.hf_id,
            row_ids=uids, seed=C.SEED,
            loader_hint=("python -c \"from steering_tutorials."
                        "streaming_trajectory_aggregation.corpora import get_loaders; "
                        "get_loaders()[%r].load(n_per_class=%d, seed=%d)\""
                        % (corpus_name, C.N_PER_CLASS, C.SEED)),
            notes=("STA %s corpus: licence is explicitly 'unstated' by the loader's "
                   "own Corpus dataclass, so export_slice would refuse it (neither "
                   "REDISTRIBUTABLE nor GATED). NOT exported -- rows never written, "
                   "only the %d Trajectory.uids that identify this exact pool. "
                   "Per-step labels on this corpus are Gemini-3.1-Pro judge output, "
                   "not human ground truth (a second, independent reason this is not "
                   "a citable public dataset)." % (corpus_name, len(uids))))
        print("[export] %s: NOT exported (licence=unstated); refetch manifest "
              "written with %d row uids, no row text" % (corpus_name, man["n_rows"]))

    print("\n[export] streaming_trajectory_aggregation: 0 slices exported "
          "(both used corpora are unvetted-licence); 2 refetch manifests written.")


if __name__ == "__main__":
    main()
