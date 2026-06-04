"""build_provenance.py — PER-HYPOTHESIS PROVENANCE & TRACING generator.

The research harness logs every experiment to a single append-only
``experiment_log.jsonl`` and a shared ``reasoning_annotations.json``; the design
docs live in ``hypotheses/<block>/<ID>_*.md`` and the campaign result data lives
in ``ideas/_campaigns/*.json``. That common plumbing makes it hard to answer the
question "for THIS hypothesis, exactly what was run, with what command, and where
are the artifacts?". This script closes that gap.

For every hypothesis that has experiments it emits
``hypotheses/PROVENANCE/<ID>.md`` containing:

  * the hypothesis id + plain title + verdict (from IDEA_TABLE.md);
  * the EXACT experiments — a table of every exp# (tag, model, layer, alpha, op,
    source, behavior, PPL, composite, off-shell) read straight from
    experiment_log.jsonl;
  * HOW TO REPRODUCE — the real campaign_sweep / run_hillclimb / rung3 command(s)
    with the actual args, and which script produced the rows;
  * the ARTIFACTS — campaign result JSON path(s), GitHub-blob links to the
    experiment rows + design doc, and a representative reasoning entry
    (diagnosis / hypothesis / prediction / verdict);
  * the RESULT + interpretation (2-3 sentences).

It also appends a concise "## Provenance & Tracing" section to the bottom of each
tested hypothesis's main design doc. Untested hypotheses get a one-line section
pointing back at the design protocol.

Run:
    cd <repo> && PYTHONPATH=src python scripts/build_provenance.py

This script READS the logs/campaigns/docs and WRITES only:
    hypotheses/PROVENANCE/<ID>.md   (new)
    hypotheses/<block>/<ID>_*.md    (appends one section)
It does NOT touch src/ or the dashboard and does NOT run git.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "autoresearch_results"
LOG = RESULTS / "experiment_log.jsonl"
ANN = RESULTS / "reasoning_annotations.json"
CAMP = ROOT / "ideas" / "_campaigns"
HYP = ROOT / "hypotheses"
PROV = HYP / "PROVENANCE"
IDEA_TABLE = ROOT / "IDEA_TABLE.md"

GH_BLOB = "https://github.com/dlmastery/steeringresearch/blob/master"
PROV_MARK = "## Provenance & Tracing"

# ---------------------------------------------------------------------------
# tag -> hypothesis-id mapping.  A tag-prefix predicate maps each experiment row
# (by its config tag) onto one or more hypothesis ids.  Order matters only for
# readability; an experiment can legitimately belong to several hypotheses
# (e.g. C6 informs E3, E4 and E27; C11-xbeh informs both E3-cross-behavior and E4).
# ---------------------------------------------------------------------------
TAG_RULES: list[tuple[str, list[str]]] = [
    ("rung1_plumbing", ["E4"]),            # bring-up / plumbing gate
    ("E3-cliff", ["E3"]),
    ("E3real-cliff", ["E3"]),
    ("E3gemma", ["E3"]),
    ("C1-E2-layer", ["E2"]),
    ("C3-E27-op", ["E27"]),
    ("C3b-E27-smallrot", ["E27"]),
    ("C4-N20-fragility", ["N20"]),
    ("C6-gemma1b-cliff", ["E3", "E4"]),    # cross-scale cliff + diffmean/pca alignment
    ("C7-source", ["E36"]),
    ("C8-normsrc", ["E36"]),
    ("C9b-relcliff", ["E7", "E36"]),       # relative cliff (E7) + diffmean==pca (E36)
    ("C9-relcliff", ["E7", "E36"]),
    ("C10-rel1b", ["E7"]),
    ("C11-xbeh", ["E3", "E4"]),            # cross-behavior cliff (E3) + dm/pca cos (E4)
    ("HC-E7", ["E7"]),
    ("E15-learned-gate", ["E15"]),         # trainable: learned multi-layer logistic gate
    ("E45-hypersteer", ["E45"]),           # trainable: description->vector hypernetwork
    ("E20-diffmean-3stack", ["E20"]),      # SAE-TS baseline (DiffMean 3-stack)
    ("E20-saets-3stack", ["E20"]),         # SAE-TS optimized 3-stack
    ("E7-confirm", ["E7"]),                # controlled n>=20 cross-scale confirmation
    ("E7-axbench", ["E7"]),                # E7 on the REAL AxBench benchmark (concept500, off-family judge)
    ("E3-axbench", ["E3"]),                # E3 alpha-coherence cliff on AxBench (off-family judge behavior+fluency)
]

# Hypotheses whose evidence comes from a gradient-trained auxiliary component
# (gate / hypernetwork / SAE+optimized vector). Their provenance banner must NOT
# claim "no weights are trained".
TRAINABLE_METHODS = {"E15", "E45", "E20"}

# Hypotheses whose evidence is an ANALYSIS campaign (JSON only, no experiment_log
# rows) or a standalone rung-3 script.  Each entry carries the campaign artifact,
# the reproduce command, and a short result line so these still get a provenance
# file even though they have no exp# table.
ANALYSIS: dict[str, dict] = {
    "E10": {
        "campaigns": ["E10_E17.json"],
        "script": "scripts/run_analysis_E10_E17.py",
        "reproduce": [
            "PYTHONPATH=src python scripts/campaign_sweep.py "
            "--model models/google/gemma-3-270m-it --quant none --hyp E10 "
            "--tag-prefix E10-ortho --layers 16 --alphas 0.1 --ops relative_add "
            "--behaviors anger formality happiness ocean  "
            "# then the E10_E17 cosine/stacking analysis over the extracted banks",
        ],
        "result": "Condition vectors for the four probe concepts are mostly "
        "near-orthogonal (|cos|<0.3) EXCEPT anger-happiness (+0.48) and "
        "happiness-ocean (+0.32), so OR-gating is only conditionally independent. "
        "Verdict PARTIAL(screening).",
        "json_keys": ["E10_cosines"],
    },
    "E17": {
        "campaigns": ["E10_E17.json"],
        "script": "scripts/run_analysis_E10_E17.py",
        "reproduce": [
            "PYTHONPATH=src python scripts/campaign_sweep.py "
            "--model models/google/gemma-3-270m-it --quant none --hyp E17 "
            "--tag-prefix E17-stack --layers 16 --alphas 0.1 --ops relative_add "
            "--behaviors anger happiness  "
            "# solo vs joint (anger+happiness) stacking comparison",
        ],
        "result": "Stacking two near-orthogonal behavior vectors (anger+happiness) "
        "retains ~101%/110% of each solo effect (anger 0.716->0.721, happy "
        "0.622->0.682) — no cross-degradation. Verdict SUPPORTED(screening).",
        "json_keys": ["E17_stacking"],
    },
    "E18": {
        "campaigns": ["E18_E22.json"],
        "script": "scripts/run_analysis_E18_E22.py",
        "reproduce": [
            "PYTHONPATH=src python scripts/campaign_sweep.py "
            "--model models/google/gemma-3-270m-it --quant none --hyp E18 "
            "--tag-prefix E18-interf --layers 16 --alphas 0.1 --ops relative_add "
            "--behaviors anger formality happiness ocean  "
            "# 2-5 vector stacks; interference vs summed off-diagonal Gram mass",
        ],
        "result": "Multi-vector stacks retain 85-94% of solo effect, but the "
        "interference is NON-monotone in summed off-diagonal Gram mass, so the "
        "simple Gram-mass predictor is incomplete. Verdict PARTIAL(screening).",
        "json_keys": ["solo"],
    },
    "E22": {
        "campaigns": ["E18_E22.json"],
        "script": "scripts/run_analysis_E18_E22.py",
        "reproduce": [
            "PYTHONPATH=src python scripts/campaign_sweep.py "
            "--model models/google/gemma-3-270m-it --quant none --hyp E22 "
            "--tag-prefix E22-budget --layers 16 "
            "--alphas 0.02 0.05 0.1 0.2 0.4 --ops relative_add "
            "--behaviors anger happiness  "
            "# cumulative ||sum alpha_i v_i|| swept against PPL to find the collapse knee",
        ],
        "result": "There is a cumulative-displacement collapse cliff: PPL rises "
        "from ~138 to ~4518 as the summed edit pushes h out of the in-distribution "
        "shell, confirming a norm budget. Verdict SUPPORTED(screening).",
        "json_keys": ["solo"],
    },
    "E28": {
        "campaigns": ["E28_E35_E40.json"],
        "script": "scripts/run_analysis_E28_E35_E40.py",
        "reproduce": [
            "PYTHONPATH=src python scripts/campaign_sweep.py "
            "--model models/google/gemma-3-270m-it --quant none --hyp E28 "
            "--tag-prefix E28-lowrank --layers 16 --alphas 0.1 --ops relative_add "
            "--behaviors anger  "
            "# SVD of the contrastive-difference space; variance explained by top-3 dims",
        ],
        "result": "The behavior plane is NOT low-rank: the top-3 dims of the "
        "contrastive-difference space explain only 66% of variance (predicted "
        ">90%). Verdict FALSIFIED(screening).",
        "json_keys": ["E28_var_top3"],
    },
    "E35": {
        "campaigns": ["E28_E35_E40.json"],
        "script": "scripts/run_analysis_E28_E35_E40.py",
        "reproduce": [
            "PYTHONPATH=src python scripts/campaign_sweep.py "
            "--model models/google/gemma-3-270m-it --quant none --hyp E35 "
            "--tag-prefix E35-sparse --layers 16 --alphas 0.1 --ops relative_add "
            "--behaviors anger  "
            "# sparsify the behavior vector to top-magnitude coords; efficacy vs sparsity",
        ],
        "result": "Sparsifying the behavior vector to its top-10% coordinates "
        "retains only ~77% of full efficacy (below the 85% target), so behaviors "
        "are not strongly sparse. Verdict PARTIAL(screening).",
        "json_keys": ["E35_full"],
    },
    "E40": {
        "campaigns": ["E28_E35_E40.json"],
        "script": "scripts/run_analysis_E28_E35_E40.py",
        "reproduce": [
            "PYTHONPATH=src python scripts/campaign_sweep.py "
            "--model models/google/gemma-3-270m-it --quant none --hyp E40 "
            "--tag-prefix E40-transport --layers 6 9 12 14 16 --alphas 0.1 "
            "--ops relative_add --behaviors anger  "
            "# cosine of the same behavior direction across layers (parallel transport)",
        ],
        "result": "The same behavior direction at adjacent layers stays "
        "cos 0.75-0.90 aligned, consistent with parallel transport of one "
        "underlying direction. Verdict SUPPORTED(screening).",
        "json_keys": [],
    },
    "N17": {
        "campaigns": ["RUNG3_N17.json"],
        "script": "scripts/rung3_n17.py",
        "reproduce": [
            "PYTHONPATH=src python scripts/rung3_n17.py  "
            "# rung-3: REAL WikiText-2 PPL, 50 pooled (model x layer x alpha) points, "
            "2 models, Spearman + 10k-bootstrap CI + held-out 270m->1b N5-law fit",
        ],
        "result": "On REAL held-out WikiText-2, off-shell displacement predicts "
        "incoherence: Spearman(off-shell, log real-PPL) = +0.585, 95% CI "
        "[+0.353,+0.758] (excludes 0), p=8.1e-6, n=50 across two model scales. "
        "Verdict SUPPORTED(rung-3 on N17); the single universal N5 collapse law "
        "did NOT generalize held-out (R2<0).",
        "json_keys": ["spearman_offshell_logPPL", "heldout_R2_on_1b"],
    },
    "N5": {
        "campaigns": ["RUNG3_N17.json"],
        "script": "scripts/rung3_n17.py",
        "reproduce": [
            "PYTHONPATH=src python scripts/rung3_n17.py  "
            "# same run as N17: fit log PPL = a + b*offshell on gemma-3-270m, "
            "predict gemma-3-1b, report held-out R2",
        ],
        "result": "The norm-budget conservation law is real WITHIN a model "
        "(monotone off-shell->PPL) but the single universal collapse curve did NOT "
        "transfer across scales: fitting on 270m and predicting 1b gave a negative "
        "held-out R2 (-1.60), so the slope is model-specific. Verdict "
        "SUPPORTED(screening) for the within-model law; cross-model universality "
        "FALSIFIED at rung-3.",
        "json_keys": ["n5_law_fit_270m", "heldout_R2_on_1b"],
    },
}

# Which campaign JSON(s) carry the row-level data for each tag-prefix, used to
# point the provenance file at the right artifact for experiment-backed hypotheses.
PREFIX_CAMPAIGN: dict[str, str] = {
    "E3-cliff": "(logged inline in experiment_log.jsonl; no separate campaign JSON)",
    "E3real-cliff": "(logged inline in experiment_log.jsonl; no separate campaign JSON)",
    "rung1_plumbing": "(logged inline in experiment_log.jsonl; bring-up gate)",
    "C1-E2-layer": "C1-E2-layer.json",
    "C3-E27-op": "C3-E27-op.json",
    "C3b-E27-smallrot": "C3b-E27-smallrot.json",
    "C4-N20-fragility": "C4-N20-fragility.json",
    "C6-gemma1b-cliff": "C6-gemma1b-cliff.json",
    "C7-source": "C7-source.json",
    "C8-normsrc": "C8-normsrc.json",
    "C9b-relcliff": "C9b-relcliff.json",
    "C9-relcliff": "C9-relcliff.json",
    "C10-rel1b": "C10-rel1b.json",
    "C11-xbeh": "C11-xbeh.json",
    "HC-E7": "HC-E7.json",
    "E15-learned-gate": "E15-learned-gate.json",
    "E45-hypersteer": "E45-hypersteer.json",
    "E20-diffmean-3stack": "E20-saets.json",
    "E20-saets-3stack": "E20-saets.json",
    "E7-confirm": "E7-confirm-gemma-3-270m-it.json + E7-confirm-gemma-3-1b-it.json",
    "E7-axbench": "E7-axbench-gemma-3-270m-it.json + E7-axbench-gemma-2-2b-it.json",
    "E3-axbench": "E3-axbench-gemma-2-2b-it.json",
}

# Reproduce command(s) for each tag-prefix (the real invocation that produced it).
PREFIX_REPRODUCE: dict[str, str] = {
    "rung1_plumbing": "PYTHONPATH=src python -m steering.runner  "
    "# Rung-0/1 plumbing gate on the offline FakeResidualLM (infra, not a Gemma claim)",
    "E3-cliff": "PYTHONPATH=src python scripts/campaign_sweep.py "
    "--model Qwen/Qwen2.5-0.5B-Instruct --quant none --hyp E3 --tag-prefix E3-cliff "
    "--behavior ocean --layers 21 --alphas 0.0 1.0 2.0 4.0 8.0 12.0 16.0 24.0 --ops add",
    "E3real-cliff": "PYTHONPATH=src python scripts/campaign_sweep.py "
    "--model models/google/gemma-3-270m-it --quant none --hyp E3 --tag-prefix E3real-cliff "
    "--behavior ocean --layers 16 --alphas 0.0 1.0 2.0 4.0 8.0 --ops add",
    "C1-E2-layer": "PYTHONPATH=src python scripts/campaign_sweep.py "
    "--model models/google/gemma-3-270m-it --quant none --hyp E2 --tag-prefix C1-E2-layer "
    "--behavior ocean --layers 2 4 6 8 10 12 14 16 --alphas 2.0 --ops add --sources diffmean",
    "C3-E27-op": "PYTHONPATH=src python scripts/campaign_sweep.py "
    "--model models/google/gemma-3-270m-it --quant none --hyp E27 --tag-prefix C3-E27-op "
    "--behavior ocean --layers 16 --alphas 1.0 2.0 4.0 --ops add rotate project_out --sources diffmean",
    "C3b-E27-smallrot": "PYTHONPATH=src python scripts/campaign_sweep.py "
    "--model models/google/gemma-3-270m-it --quant none --hyp E27 --tag-prefix C3b-E27-smallrot "
    "--behavior ocean --layers 16 --alphas 0.05 0.1 0.2 0.3 0.5 --ops rotate add --sources diffmean",
    "C4-N20-fragility": "PYTHONPATH=src python scripts/campaign_sweep.py "
    "--model models/google/gemma-3-270m-it --quant none --hyp N20 --tag-prefix C4-N20-fragility "
    "--behavior ocean --layers 2 4 6 8 10 12 14 16 --alphas 4.0 --ops add --sources diffmean",
    "C6-gemma1b-cliff": "PYTHONPATH=src python scripts/campaign_sweep.py "
    "--model models/google/gemma-3-1b-it --quant none --hyp E3 --tag-prefix C6-gemma1b-cliff "
    "--behavior ocean --layers 18 --alphas 0.0 1.0 2.0 4.0 8.0 --ops add --sources diffmean",
    "C7-source": "PYTHONPATH=src python scripts/campaign_sweep.py "
    "--model models/google/gemma-3-270m-it --quant none --hyp E36 --tag-prefix C7-source "
    "--behavior ocean --layers 16 --alphas 0.5 1.0 2.0 --ops add --sources diffmean pca",
    "C8-normsrc": "PYTHONPATH=src python scripts/campaign_sweep.py "
    "--model models/google/gemma-3-270m-it --quant none --hyp E36 --tag-prefix C8-normsrc "
    "--behavior ocean --layers 16 --alphas 5.0 10.0 20.0 40.0 --ops add --sources diffmean pca --normalize",
    "C9b-relcliff": "PYTHONPATH=src python scripts/campaign_sweep.py "
    "--model models/google/gemma-3-270m-it --quant none --hyp E7 --tag-prefix C9b-relcliff "
    "--behavior ocean --layers 16 --alphas 0.02 0.05 0.1 0.2 0.4 --ops relative_add --sources diffmean pca",
    "C9-relcliff": "PYTHONPATH=src python scripts/campaign_sweep.py "
    "--model models/google/gemma-3-270m-it --quant none --hyp E7 --tag-prefix C9-relcliff "
    "--behavior ocean --layers 16 --alphas 0.02 0.05 0.1 0.2 0.4 --ops relative_add --sources diffmean pca",
    "C10-rel1b": "PYTHONPATH=src python scripts/campaign_sweep.py "
    "--model models/google/gemma-3-1b-it --quant none --hyp E7 --tag-prefix C10-rel1b "
    "--behavior ocean --layers 18 --alphas 0.02 0.05 0.1 0.2 0.4 --ops relative_add --sources diffmean",
    "C11-xbeh": "PYTHONPATH=src python scripts/campaign_sweep.py "
    "--model models/google/gemma-3-270m-it --quant none --hyp E3 --tag-prefix C11-xbeh "
    "--behaviors ocean happiness anger formality --layers 16 --alphas 0.0 0.1 0.2 "
    "--ops relative_add --sources diffmean",
    "HC-E7": "PYTHONPATH=src python scripts/run_hillclimb.py "
    "--model models/google/gemma-3-270m-it --quant none --hyp E7 --behavior ocean "
    "--base-op relative_add --base-layer 16 --base-source diffmean --base-alpha 0.1 "
    "--alphas 0.05 0.1 0.15 0.2 --layers 12 14 --sources diffmean pca --ops relative_add add",
    "E15-learned-gate": "PYTHONPATH=src python scripts/run_e15.py "
    "--model models/google/gemma-3-270m-it --quant none "
    "# trains the multi-layer logistic gate + fixed cosine baseline, in-dist + OOD PR-AUC",
    "E45-hypersteer": "PYTHONPATH=src python scripts/run_e45.py "
    "--model models/google/gemma-3-270m-it --quant none "
    "# leave-one-behavior-out hypernetwork (Adam/MSE) description->vector cosine + efficacy",
    "E20-diffmean-3stack": "PYTHONPATH=src python scripts/run_e20.py "
    "--model models/google/gemma-3-270m-it --quant none --layer 6 "
    "# trains a sparse autoencoder + optimizes SAE-TS vectors; Gram mass + 3-stack coherence",
    "E20-saets-3stack": "PYTHONPATH=src python scripts/run_e20.py "
    "--model models/google/gemma-3-270m-it --quant none --layer 6 "
    "# trains a sparse autoencoder + optimizes SAE-TS vectors; Gram mass + 3-stack coherence",
    "E3-axbench": "PYTHONPATH=src python scripts/run_axbench_e3.py --model google/gemma-2-2b-it --quant none --layer 20 --dataset concept500 --concepts 30 --prompts 8 --alphas 0.02 0.05 0.1 0.2 0.4 0.8   # off-family judge scores behavior+fluency vs alpha",
    "E7-axbench": "PYTHONPATH=src python scripts/run_axbench_e7.py --model models/google/gemma-3-270m-it --quant none --dataset concept500 --concepts 0 --prompts 10 --knee 0.1 --judge local   # real AxBench benchmark + off-family Qwen judge",
    "E7-confirm": "PYTHONPATH=src python scripts/confirm_e7.py "
    "--model models/google/gemma-3-270m-it --quant none --layer 16 --alphas 0.05 0.1 0.15 "
    "--knee 0.1 --seeds 20 --prompts 4    # then --model .../gemma-3-1b-it --layer 18 ; then --combine. "
    "GEMINI_API_KEY set => off-family judge instrument; controls = matched-displacement random + shuffled-label.",
}

PREFIX_SCRIPT: dict[str, str] = {
    "rung1_plumbing": "scripts/.. (steering.runner plumbing gate)",
    "E3-cliff": "scripts/campaign_sweep.py",
    "E3real-cliff": "scripts/campaign_sweep.py",
    "C1-E2-layer": "scripts/campaign_sweep.py",
    "C3-E27-op": "scripts/campaign_sweep.py",
    "C3b-E27-smallrot": "scripts/campaign_sweep.py",
    "C4-N20-fragility": "scripts/campaign_sweep.py",
    "C6-gemma1b-cliff": "scripts/campaign_sweep.py",
    "C7-source": "scripts/campaign_sweep.py",
    "C8-normsrc": "scripts/campaign_sweep.py",
    "C9b-relcliff": "scripts/campaign_sweep.py",
    "C9-relcliff": "scripts/campaign_sweep.py",
    "C10-rel1b": "scripts/campaign_sweep.py",
    "C11-xbeh": "scripts/campaign_sweep.py",
    "HC-E7": "scripts/run_hillclimb.py",
    "E15-learned-gate": "scripts/run_e15.py  (steering.gate)",
    "E45-hypersteer": "scripts/run_e45.py  (steering.hypersteer)",
    "E20-diffmean-3stack": "scripts/run_e20.py  (steering.sae)",
    "E20-saets-3stack": "scripts/run_e20.py  (steering.sae)",
    "E7-confirm": "scripts/confirm_e7.py  (steering.controls + steering.stats + steering.judge)",
    "E7-axbench": "scripts/run_axbench_e7.py  (steering.axbench + steering.local_judge + steering.stats)",
    "E3-axbench": "scripts/run_axbench_e3.py  (steering.axbench + steering.local_judge)",
}


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def load_experiments() -> list[dict]:
    rows = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_reasoning() -> dict:
    return json.loads(ANN.read_text(encoding="utf-8")) if ANN.exists() else {}


def matching_prefix(tag: str) -> str | None:
    """Return the longest tag-prefix in PREFIX_REPRODUCE that the tag starts with."""
    best = None
    for pre in PREFIX_REPRODUCE:
        if tag.startswith(pre) and (best is None or len(pre) > len(best)):
            best = pre
    return best


def hyp_ids_for_tag(tag: str) -> list[str]:
    ids: list[str] = []
    for pre, hyps in TAG_RULES:
        if tag.startswith(pre):
            for h in hyps:
                if h not in ids:
                    ids.append(h)
    return ids


_VERDICT_RE = re.compile(
    r"(SUPPORTED|FALSIFIED|PARTIAL|PENDING|INCONCLUSIVE|DIRECTIONAL)", re.I)


def parse_idea_table() -> dict[str, dict]:
    """Map hypothesis id -> {title, status} parsed from IDEA_TABLE.md table rows.

    Some cells contain literal unescaped ``||h||`` which breaks a naive pipe split
    (it injects phantom empty cells and shifts column indices). So title is taken
    as the cell right after the id, and status is located by verdict keyword rather
    than a fixed column index.
    """
    out: dict[str, dict] = {}
    if not IDEA_TABLE.exists():
        return out
    for line in IDEA_TABLE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7:
            continue
        hid = cells[0]
        if not re.fullmatch(r"[EN]\d+", hid):
            continue
        title = cells[1]
        # status = the verdict-bearing cell; fall back to the canonical 7th column.
        # literal ``||h||`` injects phantom empty cells, so rejoin from the verdict
        # cell to the last cell that still carries the status (before the Rung/dir
        # columns) — heuristically, glue trailing cells that contain '(' or text.
        vidx = next((i for i, c in enumerate(cells) if i >= 2 and _VERDICT_RE.search(c)), None)
        if vidx is not None:
            tail = [c for c in cells[vidx:] if c]
            # keep gluing until we hit a pure rung number or empty trailer
            status = " ".join(t for t in tail if not re.fullmatch(r"\d+|—|-", t))
        else:
            status = cells[6] if len(cells) > 6 else "—"
        out[hid] = {"title": title, "status": status}
    return out


def find_design_doc(hid: str) -> Path | None:
    matches = sorted(HYP.glob(f"*/{hid}_*.md"))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def fmt_num(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) >= 1e6 or (v != 0 and abs(v) < 1e-3):
            return f"{v:.2e}"
        return f"{v:.{nd}f}"
    return str(v)


def exp_table(rows: list[dict]) -> str:
    head = (
        "| exp# | tag | model | layer | alpha | op | source | behavior | PPL | composite | off-shell |\n"
        "|------|-----|-------|-------|-------|----|--------|----------|-----|-----------|-----------|\n"
    )
    body = []
    for r in sorted(rows, key=lambda x: x["experiment_num"]):
        c = r["config"]
        body.append(
            "| {n} | `{tag}` | {model} | {layer} | {alpha} | {op} | {src} | {beh} | {ppl} | {comp} | {off} |".format(
                n=r["experiment_num"],
                tag=c.get("tag", "?"),
                model=c.get("model", "?").split("/")[-1],
                layer=c.get("layer", "—"),
                alpha=fmt_num(c.get("alpha"), 3),
                op=c.get("operation", "—"),
                src=c.get("source", "—"),
                beh=fmt_num(r.get("behavior_efficacy"), 3),
                ppl=fmt_num(r.get("perplexity"), 1),
                comp=fmt_num(r.get("composite"), 3),
                off=fmt_num(r.get("offshell_displacement"), 4),
            )
        )
    return head + "\n".join(body) + "\n"


def reasoning_block(rep_num: int, ann: dict) -> str:
    e = ann.get(str(rep_num))
    if not e:
        return f"_No reasoning entry recorded for exp#{rep_num}._\n"
    def trim(s: str, n=600) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[: n - 1].rstrip() + "…"
    return (
        f"Representative reasoning entry — **exp#{rep_num}** "
        f"(`autoresearch_results/reasoning_annotations.json` key `\"{rep_num}\"`):\n\n"
        f"- **Diagnosis:** {trim(e.get('diagnosis',''))}\n"
        f"- **Hypothesis:** {trim(e.get('hypothesis',''))}\n"
        f"- **Prediction:** {trim(e.get('prediction',''))}\n"
        f"- **Verdict:** {trim(e.get('verdict',''), 400) or '_(no verdict text; see status journal in design doc)_'}\n"
    )


def reproduce_block(prefixes: list[str]) -> str:
    lines = []
    for pre in prefixes:
        cmd = PREFIX_REPRODUCE.get(pre)
        script = PREFIX_SCRIPT.get(pre, "scripts/campaign_sweep.py")
        camp = PREFIX_CAMPAIGN.get(pre, "")
        if not cmd:
            continue
        lines.append(f"**Tag prefix `{pre}*`** — script `{script}`"
                     + (f", artifact `ideas/_campaigns/{camp}`" if camp and camp.endswith('.json') else
                        (f"  {camp}" if camp else "")) + ":\n")
        lines.append("```bash\n" + cmd + "\n```\n")
    return "\n".join(lines)


def build_experiment_backed(hid: str, info: dict, rows: list[dict], ann: dict) -> str:
    title = info.get("title", hid)
    status = info.get("status", "—")
    rows = sorted(rows, key=lambda x: x["experiment_num"])
    exp_nums = [r["experiment_num"] for r in rows]
    prefixes: list[str] = []
    campaigns: list[str] = []
    models: set[str] = set()
    for r in rows:
        pre = matching_prefix(r["config"].get("tag", ""))
        if pre and pre not in prefixes:
            prefixes.append(pre)
        camp = PREFIX_CAMPAIGN.get(pre or "", "")
        if camp.endswith(".json") and camp not in campaigns:
            campaigns.append(camp)
        models.add(r["config"].get("model", "?").split("/")[-1])

    rep_num = exp_nums[len(exp_nums) // 2]  # a middle experiment as representative

    art_lines = []
    for camp in campaigns:
        art_lines.append(f"- Campaign result JSON: `ideas/_campaigns/{camp}` "
                         f"([blob]({GH_BLOB}/ideas/_campaigns/{camp}))")
    if not campaigns:
        art_lines.append("- Experiment rows logged inline (no separate campaign JSON).")
    art_lines.append(f"- Experiment log (all rows, append-only): "
                     f"`autoresearch_results/experiment_log.jsonl` "
                     f"([blob]({GH_BLOB}/autoresearch_results/experiment_log.jsonl)) "
                     f"— this hypothesis = exp# {', '.join(map(str, exp_nums))}")
    art_lines.append(f"- Reasoning annotations: `autoresearch_results/reasoning_annotations.json` "
                     f"([blob]({GH_BLOB}/autoresearch_results/reasoning_annotations.json))")
    dd = find_design_doc(hid)
    if dd:
        rel = dd.relative_to(ROOT).as_posix()
        art_lines.append(f"- Design doc: `{rel}` ([blob]({GH_BLOB}/{rel}))")

    md = []
    md.append(f"# {hid} — Provenance & Tracing")
    md.append("")
    md.append(f"> **Title:** {title}  ")
    md.append(f"> **Verdict (IDEA_TABLE.md):** {status}  ")
    md.append(f"> **Models touched:** {', '.join(sorted(models))}  ")
    md.append(f"> **Experiments:** {len(rows)} rows — exp# {', '.join(map(str, exp_nums))}")
    md.append("")
    md.append("_Auto-generated by `scripts/build_provenance.py` from "
              "`experiment_log.jsonl` + `reasoning_annotations.json` + "
              "`ideas/_campaigns/`. Do not hand-edit; re-run the generator._")
    md.append("")
    if hid in TRAINABLE_METHODS:
        md.append("> **This hypothesis trains a small auxiliary component** (a "
                  "logistic gate / an MLP hypernetwork / a sparse autoencoder + an "
                  "optimized steering vector) — the base model stays frozen. These "
                  "are the project's only gradient-trained experiments; the rows "
                  "below carry method-specific metrics (gate AUC / held-out cosine / "
                  "Gram mass) in their `method_extra`, not the standard 5-axis "
                  "composite. See [`../TRAINING-PROCESS.md`](../TRAINING-PROCESS.md) §4.")
    else:
        md.append("> **No weights are trained.** The runs below are *inference-time* "
                  "steering: the model is frozen and the steering vector is extracted "
                  "in one shot (mean difference or one SVD). That is why the rows are "
                  "configs (layer/α/op/source), not training checkpoints. See "
                  "[`../TRAINING-PROCESS.md`](../TRAINING-PROCESS.md).")
    md.append("")
    md.append("## 1. The exact experiments")
    md.append("")
    md.append(exp_table(rows))
    md.append("## 2. How to reproduce")
    md.append("")
    md.append(reproduce_block(prefixes))
    md.append("Each campaign cell pre-authors a `_manual` reasoning entry tied to "
              "this hypothesis, then calls `steering.runner.run_single_experiment` "
              "(model loaded once via `load_model_cached`); every cell appends one "
              "row to `experiment_log.jsonl` with the shared `composite_fingerprint`.")
    md.append("")
    md.append("## 3. Artifacts")
    md.append("")
    md.extend(art_lines)
    md.append("")
    md.append("## 4. Reasoning trace (representative)")
    md.append("")
    md.append(reasoning_block(rep_num, ann))
    md.append("## 5. Result & interpretation")
    md.append("")
    md.append(result_interpretation(hid, rows, status))
    md.append("")
    return "\n".join(md)


def result_interpretation(hid: str, rows: list[dict], status: str) -> str:
    if hid in ANALYSIS:
        return ANALYSIS[hid]["result"]
    comps = [r.get("composite") for r in rows if isinstance(r.get("composite"), (int, float))]
    ppls = [r.get("perplexity") for r in rows if isinstance(r.get("perplexity"), (int, float))]
    offs = [r.get("offshell_displacement") for r in rows
            if isinstance(r.get("offshell_displacement"), (int, float))]
    bits = []
    if comps:
        bits.append(f"composite ranged {min(comps):+.3f}..{max(comps):+.3f}")
    if ppls:
        bits.append(f"perplexity {min(ppls):.1f}..{max(ppls):.1f}")
    if offs:
        bits.append(f"off-shell Δ‖h‖ {min(offs):.4f}..{max(offs):.4f}")
    sweep = "; ".join(bits)
    return (
        f"Across these {len(rows)} screening cells the {sweep}. "
        f"The IDEA_TABLE verdict for {hid} is **{status}** — see the design doc's "
        f"status journal and committee Q&A for the full interpretation and the "
        f"pre-registered falsifier. These rows are n=1 SCREENING geometry probes "
        f"(behavior is a synthetic-lexicon proxy; PPL and off-shell Δ‖h‖ are the "
        f"real signals); the rung-3 generation-judge confirmation is the pending step."
    )


def build_analysis_backed(hid: str, info: dict, meta: dict, ann: dict) -> str:
    title = meta.get("title", hid)
    status = meta.get("status", "—")
    md = []
    md.append(f"# {hid} — Provenance & Tracing")
    md.append("")
    md.append(f"> **Title:** {title}  ")
    md.append(f"> **Verdict (IDEA_TABLE.md):** {status}  ")
    md.append("> **Evidence type:** analysis campaign (no per-row `experiment_log.jsonl` "
              "entries; the artifact below carries the computed quantities).")
    md.append("")
    md.append("_Auto-generated by `scripts/build_provenance.py`._")
    md.append("")
    md.append("> **No weights are trained** — this analysis runs over *extracted* "
              "(frozen-model) activation banks, not a training process. See "
              "[`../TRAINING-PROCESS.md`](../TRAINING-PROCESS.md).")
    md.append("")
    md.append("## 1. The exact experiments")
    md.append("")
    md.append("This hypothesis was evaluated by an analysis campaign over the "
              "extracted activation banks rather than by individually-logged "
              "steering cells. The computed quantities live in the artifact(s) below.")
    md.append("")
    for camp in info["campaigns"]:
        path = CAMP / camp
        keys = info.get("json_keys", [])
        snippet = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for k in keys:
                if k in data:
                    snippet[k] = data[k]
        if snippet:
            md.append(f"Key values from `ideas/_campaigns/{camp}`:")
            md.append("")
            md.append("```json")
            md.append(json.dumps(snippet, indent=2)[:1200])
            md.append("```")
            md.append("")
    md.append("## 2. How to reproduce")
    md.append("")
    md.append(f"Script: `{info['script']}` (analysis driver over the cached banks).")
    md.append("")
    for cmd in info["reproduce"]:
        md.append("```bash\n" + cmd + "\n```")
    md.append("")
    md.append("## 3. Artifacts")
    md.append("")
    for camp in info["campaigns"]:
        md.append(f"- Campaign result JSON: `ideas/_campaigns/{camp}` "
                  f"([blob]({GH_BLOB}/ideas/_campaigns/{camp}))")
    dd = find_design_doc(hid)
    if dd:
        rel = dd.relative_to(ROOT).as_posix()
        md.append(f"- Design doc: `{rel}` ([blob]({GH_BLOB}/{rel}))")
    md.append(f"- Experiment log (program-wide): `autoresearch_results/experiment_log.jsonl` "
              f"([blob]({GH_BLOB}/autoresearch_results/experiment_log.jsonl))")
    md.append("")
    md.append("## 4. Result & interpretation")
    md.append("")
    md.append(info["result"])
    md.append("")
    return "\n".join(md)


def append_to_design_doc(hid: str, prov_rel: str, exp_nums: list[int],
                         reproduce_cmds: list[str], tested: bool) -> bool:
    dd = find_design_doc(hid)
    if not dd:
        return False
    text = dd.read_text(encoding="utf-8")
    if PROV_MARK in text:
        # strip the previously-appended section (and the separator we injected
        # ahead of it) so re-runs stay idempotent and don't pile up '---' rules
        head = text.split("\n" + PROV_MARK)[0].rstrip()
        # remove any run of trailing horizontal-rule separators we injected before
        head = re.sub(r"(\s*\n-{3,})+\s*$", "", head).rstrip()
        text = head + "\n"
    block = ["", "---", "", PROV_MARK, ""]
    if tested:
        block.append(f"Full per-hypothesis provenance (exact experiments, reproduce "
                     f"commands, artifact links, reasoning trace): "
                     f"[`{prov_rel}`](../{prov_rel}).")
        block.append("")
        if exp_nums:
            block.append(f"- **Experiments:** exp# {', '.join(map(str, exp_nums))} "
                         f"(`autoresearch_results/experiment_log.jsonl`).")
        else:
            block.append("- **Experiments:** analysis campaign (computed quantities "
                         "in the campaign JSON; see the provenance file).")
        if reproduce_cmds:
            block.append("- **Reproduce:**")
            block.append("")
            block.append("```bash\n" + reproduce_cmds[0].strip() + "\n```")
    else:
        block.append("No experiments run yet — see this design doc's protocol "
                     "(§7) for what would be run. Once a campaign logs rows for "
                     "this hypothesis, re-run `scripts/build_provenance.py` to "
                     "generate `hypotheses/PROVENANCE/" + hid + ".md`.")
    block.append("")
    dd.write_text(text.rstrip() + "\n" + "\n".join(block), encoding="utf-8")
    return True


def main() -> None:
    PROV.mkdir(parents=True, exist_ok=True)
    rows = load_experiments()
    ann = load_reasoning()
    table = parse_idea_table()

    # group experiment rows by hypothesis id
    by_hyp: dict[str, list[dict]] = {}
    for r in rows:
        for hid in hyp_ids_for_tag(r["config"].get("tag", "")):
            by_hyp.setdefault(hid, []).append(r)

    tested_ids = set(by_hyp) | set(ANALYSIS)
    written, appended, untested = 0, 0, 0

    # 1) experiment-backed provenance files
    for hid, hrows in sorted(by_hyp.items()):
        meta = table.get(hid, {"title": hid, "status": "—"})
        md = build_experiment_backed(hid, meta, hrows, ann)
        (PROV / f"{hid}.md").write_text(md, encoding="utf-8")
        written += 1
        prefixes = []
        for r in hrows:
            pre = matching_prefix(r["config"].get("tag", ""))
            if pre and pre not in prefixes:
                prefixes.append(pre)
        cmds = [PREFIX_REPRODUCE[p] for p in prefixes if p in PREFIX_REPRODUCE]
        exp_nums = sorted(r["experiment_num"] for r in hrows)
        if append_to_design_doc(hid, f"PROVENANCE/{hid}.md", exp_nums, cmds, tested=True):
            appended += 1

    # 2) analysis-backed provenance files (no exp rows)
    for hid, info in sorted(ANALYSIS.items()):
        if hid in by_hyp:
            continue  # already emitted with row table
        meta = table.get(hid, {"title": hid, "status": "—"})
        md = build_analysis_backed(hid, info, meta, ann)
        (PROV / f"{hid}.md").write_text(md, encoding="utf-8")
        written += 1
        if append_to_design_doc(hid, f"PROVENANCE/{hid}.md", [],
                                info["reproduce"], tested=True):
            appended += 1

    # 3) untested hypotheses — append the "no experiments yet" stub
    all_ids = set(table)
    for dd in sorted(HYP.glob("*/*.md")):
        m = re.match(r"([EN]\d+)_", dd.name)
        if not m:
            continue
        hid = m.group(1)
        all_ids.add(hid)
        if hid in tested_ids:
            continue
        if append_to_design_doc(hid, f"PROVENANCE/{hid}.md", [], [], tested=False):
            untested += 1

    print(f"PROVENANCE files written : {written}  -> {PROV}")
    print(f"design docs appended (tested)   : {appended}")
    print(f"design docs appended (untested) : {untested}")
    print(f"tested hypothesis ids: {sorted(tested_ids)}")


if __name__ == "__main__":
    main()
