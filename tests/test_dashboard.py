"""Tests for the three-tier dashboard generator (CLAUDE.md §11).

Builds dashboards from a synthetic 3-row experiment_log.jsonl + a
reasoning_annotations.json and asserts:
  - master index.html exists and carries the composite fingerprint
  - master carries a SCREENING or EVALUATION tier chip
  - every per-experiment page exists and has NO literal "|---|" or "**" leak
  - master links to each per-experiment page
  - the markdown converter renders tables/bold without leaking markup
  - the 3-tier sub-linking (master -> hypothesis -> experiment) resolves
"""

import json
from pathlib import Path

import steering.dashboard as dash
from steering.eval import composite_fingerprint


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
def _synthetic_rows():
    fp = composite_fingerprint()
    return [
        {
            "experiment_num": 1,
            "config": {"tag": "smoke-diffmean-paircount", "operation": "add",
                       "source": "diffmean", "alpha": 4.0, "behavior": "ocean",
                       "seed": 0, "rung": 0},
            "rung": 0, "layer": 2, "behavior": "ocean",
            "composite": 0.4210, "behavior_efficacy": 0.6100,
            "capability_retention": 0.9000, "mmlu_drop_pp": 0.0500,
            "perplexity": 12.50, "dppl_norm": 0.0200, "repetition_rate": 0.0100,
            "compliance_rate": 0.0000, "harmful_refusal_rate": 1.0000,
            "harmless_refusal_rate": 0.0500, "selectivity_gap": 0.9500,
            "offshell_displacement": 0.3000, "effective_rank_base": 8.0,
            "effective_rank_steer": 7.2, "norm_budget": 0.1500,
            "fisher_at_layer": 1.2345, "composite_fingerprint": fp,
            "n_seeds": 1, "tier": "SCREENING", "status": "KEEP",
            "sample_prompt": "Tell me about the ocean.",
            "sample_steered": "The vast ocean teems with wondrous life.",
            "sample_unsteered": "The ocean is a body of salt water.",
        },
        {
            "experiment_num": 2,
            "config": {"tag": "baseline-prior", "operation": "add",
                       "source": "diffmean", "alpha": 2.0, "behavior": "ocean",
                       "seed": 0, "rung": 1},
            "rung": 1, "layer": 4, "behavior": "ocean", "is_baseline": True,
            "composite": 0.3100, "behavior_efficacy": 0.4000,
            "capability_retention": 0.9500, "mmlu_drop_pp": 0.0200,
            "perplexity": 11.00, "dppl_norm": 0.0100, "repetition_rate": 0.0050,
            "compliance_rate": 0.0000, "harmful_refusal_rate": 1.0000,
            "harmless_refusal_rate": 0.0200, "selectivity_gap": 0.9800,
            "offshell_displacement": 0.1500, "norm_budget": 0.0800,
            "fisher_at_layer": 0.9876, "composite_fingerprint": fp,
            "n_seeds": 7, "tier": "EVALUATION", "status": "DISCARD",
        },
        {
            "experiment_num": 3,
            "config": {"tag": "fisher-layer-selection", "operation": "add",
                       "source": "pca", "alpha": 6.0, "behavior": "ocean",
                       "seed": 0, "rung": 0},
            "rung": 0, "layer": 6, "behavior": "ocean",
            "composite": 0.1000, "behavior_efficacy": 0.7000,
            "capability_retention": 0.7000, "mmlu_drop_pp": 0.3000,
            "perplexity": 40.00, "dppl_norm": 0.8000, "repetition_rate": 0.0500,
            "compliance_rate": 0.0000, "harmful_refusal_rate": 1.0000,
            "harmless_refusal_rate": 0.0000, "selectivity_gap": 1.0000,
            "offshell_displacement": 0.9000, "norm_budget": 0.4000,
            "fisher_at_layer": 1.5000, "composite_fingerprint": fp,
            "n_seeds": 1, "tier": "SCREENING", "status": "DISCARD",
        },
    ]


def _reasoning():
    return {
        "1": {
            "diagnosis": "## Diagnosis\n\nThe **champion** weakness is layer "
                         "selection. Prior `exp0` left the off-shell budget "
                         "unspent.\n\n| axis | state |\n|---|---|\n| behavior | weak |\n"
                         "| safety | clean |",
            "citations": "Rimsky et al., 2023 NeurIPS 'Steering Llama 2 via "
                         "Contrastive Activation Addition' (arXiv:2312.06681) — "
                         "establishes the DiffMean framework.",
            "hypothesis": "The mechanism is that **DiffMean** shifts the residual "
                          "stream because the contrast direction is high-SNR.",
            "prediction": "We predict composite in [0.40, 0.45] and behavior > 0.55.",
            "analysis": "Composite landed at +0.4210, inside the predicted band.",
            "checkpoint": "Updated the ledger and the dashboard; committed.",
        }
    }


def _build(tmp_path: Path):
    """Write synthetic state into tmp_path and build all dashboards there.

    Uses tmp_path as the repo_root so docs/ and ideas/ live under it. We seed a
    couple of ideas/<dir> so hypothesis sub-linking can resolve.
    """
    results = tmp_path / "autoresearch_results"
    results.mkdir(parents=True, exist_ok=True)
    log = results / "experiment_log.jsonl"
    with open(log, "w", encoding="utf-8") as f:
        for r in _synthetic_rows():
            f.write(json.dumps(r) + "\n")
    (results / "reasoning_annotations.json").write_text(
        json.dumps(_reasoning(), indent=2), encoding="utf-8")

    # Seed ideas dirs so hypothesis resolution works for rows 1 and 3.
    ideas = tmp_path / "ideas"
    for name, claim in (
        ("10_diffmean_paircount_knee",
         "## Claim\n\nDiffMean reaches **90%** of asymptote.\n\n"
         "## Falsifier\n\nIf below 90% it is DISCARDED.\n\n"
         "## Predicted delta range\n\n| metric | value |\n|---|---|\n| knee | 16-64 |"),
        ("20_fisher_layer_selection",
         "## Claim\n\nFisher ratio predicts efficacy.\n\n"
         "## Falsifier\n\nSpearman < 0.7 falsifies.\n"),
    ):
        d = ideas / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "IDEA.md").write_text(f"# {name}\n\n{claim}\n", encoding="utf-8")

    master = dash.build_all_dashboards(results_dir=results, repo_root=tmp_path)
    return tmp_path, master


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_master_index_exists_and_has_fingerprint(tmp_path):
    root, master = _build(tmp_path)
    assert master.exists(), "master dashboard/index.html must exist"
    mirror = root / "docs" / "dashboard" / "index.html"
    assert mirror.exists(), "docs mirror must exist"
    text = master.read_text(encoding="utf-8")
    assert composite_fingerprint() in text, "footer must carry the composite fingerprint"
    assert composite_fingerprint() in mirror.read_text(encoding="utf-8")


def test_master_has_tier_chip(tmp_path):
    _root, master = _build(tmp_path)
    text = master.read_text(encoding="utf-8")
    assert ("SCREENING" in text) or ("EVALUATION" in text), \
        "master must show a SCREENING or EVALUATION tier chip"
    assert "EVALUATION" in text, "the n=7 row should carry an EVALUATION chip"
    assert "n=" in text, "numeric cells must carry an n= badge"


def test_every_experiment_page_exists_no_markdown_leak(tmp_path):
    root, _master = _build(tmp_path)
    for exp in (1, 2, 3):
        for base in ("dashboard", "docs/dashboard"):
            page = root / base / "experiments" / f"exp{exp:03d}.html"
            assert page.exists(), f"per-experiment page {page} must exist"
            text = page.read_text(encoding="utf-8")
            assert "|---|" not in text, f"{page} leaks a markdown table separator"
            assert "**" not in text, f"{page} leaks markdown bold markers"
            assert "##" not in text, f"{page} leaks markdown heading markers"


def test_master_links_to_each_experiment_page(tmp_path):
    _root, master = _build(tmp_path)
    text = master.read_text(encoding="utf-8")
    for exp in (1, 2, 3):
        assert f"experiments/exp{exp:03d}.html" in text, \
            f"master must link to per-experiment page exp{exp:03d}"


def test_three_tier_sublinking(tmp_path):
    root, master = _build(tmp_path)
    mtext = master.read_text(encoding="utf-8")
    # master -> hypothesis sub-dashboard links
    assert "hyp/10.html" in mtext, "master must link to hypothesis H10 sub-dashboard"

    # hypothesis sub-dashboards exist (primary + docs mirror)
    sub_primary = root / "ideas" / "10_diffmean_paircount_knee" / "dashboard" / "index.html"
    sub_mirror = root / "docs" / "dashboard" / "hyp" / "10.html"
    assert sub_primary.exists(), "primary per-hypothesis sub-dashboard must exist"
    assert sub_mirror.exists(), "docs/dashboard/hyp/<id>.html mirror must exist"

    sub_text = sub_mirror.read_text(encoding="utf-8")
    # hypothesis page links back to master and down to experiment pages
    assert "index.html" in sub_text, "sub-dashboard must back-link to master"
    assert "experiments/exp001.html" in sub_text, \
        "sub-dashboard must link to its experiment pages"
    # markdown card rendered, no leak
    assert "**" not in sub_text and "|---|" not in sub_text

    # per-experiment page back-links to its hypothesis sub-dashboard
    exp_page = (root / "docs" / "dashboard" / "experiments" / "exp001.html").read_text(encoding="utf-8")
    assert "hyp/10.html" in exp_page, "experiment page must back-link to hypothesis sub-dashboard"


def test_markdown_converter_no_leak():
    md = ("## Heading\n\nSome **bold** and `code` and a table:\n\n"
          "| a | b |\n|---|---|\n| 1 | 2 |\n\n> a quote\n")
    out = dash.md_to_html(md)
    assert "**" not in out, "bold markers must be converted"
    assert "|---|" not in out, "table separator must be consumed"
    assert "<strong>bold</strong>" in out
    assert "<table>" in out and "<th>a</th>" in out
    assert "<blockquote>" in out
    assert out.count("<h") >= 1, "heading must be converted to an <h> tag"


def test_ladder_board_and_champion(tmp_path):
    rows = _synthetic_rows()
    ladder = dash.ladder_board(rows)
    assert ladder, "ladder board must have entries"
    tags = {L["tag"] for L in ladder}
    assert "smoke-diffmean-paircount" in tags

    _root, master = _build(tmp_path)
    text = master.read_text(encoding="utf-8")
    # champion = highest composite = exp1 (0.4210); its row carries class champion
    assert 'class="champion' in text, "champion row must be highlighted"


# ---------------------------------------------------------------------------
# Surface A — stack/compete matrix on the master
# ---------------------------------------------------------------------------
def test_master_has_stack_compete_matrix(tmp_path):
    _root, master = _build(tmp_path)
    text = master.read_text(encoding="utf-8")
    assert "Stack / compete matrix" in text, "master must render the stack/compete matrix card"
    # the three verdict codes must all appear as cells (transcribed from corpus §4)
    assert "STACK" in text, "matrix must contain a STACK cell"
    assert "COMPETE" in text, "matrix must contain a COMPETE cell"
    assert "CARE" in text, "matrix must contain a CARE (stack-with-care) cell"
    # the intervention families must be present as row/col headers
    for fam in ("CAA/ActAdd", "Angular/Selective", "DoLa", "CAST gate"):
        assert fam in text, f"matrix must list the {fam} family"


def test_stack_matrix_measured_overlay():
    rows = [{
        "experiment_num": 9,
        "config": {"intervention_family": "caa", "other_family": "angular"},
        "pair_verdict": "COMPETE",
    }]
    measured = dash.measured_pair_verdicts(rows)
    assert measured.get(("CAA/ActAdd", "Angular/Selective")) == "COMPETE"
    html_out = dash.render_stack_matrix(rows)
    assert "measured: COMPETE" in html_out, "measured verdict must overlay the matrix cell"


# ---------------------------------------------------------------------------
# Surface B — per-experiment sweep curve
# ---------------------------------------------------------------------------
def test_experiment_page_has_sweep_curve(tmp_path):
    root, _master = _build(tmp_path)
    page = (root / "docs" / "dashboard" / "experiments" / "exp001.html").read_text(encoding="utf-8")
    assert "Sweep curve" in page, "per-experiment page must have a sweep-curve section"
    # the embedded PNG (when matplotlib is available) or the graceful note.
    has_img = 'exp001_sweep.png' in page
    has_note = "sweep accumulates as more alpha/layer rows are logged" in page
    assert has_img or has_note, "sweep section must embed the PNG or carry the accumulation note"


# ---------------------------------------------------------------------------
# Surface C — per-experiment side-by-side samples
# ---------------------------------------------------------------------------
def test_experiment_page_has_samples_section(tmp_path):
    root, _master = _build(tmp_path)
    # exp001 has sample_steered / sample_unsteered -> content rendered
    page1 = (root / "docs" / "dashboard" / "experiments" / "exp001.html").read_text(encoding="utf-8")
    assert "Side-by-side samples" in page1, "page must have a samples section"
    assert "Steered" in page1 and "Unsteered" in page1, "two-column layout must be present"
    assert "wondrous life" in page1, "steered sample text must be rendered"
    assert "body of salt water" in page1, "unsteered sample text must be rendered"

    # exp002 has NO samples -> placeholder block, layout still ready
    page2 = (root / "docs" / "dashboard" / "experiments" / "exp002.html").read_text(encoding="utf-8")
    assert "Side-by-side samples" in page2
    assert "no samples captured for this run" in page2, "absent-samples placeholder must render"
    assert "Steered" in page2 and "Unsteered" in page2, "placeholder must keep two-column layout"
