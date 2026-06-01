"""dashboard.py — the RICH, hierarchically-linked autoresearch dashboard.

A faithful port of the ``nature_inspired_networks`` reference dashboard
(brutalist editorial lab-notebook v3) to the steering autoresearch data model.

Three linked tiers with full bidirectional click-through (CLAUDE.md §11):

  A. MASTER     dashboard/index.html  (mirror docs/dashboard/index.html)
  B. PER-HYPO   docs/dashboard/hyp/<ID>.html
                (also ideas/<dir>/dashboard/index.html where a dir exists)
  C. PER-EXP    docs/dashboard/experiments/expNNN.html
                (mirror dashboard/experiments/expNNN.html)

MASTER sections (ported from the reference render_dashboard):
  - Masthead: title + subtitle + mast-pill CTAs (repo / paper / FINDINGS /
    IDEA_TABLE / live Pages) + the composite formula-chip (fingerprint).
  - "How to read this" 4-bullet orientation block (incl. a NAVIGATION bullet).
  - KPI ribbon (7 cards).
  - Headline-ribbon: the S-1..S-8 screening findings, tagged SCREENING (n=1).
  - HYPOTHESIS-STATUS GRID: a clickable, verdict-coloured cell per hypothesis
    (E1..E50 in blocks A-F, then N1..N20), each linking to its hypothesis page.
  - Runs grouped into campaign sections with verdict-coloured rows; each row
    links to its experiment page and (when resolvable) its hypothesis page.
  - Panels as cards (radar / parcoords, 3 Pareto, geometry, ladder board,
    stack/compete matrix).
  - Footer: composite formula + fingerprint + git SHA + timestamp + QA note +
    doc cross-links.

PER-EXPERIMENT page sections (ported from render_experiment_page):
  - Asymmetric header + mast-pill row + tier badge + verdict-row provenance.
  - Key-numbers strip (kn-strip tiles).
  - Hypothesis digest card (statement / falsifier / predicted Δ pulled from
    IDEA_TABLE + ideas/<dir>).
  - Verdict card (the run's KEEP/DISCARD verdict + behavior_scorer / safety_real
    provenance chips, rendered from the reasoning verdict blob).
  - 7-step reasoning blob (rendered markdown — NO ##/**/|---| leak).
  - Configuration card.
  - Five-axis metrics quick-reference table.
  - Composite-score breakdown (term-by-term, reconciled to the logged composite).
  - Geometry probes (offshell, angular, fisher, cos_dm_pca).
  - Sweep curve (behavior + PPL vs the swept alpha/layer axis).
  - Steered-vs-unsteered samples.
  - Cross-references grid (hypothesis page + master + prev/next experiment).

PER-HYPOTHESIS page sections:
  - Header + verdict pill + mast-pills.
  - Hypothesis card (statement + falsifier + predicted Δ + current verdict).
  - Campaign writeup (ideas/_campaigns/*.md, rendered).
  - Table of ALL its experiments (linking to exp pages).
  - Back-link to master grid.

Hard rules honoured: self-contained HTML (Google Fonts link OK w/ offline
serif/mono fallback), no JS framework, single inline sort/filter script, PNG
(not SVG) for matplotlib panels (SVG sparklines fine), no emoji-as-data.

Usage:
    python -m steering.dashboard
"""

from __future__ import annotations

import html
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from .eval import COMPOSITE_FORMULA, COMPOSITE_WEIGHTS, composite_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "autoresearch_results"

# Project doc / hosting URLs (mast-pill CTAs + footer cross-links).
REPO_URL = "https://github.com/dlmastery/steeringresearch"
PAGES_URL = "https://dlmastery.github.io/steeringresearch/"
PAPER_URL = REPO_URL + "/blob/master/paper/PAPER.md"
FINDINGS_URL = REPO_URL + "/blob/master/FINDINGS.md"
IDEATABLE_URL = REPO_URL + "/blob/master/IDEA_TABLE.md"
FINGERPRINT = composite_fingerprint()

# ---------------------------------------------------------------------------
# Shared CSS — the brutalist editorial palette ported verbatim from the
# reference dashboard (:root vars, mast-pill, how-to-read, headline-ribbon,
# hyp-grid, verdict colours, table.runs, card / panel-2col, kn-strip, footer).
# A single CSS block is reused across all three tiers; the per-page <body> tag
# distinguishes them only by which sections render.
# ---------------------------------------------------------------------------
SHARED_CSS = r"""
 :root{
   --ink:#0a0a0d; --paper:#e6e1d6; --paper-dim:#a89e8c;
   --rule:#1c1c20; --rule-bright:#2a2a30;
   --panel:#111114; --panel2:#16161a;
   --accent:#bb8c4d; --accent-dim:#7a5e36;
   --v-pass:#3fb950; --v-minor:#d29922; --v-major:#f0883e; --v-broken:#f85149;
   --v-novel:#a371f7; --v-derivative:#58a6ff; --v-numerology:#8b949e;
   --v-falsified:#db6d28; --v-infra:#8b949e;
 }
 *{margin:0;padding:0;box-sizing:border-box;}
 html{scroll-behavior:smooth;}
 html,body{background:var(--ink);}
 body{font-family:'Source Serif 4','Charter','Source Serif Pro',Georgia,serif;
      color:var(--paper);padding:32px 36px 80px;line-height:1.6;
      max-width:1320px;margin:0 auto;font-size:15px;
      font-variant-numeric:tabular-nums;position:relative;}
 a{color:var(--v-derivative);text-decoration:none;border-bottom:1px solid transparent;
   transition:border-color 160ms ease;}
 a:hover{border-bottom-color:var(--v-derivative);}
 h1{font-family:'Source Serif 4',Georgia,serif;font-weight:600;
    font-size:36px;line-height:1.12;color:var(--paper);letter-spacing:-0.005em;
    margin-bottom:6px;}
 h2{font-family:'Source Serif 4',Georgia,serif;font-weight:600;
    font-size:21px;color:var(--paper);margin-bottom:12px;letter-spacing:-0.003em;}
 h3{font-family:'IBM Plex Mono',ui-monospace,monospace;font-weight:600;font-size:11px;
    text-transform:uppercase;letter-spacing:0.16em;color:var(--paper-dim);
    margin:16px 0 10px 0;}
 .sub{font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:11px;
      text-transform:uppercase;letter-spacing:0.16em;color:var(--paper-dim);
      margin:2px 0 4px;}
 .mono{font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:0.85em;}
 code{background:var(--ink);padding:1px 5px;font-family:'IBM Plex Mono',monospace;
      border:1px solid var(--rule);font-size:0.9em;color:var(--paper);}
 /* masthead CTA pills */
 .mast-row{margin:10px 0 6px 0;display:flex;flex-wrap:wrap;gap:6px;}
 .mast-pill{display:inline-block;padding:3px 10px;
            border:1px solid var(--accent-dim);color:var(--accent);
            font-family:'IBM Plex Mono',monospace;font-size:10px;
            text-transform:uppercase;letter-spacing:0.16em;font-weight:600;
            text-decoration:none;}
 .mast-pill:hover{background:var(--accent-dim);color:var(--ink);
                  border-bottom-color:var(--accent-dim);}
 .mast-pill.repo{border-color:var(--v-novel);color:var(--v-novel);}
 .mast-pill.repo:hover{background:var(--v-novel);color:var(--ink);}
 .mast-pill.lit{border-color:var(--v-derivative);color:var(--v-derivative);}
 .mast-pill.lit:hover{background:var(--v-derivative);color:var(--ink);}
 .mast-pill.paper{border-color:var(--v-pass);color:var(--v-pass);}
 .mast-pill.paper:hover{background:var(--v-pass);color:var(--ink);}
 .live-link{display:inline-block;padding:3px 10px;border:1px solid var(--v-pass);
            color:var(--v-pass);font-family:'IBM Plex Mono',monospace;
            font-size:10px;text-transform:uppercase;letter-spacing:0.18em;
            font-weight:600;text-decoration:none;}
 .live-link:hover{background:var(--v-pass);color:var(--ink);}
 /* formula chip */
 .formula-chip{background:var(--ink);border:1px solid var(--rule);
               border-left:2px solid var(--accent);padding:11px 16px;
               font-family:'IBM Plex Mono',monospace;font-size:0.8em;
               margin:8px 0 14px;color:var(--paper);word-break:break-word;}
 .formula-chip .fp{color:var(--paper-dim);font-size:0.92em;}
 /* KPI ribbon */
 .ribbon{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
         gap:1px;background:var(--rule);border:1px solid var(--rule);
         margin:14px 0 24px;}
 .kpi{background:var(--panel);padding:14px 16px;}
 .kpi .label{color:var(--paper-dim);font-size:9.5px;
             font-family:'IBM Plex Mono',monospace;text-transform:uppercase;
             letter-spacing:0.16em;}
 .kpi .value{font-family:'Source Serif 4',Georgia,serif;font-size:24px;
             font-weight:600;margin-top:4px;color:var(--paper);
             letter-spacing:-0.005em;line-height:1.1;}
 .kpi .value small{font-size:11px;color:var(--paper-dim);font-weight:400;}
 .kpi.positive{box-shadow:inset 3px 0 0 var(--v-pass);}
 .kpi.negative{box-shadow:inset 3px 0 0 var(--v-broken);}
 .kpi.neutral{box-shadow:inset 3px 0 0 var(--accent-dim);}
 /* how-to-read */
 .how-to-read{background:var(--panel);border:1px solid var(--rule);
              border-left:2px solid var(--v-pass);padding:18px 24px;
              margin:18px 0;font-family:'Source Serif 4',Georgia,serif;
              line-height:1.6;}
 .how-to-read h3{font-family:'IBM Plex Mono',monospace;font-size:11px;
                 text-transform:uppercase;letter-spacing:0.18em;
                 color:var(--paper-dim);margin:0 0 12px 0;}
 .how-to-read ul{list-style:none;margin:0;padding:0;
                 display:grid;grid-template-columns:1fr 1fr;gap:10px 22px;}
 .how-to-read li{padding-left:14px;border-left:1px solid var(--rule-bright);
                 color:var(--paper);font-size:14px;line-height:1.55;}
 .how-to-read li b{color:var(--paper);font-weight:600;}
 .how-to-read code{background:var(--ink);padding:1px 5px;
                   font-family:'IBM Plex Mono',monospace;font-size:0.88em;
                   border:1px solid var(--rule);}
 .how-to-read .chip-scr{background:#3a2e13;color:#e0b15a;padding:1px 6px;
                        border-radius:3px;font-family:'IBM Plex Mono',monospace;
                        font-size:0.82em;font-weight:600;letter-spacing:0.04em;}
 .how-to-read .chip-eval{background:#13303a;color:#5ac8e0;padding:1px 6px;
                         border-radius:3px;font-family:'IBM Plex Mono',monospace;
                         font-size:0.82em;font-weight:600;letter-spacing:0.04em;}
 @media(max-width:880px){.how-to-read ul{grid-template-columns:1fr;}}
 /* headline ribbon */
 .headline-ribbon{background:var(--panel);border:1px solid var(--rule);
                  border-left:2px solid var(--accent);padding:18px 24px;
                  margin:18px 0;font-family:'Source Serif 4',Georgia,serif;
                  line-height:1.55;}
 .headline-ribbon .tag-note{font-family:'IBM Plex Mono',monospace;font-weight:600;
                  color:var(--accent);font-size:10px;text-transform:uppercase;
                  letter-spacing:0.16em;display:block;margin-bottom:10px;}
 .headline-ribbon h2{font-size:19px;margin:0 0 8px;}
 .headline-ribbon ul{margin:6px 0 0 20px;padding:0;}
 .headline-ribbon li{margin:5px 0;font-size:0.95em;}
 .headline-ribbon code{background:var(--ink);padding:1px 5px;
                  font-family:'IBM Plex Mono',monospace;border:1px solid var(--rule);
                  font-size:0.86em;color:var(--paper);}
 /* generic card / section */
 .card,section.card{background:var(--panel);border:1px solid var(--rule);
       padding:22px 26px;margin-bottom:22px;position:relative;}
 .card::before{content:"";position:absolute;top:0;left:0;width:48px;
               height:1px;background:var(--accent);}
 .card p{margin-bottom:10px;font-size:0.96em;line-height:1.6;}
 .card ul{margin:6px 0 12px 22px;font-size:0.95em;line-height:1.6;}
 .card ul li{margin-bottom:4px;}
 .card img{max-width:100%;height:auto;background:#fff;padding:4px;
           border:1px solid var(--rule);display:block;}
 .card .cap{color:var(--paper-dim);font-size:11px;margin:8px 0 0;
            font-family:'IBM Plex Mono',monospace;line-height:1.5;}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:8px;}
 .grid .card{margin-bottom:0;}
 .panel-2col{grid-column:1 / 3;}
 @media(max-width:980px){.grid{grid-template-columns:1fr;}
   .panel-2col{grid-column:auto;}}
 /* hypothesis status grid */
 .hyp-grid-summary{font-family:'IBM Plex Mono',monospace;font-size:0.82em;
                   color:var(--paper-dim);margin:2px 0 8px;}
 .hyp-grid-summary b{color:var(--paper);font-weight:600;}
 .legend-row{margin:8px 0 14px 0;font-size:0.78em;color:var(--paper-dim);
             font-family:'IBM Plex Mono',monospace;letter-spacing:0.06em;}
 .legend-row .swatch{display:inline-block;width:12px;height:12px;
                     vertical-align:middle;margin:0 4px 0 10px;
                     border:1px solid var(--rule-bright);}
 .hyp-grid-row{display:flex;align-items:center;margin-bottom:4px;
               font-family:'IBM Plex Mono',monospace;font-size:0.78em;}
 .hyp-grid-row .gid{width:120px;color:var(--paper-dim);flex:0 0 120px;
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
 .hyp-cell{width:30px;height:24px;margin-right:3px;
           display:inline-flex;align-items:center;justify-content:center;
           font-size:0.72em;color:#0a0a0d;cursor:pointer;text-decoration:none;
           border:1px solid var(--rule);font-weight:600;}
 .hyp-cell:hover{outline:1px solid var(--accent);border-bottom-color:var(--rule);}
 .hyp-cell.empty{background:var(--rule);color:#484f58;cursor:default;
                 border-color:transparent;}
 .v-supported{background:var(--v-pass);}
 .v-falsified{background:var(--v-falsified);color:#fff;}
 .v-directional{background:var(--v-minor);}
 .v-inconclusive{background:var(--v-numerology);}
 .v-pending{background:#2a2e35;color:#6a7079;}
 /* runs tables */
 table.runs{width:100%;border-collapse:collapse;font-size:0.82em;
            background:var(--panel);border:1px solid var(--rule);}
 table.runs th{background:transparent;color:var(--paper-dim);text-align:right;
               padding:8px 11px;border-bottom:1px solid var(--rule-bright);
               font-weight:500;text-transform:uppercase;font-size:9.5px;
               letter-spacing:0.14em;cursor:pointer;user-select:none;
               font-family:'IBM Plex Mono',monospace;white-space:nowrap;}
 table.runs th:first-child,table.runs th.l{text-align:left;}
 table.runs td{padding:7px 11px;border-bottom:1px solid var(--rule);
               text-align:right;color:var(--paper);
               font-family:'IBM Plex Mono',monospace;font-size:0.95em;
               white-space:nowrap;}
 table.runs td:first-child,table.runs td.l{text-align:left;}
 table.runs tr.champion{box-shadow:inset 3px 0 0 var(--accent);
                        background:rgba(187,140,77,0.06);}
 table.runs tr.champion td{font-weight:600;}
 table.runs tr.champion td:first-child::before{content:"\2605 ";color:var(--accent);}
 table.runs tr.neg-comp td.comp{color:var(--v-broken);}
 table.runs tr.vr-supported td.l:first-child{box-shadow:inset 3px 0 0 var(--v-pass);}
 table.runs tr.vr-falsified td.l:first-child{box-shadow:inset 3px 0 0 var(--v-falsified);}
 table.runs tr.vr-directional td.l:first-child{box-shadow:inset 3px 0 0 var(--v-minor);}
 table.runs tr.vr-inconclusive td.l:first-child{box-shadow:inset 3px 0 0 var(--v-numerology);}
 .arr{font-size:9px;color:var(--accent);margin-left:3px;}
 .n-chip{display:inline-block;padding:0 5px;font-size:8.5px;
         font-family:'IBM Plex Mono',monospace;letter-spacing:0.08em;
         text-transform:uppercase;border:1px solid;margin-left:5px;
         vertical-align:middle;}
 .n-chip.eval{color:var(--v-pass);border-color:var(--v-pass);}
 .n-chip.scr{color:var(--v-minor);border-color:var(--v-minor);}
 .filter-box{width:360px;max-width:60vw;padding:8px 11px;background:var(--ink);
             border:1px solid var(--rule);color:var(--paper);
             font-family:'IBM Plex Mono',monospace;font-size:12px;margin-bottom:6px;}
 .group-section{margin-top:26px;background:var(--panel);border:1px solid var(--rule);
                padding:20px 24px;position:relative;}
 .group-section::before{content:"";position:absolute;top:0;left:0;
                        width:64px;height:1px;background:var(--accent);}
 .group-section h2{margin:0;}
 .group-section h2 .cnt{color:var(--paper-dim);font-weight:400;font-size:0.62em;
                        font-family:'IBM Plex Mono',monospace;letter-spacing:0.04em;}
 .group-desc{color:var(--paper-dim);font-size:0.92em;margin:4px 0 14px;
             max-width:980px;line-height:1.6;}
 .tablewrap{overflow-x:auto;}
 /* per-experiment header */
 .head-grid{display:grid;grid-template-columns:1fr auto;gap:24px;
            align-items:start;padding-bottom:18px;
            border-bottom:1px solid var(--rule);margin-bottom:24px;}
 .head-left .tag-display{font-family:'Source Serif 4',Georgia,serif;
    font-size:40px;font-weight:600;line-height:1.05;color:var(--paper);
    letter-spacing:-0.012em;word-break:break-word;}
 .head-right{display:flex;flex-direction:column;align-items:flex-end;gap:8px;
             min-width:220px;}
 .head-right .back{font-family:'IBM Plex Mono',monospace;font-size:11px;
    text-transform:uppercase;letter-spacing:0.18em;color:var(--paper-dim);}
 .pill{display:inline-block;background:transparent;border:1px solid var(--rule-bright);
       padding:3px 10px;font-size:0.72em;color:var(--paper-dim);
       font-family:'IBM Plex Mono',monospace;margin:0 4px 4px 0;
       text-transform:uppercase;letter-spacing:0.12em;}
 .pill.hyp{border-color:var(--v-derivative);color:var(--v-derivative);}
 .pill.grp{border-color:var(--v-novel);color:var(--v-novel);}
 .pill.ds{border-color:var(--accent-dim);color:var(--accent);}
 .verdict-row{display:flex;flex-wrap:wrap;gap:4px;justify-content:flex-end;}
 .vbadge{display:inline-block;padding:3px 9px;font-family:'IBM Plex Mono',monospace;
         font-size:0.7em;font-weight:600;text-transform:uppercase;
         letter-spacing:0.1em;border:1px solid;}
 .vbadge.keep{color:var(--v-pass);border-color:var(--v-pass);}
 .vbadge.discard{color:var(--v-broken);border-color:var(--v-broken);}
 .vbadge.prov{color:var(--paper-dim);border-color:var(--rule-bright);}
 /* key-numbers strip */
 .kn-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
           gap:1px;background:var(--rule);border:1px solid var(--rule);
           margin-bottom:24px;}
 .kn-tile{background:var(--panel);padding:18px;position:relative;overflow:hidden;}
 .kn-tile .kn-val{font-family:'Source Serif 4',Georgia,serif;font-size:26px;
                  font-weight:600;line-height:1.05;color:var(--paper);
                  letter-spacing:-0.005em;}
 .kn-tile .kn-lbl{font-family:'IBM Plex Mono',monospace;font-size:9.5px;
                  text-transform:uppercase;letter-spacing:0.16em;
                  color:var(--paper-dim);margin-top:6px;}
 .kn-tile.tint-pos{box-shadow:inset 3px 0 0 var(--v-pass);}
 .kn-tile.tint-neg{box-shadow:inset 3px 0 0 var(--v-broken);}
 .kn-tile.tint-neu{box-shadow:inset 3px 0 0 var(--accent-dim);}
 /* generic key/value + breakdown tables */
 .kvtable{width:100%;border-collapse:collapse;font-size:0.9em;}
 .kvtable td,.kvtable th{padding:8px 12px;border-bottom:1px solid var(--rule);
                         text-align:left;}
 .kvtable th{color:var(--paper-dim);font-family:'IBM Plex Mono',monospace;
             font-size:10px;text-transform:uppercase;letter-spacing:0.16em;
             border-bottom:1px solid var(--rule-bright);font-weight:500;}
 .kvtable td.k{color:var(--paper-dim);width:46%;}
 .kvtable td.v{color:var(--paper);font-family:'IBM Plex Mono',monospace;}
 .breakdown td.term{font-family:'IBM Plex Mono',monospace;}
 .pos{color:var(--v-pass);} .neg{color:var(--v-broken);} .mut{color:var(--paper-dim);}
 .warn{color:var(--v-major);font-style:italic;}
 /* reasoning + markdown body */
 .md-body p{margin:8px 0 12px;} .md-body p:first-child{margin-top:0;}
 .md-body strong,.md-body b{color:var(--paper);font-weight:600;}
 .md-body em,.md-body i{font-style:italic;}
 .md-body code{font-family:'IBM Plex Mono',monospace;font-size:0.86em;
    background:var(--panel2);padding:1px 5px;border-radius:3px;border:none;}
 .md-body pre{font-family:'IBM Plex Mono',monospace;font-size:0.85em;
    background:var(--panel2);padding:10px 12px;border-radius:5px;
    border:1px solid var(--rule);overflow-x:auto;margin:10px 0;white-space:pre-wrap;}
 .md-body ul,.md-body ol{margin:8px 0 12px 22px;}
 .md-body li{margin:3px 0;}
 .md-body blockquote{border-left:3px solid var(--accent-dim);padding:6px 14px;
    margin:10px 0;color:var(--paper-dim);}
 .md-body table{border-collapse:collapse;margin:10px 0;font-size:0.92em;width:auto;}
 .md-body th,.md-body td{border:1px solid var(--rule);padding:6px 10px;text-align:left;}
 .md-body th{background:var(--panel2);font-weight:600;}
 .md-body h2,.md-body h3,.md-body h4{color:var(--paper);text-transform:none;
    letter-spacing:0;font-family:'Source Serif 4',Georgia,serif;margin:12px 0 6px;}
 .reason-section{margin-bottom:16px;}
 .reason-section .lbl{color:var(--accent);font-family:'IBM Plex Mono',monospace;
    font-size:10px;text-transform:uppercase;letter-spacing:0.18em;
    margin-bottom:6px;font-weight:600;}
 .quote{border-left:2px solid var(--accent);padding:10px 18px;background:var(--ink);
        font-size:0.95em;color:var(--paper);margin:10px 0;line-height:1.55;}
 /* composite stacked bar */
 .comp-stack{display:flex;min-height:34px;border:1px solid var(--rule);
             margin:14px 0 8px;font-family:'IBM Plex Mono',monospace;
             font-size:9.5px;flex-wrap:wrap;}
 .comp-stack .seg{display:flex;align-items:center;justify-content:center;
                  color:var(--ink);font-weight:600;border-right:1px solid var(--ink);
                  padding:4px 6px;white-space:nowrap;overflow:hidden;}
 .comp-stack .seg.pos{background:var(--v-pass);}
 .comp-stack .seg.cost{background:var(--v-major);}
 /* samples + stack matrix */
 .charts{display:grid;grid-template-columns:1fr 1fr;gap:18px;}
 @media(max-width:980px){.charts{grid-template-columns:1fr;}
   .head-grid{grid-template-columns:1fr;}.head-right{align-items:flex-start;}
   .verdict-row{justify-content:flex-start;}}
 pre.sample{background:var(--ink);border:1px solid var(--rule);
            border-left:2px solid var(--accent-dim);padding:10px 12px;
            white-space:pre-wrap;word-break:break-word;max-height:340px;
            overflow:auto;font-size:0.82em;font-family:'IBM Plex Mono',monospace;
            color:var(--paper);}
 table.stackmx{width:auto;border-collapse:collapse;}
 table.stackmx td.sc{text-align:center;font-weight:600;font-size:11px;
            white-space:nowrap;padding:7px 9px;border:1px solid var(--rule);}
 table.stackmx th{white-space:nowrap;padding:7px 9px;color:var(--paper-dim);
            font-family:'IBM Plex Mono',monospace;font-size:10px;
            text-transform:uppercase;letter-spacing:0.1em;border:1px solid var(--rule);}
 .sc.stack{background:#16331f;color:#7fe0a0;} .sc.care{background:#3a3010;color:#e6c25a;}
 .sc.compete{background:#3a1616;color:#f08080;} .sc.self{background:#1a1d24;color:var(--paper-dim);}
 .sc.measured{outline:2px solid #fff;outline-offset:-2px;}
 .sc .meas{font-size:9px;color:#fff;font-weight:700;}
 .sc.scl{display:inline-block;padding:1px 6px;border-radius:4px;font-weight:600;}
 /* deep-inspection accordions */
 details.deep{margin:14px 0;border-top:1px solid var(--rule);padding-top:14px;}
 details.deep > summary{cursor:pointer;list-style:none;
    font-family:'IBM Plex Mono',monospace;font-size:11px;text-transform:uppercase;
    letter-spacing:0.18em;color:var(--paper-dim);padding:6px 0;}
 details.deep > summary:hover{color:var(--accent);}
 details.deep > summary::-webkit-details-marker{display:none;}
 details.deep > summary::before{content:"\25B8";margin-right:10px;color:var(--accent);}
 details.deep[open] > summary::before{content:"\25BE";}
 details.deep > .body{padding:14px 0 6px 24px;font-size:0.95em;
    border-left:1px solid var(--rule);margin-left:5px;margin-top:6px;}
 /* cross-references grid */
 .xrefs-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
             gap:10px;margin-top:6px;}
 .xref-card{display:block;background:var(--ink);border:1px solid var(--rule);
            padding:12px 14px;text-decoration:none;
            transition:border-color 160ms ease,transform 160ms ease;}
 .xref-card:hover{border-color:var(--accent);transform:translateY(-1px);border-bottom-color:var(--accent);}
 .xref-card .xref-lbl{font-family:'IBM Plex Mono',monospace;font-size:9px;
    text-transform:uppercase;letter-spacing:0.18em;color:var(--paper-dim);margin-bottom:6px;}
 .xref-card .xref-tag{font-family:'IBM Plex Mono',monospace;font-size:0.88em;
    color:var(--paper);font-weight:500;}
 .xref-card .xref-meta{font-family:'IBM Plex Mono',monospace;font-size:0.78em;
    color:var(--paper-dim);margin-top:4px;}
 /* footer */
 .doc-footer{margin:24px 0 6px;font-family:'IBM Plex Mono',monospace;
             font-size:10px;color:var(--paper-dim);letter-spacing:0.12em;
             text-transform:uppercase;}
 .doc-footer a{color:var(--accent);border-bottom:1px dotted var(--accent-dim);}
 .doc-footer a:hover{color:var(--v-pass);border-bottom-color:var(--v-pass);}
 .meta{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--paper-dim);
       margin-top:18px;padding-top:18px;border-top:1px solid var(--rule);
       line-height:1.9;letter-spacing:0.04em;}
 .meta code{background:var(--ink);padding:1px 5px;color:var(--paper);border:1px solid var(--rule);}
 .empty{color:var(--paper-dim);font-style:italic;font-size:0.9em;}
 /* dsbench-style tab bar (ported, re-skinned to the editorial palette) */
 .tabs{display:flex;gap:0;flex-wrap:wrap;border-bottom:1px solid var(--rule-bright);
       margin:30px 0 0 0;}
 .tab{padding:9px 16px;cursor:pointer;color:var(--paper-dim);
      border-bottom:2px solid transparent;font-family:'IBM Plex Mono',monospace;
      font-size:11px;text-transform:uppercase;letter-spacing:0.14em;font-weight:600;
      user-select:none;}
 .tab:hover{color:var(--paper);}
 .tab.active{color:var(--accent);border-bottom-color:var(--accent);}
 .tab-pane{display:none;padding:16px 0;}
 .tab-pane.active{display:block;}
 /* dsbench-style rich filter bar (regex + dropdown + toggle pills) */
 .toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:12px 0;
          padding:10px 14px;background:var(--panel);border:1px solid var(--rule);}
 .toolbar label{font-family:'IBM Plex Mono',monospace;font-size:10px;
                text-transform:uppercase;letter-spacing:0.12em;color:var(--paper-dim);
                display:flex;align-items:center;gap:6px;}
 .toolbar select{background:var(--ink);color:var(--paper);border:1px solid var(--rule);
                 padding:6px 8px;font-family:'IBM Plex Mono',monospace;font-size:11px;}
 .toolbar .fpill{background:var(--ink);border:1px solid var(--rule-bright);
                 padding:4px 11px;border-radius:13px;font-family:'IBM Plex Mono',monospace;
                 font-size:10px;letter-spacing:0.1em;text-transform:uppercase;
                 cursor:pointer;color:var(--paper-dim);user-select:none;}
 .toolbar .fpill:hover{color:var(--paper);border-color:var(--accent-dim);}
 .toolbar .fpill.active{background:var(--accent);color:var(--ink);
                        border-color:var(--accent);font-weight:600;}
 .toolbar .fpill.v-active{background:var(--v-pass);color:var(--ink);border-color:var(--v-pass);}
 .toolbar .fpill.bad-active{background:var(--v-falsified);color:#fff;border-color:var(--v-falsified);}
 /* auto-expand inline detail row */
 tr.detail-row{display:none;}
 tr.detail-row.open{display:table-row;}
 tr.detail-row > td{background:var(--ink);border-bottom:1px solid var(--rule-bright);
                    box-shadow:inset 3px 0 0 var(--accent);padding:0;}
 .detail-box{padding:12px 16px;font-family:'IBM Plex Mono',monospace;font-size:11px;
             line-height:1.7;color:var(--paper);}
 .detail-box .dl{display:flex;flex-wrap:wrap;gap:5px 18px;margin-bottom:8px;}
 .detail-box .dl span{color:var(--paper-dim);}
 .detail-box .dl b{color:var(--paper);font-weight:600;}
 .detail-box .vsnip{border-left:2px solid var(--accent-dim);padding:4px 12px;
                    margin:6px 0;color:var(--paper-dim);font-style:italic;
                    max-width:880px;}
 table.runs tr.task-row[data-expandable] td:first-child{cursor:pointer;}
 /* hill-climb tier */
 .hc-callout{background:var(--panel);border:1px solid var(--rule);
             border-left:2px solid var(--v-pass);padding:16px 22px;margin:14px 0;}
 .hc-callout b{color:var(--paper);}
 .hc-empty{background:var(--panel);border:1px solid var(--rule);
           border-left:2px solid var(--v-minor);padding:18px 24px;margin:14px 0;
           color:var(--paper-dim);line-height:1.6;}
 .hc-empty b{color:var(--paper);}
 .hc-empty code{background:var(--ink);padding:1px 5px;border:1px solid var(--rule);
                font-family:'IBM Plex Mono',monospace;font-size:0.88em;color:var(--paper);}
"""

_FONT_LINK = (
    "<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">"
    "<link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>"
    "<link href=\"https://fonts.googleapis.com/css2?"
    "family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600;8..60,700&"
    "family=IBM+Plex+Mono:wght@400;500;600&display=swap\" rel=\"stylesheet\">"
)


# ===========================================================================
# I/O helpers
# ===========================================================================
def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "no-git"
    except Exception:
        return "no-git"


def load_rows(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not log_path.exists():
        return rows
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_reasoning(results_dir: Path) -> dict:
    path = results_dir / "reasoning_annotations.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _esc(s: object) -> str:
    return html.escape("" if s is None else str(s))


def _flatten(row: dict) -> dict:
    """Lift nested config fields to the top level for table rendering."""
    flat = dict(row)
    cfg = row.get("config", {}) or {}
    for k in ("tag", "operation", "source", "alpha", "behavior", "model",
              "quant", "hypothesis_id"):
        if flat.get(k) is None:
            flat[k] = cfg.get(k)
    return flat


def _tier_of(row: dict) -> str:
    explicit = row.get("tier")
    n = int(row.get("n_seeds", 1) or 1)
    if explicit in ("SCREENING", "EVALUATION"):
        return explicit
    return "EVALUATION" if n >= 7 else "SCREENING"


def _num(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key)
    try:
        x = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x):
        return default
    return x


_MODEL_SHORT = {
    "fake": "fake",
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B",
    "models/google/gemma-3-270m-it": "gemma-3-270m",
    "models/google/gemma-3-1b-it": "gemma-3-1b",
}


def _short_model(m: object) -> str:
    s = str(m or "")
    return _MODEL_SHORT.get(s, s.split("/")[-1] if s else "—")


# ===========================================================================
# Hypothesis registry (IDEA_TABLE.md) — the canonical 70-hypothesis status set
# ===========================================================================
# Verdict status -> (css class, swatch glyph) for the grid + run-row tint.
VERDICT_CLASS = {
    "SUPPORTED": "v-supported",
    "FALSIFIED": "v-falsified",
    "DIRECTIONAL": "v-directional",
    "INCONCLUSIVE": "v-inconclusive",
    "PENDING": "v-pending",
}
VERDICT_LEGEND = [
    ("SUPPORTED", "v-supported", "supported"),
    ("FALSIFIED", "v-falsified", "falsified"),
    ("DIRECTIONAL", "v-directional", "directional"),
    ("INCONCLUSIVE", "v-inconclusive", "inconclusive"),
    ("PENDING", "v-pending", "pending"),
]

# Block layout for the grid (E1..E50 in A-F, then N1..N20).
HYP_BLOCKS = [
    ("Block A · foundations", [f"E{i}" for i in range(1, 9)]),
    ("Block B · CAST / gating", [f"E{i}" for i in range(9, 17)]),
    ("Block C · stacking", [f"E{i}" for i in range(17, 27)]),
    ("Block D · geometry", [f"E{i}" for i in range(27, 34)]),
    ("Block E · mechanistic", [f"E{i}" for i in range(34, 41)]),
    ("Block F · robustness", [f"E{i}" for i in range(41, 51)]),
    ("Novel N1-N12", [f"N{i}" for i in range(1, 13)]),
    ("Novel N13-N20", [f"N{i}" for i in range(13, 21)]),
]


def parse_idea_table(idea_table_md: Path) -> dict[str, dict]:
    """Parse IDEA_TABLE.md into {id: {title, hypothesis, metric, status,
    verdict, rung, idea_dir}}.

    The status column holds e.g. ``SUPPORTED(scr): ...`` / ``FALSIFIED(scr): ...``
    / ``DIRECTIONAL(...)`` / ``INCONCLUSIVE(...)`` / ``PENDING``. The leading
    verdict keyword is extracted; the trailing prose is kept as the verdict
    detail. Rows look like::

        | E2 | Fisher layer selection | A | A1 (site/layer) | <hypothesis> |
            <metric+threshold> | FALSIFIED(scr): rho=+0.14 ... | 2 | `ideas/20_.../`
    """
    out: dict[str, dict] = {}
    if not idea_table_md.exists():
        return out
    text = idea_table_md.read_text(encoding="utf-8", errors="replace")
    id_re = re.compile(r"^(E\d{1,2}|N\d{1,2})$")
    verdict_kw = re.compile(
        r"\b(SUPPORTED|FALSIFIED|DIRECTIONAL|INCONCLUSIVE|PENDING)\b")
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not cells or not id_re.match(cells[0]):
            continue
        hid = cells[0]
        title = cells[1] if len(cells) > 1 else ""
        hypothesis = cells[4] if len(cells) > 4 else ""
        metric = cells[5] if len(cells) > 5 else ""
        status_cell = cells[6] if len(cells) > 6 else ""
        rung = cells[7] if len(cells) > 7 else ""
        idea_dir = cells[8] if len(cells) > 8 else ""
        m = verdict_kw.search(status_cell)
        verdict = m.group(1) if m else "PENDING"
        out[hid] = {
            "id": hid,
            "title": title,
            "hypothesis": hypothesis,
            "metric": metric,
            "status_cell": status_cell,
            "verdict": verdict,
            "verdict_detail": status_cell,
            "rung": rung,
            "idea_dir": idea_dir.strip("`").rstrip("/"),
        }
    return out


# ---------------------------------------------------------------------------
# Tag <-> hypothesis resolution (drives run-row tint, hyp-page grouping, links)
# ---------------------------------------------------------------------------
def hypothesis_for_tag(tag: Optional[str]) -> Optional[str]:
    """Map an experiment-row tag prefix to its primary hypothesis id (or None
    for the infra/plumbing rows)."""
    t = str(tag or "").lower()
    if not t or "plumbing" in t or "fakelm" in t:
        return None
    # hill-climb tags (HC-*) carry their target hypothesis id as a token,
    # e.g. HC-E3-layer-18 -> E3, HC_N20_alpha -> N20.
    if t.startswith("hc-") or t.startswith("hc_"):
        m = re.search(r"\b([en]\d{1,2})\b", t)
        if m:
            return m.group(1).upper()
        return None
    if t.startswith("c1-e2") or t.startswith("c1_e2"):
        return "E2"
    if t.startswith("c3"):          # C3 / C3b operation comparison -> E27
        return "E27"
    if t.startswith("c4-n20") or t.startswith("c4_n20"):
        return "N20"
    if t.startswith("c6"):          # cross-scale alpha cliff -> E3
        return "E3"
    if t.startswith("e3"):
        return "E3"
    return None


def is_hillclimb(row: dict) -> bool:
    """A hill-climb (rung-2.5 coordinate-descent) row is flagged by a tag prefix
    ``HC-`` / ``HC_`` (case-insensitive) or an explicit config marker
    (``config.hillclimb`` truthy or ``config.phase == 'hillclimb'``)."""
    cfg = row.get("config", {}) or {}
    tag = str(cfg.get("tag") or row.get("tag") or "")
    t = tag.lower()
    if t.startswith("hc-") or t.startswith("hc_"):
        return True
    if cfg.get("hillclimb") or row.get("hillclimb"):
        return True
    if str(cfg.get("phase") or row.get("phase") or "").lower() == "hillclimb":
        return True
    return False


def hypothesis_label(row: dict) -> str:
    hid = hypothesis_for_tag(row.get("tag") or (row.get("config", {}) or {}).get("tag"))
    return hid or "—"


def _idea_id(idea_dir: Path) -> str:
    """The leading numeric id of an ideas dir, e.g. '10' from '10_foo_bar'."""
    m = re.match(r"(\d+)", idea_dir.name)
    return m.group(1) if m else idea_dir.name


def resolve_idea_dir(row: dict, idea_dirs: list[Path]) -> Optional[Path]:
    """Token-overlap fallback: map a row to an ideas/<dir> when no E/N tag
    prefix resolves (used for legacy / non-campaign tags). Priority: explicit
    config hypothesis_id, then tag-token overlap with the dir slug."""
    if not idea_dirs:
        return None
    cfg = row.get("config", {}) or {}
    hyp = row.get("hypothesis_id") or cfg.get("hypothesis_id")
    tag = str(row.get("tag") or cfg.get("tag") or "").lower()
    if hyp is not None:
        h = str(hyp).lower().lstrip("h").lstrip("e").lstrip("n")
        for d in idea_dirs:
            if _idea_id(d) == h or h in d.name.lower():
                return d
    tag_tokens = set(re.split(r"[^a-z0-9]+", tag)) - {""}
    best, best_score = None, 0
    for d in idea_dirs:
        slug_tokens = {t for t in (set(re.split(r"[^a-z0-9]+", d.name.lower())) - {""})
                       if not t.isdigit()}
        score = len(tag_tokens & slug_tokens)
        if score > best_score:
            best, best_score = d, score
    return best if best_score > 0 else None


def resolve_hyp_id(row: dict, repo_root: Path, idea_dirs: list[Path]) -> Optional[str]:
    """Unified row -> hypothesis-page id. Prefers the E/N registry id (real
    campaign data); falls back to the numeric ideas/<dir> id (legacy / tests).
    Both ids resolve to ``hyp/<id>.html``."""
    hid = hypothesis_for_tag(row.get("config", {}).get("tag") if row.get("config") else row.get("tag"))
    if hid:
        return hid
    d = resolve_idea_dir(row, idea_dirs)
    if d is not None:
        return _idea_id(d)
    return None


# ---------------------------------------------------------------------------
# ideas/<dir> resolution + IDEA.md parsing (statement / falsifier / predicted)
# ---------------------------------------------------------------------------
def list_idea_dirs(repo_root: Path) -> list[Path]:
    ideas = repo_root / "ideas"
    if not ideas.exists():
        return []
    return [d for d in sorted(ideas.iterdir())
            if d.is_dir() and not d.name.startswith("_")]


def _idea_dir_for_hyp(hid: str, repo_root: Path,
                      table: dict[str, dict]) -> Optional[Path]:
    """Resolve an ideas/<dir> for a hypothesis id, via the IDEA_TABLE idea_dir
    column then a numeric / token fallback."""
    info = table.get(hid, {})
    rel = info.get("idea_dir", "")
    if rel and rel.startswith("ideas/"):
        cand = repo_root / rel
        if cand.exists() and cand.is_dir():
            return cand
    # numeric mapping (10_/20_/30_ dirs encode E1/E2/E3 by their leading digits)
    num_map = {"E1": "10", "E2": "20", "E3": "30"}
    pref = num_map.get(hid)
    if pref:
        for d in list_idea_dirs(repo_root):
            if d.name.startswith(pref + "_"):
                return d
    return None


def _idea_dir_for_id(hid: str, repo_root: Path, table: dict[str, dict],
                     idea_dirs: list[Path]) -> Optional[Path]:
    """Resolve an ideas/<dir> for ANY hyp id — an E/N registry id (via
    _idea_dir_for_hyp) or a numeric idea-dir id (direct lookup)."""
    if hid in table:
        d = _idea_dir_for_hyp(hid, repo_root, table)
        if d is not None:
            return d
    if hid and hid.isdigit():
        for d in idea_dirs:
            if _idea_id(d) == hid:
                return d
    return None


def _hyp_info(hid: str, table: dict[str, dict], repo_root: Path,
              idea_dirs: list[Path]) -> dict:
    """Return a hypothesis-info dict for ANY hid. E/N ids come straight from the
    IDEA_TABLE registry; numeric idea-dir ids are synthesised from the idea
    dir's IDEA.md (statement -> hypothesis, with a PENDING verdict)."""
    if hid in table:
        info = dict(table[hid])
        if not info.get("title") or not info.get("hypothesis"):
            d = _idea_dir_for_id(hid, repo_root, table, idea_dirs)
            if d is not None:
                idea = _parse_idea_md(d)
                info.setdefault("title", idea.get("title", ""))
                info["title"] = info.get("title") or idea.get("title", "")
                info["hypothesis"] = info.get("hypothesis") or idea.get("statement", "")
        return info
    d = _idea_dir_for_id(hid, repo_root, table, idea_dirs)
    if d is not None:
        idea = _parse_idea_md(d)
        return {"id": hid, "title": idea.get("title", ""),
                "hypothesis": idea.get("statement", ""), "metric": "",
                "status_cell": "", "verdict": "PENDING", "verdict_detail": "",
                "rung": "", "idea_dir": str(d)}
    return {"id": hid, "title": "", "hypothesis": "", "metric": "",
            "verdict": "PENDING", "verdict_detail": ""}


def _parse_idea_md(idea_dir: Path) -> dict:
    out = {"statement": "", "falsifier": "", "predicted": "", "title": idea_dir.name}
    for fname in ("IDEA.md", "README.md"):
        p = idea_dir / fname
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^#\s+(.*)$", text, re.MULTILINE)
        if m and out["title"] == idea_dir.name:
            out["title"] = m.group(1).strip()

        def _section(name):
            mm = re.search(rf"^#{{1,6}}\s*{name}.*?$\n(.*?)(?=^#{{1,6}}\s|\Z)",
                           text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
            return mm.group(1).strip() if mm else ""

        if not out["statement"]:
            out["statement"] = (_section("Claim") or _section("Hypothesis")
                                or _section("TL;DR"))
        if not out["falsifier"]:
            out["falsifier"] = _section("Falsifier")
        if not out["predicted"]:
            out["predicted"] = (_section("Pre-registered prediction")
                                or _section("Predicted delta")
                                or _section("Predicted delta range")
                                or _section("Prediction"))
    return out


# ===========================================================================
# Minimal GFM-ish markdown -> HTML converter (NO literal ## / ** / |---| leak)
# ===========================================================================
def _md_inline(text: str) -> str:
    out = html.escape(text)
    code_spans: list[str] = []

    def _stash_code(m):
        code_spans.append(m.group(1))
        return f"\x00CODE{len(code_spans) - 1}\x00"

    out = re.sub(r"`([^`]+)`", _stash_code, out)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                 lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>',
                 out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", out)
    out = re.sub(r"(?<![\w_])_([^_\n]+)_(?![\w_])", r"<em>\1</em>", out)
    # any stray bold/italic markers left over must NOT leak as literal "**"
    out = out.replace("**", "").replace("__", "")
    for i, c in enumerate(code_spans):
        out = out.replace(f"\x00CODE{i}\x00", f"<code>{html.escape(c)}</code>")
    return out


def md_to_html(md: str) -> str:
    """Convert a GFM-ish markdown fragment to HTML. Supports ATX headings,
    bold/italic/code, fenced code, blockquotes, lists, and pipe tables (the
    |---| separator row is consumed, never leaked)."""
    if not md:
        return ""
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)

    def close_list(stack):
        while stack:
            out.append(f"</{stack.pop()}>")

    list_stack: list[str] = []

    while i < n:
        line = lines[i]

        if line.strip().startswith("```"):
            close_list(list_stack)
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(html.escape(lines[i]))
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            continue

        if ("|" in line and i + 1 < n
                and re.match(r"^\s*\|?[\s:|-]*-[-\s:|]*\|?\s*$", lines[i + 1])
                and "-" in lines[i + 1]):
            close_list(list_stack)
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            thead = "".join(f"<th>{_md_inline(h)}</th>" for h in header)
            body = ""
            for r in rows:
                body += "<tr>" + "".join(f"<td>{_md_inline(c)}</td>" for c in r) + "</tr>"
            out.append(f"<table><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>")
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            close_list(list_stack)
            lvl = min(len(m.group(1)) + 1, 6)
            out.append(f"<h{lvl}>{_md_inline(m.group(2).strip())}</h{lvl}>")
            i += 1
            continue

        if line.strip().startswith(">"):
            close_list(list_stack)
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(_md_inline(re.sub(r"^\s*>\s?", "", lines[i])))
                i += 1
            out.append("<blockquote>" + "<br>".join(buf) + "</blockquote>")
            continue

        m = re.match(r"^\s*[-*+]\s+(.*)$", line)
        if m:
            if not list_stack or list_stack[-1] != "ul":
                close_list(list_stack)
                list_stack.append("ul")
                out.append("<ul>")
            out.append(f"<li>{_md_inline(m.group(1))}</li>")
            i += 1
            continue

        m = re.match(r"^\s*\d+[.)]\s+(.*)$", line)
        if m:
            if not list_stack or list_stack[-1] != "ol":
                close_list(list_stack)
                list_stack.append("ol")
                out.append("<ol>")
            out.append(f"<li>{_md_inline(m.group(1))}</li>")
            i += 1
            continue

        if not line.strip():
            close_list(list_stack)
            i += 1
            continue

        close_list(list_stack)
        buf = [line]
        i += 1
        while i < n and lines[i].strip() and not re.match(
            r"^\s*([-*+]\s|\d+[.)]\s|#{1,6}\s|>|```)", lines[i]
        ) and "|" not in lines[i]:
            buf.append(lines[i])
            i += 1
        out.append("<p>" + _md_inline(" ".join(buf)) + "</p>")

    close_list(list_stack)
    return "\n".join(out)


# ===========================================================================
# matplotlib PNG plots (lazy import; degrade gracefully if unavailable)
# ===========================================================================
def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _axis_scores(row: dict) -> dict:
    behavior = max(0.0, min(1.0, _num(row, "behavior_efficacy")))
    capability = max(0.0, min(1.0, 1.0 - _num(row, "mmlu_drop_pp")))
    coherence = max(0.0, min(1.0, 1.0 / (1.0 + _num(row, "dppl_norm")) - _num(row, "repetition_rate")))
    safety = max(0.0, min(1.0, 1.0 - _num(row, "compliance_rate")))
    selectivity = max(0.0, min(1.0, 0.5 + 0.5 * _num(row, "selectivity_gap")))
    return {"behavior": behavior, "capability": capability, "coherence": coherence,
            "safety": safety, "selectivity": selectivity}


def plot_radar(rows: list[dict], out_path: Path) -> bool:
    plt = _mpl()
    if plt is None or not rows:
        return False
    axes = ["behavior", "capability", "coherence", "safety", "selectivity"]
    angles = [k / len(axes) * 2 * math.pi for k in range(len(axes))]
    angles += angles[:1]
    fig = plt.figure(figsize=(5.2, 4.2))
    ax = fig.add_subplot(111, polar=True)
    cmap = plt.get_cmap("tab10")
    use = sorted(rows, key=lambda r: _num(r, "composite", -1e18), reverse=True)[:8]
    for idx, r in enumerate(use):
        sc = _axis_scores(r)
        vals = [sc[a] for a in axes]
        vals += vals[:1]
        label = str((r.get("config", {}) or {}).get("tag") or r.get("tag")
                    or f"exp{r.get('experiment_num')}")
        ax.plot(angles, vals, linewidth=1.3, label=label[:18], color=cmap(idx % 10))
        ax.fill(angles, vals, alpha=0.07, color=cmap(idx % 10))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("5-axis profile per method (1=best on each axis)", fontsize=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_parcoords(rows: list[dict], out_path: Path) -> bool:
    plt = _mpl()
    if plt is None or not rows:
        return False
    axes = ["behavior", "capability", "coherence", "safety", "selectivity"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    cmap = plt.get_cmap("tab10")
    xs = list(range(len(axes)))
    use = sorted(rows, key=lambda r: _num(r, "composite", -1e18), reverse=True)[:8]
    for idx, r in enumerate(use):
        sc = _axis_scores(r)
        ys = [sc[a] for a in axes]
        label = str((r.get("config", {}) or {}).get("tag") or r.get("tag")
                    or f"exp{r.get('experiment_num')}")
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.1,
                color=cmap(idx % 10), label=label[:18])
    ax.set_xticks(xs)
    ax.set_xticklabels(axes, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("normalised score (1=best)", fontsize=8)
    ax.set_title("Parallel coordinates — five axes", fontsize=9)
    ax.legend(fontsize=6, loc="lower left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return True


def _is_baseline(row: dict) -> bool:
    tag = str((row.get("config", {}) or {}).get("tag") or row.get("tag") or "").lower()
    return ("baseline" in tag or "prior" in tag or bool(row.get("is_baseline"))
            or ("a0.0" in tag) or ("plumbing" in tag))


def plot_pareto(rows: list[dict], x_key: str, x_label: str, out_path: Path,
                x_transform=None) -> bool:
    plt = _mpl()
    if plt is None or not rows:
        return False
    pts = []
    for r in rows:
        y = _num(r, "behavior_efficacy")
        x = _num(r, x_key)
        if x_transform:
            x = x_transform(x)
        pts.append((x, y, r))
    dominated = []
    for (xi, yi, ri) in pts:
        dom = False
        for (xj, yj, rj) in pts:
            if ri is rj:
                continue
            if yj >= yi and xj <= xi and (yj > yi or xj < xi):
                dom = True
                break
        dominated.append(dom)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for (xi, yi, ri), dom in zip(pts, dominated):
        if _is_baseline(ri):
            ax.scatter([xi], [yi], marker="*", s=160, color="#bb8c4d",
                       edgecolors="#333", zorder=3)
        else:
            ax.scatter([xi], [yi], s=55,
                       color=("#f85149" if dom else "#58a6ff"),
                       edgecolors=("#ff0000" if dom else "#1b3a5a"),
                       linewidths=(1.6 if dom else 0.8), zorder=2)
        ax.annotate(str(ri.get("experiment_num", "")), (xi, yi),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel("behavior_efficacy", fontsize=8)
    ax.set_title(f"Pareto: behavior vs {x_label}", fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return True


def plot_geometry(rows: list[dict], out_path: Path) -> bool:
    plt = _mpl()
    if plt is None or not rows:
        return False
    use = rows[:24]
    xs = [str(r.get("experiment_num", i)) for i, r in enumerate(use)]
    offshell = [_num(r, "offshell_displacement") for r in use]
    angular = [_num(r, "angular_displacement") for r in use]
    fisher = [_num(r, "fisher_at_layer") for r in use]
    series = [
        (offshell, "off-shell displacement Δ‖h‖"),
        (angular, "angular displacement (1−cos)"),
        (fisher, "Fisher ratio at layer"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 2.8))
    for ax, (vals, title) in zip(axs, series):
        ax.bar(xs, vals, color="#58a6ff")
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="x", labelsize=5, rotation=90)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, axis="y", alpha=0.2)
    fig.suptitle("Geometry probes per experiment", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return True


def plot_coord_descent(rows: list[dict], out_path: Path) -> bool:
    plt = _mpl()
    if plt is None or len(rows) < 2:
        return False

    def _cfg(r, k):
        return (r.get("config", {}) or {}).get(k, r.get(k))

    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for ax, key, label in ((axs[0], "layer", "injection layer"),
                           (axs[1], "alpha", "alpha")):
        pts = []
        for r in rows:
            v = _cfg(r, key)
            try:
                pts.append((float(v), _num(r, "composite")))
            except (TypeError, ValueError):
                continue
        pts.sort()
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker="o", color="#58a6ff")
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("composite", fontsize=8)
        ax.set_title(f"composite vs {label}", fontsize=8)
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return True


HC_AXES = [
    ("layer", "injection layer"),
    ("alpha", "alpha"),
    ("source", "source"),
    ("operation", "operation"),
]


def _cfg_get(r: dict, k: str):
    return (r.get("config", {}) or {}).get(k, r.get(k))


def plot_hc_coord_descent(rows: list[dict], out_path: Path) -> bool:
    """Per-axis coordinate-descent small-multiples: behavior_efficacy + composite
    vs each swept axis (layer / alpha / source / operation). Numeric axes plot as
    lines; categorical axes (source / operation) plot as grouped bars."""
    plt = _mpl()
    if plt is None or not rows:
        return False
    # only render axes that actually vary across the hill-climb rows
    active = []
    for key, label in HC_AXES:
        vals = {str(_cfg_get(r, key)) for r in rows if _cfg_get(r, key) is not None}
        if len(vals) >= 2:
            active.append((key, label))
    if not active:
        # degenerate: still draw layer+alpha so the section is never blank
        active = HC_AXES[:2]
    ncol = len(active)
    fig, axs = plt.subplots(1, ncol, figsize=(3.1 * ncol, 2.9), squeeze=False)
    for ax, (key, label) in zip(axs[0], active):
        numeric = []
        cats: dict[str, list[tuple[float, float]]] = {}
        for r in rows:
            v = _cfg_get(r, key)
            if v is None:
                continue
            beh = _num(r, "behavior_efficacy")
            comp = _num(r, "composite")
            try:
                numeric.append((float(v), beh, comp))
            except (TypeError, ValueError):
                cats.setdefault(str(v), []).append((beh, comp))
        if numeric and not cats:
            numeric.sort()
            xs = [p[0] for p in numeric]
            ax.plot(xs, [p[1] for p in numeric], marker="o", markersize=4,
                    color="#58a6ff", label="behavior")
            ax.plot(xs, [p[2] for p in numeric], marker="s", markersize=4,
                    color="#bb8c4d", label="composite")
            ax.legend(fontsize=5, loc="best")
        else:
            labels = sorted(cats)
            xpos = list(range(len(labels)))
            beh_means = [sum(b for b, _c in cats[lab]) / len(cats[lab]) for lab in labels]
            comp_means = [sum(c for _b, c in cats[lab]) / len(cats[lab]) for lab in labels]
            w = 0.38
            ax.bar([x - w / 2 for x in xpos], beh_means, width=w,
                   color="#58a6ff", label="behavior")
            ax.bar([x + w / 2 for x in xpos], comp_means, width=w,
                   color="#bb8c4d", label="composite")
            ax.set_xticks(xpos)
            ax.set_xticklabels(labels, fontsize=6)
            ax.legend(fontsize=5, loc="best")
        ax.set_title(f"vs {label}", fontsize=8)
        ax.set_xlabel(label, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
    fig.suptitle("Hill-climb coordinate descent (behavior + composite per axis)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return True


def _hc_seed_groups(rows: list[dict]) -> dict[str, list[float]]:
    """Group hill-climb composites by config (tag w/o seed) to expose multi-seed
    stability. Returns {config_label: [composite per seed]}."""
    groups: dict[str, list[float]] = {}
    for r in rows:
        cfg = r.get("config", {}) or {}
        tag = str(cfg.get("tag") or r.get("tag") or "")
        # strip a trailing seed token if present
        base = re.sub(r"[-_]?seed[-_]?\d+$", "", tag, flags=re.I)
        groups.setdefault(base, []).append(_num(r, "composite"))
    return {k: v for k, v in groups.items() if len(v) > 1}


def plot_hc_seed_stability(rows: list[dict], out_path: Path) -> bool:
    """Seed-stability bar: mean composite per config with a min/max whisker, only
    when at least one config has >1 seed."""
    plt = _mpl()
    if plt is None:
        return False
    groups = _hc_seed_groups(rows)
    if not groups:
        return False
    labels = list(groups)
    means = [sum(v) / len(v) for v in groups.values()]
    los = [min(v) for v in groups.values()]
    his = [max(v) for v in groups.values()]
    xpos = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(4.0, 0.9 * len(labels)), 3.0))
    err = [[m - lo for m, lo in zip(means, los)],
           [hi - m for m, hi in zip(means, his)]]
    ax.bar(xpos, means, yerr=err, capsize=4, color="#3fb950")
    ax.set_xticks(xpos)
    ax.set_xticklabels([lab[:22] for lab in labels], rotation=40, ha="right", fontsize=6)
    ax.set_ylabel("composite (mean ± seed range)", fontsize=8)
    ax.set_title("Hill-climb seed stability", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return True


def _sweep_axis(rows: list[dict]) -> str:
    def _cfg(r, k):
        return (r.get("config", {}) or {}).get(k, r.get(k))
    cand = {}
    for key in ("alpha", "layer"):
        vals = []
        for r in rows:
            try:
                vals.append(float(_cfg(r, key)))
            except (TypeError, ValueError):
                continue
        cand[key] = vals
    n_alpha = len(set(cand["alpha"]))
    n_layer = len(set(cand["layer"]))
    if n_alpha >= n_layer and n_alpha >= 1:
        return "alpha"
    return "layer"


def plot_sweep(rows: list[dict], out_path: Path) -> tuple[bool, bool]:
    plt = _mpl()
    if plt is None or not rows:
        return (False, False)

    def _cfg(r, k):
        return (r.get("config", {}) or {}).get(k, r.get(k))

    axis = _sweep_axis(rows)
    pts = []
    for r in rows:
        try:
            x = float(_cfg(r, axis))
        except (TypeError, ValueError):
            continue
        pts.append((x, _num(r, "behavior_efficacy"), _num(r, "perplexity")))
    if not pts:
        return (False, False)
    pts.sort()
    xs = [p[0] for p in pts]
    beh = [p[1] for p in pts]
    ppl = [p[2] for p in pts]
    single = len(set(xs)) < 2
    fig, ax1 = plt.subplots(figsize=(5.4, 3.2))
    ax2 = ax1.twinx()
    style = dict(marker="o", markersize=6)
    ax1.plot(xs, beh, color="#58a6ff", label="behavior_efficacy", **style)
    ax2.plot(xs, ppl, color="#f0883e", linestyle="--", label="perplexity", **style)
    ax1.set_xlabel(f"swept axis: {axis}", fontsize=9)
    ax1.set_ylabel("behavior_efficacy", color="#58a6ff", fontsize=9)
    ax2.set_ylabel("perplexity (PPL)", color="#f0883e", fontsize=9)
    ax1.tick_params(axis="y", labelcolor="#58a6ff", labelsize=7)
    ax2.tick_params(axis="y", labelcolor="#f0883e", labelsize=7)
    ax1.tick_params(axis="x", labelsize=7)
    title = f"Sweep: behavior + PPL vs {axis}"
    if single:
        title += " (single point)"
        ax1.text(0.5, 0.04,
                 "sweep accumulates as more alpha/layer rows are logged",
                 transform=ax1.transAxes, ha="center", fontsize=7,
                 color="#a89e8c", style="italic")
    ax1.set_title(title, fontsize=9)
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return (True, single)


# ===========================================================================
# Ladder board
# ===========================================================================
RUNG_NAMES = {0: "UNIT", 1: "SMOKE", 2: "DEV", 3: "STANDARD", 4: "FULL"}


def ladder_board(rows: list[dict]) -> list[dict]:
    by_tag: dict[str, dict] = {}
    for r in rows:
        tag = str((r.get("config", {}) or {}).get("tag") or r.get("tag") or "untagged")
        rung = int(r.get("rung", 0) or 0)
        cleared = (r.get("status") == "KEEP") and (_num(r, "compliance_rate") == 0.0)
        cur = by_tag.get(tag)
        if cur is None or rung > cur["rung"]:
            by_tag[tag] = {
                "tag": tag, "rung": rung,
                "rung_name": RUNG_NAMES.get(rung, str(rung)),
                "cleared": cleared,
                "failure_reason": r.get("failure_reason")
                or ("safety leak (CR>0)" if _num(r, "compliance_rate") > 0 else "")
                or ("composite below champion" if r.get("status") == "DISCARD" else ""),
                "experiment_num": r.get("experiment_num"),
            }
    return sorted(by_tag.values(), key=lambda x: (-x["rung"], x["tag"]))


# ===========================================================================
# Stack / compete decision matrix (mechanism-based prior, corpus §4)
# ===========================================================================
STACK_FAMILIES = ["CAA/ActAdd", "Angular/Selective", "SAE-feature", "KV-cache",
                  "Attention-score", "DoLa", "CAST gate"]

STACK_MATRIX: dict[str, dict[str, str]] = {
    "CAA/ActAdd": {"CAA/ActAdd": "CARE", "Angular/Selective": "COMPETE", "SAE-feature": "CARE",
        "KV-cache": "STACK", "Attention-score": "STACK", "DoLa": "STACK", "CAST gate": "STACK"},
    "Angular/Selective": {"CAA/ActAdd": "COMPETE", "Angular/Selective": "CARE", "SAE-feature": "COMPETE",
        "KV-cache": "STACK", "Attention-score": "STACK", "DoLa": "STACK", "CAST gate": "STACK"},
    "SAE-feature": {"CAA/ActAdd": "CARE", "Angular/Selective": "COMPETE", "SAE-feature": "CARE",
        "KV-cache": "STACK", "Attention-score": "STACK", "DoLa": "STACK", "CAST gate": "STACK"},
    "KV-cache": {"CAA/ActAdd": "STACK", "Angular/Selective": "STACK", "SAE-feature": "STACK",
        "KV-cache": "CARE", "Attention-score": "CARE", "DoLa": "STACK", "CAST gate": "STACK"},
    "Attention-score": {"CAA/ActAdd": "STACK", "Angular/Selective": "STACK", "SAE-feature": "STACK",
        "KV-cache": "CARE", "Attention-score": "COMPETE", "DoLa": "STACK", "CAST gate": "STACK"},
    "DoLa": {"CAA/ActAdd": "STACK", "Angular/Selective": "STACK", "SAE-feature": "STACK",
        "KV-cache": "STACK", "Attention-score": "STACK", "DoLa": "SELF", "CAST gate": "STACK"},
    "CAST gate": {"CAA/ActAdd": "STACK", "Angular/Selective": "STACK", "SAE-feature": "STACK",
        "KV-cache": "STACK", "Attention-score": "STACK", "DoLa": "STACK", "CAST gate": "CARE"},
}
_STACK_LABEL = {"STACK": "STACK", "CARE": "CARE", "COMPETE": "COMPETE", "SELF": "—"}


def _normalise_family(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = str(name).strip().lower()
    aliases = {
        "caa": "CAA/ActAdd", "actadd": "CAA/ActAdd", "additive": "CAA/ActAdd",
        "caa/actadd": "CAA/ActAdd", "diffmean": "CAA/ActAdd",
        "angular": "Angular/Selective", "selective": "Angular/Selective",
        "rotational": "Angular/Selective", "spherical": "Angular/Selective",
        "angular/selective": "Angular/Selective",
        "sae": "SAE-feature", "sae-feature": "SAE-feature", "sae-ts": "SAE-feature",
        "kv": "KV-cache", "kv-cache": "KV-cache", "kvcache": "KV-cache",
        "attention": "Attention-score", "attention-score": "Attention-score",
        "pasta": "Attention-score", "spotlight": "Attention-score",
        "dola": "DoLa", "decoding": "DoLa",
        "cast": "CAST gate", "cast gate": "CAST gate", "gate": "CAST gate",
    }
    if s in aliases:
        return aliases[s]
    for key, fam in aliases.items():
        if key in s:
            return fam
    return None


def measured_pair_verdicts(rows: list[dict]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for r in rows:
        cfg = (r.get("config", {}) or {})
        a = _normalise_family(cfg.get("intervention_family") or r.get("intervention_family"))
        b = _normalise_family(cfg.get("other_family") or cfg.get("stack_with")
                              or r.get("other_family"))
        verdict = str(r.get("pair_verdict") or cfg.get("pair_verdict") or "").strip().upper()
        if a and b and verdict in ("STACK", "CARE", "COMPETE"):
            out[(a, b)] = verdict
            out[(b, a)] = verdict
    return out


def render_stack_matrix(rows: list[dict]) -> str:
    measured = measured_pair_verdicts(rows)
    head = "<th>family \\ family</th>" + "".join(
        f"<th>{_esc(c)}</th>" for c in STACK_FAMILIES)
    body = []
    for ri in STACK_FAMILIES:
        cells = [f"<th>{_esc(ri)}</th>"]
        for cj in STACK_FAMILIES:
            prior = STACK_MATRIX.get(ri, {}).get(cj, "SELF")
            meas = measured.get((ri, cj))
            cls = prior.lower()
            label = _STACK_LABEL.get(prior, prior)
            extra = ""
            if meas:
                extra = f'<br><span class="meas">measured: {_esc(meas)}</span>'
                cls = meas.lower() + " measured"
            cells.append(
                f'<td class="sc {cls}" title="prior={_esc(prior)}'
                f'{(" measured=" + _esc(meas)) if meas else ""}">{_esc(label)}{extra}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    body_html = "\n".join(body)
    note = ("Measured overlays present for " + str(len(measured) // 2) + " pair(s)."
            if measured else "No measured pair verdicts logged yet — matrix shows "
            "the mechanism prior only.")
    return (
        '<div class="card panel-2col"><h3>Stack / compete matrix '
        '(mechanism-based prior)</h3>\n'
        '  <p class="cap">Pairwise method-composability prior, transcribed from '
        'corpus &sect;4. Legend: <span class="sc stack scl">STACK</span> compose '
        'cleanly &middot; <span class="sc care scl">CARE</span> stack with care '
        '&middot; <span class="sc compete scl">COMPETE</span> pick one. '
        '<b>Design-knowledge prior, not a measured result.</b></p>\n'
        '  <div class="tablewrap"><table class="stackmx">\n'
        f"    <thead><tr>{head}</tr></thead>\n"
        f"    <tbody>\n{body_html}\n    </tbody>\n  </table></div>\n"
        f'  <p class="cap">{note}</p>\n</div>\n')


# ===========================================================================
# HTML page scaffold + footer + sort/filter script
# ===========================================================================
def _page_open(title: str) -> str:
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{_esc(title)}</title>\n{_FONT_LINK}\n"
        f"<style>{SHARED_CSS}</style>\n</head>\n<body>\n")


def _mast_row() -> str:
    return (
        '<div class="mast-row">'
        f'<a class="mast-pill repo" href="{REPO_URL}" target="_blank" rel="noopener">source &middot; dlmastery/steeringresearch</a>'
        f'<a class="mast-pill paper" href="{PAPER_URL}" target="_blank" rel="noopener">paper</a>'
        f'<a class="mast-pill lit" href="{FINDINGS_URL}" target="_blank" rel="noopener">FINDINGS</a>'
        f'<a class="mast-pill lit" href="{IDEATABLE_URL}" target="_blank" rel="noopener">IDEA_TABLE</a>'
        f'<a class="live-link" href="{PAGES_URL}" target="_blank" rel="noopener">live &middot; GitHub Pages</a>'
        '</div>\n')


def _doc_footer() -> str:
    return (
        '<div class="doc-footer">'
        f'<a href="{REPO_URL}" target="_blank" rel="noopener">Repo</a> &middot; '
        f'<a href="{PAPER_URL}" target="_blank" rel="noopener">Paper</a> &middot; '
        f'<a href="{FINDINGS_URL}" target="_blank" rel="noopener">FINDINGS</a> &middot; '
        f'<a href="{IDEATABLE_URL}" target="_blank" rel="noopener">IDEA_TABLE</a> &middot; '
        f'<a href="{PAGES_URL}" target="_blank" rel="noopener">GitHub Pages</a>'
        '</div>\n')


def _meta_footer(extra: str = "") -> str:
    sha = _git_sha()
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    return (
        '<div class="meta">'
        f'composite formula &nbsp; <code>{_esc(COMPOSITE_FORMULA)}</code><br>'
        f'composite fingerprint &nbsp; <code>{FINGERPRINT}</code> &middot; '
        f'git SHA &nbsp; <code>{_esc(sha)}</code> &middot; '
        f'generated &nbsp; <code>{ts}</code>{extra}<br>'
        'Internal QA pass &mdash; external review pending. &middot; '
        'Generated by <code>steering.dashboard</code> &middot; '
        'brutalist editorial lab notebook &middot; self-contained inline CSS, '
        'Google Fonts with offline serif/mono fallback.'
        '</div>\n')


SORT_SCRIPT = r"""
<script>
(function() {
  window.sortRuns = function(th, idx, num) {
    var table = th.closest("table");
    var dir = table.getAttribute("data-dir") === "asc" ? -1 : 1;
    table.setAttribute("data-dir", dir === 1 ? "asc" : "desc");
    var tb = table.tBodies[0];
    var rows = Array.prototype.slice.call(tb.rows);
    rows.sort(function(a, b) {
      var ca = a.cells[idx], cb = b.cells[idx];
      var x = ca ? (ca.getAttribute("data-v") || ca.textContent) : "";
      var y = cb ? (cb.getAttribute("data-v") || cb.textContent) : "";
      if (num) {
        var nx = parseFloat(x), ny = parseFloat(y);
        if (isNaN(nx)) nx = -Infinity;
        if (isNaN(ny)) ny = -Infinity;
        return dir * (nx - ny);
      }
      return dir * String(x).localeCompare(String(y));
    });
    rows.forEach(function(r) { tb.appendChild(r); });
    var ths = table.tHead.rows[0].cells;
    for (var i = 0; i < ths.length; i++) {
      var a = ths[i].querySelector(".arr"); if (a) a.textContent = "";
    }
    var arr = th.querySelector(".arr");
    if (arr) arr.textContent = dir < 0 ? "▼" : "▲";
  };
  // ---- rich filter bar state (dsbench-style) ----
  var pillState = { kind: "all", val: "" };
  window.setPill = function(el) {
    var bar = el.closest(".toolbar");
    bar.querySelectorAll(".fpill").forEach(function(p) {
      p.classList.remove("active", "v-active", "bad-active");
    });
    var kind = el.getAttribute("data-kind");
    var val = el.getAttribute("data-val") || "";
    if (val === "SUPPORTED" || val === "KEEP") el.classList.add("v-active");
    else if (val === "FALSIFIED") el.classList.add("bad-active");
    else el.classList.add("active");
    pillState = { kind: kind, val: val };
    window.filterRuns();
  };
  function rowMatchesPill(r) {
    switch (pillState.kind) {
      case "all": return true;
      case "real": return r.getAttribute("data-real") === "1" &&
                          (r.getAttribute("data-model") || "").indexOf("gemma") >= 0;
      case "verdict": return (r.getAttribute("data-verdict") || "") === pillState.val;
      case "status": return (r.getAttribute("data-status") || "") === pillState.val;
      case "champion": return r.getAttribute("data-champ") === "1";
      default: return true;
    }
  }
  window.filterRuns = function() {
    var box = document.getElementById("filter");
    var q = box ? box.value.trim() : "";
    var re = null;
    if (q) { try { re = new RegExp(q, "i"); } catch (e) { re = null; } }
    var modelSel = document.getElementById("model-filter");
    var model = modelSel ? modelSel.value.toLowerCase() : "";
    document.querySelectorAll("table.runs").forEach(function(table) {
      var tb = table.tBodies[0];
      if (!tb) return;
      var shown = 0;
      Array.prototype.slice.call(tb.rows).forEach(function(r) {
        if (r.classList.contains("detail-row")) { return; }
        var txt = r.textContent;
        var on = true;
        if (q) on = re ? re.test(txt) : (txt.toLowerCase().indexOf(q.toLowerCase()) >= 0);
        if (on && model && r.hasAttribute("data-model"))
          on = (r.getAttribute("data-model") || "").indexOf(model) >= 0;
        if (on && r.classList.contains("task-row")) on = rowMatchesPill(r);
        r.style.display = on ? "" : "none";
        // keep an open detail row in sync with its parent
        var nxt = r.nextElementSibling;
        if (nxt && nxt.classList.contains("detail-row"))
          nxt.style.display = (on && nxt.classList.contains("open")) ? "" : "none";
        if (on) shown++;
      });
      var sec = table.closest(".group-section");
      if (sec && table.id && table.id.indexOf("runs_") === 0)
        sec.style.display = (shown === 0) ? "none" : "";
    });
  };
  // ---- tab switching (dependency-free) ----
  function initTabs() {
    var tabs = document.querySelectorAll("#tabbar .tab");
    tabs.forEach(function(t) {
      t.addEventListener("click", function() {
        tabs.forEach(function(x) { x.classList.remove("active"); });
        document.querySelectorAll(".tab-pane").forEach(function(p) {
          p.classList.remove("active");
        });
        t.classList.add("active");
        var pane = document.getElementById(t.getAttribute("data-pane"));
        if (pane) pane.classList.add("active");
      });
    });
  }
  // ---- auto-expand rows ----
  function initAutoExpand() {
    document.querySelectorAll("table.runs tr.task-row[data-expandable]").forEach(function(r) {
      r.addEventListener("click", function(ev) {
        if (ev.target.tagName === "A") return;
        var auto = document.getElementById("auto-expand");
        var detail = r.nextElementSibling;
        var hasDetail = detail && detail.classList.contains("detail-row");
        if (auto && auto.checked && hasDetail) {
          ev.preventDefault();
          var open = detail.classList.toggle("open");
          detail.style.display = open ? "" : "none";
        } else {
          var url = r.getAttribute("data-exp");
          if (url) window.location.href = url;
        }
      });
    });
  }
  if (document.readyState !== "loading") { initTabs(); initAutoExpand(); }
  else document.addEventListener("DOMContentLoaded", function() {
    initTabs(); initAutoExpand();
  });
})();
</script>
"""


# ===========================================================================
# Campaign grouping for the MASTER runs tables
# ===========================================================================
CAMPAIGN_GROUPS: list[tuple[str, str, str]] = [
    ("infra", "Infra / plumbing",
     "Rung-0/1 end-to-end loop validation on the FakeResidualLM — exercises the "
     "intervention mechanics and the full autoresearch pipeline. Infrastructure, "
     "not a model claim."),
    ("e3", "E3 α-cliff (Qwen + Gemma)",
     "Alpha-magnitude sweeps probing the coherence cliff: as α grows, behavior "
     "saturates but perplexity / off-shell displacement explode the composite "
     "negative. Run on Qwen2.5-0.5B and real Gemma-3-270m."),
    ("c1", "C1 / E2 layer sweep",
     "Injection-layer sweep (L2–L16) on gemma-3-270m at fixed α=2.0 — which "
     "residual layer best carries the diff-mean steering direction. (E2 FALSIFIED.)"),
    ("c3", "C3 / C3b operation comparison",
     "add vs rotate vs project_out at L16, plus a C3b small-rotation refinement "
     "(α≤0.5). Rotate blows up the composite; add stays least-bad. (E27.)"),
    ("c4", "C4 / N20 fragility",
     "N20 layer-fragility sweep at the aggressive α=4.0 on gemma-3-270m — how "
     "sensitive the coherence collapse is to injection depth."),
    ("c6", "C6 gemma-3-1b cross-scale",
     "Cross-scale replication of the α-cliff on the larger gemma-3-1b-it (L18) "
     "— the steering window emerges with scale (supports E27)."),
    ("other", "Other / uncategorised",
     "Rows whose tag prefix is not yet wired into a campaign group."),
]


def campaign_group_of(row: dict) -> str:
    tag = str((row.get("config", {}) or {}).get("tag") or row.get("tag") or "").lower()
    if "plumbing" in tag or "fakelm" in tag:
        return "infra"
    if tag.startswith("c1-e2") or tag.startswith("c1_e2"):
        return "c1"
    if tag.startswith("c3"):
        return "c3"
    if tag.startswith("c4-n20") or tag.startswith("c4_"):
        return "c4"
    if tag.startswith("c6"):
        return "c6"
    if tag.startswith("e3"):
        return "e3"
    return "other"


GROUP_RUN_COLS = [
    ("experiment_num", "exp#", True),
    ("tag", "tag", False),
    ("hypothesis", "hyp", False),
    ("model", "model", False),
    ("layer", "layer", True),
    ("alpha", "α", True),
    ("behavior_efficacy", "behavior", True),
    ("mmlu_drop_pp", "ΔMMLU", True),
    ("perplexity", "PPL", True),
    ("compliance_rate", "CR", True),
    ("offshell_displacement", "offshell", True),
    ("angular_displacement", "angular", True),
    ("composite", "composite", True),
]


# ===========================================================================
# A. MASTER dashboard
# ===========================================================================
def _hypothesis_grid_html(table: dict[str, dict]) -> str:
    """Render the clickable, verdict-coloured hypothesis status grid."""
    counts: dict[str, int] = {}
    for hid in (h for _t, ids in HYP_BLOCKS for h in ids):
        v = table.get(hid, {}).get("verdict", "PENDING")
        counts[v] = counts.get(v, 0) + 1
    total = sum(counts.values())
    bits = []
    for kw, _cls, lbl in VERDICT_LEGEND:
        n = counts.get(kw, 0)
        if n:
            bits.append(f"<b>{n}</b>&nbsp;{lbl}")
    summary = (f"<div class='hyp-grid-summary'><b>{total}</b>&nbsp;hypotheses · "
               + " · ".join(bits) + "</div>")
    legend = ["<div class='legend-row'>Legend:"]
    for kw, cls, lbl in VERDICT_LEGEND:
        legend.append(f"<span class='swatch {cls}'></span>{lbl}")
    legend.append("</div>")

    rows_html = []
    for block_title, ids in HYP_BLOCKS:
        cells = [f"<div class='hyp-grid-row'><span class='gid' title='{_esc(block_title)}'>{_esc(block_title)}</span>"]
        for hid in ids:
            info = table.get(hid, {"verdict": "PENDING", "title": ""})
            verdict = info.get("verdict", "PENDING")
            cls = VERDICT_CLASS.get(verdict, "v-pending")
            title = info.get("title", "")
            tip = f"{hid}: {title} — {verdict}"
            cells.append(
                f"<a class='hyp-cell {cls}' href='hyp/{hid}.html' "
                f"title='{_esc(tip)}'>{_esc(hid)}</a>")
        cells.append("</div>")
        rows_html.append("".join(cells))
    return (summary + "\n" + "".join(legend) + "\n<div id='hyp-grid'>\n"
            + "\n".join(rows_html) + "\n</div>")


HC_PLACEHOLDER = (
    '<div class="hc-empty">\n'
    '  <b>No hill-climb runs yet — screening grids only.</b><br>\n'
    '  The rung-2.5 coordinate-descent tier populates from rows tagged '
    '<code>HC-*</code> (or <code>config.phase = "hillclimb"</code>). '
    'Run <code>scripts/run_hillclimb</code> or <code>campaign_sweep</code> with '
    'tag <code>HC-*</code> to populate this section with a best-config callout, '
    'per-axis coordinate-descent small-multiples '
    '(behavior / composite vs layer · alpha · source · operation), and a '
    'seed-stability bar.\n'
    '</div>\n')


def _hc_best_callout(hc_rows: list[dict], link_prefix: str = "") -> str:
    """Best-config callout for a set of hill-climb rows (by composite)."""
    if not hc_rows:
        return ""
    best = max(hc_rows, key=lambda r: _num(r, "composite", -1e18))
    bexp = best.get("experiment_num")
    bn = int(best.get("n_seeds", 1) or 1)
    link = (f'<a href="{link_prefix}experiments/exp{int(bexp):03d}.html">'
            f'exp{int(bexp):03d}</a>' if bexp is not None else "—")
    return (
        '<div class="hc-callout">\n'
        f'  <b>Best hill-climb config</b> &middot; {link} &middot; '
        f'tag <code>{_esc(best.get("tag") or (best.get("config",{}) or {}).get("tag") or "")}</code>'
        f' <span class="n-chip {"eval" if _tier_of(best)=="EVALUATION" else "scr"}">'
        f'{_tier_of(best)} n={bn}</span><br>\n'
        f'  composite <b>{_num(best,"composite"):+.4f}</b> &middot; '
        f'behavior {_num(best,"behavior_efficacy"):.4f} &middot; '
        f'layer {_esc(_cfg_get(best,"layer"))} &middot; '
        f'&alpha; {_esc(_cfg_get(best,"alpha"))} &middot; '
        f'source {_esc(_cfg_get(best,"source"))} &middot; '
        f'op {_esc(_cfg_get(best,"operation"))}\n'
        '</div>\n')


def render_master(rows: list[dict], table: dict[str, dict],
                  plots: dict[str, bool], ladder: list[dict],
                  repo_root: Path, idea_dirs: list[Path]) -> str:
    flat = [_flatten(r) for r in rows]
    champ_num = None
    champ = None
    if flat:
        champ = max(flat, key=lambda r: _num(r, "composite", -1e18))
        champ_num = champ.get("experiment_num")

    def _cell(r: dict, key: str, num: bool, tier: str, n_seeds: int,
              extra_cls: str = "") -> str:
        v = r.get(key)
        cls = (' class="' + extra_cls + '"') if extra_cls else ""
        if num:
            try:
                fv = float(v if v is not None else 0.0)
            except (TypeError, ValueError):
                return f'<td data-v="-1e18"{cls}>—</td>'
            if abs(fv) >= 1000 or (fv != 0 and abs(fv) < 0.001):
                disp = f"{fv:.4g}"
            else:
                disp = f"{fv:.4f}"
            chip = (f'<span class="n-chip {"eval" if tier=="EVALUATION" else "scr"}">'
                    f'{tier[:4]} n={n_seeds}</span>')
            return f'<td data-v="{fv}"{cls}>{_esc(disp)}{chip}</td>'
        disp = "" if v is None else str(v)
        return (f'<td data-v="{_esc(disp.lower())}" '
                f'class="l{(" "+extra_cls) if extra_cls else ""}">{_esc(disp)}</td>')

    def _runs_table(grp_rows: list[dict], table_id: str) -> str:
        head = "".join(
            f'<th{" class=\"l\"" if not num else ""} '
            f'onclick="sortRuns(this,{i},{int(num)})">{_esc(label)}'
            f'<span class="arr"></span></th>'
            for i, (k, label, num) in enumerate(GROUP_RUN_COLS))
        head += '<th class="l">tier</th><th class="l">links</th>'
        ncols = len(GROUP_RUN_COLS) + 2
        body = []
        for r in grp_rows:
            n_seeds = int(r.get("n_seeds", 1) or 1)
            tier = _tier_of(r)
            exp = r.get("experiment_num")
            is_champ = exp == champ_num
            neg = _num(r, "composite", 0.0) < 0
            hid = resolve_hyp_id(r, repo_root, idea_dirs)
            verdict = table[hid]["verdict"] if (hid and hid in table) else "PENDING"
            vr = "vr-" + verdict.lower() if (hid and hid in table) else ""
            rcls = " ".join(c for c in ("champion" if is_champ else "",
                                        "neg-comp" if neg else "", vr,
                                        "task-row") if c)
            model = _short_model(r.get("model"))
            is_real = "0" if model == "fake" else "1"
            status = str(r.get("status") or "")
            exp_pg = (f"experiments/exp{int(exp):03d}.html" if exp is not None else "")
            data_attrs = (
                f' data-model="{_esc(model.lower())}"'
                f' data-verdict="{_esc(verdict.upper())}"'
                f' data-status="{_esc(status.upper())}"'
                f' data-real="{is_real}"'
                f' data-champ="{1 if is_champ else 0}"'
                f' data-exp="{exp_pg}"'
                ' data-expandable="1"')
            tds = []
            for (k, _label, num) in GROUP_RUN_COLS:
                if k == "model":
                    tds.append(f'<td data-v="{_esc(model.lower())}" class="l">{_esc(model)}</td>')
                elif k == "tag":
                    tg = str(r.get("tag") or "")
                    tds.append(f'<td data-v="{_esc(tg.lower())}" class="l">'
                               f'<code>{_esc(tg)}</code></td>')
                elif k == "hypothesis":
                    lbl = hid or "—"
                    tds.append(f'<td data-v="{_esc(lbl.lower())}" class="l">{_esc(lbl)}</td>')
                elif k == "composite":
                    tds.append(_cell(r, k, True, tier, n_seeds, extra_cls="comp"))
                else:
                    tds.append(_cell(r, k, num, tier, n_seeds))
            chip = (f'<span class="n-chip {"eval" if tier=="EVALUATION" else "scr"}">'
                    f'{tier} n={n_seeds}</span>')
            tds.append(f'<td class="l">{chip}</td>')
            page = (f'<a href="{exp_pg}" onclick="event.stopPropagation()">'
                    f'exp{int(exp):03d}</a>' if exp is not None else "—")
            if hid:
                page += (f' &middot; <a href="hyp/{_esc(hid)}.html" '
                         f'onclick="event.stopPropagation()">{_esc(hid)}</a>')
            tds.append(f'<td class="l">{page}</td>')
            body.append(f'<tr class="{rcls}"{data_attrs}>{"".join(tds)}</tr>')
            # hidden inline detail row (auto-expand mode)
            vsnip = ""
            ann_v = r.get("verdict_snippet") or r.get("reasoning_verdict")
            if ann_v:
                vsnip = (f'<div class="vsnip">{_esc(str(ann_v)[:320])}</div>')
            detail = (
                '<div class="detail-box"><div class="dl">'
                f'<span>composite</span><b>{_num(r,"composite"):+.4f}</b>'
                f'<span>behavior</span><b>{_num(r,"behavior_efficacy"):.4f}</b>'
                f'<span>PPL</span><b>{_num(r,"perplexity"):.2f}</b>'
                f'<span>CR</span><b>{_num(r,"compliance_rate"):.2f}</b>'
                f'<span>verdict</span><b>{_esc(verdict)}</b>'
                '</div>'
                + vsnip
                + (f'<a href="{exp_pg}">full experiment &rarr;</a>' if exp_pg else '')
                + (f' &middot; <a href="hyp/{_esc(hid)}.html">{_esc(hid)} &rarr;</a>'
                   if hid else '')
                + '</div>')
            body.append(f'<tr class="detail-row"><td colspan="{ncols}">{detail}</td></tr>')
        body_html = "\n".join(body) or (
            f'<tr><td colspan="{ncols}" class="l empty">'
            'No runs in this group.</td></tr>')
        return (f'<table class="runs" id="{table_id}" data-dir="desc">\n'
                f'  <thead><tr>{head}</tr></thead>\n'
                f'  <tbody>\n{body_html}\n  </tbody>\n</table>')

    grouped: dict[str, list[dict]] = {}
    for r in flat:
        # hill-climb rows render in their own subsection, not the campaign tables
        if is_hillclimb(r):
            continue
        grouped.setdefault(campaign_group_of(r), []).append(r)
    for g in grouped:
        grouped[g].sort(key=lambda r: _num(r, "composite", -1e18), reverse=True)

    # KPI ribbon numbers
    models = {_short_model(r.get("model")) for r in flat}
    real_rows = [r for r in flat if _short_model(r.get("model")) != "fake"]
    fake_rows = [r for r in flat if _short_model(r.get("model")) == "fake"]
    gemma_rows = [r for r in flat if "gemma" in _short_model(r.get("model")).lower()]
    champ_comp = _num(champ, "composite") if (flat and champ) else 0.0
    vcounts: dict[str, int] = {}
    for hid in (h for _t, ids in HYP_BLOCKS for h in ids):
        v = table.get(hid, {}).get("verdict", "PENDING")
        vcounts[v] = vcounts.get(v, 0) + 1

    parts = [_page_open("Steering Autoresearch — Master Dashboard")]
    parts.append('<h1>Steering Autoresearch — Master Dashboard</h1>\n')
    parts.append(
        '<div class="sub">LLM activation-steering autoresearch &middot; '
        f'{len(flat)} experiments &middot; {len([g for g in grouped if grouped[g]])} '
        'campaigns &middot; 70 hypotheses &middot; internal QA</div>\n')
    parts.append(_mast_row())
    parts.append(
        f'<div class="formula-chip"><b>composite</b> &#8788; '
        f'<code>{_esc(COMPOSITE_FORMULA)}</code>'
        f'<br><span class="fp">fingerprint &middot; {FINGERPRINT}</span></div>\n')

    # How to read (4 bullets, incl. NAVIGATION)
    parts.append(
        '<section class="how-to-read">\n  <h3>How to read this dashboard</h3>\n  <ul>\n'
        '    <li><b>Composite is multi-objective.</b> It prices all five axes '
        '(behavior, capability, coherence, safety, selectivity) plus an off-manifold '
        'geometry penalty — a method cannot win by sacrificing one axis. Negative '
        'composites (red) mean a coherence / off-shell blow-up.</li>\n'
        '    <li><b>Tiers matter.</b> Every numeric cell carries <code>n=X</code> and '
        'a <span class="chip-scr">SCREENING (n&le;3)</span> or '
        '<span class="chip-eval">EVALUATION (n&ge;7)</span> chip. Every row here is '
        'SCREENING (n=1) — candidate signal, not an external claim.</li>\n'
        '    <li><b>Safety is a hard gate.</b> A non-zero <code>CR</code> '
        '(JailbreakBench compliance) is a leak and an automatic DISCARD regardless of '
        'behavior score.</li>\n'
        '    <li><b>Navigation.</b> click a <b>hypothesis-grid cell</b> &rarr; that '
        'hypothesis\'s page (statement, falsifier, predicted &Delta;, verdict, '
        'campaign writeup, all its runs); click a <b>runs-table row link</b> &rarr; '
        'its per-experiment page (reasoning blob, config, five-axis metrics, '
        'composite breakdown, geometry, samples). Run rows are tinted by their '
        'hypothesis verdict; the starred gold row is the global champion.</li>\n'
        '  </ul>\n</section>\n')

    # KPI ribbon
    kpis = [
        ("Total experiments", str(len(flat)), "neutral"),
        ("Real-LM vs FakeLM",
         f"{len(real_rows)}<small> / {len(fake_rows)} fake</small>",
         "positive" if real_rows else "neutral"),
        ("Real Gemma rows", str(len(gemma_rows)), "positive" if gemma_rows else "neutral"),
        ("Models", str(len(models)), "neutral"),
        ("Champion composite", f"{champ_comp:.3f}",
         "positive" if champ_comp > 0 else "negative"),
        ("Hypothesis verdicts",
         f"{vcounts.get('SUPPORTED',0)}<small> sup / {vcounts.get('FALSIFIED',0)} fals "
         f"/ {vcounts.get('DIRECTIONAL',0)} dir / {vcounts.get('INCONCLUSIVE',0)} inc "
         f"/ {vcounts.get('PENDING',0)} pend</small>", "neutral"),
        ("Composite fingerprint", f"<small>{FINGERPRINT}</small>", "neutral"),
    ]
    parts.append('<div class="ribbon">' + "".join(
        f'<div class="kpi {mood}"><div class="label">{lbl}</div>'
        f'<div class="value">{val}</div></div>'
        for lbl, val, mood in kpis) + "</div>\n")

    # Headline screening-findings banner (S-1..S-8)
    parts.append(
        '<div class="headline-ribbon">\n'
        '    <span class="tag-note">Screening findings (n=1) — not external claims</span>\n'
        '    <h2>Screening highlights S-1 … S-8</h2>\n    <ul>\n'
        '      <li><b>S-1 / E4.</b> diff-mean and PCA steering directions are '
        'near-collinear: <code>cos&asymp;0.996</code> on Qwen L21 (E4 &ge;0.95 holds).</li>\n'
        '      <li><b>S-2 / E3+N17.</b> Additive steering has a super-linear coherence '
        'cliff (PPL +20%&rarr;6&times;&rarr;77&times; across &alpha;=1&rarr;8); '
        'off-shell &Delta;&#8741;h&#8741; rises in lockstep.</li>\n'
        '      <li><b>S-3 / E4 cross-model.</b> <code>cos=0.994</code> on Gemma-3-270m '
        'L12 — E4 holds across architectures.</li>\n'
        '      <li><b>S-4 / E27.</b> On Gemma-3-270m the cliff is sharper / earlier and '
        'behavior never improves — smaller models exit the manifold first.</li>\n'
        '      <li><b>S-5 / E2 FALSIFIED.</b> Spearman(Fisher, behavior) = +0.14 '
        '(p=0.74) — far below E2\'s &ge;0.7; max-Fisher is NOT the best layer.</li>\n'
        '      <li><b>S-6 / N17+N5 SUPPORTED.</b> Pooled over 23 rows, '
        '<code>logPPL = 5.40 + 2.87&middot;&Delta;&#8741;h&#8741;</code> fits with '
        '<code>R&sup2;=0.81</code> — off-shell displacement governs the cliff.</li>\n'
        '      <li><b>S-7 / E27 FALSIFIED, N16 SUPPORTED.</b> add is gentler than '
        'full-vector rotate; angular (1&minus;cos) predicts rotation logPPL at '
        '<code>R&sup2;=0.997</code> — the Cylindrical Representation Hypothesis.</li>\n'
        '      <li><b>S-8 / E27 cross-scale.</b> The cliff replicates on gemma-3-1b '
        '@L18 with a clean steering window — the window emerges with scale.</li>\n'
        '    </ul>\n  </div>\n')

    def card(name: str, title: str, cap: str, wide: bool = False) -> str:
        cls = "card panel-2col" if wide else "card"
        if not plots.get(name):
            inner = (f'<p class="cap">({_esc(cap)} — plot unavailable: matplotlib '
                     "missing or no data)</p>")
        else:
            inner = (f'<img src="{name}" alt="{_esc(cap)}">'
                     f'<p class="cap">{_esc(cap)}</p>')
        return f'<div class="{cls}"><h3>{_esc(title)}</h3>{inner}</div>'

    # =====================================================================
    # TAB BAR (dsbench interaction model ported to the editorial palette).
    # Five panes: Runs (default), Geometry, Ladder, Hypotheses, Raw.
    # =====================================================================
    parts.append(
        '<div class="tabs" id="tabbar">\n'
        '  <div class="tab active" data-pane="pane-runs">Runs</div>\n'
        '  <div class="tab" data-pane="pane-geometry">Geometry</div>\n'
        '  <div class="tab" data-pane="pane-ladder">Ladder</div>\n'
        '  <div class="tab" data-pane="pane-hypotheses">Hypotheses</div>\n'
        '  <div class="tab" data-pane="pane-raw">Raw</div>\n'
        '</div>\n')

    # ---- PANE: RUNS (default) ----
    hc_rows = [r for r in flat if is_hillclimb(r)]
    parts.append('<div class="tab-pane active" id="pane-runs">\n')
    parts.append(
        '<h2 style="margin-top:18px">Runs by campaign '
        '<span class="sub" style="display:inline">filter, sort, or auto-expand a '
        'row</span></h2>\n')
    # rich filter bar (regex + model dropdown + toggle pills + auto-expand)
    parts.append(
        '<div class="toolbar" id="runs-toolbar">\n'
        '  <label>filter <input type="text" id="filter" class="filter-box" '
        'style="width:240px;margin:0" placeholder="regex over tag / model / hyp / exp#…" '
        'oninput="filterRuns()"></label>\n'
        '  <label>model <select id="model-filter" onchange="filterRuns()">\n'
        '    <option value="">all</option>\n'
        '    <option value="gemma-3-270m">gemma-3-270m</option>\n'
        '    <option value="gemma-3-1b">gemma-3-1b</option>\n'
        '    <option value="qwen">qwen</option>\n'
        '    <option value="fake">fake</option>\n'
        '  </select></label>\n'
        '  <span class="fpill active" data-kind="all" onclick="setPill(this)">all</span>\n'
        '  <span class="fpill" data-kind="real" onclick="setPill(this)">real-Gemma</span>\n'
        '  <span class="fpill" data-kind="verdict" data-val="SUPPORTED" onclick="setPill(this)">SUPPORTED</span>\n'
        '  <span class="fpill" data-kind="verdict" data-val="FALSIFIED" onclick="setPill(this)">FALSIFIED</span>\n'
        '  <span class="fpill" data-kind="status" data-val="KEEP" onclick="setPill(this)">KEEP</span>\n'
        '  <span class="fpill" data-kind="champion" onclick="setPill(this)">champion</span>\n'
        '  <label style="margin-left:auto"><input type="checkbox" id="auto-expand"> '
        'auto-expand row</label>\n'
        '</div>\n'
        '<p class="cap">default sort: composite desc &middot; click any header to '
        're-sort &middot; with auto-expand on, clicking a row reveals an inline '
        'detail panel instead of navigating.</p>\n')
    for gkey, gtitle, gdesc in CAMPAIGN_GROUPS:
        grp_rows = grouped.get(gkey)
        if not grp_rows:
            continue
        n = len(grp_rows)
        parts.append(
            f'<section class="group-section">\n  <h2>{_esc(gtitle)} '
            f'<span class="cnt">({n} run{"s" if n != 1 else ""})</span></h2>\n'
            f'  <div class="group-desc">{_esc(gdesc)}</div>\n'
            f'  <div class="tablewrap">{_runs_table(grp_rows, f"runs_{gkey}")}</div>\n'
            "</section>\n")
    # Hill-climb subsection (rung-2.5)
    parts.append(
        '<section class="group-section" id="hillclimb-section">\n'
        '  <h2>Hill-climb (rung-2.5 coordinate descent) '
        f'<span class="cnt">({len(hc_rows)} run{"s" if len(hc_rows)!=1 else ""})</span></h2>\n'
        '  <div class="group-desc">Coordinate-descent refinement around a screening '
        'champion — sweeps one axis (layer / alpha / source / operation) at a time. '
        'Populated from rows tagged <code>HC-*</code>.</div>\n')
    if hc_rows:
        parts.append(_hc_best_callout(hc_rows))
        if plots.get("plot_hc_coord.png"):
            parts.append(
                '  <div class="card panel-2col"><h3>Per-axis coordinate descent</h3>'
                '<img src="plot_hc_coord.png" alt="hill-climb coordinate descent">'
                '<p class="cap">behavior_efficacy + composite vs each swept axis.</p></div>\n')
        if plots.get("plot_hc_seed.png"):
            parts.append(
                '  <div class="card panel-2col"><h3>Seed stability</h3>'
                '<img src="plot_hc_seed.png" alt="hill-climb seed stability">'
                '<p class="cap">mean composite per config with the seed min/max range.</p></div>\n')
        parts.append(f'  <div class="tablewrap">{_runs_table(hc_rows, "runs_hillclimb")}</div>\n')
    else:
        parts.append(HC_PLACEHOLDER)
    parts.append('</section>\n')
    parts.append('</div>\n')  # end pane-runs

    # ---- PANE: GEOMETRY ----
    parts.append('<div class="tab-pane" id="pane-geometry">\n')
    parts.append('<h2 style="margin-top:18px">Geometry</h2>\n')
    parts.append(card("plot_geometry.png", "Geometry probes",
        "Off-shell displacement, angular displacement, and Fisher ratio per experiment.", wide=True))
    parts.append(_geometry_table(flat))
    parts.append('</div>\n')  # end pane-geometry

    # ---- PANE: LADDER ----
    parts.append('<div class="tab-pane" id="pane-ladder">\n')
    parts.append('<h2 style="margin-top:18px">Ladder board</h2>\n')
    ladder_rows = "".join(
        f'<tr><td class="l">{_esc(L["tag"])}</td>'
        f'<td class="l">{L["rung"]} ({_esc(L["rung_name"])})</td>'
        f'<td class="l">{"cleared" if L["cleared"] else "failed"}</td>'
        f'<td class="l">{_esc(L["failure_reason"] or "—")}</td></tr>'
        for L in ladder) or '<tr><td colspan="4" class="l empty">No runs yet.</td></tr>'
    parts.append(
        '<section class="group-section">\n'
        '  <div class="group-desc">Per method (tag): the highest ladder rung reached '
        'and whether the safety/quality gate cleared.</div>\n'
        '  <div class="tablewrap"><table class="runs">\n'
        '    <thead><tr><th class="l">method (tag)</th><th class="l">highest rung</th>'
        '<th class="l">gate</th><th class="l">failure_reason</th></tr></thead>\n'
        f"    <tbody>{ladder_rows}</tbody>\n  </table></div>\n</section>\n")
    parts.append(render_stack_matrix(rows))
    parts.append('</div>\n')  # end pane-ladder

    # ---- PANE: HYPOTHESES ----
    parts.append('<div class="tab-pane" id="pane-hypotheses">\n')
    parts.append(
        '<section class="card panel-2col" style="margin-top:18px">\n'
        '  <h2>Hypothesis status grid</h2>\n'
        '  <p class="cap" style="margin-bottom:10px">All 70 registry hypotheses '
        '(E1–E50 in blocks A–F, then N1–N20). Cell colour = current verdict from '
        'IDEA_TABLE.md; click a cell to open that hypothesis page.</p>\n'
        + _hypothesis_grid_html(table) + "\n</section>\n")
    parts.append('<h2>5-axis profiles</h2>\n<div class="grid">\n')
    parts.append(card("plot_radar.png", "5-axis radar",
        "5-axis radar per method: behavior, capability, coherence, safety, selectivity (1 = best)."))
    parts.append(card("plot_parcoords.png", "Parallel coordinates",
        "Parallel coordinates across the same five axes — lines high everywhere are multi-objective wins."))
    parts.append(card("plot_pareto_capability.png", "Pareto · capability",
        "behavior vs capability (MMLU drop on x; left = better capability retention)."))
    parts.append(card("plot_pareto_coherence.png", "Pareto · coherence",
        "behavior vs coherence (ΔPPL_norm on x; left = better coherence)."))
    parts.append(card("plot_pareto_safety.png", "Pareto · safety",
        "behavior vs safety (compliance rate on x; left = safer / no leak)."))
    parts.append("</div>\n")
    parts.append(_hypotheses_table(table, rows_by_hyp=_count_rows_by_hyp(flat, repo_root, idea_dirs)))
    parts.append('</div>\n')  # end pane-hypotheses

    # ---- PANE: RAW ----
    parts.append('<div class="tab-pane" id="pane-raw">\n')
    parts.append('<h2 style="margin-top:18px">Raw experiment-log rows</h2>\n')
    raw_src = (f'<a href="{REPO_URL}/blob/master/autoresearch_results/'
               'experiment_log.jsonl" target="_blank" rel="noopener">'
               'autoresearch_results/experiment_log.jsonl</a>')
    PREVIEW = 12
    parts.append(
        '<p class="cap">One slimmed JSON object per '
        f'<code>experiment_log.jsonl</code> row (sample blobs dropped). '
        f'Full {len(rows)}-row source of truth: {raw_src}. '
        f'Preview below shows the first {min(PREVIEW, len(rows))} rows.</p>\n')
    raw_pre = _esc("\n".join(json.dumps(_raw_row(r), sort_keys=True)
                             for r in rows[:PREVIEW]))
    parts.append(
        '<details class="deep"><summary>show raw JSON preview ('
        f'first {min(PREVIEW, len(rows))} of {len(rows)} rows)</summary>\n'
        f'<div class="body"><pre>{raw_pre}</pre></div></details>\n')
    parts.append('</div>\n')  # end pane-raw

    parts.append(_doc_footer())
    parts.append(_meta_footer())
    parts.append(SORT_SCRIPT)
    parts.append("</body>\n</html>\n")
    return "".join(parts)


def _raw_row(r: dict) -> dict:
    """Slim a row for the Raw tab JSON dump (drop verbose sample/prose blobs)."""
    drop = {"samples", "description", "sample_steered", "sample_unsteered",
            "sample_prompt"}
    return {k: v for k, v in r.items() if k not in drop}


def _count_rows_by_hyp(flat: list[dict], repo_root: Path,
                       idea_dirs: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in flat:
        hid = resolve_hyp_id(r, repo_root, idea_dirs)
        if hid:
            counts[hid] = counts.get(hid, 0) + 1
    return counts


def _geometry_table(flat: list[dict]) -> str:
    """Sortable geometry-focused table: offshell / angular / fisher / cos_dm_pca."""
    cols = [
        ("experiment_num", "exp#", True),
        ("tag", "tag", False),
        ("model", "model", False),
        ("offshell_displacement", "offshell Δ‖h‖", True),
        ("angular_displacement", "angular (1−cos)", True),
        ("fisher_at_layer", "Fisher@layer", True),
        ("cosine_dm_pca", "cos(DM,PCA)", True),
        ("composite", "composite", True),
    ]
    head = "".join(
        f'<th{" class=\"l\"" if not num else ""} '
        f'onclick="sortRuns(this,{i},{int(num)})">{_esc(label)}'
        f'<span class="arr"></span></th>'
        for i, (k, label, num) in enumerate(cols))
    body = []
    rows_sorted = sorted(flat, key=lambda r: _num(r, "offshell_displacement"),
                         reverse=True)
    for r in rows_sorted:
        tds = []
        for (k, _l, num) in cols:
            if k == "model":
                m = _short_model(r.get("model"))
                tds.append(f'<td data-v="{_esc(m.lower())}" class="l">{_esc(m)}</td>')
            elif k == "tag":
                tg = str(r.get("tag") or "")
                tds.append(f'<td data-v="{_esc(tg.lower())}" class="l">'
                           f'<code>{_esc(tg)}</code></td>')
            elif num:
                v = r.get(k)
                try:
                    fv = float(v if v is not None else 0.0)
                    disp = f"{fv:.4f}"
                except (TypeError, ValueError):
                    fv, disp = -1e18, "—"
                tds.append(f'<td data-v="{fv}">{_esc(disp)}</td>')
            else:
                tds.append(f'<td class="l">{_esc(r.get(k))}</td>')
        body.append(f'<tr>{"".join(tds)}</tr>')
    body_html = "\n".join(body) or (
        f'<tr><td colspan="{len(cols)}" class="l empty">No rows.</td></tr>')
    return ('<section class="group-section"><h2>Geometry table '
            '<span class="cnt">(sortable)</span></h2>\n'
            '  <div class="group-desc">Sorted by off-shell displacement (desc). '
            'Click a header to re-sort.</div>\n'
            '  <div class="tablewrap"><table class="runs" data-dir="desc">\n'
            f'    <thead><tr>{head}</tr></thead>\n'
            f'    <tbody>\n{body_html}\n    </tbody>\n  </table></div>\n</section>\n')


def _hypotheses_table(table: dict[str, dict], rows_by_hyp: dict[str, int]) -> str:
    """Table of all 70 hypotheses: verdict + #runs + link."""
    body = []
    for _bt, ids in HYP_BLOCKS:
        for hid in ids:
            info = table.get(hid, {"verdict": "PENDING", "title": ""})
            verdict = info.get("verdict", "PENDING")
            vcls = VERDICT_CLASS.get(verdict, "v-pending")
            nruns = rows_by_hyp.get(hid, 0)
            body.append(
                f'<tr data-verdict="{_esc(verdict.upper())}">'
                f'<td class="l"><a href="hyp/{_esc(hid)}.html">{_esc(hid)}</a></td>'
                f'<td class="l">{_esc(info.get("title",""))}</td>'
                f'<td class="l"><span class="hyp-cell {vcls}" '
                f'style="cursor:default;width:auto;padding:1px 7px;display:inline-block">'
                f'{_esc(verdict)}</span></td>'
                f'<td>{nruns}</td>'
                f'<td class="l"><a href="hyp/{_esc(hid)}.html">{_esc(hid)} page &rarr;</a></td>'
                '</tr>')
    body_html = "\n".join(body)
    return ('<section class="group-section"><h2>All 70 hypotheses</h2>\n'
            '  <div class="group-desc">Verdict, run count, and link per registry '
            'hypothesis.</div>\n'
            '  <div class="tablewrap"><table class="runs" data-dir="desc">\n'
            '    <thead><tr><th class="l">id</th><th class="l">title</th>'
            '<th class="l">verdict</th><th># runs</th><th class="l">link</th>'
            '</tr></thead>\n'
            f'    <tbody>\n{body_html}\n    </tbody>\n  </table></div>\n</section>\n')


# ===========================================================================
# C. PER-EXPERIMENT page
# ===========================================================================
REASONING_STEPS = [
    ("1. Diagnose", "diagnosis"),
    ("2. Cite", "citations"),
    ("3. Hypothesize", "hypothesis"),
    ("4. Predict", "prediction"),
    ("5. Verdict", "verdict"),
    ("6. Learn", "learning"),
]

AXIS_METRICS = [
    ("behavior_efficacy", "Axis 1 — behavior efficacy", "high"),
    ("capability_retention", "Axis 2 — capability retention", "~baseline"),
    ("mmlu_drop_pp", "  MMLU drop (pp)", "~0"),
    ("perplexity", "Axis 3 — perplexity", "low"),
    ("dppl_norm", "  ΔPPL_norm", "~0"),
    ("repetition_rate", "  repetition rate", "low"),
    ("compliance_rate", "Axis 4 — JailbreakBench CR", "0 (no leak)"),
    ("harmful_refusal_rate", "  harmful refusal rate", "high"),
    ("harmless_refusal_rate", "Axis 5 — harmless refusal (over-refusal)", "low"),
    ("selectivity_gap", "  selectivity gap", "high"),
]

GEOMETRY_METRICS = [
    ("offshell_displacement", "off-shell displacement Δ‖h‖"),
    ("angular_displacement", "angular displacement (1−cos)"),
    ("fisher_at_layer", "Fisher ratio at layer"),
    ("cosine_dm_pca", "cos(DiffMean, PCA-top1)"),
]


def _extract_samples(row: dict) -> list[dict]:
    cfg = row.get("config", {}) or {}

    def _g(d, *keys):
        for k in keys:
            v = d.get(k)
            if v not in (None, ""):
                return v
        return None

    samples = row.get("samples") or cfg.get("samples")
    if isinstance(samples, list) and samples:
        out = []
        for it in samples:
            if not isinstance(it, dict):
                continue
            out.append({
                "prompt": _g(it, "prompt", "sample_prompt", "input") or "",
                "steered": _g(it, "steered", "sample_steered", "steered_text") or "",
                "unsteered": _g(it, "unsteered", "sample_unsteered",
                                "unsteered_text", "baseline") or "",
            })
        if out:
            return out

    steered = _g(row, "sample_steered", "steered_text") or _g(cfg, "sample_steered", "steered_text")
    unsteered = _g(row, "sample_unsteered", "unsteered_text") or _g(cfg, "sample_unsteered", "unsteered_text")
    prompt = _g(row, "sample_prompt", "prompt") or _g(cfg, "sample_prompt", "prompt") or ""
    if steered or unsteered:
        return [{"prompt": prompt, "steered": steered or "", "unsteered": unsteered or ""}]
    return []


def _samples_section(row: dict) -> str:
    samples = _extract_samples(row)
    if not samples:
        return (
            '<section class="card"><h2>Side-by-side samples (steered vs unsteered)</h2>\n'
            '  <p class="cap">No samples captured for this run. The two-column layout '
            'below is ready; it populates once the runner logs <code>sample_steered</code>'
            ' / <code>sample_unsteered</code> (or a <code>samples</code> list) on the '
            'experiment row.</p>\n  <div class="charts">\n'
            '    <div><h3>Unsteered (baseline)</h3>'
            '<pre class="sample empty">(no samples captured for this run)</pre></div>\n'
            '    <div><h3>Steered</h3>'
            '<pre class="sample empty">(no samples captured for this run)</pre></div>\n'
            "  </div>\n</section>\n")
    blocks = []
    for i, s in enumerate(samples, 1):
        prompt_html = (f'  <p class="cap">Prompt {i}: <code>{_esc(s["prompt"])}</code></p>\n'
                       if s.get("prompt") else "")
        blocks.append(
            prompt_html + '  <div class="charts">\n'
            '    <div><h3>Unsteered (baseline)</h3>'
            f'<pre class="sample">{_esc(s["unsteered"]) or "(empty)"}</pre></div>\n'
            '    <div><h3>Steered</h3>'
            f'<pre class="sample">{_esc(s["steered"]) or "(empty)"}</pre></div>\n  </div>\n')
    return ('<section class="card"><h2>Side-by-side samples (steered vs unsteered)</h2>\n'
            f'  <p class="cap">{len(samples)} captured generation pair(s) from the '
            "experiment row.</p>\n" + "".join(blocks) + "</section>\n")


def _composite_stack(flat: dict) -> tuple[str, str]:
    """Composite term-by-term breakdown (stacked bar + table). Returns
    (stacked_bar_html, breakdown_table_html); reconciled to the logged composite."""
    def w(name):
        return COMPOSITE_WEIGHTS.get(name, 0.0)
    be = _num(flat, "behavior_efficacy")
    terms = [
        ("behavior_efficacy", 1.0, be, be),
        ("− λ_cap · MMLU_drop", -w("lambda_cap"), max(0.0, _num(flat, "mmlu_drop_pp")),
         -w("lambda_cap") * max(0.0, _num(flat, "mmlu_drop_pp"))),
        ("− λ_coh · ΔPPL_norm", -w("lambda_coh"), max(0.0, _num(flat, "dppl_norm")),
         -w("lambda_coh") * max(0.0, _num(flat, "dppl_norm"))),
        ("− λ_rep · repetition", -w("lambda_coh_rep"), max(0.0, _num(flat, "repetition_rate")),
         -w("lambda_coh_rep") * max(0.0, _num(flat, "repetition_rate"))),
        ("− λ_safe · compliance", -w("lambda_safe"), _num(flat, "compliance_rate"),
         -w("lambda_safe") * _num(flat, "compliance_rate")),
        ("− λ_sel · harmless_refusal", -w("lambda_sel"), max(0.0, _num(flat, "harmless_refusal_rate")),
         -w("lambda_sel") * max(0.0, _num(flat, "harmless_refusal_rate"))),
        ("− λ_geo · offshell", -w("lambda_geo"), max(0.0, _num(flat, "offshell_displacement")),
         -w("lambda_geo") * max(0.0, _num(flat, "offshell_displacement"))),
    ]
    total = sum(t[3] for t in terms)
    reported = _num(flat, "composite")

    # stacked bar — magnitudes; positive seg green, cost segs orange.
    mags = [(t[0], t[3]) for t in terms if abs(t[3]) > 1e-9]
    tot_mag = sum(abs(m) for _n, m in mags) or 1.0
    segs = []
    for name, val in mags:
        pct = 100.0 * abs(val) / tot_mag
        if pct < 0.6:
            continue
        cls = "pos" if val >= 0 else "cost"
        sign = "+" if val >= 0 else "−"
        segs.append(f'<div class="seg {cls}" style="width:{pct:.2f}%" '
                    f'title="{_esc(name)} = {val:+.4f}">{_esc(name.split(" ")[-1])} '
                    f'{sign}{abs(val):.3f}</div>')
    delta = total - reported
    stack = (
        '<div class="comp-stack">' + "".join(segs) + "</div>\n"
        '<div style="display:flex;justify-content:space-between;'
        'font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:var(--paper-dim);'
        'margin-top:6px">'
        f'<span>Σ recomputed: <b style="color:var(--paper)">{total:+.5f}</b></span>'
        f'<span>reported: <b style="color:var(--paper)">{reported:+.5f}</b> '
        f'&middot; Δ {delta:+.6f}</span></div>\n')

    brk_rows = "".join(
        f'<tr><td class="term">{_esc(t[0])}</td>'
        f'<td class="term mut" style="text-align:right">{t[1]:+.3f}</td>'
        f'<td class="term" style="text-align:right">{t[2]:+.4f}</td>'
        f'<td class="term {"pos" if t[3] >= 0 else "neg"}" style="text-align:right">'
        f'{t[3]:+.5f}</td></tr>' for t in terms)
    recon = "" if abs(delta) < 5e-3 else (
        f'<p class="warn">Reconstructed {total:+.4f} differs from reported '
        f'{reported:+.4f} — review weighting.</p>')
    table = (
        '<table class="kvtable breakdown"><thead><tr><th>term</th>'
        '<th style="text-align:right">weight</th><th style="text-align:right">raw</th>'
        '<th style="text-align:right">contribution</th></tr></thead><tbody>'
        f'{brk_rows}<tr style="border-top:2px solid var(--rule-bright)">'
        f'<td class="term" colspan="3"><b>Σ (recomputed)</b></td>'
        f'<td class="term" style="text-align:right"><b>{total:+.5f}</b></td></tr>'
        f'</tbody></table>{recon}')
    return stack, table


def render_experiment(row: dict, ann: dict, table: dict[str, dict],
                      repo_root: Path, idea_dirs: list[Path],
                      prev_exp: Optional[int] = None,
                      next_exp: Optional[int] = None,
                      sweep_plot: Optional[str] = None,
                      sweep_single: bool = False) -> str:
    flat = _flatten(row)
    exp = flat.get("experiment_num")
    exp_id = f"exp{int(exp):03d}" if exp is not None else "exp"
    tag = str(flat.get("tag") or "")
    n_seeds = int(flat.get("n_seeds", 1) or 1)
    tier = _tier_of(flat)
    status = str(flat.get("status") or "")
    hid = resolve_hyp_id(row, repo_root, idea_dirs)
    hinfo = _hyp_info(hid, table, repo_root, idea_dirs) if hid else {}
    model = _short_model(flat.get("model"))
    comp = _num(flat, "composite")

    back = '<a class="back" href="../index.html">&larr; back to master</a>'

    # verdict-row provenance chips
    vbadges = [f'<span class="vbadge {"keep" if status=="KEEP" else "discard"}">{_esc(status or "?")}</span>',
               f'<span class="vbadge prov">tier {tier.lower()} n={n_seeds}</span>']
    if flat.get("behavior_scorer"):
        vbadges.append(f'<span class="vbadge prov">behavior: {_esc(flat.get("behavior_scorer"))}</span>')
    if flat.get("safety_real") is not None:
        vbadges.append(f'<span class="vbadge prov">safety_real: {_esc(flat.get("safety_real"))}</span>')

    hyp_pill = (f'<span class="pill hyp">{_esc(hid)}</span>' if hid
                else '<span class="pill">infra</span>')
    grp_pill = ""
    gkey = campaign_group_of(row)
    for k, gt, _gd in CAMPAIGN_GROUPS:
        if k == gkey:
            grp_pill = f'<span class="pill grp">{_esc(gt)}</span>'
            break

    parts = [_page_open(f"Experiment {exp_id} — {tag}")]
    parts.append('<div class="grain"></div>\n')
    parts.append(
        '<div class="head-grid">\n  <div class="head-left">\n'
        f'    <div class="tag-display">{_esc(exp_id)}</div>\n'
        f'    <div class="sub">{hyp_pill}{grp_pill}'
        f'<span class="pill ds">{_esc(model)}</span>'
        f'<span class="pill">layer {_esc(flat.get("layer"))}</span>'
        f'<span class="pill">α {_esc(flat.get("alpha"))}</span>'
        f'&nbsp;&middot;&nbsp; tag <span class="mono" style="color:var(--paper)">'
        f'{_esc(tag)}</span></div>\n'
        f'    {_mast_row()}'
        '  </div>\n  <div class="head-right">\n'
        f'    {back}\n'
        f'    <div class="verdict-row">{"".join(vbadges)}</div>\n'
        '  </div>\n</div>\n')

    # Key-numbers strip
    def kn(val, lbl, tint):
        return (f'<div class="kn-tile tint-{tint}"><div class="kn-val">{val}</div>'
                f'<div class="kn-lbl">{_esc(lbl)}</div></div>')
    parts.append('<div class="kn-strip">'
        + kn(f"{comp:+.4f}", "composite", "pos" if comp > 0 else "neg")
        + kn(f"{_num(flat,'behavior_efficacy'):.3f}", "behavior", "neu")
        + kn(f"{_num(flat,'perplexity'):.1f}", "perplexity", "neu")
        + kn(f"{_num(flat,'mmlu_drop_pp')*100:.1f}pp", "MMLU drop", "neu")
        + kn(f"{_num(flat,'compliance_rate'):.2f}", "CR (jailbreak)",
             "neg" if _num(flat,'compliance_rate') > 0 else "pos")
        + kn(f"n={n_seeds}", f"tier {tier.lower()}", "neu")
        + "</div>\n")

    # Hypothesis digest card
    if hid and (hinfo.get("title") or hinfo.get("hypothesis")):
        idir = _idea_dir_for_id(hid, repo_root, table, idea_dirs)
        extra = ""
        if idir:
            info = _parse_idea_md(idir)
            if info.get("falsifier"):
                extra += ('<h3>Falsifier</h3><div class="md-body">'
                          + md_to_html(info["falsifier"]) + "</div>")
            if info.get("predicted"):
                extra += ('<h3>Predicted Δ</h3><div class="md-body">'
                          + md_to_html(info["predicted"]) + "</div>")
        verdict = hinfo.get("verdict", "PENDING")
        vcls = VERDICT_CLASS.get(verdict, "v-pending")
        parts.append(
            '<section class="card"><h2>Hypothesis '
            f'<a href="../hyp/{_esc(hid)}.html" class="mono" '
            f'style="font-size:0.6em;letter-spacing:0.12em">{_esc(hid)} page &rarr;</a></h2>\n'
            f'  <p><span class="hyp-cell {vcls}" style="cursor:default;width:auto;'
            f'padding:2px 8px;display:inline-block">{_esc(verdict)}</span> '
            f'<b>{_esc(hid)} — {_esc(hinfo.get("title",""))}</b></p>\n'
            f'  <div class="md-body">{md_to_html(hinfo.get("hypothesis",""))}</div>\n'
            f'  <p class="cap">Metric + threshold: {_esc(hinfo.get("metric",""))}</p>\n'
            f'  {extra}\n</section>\n')
    else:
        parts.append(
            '<section class="card"><h2>Hypothesis</h2>'
            '<p class="empty">No hypothesis resolved for this row — '
            'infrastructure / plumbing run.</p></section>\n')

    # Verdict card (from reasoning verdict blob)
    verdict_blob = ann.get("verdict") if ann else None
    learning_blob = ann.get("learning") if ann else None
    vcard = ""
    if verdict_blob:
        vcard += f'<div class="quote md-body">{md_to_html(str(verdict_blob))}</div>'
    if learning_blob:
        vcard += ('<h3>Learning</h3><div class="md-body">'
                  + md_to_html(str(learning_blob)) + "</div>")
    if not vcard:
        vcard = (f'<p>Run verdict: <b>{_esc(status)}</b> — see the reasoning blob '
                 'below. (No reasoning verdict field archived for this row.)</p>')
    parts.append(
        '<section class="card"><h2>Verdict '
        '<span class="mono" style="font-size:0.55em;letter-spacing:0.18em;'
        'color:var(--paper-dim)">reasoning_annotations.json</span></h2>\n'
        f'  {vcard}\n</section>\n')

    # 7-step reasoning blob
    steps_html = []
    for label, key in REASONING_STEPS:
        val = ann.get(key) if ann else None
        if val:
            steps_html.append(
                f'<div class="reason-section"><div class="lbl">{_esc(label)}</div>'
                f'<div class="md-body">{md_to_html(str(val))}</div></div>')
        else:
            steps_html.append(
                f'<div class="reason-section"><div class="lbl">{_esc(label)}</div>'
                f'<p class="warn">(reasoning field "{_esc(key)}" missing.)</p></div>')
    if not ann:
        steps_html.insert(0, '<p class="warn">(No reasoning annotation found for '
                          "this experiment.)</p>")
    parts.append('<section class="card"><h2>Reasoning entry</h2>\n'
                 + "\n".join(steps_html) + "\n</section>\n")

    # Configuration
    cfg = row.get("config", {}) or {}
    cfg_json = _esc(json.dumps(cfg, indent=2))
    parts.append(
        '<section class="card"><h2>Configuration</h2>\n'
        f'  <pre>{cfg_json}</pre>\n'
        '  <p class="cap">Source: experiment_log.jsonl config block.</p>\n</section>\n')

    # Five-axis metrics quick reference
    def _ci_cells(key):
        lo = row.get(f"{key}_ci_low")
        hi = row.get(f"{key}_ci_high")
        if lo is None and hi is None:
            return '<td class="mut">(CI not computed)</td><td class="mut">(CI not computed)</td>'
        return f"<td>{_esc(lo)}</td><td>{_esc(hi)}</td>"
    metric_rows = []
    for key, label, good in AXIS_METRICS:
        v = row.get(key)
        disp = f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v))
        metric_rows.append(
            f'<tr><td class="k">{_esc(label)}</td>'
            f'<td class="v">{_esc(disp)} <span class="n-chip scr">n={n_seeds}</span></td>'
            f'<td class="mut">{_esc(good)}</td>{_ci_cells(key)}</tr>')
    parts.append(
        '<section class="card"><h2>Five-axis metrics — quick reference</h2>\n'
        '  <table class="kvtable"><thead><tr><th>metric</th><th>value (n)</th>'
        '<th>good=</th><th>CI low</th><th>CI high</th></tr></thead>\n'
        f"  <tbody>{''.join(metric_rows)}</tbody></table>\n</section>\n")

    # Composite breakdown
    stack, brk = _composite_stack(flat)
    parts.append(
        '<section class="card"><h2>Composite-score breakdown</h2>\n'
        f'  {stack}\n  {brk}\n'
        f'  <div class="formula-chip" style="margin-top:14px">composite &#8788; '
        f'<code>{_esc(COMPOSITE_FORMULA)}</code></div>\n</section>\n')

    # Geometry probes
    geo_rows = []
    for key, label in GEOMETRY_METRICS:
        v = row.get(key)
        if v is None:
            continue
        disp = f"{v:.4f}" if isinstance(v, float) else str(v)
        geo_rows.append(f'<tr><td class="k">{_esc(label)}</td>'
                        f'<td class="v">{_esc(disp)} <span class="n-chip scr">n={n_seeds}</span></td></tr>')
    if not geo_rows:
        geo_rows.append('<tr><td colspan="2" class="empty">No geometry probes logged.</td></tr>')
    parts.append(
        '<section class="card"><h2>Geometry probes</h2>\n'
        '  <table class="kvtable"><thead><tr><th>probe</th><th>value (n)</th></tr></thead>\n'
        f"  <tbody>{''.join(geo_rows)}</tbody></table>\n</section>\n")

    # Sweep curve
    if sweep_plot:
        note = (' <span class="cap">sweep accumulates as more alpha/layer rows are '
                "logged</span>") if sweep_single else ""
        sweep_body = (f'<img src="{_esc(sweep_plot)}" alt="sweep curve">'
                      f'<p class="cap">behavior_efficacy (left axis) and perplexity '
                      f'(right axis) vs the swept alpha/layer axis for this hypothesis '
                      f'group.{note}</p>')
    else:
        sweep_body = ('<p class="cap">(sweep curve unavailable — matplotlib missing or '
                      'no alpha/layer value logged for this run; sweep accumulates as '
                      'more alpha/layer rows are logged.)</p>')
    parts.append(
        '<section class="card"><h2>Sweep curve (behavior + PPL vs alpha/layer)</h2>\n'
        f'  {sweep_body}\n</section>\n')

    # Samples
    parts.append(_samples_section(row))

    # Cross-references grid
    xrefs = []
    if hid:
        xrefs.append(
            f'<a class="xref-card" href="../hyp/{_esc(hid)}.html">'
            f'<div class="xref-lbl">&#8599; hypothesis</div>'
            f'<div class="xref-tag">{_esc(hid)}</div>'
            f'<div class="xref-meta">{_esc(hinfo.get("title",""))}</div></a>')
    if prev_exp is not None:
        xrefs.append(
            f'<a class="xref-card" href="exp{int(prev_exp):03d}.html">'
            f'<div class="xref-lbl">&#8592; previous experiment</div>'
            f'<div class="xref-tag">exp{int(prev_exp):03d}</div></a>')
    if next_exp is not None:
        xrefs.append(
            f'<a class="xref-card" href="exp{int(next_exp):03d}.html">'
            f'<div class="xref-lbl">&#8594; next experiment</div>'
            f'<div class="xref-tag">exp{int(next_exp):03d}</div></a>')
    xrefs.append(
        '<a class="xref-card" href="../index.html">'
        '<div class="xref-lbl">&#8599; master dashboard</div>'
        '<div class="xref-tag">index</div>'
        '<div class="xref-meta">runs table + hypothesis grid</div></a>')
    parts.append(
        '<section class="card"><h2>Cross-references</h2>\n'
        f'  <div class="xrefs-grid">{"".join(xrefs)}</div>\n</section>\n')

    parts.append(_doc_footer())
    extra = f' &middot; experiment <code>{_esc(exp_id)}</code>'
    if hid:
        extra += f' &middot; hypothesis <code>{_esc(hid)}</code>'
    parts.append(_meta_footer(extra=extra))
    parts.append("</body>\n</html>\n")
    return "".join(parts)


# ===========================================================================
# B. PER-HYPOTHESIS page
# ===========================================================================
def _campaign_writeup_for(hid: str, repo_root: Path) -> str:
    """Find and render the most relevant ideas/_campaigns/*.md writeup for a
    hypothesis (matched by the campaign tag that maps to this hypothesis)."""
    camp = repo_root / "ideas" / "_campaigns"
    if not camp.exists():
        return ""
    # campaign-md -> hypothesis: scan each *_results.md / *.md for the hyp id.
    best = None
    for p in sorted(camp.glob("*.md")):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        # match e.g. "(E27)" or "E2 " or "N20" in the heading / body
        if re.search(rf"\b{re.escape(hid)}\b", text):
            best = (p, text)
            # prefer a *_results.md write-up over a json sidecar name
            if p.name.endswith("_results.md"):
                break
    if not best:
        return ""
    p, text = best
    return (f'<p class="cap">Campaign writeup: <code>ideas/_campaigns/{_esc(p.name)}</code></p>\n'
            f'<div class="md-body">{md_to_html(text)}</div>')


_DESIGN_DOC_BLOB = "https://github.com/dlmastery/steeringresearch/blob/master/"


def _design_doc_link(hid: str, repo_root: Path) -> str:
    """Link to the rich 12-section design doc hypotheses/<block>/<ID>_*.md if it exists."""
    hyp_root = Path(repo_root) / "hypotheses"
    if not hyp_root.exists():
        return ""
    for p in sorted(hyp_root.glob(f"*/{hid}_*.md")):
        rel = p.relative_to(repo_root).as_posix()
        return (f'<a class="mast-pill paper" href="{_DESIGN_DOC_BLOB}{rel}" '
                f'target="_blank" rel="noopener">design doc &middot; {_esc(hid)}</a>')
    return ""


def render_hypothesis(hid: str, table: dict[str, dict], rows: list[dict],
                      repo_root: Path, idea_dirs: list[Path],
                      plots: dict[str, bool]) -> str:
    info = _hyp_info(hid, table, repo_root, idea_dirs)
    verdict = info.get("verdict", "PENDING")
    vcls = VERDICT_CLASS.get(verdict, "v-pending")
    flat = [_flatten(r) for r in rows]
    flat.sort(key=lambda r: _num(r, "composite", -1e18), reverse=True)
    best = flat[0] if flat else None

    idir = _idea_dir_for_id(hid, repo_root, table, idea_dirs)
    idea = _parse_idea_md(idir) if idir else {"statement": "", "falsifier": "", "predicted": "", "title": ""}

    parts = [_page_open(f"Hypothesis {hid} — {info.get('title','')}")]
    parts.append('<div class="grain"></div>\n')
    back = '<a class="back" href="../index.html">&larr; back to master grid</a>'
    parts.append(
        '<div class="head-grid">\n  <div class="head-left">\n'
        f'    <div class="tag-display">{_esc(hid)}</div>\n'
        f'    <div class="sub"><span class="hyp-cell {vcls}" style="cursor:default;'
        f'width:auto;padding:2px 8px;display:inline-block">{_esc(verdict)}</span> '
        f'&nbsp; {_esc(info.get("title",""))} &nbsp;&middot;&nbsp; '
        f'{len(flat)} run{"s" if len(flat)!=1 else ""}</div>\n'
        f'    {_mast_row()}'
        f'    <div style="margin-top:6px">{_design_doc_link(hid, repo_root)}</div>'
        '  </div>\n  <div class="head-right">\n'
        f'    {back}\n  </div>\n</div>\n')

    # Hypothesis card
    card = (f'<p><span class="hyp-cell {vcls}" style="cursor:default;width:auto;'
            f'padding:2px 8px;display:inline-block">{_esc(verdict)}</span> '
            f'<b>{_esc(hid)} — {_esc(info.get("title",""))}</b></p>\n'
            f'<h3>Falsifiable hypothesis</h3><div class="md-body">'
            f'{md_to_html(info.get("hypothesis",""))}</div>\n'
            f'<h3>Primary metric + threshold</h3><div class="md-body">'
            f'{md_to_html(info.get("metric",""))}</div>\n'
            f'<h3>Current verdict (IDEA_TABLE.md)</h3><div class="md-body">'
            f'{md_to_html(info.get("verdict_detail",""))}</div>\n')
    if idea.get("statement"):
        card += ('<h3>Pre-registered claim (idea dir)</h3><div class="md-body">'
                 + md_to_html(idea["statement"]) + "</div>")
    if idea.get("falsifier"):
        card += ('<h3>Falsifier (idea dir)</h3><div class="md-body">'
                 + md_to_html(idea["falsifier"]) + "</div>")
    if idea.get("predicted"):
        card += ('<h3>Predicted Δ (idea dir)</h3><div class="md-body">'
                 + md_to_html(idea["predicted"]) + "</div>")
    parts.append(f'<section class="card"><h2>Hypothesis card</h2>\n{card}</section>\n')

    # Best-config callout
    if best is not None:
        bn = int(best.get("n_seeds", 1) or 1)
        bexp = best.get("experiment_num")
        link = (f'<a href="../experiments/exp{int(bexp):03d}.html">exp{int(bexp):03d}</a>'
                if bexp is not None else "—")
        parts.append(
            '<section class="card"><h2>Best-config callout</h2>\n'
            f'  <p>{link} &middot; tag <code>{_esc(best.get("tag") or "")}</code> '
            f'&middot; composite <b>{_num(best,"composite"):+.4f}</b> '
            f'<span class="n-chip scr">n={bn}</span><br>\n'
            f'  behavior {_num(best,"behavior_efficacy"):.4f} &middot; '
            f'ΔMMLU {_num(best,"mmlu_drop_pp"):.4f} &middot; '
            f'PPL {_num(best,"perplexity"):.4f} &middot; '
            f'CR {_num(best,"compliance_rate"):.4f} &middot; '
            f'over_refusal {_num(best,"harmless_refusal_rate"):.4f}</p>\n</section>\n')

    # Coordinate descent / seed stability plots
    if plots.get(f"hyp{hid}_coord.png"):
        parts.append(
            '<section class="card"><h2>Coordinate descent</h2>\n'
            f'  <img src="hyp{hid}_coord.png" alt="composite vs layer / alpha">\n'
            '  <p class="cap">composite vs injection layer and vs alpha for this '
            'hypothesis sweep.</p>\n</section>\n')

    # Hill-climb (rung-2.5) tier for this hypothesis
    hc_rows = [r for r in flat if is_hillclimb(r)]
    hc_inner = ['<section class="card" id="hillclimb-hyp"><h2>Hill-climb '
                '(rung-2.5 coordinate descent)</h2>\n']
    if hc_rows:
        hc_inner.append(_hc_best_callout(hc_rows, link_prefix="../"))
        if plots.get(f"hyp{hid}_hc_coord.png"):
            hc_inner.append(
                f'  <img src="hyp{hid}_hc_coord.png" alt="hill-climb coordinate descent">\n'
                '  <p class="cap">behavior_efficacy + composite vs each swept axis '
                '(layer / alpha / source / operation).</p>\n')
        if plots.get(f"hyp{hid}_hc_seed.png"):
            hc_inner.append(
                f'  <img src="hyp{hid}_hc_seed.png" alt="hill-climb seed stability">\n'
                '  <p class="cap">mean composite per config with the seed min/max '
                'range (only configs with &gt;1 seed shown).</p>\n')
    else:
        hc_inner.append(HC_PLACEHOLDER)
    hc_inner.append('</section>\n')
    parts.append("".join(hc_inner))

    # Campaign writeup
    writeup = _campaign_writeup_for(hid, repo_root)
    if writeup:
        parts.append(f'<section class="card"><h2>Campaign writeup</h2>\n{writeup}\n</section>\n')

    # Experiments table
    if flat:
        body = []
        for r in flat:
            n = int(r.get("n_seeds", 1) or 1)
            tier = _tier_of(r)
            exp = r.get("experiment_num")
            page = (f'<a href="../experiments/exp{int(exp):03d}.html">exp{int(exp):03d}</a>'
                    if exp is not None else "—")
            chip = f'<span class="n-chip {"eval" if tier=="EVALUATION" else "scr"}">{tier} n={n}</span>'
            body.append(
                f'<tr><td class="l">{page}</td>'
                f'<td class="l"><code>{_esc(r.get("tag") or "")}</code></td>'
                f'<td class="l">{_esc(_short_model(r.get("model")))}</td>'
                f'<td>{_esc(r.get("layer"))}</td><td>{_esc(r.get("alpha"))}</td>'
                f'<td>{_num(r,"behavior_efficacy"):.4f}</td>'
                f'<td>{_num(r,"perplexity"):.2f}</td>'
                f'<td>{_num(r,"compliance_rate"):.2f}</td>'
                f'<td class="comp">{_num(r,"composite"):+.4f}</td>'
                f'<td class="l">{chip}</td></tr>')
        body_html = "\n".join(body)
        parts.append(
            '<section class="card"><h2>Experiments for this hypothesis</h2>\n'
            '  <div class="tablewrap"><table class="runs">\n'
            '    <thead><tr><th class="l">exp</th><th class="l">tag</th>'
            '<th class="l">model</th><th>layer</th><th>α</th><th>behavior</th>'
            '<th>PPL</th><th>CR</th><th>composite</th><th class="l">tier</th></tr></thead>\n'
            f'    <tbody>{body_html}</tbody></table></div>\n</section>\n')
    else:
        parts.append(
            '<section class="card"><h2>Experiments for this hypothesis</h2>\n'
            '  <p class="empty">No runs logged for this hypothesis yet — it is '
            'pre-registered in IDEA_TABLE.md and awaiting its first experiment.</p>\n'
            '</section>\n')

    parts.append(_doc_footer())
    parts.append(_meta_footer(extra=f' &middot; hypothesis <code>{_esc(hid)}</code>'))
    parts.append("</body>\n</html>\n")
    return "".join(parts)


# ===========================================================================
# Build orchestration
# ===========================================================================
def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def build_all_dashboards(results_dir: Path | None = None,
                         repo_root: Path | None = None) -> Path:
    """Regenerate the full three-tier dashboard (A master + B per-hypothesis +
    C per-experiment) and the docs/ mirrors. Returns the master index path."""
    repo_root = Path(repo_root) if repo_root else REPO_ROOT
    results_dir = Path(results_dir) if results_dir else (repo_root / "autoresearch_results")

    log_path = results_dir / "experiment_log.jsonl"
    rows = load_rows(log_path)
    reasoning = load_reasoning(results_dir)
    table = parse_idea_table(repo_root / "IDEA_TABLE.md")
    # Ensure the full 70-hypothesis registry exists even if the table is sparse.
    for hid in (h for _t, ids in HYP_BLOCKS for h in ids):
        table.setdefault(hid, {"id": hid, "title": "", "hypothesis": "",
                               "metric": "", "status_cell": "", "verdict": "PENDING",
                               "verdict_detail": "", "rung": "", "idea_dir": ""})

    idea_dirs = list_idea_dirs(repo_root)

    dash = repo_root / "dashboard"
    docs_dash = repo_root / "docs" / "dashboard"
    dash.mkdir(parents=True, exist_ok=True)
    docs_dash.mkdir(parents=True, exist_ok=True)

    # ---- master plots ----
    plots: dict[str, bool] = {}
    plots["plot_radar.png"] = plot_radar(rows, dash / "plot_radar.png")
    plots["plot_parcoords.png"] = plot_parcoords(rows, dash / "plot_parcoords.png")
    plots["plot_pareto_capability.png"] = plot_pareto(
        rows, "mmlu_drop_pp", "MMLU drop (capability cost)", dash / "plot_pareto_capability.png")
    plots["plot_pareto_coherence.png"] = plot_pareto(
        rows, "dppl_norm", "ΔPPL_norm (coherence cost)", dash / "plot_pareto_coherence.png")
    plots["plot_pareto_safety.png"] = plot_pareto(
        rows, "compliance_rate", "compliance rate (safety cost)", dash / "plot_pareto_safety.png")
    plots["plot_geometry.png"] = plot_geometry(rows, dash / "plot_geometry.png")

    # ---- hill-climb (rung-2.5) master plots — only when HC-* rows exist ----
    hc_rows_all = [r for r in rows if is_hillclimb(r)]
    if hc_rows_all:
        plots["plot_hc_coord.png"] = plot_hc_coord_descent(
            hc_rows_all, dash / "plot_hc_coord.png")
        plots["plot_hc_seed.png"] = plot_hc_seed_stability(
            hc_rows_all, dash / "plot_hc_seed.png")

    # mirror master PNGs to docs
    for pname in plots:
        src = dash / pname
        if src.exists():
            _write_bytes(docs_dash / pname, src.read_bytes())

    ladder = ladder_board(rows)

    # ---- A. master ----
    master_html = render_master(rows, table, plots, ladder, repo_root, idea_dirs)
    _write(dash / "index.html", master_html)
    _write(docs_dash / "index.html", master_html)

    # ---- C. per-experiment pages ----
    exp_dir = dash / "experiments"
    docs_exp_dir = docs_dash / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    docs_exp_dir.mkdir(parents=True, exist_ok=True)

    def _sweep_group_key(r: dict) -> str:
        cfg = r.get("config", {}) or {}
        hyp = hypothesis_for_tag(r.get("config", {}).get("tag") if r.get("config") else None) or ""
        beh = str(cfg.get("behavior") or r.get("behavior") or "")
        layer = str(cfg.get("layer") or r.get("layer") or "")
        # group rows that share hypothesis + behavior + (for layer sweeps) alpha.
        return f"{hyp}|{beh}|{layer if hyp=='' else ''}"

    sweep_groups: dict[str, list[dict]] = {}
    for r in rows:
        sweep_groups.setdefault(_sweep_group_key(r), []).append(r)

    exp_nums = sorted(int(r["experiment_num"]) for r in rows
                      if r.get("experiment_num") is not None)
    for row in rows:
        exp = row.get("experiment_num")
        if exp is None:
            continue
        exp_id = f"exp{int(exp):03d}"
        ann = reasoning.get(str(exp), {}) if reasoning else {}
        siblings = sweep_groups.get(_sweep_group_key(row), [row])
        sweep_name = f"{exp_id}_sweep.png"
        ok, single = plot_sweep(siblings, exp_dir / sweep_name)
        sweep_plot = sweep_name if ok else None

        idx = exp_nums.index(int(exp)) if int(exp) in exp_nums else -1
        prev_exp = exp_nums[idx - 1] if idx > 0 else None
        next_exp = exp_nums[idx + 1] if 0 <= idx < len(exp_nums) - 1 else None

        page = render_experiment(row, ann, table, repo_root, idea_dirs,
                                 prev_exp=prev_exp, next_exp=next_exp,
                                 sweep_plot=sweep_plot, sweep_single=single)
        _write(exp_dir / f"{exp_id}.html", page)
        _write(docs_exp_dir / f"{exp_id}.html", page)
        if ok:
            src = exp_dir / sweep_name
            if src.exists():
                _write_bytes(docs_exp_dir / sweep_name, src.read_bytes())

    # ---- B. per-hypothesis pages ----
    # Group rows by their resolved hyp id (E/N registry id OR numeric idea-dir
    # id for legacy / non-campaign tags).
    rows_by_hyp: dict[str, list[dict]] = {}
    for row in rows:
        hid = resolve_hyp_id(row, repo_root, idea_dirs) or ""
        if hid:
            rows_by_hyp.setdefault(hid, []).append(row)

    # Page set = the full E/N registry (so every grid cell resolves) PLUS any
    # numeric idea-dir id that runs resolved to (so legacy links resolve).
    registry_ids = [h for _t, ids in HYP_BLOCKS for h in ids]
    extra_ids = [h for h in rows_by_hyp if h not in registry_ids]

    hyp_docs_dir = docs_dash / "hyp"
    hyp_dash_dir = dash / "hyp"
    for hid in registry_ids + sorted(extra_ids):
        hrows = rows_by_hyp.get(hid, [])
        hplots: dict[str, bool] = {}
        if hrows:
            hplots[f"hyp{hid}_coord.png"] = plot_coord_descent(
                hrows, hyp_docs_dir / f"hyp{hid}_coord.png")
            hc_hrows = [r for r in hrows if is_hillclimb(r)]
            if hc_hrows:
                hplots[f"hyp{hid}_hc_coord.png"] = plot_hc_coord_descent(
                    hc_hrows, hyp_docs_dir / f"hyp{hid}_hc_coord.png")
                hplots[f"hyp{hid}_hc_seed.png"] = plot_hc_seed_stability(
                    hc_hrows, hyp_docs_dir / f"hyp{hid}_hc_seed.png")
        sub_html = render_hypothesis(hid, table, hrows, repo_root, idea_dirs, hplots)
        _write(hyp_docs_dir / f"{hid}.html", sub_html)
        _write(hyp_dash_dir / f"{hid}.html", sub_html)
        for pname in hplots:
            src = hyp_docs_dir / pname
            if src.exists():
                _write_bytes(hyp_dash_dir / pname, src.read_bytes())

        # also write the legacy ideas/<dir>/dashboard/index.html where a dir exists
        idir = _idea_dir_for_id(hid, repo_root, table, idea_dirs)
        if idir is not None:
            _write(idir / "dashboard" / "index.html", sub_html)

    return dash / "index.html"


def build_dashboard(results_dir: Path | None = None) -> Path:
    return build_all_dashboards(results_dir=results_dir, repo_root=REPO_ROOT)


def main() -> None:
    out = build_all_dashboards()
    print(f"Master dashboard written: {out}")
    print(f"Mirror: {REPO_ROOT / 'docs' / 'dashboard' / 'index.html'}")


if __name__ == "__main__":
    main()
