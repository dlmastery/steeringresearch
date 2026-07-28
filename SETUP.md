# SETUP — reproducing this work on another machine

Everything below was learned the hard way on the original host. The traps in §5 each
cost real time; read them before debugging anything.

---

## 1. What you need

| | |
|---|---|
| GPU | one CUDA GPU, **16 GB** is enough (developed on an RTX 4090 Laptop). CPU-only works for the judge-free geometry experiments but not for generation. |
| Disk | **~90 GB** — SVD corpus 38 GB + extraction 41 GB, models ~10 GB |
| RAM | 16 GB minimum; 32 GB comfortable. **RAM, not VRAM, is the usual bottleneck** — a browser holding 28 GB will get your jobs OOM-reaped. |
| Python | **3.12** (3.13 does NOT work — see §5) |

---

## 2. Environment

```bash
conda create -n steering python=3.12 -y
conda activate steering

# CUDA build — do NOT let pip pick the CPU wheel
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install "transformers==4.55.0" "huggingface_hub==0.36.2" \
            "datasets==2.17.1" accelerate bitsandbytes \
            "numpy==1.26.4" scipy scikit-learn pandas matplotlib \
            sentence-transformers truststore soundfile librosa
```

Verify before going further — all three must be true:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers, huggingface_hub as h; print(transformers.__version__, h.__version__)"
python -c "import truststore; truststore.inject_into_ssl(); print('ssl ok')"
```

> **Pin `huggingface_hub<1.0`.** `transformers 4.55` refuses to import against hub 1.x
> with `ImportError: huggingface-hub>=0.34.0,<1.0 is required`.

---

## 3. Model access

```bash
huggingface-cli login          # needs a READ token
```

Gated repos used here: **`google/embeddinggemma-300m`** and the Gemma family. Accept
each model's licence on its HF page first.

**Verify access with a real weight fetch, never a file listing** — repo *metadata* is
public while weights are gated, so `list_repo_files` returns 200 on a repo you cannot
actually download:

```bash
python -c "
import truststore; truststore.inject_into_ssl()
from huggingface_hub import hf_hub_download
print(hf_hub_download('google/embeddinggemma-300m','config.json'))"
```

Models used:

| role | id |
|---|---|
| steering target | `DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated` |
| cross-scale target | `DavidAU/gemma-3-4b-it-heretic-uncensored-abliterated-Extreme` |
| judge (off-family) | `Qwen/Qwen3-4B-Instruct-2507` — **calibrated at AUC 0.665–0.751, below the 0.85 gate; see `autoresearch_results/JUDGE_CARD.md`. Do not build judge-dependent claims on it.** |
| voice backbone | `microsoft/wavlm-base-plus` |

---

## 4. Reproduce the results

```bash
git clone https://github.com/dlmastery/steeringresearch
git clone https://github.com/dlmastery/auto-research-voice-based-disease-detection
git clone https://github.com/dlmastery/yoganext
```

### Steering — judge-free, so nothing depends on the weak judge

```bash
cd steeringresearch
export PYTHONPATH=src

# M-a — the EVALUATION-tier finding (~15 min GPU)
#   rotation costs +91.24 PPL more than addition at matched displacement,
#   CI95 [87.98, 94.71], 8/8 resamples, random control +16.80
python scripts/run_ma_direction_seeds.py --seeds 8 --layer 12 --n-ppl 25

# the hill-climb that produced the champion (f=0.05, r=1.0)
python scripts/hillclimb_angle_radius.py --layer 12 --n-ppl 40

# regenerate the 3-tier dashboard from the clean v2 log
python scripts/adapt_v2_for_dashboard.py
python -c "from pathlib import Path; from steering.dashboard import build_all_dashboards; print(build_all_dashboards(results_dir=Path('.dash_v2').resolve()))"

pytest tests -q          # 261 tests
```

### Voice — F1 needs no audio at all

```bash
cd auto-research-voice-based-disease-detection

# F1: age alone -> ROC-AUC 0.871. 167 KB metadata, CPU, seconds.
python scripts/fetch_svd_resumable.py --only-missing   # or just the metadata
python scripts/audit_demographic_baseline.py --dataset svd

# F3: the full corpus. 38 GB download, RESUMABLE (re-run to continue).
python scripts/fetch_svd_resumable.py                  # ~38 GB
python scripts/preprocess_audio.py --corpus svd --workers 8   # -> 28,509 recs / 1,679 speakers
python scripts/run_benchmark.py --corpus svd --backbone wavlm --head logreg --folds 5 --repeats 8
```

Expect F3 to reproduce as **NOT CLEARED**: age-only rec-AUC ≈ 0.874 vs best audio
≈ 0.739. That is the finding, not a failure.

### yoganext

```bash
cd yoganext && npm install
npm run verify:agent     # must print 24/24 tools, 25/25 UI capabilities
npm run dev
```

---

## 5. Host traps — every one of these cost real time

1. **Python 3.13 is unusable.** The Windows-Store `python` on PATH was 3.13 with
   `torch 2.10.0+cpu` (no CUDA) and a `transformers` that will not import. Always invoke
   the 3.12 interpreter explicitly.
2. **`urlretrieve` does not resume.** The 22 GB SVD archive restarted from zero on every
   interruption and could never finish. `scripts/fetch_svd_resumable.py` uses HTTP
   `Range:` and a `.part` file. Re-running continues; it never re-downloads.
3. **Windows cp1252 console kills unicode.** Printing a German pathology filename
   (`Bulbärparalyse`) raised `UnicodeEncodeError` and killed a 38 GB download outright.
   Use `PYTHONIOENCODING=utf-8` and `.encode('ascii','replace')` in anything that prints
   paths.
4. **HF symlinks need admin on Windows.** `snapshot_download` fails with
   `WinError 1314`. Pass `local_dir=` (copies instead of symlinking) or enable Developer
   Mode.
5. **Piping a long job through `head -n` SIGPIPE-kills it.** Redirect to a file instead.
6. **`nohup … &` inside a backgrounded call orphans the process.** Launch detached
   directly.
7. **ONE GPU.** Re-check `nvidia-smi --query-compute-apps` before every launch. Three
   concurrent model loads got two jobs killed here.
8. **Do not commit `cache/`.** The full-corpus embedding cache is 151 MB, over GitHub's
   100 MB limit, and is derived — regenerate it instead.

---

## 6. Read these first

| file | why |
|---|---|
| `CLAUDE.md` §18 | portable session state: what is done, what is broken, next actions |
| `autoresearch_results/PROVENANCE.md` | **the legacy 124-row log is not one comparable scale** — do not rank it |
| `autoresearch_results/JUDGE_CARD.md` | the judge fails its own gate; no judge-dependent claim is admissible |
| `FINDINGS.md` (both repos) | the rigor-gated results, positives and negatives |
