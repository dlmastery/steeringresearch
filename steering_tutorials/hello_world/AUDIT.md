# Paper Audit — `hello_world` (Safety Probe)

Independent paper/verification audit. Auditor did not modify lesson code or the README.

## Cited paper
- Alain & Bengio 2016, "Understanding intermediate layers using linear classifier probes", arXiv:1610.01644.

## Checks

| Check | Verdict | Evidence |
|---|---|---|
| 1. Paper real + attribution correct | **PASS** | WebFetch of `arxiv.org/abs/1610.01644` confirms exact title "Understanding intermediate layers using linear classifier probes", authors **Guillaume Alain, Yoshua Bengio**, 2016. Method = linear classifiers ("probes") trained independently on each frozen layer to measure what features are linearly decodable — exactly how the README cites it. The README/`probe.py` still tag it `[UNVERIFIED]`; it is now **verified** and that tag can be dropped. |
| 2. Method fidelity | **PASS (with honest simplification)** | `probe.py` trains a 3-layer MLP (1152→128→32→1) on frozen layer-12 mean-pooled activations; `model_utils.extract_features` uses a forward hook + `@torch.no_grad`, so the LLM is genuinely frozen. This is a shallow-MLP variant of Alain-Bengio's *linear* probe; the README states the deviation plainly ("a hair fancier than linear") and adds a plain logistic-regression **linear** probe as the faithful reference. |
| 3. Claim accuracy | **PASS** | The central claim ("harm is *largely linearly decodable* from a mid-layer residual stream") is the Alain-Bengio thesis, not an overclaim. The load-bearing evidence is that the logreg linear probe (0.860 / 0.944) essentially ties the MLP (0.870 / 0.941) — correctly presented as *support* for linear decodability, not as a novel SOTA result. |
| 4. Results honesty | **PASS** | Numbers reconcile with artifacts: `metrics.json` single-split acc 0.95 / ROC-AUC 0.98, confusion `[[19,1],[1,19]]`; `cv_report.json` 5-fold acc **0.870 ± 0.026**, ROC-AUC **0.9415 ± 0.0138**, logreg 0.860 / 0.9435; `ood_metrics.json` XSTest acc 0.69, ROC-AUC 0.8885, recall 0.4267, confusion `[[143,7],[86,64]]` — all match the README. Screening-tier (n=200, single seed) is stated; the lucky single-split 0.950 is explicitly flagged as sitting *above* the CV 95% CI ("quote the CV mean, not the point estimate"); OOD recall collapse (0.427) is reported as prominently as the wins. |

## Note on the 1B self-judge flag
Not applicable here. This lesson uses the abliterated Gemma-3-1B purely as a **frozen feature extractor** for a probe — there is **no LLM-judge** in the loop, so the known 1B self-judge refusal-inflation effect (flagged in the FLAS audit, relevant to lesson 2) cannot affect these numbers. Labels come from JailbreakBench's prompt-level ground truth, not from a model.

## Overall verdict
The single cited paper is real and correctly attributed (drop the `[UNVERIFIED]` tag). The implementation is a faithful, honestly-labeled shallow-MLP extension of linear probing, the claim is not overstated, and every headline number reconciles with the committed artifacts with the small-n and calibration caveats stated up front. No FAIL or CONCERN flags.

Internal QA pass — independent external review pending (auditor shares a model family with the author).
