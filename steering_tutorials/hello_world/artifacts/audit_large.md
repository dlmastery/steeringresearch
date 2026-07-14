# Large-set (Toxic-Chat) leakage & length-confound audit

Dataset: Toxic-Chat balanced (n=748), split train=524/val=112/test=112, seed=0. Natural base rate ~7% toxic (we evaluate on a balanced set for readable metrics).
Deployed probe on this test set: acc=0.875 / auc=0.965.

## Check 0 — split disjointness & duplicates
- cross-split prompt overlaps: 0
- exact-duplicate prompts in dataset: 0
- label balance: train {'harmful(1)': 262, 'benign(0)': 262}, val {'harmful(1)': 56, 'benign(0)': 56}, test {'harmful(1)': 56, 'benign(0)': 56}
- VERDICT: PASS — splits disjoint, no cross-split prompt leaks

## Check 1 — label-shuffle control on activations
- TRUE labels:     acc=0.911, auc=0.976
- SHUFFLED labels: acc=0.580, auc=0.557  (must collapse; fail if acc>0.65)
- VERDICT: PASS — true=0.911, shuffled=0.580 ~chance; no features->label leakage path

## Check 2 — surface baselines (the length confound)
- median length: harmful 180 chars vs benign 56 chars (3.2x)
- deployed probe:      acc=0.875, auc=0.965
- length-only [char,word]: acc=0.643, auc=0.728
- TF-IDF (1,2)-gram:       acc=0.741, auc=0.857 (4727 feats)
- probe lift over length: +0.232 acc; over TF-IDF: +0.134 acc

## Check 3 — length-matched (within comparable-length bins)
| bin | char-len range | n | harm/benign | probe acc | length-only acc | probe−length |
|---|---|---|---|---|---|---|
| 1 | 13–46 | 28 | 8/20 | 0.893 | 0.714 | +0.179 |
| 2 | 46–90 | 28 | 11/17 | 0.857 | 0.607 | +0.250 |
| 3 | 90–296 | 28 | 15/13 | 0.821 | 0.429 | +0.393 |
| 4 | 296–1535 | 28 | 22/6 | 0.929 | 0.821 | +0.107 |

- middle (overlap-length) bins: probe 0.839 vs length-only 0.518 (+0.321)

## Overall verdict
LEGITIMATE — no leakage (shuffle collapses to chance, splits disjoint). Toxic prompts ARE ~3.2x longer, and a length-only baseline is non-trivial (acc 0.643), so length is a PARTIAL confound. BUT within comparable-length (middle) bins the probe scores 0.839 vs the length-only baseline's 0.518 (+0.321) — the probe reads toxicity, not merely length.

Flags: none