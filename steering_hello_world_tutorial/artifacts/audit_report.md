# Safety-probe leakage & artifact audit

Dataset: JailbreakBench harmful vs. benign (n=200), split train=130/val=30/test=40, seed=0.
Headline under audit: MLP probe acc=0.95 / auc=0.98 (metrics.json).

## Check 1 -- split disjointness & duplicates
- cross-split prompt overlaps: 0 (train/test=0, train/val=0, val/test=0)
- exact-duplicate prompts in full dataset: 0
- label balance: train {'harmful(1)': 65, 'benign(0)': 65}, val {'harmful(1)': 15, 'benign(0)': 15}, test {'harmful(1)': 20, 'benign(0)': 20}
- VERDICT: PASS -- splits are disjoint at both prompt and index level

## Check 2 -- label-shuffle (permutation) control on activations
- TRUE labels:     test acc=0.925, auc=0.968
- SHUFFLED labels: test acc=0.425, auc=0.480  (fail if acc>0.65)
- VERDICT: PASS -- true=0.925 high, shuffled=0.425 ~chance; no features->label leakage path

## Check 3 -- trivial text-confound baselines (no activations)
- length-only [char,word]: acc=0.650, auc=0.647
- TF-IDF (1,2)-gram:       acc=0.575, auc=0.635 (424 feats)
- Trivial text baselines stay well below the probe -- the probe's score is not explained by surface text features alone.
- VERDICT: OK -- text-only baselines (tfidf 0.575) trail the probe

## Check 4 -- scaler discipline
- train_probe.py step 4: Scaler.fit(X[tr]) then transform train/val/test with that train-fit scaler
- VERDICT: PASS -- standardization is fit on the train split only; no test statistics leak into the scaler.

## Overall verdict
LEGITIMATE -- no leakage and no dominant text artifact; the headline reflects real activation-level signal.

Flags: none