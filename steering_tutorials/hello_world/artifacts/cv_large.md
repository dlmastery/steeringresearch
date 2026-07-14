# 5-Fold Cross-Validation — Safety Probe (Toxic-Chat)

Frozen-LLM activations `X` = **748x1152**, balanced ([374, 374]). CPU-only; the Gemma model is never loaded — we reuse `artifacts/features_large.npz`. This mean ± CI across folds is the TRUSTWORTHY headline (every example is held out exactly once).

Estimator: the deployed 3-layer `MLPProbe` (1152->128->32->1, dropout 0.3), trained with the exact `train_probe.py` recipe (Adam lr=0.001, weight_decay=0.001, BCE, early stop). StandardScaler fit per-fold on train only.

## MLP probe — mean ± 95% CI across folds

| metric | mean | std | 95% CI | single-split |
|---|---|---|---|---|
| accuracy | 0.9518 | 0.0270 | [0.9281, 0.9755] | 0.8750 |
| balanced_accuracy | 0.9518 | 0.0272 | [0.9280, 0.9756] | 0.8750 |
| precision | 0.9441 | 0.0419 | [0.9074, 0.9808] | 0.8621 |
| recall | 0.9626 | 0.0130 | [0.9512, 0.9740] | 0.8929 |
| specificity | 0.9409 | 0.0474 | [0.8994, 0.9825] | 0.8571 |
| f1 | 0.9529 | 0.0250 | [0.9310, 0.9748] | 0.8772 |
| mcc | 0.9046 | 0.0529 | [0.8582, 0.9509] | 0.7505 |
| cohen_kappa | 0.9036 | 0.0541 | [0.8562, 0.9511] | 0.7500 |
| roc_auc | 0.9839 | 0.0086 | [0.9764, 0.9914] | 0.9649 |
| pr_auc | 0.9819 | 0.0107 | [0.9725, 0.9912] | 0.9696 |
| log_loss | 0.1701 | 0.0634 | [0.1145, 0.2257] | 0.2481 |
| brier | 0.0385 | 0.0175 | [0.0232, 0.0539] | 0.0772 |

## Linear-probe reference — LogisticRegression, 5-fold CV

| metric | mean | std |
|---|---|---|
| accuracy | 0.9398 | 0.0229 |
| roc_auc | 0.9830 | 0.0106 |

## Trustworthy headline

Across 5 stratified folds the MLP probe scores **accuracy 0.952 ± 0.024** (95% CI [0.928, 0.976]) and **roc_auc 0.984 ± 0.008**. A plain logistic-regression linear probe reaches 0.940 accuracy / 0.983 roc_auc, so the harmful-vs-benign signal is strongly (and largely linearly) decodable from this layer.
