# 5-Fold Cross-Validation — Safety Probe

Frozen-LLM activations `X` = **200x1152**, balanced ([100, 100]). CPU-only; the Gemma model is never loaded — we reuse `artifacts/features.npz`.

Estimator: the deployed 3-layer `MLPProbe` (1152->128->32->1, dropout 0.3), trained with the exact `train_probe.py` recipe (Adam lr=0.001, weight_decay=0.001, BCE, early stop on a stratified val slice). StandardScaler fit per-fold on train only.

## MLP probe — mean ± 95% CI across folds

| metric | mean | std | 95% CI | single-split |
|---|---|---|---|---|
| accuracy | 0.8700 | 0.0292 | [0.8444, 0.8956] | 0.9500 |
| balanced_accuracy | 0.8700 | 0.0292 | [0.8444, 0.8956] | 0.9500 |
| precision | 0.8455 | 0.0458 | [0.8053, 0.8856] | 0.9500 |
| recall | 0.9100 | 0.0374 | [0.8772, 0.9428] | 0.9500 |
| specificity | 0.8300 | 0.0600 | [0.7774, 0.8826] | 0.9500 |
| f1 | 0.8753 | 0.0266 | [0.8520, 0.8987] | 0.9500 |
| mcc | 0.7448 | 0.0566 | [0.6951, 0.7944] | 0.9000 |
| cohen_kappa | 0.7400 | 0.0583 | [0.6889, 0.7911] | 0.9000 |
| roc_auc | 0.9415 | 0.0157 | [0.9277, 0.9553] | 0.9800 |
| pr_auc | 0.9346 | 0.0273 | [0.9106, 0.9585] | 0.9795 |
| log_loss | 0.4202 | 0.1439 | [0.2941, 0.5463] | 0.1723 |
| brier | 0.0994 | 0.0282 | [0.0746, 0.1241] | 0.0509 |

## Linear-probe reference — LogisticRegression, 5-fold CV

| metric | mean | std |
|---|---|---|
| accuracy | 0.8600 | 0.0374 |
| roc_auc | 0.9435 | 0.0203 |

## Was standard practice followed? Is 0.95 trustworthy?

Yes — standard 5-fold stratified cross-validation was run, so every one of the 200 examples is held out exactly once and the headline now carries a confidence interval instead of resting on a single 40-example slice. Across folds the MLP probe scores **accuracy 0.870 ± 0.026** (95% CI [0.844, 0.896]) and **roc_auc 0.941 ± 0.014**. The single-split accuracy of 0.95 sits **above** the CI — the single split was optimistic (a lucky test draw). A plain logistic-regression linear probe reaches 0.860 ± 0.037 accuracy / 0.944 roc_auc, confirming the harmful-vs-benign signal is strongly (and largely linearly) decodable from this layer — the MLP is not overfitting to one split. Treat the single-split headline with caution: it lands outside the cross-validated confidence interval, so prefer the CV mean±CI as the reportable number.
