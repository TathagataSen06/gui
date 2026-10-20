# Argus on public benchmark data

Argus's models (unchanged configuration) and entity-resolution building blocks, run on three public research datasets and compared with published results, under the same protocol where one exists; each table says how the published protocols differ. None of the datasets identifies a person. The person-level pipeline (`argus ingest` onwards) still accepts only synthetic data.

Where Argus falls short on this data is analyzed in [`docs/FAILURE_MODES.md`](../docs/FAILURE_MODES.md#6-on-public-benchmark-data).

Environment: Python 3.13.16, scikit-learn 1.9.1, x86_64. Regenerate with `argus realworld run`.

| Dataset | Tests | Status |
|---|---|---|
| Elliptic Data Set (Bitcoin transaction graph) | graph features, supervised models | **not run** (data host not reachable) |
| Credit Card Fraud Detection (ULB Machine Learning Group / Worldline) | supervised models | measured |
| DBLP-ACM bibliographic record linkage benchmark (Leipzig) | blocking, comparators, embedding, Fellegi-Sunter, clustering | measured |

## Elliptic Bitcoin transactions: graph features and models

**Not run in this environment.** Could not obtain elliptic_txs_features.csv for elliptic.

To run it, allow one of these hosts in the environment's network policy: `data.pyg.org`; or place the files by hand: Download the three CSVs from the Kaggle page (free account) and place elliptic_txs_features.csv, elliptic_txs_edgelist.csv and elliptic_txs_classes.csv in data/external/elliptic/. Then run `argus realworld run --dataset elliptic`.

No result is reported for this dataset.


## ULB credit-card fraud: supervised models at 0.17% prevalence

Data: Credit Card Fraud Detection (ULB Machine Learning Group / Worldline). License: Open Database License (ODbL) v1.0 / Database Contents License (DbCL) v1.0.

- `creditcard.csv`: 150,828,752 bytes, sha256 `a57f43b36bd695c8...`, from <https://gitlab.com/data/d/openml/1597/-/raw/master/dataset/tables/data.csv> (mirror)

284,807 transactions, 492 frauds (0.173%). Features: V1-V28 and log(1 + Amount); `Time` only orders rows.

**Out-of-time** (sorted by Time; first 80% train, last 20% test): 56,962 test rows, 75 frauds. 95% bootstrap intervals.

| Model | ROC-AUC | AUPRC | Source |
|---|---:|---:|---|
| **Argus** logistic regression | 0.980 (0.963-0.993) | 0.750 (0.658-0.839) | measured here |
| **Argus** gradient boosting | 0.907 (0.848-0.958) | 0.669 (0.551-0.786) | measured here |
| H2O AutoML | 0.992 | - | [Grover et al., Fraud Dataset Benchmark (arXiv:2208.14417), README results table](https://github.com/amazon-science/fraud-dataset-benchmark) (preprint; FDB's sort is not stable at tied Time values) |
| AutoGluon | 0.990 | - | [Grover et al., Fraud Dataset Benchmark (arXiv:2208.14417), README results table](https://github.com/amazon-science/fraud-dataset-benchmark) (reported to two decimals) |
| Auto-sklearn | 0.988 | - | [Grover et al., Fraud Dataset Benchmark (arXiv:2208.14417), README results table](https://github.com/amazon-science/fraud-dataset-benchmark) |
| Amazon Fraud Detector (proprietary) | 0.985 | - | [Grover et al., Fraud Dataset Benchmark (arXiv:2208.14417), README results table](https://github.com/amazon-science/fraud-dataset-benchmark) |

Alert budgets on the out-of-time test set (top-scored transactions alerted):

| Model | Budget | Alerts | Frauds caught | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Argus logistic regression | 0.1% | 57 | 46 | 0.807 | 0.613 |
| Argus logistic regression | 0.5% | 285 | 63 | 0.221 | 0.840 |
| Argus gradient boosting | 0.1% | 57 | 47 | 0.825 | 0.627 |
| Argus gradient boosting | 0.5% | 285 | 58 | 0.204 | 0.773 |

**Random split** (stratified random 60% train / 40% test, 5 seeds), mean and standard deviation over seeds:

| Model | AUPRC | ROC-AUC | Source |
|---|---:|---:|---|
| **Argus** logistic regression | 0.776 ± 0.015 | 0.978 ± 0.003 | measured here |
| **Argus** gradient boosting | 0.578 ± 0.089 | 0.777 ± 0.104 | measured here |
| Logistic regression, no re-sampling | 0.587 | - | [Liu et al., Self-paced Ensemble, ICDE 2020, re-sampling table (authors' figure)](https://github.com/ZhiningLiu1998/self-paced-ensemble) (170,885 training rows; split procedure not published and probably not stratified) |
| Gradient boosting, 10 trees, no re-sampling | 0.803 | - | [Liu et al., Self-paced Ensemble, ICDE 2020, re-sampling table (authors' figure)](https://github.com/ZhiningLiu1998/self-paced-ensemble) (10-tree GBDT, not a full-size model) |
| Self-paced ensemble (10 x GBDT10) | 0.849 | - | [Liu et al., Self-paced Ensemble, ICDE 2020, re-sampling table (authors' figure)](https://github.com/ZhiningLiu1998/self-paced-ensemble) |

Argus gradient boosting is unstable on this data: its ROC-AUC ranged from 0.571 to 0.857 across the seeds above (per seed: 0.827, 0.812, 0.857, 0.818, 0.571).

Caveats:

- Only the transaction scorer is tested: the data has no cardholder, merchant or device key, so no entity resolution or graph features.
- The data spans two days, so 'out-of-time' means a roughly ten-hour hold-out, not drift over months.
- The out-of-time test set has about 75 frauds, hence the wide intervals. Random-split numbers are not comparable with out-of-time ones.

Citation: A. Dal Pozzolo, O. Caelen, R. A. Johnson and G. Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. IEEE SSCI, 2015.


## DBLP-ACM: entity resolution

Data: DBLP-ACM bibliographic record linkage benchmark (Leipzig). License: No explicit license published; distributed by the Database Group, University of Leipzig, for entity-resolution research.

- `DBLP2.csv`: 405,262 bytes, sha256 `0acd567508a21be8...`, from <https://raw.githubusercontent.com/skogler/systemds-amls-project-data/master/DBLP-ACM/DBLP2.csv> (mirror)
- `ACM.csv`: 366,372 bytes, sha256 `572f5ae4d40f64b6...`, from <https://raw.githubusercontent.com/skogler/systemds-amls-project-data/master/DBLP-ACM/ACM.csv> (mirror)
- `DBLP-ACM_perfectMapping.csv`: 69,253 bytes, sha256 `689d5cccce00f27f...`, from <https://raw.githubusercontent.com/skogler/systemds-amls-project-data/master/DBLP-ACM/DBLP-ACM_perfectMapping.csv> (mirror)
- `pre_blocked/train.txt`: 2,894,612 bytes, sha256 `4f757d2d5dae4f7d...`, from <https://raw.githubusercontent.com/megagonlabs/ditto/master/data/er_magellan/Structured/DBLP-ACM/train.txt> (official)
- `pre_blocked/valid.txt`: 966,837 bytes, sha256 `b13444a2bf1a503e...`, from <https://raw.githubusercontent.com/megagonlabs/ditto/master/data/er_magellan/Structured/DBLP-ACM/valid.txt> (official)
- `pre_blocked/test.txt`: 961,602 bytes, sha256 `ded891e469b1d08e...`, from <https://raw.githubusercontent.com/megagonlabs/ditto/master/data/er_magellan/Structured/DBLP-ACM/test.txt> (official)

2,616 DBLP and 2,294 ACM records, 2,224 true matches.

**Whole problem.** Blocking kept 100,762 of 6,001,104 pairs (reduction 0.983) and 100.0% of the true matches. Precision and recall are over all 2,224 true matches, so blocking misses count.

| Matcher | Precision | Recall | F1 | Labels | Source |
|---|---:|---:|---:|---|---|
| **Argus** Fellegi-Sunter (u from random pairs, m by EM) | 0.987 | 0.992 | 0.989 | none | measured here |
| **Argus** logistic regression | 0.995 | 0.980 | 0.988 | 5-fold out-of-fold, grouped by DBLP record | measured here |
| **Argus** gradient boosting | 0.995 | 0.982 | 0.988 | 5-fold out-of-fold, grouped by DBLP record | measured here |

Published results on the whole problem (not on the pre-blocked split). Their protocols differ from this one and from each other; the notes say how:

| Method | Precision | Recall | F1 | Source |
|---|---:|---:|---:|---|
| FEBRL Fellegi-Sunter | - | - | 0.962 | [Kopcke, Thor & Rahm, PVLDB 2010, as tabulated by Zhang et al. (arXiv:1712.09691) Table 2](https://arxiv.org/abs/1712.09691) (best configuration chosen with the ground truth, not a held-out score) |
| FEBRL SVM (best of five) | - | - | 0.976 | [Kopcke, Thor & Rahm, PVLDB 2010, as tabulated by Zhang et al. (arXiv:1712.09691) Table 2](https://arxiv.org/abs/1712.09691) (best configuration chosen with the ground truth) |
| MARLIN SVM | - | - | 0.974 | [Kopcke, Thor & Rahm, PVLDB 2010, as tabulated by Zhang et al. (arXiv:1712.09691) Table 2](https://arxiv.org/abs/1712.09691) (best configuration chosen with the ground truth) |
| v-signature (unsupervised) | 0.977 | 0.974 | 0.976 | [Zhang et al., arXiv:1712.09691, Table 1](https://arxiv.org/abs/1712.09691) (best of a parameter sweep scored on the ground truth) |
| ZeroER (unsupervised) | - | - | 0.960 | [Wu et al., ZeroER, SIGMOD 2020, Table 3](https://arxiv.org/abs/1908.06049) (own LSH blocking; fit on all candidates) |
| Fellegi-Sunter via ECM (recordlinkage toolkit) | - | - | 0.090 | [Wu et al., ZeroER, SIGMOD 2020, Table 3](https://arxiv.org/abs/1908.06049) (binarized features; the paper attributes the failure to that) |
| Random forest on Magellan features | - | - | 0.980 | [Wu et al., ZeroER, SIGMOD 2020, Table 3](https://arxiv.org/abs/1908.06049) (supervised, scored on a random 50% of candidate pairs) |
| Logistic regression on Magellan features | - | - | 0.960 | [Wu et al., ZeroER, SIGMOD 2020, Table 3](https://arxiv.org/abs/1908.06049) (supervised, scored on a random 50% of candidate pairs) |

**Pre-blocked split** (pre-blocked candidate pairs (DeepMatcher/Ditto split 3:1:1); threshold chosen on valid, F1 on test): 7,417 / 2,473 / 2,473 pairs.

| Model | Precision | Recall | F1 | Source |
|---|---:|---:|---:|---|
| **Argus** logistic regression (threshold 0.65; F1 at 0.5: 0.970) | 0.973 | 0.984 | 0.979 | measured here |
| **Argus** gradient boosting (threshold 0.70; F1 at 0.5: 0.987) | 0.989 | 0.991 | 0.990 | measured here |
| Magellan | - | - | 0.984 | [Li et al., Ditto, PVLDB 14(1) 2021, Tables 5 and 10 (as reported by Mudgal et al. 2018)](https://arxiv.org/abs/2004.00584) |
| SVM on Magellan features (best of 6 classical learners) | - | - | 0.984 | [Kasai et al., ACL 2019, Tables 3 and 4](https://arxiv.org/abs/1906.08042) (mean of 5 runs) |
| DeepMatcher (Hybrid) | - | - | 0.984 | [Li et al., Ditto, PVLDB 14(1) 2021, Tables 5 and 10 (as reported by Mudgal et al. 2018)](https://arxiv.org/abs/2004.00584) |
| Deep ER, Kasai et al. | 0.983 | 0.986 | 0.985 | [Kasai et al., ACL 2019, Tables 3 and 4](https://arxiv.org/abs/1906.08042) (mean of 5 runs) |
| Ditto (RoBERTa) | - | - | 0.990 | [Li et al., Ditto, PVLDB 14(1) 2021, Tables 5 and 10](https://arxiv.org/abs/2004.00584) |

Caveats:

- The entity is a publication, not a person. Author names are public bibliographic metadata, used only as a string field and never written to any report.
- Field normalization (venue aliases, HTML entities) was written for this dataset; the Argus resolver's person-specific comparators do not apply to bibliographic records.
- DBLP-ACM is an easy benchmark; most published methods score above 0.96 F1. It shows the components work on real records, not that they are state of the art.
- One change was made after the first run on this data: levels never seen among random pairs had been given an arbitrary u of 1e-6 instead of the same additive smoothing as the other levels. Fixing that moved the unsupervised Fellegi-Sunter F1 from 0.974 to the value above; the supervised and pre-blocked results did not change.

Citation: H. Kopcke, A. Thor and E. Rahm. Evaluation of entity resolution approaches on real-world match problems. PVLDB 3(1), 2010. Pre-blocked split: S. Mudgal et al., Deep Learning for Entity Matching, SIGMOD 2018; serialized by Y. Li et al., Ditto, PVLDB 14(1), 2021.
