# Argus benchmark (multi-seed), point-in-time

3 independent synthetic worlds of 1,500 people (seeds 42, 7, 2024), 182-day window, 7 planted networks each (2 shell-layering, 2 structuring, 2 mule-funnel, 1 insider), 21 benign look-alike decoys each, plus a clean control world per seed with no planted networks (scored with the model trained on the main world). Alert threshold: risk >= 45.

**Point-in-time:** models are trained on features from the events before the first case opened and labelled by the cases opened afterwards; every score uses only days 91-182 of the window. Cross-validation predictions are made on those later features.

| Metric | Mean | Std | Per seed |
|---|---:|---:|---|
| ER pairwise precision (persons, excl. anonymous lines) | 0.996 | 0.0009 | 0.996, 0.998, 0.996 |
| ER pairwise recall (persons, excl. anonymous lines) | 0.927 | 0.0028 | 0.924, 0.925, 0.930 |
| ER pairwise F1 (persons, excl. anonymous lines) | 0.960 | 0.0012 | 0.959, 0.960, 0.962 |
| ER pairwise recall (all person records) | 0.885 | 0.0037 | 0.884, 0.881, 0.890 |
| Model CV ROC-AUC (labelled entities) | 0.948 | 0.0066 | 0.950, 0.939, 0.954 |
| Model CV average precision | 0.848 | 0.015 | 0.838, 0.836, 0.870 |
| Rules-only CV ROC-AUC | 0.909 | 0.020 | 0.903, 0.888, 0.935 |
| Rules-only CV average precision | 0.667 | 0.037 | 0.660, 0.626, 0.716 |
| Logistic regression CV ROC-AUC | 0.945 | 0.011 | 0.936, 0.939, 0.961 |
| Gradient boosting CV ROC-AUC | 0.952 | 0.0083 | 0.960, 0.940, 0.954 |
| Blend CV ROC-AUC | 0.953 | 0.0062 | 0.950, 0.948, 0.962 |
| Alert precision (entities) | 0.885 | 0.065 | 0.826, 0.854, 0.976 |
| Alert recall (entities incl. fragments) | 0.461 | 0.018 | 0.437, 0.471, 0.476 |
| Ring-member recall (person level) | 0.615 | 0.033 | 0.568, 0.634, 0.643 |
| Ring-member recall excl. watch-listed seeds | 0.586 | 0.059 | 0.513, 0.588, 0.657 |
| Rings detected (of 7) | 5.000 | 0.816 | 4.000, 5.000, 6.000 |
| Precision@25 | 0.947 | 0.075 | 0.840, 1.000, 1.000 |
| Clean-world person FPR | 0.0004 | 0.0006 | 0.000, 0.0013, 0.000 |
| Clean-world alerts | 3.333 | 0.943 | 4.000, 2.000, 4.000 |
| Rings active >= 14 days in the scoring window | 5.667 | 0.471 | 6.000, 6.000, 5.000 |
| Rings detected among active rings | 4.667 | 0.471 | 4.000, 5.000, 5.000 |
| Ring-member recall among active rings | 0.665 | 0.051 | 0.595, 0.714, 0.686 |

## Member recall by typology

| Typology | Mean member recall | Rings |
|---|---:|---:|
| insider_collusion | 0.33 | 3 |
| mule_funnel | 0.47 | 6 |
| shell_layering | 0.54 | 6 |
| structuring | 0.87 | 6 |

## Member recall by role (all seeds)

| Typology | Role | Alerted / members |
|---|---|---:|
| insider_collusion | handler | 2 / 3 |
| insider_collusion | insider | 0 / 3 |
| mule_funnel | herder | 6 / 6 |
| mule_funnel | mule | 18 / 44 |
| shell_layering | facilitator | 0 / 6 |
| shell_layering | nominee | 11 / 16 |
| shell_layering | organizer | 4 / 6 |
| structuring | coordinator | 4 / 6 |
| structuring | smurf | 33 / 37 |

## Alert rate among benign look-alikes (main worlds)

| Decoy | Mean alert rate |
|---|---:|
| cash_business | 0.000 |
| freelancer | 0.000 |
| landlord | 0.000 |
| night_shift | 0.143 |
| rosca | 0.028 |
| scam victims | 0.000 |

Total runtime: 1423s (generation, pipeline, evaluation and clean control for all seeds; one seed at a time, each in a fresh process).
