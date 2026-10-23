# Argus benchmark (multi-seed)

3 independent synthetic worlds of 1,500 people (seeds 42, 7, 2024), 182-day window, 7 planted networks each (2 shell-layering, 2 structuring, 2 mule-funnel, 1 insider), 21 benign look-alike decoys each, plus a clean control world per seed with no planted networks (scored with the model trained on the main world). Alert threshold: risk >= 45.

| Metric | Mean | Std | Per seed |
|---|---:|---:|---|
| ER pairwise precision (persons, excl. anonymous lines) | 0.996 | 0.0009 | 0.996, 0.998, 0.996 |
| ER pairwise recall (persons, excl. anonymous lines) | 0.927 | 0.0028 | 0.924, 0.925, 0.930 |
| ER pairwise F1 (persons, excl. anonymous lines) | 0.960 | 0.0012 | 0.959, 0.960, 0.962 |
| ER pairwise recall (all person records) | 0.885 | 0.0037 | 0.884, 0.881, 0.890 |
| Model CV ROC-AUC (labelled entities) | 0.982 | 0.0042 | 0.987, 0.977, 0.981 |
| Model CV average precision | 0.945 | 0.011 | 0.930, 0.956, 0.948 |
| Rules-only CV ROC-AUC | 0.971 | 0.0054 | 0.978, 0.965, 0.971 |
| Rules-only CV average precision | 0.792 | 0.029 | 0.773, 0.770, 0.834 |
| Logistic regression CV ROC-AUC | 0.966 | 0.0095 | 0.975, 0.971, 0.953 |
| Gradient boosting CV ROC-AUC | 0.981 | 0.0039 | 0.986, 0.977, 0.981 |
| Blend CV ROC-AUC | 0.978 | 0.0066 | 0.987, 0.977, 0.971 |
| Alert precision (entities) | 0.919 | 0.031 | 0.897, 0.898, 0.962 |
| Alert recall (entities incl. fragments) | 0.605 | 0.0050 | 0.598, 0.609, 0.607 |
| Ring-member recall (person level) | 0.890 | 0.0091 | 0.886, 0.902, 0.881 |
| Ring-member recall excl. watch-listed seeds | 0.905 | 0.014 | 0.919, 0.912, 0.886 |
| Rings detected (of 7) | 7.000 | 0.000 | 7.000, 7.000, 7.000 |
| Precision@25 | 1.000 | 0.000 | 1.000, 1.000, 1.000 |
| Clean-world person FPR | 0.0016 | 0.0006 | 0.0007, 0.0020, 0.0020 |
| Clean-world alerts | 2.333 | 0.943 | 1.000, 3.000, 3.000 |

## Member recall by typology

| Typology | Mean member recall | Rings |
|---|---:|---:|
| insider_collusion | 0.67 | 3 |
| mule_funnel | 0.97 | 6 |
| shell_layering | 0.75 | 6 |
| structuring | 0.91 | 6 |

## Member recall by role (all seeds)

| Typology | Role | Alerted / members |
|---|---|---:|
| insider_collusion | handler | 2 / 3 |
| insider_collusion | insider | 2 / 3 |
| mule_funnel | herder | 6 / 6 |
| mule_funnel | mule | 43 / 44 |
| shell_layering | facilitator | 0 / 6 |
| shell_layering | nominee | 16 / 16 |
| shell_layering | organizer | 5 / 6 |
| structuring | coordinator | 4 / 6 |
| structuring | smurf | 35 / 37 |

## Alert rate among benign look-alikes (main worlds)

| Decoy | Mean alert rate |
|---|---:|
| cash_business | 0.000 |
| freelancer | 0.000 |
| landlord | 0.000 |
| night_shift | 0.143 |
| rosca | 0.028 |
| scam victims | 0.019 |

Total runtime: 932s (generation, pipeline, evaluation and clean control for all seeds; one seed at a time, each in a fresh process).
