# Argus benchmark (multi-seed), point-in-time

3 independent synthetic worlds of 1,500 people (seeds 42, 7, 2024), 182-day window, 7 planted networks each (2 shell-layering, 2 structuring, 2 mule-funnel, 1 insider), 21 benign look-alike decoys each, plus a clean control world per seed with no planted networks (scored with the model trained on the main world). Alert threshold: risk >= 45.

**Point-in-time:** models are trained on features from the events before the first case opened and labelled only by outcomes decided by the end of the window (confirmed cases closed by then; cases still open are unlabelled); every score uses only days 92-182 of the window (day 1 = the first day). Cross-validation predictions are made on those later features.

| Metric | Mean | Std | Per seed |
|---|---:|---:|---|
| ER pairwise precision (persons, excl. anonymous lines) | 0.996 | 0.0009 | 0.996, 0.998, 0.996 |
| ER pairwise recall (persons, excl. anonymous lines) | 0.927 | 0.0028 | 0.924, 0.925, 0.930 |
| ER pairwise F1 (persons, excl. anonymous lines) | 0.960 | 0.0012 | 0.959, 0.960, 0.962 |
| ER pairwise recall (all person records) | 0.885 | 0.0037 | 0.884, 0.881, 0.890 |
| Model CV ROC-AUC (labelled entities) | 0.814 | 0.096 | 0.680, 0.860, 0.903 |
| Model CV average precision | 0.476 | 0.125 | 0.309, 0.608, 0.513 |
| Rules-only CV ROC-AUC | 0.903 | 0.022 | 0.905, 0.876, 0.929 |
| Rules-only CV average precision | 0.552 | 0.073 | 0.454, 0.578, 0.626 |
| Logistic regression CV ROC-AUC | 0.778 | 0.091 | 0.672, 0.768, 0.894 |
| Gradient boosting CV ROC-AUC | 0.849 | 0.048 | 0.780, 0.877, 0.888 |
| Blend CV ROC-AUC | 0.814 | 0.096 | 0.680, 0.860, 0.903 |
| Alert precision (entities) | 0.975 | 0.035 | 1.000, 0.926, 1.000 |
| Alert recall (entities incl. fragments) | 0.326 | 0.040 | 0.310, 0.287, 0.381 |
| Ring-member recall (person level) | 0.479 | 0.063 | 0.523, 0.390, 0.524 |
| Ring-member recall excl. watch-listed seeds | 0.489 | 0.056 | 0.540, 0.412, 0.514 |
| Rings detected (of 7) | 4.000 | 0.816 | 4.000, 3.000, 5.000 |
| Precision@25 | 0.973 | 0.038 | 1.000, 0.920, 1.000 |
| Clean-world person FPR | 0.000 | 0.000 | 0.000, 0.000, 0.000 |
| Clean-world alerts | 0.000 | 0.000 | 0.000, 0.000, 0.000 |
| Rings scheduled >= 14 days in the scoring window | 5.667 | 0.471 | 6.000, 6.000, 5.000 |
| Rings detected among active rings | 3.667 | 0.471 | 4.000, 3.000, 4.000 |
| Ring-member recall among active rings | 0.535 | 0.059 | 0.548, 0.457, 0.600 |

## Member recall by typology

| Typology | Mean member recall | Rings |
|---|---:|---:|
| insider_collusion | 0.17 | 3 |
| mule_funnel | 0.32 | 6 |
| shell_layering | 0.48 | 6 |
| structuring | 0.71 | 6 |

## Member recall by role (all seeds)

| Typology | Role | Alerted / members |
|---|---|---:|
| insider_collusion | handler | 1 / 3 |
| insider_collusion | insider | 0 / 3 |
| mule_funnel | herder | 4 / 6 |
| mule_funnel | mule | 13 / 44 |
| shell_layering | facilitator | 0 / 6 |
| shell_layering | nominee | 9 / 16 |
| shell_layering | organizer | 4 / 6 |
| structuring | coordinator | 0 / 6 |
| structuring | smurf | 30 / 37 |

## Alert rate among benign look-alikes (main worlds)

| Decoy | Mean alert rate |
|---|---:|
| cash_business | 0.000 |
| freelancer | 0.000 |
| landlord | 0.000 |
| night_shift | 0.000 |
| rosca | 0.028 |
| scam victims | 0.000 |

Total runtime: 1628s (generation, pipeline, evaluation and clean control for all seeds; one seed at a time, each in a fresh process).
