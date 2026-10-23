# Where Argus fails

This write-up is based on measured runs, not on intuition:

- three independent 1,500-person worlds (seeds 42, 7 and 2024), summarised in
  [`reports/benchmark.md`](../reports/benchmark.md);
- per-seed reports in `data/bench/seed_*/reports/evaluation.md`;
- targeted diagnostics written while building the pipeline.

Numbers quoted without a range are from seed 42.

## 1. Entity resolution

**What works.** Pairwise precision is about 0.996 and recall about 0.92 on
person records, excluding deliberately anonymous phone lines. Bank,
telco and badge records tie together well because they share phone numbers,
e-mail addresses or employers.

**Where it fails.**

| Failure | Evidence | Why |
|---|---|---|
| **DOB-less records with common names stay split** | 264-284 missed records per world. Almost all are classified as `weak_name_evidence`: personal mailboxes with only a first-name display name ("Nick"), badges of people called "Shannon Smith" whose KYC lacks an e-mail | E-mail and badge records carry no date of birth or address. When no identifier is shared, the only evidence is a name, and for common names the term-frequency-adjusted weight correctly stays below the threshold. Lowering the threshold merges namesakes. |
| **Anonymous lines and burners are unresolvable** | Recall drops from ~0.92 to ~0.88 when they are included | This is intended. A prepaid SIM bought with cash has no identity attributes. Real systems link them through co-travel (two phones always at the same cell), which this prototype deliberately does not implement. |
| **Remaining over-merges (6-10 clusters per world)** | e.g. two different "William B. Miller"s, two "Jonathan Whitaker"s; "Theresa Marsh" with a nickname record | Same first name, middle initial and surname, with DOB present on only one side; the cannot-link constraint can only act when both clusters have DOBs. |
| **Household dependence** | Before the household guard, 88 of 88 cannot-link refusals in a 400-person world were *different people* the pairwise scorer wanted to merge | Fellegi-Sunter assumes fields are independent. In a household, surname and address are correlated, so their weights double count. The guard (ignore address when first names conflict) cut refusals from 465 to 133 at 1,500 people. |
| **Work vs. personal e-mail** | Fixed. ER recall rose from 0.900 to 0.924 once "different mailbox *types*" stopped counting as disagreement | People hold several mailboxes; a naive comparison penalises every badge-to-bank link. |
| **Role mailboxes are not linked to their company** | `org_records` recall is 0 on a handful of pairs | `careers@<company>.example` is typed as a person-like mailbox and never compared with the company's bank record. |
| **u-probabilities are estimated on random pairs** | Without corroboration rules, "Tina Gonzalez" and "Tina Gonzales" linked with p = 0.96 | A fuzzy surname agreement is rare between random pairs but common inside a phonetic block. A proper EM fit on blocked pairs would be more principled than the corroboration rule. |

Errors propagate downstream. An over-merged entity inherits a ring member's
risk, so the namesake gets alerted. A split entity hides part of a subject's
behaviour: in seed 42, one shell-layering nominee was split into a scored
bank entity and an unscored phone-only entity.

## 2. Relationship graph

| Failure | Evidence | Why |
|---|---|---|
| **Louvain on the full social graph does not find rings** | In a 400-person world with all channels at resolution 1.0, the community holding the most members of a ring had 58-121 nodes. For 6 of 7 rings it held at most 5/8 of the members; the exception sat inside a 121-node community. | Ring members also have families, friends and coworkers, and those ties are stronger and more numerous than ring ties. This mirrors real investigations. |
| **Even on the investigative subgraph, rings are split** | Largest community share per ring is 0.13-1.00 across seeds; 3, 5 and 6 of 7 rings touch a flagged community (seeds 42, 7, 2024) | Shell rings pass money through companies controlled by *different* nominees, and burners create separate phone-only nodes. Community detection is a lead generator, not a detector. |
| **Community flags are relative** | Robust z-scores are computed against the other communities in the *same* world | In a world full of rings, or one with none, the baseline shifts. Absolute thresholds would be needed operationally. |
| **Most "circular payment paths" are innocent** | 77 temporal cycles, of which only 11 involve a ring and 4 have three or more hops; the cycle rule fires on 127 benign entities | Family loans repaid within days are value-preserving 2-cycles. Restricting the rule to cycles of three or more hops would cut false positives but miss round trips that collapse to two parties after merging companies into their beneficial owners. |
| **Approximate centrality** | Betweenness uses 64 sampled sources | Exact betweenness costs O(V·E) in pure Python; ranks of low-centrality nodes are noisy. |

## 3. Risk scoring

Member recall by role, summed over three seeds (from `reports/benchmark.md`):

| Typology | Role | Alerted / members |
|---|---|---:|
| mule_funnel | herder / mule | 6 / 6, 43 / 44 |
| structuring | coordinator / smurf | 4 / 6, 35 / 37 |
| shell_layering | nominee / organizer | 16 / 16, 5 / 6 |
| shell_layering | **facilitator (accountant)** | **0 / 6** |
| insider_collusion | insider / handler | 2 / 3, 2 / 3 |

| Failure | Evidence | Why / mitigation |
|---|---|---|
| **Peripheral roles without money are invisible** | The shell rings' accountant was never alerted (0 / 6) | Their footprint is a reconciliation e-mail and a phone call per cycle, with no money flowing through their accounts. Every money feature is zero, and comms features cannot separate them from any colleague. Network-propagation scoring (risk flowing from confirmed members to frequent contacts) would catch them, and would also pull in innocent relatives (see guilt by association below). |
| **The supervised model cannot learn rare typologies** | Without the rule floor the insider (1 ring = 2 positives) would score about **24/100** with five rules firing and model p ≈ 0.001 | With grouped CV, the insider's fold has no other insider example to learn from. The **rule floor** (`risk >= 0.8 x rules`) fixed it (the insider moved to 48), at the cost of 3 of the 6 benign alerts in seed 42. |
| **Rules alone are not enough** | Rules-only CV average precision 0.77-0.83 (mean 0.79), versus 0.93-0.96 for the chosen model | Benign behaviour trips individual rules constantly: 400 benign entities contact anonymous prepaid lines (R10), 558 contact high-risk entities (R09), 140 receive from many payers (R04). Only combinations are rare. |
| **Guilt by association** | The 2-pass "calls with high-risk entities" feature makes family of watch-listed people look risky. Scam victims are in heavy contact with the herder's burner and pay mules; in seed 42 their median rank is ~690 of ~1,950, the best ~96, and across seeds 1.9% of victims were alerted. | Victims sit far higher in the queue than their own behaviour warrants. In a real system that is a harm to the people the system should protect, not just a metric. |
| **Benign extremes** | Seed 42's six false alerts are ordinary people: four with circular payment paths (three of them alerted only through the rule floor) and two night-shift workers with after-hours restricted access, after-hours small-group meetings and calls to a prepaid line. With a third night-shift worker who also has a loan cycle, 3 of the 7 night-shift decoys are alerted (ranks 50-58, risk 45-51, model p ≈ 0.5), the pattern of the insider typology. The other seeds alert no night-shift worker; one ROSCA organiser is alerted in seed 7 (rank 57). | Each behaviour is innocent; the sum looks like a typology. Analyst dispositions (recorded by the console) are the intended feedback loop, but no model retraining from them is implemented. |
| **Labels are optimistic** | Every member of every ring is a confirmed case | Real case outcomes are partial, delayed and biased toward what investigators already look for. Here the model learns from a perfect label set, which inflates CV metrics. |
| **Look-ahead in the default scores** | With `--point-in-time` (train on the quarter before the first case opened, score only the latest quarter) cross-validated average precision falls from 0.94 to 0.85, member recall from 0.89 to 0.62 and rings detected from 7 to 5 (4-6) of 7; see `reports/benchmark_point_in_time.md` | The default lets every feature see all six months, including activity after the cases used as labels were opened. Part of the drop is the shorter window rather than look-ahead: rings that stop early barely show in the last quarter (mules 18 of 44, the insider 0 of 3), and recall among rings active at least 14 days in that quarter is 0.67. Labels also become noisier: a ring whose activity starts after the training quarter is labelled positive with no activity to learn from. |
| **Entity-level vs. person-level metrics differ** | Entity-level ROC-AUC of the final risk score is 0.81-0.83, while the chosen model's CV AUC is 0.98-0.99 | The entity-level evaluation also counts low-signal fragments of ring members, such as split-off mailboxes and burner entities, and the rule floor reorders the low end. Person-level ring-member recall (0.88-0.90) is the fairer headline. |

## 4. The benchmark itself

- **Circularity.** The same author wrote the typologies and the detectors.
  Features such as "calls within four hours before a large transfer" exist
  because the generator plants exactly that, so the benchmark measures
  consistency and robustness to noise, not real-world efficacy. An honest use
  is to plant a typology the detectors were *not* designed for and watch it
  fail.
- **Prevalence is unrealistic.** About 3-4% of labelled entities are positive,
  versus roughly 0.1% or less in real portfolios. At real prevalence, the same
  per-person false-positive rate (0.07-0.20% on the clean worlds, mean 0.16%)
  would give more false alerts than true ones: at 0.1% prevalence and 89%
  recall, roughly two of every three alerts would be false.
- **A static, cooperative world.** There is no concept drift, no adaptive
  adversaries, no account takeovers, no cross-border flows and no missing feed
  outages, and ring operational security is limited to burner discipline.
- **Simplified geography and formats.** One metro area, US formats and
  English names only.
- **Throughput.** A 1,500-person world (≈1.2 M events) takes about 3 minutes
  end to end and peaks at about 1.3 GB of RAM; building and scoring its clean
  control takes about 3 minutes more (306-316 s per seed for both in
  `reports/benchmark.json`, measured while the point-in-time benchmark ran
  alongside). Point-in-time scoring re-runs the graph stage on two snapshots
  and takes about 8 minutes per seed. Ingestion and generation are
  Python-bound; that is fine for a benchmark and not for production.
- **A memory bug, fixed.** Until a join-order fix, `argus evaluate
  --clean-control` peaked at 12.7 GB: written as a join condition, the
  self-call filter in the call-pair query let DuckDB plan a nested-loop join of
  ~60 million rows on clean worlds. The filter now runs after the query; the
  features are identical.

## 5. Things that look like failures but are by design

- Burner phones and anonymous prepaid lines remain separate "Unregistered
  line" entities. Linking them would require location co-travel analytics,
  which this project intentionally leaves out.
- The ingest guard rejects data that is not synthetic, even when it is
  well-formed. That is the point.

## 6. On public benchmark data

Measured with `argus realworld run`; tables, data provenance and published
baselines are in [`reports/realworld.md`](../reports/realworld.md).

| Failure | Evidence | Why |
|---|---|---|
| **Gradient boosting diverged at real-world class imbalance (fixed)** | On the ULB credit-card data (0.17% fraud), Argus's original gradient boosting scored ROC-AUC 0.571-0.857 over five random 60/40 splits and 0.907 out-of-time, while logistic regression scores 0.976-0.983 and 0.980. On the worst split, test ROC-AUC rises to 0.921 by tree 16, then collapses to 0.515 at tree 21 (0.545 at tree 25). The largest raw score grows from 23 in magnitude after the first tree to about 1.7e11, and the training loss rises from 0.013 to 7.4 million. The other split examined also diverges (raw scores near 1e116) but keeps more of its ranking (0.857). Reproduce with `scripts/diagnose_gbm_creditcard.py`. | Log-loss leaf values are Newton steps: summed residuals divided by summed p(1-p). At 0.17% prevalence, a small leaf holding a fraud among confidently scored legitimate rows has a near-zero denominator, and scikit-learn only guards against denominators below 1e-150. The configuration (250 trees, depth 3, learning rate 0.05, subsample 0.8, at least 5 samples per leaf) was chosen on synthetic entities with 3-4% positives and about 2,000 rows, where it is stable: the seed-42 model's raw scores stay within ±9.4 and its training loss falls from 0.22 to 0.0017 (lowest 0.0016, at tree 230). It was never tested at real prevalence, which is what this check is for. **Fix:** scikit-learn's `HistGradientBoostingClassifier` with an L2 leaf penalty of 0.1, chosen on synthetic data only (`scripts/select_booster.py`): the old booster's raw scores reached 2e9-1e82 on synthetic 0.17%-prevalence data, settings with a penalty of at least 0.1 stay near 12-15, and 0.1 had the best average precision on development worlds that are not benchmark seeds. On the card data it is now stable (ROC-AUC 0.965-0.982 across the five splits; on the worst split the largest raw score is 22 and the training loss falls to its lowest at the last tree). The cost on synthetic data was small (model average precision 0.954 to 0.939 before the blend below). |
| **Gradient boosting alone still does not hold up over time** | Out-of-time ROC-AUC 0.870 (95% CI 0.793-0.938) for the regularized booster, against 0.980 for logistic regression; on random splits the two are close (0.974 and 0.978). | The booster fits patterns that do not carry into the last hours of the data. Choosing the model on the latest part of the training period, as the harness now does, avoids it: that choice picked the blend of the two models, at 0.986 (0.976-0.996), inside the 0.985-0.992 AutoML range; alerting the top 0.1% of transactions (57) catches 48 of 75 frauds. The blend was adopted because it improved detection on synthetic development worlds, before it was run on this data. The test set holds 75 frauds, so the intervals are wide. |
| **The transaction scorer is all the card data can test** | No cardholder, merchant or device key | Entity resolution, graph features, rules and the risk blend cannot be exercised on it. |
| **Entity-resolution evidence comes from an easy, non-person benchmark** | DBLP-ACM: blocking kept all 2,224 true pairs; Fellegi-Sunter with no labels F1 0.989; supervised 0.988 (logistic regression) and 0.991 (gradient boosting); pre-blocked split test F1 0.988 (gradient boosting; logistic regression 0.979) | Published methods mostly score 0.96-0.99 on it. This validates Argus's Jaro-Winkler comparator, character n-gram embedding and supervised models on real records. The blocking (embedding neighbours plus rare title tokens), the EM-fitted Fellegi-Sunter matcher and the one-to-one linking are written for this benchmark in `argus/realworld/dblp_acm.py`; they are not the person resolver's `block`, `FellegiSunterScorer` or `ConstrainedUnionFind`, and the resolver's hand-set m-probabilities are not tested. One fix was made after the first run (consistent smoothing of u-probabilities); it moved the unsupervised F1 from 0.974 to 0.989 and is disclosed in the report. |
| **Graph features are unvalidated on real data** | Elliptic was not run: its host (`data.pyg.org`) is blocked in the environment the report was produced in | The harness, protocol and published baselines are in place; `argus realworld run --dataset elliptic` fills the gap once the data is reachable. |
| **Data came from mirrors** | Credit-card and DBLP-ACM files were fetched from public mirrors (GitLab's OpenML mirror, a GitHub copy of the Leipzig files) | The credit-card file matches its documented byte size and its row and fraud counts. The Leipzig files match the documented record and match counts; no byte size is documented for them. The pre-blocked split comes from the Ditto repository and is identified only by its SHA-256. Each file's SHA-256 is recorded. They are not the official downloads, which need a Kaggle login or blocked hosts. |
| **Multi-source fusion, typology rules and the risk blend remain untested on real data** | No public dataset links the bank, phone, badge and e-mail records of the same people | By design: such a dataset should not exist. |
