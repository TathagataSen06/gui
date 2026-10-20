"""External validation on public, de-identified research benchmarks.

The synthetic benchmark (``argus.evaluate``) measures Argus against networks it
plants itself, which says nothing about real-world efficacy. This package
re-runs Argus's *methods* on three public datasets that are standard in the
fraud and record-linkage literature and compares the result with published
numbers:

* **Elliptic** (Weber et al., 2019): Bitcoin transaction graph with illicit /
  licit labels. Exercises the graph features and the supervised models.
* **ULB credit-card fraud** (Dal Pozzolo et al., 2015): PCA-transformed card
  transactions. Exercises the supervised models under 0.17% prevalence.
* **DBLP-ACM** (Kopcke et al., 2010): bibliographic records from two sources.
  Exercises blocking, string comparators, the n-gram embedding and union-find.

None of these datasets identifies a person: Elliptic nodes are anonymized
transaction ids, the card data has no cardholder key, and DBLP-ACM lists
publications. Data is downloaded to ``data/external/`` on demand and never
committed. The person-level pipeline (``argus ingest`` onwards) still refuses
anything that is not synthetic; nothing here relaxes that guard.
"""
