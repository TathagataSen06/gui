"""Published results that Argus's numbers are compared with.

Every entry was checked against its primary source (paper table, authors'
figure, or repository README). An entry is listed only under the protocol it
was measured with; numbers from different protocols are never put in one
table. ``note`` records what makes a comparison imperfect.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Baseline:
    dataset: str
    protocol: str
    method: str
    metrics: dict[str, float]
    source: str
    url: str
    note: str = ""


WEBER = "Weber et al., KDD'19 AD in Finance workshop, Table 1"
WEBER_URL = "https://arxiv.org/abs/1908.02591"
EPP = "Elmougy & Liu, KDD'23 (Elliptic++), authors' notebook, original Elliptic features"
EPP_URL = "https://github.com/git-disl/EllipticPlusPlus"
FDB = "Grover et al., Fraud Dataset Benchmark (arXiv:2208.14417), README results table"
FDB_URL = "https://github.com/amazon-science/fraud-dataset-benchmark"
SPE = "Liu et al., Self-paced Ensemble, ICDE 2020, re-sampling table (authors' figure)"
SPE_URL = "https://github.com/ZhiningLiu1998/self-paced-ensemble"
KOPCKE = "Kopcke, Thor & Rahm, PVLDB 2010, as tabulated by Zhang et al. (arXiv:1712.09691) Table 2"
KOPCKE_URL = "https://arxiv.org/abs/1712.09691"
ZEROER = "Wu et al., ZeroER, SIGMOD 2020, Table 3"
ZEROER_URL = "https://arxiv.org/abs/1908.06049"
DITTO = "Li et al., Ditto, PVLDB 14(1) 2021, Tables 5 and 10"
DITTO_URL = "https://arxiv.org/abs/2004.00584"
KASAI = "Kasai et al., ACL 2019, Tables 3 and 4"
KASAI_URL = "https://arxiv.org/abs/1906.08042"


def _w(method: str, p: float, r: float, f1: float, micro: float) -> Baseline:
    return Baseline("elliptic", "temporal_1_34", method, {"precision": p, "recall": r, "f1": f1, "micro_f1": micro}, WEBER, WEBER_URL,
                    "single unseeded run; AF includes the time step")


BASELINES: list[Baseline] = [
    # ---- Elliptic: train on time steps 1-34, test on 35-49, illicit class ----
    _w("Logistic regression (sklearn defaults), LF", 0.348, 0.668, 0.457, 0.920),
    _w("Logistic regression (sklearn defaults), AF", 0.404, 0.593, 0.481, 0.931),
    _w("Random forest (50 trees, 50 max features), LF", 0.803, 0.611, 0.694, 0.966),
    _w("Random forest (50 trees, 50 max features), AF", 0.956, 0.670, 0.788, 0.977),
    _w("Random forest, AF + GCN node embeddings (paper's best)", 0.971, 0.675, 0.796, 0.978),
    _w("MLP, AF", 0.694, 0.617, 0.653, 0.962),
    _w("GCN (2 layers)", 0.812, 0.512, 0.628, 0.961),
    Baseline("elliptic", "temporal_1_34", "EvolveGCN", {"precision": 0.850, "recall": 0.624, "f1": 0.720}, "Weber et al., Table 2", WEBER_URL),
    Baseline("elliptic", "temporal_1_34", "Random forest (50 trees), original features", {"precision": 0.940, "recall": 0.724, "f1": 0.818, "micro_f1": 0.979}, EPP, EPP_URL, "unseeded"),
    Baseline("elliptic", "temporal_1_34", "XGBoost (defaults), original features", {"precision": 0.812, "recall": 0.717, "f1": 0.761, "micro_f1": 0.971}, EPP, EPP_URL),
    # ---- Credit card: out-of-time 80/20 ----
    Baseline("creditcard", "out_of_time", "H2O AutoML", {"roc_auc": 0.992}, FDB, FDB_URL, "preprint; FDB's sort is not stable at tied Time values"),
    Baseline("creditcard", "out_of_time", "AutoGluon", {"roc_auc": 0.99}, FDB, FDB_URL, "reported to two decimals"),
    Baseline("creditcard", "out_of_time", "Auto-sklearn", {"roc_auc": 0.988}, FDB, FDB_URL),
    Baseline("creditcard", "out_of_time", "Amazon Fraud Detector (proprietary)", {"roc_auc": 0.985}, FDB, FDB_URL),
    # ---- Credit card: ~60% training set ----
    Baseline("creditcard", "random_60_40", "Logistic regression, no re-sampling", {"auprc": 0.587}, SPE, SPE_URL,
             "170,885 training rows; split procedure not published and probably not stratified"),
    Baseline("creditcard", "random_60_40", "Gradient boosting, 10 trees, no re-sampling", {"auprc": 0.803}, SPE, SPE_URL, "10-tree GBDT, not a full-size model"),
    Baseline("creditcard", "random_60_40", "Self-paced ensemble (10 x GBDT10)", {"auprc": 0.849}, SPE, SPE_URL),
    # ---- DBLP-ACM: whole problem scored against all 2,224 gold pairs ----
    Baseline("dblp_acm", "full_table", "FEBRL Fellegi-Sunter", {"f1": 0.962}, KOPCKE, KOPCKE_URL, "best configuration chosen with the ground truth, not a held-out score"),
    Baseline("dblp_acm", "full_table", "FEBRL SVM (best of five)", {"f1": 0.976}, KOPCKE, KOPCKE_URL, "best configuration chosen with the ground truth"),
    Baseline("dblp_acm", "full_table", "MARLIN SVM", {"f1": 0.974}, KOPCKE, KOPCKE_URL, "best configuration chosen with the ground truth"),
    Baseline("dblp_acm", "full_table", "v-signature (unsupervised)", {"precision": 0.977, "recall": 0.974, "f1": 0.976}, "Zhang et al., arXiv:1712.09691, Table 1", KOPCKE_URL, "best of a parameter sweep scored on the ground truth"),
    Baseline("dblp_acm", "full_table", "ZeroER (unsupervised)", {"f1": 0.96}, ZEROER, ZEROER_URL, "own LSH blocking; fit on all candidates"),
    Baseline("dblp_acm", "full_table", "Fellegi-Sunter via ECM (recordlinkage toolkit)", {"f1": 0.09}, ZEROER, ZEROER_URL, "binarized features; the paper attributes the failure to that"),
    Baseline("dblp_acm", "full_table", "Random forest on Magellan features", {"f1": 0.98}, ZEROER, ZEROER_URL, "supervised, scored on a random 50% of candidate pairs"),
    Baseline("dblp_acm", "full_table", "Logistic regression on Magellan features", {"f1": 0.96}, ZEROER, ZEROER_URL, "supervised, scored on a random 50% of candidate pairs"),
    # ---- DBLP-ACM: DeepMatcher pre-blocked 3:1:1 split, F1 on test ----
    Baseline("dblp_acm", "pre_blocked", "Magellan", {"f1": 0.984}, DITTO + " (as reported by Mudgal et al. 2018)", DITTO_URL),
    Baseline("dblp_acm", "pre_blocked", "SVM on Magellan features (best of 6 classical learners)", {"f1": 0.9835}, KASAI, KASAI_URL, "mean of 5 runs"),
    Baseline("dblp_acm", "pre_blocked", "DeepMatcher (Hybrid)", {"f1": 0.984}, DITTO + " (as reported by Mudgal et al. 2018)", DITTO_URL),
    Baseline("dblp_acm", "pre_blocked", "Deep ER, Kasai et al.", {"precision": 0.983, "recall": 0.986, "f1": 0.9845}, KASAI, KASAI_URL, "mean of 5 runs"),
    Baseline("dblp_acm", "pre_blocked", "Ditto (RoBERTa)", {"f1": 0.9899}, DITTO, DITTO_URL),
]


def for_protocol(dataset: str, protocol: str) -> list[Baseline]:
    return [b for b in BASELINES if b.dataset == dataset and b.protocol == protocol]
