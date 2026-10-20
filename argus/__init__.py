"""Argus: a synthetic-data investigative analytics benchmark.

The package is split into pipeline stages that mirror a real financial
intelligence / fraud analytics stack:

``generate``  -> fabricate a synthetic world (people, orgs, four noisy data feeds,
                 planted criminal networks and their ground truth)
``ingest``    -> map heterogeneous raw feeds into one common event schema (DuckDB)
``resolve``   -> entity resolution: blocking, pairwise matching, constrained clustering
``graph``     -> relationship graph, centrality, communities, money-flow cycles
``score``     -> explainable rule-based and machine-learned risk scores
``app``       -> Streamlit operator console
``evaluate``  -> benchmark metrics against the generator's ground truth

Research and fraud-detection benchmarking only. Not designed for, and must not
be used for, monitoring real individuals. See README.md.
"""

__version__ = "0.1.0"

SYNTHETIC_MARKER = "ARGUS-SYNTHETIC-v1"
