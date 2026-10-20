"""Optional dense-embedding similarity for record pairs.

The default embedder is LSA over character n-grams (TF-IDF -> truncated SVD).
It is deterministic, CPU-only and captures what token-level comparisons miss
(re-ordered tokens, spacing, partial typos across fields). Any object with an
``embed(texts) -> np.ndarray`` method (e.g. a sentence-transformer wrapper)
can be swapped in via :class:`Embedder`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class CharNgramEmbedder:
    """Character n-gram LSA embedding (unit-normalized rows)."""

    def __init__(self, dim: int = 48, ngram_range: tuple[int, int] = (2, 4), seed: int = 0) -> None:
        self.dim = dim
        self.ngram_range = ngram_range
        self.seed = seed

    def embed(self, texts: list[str]) -> np.ndarray:
        texts = [t if t.strip() else "_" for t in texts]
        tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=self.ngram_range, min_df=2, sublinear_tf=True).fit_transform(texts)
        k = max(2, min(self.dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
        dense = TruncatedSVD(n_components=k, random_state=self.seed).fit_transform(tfidf)
        return normalize(dense)


def pair_cosine(emb: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Cosine similarity for each row ``(i, j)`` of ``pairs``."""
    if len(pairs) == 0:
        return np.zeros(0)
    return np.einsum("ij,ij->i", emb[pairs[:, 0]], emb[pairs[:, 1]])
