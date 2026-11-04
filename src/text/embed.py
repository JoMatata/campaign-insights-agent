"""Text embeddings behind one small interface.

SentenceTransformerEmbedder is the default. If the model cannot be loaded (no network,
no torch) the code falls back to TF-IDF + TruncatedSVD, which needs no download and
still gives dense vectors that work with the same search and clustering code.
"""
import logging
import os

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src import config

log = logging.getLogger(__name__)


class Embedder:
    name = "base"

    def fit(self, texts):
        return self

    def encode(self, texts) -> np.ndarray:
        """Return L2-normalised vectors, so a dot product is cosine similarity."""
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name=config.ST_MODEL):
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from sentence_transformers import SentenceTransformer  # heavy import, keep it lazy
        self.model = SentenceTransformer(model_name)
        self.name = model_name.split("/")[-1]

    def encode(self, texts):
        return self.model.encode(list(texts), batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=False)


class TfidfSvdEmbedder(Embedder):
    name = "tfidf-svd"

    def __init__(self, dim=128, seed=0):
        # char n-grams cope better with Sheng spellings and typos than word tokens alone
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                   sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=dim, random_state=seed)

    def fit(self, texts):
        self.svd.fit(self.vec.fit_transform(texts))
        return self

    def encode(self, texts):
        return normalize(self.svd.transform(self.vec.transform(texts)))


def get_embedder(kind=config.EMBEDDER):
    if kind in ("auto", "st"):
        try:
            return SentenceTransformerEmbedder()
        except Exception as e:  # missing package, no network, bad cache
            if kind == "st":
                raise
            log.warning("sentence-transformers unavailable (%s); using TF-IDF + SVD", e)
    return TfidfSvdEmbedder()
