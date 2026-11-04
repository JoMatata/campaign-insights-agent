import numpy as np
import pytest

from src.text import classify, embed
from src.text.cluster import choose_k, top_terms
from src.text.embed import TfidfSvdEmbedder, get_embedder

TEXTS = ["mpesa payment failed again", "m-pesa stk push never came", "pesa imekatwa refund please",
         "rider never came with my order", "delivery took five days", "order imechelewa sana",
         "the juice was expired", "crisps were stale and broken", "packaging was leaking oil"] * 3


def test_falls_back_to_tfidf_when_model_unavailable(monkeypatch):
    def broken(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(embed, "SentenceTransformerEmbedder", broken)
    assert isinstance(get_embedder("auto"), TfidfSvdEmbedder)
    with pytest.raises(OSError):
        get_embedder("st")


def test_tfidf_vectors_are_normalised_and_meaningful():
    e = TfidfSvdEmbedder(dim=8).fit(TEXTS)
    X = e.encode(TEXTS[:9])
    assert np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-6)
    q = e.encode(["mpesa payment failed"])[0]
    assert np.argmax(X @ q) in (0, 1, 2)


def test_choose_k_finds_obvious_clusters():
    rng = np.random.default_rng(0)
    centres = np.eye(5)[:3] * 10
    X = np.vstack([c + rng.normal(0, 0.3, (40, 5)) for c in centres])
    k, scores = choose_k(X, k_range=range(2, 7))
    assert k == 3 and scores[3] > scores[5]


def test_top_terms_picks_distinctive_words():
    terms = top_terms(TEXTS[:3] * 3, TEXTS, n=5)
    assert "payment failed" in terms and not any("rider" in t for t in terms)


def test_classifiers_run_on_small_set():
    e = TfidfSvdEmbedder(dim=8).fit(TEXTS)
    y = np.array((["payment"] * 3 + ["delivery"] * 3 + ["quality"] * 3) * 3)
    X = e.encode(TEXTS)
    assert classify.score(y, classify.baseline_logreg(X, y))["macro_f1"] > 0.5
    assert len(classify.zero_shot(X, e)) == len(TEXTS)
