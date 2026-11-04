"""In-memory vector index over the feedback table.

3,000 snippets fit in a numpy array, so a brute-force dot product is fast enough
and much easier to explain than a vector database.
"""
import joblib
import numpy as np
import pandas as pd

from src import config
from src.agent.sql_guard import connect
from src.text.embed import get_embedder


class FeedbackIndex:
    def __init__(self, df, embedder, vectors):
        self.df = df.reset_index(drop=True)
        self.embedder = embedder
        self.vectors = vectors

    @classmethod
    def build(cls, df, embedder=None):
        embedder = embedder or get_embedder()
        embedder.fit(df["text"].tolist())
        return cls(df, embedder, embedder.encode(df["text"].tolist()))

    @classmethod
    def load_or_build(cls, db_path=config.DB_PATH, cache=config.INDEX_PATH, kind=config.EMBEDDER):
        con = connect(db_path)
        df = con.execute("SELECT * FROM feedback ORDER BY feedback_id").df()
        con.close()
        embedder = get_embedder(kind)
        if cache.exists():
            saved = joblib.load(cache)
            if saved["name"] == embedder.name and saved["n"] == len(df):
                # the transformer model is not pickled; TF-IDF must be, since it is fitted
                return cls(df, saved.get("embedder") or embedder, saved["vectors"])
        obj = cls.build(df, embedder)
        joblib.dump({"name": embedder.name, "n": len(df), "vectors": obj.vectors,
                     "embedder": embedder if embedder.name == "tfidf-svd" else None}, cache)
        return obj

    def search(self, query, k=10, county=None, campaign_id=None, source=None,
               start_date=None, end_date=None):
        scores = self.vectors @ self.embedder.encode([query])[0]
        mask = np.ones(len(self.df), dtype=bool)
        for col, val in (("county", county), ("campaign_id", campaign_id), ("source", source)):
            if val:
                mask &= (self.df[col] == val).to_numpy()
        dates = self.df["created_at"].astype(str)
        if start_date:
            mask &= (dates >= start_date).to_numpy()
        if end_date:
            mask &= (dates <= end_date).to_numpy()
        scores = np.where(mask, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        top = top[np.isfinite(scores[top])]
        out = self.df.iloc[top].copy()
        out["score"] = scores[top].round(3)
        return out


def load_truth():
    return pd.read_csv(config.TRUTH_PATH)
