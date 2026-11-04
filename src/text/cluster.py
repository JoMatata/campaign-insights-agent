"""KMeans clustering of feedback embeddings, with silhouette-based k and term labels."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score

# template fillers and common Swahili function words that say nothing about the topic
STOP = list(ENGLISH_STOP_WORDS | {
    "honestly", "tausi", "bana", "hi", "wueh", "pls", "please", "sort", "fix", "kwa", "kweli",
    "na", "ni", "sana", "sio", "hii", "also", "plus", "will", "order", "days", "kes", "mara",
})


def top_terms(texts, background, n=8):
    """Terms that are more frequent in `texts` than in the whole corpus."""
    vec = TfidfVectorizer(stop_words=STOP, min_df=3, ngram_range=(1, 2),
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]+\b")
    vec.fit(background)
    sub = np.asarray(vec.transform(texts).mean(axis=0)).ravel()
    bg = np.asarray(vec.transform(background).mean(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    return [terms[i] for i in np.argsort(-(sub - bg))[:n]]


def representative(vectors, n=5):
    """Indices of the rows closest to the centroid."""
    centroid = vectors.mean(axis=0)
    return list(np.argsort(-(vectors @ centroid))[:n])


def choose_k(X, k_range=range(4, 13), seed=0):
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        scores[k] = float(silhouette_score(X, labels, metric="cosine"))
    return max(scores, key=scores.get), scores


def cluster_feedback(texts, X, truth=None, k=None, llm=None, seed=0):
    texts = list(texts)
    best_k, scores = choose_k(X, seed=seed) if k is None else (k, {})
    km = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(X)
    clusters = []
    for c in range(best_k):
        idx = np.where(km.labels_ == c)[0]
        members = [texts[i] for i in idx]
        info = {"cluster": c, "size": len(idx), "top_terms": top_terms(members, texts),
                "examples": [members[i] for i in representative(X[idx], n=3)]}
        if truth is not None:
            vals, counts = np.unique(np.asarray(truth)[idx], return_counts=True)
            info["majority_theme"] = str(vals[counts.argmax()])
            info["purity"] = round(counts.max() / len(idx), 3)
        if llm is not None:
            info["llm_label"] = llm.complete(
                "Give a 2-5 word label for this group of customer feedback. Reply with the label only.\n"
                + "\n".join(members[:15])).strip()
        clusters.append(info)
    out = {"k": best_k, "silhouette_by_k": scores,
           "silhouette": float(silhouette_score(X, km.labels_, metric="cosine")),
           "clusters": sorted(clusters, key=lambda c: -c["size"])}
    if truth is not None:
        out["adjusted_rand_index"] = float(adjusted_rand_score(truth, km.labels_))
    return out
