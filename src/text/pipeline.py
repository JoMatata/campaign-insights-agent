"""Run clustering and the classifier comparison, and save the metrics.

Run: python -m src.text.pipeline [--embedder auto|st|tfidf] [--llm]
"""
import argparse
import json
import os
import time

import pandas as pd

from src import config
from src.text import classify
from src.text.cluster import cluster_feedback
from src.text.index import FeedbackIndex, load_truth


def run(kind=config.EMBEDDER, use_llm=False):
    t0 = time.perf_counter()
    index = FeedbackIndex.load_or_build(kind=kind)
    emb = index.embedder
    truth = index.df[["feedback_id"]].merge(load_truth(), on="feedback_id", how="left")["theme"]
    llm = None
    if use_llm:
        from src.agent.llm import get_llm
        llm = get_llm()

    clusters = cluster_feedback(index.df["text"], index.vectors, truth=truth, llm=llm)

    labelled = pd.read_csv(config.LABELLED_PATH)
    X = emb.encode(labelled["text"].tolist())
    y = labelled["theme"].to_numpy()
    results = {"baseline_logreg_cv5": classify.score(y, classify.baseline_logreg(X, y)),
               "zero_shot_embedding": classify.score(y, classify.zero_shot(X, emb))}
    if llm is not None:
        results["llm_zero_shot"] = classify.score(y, classify.llm_classify(labelled["text"].tolist(), llm))

    out = {"embedder": emb.name, "n_feedback": len(index.df), "n_labelled": len(labelled),
           "llm_used": llm.name if llm else None, "clustering": clusters,
           "classification": results, "seconds": round(time.perf_counter() - t0, 1)}
    config.REPORTS_DIR.mkdir(exist_ok=True)
    path = config.REPORTS_DIR / f"text_results_{emb.name}.json"
    path.write_text(json.dumps(out, indent=2))
    return out, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default=config.EMBEDDER)
    ap.add_argument("--llm", action="store_true", help="also run the LLM classifier and labels")
    args = ap.parse_args()
    if args.llm and not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("--llm needs ANTHROPIC_API_KEY")
    out, path = run(args.embedder, args.llm)
    c = out["clustering"]
    print(f"embedder={out['embedder']}  k={c['k']}  silhouette={c['silhouette']:.3f}  "
          f"ARI vs themes={c['adjusted_rand_index']:.3f}")
    for cl in c["clusters"]:
        print(f"  [{cl['size']:4d}] {cl['majority_theme']:<13} purity={cl['purity']:.2f}  "
              f"{', '.join(cl['top_terms'][:6])}")
    for name, r in out["classification"].items():
        print(f"{name:<22} macro-F1={r['macro_f1']:.3f}  acc={r['accuracy']:.3f}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
