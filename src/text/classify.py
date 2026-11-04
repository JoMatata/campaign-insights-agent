"""Topic classification: a trained baseline against zero-shot and LLM alternatives.

All three are scored on the same 300 labelled rows. The baseline uses 5-fold
cross-validation so it is never scored on rows it was trained on.
"""
import json
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# short descriptions for zero-shot; English plus the Swahili a customer might use
LABELS = {
    "delivery": "late delivery, order has not arrived, rider lost, imechelewa, mzigo haujafika",
    "pricing": "too expensive, price increase, discount or promo code not applied, bei iko juu, ghali",
    "payment": "M-Pesa payment failed, STK push, money deducted, refund, pesa imekatwa",
    "quality": "product expired, stale, leaking, damaged or bad taste, quality imeshuka",
    "ad_annoyance": "annoying ad, seeing the same ad too often, stop showing me ads, ad inaboo",
    "service": "customer care does not respond, rude support agent, no reply, hawashiki simu",
    "positive": "happy customer, great quality, fast delivery, will order again, poa sana, fiti",
}


def score(y_true, y_pred):
    return {"macro_f1": round(f1_score(y_true, y_pred, average="macro"), 3),
            "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "per_class_f1": {k: round(v["f1-score"], 3) for k, v in classification_report(
                y_true, y_pred, output_dict=True, zero_division=0).items() if k in LABELS}}


def baseline_logreg(X, y, seed=0):
    clf = LogisticRegression(max_iter=2000, C=2.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_predict(clf, X, y, cv=cv)


def zero_shot(X, embedder):
    names = list(LABELS)
    L = embedder.encode(list(LABELS.values()))
    return np.array(names)[np.argmax(X @ L.T, axis=1)]


def llm_classify(texts, llm, batch=25):
    """Ask the LLM for one label per line, in batches. Unparseable answers become 'unknown'."""
    preds = []
    labels = ", ".join(LABELS)
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        numbered = "\n".join(f"{j}. {t}" for j, t in enumerate(chunk))
        reply = llm.complete(
            f"Classify each customer feedback snippet into exactly one of: {labels}. "
            "Pick the main complaint if there are several. Swahili and Sheng may appear.\n"
            f'Reply with JSON only: {{"labels": [...]}} with {len(chunk)} items in order.\n\n{numbered}')
        try:
            got = json.loads(re.search(r"\{.*\}", reply, re.DOTALL).group(0))["labels"]
        except (AttributeError, KeyError, json.JSONDecodeError):
            got = []
        got = [g if g in LABELS else "unknown" for g in got][:len(chunk)]
        preds += got + ["unknown"] * (len(chunk) - len(got))
    return np.array(preds)
