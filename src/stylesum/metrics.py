"""Evaluation metrics: content (ROUGE) and style (readability, formality markers,
style-classifier accuracy)."""

from __future__ import annotations

import pickle
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import textstat
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .extractive import split_sentences

# --------------------------------------------------------------------------- #
# Content: ROUGE
# --------------------------------------------------------------------------- #
_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)


def rouge(preds: Sequence[str], refs: Sequence[str]) -> dict[str, float]:
    """Mean ROUGE F1 (x100). rougeLsum treats each sentence as a line."""
    assert len(preds) == len(refs)
    if not preds:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeLsum": 0.0}
    acc = {k: [] for k in ("rouge1", "rouge2", "rougeLsum")}
    for p, r in zip(preds, refs):
        s = _SCORER.score("\n".join(split_sentences(r)), "\n".join(split_sentences(p)))
        for k in acc:
            acc[k].append(s[k].fmeasure)
    return {k: round(100 * float(np.mean(v)), 2) for k, v in acc.items()}


# --------------------------------------------------------------------------- #
# Style: surface markers
# --------------------------------------------------------------------------- #
_CONTRACTION = re.compile(r"\b\w+'(s|t|re|ve|ll|d|m)\b", re.I)
_FIRST_PERSON = re.compile(r"\b(i|me|my|mine|i'm|i've|i'd|i'll)\b", re.I)
_WORD = re.compile(r"[A-Za-z']+")


def style_markers(text: str) -> dict[str, float]:
    words = _WORD.findall(text)
    n = max(len(words), 1)
    return {
        "words": len(words),
        "fkgl": textstat.flesch_kincaid_grade(text) if words else 0.0,
        "flesch": textstat.flesch_reading_ease(text) if words else 0.0,
        "avg_word_len": sum(len(w) for w in words) / n,
        "contractions_pct": 100 * len(_CONTRACTION.findall(text)) / n,
        "first_person_pct": 100 * len(_FIRST_PERSON.findall(text)) / n,
    }


def mean_markers(texts: Sequence[str]) -> dict[str, float]:
    rows = [style_markers(t) for t in texts]
    if not rows:
        return {}
    return {k: round(float(np.mean([r[k] for r in rows])), 2) for k in rows[0]}


# --------------------------------------------------------------------------- #
# Style: classifier ("does the output look like the requested style?")
# --------------------------------------------------------------------------- #
class StyleClassifier:
    """TF-IDF + logistic regression trained on the *reference summaries* of the
    training set. Style accuracy = fraction of generated summaries classified as
    the style that was requested (cf. style-transfer evaluation, e.g. Rao &
    Tetreault 2018)."""

    def __init__(self) -> None:
        self.pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                            max_features=50_000),
            LogisticRegression(max_iter=2000, C=4.0),
        )

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> StyleClassifier:
        self.pipe.fit(list(texts), list(labels))
        return self

    def predict(self, texts: Sequence[str]) -> list[str]:
        return list(self.pipe.predict(list(texts)))

    def accuracy(self, texts: Sequence[str], labels: Sequence[str]) -> float:
        if not texts:
            return 0.0
        pred = self.predict(texts)
        return round(100 * float(np.mean([p == y for p, y in zip(pred, labels)])), 2)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pipe, f)

    @classmethod
    def load(cls, path: str | Path) -> StyleClassifier:
        obj = cls()
        with open(path, "rb") as f:
            obj.pipe = pickle.load(f)  # noqa: S301 - our own artifact
        return obj
