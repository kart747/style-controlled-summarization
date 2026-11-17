"""Extractive baselines: Lead-k and TextRank.

Extractive methods copy sentences from the source, so by construction they
*cannot* change style: the same input always yields the same summary whatever
register is requested. They are the reference point that motivates the
abstractive, prefix-controlled model.
"""

from __future__ import annotations

import re

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Split after . ! ? when followed by whitespace + capital/quote/digit.
# Avoid splitting on common abbreviations.
_ABBREVS = ["Mr", "Mrs", "Ms", "Dr", "Prof", "St", "vs", "e.g", "i.e", "et al", "Fig", "No"]
_ABBREV = "".join(rf"(?<!\b{re.escape(a)}\.)" for a in _ABBREVS)
_SENT_RE = re.compile(r"(?<=[.!?])" + _ABBREV + r"\s+(?=[\"'(\[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def lead_k(text: str, k: int = 3) -> str:
    return " ".join(split_sentences(text)[:k])


def textrank(text: str, k: int = 3, min_words: int = 4) -> str:
    """Graph-based ranking (Mihalcea & Tarau, 2004) over TF-IDF cosine similarity.
    Selected sentences are returned in their original order."""
    sents = split_sentences(text)
    cand = [i for i, s in enumerate(sents) if len(s.split()) >= min_words]
    if len(cand) <= k:
        return " ".join(sents[i] for i in cand) or " ".join(sents[:k])
    try:
        tfidf = TfidfVectorizer(stop_words="english").fit_transform([sents[i] for i in cand])
    except ValueError:  # only stop-words
        return lead_k(text, k)
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0.0)
    scores = nx.pagerank(nx.from_numpy_array(sim), max_iter=200)
    top = sorted(sorted(scores, key=scores.get, reverse=True)[:k])
    return " ".join(sents[cand[i]] for i in top)


EXTRACTIVE = {"lead3": lead_k, "textrank": textrank}
