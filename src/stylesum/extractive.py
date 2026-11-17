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


def textrank_scores(sents: list[str]) -> list[float]:
    """PageRank centrality of each sentence in a TF-IDF cosine-similarity graph
    (Mihalcea & Tarau, 2004). Falls back to a positional prior when the graph is
    degenerate."""
    n = len(sents)
    prior = [1.0 / (i + 1) for i in range(n)]
    if n < 2:
        return prior
    try:
        tfidf = TfidfVectorizer(stop_words="english").fit_transform(sents)
    except ValueError:  # only stop-words
        return prior
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0.0)
    if not sim.any():
        return prior
    pr = nx.pagerank(nx.from_numpy_array(sim), max_iter=200)
    return [pr[i] for i in range(n)]


def textrank(text: str, k: int = 3, min_words: int = 4) -> str:
    """Top-k TextRank sentences, returned in their original order."""
    sents = split_sentences(text)
    cand = [i for i, s in enumerate(sents) if len(s.split()) >= min_words]
    if len(cand) <= k:
        return " ".join(sents[i] for i in cand) or " ".join(sents[:k])
    scores = textrank_scores([sents[i] for i in cand])
    top = sorted(sorted(range(len(cand)), key=scores.__getitem__, reverse=True)[:k])
    return " ".join(sents[cand[i]] for i in top)


def compress_to_budget(text: str, count_tokens, budget: int) -> str:
    """Hybrid extract-then-abstract pre-step for long documents.

    If ``text`` exceeds ``budget`` tokens, greedily keep the highest-TextRank
    sentences that still fit, then restore document order. ``count_tokens`` is a
    callable ``str -> int`` (e.g. based on the model's tokenizer)."""
    if count_tokens(text) <= budget:
        return text
    sents = split_sentences(text)
    scores = textrank_scores(sents)
    lengths = [count_tokens(s) for s in sents]
    keep, used = [], 0
    for i in sorted(range(len(sents)), key=scores.__getitem__, reverse=True):
        if used + lengths[i] <= budget:
            keep.append(i)
            used += lengths[i]
    if not keep:  # a single huge sentence: hard-truncate by words
        words = text.split()
        while words and count_tokens(" ".join(words)) > budget:
            words = words[: int(len(words) * 0.9)]
        return " ".join(words)
    return " ".join(sents[i] for i in sorted(keep))


EXTRACTIVE = {"lead3": lead_k, "textrank": textrank}
