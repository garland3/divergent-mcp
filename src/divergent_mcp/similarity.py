"""Similarity between ideas, with no heavy dependencies.

Two sources: a lexical TF-IDF model that always works offline, and optional dense
embeddings. Dense vectors are mean-centred before cosine, which removes the part
every idea shares because they are all about the same topic and makes one
threshold usable across embedding models.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from itertools import pairwise

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    ["a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to", "in", "on", "at", "by", "for", "with", "from", "into", "over", "under", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those", "their", "there", "they", "them", "you", "your", "we", "our", "can", "could", "would", "should", "will", "may", "might", "must", "not", "no", "do", "does", "did", "has", "have", "had", "each", "every", "any", "all", "more", "most", "very", "just", "also", "than", "which", "who", "whom", "what", "when", "where", "how", "why", "while", "about", "up", "out", "off", "only", "own", "same", "such", "too", "use", "uses", "used", "using", "make", "makes", "made", "get", "gets", "one", "two"]
)

Matrix = list[list[float]]


def _stem(word: str) -> str:
    for suffix in ("ingly", "edly", "ing", "ies", "ed", "es", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokens(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


def _features(text: str) -> Counter[str]:
    toks = tokens(text)
    feats: Counter[str] = Counter(toks)
    feats.update(f"{a}_{b}" for a, b in pairwise(toks))
    return feats


def lexical_matrix(texts: list[str]) -> Matrix:
    """Pairwise cosine over TF-IDF unigram+bigram vectors fitted on the pool itself."""
    n = len(texts)
    feats = [_features(t) for t in texts]
    df: Counter[str] = Counter()
    for f in feats:
        df.update(f.keys())
    vecs: list[dict[str, float]] = []
    for f in feats:
        v = {k: (1.0 + math.log(c)) * (math.log((1 + n) / (1 + df[k])) + 1.0) for k, c in f.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({k: x / norm for k, x in v.items()})
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        sim[i][i] = 1.0
        for j in range(i + 1, n):
            a, b = (vecs[i], vecs[j]) if len(vecs[i]) <= len(vecs[j]) else (vecs[j], vecs[i])
            s = sum(x * b[k] for k, x in a.items() if k in b)
            sim[i][j] = sim[j][i] = s
    return sim


def dense_matrix(vectors: list[list[float]]) -> Matrix:
    """Pairwise cosine over mean-centred dense vectors."""
    n = len(vectors)
    if n == 0:
        return []
    dim = len(vectors[0])
    mean = [sum(v[d] for v in vectors) / n for d in range(dim)] if n > 2 else [0.0] * dim
    centred = []
    for v in vectors:
        c = [x - m for x, m in zip(v, mean)]
        norm = math.sqrt(sum(x * x for x in c)) or 1.0
        centred.append([x / norm for x in c])
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        sim[i][i] = 1.0
        ci = centred[i]
        for j in range(i + 1, n):
            s = sum(map(float.__mul__, ci, centred[j]))
            sim[i][j] = sim[j][i] = s
    return sim


def combine(lexical: Matrix, dense: Matrix | None, lex_threshold: float, dense_threshold: float) -> Matrix:
    """One matrix on a common scale where 1.0 means "at the duplicate threshold".

    Each source is divided by its own threshold and the larger wins, so two ideas
    count as duplicates if either the wording or the meaning says so.
    """
    n = len(lexical)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            v = lexical[i][j] / lex_threshold
            if dense is not None:
                v = max(v, dense[i][j] / dense_threshold)
            out[i][j] = max(0.0, v)
    return out


def cluster(sim: Matrix, order: list[int] | None = None, threshold: float = 1.0) -> list[list[int]]:
    """Greedy leader clustering: an item joins the first cluster whose leader it duplicates."""
    order = list(range(len(sim))) if order is None else order
    clusters: list[list[int]] = []
    for i in order:
        for c in clusters:
            if sim[i][c[0]] >= threshold:
                c.append(i)
                break
        else:
            clusters.append([i])
    return clusters


def isolation(sim: Matrix, members: list[int], k: int = 5) -> dict[int, float]:
    """How far each member sits from its k nearest neighbours, rank-scaled to 0..1."""
    if len(members) <= 1:
        return {m: 1.0 for m in members}
    raw = {}
    for i in members:
        near = sorted((sim[i][j] for j in members if j != i), reverse=True)[:k]
        raw[i] = -sum(near) / len(near)
    ranked = sorted(members, key=lambda m: raw[m])
    return {m: r / (len(ranked) - 1) for r, m in enumerate(ranked)}
