from __future__ import annotations

import numpy as np


def cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Adjacent-pair cosine similarity. Embeddings must be L2-normalized."""
    if len(embeddings) < 2:
        return np.zeros(0, dtype=np.float32)
    return np.sum(embeddings[:-1] * embeddings[1:], axis=1).astype(np.float32)


def block_similarity(embeddings: np.ndarray, block_size: int = 1) -> np.ndarray:
    """Gap-level similarity using mean block embeddings (TextTiling block comparison).

    For each gap i, compare the mean of the ``block_size`` sentences immediately
    before vs the ``block_size`` sentences immediately after.  When block_size=1
    this is identical to ``cosine_similarity``.  Larger values smooth out
    sentence-level noise and reveal gradual topic shifts that span several
    sentences — useful for articles or longer presentations where topic
    transitions are not marked by a single sharp sentence boundary.

    Embeddings must be L2-normalised (as produced by ``Embedder.embed``).
    """
    if block_size <= 1:
        return cosine_similarity(embeddings)
    n = len(embeddings)
    if n < 2:
        return np.zeros(0, dtype=np.float32)
    sim = np.zeros(n - 1, dtype=np.float32)
    for i in range(n - 1):
        left = embeddings[max(0, i - block_size + 1): i + 1]
        right = embeddings[i + 1: min(n, i + 1 + block_size)]
        lv = left.mean(axis=0)
        rv = right.mean(axis=0)
        ln = float(np.linalg.norm(lv))
        rn = float(np.linalg.norm(rv))
        if ln > 0 and rn > 0:
            sim[i] = float(np.dot(lv / ln, rv / rn))
    return sim


def depth_scores(sim: np.ndarray) -> np.ndarray:
    """TextTiling-style depth score for each gap.

    For gap i, walk outward until the similarity stops climbing on each side;
    the highest values seen are the local peaks. depth = (left_peak - sim[i])
    + (right_peak - sim[i]). Higher depth = stronger boundary candidate.
    """
    n = len(sim)
    depths = np.zeros(n, dtype=np.float32)
    for i in range(n):
        left_peak = sim[i]
        for j in range(i - 1, -1, -1):
            if sim[j] >= left_peak:
                left_peak = sim[j]
            else:
                break
        right_peak = sim[i]
        for j in range(i + 1, n):
            if sim[j] >= right_peak:
                right_peak = sim[j]
            else:
                break
        depths[i] = (left_peak - sim[i]) + (right_peak - sim[i])
    return depths


def threshold_for_sensitivity(depths: np.ndarray, sensitivity: float) -> float:
    """Map sensitivity (0–2) to a depth threshold via mean + k*std.

    sensitivity 1.0 → k=1.0 (cuts deep valleys only, ~conservative)
    sensitivity 2.0 → k=0.0 (cuts at every above-mean valley, aggressive)
    sensitivity 0.5 → k=1.5 (very conservative)
    """
    if len(depths) == 0:
        return 0.0
    k = max(0.0, 2.0 - sensitivity)
    return float(depths.mean() + k * depths.std())
