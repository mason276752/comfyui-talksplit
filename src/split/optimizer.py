from __future__ import annotations

import numpy as np

from .boundary import threshold_for_sensitivity


def optimize_boundaries(
    depths: np.ndarray,
    min_sentences: int = 2,
    max_sentences: int = 15,
    sensitivity: float = 1.0,
    target_paragraphs: int | None = None,
    threshold: float | None = None,
) -> list[int]:
    """Pick gap indices to split on.

    A *gap* index ``g`` means "split between sentence g and g+1".

    The DP maximizes the sum of depths at chosen gaps subject to every
    paragraph having between ``min_sentences`` and ``max_sentences`` sentences.
    If the depth threshold filters out everything that satisfies length, the
    candidate set is relaxed to all gaps so length constraints still hold.

    If ``threshold`` is given it overrides the sensitivity-derived value —
    useful when ``depths`` has been post-processed (e.g. marker-boosted) and
    the threshold should reflect the *natural* distribution rather than the
    boosted one.
    """
    n_gaps = len(depths)
    n_sentences = n_gaps + 1
    if n_sentences <= min_sentences:
        return []

    if target_paragraphs is not None:
        return _target_count_dp(depths, min_sentences, max_sentences, target_paragraphs)

    if threshold is None:
        threshold = threshold_for_sensitivity(depths, sensitivity)
    candidates = {i for i, d in enumerate(depths) if d >= threshold}

    splits = _length_constrained_dp(depths, min_sentences, max_sentences, candidates)
    if splits is None:
        # relax: any gap is fair game so we at least respect length
        splits = _length_constrained_dp(
            depths, min_sentences, max_sentences, set(range(n_gaps))
        )
    return splits or []


def _length_constrained_dp(
    depths: np.ndarray,
    min_len: int,
    max_len: int,
    candidates: set[int],
) -> list[int] | None:
    n_gaps = len(depths)
    n_sentences = n_gaps + 1

    f = np.full(n_gaps, -np.inf, dtype=np.float64)
    parent = np.full(n_gaps, -2, dtype=int)  # -1 = first split (no predecessor), -2 = unset

    for g in range(n_gaps):
        if g not in candidates:
            continue
        first_len = g + 1
        if min_len <= first_len <= max_len:
            f[g] = depths[g]
            parent[g] = -1
        lo = max(0, g - max_len)
        hi = g - min_len
        for gp in range(lo, hi + 1):
            if f[gp] == -np.inf:
                continue
            cand = f[gp] + depths[g]
            if cand > f[g]:
                f[g] = cand
                parent[g] = gp

    best_g, best_score = -1, -np.inf
    if min_len <= n_sentences <= max_len:
        best_score = 0.0  # zero splits is a valid outcome
    for g in range(n_gaps):
        if f[g] == -np.inf:
            continue
        last_len = n_sentences - 1 - g
        if min_len <= last_len <= max_len and f[g] > best_score:
            best_score = f[g]
            best_g = g

    if best_score == -np.inf:
        return None
    if best_g == -1:
        return []
    splits: list[int] = []
    g = best_g
    while g >= 0:
        splits.append(int(g))
        g = parent[g]
    splits.reverse()
    return splits


def _target_count_dp(
    depths: np.ndarray,
    min_len: int,
    max_len: int,
    target: int,
) -> list[int]:
    """Pick exactly ``target - 1`` splits maximizing summed depth, length-constrained."""
    n_gaps = len(depths)
    n_sentences = n_gaps + 1
    n_splits = target - 1
    if n_splits <= 0:
        return []
    if n_splits > n_gaps:
        return list(range(n_gaps))

    NEG = -np.inf
    # f[s][g] = best depth sum using s splits ending at gap g
    f = np.full((n_splits + 1, n_gaps), NEG, dtype=np.float64)
    parent = np.full((n_splits + 1, n_gaps), -2, dtype=int)

    for g in range(n_gaps):
        if min_len <= g + 1 <= max_len:
            f[1, g] = depths[g]
            parent[1, g] = -1

    for s in range(2, n_splits + 1):
        for g in range(n_gaps):
            lo = max(0, g - max_len)
            hi = g - min_len
            best = NEG
            best_gp = -2
            for gp in range(lo, hi + 1):
                if f[s - 1, gp] == NEG:
                    continue
                v = f[s - 1, gp] + depths[g]
                if v > best:
                    best = v
                    best_gp = gp
            f[s, g] = best
            parent[s, g] = best_gp

    best_g, best_score = -1, NEG
    for g in range(n_gaps):
        if f[n_splits, g] == NEG:
            continue
        last_len = n_sentences - 1 - g
        if min_len <= last_len <= max_len and f[n_splits, g] > best_score:
            best_score = f[n_splits, g]
            best_g = g
    if best_g == -1:
        # constraints unsatisfiable — pick top-k by depth
        return sorted(np.argsort(-depths)[:n_splits].tolist())

    splits: list[int] = []
    g, s = best_g, n_splits
    while g >= 0 and s >= 1:
        splits.append(int(g))
        g = parent[s, g]
        s -= 1
    splits.reverse()
    return splits
