"""Discourse-marker boost for boundary scores.

Embedding-only segmentation responds to *content* shifts and often misses
explicit transition cues like 'Let me switch gears' or '接下來我想換個話題'
because those sentences semantically blend with what comes after. We patch
that blind spot by adding a depth bonus at gaps preceding sentences that
begin with a known transition phrase.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

ZH_MARKERS = (
    "接下來", "接著", "另外", "首先", "其次", "再來", "再者",
    "最後", "總之", "換個話題", "順便", "至於", "話說", "對了",
    "說到", "反過來", "另一方面",
)

EN_MARKERS = (
    "let me switch", "let me turn", "let me move",
    "moving on", "next,", "now,",
    "first,", "second,", "third,", "fourth,", "finally,",
    "in conclusion", "to recap", "to sum up", "in summary",
    "speaking of", "on another note", "on a different note",
    "meanwhile,", "lastly,",
)

DEFAULT_MARKERS: tuple[str, ...] = ZH_MARKERS + EN_MARKERS


def find_marker_gaps(sentences, markers: Sequence[str] = DEFAULT_MARKERS) -> list[int]:
    """Return gap indices preceding sentences that begin with a marker phrase."""
    lowered = [m.lower() for m in markers]
    hits: list[int] = []
    for i, sent in enumerate(sentences):
        if i == 0:
            continue
        head = sent.text.lstrip().lower()
        if any(head.startswith(m) for m in lowered):
            hits.append(i - 1)
    return hits


def boost_depths(
    sentences,
    depths: np.ndarray,
    markers: Sequence[str] = DEFAULT_MARKERS,
    bonus: float | None = None,
) -> np.ndarray:
    """Add a bonus to ``depths`` at gaps preceding marker-tagged sentences.

    ``bonus`` defaults to one standard deviation of the depth distribution,
    which is enough to push a borderline-below-threshold gap above it without
    swamping naturally-deep valleys.
    """
    if len(depths) == 0:
        return depths
    if bonus is None:
        bonus = float(depths.std())
    boosted = depths.astype(np.float32, copy=True)
    for gap in find_marker_gaps(sentences, markers):
        if 0 <= gap < len(boosted):
            boosted[gap] = boosted[gap] + bonus
    return boosted


def parse_markers(text: str) -> tuple[str, ...]:
    """Parse a newline / comma-separated list of marker phrases."""
    parts: Iterable[str] = (
        part.strip() for line in text.splitlines() for part in line.split(",")
    )
    return tuple(p for p in parts if p)
