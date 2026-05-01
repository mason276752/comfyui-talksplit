"""Tests for discourse markers and clause-level splitting."""
from __future__ import annotations

import numpy as np

from split.markers import (
    DEFAULT_MARKERS,
    boost_depths,
    find_marker_gaps,
    parse_markers,
)
from split.segmenter import SOFT_COMMA, split_sentences


def test_find_marker_gaps_zh():
    text = "今天天氣不錯。我去散步了。接下來我想換個話題。談談電影。"
    sents = split_sentences(text)
    # sentences: 0 today, 1 walk, 2 next-let-me-switch, 3 movies
    gaps = find_marker_gaps(sents)
    assert 1 in gaps  # gap before sentence 2 starting with "接下來"


def test_find_marker_gaps_en():
    text = "Today the weather is nice. I went for a walk. Let me switch gears. I want to talk about cinema."
    sents = split_sentences(text)
    gaps = find_marker_gaps(sents)
    assert 1 in gaps  # gap before "Let me switch gears"


def test_boost_depths_only_at_marker_gaps():
    text = "甲一。甲二。接下來我想談談乙。乙一。乙二。"
    sents = split_sentences(text)
    depths = np.array([0.1, 0.05, 0.2, 0.05], dtype=np.float32)
    boosted = boost_depths(sents, depths)
    # Sentence 2 starts with "接下來" → gap 1 should be boosted
    assert boosted[1] > depths[1]
    # other gaps untouched
    assert boosted[0] == depths[0]
    assert boosted[3] == depths[3]


def test_boost_depths_no_markers_passes_through():
    sents = split_sentences("一。二。三。四。")
    depths = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    boosted = boost_depths(sents, depths)
    np.testing.assert_array_equal(boosted, depths)


def test_clause_level_splits_at_commas():
    text = "他覺得很開心，因為今天天氣很好。"
    plain = split_sentences(text)
    clauses = split_sentences(text, extra_terminators=SOFT_COMMA)
    assert len(plain) == 1
    assert len(clauses) == 2
    assert "開心" in clauses[0].text
    assert "天氣" in clauses[1].text


def test_clause_level_keeps_decimals():
    text = "Pi is about 3.14, which is irrational."
    clauses = split_sentences(text, extra_terminators=SOFT_COMMA)
    # decimal in 3.14 not split because '.' is followed by digit; only comma splits
    assert any("3.14" in s.text for s in clauses)


def test_parse_markers_supports_lines_and_commas():
    parsed = parse_markers("hello, world\nfoo,bar")
    assert set(parsed) == {"hello", "world", "foo", "bar"}


def test_default_markers_nonempty():
    assert "接下來" in DEFAULT_MARKERS
    assert "let me switch" in DEFAULT_MARKERS
