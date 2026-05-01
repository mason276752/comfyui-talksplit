import numpy as np

from split.segmenter import normalize_paragraph, split_sentences
from split.boundary import cosine_similarity, depth_scores, threshold_for_sensitivity
from split.optimizer import optimize_boundaries


def test_split_zh_with_punctuation():
    text = "你好。我很好！你呢？我也不錯；今天天氣真好。"
    sents = split_sentences(text)
    assert len(sents) == 5
    assert sents[0].text.startswith("你好")
    assert text[sents[0].start : sents[0].end] == sents[0].text


def test_split_en_with_periods():
    text = "Good morning. Today is great. Thank you."
    sents = split_sentences(text)
    assert len(sents) == 3
    assert sents[0].text == "Good morning."
    assert sents[2].text == "Thank you."


def test_split_en_decimal_not_split():
    text = "Pi is 3.14. That is interesting."
    sents = split_sentences(text)
    assert len(sents) == 2
    assert "3.14" in sents[0].text


def test_split_unpunctuated_chunks():
    text = "嗯" * 200  # 200 chars without any punctuation
    sents = split_sentences(text, fallback_chunk=30)
    assert len(sents) >= 6
    # last one no longer than chunk size
    assert all(len(s.text) <= 30 for s in sents)


def test_long_terminated_sentence_kept_whole():
    long_en = "I know you have heard this advice many times before, but I still think most of it shares one common blind spot worth examining."
    sents = split_sentences(long_en, fallback_chunk=30)
    assert len(sents) == 1
    assert sents[0].text == long_en


def test_cosine_and_depth_shapes():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(10, 8)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim = cosine_similarity(embeddings)
    depths = depth_scores(sim)
    assert sim.shape == (9,)
    assert depths.shape == (9,)


def test_optimizer_respects_length_constraints():
    # Construct a depth signal with two clear peaks
    depths = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    splits = optimize_boundaries(depths, min_sentences=2, max_sentences=6, sensitivity=1.0)
    # All paragraph lengths must be in [2, 6]
    bounds = [-1] + splits + [len(depths)]
    lens = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    assert all(2 <= L <= 6 for L in lens), lens
    # Splits must be plain Python ints (so JSON output works downstream)
    assert all(isinstance(s, int) and not isinstance(s, bool) for s in splits)


def test_target_paragraphs_count():
    depths = np.linspace(0.1, 1.0, 19)
    splits = optimize_boundaries(
        depths, min_sentences=2, max_sentences=10, target_paragraphs=4
    )
    assert len(splits) == 3


def test_threshold_sensitivity_monotonic():
    depths = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    t_low = threshold_for_sensitivity(depths, 0.5)
    t_mid = threshold_for_sensitivity(depths, 1.0)
    t_high = threshold_for_sensitivity(depths, 2.0)
    assert t_low > t_mid > t_high


def test_normalize_paragraph_strips_blank_lines_in_chinese():
    assert normalize_paragraph("漩渦。\n\n接下來我想換個話題。") == "漩渦。接下來我想換個話題。"


def test_normalize_paragraph_preserves_english_spaces():
    assert normalize_paragraph("Sentence one.\n\nSentence two.") == "Sentence one. Sentence two."
