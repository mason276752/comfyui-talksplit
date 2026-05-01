"""Smoke tests for the ComfyUI bridge.

Loads the root ``__init__.py`` exactly the way ComfyUI would, verifies the
node mappings are present, and exercises the nodes that don't need the
embedding model (so this stays fast and offline).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def comfy_module():
    spec = importlib.util.spec_from_file_location(
        "_talksplit_root_test", os.path.join(REPO_ROOT, "__init__.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_talksplit_root_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def nodes_module():
    return sys.modules["_talksplit_internal.comfy_nodes"]


def test_node_mappings_exposed(comfy_module):
    expected = {
        "TalksplitSentences",
        "TalksplitEmbed",
        "TalksplitScore",
        "TalksplitOptimize",
        "TalksplitAssemble",
        "TalksplitPlot",
        "TalksplitAuto",
    }
    assert expected.issubset(comfy_module.NODE_CLASS_MAPPINGS)
    assert expected.issubset(comfy_module.NODE_DISPLAY_NAME_MAPPINGS)


def test_every_node_has_required_attrs(comfy_module):
    for name, cls in comfy_module.NODE_CLASS_MAPPINGS.items():
        assert callable(getattr(cls, "INPUT_TYPES", None)), name
        assert isinstance(cls.RETURN_TYPES, tuple), name
        assert isinstance(cls.FUNCTION, str), name
        assert isinstance(cls.CATEGORY, str), name
        # the function attribute must point to a real method
        assert hasattr(cls, cls.FUNCTION), f"{name}.{cls.FUNCTION} missing"


def test_sentences_node_runs(nodes_module):
    cls = nodes_module.TalksplitSentences
    sents, text = cls().run("你好。今天天氣不錯！", fallback_chunk=30, clause_level=False)
    assert len(sents) == 2
    assert text.startswith("你好")


def test_split_to_list_node(nodes_module):
    cls = nodes_module.TalksplitSplitToList
    paragraphs = "第一段內容。\n\n第二段內容。\n\n第三段內容。"
    (items,) = cls().run(paragraphs)
    assert items == ["第一段內容。", "第二段內容。", "第三段內容。"]


def test_split_to_list_drops_empty(nodes_module):
    cls = nodes_module.TalksplitSplitToList
    (items,) = cls().run("a\n\n\n\nb\n\n   \n\nc")
    assert items == ["a", "b", "c"]


def test_split_to_list_marked_as_list_output(nodes_module):
    cls = nodes_module.TalksplitSplitToList
    assert getattr(cls, "OUTPUT_IS_LIST", None) == (True,)


def test_pick_paragraph_basic(nodes_module):
    cls = nodes_module.TalksplitPickParagraph
    text = "alpha\n\nbeta\n\ngamma"
    para, count = cls().run(text, 1)
    assert para == "beta"
    assert count == 3


def test_pick_paragraph_negative_index(nodes_module):
    cls = nodes_module.TalksplitPickParagraph
    para, _ = cls().run("alpha\n\nbeta\n\ngamma", -1)
    assert para == "gamma"


def test_pick_paragraph_clamps_out_of_range(nodes_module):
    cls = nodes_module.TalksplitPickParagraph
    para, count = cls().run("alpha\n\nbeta", 99)
    assert para == "beta"
    assert count == 2


def test_pick_paragraph_empty_input(nodes_module):
    cls = nodes_module.TalksplitPickParagraph
    para, count = cls().run("", 0)
    assert para == ""
    assert count == 0


def test_score_optimize_assemble_pipeline_offline(nodes_module):
    """Pipeline from Score onward, fed with synthetic embeddings (no model load)."""
    sentences_cls = nodes_module.TalksplitSentences
    score_cls = nodes_module.TalksplitScore
    optimize_cls = nodes_module.TalksplitOptimize
    assemble_cls = nodes_module.TalksplitAssemble

    text = "甲一。甲二。甲三。乙一。乙二。乙三。丙一。丙二。"
    sents, _ = sentences_cls().run(text, fallback_chunk=30, clause_level=False)

    # Hand-craft embeddings that cluster into 3 topics with clear valleys
    rng = np.random.default_rng(0)
    base = rng.normal(size=(3, 16)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    pattern = [0, 0, 0, 1, 1, 1, 2, 2]
    embs = np.stack([base[p] for p in pattern[: len(sents)]])

    (depths_bundle,) = score_cls().run(embs)
    (splits,) = optimize_cls().run(
        depths_bundle,
        sensitivity=1.0,
        min_sentences=2,
        max_sentences=5,
        target_paragraphs=0,
    )
    assert 1 <= len(splits) <= 4

    out = assemble_cls().run(text, sents, splits)
    paragraphs = out["result"][0]
    assert paragraphs.count("\n\n") == len(splits)
