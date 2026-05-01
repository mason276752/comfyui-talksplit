"""ComfyUI custom node wrappers.

Granular nodes (Sentences -> Embed -> Score -> Optimize -> Assemble) for
flexible workflows, plus an Auto node that runs the whole pipeline.
"""
from __future__ import annotations

import io

from .boundary import cosine_similarity, depth_scores, threshold_for_sensitivity
from .embedder import Embedder
from .markers import DEFAULT_MARKERS, boost_depths, parse_markers
from .optimizer import optimize_boundaries
from .segmenter import SOFT_COMMA, Sentence, normalize_paragraph, split_sentences

_T_SENTENCES = "TS_SENTENCES"
_T_EMBEDDINGS = "TS_EMBEDDINGS"
_T_DEPTHS = "TS_DEPTHS"
_T_SPLITS = "TS_SPLITS"

_CATEGORY = "talksplit"


class TalksplitSentences:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "fallback_chunk": ("INT", {"default": 30, "min": 5, "max": 500, "step": 1}),
                "clause_level": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (_T_SENTENCES, "STRING")
    RETURN_NAMES = ("sentences", "text")
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, text: str, fallback_chunk: int, clause_level: bool):
        sents = split_sentences(
            text,
            fallback_chunk=fallback_chunk,
            extra_terminators=SOFT_COMMA if clause_level else "",
        )
        return (sents, text)


class TalksplitEmbed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sentences": (_T_SENTENCES,),
                "model_name": ("STRING", {"default": "BAAI/bge-m3"}),
            },
            "optional": {
                "device": ("STRING", {"default": ""}),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 256}),
            },
        }

    RETURN_TYPES = (_T_EMBEDDINGS,)
    RETURN_NAMES = ("embeddings",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, sentences, model_name: str, device: str = "", batch_size: int = 32):
        if not sentences:
            return (None,)
        embedder = _get_embedder(model_name, device)
        embs = embedder.embed([s.text for s in sentences], batch_size=batch_size)
        return (embs,)


class TalksplitScore:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"embeddings": (_T_EMBEDDINGS,)}}

    RETURN_TYPES = (_T_DEPTHS,)
    RETURN_NAMES = ("depths",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, embeddings):
        sim = cosine_similarity(embeddings)
        depths = depth_scores(sim)
        # bundle is (sim, depths, raw_depths_for_threshold). raw == depths until
        # something post-processes (e.g. MarkerBoost) and wants the threshold to
        # stay anchored to the natural distribution.
        return ((sim, depths, depths),)


class TalksplitMarkerBoost:
    """Boost depth scores at gaps preceding sentences that begin with a
    discourse marker like 'Let me switch' or '接下來'."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sentences": (_T_SENTENCES,),
                "depths": (_T_DEPTHS,),
            },
            "optional": {
                "custom_markers": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Newline / comma-separated phrases. Leave empty to use defaults.",
                }),
            },
        }

    RETURN_TYPES = (_T_DEPTHS,)
    RETURN_NAMES = ("depths",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, sentences, depths, custom_markers: str = ""):
        sim, dps, raw = _unpack_depths(depths)
        markers = parse_markers(custom_markers) if custom_markers.strip() else DEFAULT_MARKERS
        boosted = boost_depths(sentences, dps, markers=markers)
        return ((sim, boosted, raw),)


class TalksplitOptimize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depths": (_T_DEPTHS,),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "min_sentences": ("INT", {"default": 2, "min": 1, "max": 200}),
                "max_sentences": ("INT", {"default": 15, "min": 1, "max": 500}),
                "target_paragraphs": ("INT", {"default": 0, "min": 0, "max": 200}),
            }
        }

    RETURN_TYPES = (_T_SPLITS,)
    RETURN_NAMES = ("splits",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, depths, sensitivity, min_sentences, max_sentences, target_paragraphs):
        _sim, dps, raw = _unpack_depths(depths)
        target = target_paragraphs if target_paragraphs > 0 else None
        threshold = threshold_for_sensitivity(raw, sensitivity) if raw is not dps else None
        splits = optimize_boundaries(
            dps,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            sensitivity=sensitivity,
            target_paragraphs=target,
            threshold=threshold,
        )
        return (splits,)


class TalksplitAssemble:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True}),
                "sentences": (_T_SENTENCES,),
                "splits": (_T_SPLITS,),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("paragraphs",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True

    def run(self, text, sentences, splits):
        out = _assemble(text, sentences, splits)
        return {"ui": {"text": [out]}, "result": (out,)}


class TalksplitPlot:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depths": (_T_DEPTHS,),
                "splits": (_T_SPLITS,),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("plot",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, depths, splits, sensitivity):
        sim, dps, raw = _unpack_depths(depths)
        threshold = threshold_for_sensitivity(raw, sensitivity)
        return (_plot_to_tensor(sim, dps, splits, threshold),)


class TalksplitAuto:
    """One-shot node: text in, paragraphs out."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "min_sentences": ("INT", {"default": 2, "min": 1, "max": 200}),
                "max_sentences": ("INT", {"default": 15, "min": 1, "max": 500}),
                "target_paragraphs": ("INT", {"default": 0, "min": 0, "max": 200}),
                "model_name": ("STRING", {"default": "BAAI/bge-m3"}),
                "use_markers": ("BOOLEAN", {"default": True}),
                "clause_level": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "device": ("STRING", {"default": ""}),
                "fallback_chunk": ("INT", {"default": 30, "min": 5, "max": 500}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("paragraphs",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True

    def run(self, text, sensitivity, min_sentences, max_sentences, target_paragraphs,
            model_name, use_markers, clause_level,
            device: str = "", fallback_chunk: int = 30):
        sentences = split_sentences(
            text,
            fallback_chunk=fallback_chunk,
            extra_terminators=SOFT_COMMA if clause_level else "",
        )
        if len(sentences) < 2:
            return {"ui": {"text": [text]}, "result": (text,)}
        embedder = _get_embedder(model_name, device)
        embs = embedder.embed([s.text for s in sentences])
        sim = cosine_similarity(embs)
        dps = depth_scores(sim)
        raw_threshold = threshold_for_sensitivity(dps, sensitivity)
        if use_markers:
            dps = boost_depths(sentences, dps)
        target = target_paragraphs if target_paragraphs > 0 else None
        splits = optimize_boundaries(
            dps,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            sensitivity=sensitivity,
            target_paragraphs=target,
            threshold=raw_threshold if use_markers else None,
        )
        out = _assemble(text, sentences, splits)
        return {"ui": {"text": [out]}, "result": (out,)}


def _unpack_depths(bundle):
    """Unpack the DEPTHS socket bundle. Older 2-tuples (sim, depths) are
    accepted for backward compatibility."""
    if len(bundle) == 2:
        sim, dps = bundle
        return sim, dps, dps
    sim, dps, raw = bundle
    return sim, dps, raw


# Cache embedders so reusing the same model across nodes / runs doesn't
# reload weights from disk every time.
_embedder_cache: dict[tuple[str, str], Embedder] = {}


def _get_embedder(model_name: str, device: str) -> Embedder:
    key = (model_name, device or "")
    cached = _embedder_cache.get(key)
    if cached is None:
        cached = Embedder(model_name=model_name, device=device or None)
        _embedder_cache[key] = cached
    return cached


def _assemble(text: str, sentences: list[Sentence], splits: list[int]) -> str:
    if not sentences:
        return ""
    bounds = [-1] + sorted(splits) + [len(sentences) - 1]
    paras: list[str] = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i] + 1, bounds[i + 1]
        if start > end:
            continue
        seg = sentences[start:end + 1]
        paras.append(normalize_paragraph(text[seg[0].start : seg[-1].end]))
    return "\n\n".join(paras)


def _plot_to_tensor(sim, depths, splits, threshold):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    x = np.arange(len(sim))
    ax1.plot(x, sim, color="steelblue", label="similarity")
    ax1.set_ylabel("similarity")
    ax1.legend(loc="upper right")
    ax2.plot(x, depths, color="darkorange", label="depth")
    ax2.axhline(threshold, color="gray", linestyle="--", label=f"threshold={threshold:.3f}")
    ax2.set_ylabel("depth")
    ax2.set_xlabel("gap index")
    ax2.legend(loc="upper right")
    for g in splits:
        ax1.axvline(g, color="crimson", alpha=0.4)
        ax2.axvline(g, color="crimson", alpha=0.4)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


NODE_CLASS_MAPPINGS = {
    "TalksplitSentences": TalksplitSentences,
    "TalksplitEmbed": TalksplitEmbed,
    "TalksplitScore": TalksplitScore,
    "TalksplitMarkerBoost": TalksplitMarkerBoost,
    "TalksplitOptimize": TalksplitOptimize,
    "TalksplitAssemble": TalksplitAssemble,
    "TalksplitPlot": TalksplitPlot,
    "TalksplitAuto": TalksplitAuto,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TalksplitSentences": "Talksplit · Sentences",
    "TalksplitEmbed": "Talksplit · Embed",
    "TalksplitScore": "Talksplit · Score",
    "TalksplitMarkerBoost": "Talksplit · Marker Boost",
    "TalksplitOptimize": "Talksplit · Optimize",
    "TalksplitAssemble": "Talksplit · Assemble",
    "TalksplitPlot": "Talksplit · Plot",
    "TalksplitAuto": "Talksplit · Auto",
}
