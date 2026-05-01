from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .boundary import cosine_similarity, depth_scores, threshold_for_sensitivity
from .embedder import Embedder
from .markers import boost_depths
from .optimizer import optimize_boundaries
from .segmenter import SOFT_COMMA, Sentence, normalize_paragraph, split_sentences


MODE_DEFAULTS = {
    # speech transcripts: short paragraphs, aggressive cuts
    "speech": {"sensitivity": 1.0, "min_sentences": 2, "max_sentences": 15},
    # written articles / blog posts: longer paragraphs, conservative cuts
    "article": {"sensitivity": 0.7, "min_sentences": 4, "max_sentences": 30},
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="talksplit",
        description="Split a speech transcript into semantically coherent paragraphs.",
    )
    parser.add_argument("input", help="Path to input text file, or '-' for stdin.")
    parser.add_argument("-o", "--output", help="Output path (default: stdout).")
    parser.add_argument(
        "--mode",
        choices=list(MODE_DEFAULTS),
        default="speech",
        help="Preset defaults. 'speech' (1.0 / 2 / 15) for transcripts; "
             "'article' (0.7 / 4 / 30) for longer prose. Explicit flags below override.",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=None,
        help="0–2; higher = more splits. Overrides --mode default.",
    )
    parser.add_argument("--min-sentences", type=int, default=None,
                        help="Overrides --mode default.")
    parser.add_argument("--max-sentences", type=int, default=None,
                        help="Overrides --mode default.")
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="If given, force exactly this many paragraphs (overrides sensitivity).",
    )
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--device", default=None, help="torch device: cpu/cuda/mps")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--plot", default=None, help="Save similarity/depth plot to PNG.")
    parser.add_argument("--fallback-chunk", type=int, default=30)
    parser.add_argument(
        "--no-markers",
        action="store_true",
        help="Disable discourse-marker boost (e.g. 'Let me switch', '接下來').",
    )
    parser.add_argument(
        "--clause-level",
        action="store_true",
        help="Allow paragraph splits at commas, not just sentence ends.",
    )

    args = parser.parse_args(argv)

    defaults = MODE_DEFAULTS[args.mode]
    sensitivity = args.sensitivity if args.sensitivity is not None else defaults["sensitivity"]
    min_sentences = args.min_sentences if args.min_sentences is not None else defaults["min_sentences"]
    max_sentences = args.max_sentences if args.max_sentences is not None else defaults["max_sentences"]

    text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")

    sentences = split_sentences(
        text,
        fallback_chunk=args.fallback_chunk,
        extra_terminators=SOFT_COMMA if args.clause_level else "",
    )
    if len(sentences) < 2:
        _write(text, args.output)
        return 0

    embedder = Embedder(args.model, device=args.device)
    embeddings = embedder.embed([s.text for s in sentences])

    sim = cosine_similarity(embeddings)
    depths = depth_scores(sim)
    raw_threshold = threshold_for_sensitivity(depths, sensitivity)

    if not args.no_markers:
        depths = boost_depths(sentences, depths, threshold=raw_threshold)

    splits = optimize_boundaries(
        depths,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
        sensitivity=sensitivity,
        target_paragraphs=args.target,
        threshold=raw_threshold,
    )

    paragraphs = _make_paragraphs(text, sentences, splits)

    if args.plot:
        from .visualize import plot_curve
        plot_curve(sim, depths, splits, raw_threshold, args.plot)


    if args.format == "json":
        out = json.dumps(
            {
                "paragraphs": [
                    {"text": p_text, "sentence_indices": idx} for p_text, idx in paragraphs
                ],
                "splits": splits,
            },
            ensure_ascii=False,
            indent=2,
        )
    else:
        out = "\n\n".join(p_text for p_text, _ in paragraphs)

    _write(out, args.output)
    return 0


def _make_paragraphs(
    text: str, sentences: list[Sentence], splits: list[int]
) -> list[tuple[str, list[int]]]:
    bounds = [-1] + sorted(splits) + [len(sentences) - 1]
    out: list[tuple[str, list[int]]] = []
    for i in range(len(bounds) - 1):
        start = bounds[i] + 1
        end = bounds[i + 1]
        if start > end:
            continue
        seg = sentences[start:end + 1]
        para = normalize_paragraph(text[seg[0].start : seg[-1].end])
        out.append((para, list(range(start, end + 1))))
    return out


def _write(text: str, path: str | None) -> None:
    if path:
        Path(path).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
