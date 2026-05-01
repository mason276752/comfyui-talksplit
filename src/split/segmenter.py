import re
from dataclasses import dataclass

_HARD = "。！？；!?;\n"
# Hard terminators always split. ASCII '.' splits when followed by whitespace
# or end-of-text — avoids breaking decimals (3.14) but may over-split
# abbreviations (Mr. Smith), which the embedder typically rejoins anyway.
_HARD_PATTERN = rf"[{re.escape(_HARD)}]+|\.+(?=\s|$)"
_SPLIT_RE = re.compile(f"({_HARD_PATTERN})")
# Common "soft" terminators users might enable for clause-level splits.
SOFT_COMMA = ",，"

_WS_RUN = re.compile(r"\s+")
_CJK = r"[　-〿一-鿿＀-￯]"
_CJK_GAP = re.compile(rf"(?<={_CJK}) (?={_CJK})")

# Zero-width implicit break: split BEFORE these Chinese transition phrases
# when they appear mid-text (not already at a newline boundary).
# Covers temporal shifts, discourse markers, and answer/question openers
# common in unpunctuated presentation transcripts.
_ZH_IMPLICIT_RE = re.compile(
    # Case 1: general discourse markers (must not be at a line start)
    r"(?<=[^\n])(?="
        r"以前(?!從|都|也|已|就)|現在開始|這時候問題|"
        r"接下來|接著|另外|首先|其次|再來|再者|最後(?!也)|總之|"
        r"最壞的|次差的|"
        r"答案(?:是|很|：)|而是因為"
    r")"
    # Case 2: 未來 as a discourse opener — not when preceded by 的 (prep phrase)
    r"|(?<=[^\n的])(?=未來)"
    # Case 3: ordinal enumeration openers 第N是/個/點/層/類/步/項/條/部
    # Excluded: 第N次/名/年/天/回/輪 — those are NOT section markers.
    r"|(?<=[^\n])(?=第[一二三四五六七八九十百]+(?=[，、是個點層類步項條部])"
    r"|其[一二三四五六七八九十]+(?=[，、是]))"
)


@dataclass
class Sentence:
    text: str
    start: int
    end: int


def normalize_paragraph(s: str) -> str:
    """Collapse internal whitespace runs to single spaces, then drop spaces
    between adjacent CJK characters so Chinese output stays unspaced."""
    s = _WS_RUN.sub(" ", s)
    s = _CJK_GAP.sub("", s)
    return s.strip()


def split_sentences(
    text: str,
    fallback_chunk: int = 30,
    extra_terminators: str = "",
) -> list[Sentence]:
    """Split text into sentences (or finer clauses).

    Primary signal: sentence-terminating punctuation. A real terminated
    piece is always kept whole, regardless of length. Only trailing content
    with no terminator (e.g. raw unpunctuated transcripts) gets
    window-chopped into ``fallback_chunk`` pieces.

    ``extra_terminators`` adds extra characters that should also end a piece.
    Pass ``SOFT_COMMA`` (or any subset) to allow paragraph splits to fall at
    commas — useful for long, comma-strung sentences.

    Additionally, zero-width implicit breaks are inserted before common
    Chinese transition phrases (e.g. 以前, 未來, 接下來) so that
    unpunctuated presentation transcripts still produce meaningful sentence
    units without any LLM pre-processing.
    """
    if extra_terminators:
        pattern = f"({_HARD_PATTERN}|[{re.escape(extra_terminators)}]+)"
        split_re = re.compile(pattern)
    else:
        split_re = _SPLIT_RE

    # Collect cut points from both explicit terminators and implicit breaks.
    # Explicit terminator at [m.start:m.end] → next sentence starts at m.end.
    # Implicit break at m.start (zero-width)   → next sentence starts at m.start.
    cut_points: list[int] = []
    for m in split_re.finditer(text):
        cut_points.append(m.end())
    for m in _ZH_IMPLICIT_RE.finditer(text):
        cut_points.append(m.start())
    cut_points = sorted(set(cut_points))

    sentences: list[Sentence] = []
    prev = 0
    for cp in cut_points:
        if cp <= prev:
            continue
        chunk = text[prev:cp]
        stripped = chunk.strip()
        if stripped:
            leading = len(chunk) - len(chunk.lstrip())
            base = prev + leading
            sentences.append(Sentence(stripped, base, base + len(stripped)))
        prev = cp

    # Leftover with no terminator — only here we fall back to chunking.
    if prev < len(text):
        _emit(sentences, text[prev:], prev, chunk_long=True, fallback_chunk=fallback_chunk)
    return sentences


def _emit(out: list[Sentence], buf: str, start: int,
          chunk_long: bool, fallback_chunk: int) -> None:
    stripped = buf.strip()
    if not stripped:
        return
    leading = len(buf) - len(buf.lstrip())
    base = start + leading
    if not chunk_long or len(stripped) <= fallback_chunk * 3:
        out.append(Sentence(stripped, base, base + len(stripped)))
        return
    for i in range(0, len(stripped), fallback_chunk):
        chunk = stripped[i:i + fallback_chunk]
        if chunk.strip():
            out.append(Sentence(chunk, base + i, base + i + len(chunk)))
