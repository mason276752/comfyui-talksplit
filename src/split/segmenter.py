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
    """
    if extra_terminators:
        pattern = f"({_HARD_PATTERN}|[{re.escape(extra_terminators)}]+)"
        split_re = re.compile(pattern)
    else:
        split_re = _SPLIT_RE
    sentences: list[Sentence] = []
    pieces = split_re.split(text)

    pos = 0
    buf = ""
    buf_start = 0
    for piece in pieces:
        if not piece:
            continue
        if buf == "":
            buf_start = pos
        buf += piece
        pos += len(piece)
        if split_re.fullmatch(piece):
            _emit(sentences, buf, buf_start, chunk_long=False, fallback_chunk=fallback_chunk)
            buf = ""
    if buf:
        # Leftover with no terminator — only here we fall back to chunking.
        _emit(sentences, buf, buf_start, chunk_long=True, fallback_chunk=fallback_chunk)
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
