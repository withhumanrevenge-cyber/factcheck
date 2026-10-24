"""Quoted-span verification against a stored page snapshot.

A span passes only if, after normalisation, it occurs in the snapshot text our own fetcher
stored. A pass proves the words were on the page at fetch time; it does not prove they mean
what the sub-agent thinks (review S5), so the decider also gets the surrounding sentences.

Normalisation: Unicode NFKC, curly quotes and dashes folded, soft hyphens and zero-width
characters removed, words broken across lines by a hyphen re-joined (PDF text), whitespace
collapsed, case folded. An ellipsis inside a quote ("...", "…", "[…]") splits it into
fragments that must all occur, in order.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_TRANSLATE = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "″": '"',
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
        " ": " ",
        "­": None,
        "​": None,
        "‌": None,
        "‍": None,
        "﻿": None,
    }
)
_LINE_BREAK_HYPHEN = re.compile(r"(\w)-[ \t]*\r?\n[ \t]*(\w)")
_ELLIPSIS = re.compile(r"\s*(?:\[\s*(?:\.\.\.|…)\s*\]|\.\.\.|…)\s*")
_SPACE = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE)
    text = _LINE_BREAK_HYPHEN.sub(r"\1\2", text)
    return _SPACE.sub(" ", text).strip().casefold()


@dataclass(frozen=True)
class SpanCheck:
    verified: bool
    reason: str  # "ok" | "too_short" | "not_found" | "empty"
    start: int | None = None  # offsets in the normalised snapshot
    end: int | None = None
    context: str = ""


def check_span(
    snapshot: str, span: str, *, min_words: int = 6, context_sentences: int = 1
) -> SpanCheck:
    norm_page = normalise(snapshot)
    fragments = [f for f in (normalise(p) for p in _ELLIPSIS.split(span)) if f]
    if not fragments:
        return SpanCheck(False, "empty")
    if sum(len(f.split()) for f in fragments) < min_words:
        return SpanCheck(False, "too_short")

    pos = 0
    start = None
    for frag in fragments:
        i = norm_page.find(frag, pos)
        if i < 0:
            return SpanCheck(False, "not_found")
        if start is None:
            start = i
        pos = i + len(frag)
    return SpanCheck(
        True,
        "ok",
        start,
        pos,
        context=_context(norm_page, start, pos, context_sentences),
    )


def _context(text: str, start: int, end: int, n: int) -> str:
    """The sentences containing [start, end) plus n sentences either side."""
    bounds = [0] + [m.end() for m in _SENTENCE_END.finditer(text)] + [len(text)]
    first = max(i for i, b in enumerate(bounds[:-1]) if b <= start)
    last = max(i for i, b in enumerate(bounds[:-1]) if b < end)
    lo = bounds[max(0, first - n)]
    hi = bounds[min(len(bounds) - 1, last + n + 1)]
    return text[lo:hi].strip()
