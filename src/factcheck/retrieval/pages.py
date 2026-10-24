"""Turns a fetched page into the text that spans are checked against, plus its dates.

The date comes from page metadata (article:published_time, datePublished in JSON-LD, a
<time datetime>), never from a model (review S4). A ClaimReview block, when present, is kept
so the decider can see a fact-checker's own rating beside the quote (review S5).
PDFs are fetched and stored but not yet turned into text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from html.parser import HTMLParser
from typing import Any

from factcheck.retrieval.fetch import FetchResult

SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "head", "nav", "footer", "form"}
BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th",
    "section", "article", "header", "blockquote", "pre", "figcaption", "dd", "dt", "table",
}  # fmt: skip
DATE_META = {
    "article:published_time",
    "og:published_time",
    "datepublished",
    "date",
    "pubdate",
    "publishdate",
    "dc.date",
    "dc.date.issued",
}
_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


@dataclass
class Page:
    text: str
    title: str = ""
    published_at: date | None = None
    date_source: str = "unknown"  # "page_metadata" | "unknown"
    claim_reviews: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    m = _ISO_DATE.search(value)
    if not m:
        return None
    try:
        return date(int(m[1]), int(m[2]), int(m[3]))
    except ValueError:
        return None


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: list[str] = []
        self.json_ld: list[str] = []
        self.meta_dates: list[str] = []
        self.time_dates: list[str] = []
        self._skip = 0
        self._in_title = False
        self._in_json_ld = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            if key in DATE_META and a.get("content"):
                self.meta_dates.append(a["content"])
            return
        if tag == "time" and a.get("datetime"):
            self.time_dates.append(a["datetime"])
        if tag == "title":
            self._in_title = True
        if tag == "script" and a.get("type", "").lower() == "application/ld+json":
            self._in_json_ld = True
        if tag in SKIP_TAGS:
            self._skip += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag == "script":
            self._in_json_ld = False
        if tag in SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_json_ld:
            self.json_ld.append(data)
        elif self._in_title:
            self.title.append(data)
        elif not self._skip:
            self.parts.append(data)


def _walk_json_ld(node: Any) -> list[dict[str, Any]]:
    if isinstance(node, list):
        return [d for n in node for d in _walk_json_ld(n)]
    if isinstance(node, dict):
        found = [node]
        for key in ("@graph", "mainEntity", "itemListElement"):
            if key in node:
                found += _walk_json_ld(node[key])
        return found
    return []


def _types(node: dict[str, Any]) -> set[str]:
    t = node.get("@type", [])
    return {t} if isinstance(t, str) else set(t)


def _clean(text: str) -> str:
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def page_from_html(html: str) -> Page:
    ex = _Extractor()
    ex.feed(html)
    ex.close()

    nodes: list[dict[str, Any]] = []
    for block in ex.json_ld:
        try:
            nodes += _walk_json_ld(json.loads(block))
        except json.JSONDecodeError:
            continue
    ld_dates = [n.get("datePublished") for n in nodes if isinstance(n.get("datePublished"), str)]
    reviews = [
        {
            "claim": n.get("claimReviewed"),
            "rating": (n.get("reviewRating") or {}).get("alternateName"),
            "author": (n.get("author") or {}).get("name")
            if isinstance(n.get("author"), dict)
            else None,
            "date": n.get("datePublished"),
        }
        for n in nodes
        if "ClaimReview" in _types(n)
    ]

    published = None
    for candidate in [*ld_dates, *ex.meta_dates, *ex.time_dates]:
        published = parse_date(candidate)
        if published:
            break
    return Page(
        text=_clean("".join(ex.parts)),
        title=_clean("".join(ex.title)),
        published_at=published,
        date_source="page_metadata" if published else "unknown",
        claim_reviews=reviews,
    )


def page_from_fetch(result: FetchResult) -> Page:
    if not result.ok:
        return Page(text="", error=result.error)
    if result.content_type == "application/pdf":
        return Page(text="", error="pdf_not_supported_yet")
    raw = result.body.decode(result.charset or "utf-8", errors="replace")
    if result.content_type == "text/plain":
        return Page(text=_clean(raw))
    return page_from_html(raw)


_WORD = re.compile(r"[a-z0-9]{3,}")
_STOP = {
    "the", "and", "for", "that", "this", "with", "was", "were", "are", "has", "have", "had",
    "from", "but", "not", "its", "his", "her", "their", "they", "will", "been", "into", "than",
}  # fmt: skip


def select_passages(text: str, claim: str, max_chars: int) -> str:
    """Keeps the paragraphs that share the most words with the claim, in page order, up to
    `max_chars`. Keeps token use inside Groq's per-minute limit."""
    if len(text) <= max_chars:
        return text
    claim_words = set(_WORD.findall(claim.lower())) - _STOP
    paras = [p for p in text.split("\n") if p.strip()]
    scored = sorted(
        range(len(paras)),
        key=lambda i: -len(claim_words & set(_WORD.findall(paras[i].lower()))),
    )
    keep: set[int] = set()
    used = 0
    for i in scored:
        if used + len(paras[i]) + 1 > max_chars:
            continue
        keep.add(i)
        used += len(paras[i]) + 1
    return "\n".join(paras[i] for i in sorted(keep))
