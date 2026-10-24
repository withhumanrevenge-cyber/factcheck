"""Evidence items and the code-side facts about them.

Tier and date are set in code, never taken from a sub-agent (review S4): tier from a domain
list, date from page metadata or ClaimReview, else "unknown".
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

UNKNOWN_TIER = 3


class EvidenceItem(BaseModel):
    id: str
    claim_id: str
    url: str
    domain: str
    tier: int = Field(ge=1, le=3)
    tier_source: Literal["domain_list", "default"] = "default"
    published_at: date | None = None
    date_source: Literal["page_metadata", "claimreview", "unknown"] = "unknown"
    retrieved_at: str | None = None
    content_hash: str | None = None  # hash of the stored snapshot the span was checked against
    quoted_span: str
    context: str = ""  # ±1–2 sentences around the span, from the snapshot
    span_verified: bool = False
    origin: Literal["search", "factcheck_api", "archive", "reverse_image", "user_paste", "bundle"]
    # What the sub-agent thought the item said; logged when an item is dropped (review C4).
    sub_agent_stance: str | None = None


def normalise_domain(domain: str) -> str:
    d = domain.strip().lower().rstrip(".")
    return d[4:] if d.startswith("www.") else d


def tier_for_domain(domain: str, tiers: dict[str, int]) -> tuple[int, str]:
    """Look a domain up in the curated tier list; subdomains inherit their parent's tier."""
    d = normalise_domain(domain)
    while d:
        if d in tiers:
            return tiers[d], "domain_list"
        _, _, d = d.partition(".")
    return UNKNOWN_TIER, "default"
