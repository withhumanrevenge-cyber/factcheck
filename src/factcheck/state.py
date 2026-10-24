"""Builds the state a decider reads. Only verified evidence ever goes in."""

from __future__ import annotations

from typing import Any

from factcheck.evidence import EvidenceItem


def claim_state(claim: str, as_of: str, items: list[EvidenceItem]) -> dict[str, Any]:
    unverified = [i.id for i in items if not i.span_verified]
    if unverified:
        raise ValueError(f"unverified evidence must not reach a decider: {unverified}")
    return {
        "claim": claim,
        "as_of": as_of,
        "evidence": [
            {
                "id": i.id,
                "source": i.domain,
                "tier": i.tier,
                "published": i.published_at.isoformat() if i.published_at else "unknown",
                "quote": i.quoted_span,
                "context": i.context,
            }
            for i in items
        ],
    }
