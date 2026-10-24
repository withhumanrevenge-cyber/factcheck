"""Hand-built claim-and-evidence bundles for the decider spike.

One JSON file per bundle. Evidence in a bundle counts as verified: the person building it
checked each quote against the page. See data/bundles/README.md for how to build them.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from factcheck.evidence import EvidenceItem
from factcheck.questions import STANCE_OPTIONS, VERDICTS

HARM_LABELS = ("low", "medium", "high")


class BundleEvidence(BaseModel):
    id: str
    url: str
    domain: str
    tier: int = Field(ge=1, le=3)
    published_at: date | None = None
    quote: str
    context: str = ""
    label_stance: str

    @field_validator("label_stance")
    @classmethod
    def _stance(cls, v: str) -> str:
        if v not in STANCE_OPTIONS:
            raise ValueError(f"label_stance must be one of {list(STANCE_OPTIONS)}")
        return v


class Bundle(BaseModel):
    id: str
    claim: str
    as_of: date
    label_verdict: str
    label_harm: str
    topic: str
    source_publisher: str  # the fact-check the bundle was built from
    source_url: str
    source_published: date | None = None  # must be after the models' training cutoffs
    evidence: list[BundleEvidence] = Field(min_length=1, max_length=8)
    synthetic: bool = False  # format examples only; skipped by the runner by default
    notes: str = ""

    @field_validator("label_verdict")
    @classmethod
    def _verdict(cls, v: str) -> str:
        if v not in VERDICTS:
            raise ValueError(f"label_verdict must be one of {list(VERDICTS)}")
        return v

    @field_validator("label_harm")
    @classmethod
    def _harm(cls, v: str) -> str:
        if v not in HARM_LABELS:
            raise ValueError(f"label_harm must be one of {list(HARM_LABELS)}")
        return v

    @property
    def split(self) -> str:
        return assign_split(self.source_publisher, self.topic)

    def evidence_items(self) -> list[EvidenceItem]:
        return [
            EvidenceItem(
                id=e.id,
                claim_id=self.id,
                url=e.url,
                domain=e.domain,
                tier=e.tier,
                tier_source="domain_list",
                published_at=e.published_at,
                date_source="page_metadata" if e.published_at else "unknown",
                quoted_span=e.quote,
                context=e.context,
                span_verified=True,
                origin="bundle",
            )
            for e in self.evidence
        ]


def assign_split(publisher: str, topic: str) -> str:
    """Dev or test, by publisher and topic together, so paraphrases of one story
    never land on both sides (review S10)."""
    key = f"{publisher.strip().lower()}|{topic.strip().lower()}".encode()
    return "dev" if hashlib.sha256(key).digest()[0] % 2 == 0 else "test"


def load_bundles(directory: Path, *, include_synthetic: bool = False) -> list[Bundle]:
    bundles = []
    for path in sorted(directory.rglob("*.json")):
        b = Bundle.model_validate(json.loads(path.read_text(encoding="utf-8")))
        if b.synthetic and not include_synthetic:
            continue
        bundles.append(b)
    ids = [b.id for b in bundles]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate bundle ids: {sorted(dupes)}")
    return bundles
