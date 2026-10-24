"""Decision thresholds. Placeholders until tuned on the dev split and frozen (Phase 4).

Bump `version` whenever a value changes: cached results are keyed by it (review S9).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Policy:
    version: str = "0.1-placeholder"
    keep_claim: float = 0.70  # P(check_worthy)
    count_stance: float = 0.60  # P(stance) for an item to count as evidence
    sufficient: float = 0.70  # P(sufficient)
    verdict: dict[str, float] = field(
        default_factory=lambda: {
            "supported": 0.80,
            "refuted": 0.80,
            "mixed": 0.85,
            "missing_context": 0.85,
            "outdated": 0.85,
        }
    )
    high_harm_verdict: float = 0.90  # verdict gate when harm is high
    high_harm_p: float = 0.25  # P(harm = high) at which a claim is treated as high-harm
    consistent: float = 0.80  # P(consistent) to show an explanation
    max_rounds: int = 2
    max_items: int = 8
    min_independent_domains: int = 2  # for Supported or Refuted
    max_supporting_tier: int = 2  # at least one item at this tier or better
    min_span_words: int = 6
    # Shown-confidence buckets until a calibration map is fitted (review S1).
    bucket_high: float = 0.90
    bucket_medium: float = 0.80


DEFAULT_POLICY = Policy()
