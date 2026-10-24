"""Code-side gates between a decider's answers and what the user is shown.

Rule across all of them: when something is missing or uncertain, show less, never more.

- No counted evidence: Unverifiable without asking the decider (review C5).
- Every shown verdict needs matching stances in code, not just the decider's word (C5).
- Gates for all five decisive verdicts, and a stricter one for high-harm claims (S1).
- An answer without probabilities (an LLM decider) can show only Supported or Refuted,
  never for high harm, capped at medium confidence (C3).
- A Gateway fallback answer is a veto only: the primary model was unsure, so the
  claim is Unverifiable whatever the fallback said (C3).
- An outdated verdict needs dates that line up, checked in code (Phase 0 findings).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from factcheck.decider.base import BooleanAnswer, ChoiceAnswer, Decision, ScoreAnswer
from factcheck.evidence import EvidenceItem, normalise_domain
from factcheck.policy import DEFAULT_POLICY, Policy
from factcheck.questions import HIGH_HARM_LEVEL

UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class Counted:
    item: EvidenceItem
    stance: str
    probability: float | None


@dataclass(frozen=True)
class Outcome:
    shown_verdict: str  # what the user sees
    model_verdict: str | None  # what the decider answered, before gates
    reason: str  # "shown", or why it was withheld
    probability: float | None = None
    confidence: str | None = None  # "high" | "medium" | "low" when shown


def counted_items(
    items: list[EvidenceItem], stance_decision: Decision, policy: Policy = DEFAULT_POLICY
) -> list[Counted]:
    """Items whose stance is not unrelated, at or above the counting threshold."""
    out: list[Counted] = []
    for i, item in enumerate(items):
        ans = stance_decision.answers[f"stance_{i}"]
        assert isinstance(ans, ChoiceAnswer)
        if ans.choice == "unrelated":
            continue
        p = ans.p()
        if p is not None and p < policy.count_stance:
            continue
        out.append(Counted(item, ans.choice, p))
    return out


def outdated_date_rule(counted: list[Counted], as_of: date | None) -> str:
    """Dates are checked here, not by the decider, which is weak at date reasoning.

    An outdated verdict needs a superseding item with a known date on or before as_of, and
    newer than the newest dated item that supports the claim."""
    superseded = [c.item.published_at for c in counted if c.stance == "superseded"]
    if not superseded:
        return "no_superseded_evidence"
    dated = [d for d in superseded if d is not None and (as_of is None or d <= as_of)]
    if not dated:
        return "superseded_date_unknown"
    supporting = [
        c.item.published_at for c in counted if c.stance == "supports" and c.item.published_at
    ]
    if supporting and max(dated) <= max(supporting):
        return "superseded_not_newer_than_support"
    return "ok"


def stance_rule(
    verdict: str,
    counted: list[Counted],
    policy: Policy = DEFAULT_POLICY,
    as_of: date | None = None,
) -> str:
    """Returns "ok", or the reason the counted stances do not back this verdict."""
    by = {s: [c for c in counted if c.stance == s] for s in {c.stance for c in counted}}
    if verdict in ("supported", "refuted"):
        same, opposite = (
            ("supports", "refutes") if verdict == "supported" else ("refutes", "supports")
        )
        backing = by.get(same, [])
        if not backing:
            return "no_matching_evidence"
        if len({normalise_domain(c.item.domain) for c in backing}) < policy.min_independent_domains:
            return "too_few_independent_domains"
        best = min(c.item.tier for c in backing)
        if best > policy.max_supporting_tier:
            return "no_high_tier_evidence"
        if any(c.item.tier <= best for c in by.get(opposite, [])):
            return "contradicted_by_equal_or_better_source"
        return "ok"
    if verdict == "mixed":
        both = by.get("supports") and by.get("refutes")
        return "ok" if both or by.get("mixed") else "no_mixed_evidence"
    if verdict == "missing_context":
        return "ok" if by.get("context") else "no_context_evidence"
    if verdict == "outdated":
        return outdated_date_rule(counted, as_of)
    return "ok"


def is_high_harm(
    harm: ScoreAnswer, policy: Policy = DEFAULT_POLICY, sensitive_topic: bool = False
) -> bool:
    if sensitive_topic:
        return True
    p_high = harm.p(HIGH_HARM_LEVEL)
    if p_high is None:
        return harm.level >= HIGH_HARM_LEVEL
    return p_high >= policy.high_harm_p


def confidence_bucket(p: float, policy: Policy = DEFAULT_POLICY) -> str:
    if p >= policy.bucket_high:
        return "high"
    if p >= policy.bucket_medium:
        return "medium"
    return "low"


def decide_outcome(
    verdict_decision: Decision | None,
    counted: list[Counted],
    policy: Policy = DEFAULT_POLICY,
    *,
    sensitive_topic: bool = False,
    as_of: date | None = None,
) -> Outcome:
    if not counted or verdict_decision is None:
        return Outcome(UNVERIFIABLE, None, "no_counted_evidence")

    verdict = verdict_decision.answers["verdict"]
    sufficient = verdict_decision.answers["sufficient"]
    harm = verdict_decision.answers["harm"]
    assert isinstance(verdict, ChoiceAnswer)
    assert isinstance(sufficient, BooleanAnswer)
    assert isinstance(harm, ScoreAnswer)
    v = verdict.choice

    def withheld(reason: str) -> Outcome:
        return Outcome(UNVERIFIABLE, v, reason, verdict.p())

    if verdict_decision.fallback_triggered:
        return withheld("fallback_answered")
    if v == UNVERIFIABLE:
        return Outcome(UNVERIFIABLE, v, "decider_unverifiable", verdict.p())

    high_harm = is_high_harm(harm, policy, sensitive_topic)
    rule = stance_rule(v, counted, policy, as_of)

    if not verdict_decision.has_probabilities:
        if v not in ("supported", "refuted"):
            return withheld("no_probabilities")
        if high_harm:
            return withheld("no_probabilities_high_harm")
        if rule != "ok":
            return withheld(rule)
        return Outcome(v, v, "shown", None, "medium")

    p_verdict = verdict.p()
    assert p_verdict is not None and sufficient.probability is not None
    if sufficient.probability < policy.sufficient:
        return withheld("insufficient")
    gate = policy.verdict[v]
    if high_harm:
        gate = max(gate, policy.high_harm_verdict)
    if p_verdict < gate:
        return withheld("below_high_harm_threshold" if high_harm else "below_threshold")
    if rule != "ok":
        return withheld(rule)
    return Outcome(v, v, "shown", p_verdict, confidence_bucket(p_verdict, policy))


def should_search_again(
    sufficient_p: float | None, rounds_done: int, policy: Policy = DEFAULT_POLICY
) -> bool:
    """Code decides another round, not the decider (review S2)."""
    if rounds_done >= policy.max_rounds:
        return False
    return sufficient_p is None or sufficient_p < policy.sufficient
