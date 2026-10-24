from factcheck.decider import BooleanAnswer, ChoiceAnswer, Decision, ScoreAnswer
from factcheck.evidence import EvidenceItem
from factcheck.gates import Counted


def item(id: str, domain: str, tier: int) -> EvidenceItem:
    return EvidenceItem(
        id=id,
        claim_id="c1",
        url=f"https://{domain}/{id}",
        domain=domain,
        tier=tier,
        quoted_span="quote",
        span_verified=True,
        origin="bundle",
    )


def counted(*specs: tuple[str, str, int, str]) -> list[Counted]:
    """specs: (id, domain, tier, stance)"""
    return [Counted(item(i, d, t), s, 0.9) for i, d, t, s in specs]


def verdict_decision(
    verdict: str,
    p_verdict: float | None = 0.92,
    p_sufficient: float | None = 0.9,
    harm_probs: dict[int, float] | None = None,
    harm_level: int = 0,
    fallback: bool = False,
) -> Decision:
    if p_verdict is None:
        v = ChoiceAnswer(verdict, None)
        s = BooleanAnswer(True, None)
        h = ScoreAnswer(harm_level, None, None)
    else:
        v = ChoiceAnswer(verdict, {verdict: p_verdict})
        s = BooleanAnswer(p_sufficient >= 0.5, p_sufficient)
        probs = harm_probs or {0: 0.9, 1: 0.1, 2: 0.0}
        h = ScoreAnswer(max(probs, key=probs.get), None, probs)
    return Decision(
        answers={"verdict": v, "sufficient": s, "harm": h},
        decider="jev" if p_verdict is not None else "llm",
        model_requested="typesafe-ai/jev",
        model_used="other/model" if fallback else "typesafe-ai/jev",
        fallback_triggered=fallback,
    )
