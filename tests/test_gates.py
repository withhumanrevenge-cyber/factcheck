from datetime import date

from helpers import counted, dated, item, verdict_decision

from factcheck.decider import ChoiceAnswer, Decision
from factcheck.gates import (
    counted_items,
    decide_outcome,
    outdated_date_rule,
    should_search_again,
    stance_rule,
)

AS_OF = date(2026, 10, 1)


def test_outdated_needs_a_dated_superseding_item():
    assert outdated_date_rule([dated("superseded", None)], AS_OF) == "superseded_date_unknown"
    assert (
        outdated_date_rule([dated("superseded", "2026-12-01")], AS_OF) == "superseded_date_unknown"
    )
    assert outdated_date_rule([dated("superseded", "2026-05-01")], AS_OF) == "ok"


def test_outdated_change_must_come_after_the_support():
    older_change = [dated("supports", "2026-06-01", "e1"), dated("superseded", "2026-03-01", "e2")]
    assert outdated_date_rule(older_change, AS_OF) == "superseded_not_newer_than_support"
    newer_change = [dated("supports", "2025-06-01", "e1"), dated("superseded", "2026-03-01", "e2")]
    assert outdated_date_rule(newer_change, AS_OF) == "ok"


def test_outdated_verdict_withheld_when_dates_do_not_line_up():
    evidence = [dated("supports", "2026-06-01", "e1"), dated("superseded", "2026-03-01", "e2")]
    out = decide_outcome(verdict_decision("outdated", 0.95), evidence, as_of=AS_OF)
    assert out.reason == "superseded_not_newer_than_support"


TWO_REFUTING = counted(("e1", "city.example", 1, "refutes"), ("e2", "news.example", 2, "refutes"))


def test_no_counted_evidence_is_unverifiable_without_a_decision():
    out = decide_outcome(None, [])
    assert out.shown_verdict == "unverifiable" and out.reason == "no_counted_evidence"


def test_refuted_shown_when_gates_pass():
    out = decide_outcome(verdict_decision("refuted", 0.93), TWO_REFUTING)
    assert out.shown_verdict == "refuted"
    assert out.confidence == "high"


def test_verdict_must_match_stances():
    out = decide_outcome(verdict_decision("supported", 0.95), TWO_REFUTING)
    assert out.shown_verdict == "unverifiable" and out.reason == "no_matching_evidence"


def test_needs_two_independent_domains():
    same_site = counted(
        ("e1", "city.example", 1, "refutes"), ("e2", "www.city.example", 1, "refutes")
    )
    assert stance_rule("refuted", same_site) == "too_few_independent_domains"


def test_needs_a_high_tier_item():
    low = counted(("e1", "a.example", 3, "refutes"), ("e2", "b.example", 3, "refutes"))
    assert stance_rule("refuted", low) == "no_high_tier_evidence"


def test_equal_or_better_opposing_source_blocks():
    mixed = TWO_REFUTING + counted(("e3", "stats.example", 1, "supports"))
    assert stance_rule("refuted", mixed) == "contradicted_by_equal_or_better_source"
    weaker = TWO_REFUTING + counted(("e3", "blog.example", 3, "supports"))
    assert stance_rule("refuted", weaker) == "ok"


def test_other_verdicts_need_their_kind_of_evidence():
    assert stance_rule("missing_context", TWO_REFUTING) == "no_context_evidence"
    assert stance_rule("outdated", TWO_REFUTING) == "no_superseded_evidence"
    assert stance_rule("mixed", TWO_REFUTING) == "no_mixed_evidence"
    with_context = TWO_REFUTING + counted(("e3", "c.example", 2, "context"))
    assert stance_rule("missing_context", with_context) == "ok"


def test_insufficient_evidence_withheld():
    out = decide_outcome(verdict_decision("refuted", 0.95, p_sufficient=0.5), TWO_REFUTING)
    assert out.reason == "insufficient"


def test_below_threshold_withheld():
    out = decide_outcome(verdict_decision("refuted", 0.75), TWO_REFUTING)
    assert out.reason == "below_threshold"


def test_high_harm_raises_the_bar():
    harmful = {0: 0.1, 1: 0.6, 2: 0.3}  # argmax is medium, but P(high) >= 0.25
    out = decide_outcome(verdict_decision("refuted", 0.85, harm_probs=harmful), TWO_REFUTING)
    assert out.reason == "below_high_harm_threshold"
    out = decide_outcome(verdict_decision("refuted", 0.92, harm_probs=harmful), TWO_REFUTING)
    assert out.shown_verdict == "refuted"


def test_sensitive_topic_is_high_harm():
    out = decide_outcome(verdict_decision("refuted", 0.85), TWO_REFUTING, sensitive_topic=True)
    assert out.reason == "below_high_harm_threshold"


def test_fallback_answer_is_only_a_veto():
    out = decide_outcome(verdict_decision("refuted", 0.99, fallback=True), TWO_REFUTING)
    assert out.shown_verdict == "unverifiable" and out.reason == "fallback_answered"


def test_no_probabilities_capped_at_medium():
    out = decide_outcome(verdict_decision("refuted", None), TWO_REFUTING)
    assert out.shown_verdict == "refuted" and out.confidence == "medium"


def test_no_probabilities_never_shown_for_high_harm_or_soft_verdicts():
    out = decide_outcome(verdict_decision("refuted", None, harm_level=2), TWO_REFUTING)
    assert out.reason == "no_probabilities_high_harm"
    with_context = TWO_REFUTING + counted(("e3", "c.example", 2, "context"))
    out = decide_outcome(verdict_decision("missing_context", None), with_context)
    assert out.reason == "no_probabilities"


def test_counted_items_drops_unrelated_and_unsure():
    items = [item("e1", "a.example", 1), item("e2", "b.example", 1), item("e3", "c.example", 1)]
    d = Decision(
        answers={
            "stance_0": ChoiceAnswer("refutes", {"refutes": 0.9}),
            "stance_1": ChoiceAnswer("unrelated", {"unrelated": 0.95}),
            "stance_2": ChoiceAnswer("supports", {"supports": 0.4}),
        },
        decider="jev",
        model_requested="m",
        model_used="m",
    )
    assert [c.item.id for c in counted_items(items, d)] == ["e1"]


def test_code_decides_whether_to_search_again():
    assert should_search_again(0.5, rounds_done=1)
    assert not should_search_again(0.9, rounds_done=1)
    assert not should_search_again(0.1, rounds_done=2)
    assert should_search_again(None, rounds_done=0)
