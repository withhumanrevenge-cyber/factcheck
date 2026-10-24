from pathlib import Path

import pytest

from factcheck.decider import BooleanAnswer, ChoiceAnswer, Decision, ScoreAnswer
from factcheck.eval.bundle import assign_split, load_bundles
from factcheck.eval.metrics import (
    bootstrap_ci,
    expected_calibration_error,
    mean_correct,
    risk_coverage,
    summarise,
)
from factcheck.eval.runner import run_bundle

BUNDLES = Path(__file__).parent.parent / "data" / "bundles"


def test_example_bundles_validate_and_are_skipped_by_default():
    assert load_bundles(BUNDLES) == []
    examples = load_bundles(BUNDLES, include_synthetic=True)
    assert {b.id for b in examples} == {"example-refuted", "example-missing-context"}


def test_split_is_stable_and_groups_by_story():
    assert assign_split("Pub", "Topic A") == assign_split(" pub ", "topic a")
    splits = {assign_split("Pub", f"topic {i}") for i in range(20)}
    assert splits == {"dev", "test"}


class ScriptedDecider:
    """Answers stance and verdict requests from fixed answers, like a perfect Jev."""

    name = "scripted"

    def __init__(self, stances: list[str], verdict: str, p: float = 0.95):
        self.stances, self.verdict, self.p = stances, verdict, p
        self.calls = 0

    def decide(self, state, questions):
        self.calls += 1
        if "verdict" in questions:
            answers = {
                "verdict": ChoiceAnswer(self.verdict, {self.verdict: self.p}),
                "sufficient": BooleanAnswer(True, 0.9),
                "harm": ScoreAnswer(0, 0.1, {0: 0.9, 1: 0.1, 2: 0.0}),
            }
        else:
            answers = {
                qid: ChoiceAnswer(s, {s: 0.9})
                for qid, s in zip(questions, self.stances, strict=True)
            }
        return Decision(answers, "jev", "m", "m", latency_ms=10, input_tokens=100, cost_usd=0.001)


def test_run_bundle_end_to_end():
    b = next(x for x in load_bundles(BUNDLES, include_synthetic=True) if x.id == "example-refuted")
    decider = ScriptedDecider(["refutes", "refutes", "unrelated"], "refuted")
    rec = run_bundle(b, decider)
    assert decider.calls == 2
    assert rec["counted_ids"] == ["e1", "e2"]
    assert rec["model_verdict"] == "refuted"
    assert rec["outcome"]["shown_verdict"] == "refuted"
    assert rec["latency_ms"] == 20 and rec["cost_usd"] == pytest.approx(0.002)
    s = summarise([rec])
    assert s["stance_accuracy"] == 1.0 and s["selective_accuracy"] == 1.0


def test_run_bundle_skips_verdict_when_nothing_counts():
    b = next(x for x in load_bundles(BUNDLES, include_synthetic=True) if x.id == "example-refuted")
    decider = ScriptedDecider(["unrelated"] * 3, "refuted")
    rec = run_bundle(b, decider)
    assert decider.calls == 1
    assert rec["outcome"]["reason"] == "no_counted_evidence"


def test_ece_perfectly_calibrated_is_zero():
    pairs = [(0.75, True)] * 3 + [(0.75, False)]
    ece, bins = expected_calibration_error(pairs)
    assert ece == pytest.approx(0.0)
    assert [b.n for b in bins if b.n] == [4]


def test_ece_overconfident():
    ece, _ = expected_calibration_error([(0.95, False), (0.95, True)])
    assert ece == pytest.approx(0.45)


def test_risk_coverage_answers_most_confident_first():
    points, aurc = risk_coverage([(0.9, True), (0.5, False), (0.8, True), (0.6, False)])
    assert points == [(0.25, 0.0), (0.5, 0.0), (0.75, 1 / 3), (1.0, 0.5)]
    assert aurc == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    pairs = [("a", "a")] * 7 + [("a", "b")] * 3
    ci = bootstrap_ci(pairs, mean_correct, seed=1)
    assert ci == bootstrap_ci(pairs, mean_correct, seed=1)
    assert ci[0] <= 0.7 <= ci[1]
