"""The static questions sent to a decider.

Instructions and criteria are fixed text written here. Claim and evidence text only ever goes
into the state, never into an instruction. Paths in backticks point into the state built by
`factcheck.state`.

Changes from the design doc, following the architecture review:
- Stance has `context` and `superseded` options, so evidence for missing_context and
  outdated verdicts has a label (review S3), and "only repeats the claim" counts as
  unrelated, so a debunk quoting the false claim is not read as support (review S5).
- Stance is asked first and the verdict second, on counted items only (review C5).
- There is no next_action question: code decides whether to search again (review S2).
"""

from __future__ import annotations

from factcheck.decider.base import BooleanQuestion, ChoiceQuestion, ScoreQuestion

STANCE_OPTIONS: dict[str, str] = {
    "supports": "The quote directly states that the claim, as worded, is true.",
    "refutes": "The quote directly states that the claim, as worded, is false.",
    "mixed": "The quote confirms one part of the claim and contradicts another part.",
    "context": "The quote does not contradict the claim but adds facts that change how it should be read.",
    "superseded": "The quote shows the claim was true at an earlier time but is no longer true.",
    "unrelated": "The quote does not address the claim, or only repeats the claim without judging it.",
}

VERDICT_OPTIONS: dict[str, str] = {
    "supported": "The evidence directly confirms the claim as stated.",
    "refuted": "The evidence directly contradicts the claim.",
    "mixed": "Parts of the claim hold and parts do not.",
    "missing_context": "The claim is accurate but misleading without the context the evidence gives.",
    "outdated": "The claim was true earlier and is no longer true.",
    "unverifiable": "The evidence does not settle the claim.",
}

VERDICTS = tuple(VERDICT_OPTIONS)

HARM_LEVELS = (
    "low: a wrong verdict would mislead people about a trivial matter",
    "medium: a wrong verdict could mislead people about public affairs or money",
    "high: a wrong verdict could harm health or safety, affect an election, or damage a named person",
)
HIGH_HARM_LEVEL = 2


def stance_question(index: int) -> ChoiceQuestion:
    return ChoiceQuestion(
        instructions=(
            f"What does `evidence[{index}].quote`, read together with `evidence[{index}].context`, "
            "say about `claim`?"
        ),
        options=STANCE_OPTIONS,
    )


def stance_questions(n_items: int) -> dict[str, ChoiceQuestion]:
    return {f"stance_{i}": stance_question(i) for i in range(n_items)}


VERDICT = ChoiceQuestion(
    instructions="Using only the items in `evidence`, what is the status of `claim`?",
    options=VERDICT_OPTIONS,
)

SUFFICIENT = BooleanQuestion(
    instructions="Do the items in `evidence`, taken together, settle whether `claim` is true or false?",
    criteria_true="The evidence is enough to decide the claim either way.",
    criteria_false="More or better evidence would be needed to decide the claim.",
)

HARM = ScoreQuestion(
    instructions="How harmful would it be to show a wrong verdict on `claim`?",
    levels=HARM_LEVELS,
)

CHECK_WORTHY = BooleanQuestion(
    instructions="Is `claim` a statement of fact that could be checked against public sources?",
    criteria_true="A factual statement about the world that sources could confirm or contradict.",
    criteria_false="An opinion, a prediction, a question, a joke, or too vague to check.",
)

CONSISTENT = BooleanQuestion(
    instructions=(
        "Does `explanation` agree with `verdict`, and does it state only facts found in `evidence`?"
    ),
    criteria_true="The explanation matches the verdict and every fact in it appears in the evidence.",
    criteria_false="The explanation contradicts the verdict or states facts not in the evidence.",
)


def verdict_questions() -> dict[str, ChoiceQuestion | BooleanQuestion | ScoreQuestion]:
    return {"verdict": VERDICT, "sufficient": SUFFICIENT, "harm": HARM}
