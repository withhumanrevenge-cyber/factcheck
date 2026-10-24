"""The Decider interface: typed questions about a state in, typed answers out.

Jev is one implementation; an LLM with a strict JSON schema is the other. Code downstream
never assumes probabilities exist: `Decision.has_probabilities` says whether they do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class BooleanQuestion:
    instructions: str
    criteria_true: str | None = None
    criteria_false: str | None = None
    type: Literal["boolean"] = "boolean"


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: str
    options: dict[str, str]  # option name -> description, sent in this order
    type: Literal["choice"] = "choice"


@dataclass(frozen=True)
class ScoreQuestion:
    instructions: str
    levels: tuple[str, ...]  # lowest to highest
    type: Literal["score"] = "score"


Question = BooleanQuestion | ChoiceQuestion | ScoreQuestion


@dataclass(frozen=True)
class BooleanAnswer:
    value: bool
    probability: float | None  # P(true); None when the decider gives no probability


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float] | None

    def p(self, option: str | None = None) -> float | None:
        if self.probabilities is None:
            return None
        return self.probabilities.get(option or self.choice, 0.0)


@dataclass(frozen=True)
class ScoreAnswer:
    level: int  # most likely level, 0 = lowest
    score: float | None  # probability-weighted mean level, when the decider gives one
    probabilities: dict[int, float] | None

    def p(self, level: int) -> float | None:
        if self.probabilities is None:
            return None
        return self.probabilities.get(level, 0.0)


Answer = BooleanAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class Decision:
    answers: dict[str, Answer]
    decider: str  # "jev" | "llm"
    model_requested: str
    model_used: str
    fallback_triggered: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def has_probabilities(self) -> bool:
        for a in self.answers.values():
            if isinstance(a, BooleanAnswer) and a.probability is None:
                return False
            if isinstance(a, ChoiceAnswer | ScoreAnswer) and a.probabilities is None:
                return False
        return True


class DeciderError(RuntimeError):
    """A decider call failed or returned something we cannot use."""


class Decider(Protocol):
    name: str

    def decide(self, state: Any, questions: dict[str, Question]) -> Decision: ...
