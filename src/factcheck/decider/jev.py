"""Jev through Vercel AI Gateway's Decision API (POST /v1/evaluate).

The Gateway names the model that actually answered in the top-level `model` field and sets
`x-ai-gateway-decision-fallback-triggered` when a fallback ran. A language-model fallback
returns `probabilities: {}` for Choice and Score, which we read as "no probabilities".
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from factcheck.config import GATEWAY_URL
from factcheck.decider.base import (
    Answer,
    BooleanAnswer,
    BooleanQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    DeciderError,
    Decision,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from factcheck.decider.http import post_json

FALLBACK_HEADER = "x-ai-gateway-decision-fallback-triggered"


def question_payload(q: Question) -> dict[str, Any]:
    if isinstance(q, BooleanQuestion):
        d: dict[str, Any] = {"type": "boolean", "instructions": q.instructions}
        if q.criteria_true or q.criteria_false:
            d["criteria"] = {"true": q.criteria_true or "", "false": q.criteria_false or ""}
        return d
    if isinstance(q, ChoiceQuestion):
        return {"type": "choice", "instructions": q.instructions, "criteria": dict(q.options)}
    if isinstance(q, ScoreQuestion):
        return {"type": "score", "instructions": q.instructions, "criteria": list(q.levels)}
    raise TypeError(f"unknown question type: {q!r}")


def parse_answer(qid: str, q: Question, raw: dict[str, Any]) -> Answer:
    if isinstance(q, BooleanQuestion):
        p = raw.get("probability")
        if p is None:
            raise DeciderError(f"answer {qid!r} has no probability: {raw}")
        return BooleanAnswer(value=p >= 0.5, probability=float(p))
    if isinstance(q, ChoiceQuestion):
        choice = raw.get("choice")
        if choice not in q.options:
            raise DeciderError(f"answer {qid!r} chose unknown option {choice!r}")
        probs = raw.get("probabilities") or None
        return ChoiceAnswer(
            choice=choice,
            probabilities={k: float(v) for k, v in probs.items()} if probs else None,
        )
    if isinstance(q, ScoreQuestion):
        probs = raw.get("probabilities") or None
        score = raw.get("score")
        if probs:
            by_level = {int(k): float(v) for k, v in probs.items()}
            level = max(by_level, key=by_level.get)
        elif score is not None:
            by_level, level = None, round(float(score))
        else:
            raise DeciderError(f"answer {qid!r} has neither score nor probabilities: {raw}")
        return ScoreAnswer(
            level=level, score=float(score) if score is not None else None, probabilities=by_level
        )
    raise TypeError(f"unknown question type: {q!r}")


def parse_response(
    body: dict[str, Any],
    headers: httpx.Headers | dict[str, str],
    questions: dict[str, Question],
    model_requested: str,
    latency_ms: float | None = None,
) -> Decision:
    answers_raw = body.get("answers") or {}
    missing = [qid for qid in questions if qid not in answers_raw]
    if missing:
        raise DeciderError(f"response is missing answers for {missing}")
    answers = {qid: parse_answer(qid, q, answers_raw[qid]) for qid, q in questions.items()}

    model_used = body.get("model") or model_requested
    fallback = headers.get(FALLBACK_HEADER) == "true" or model_used != model_requested
    usage = body.get("usage") or {}
    cost = ((body.get("providerMetadata") or {}).get("gateway") or {}).get("cost")
    return Decision(
        answers=answers,
        decider="jev",
        model_requested=model_requested,
        model_used=model_used,
        fallback_triggered=fallback,
        input_tokens=usage.get("inputTokens"),
        output_tokens=usage.get("outputTokens"),
        cost_usd=float(cost) if cost is not None else None,
        latency_ms=latency_ms,
        raw=body,
    )


class JevDecider:
    """Decider backed by a Gateway decision model (Jev by default)."""

    def __init__(
        self,
        api_key: str,
        model: str = "typesafe-ai/jev",
        *,
        name: str = "jev",
        base_url: str = GATEWAY_URL,
        fallback: dict[str, Any] | None = None,
        zero_data_retention: bool = False,
        client: httpx.Client | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
    ):
        self.name = name
        self.model = model
        self._url = f"{base_url}/v1/evaluate"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._fallback = fallback
        self._zdr = zero_data_retention
        self._client = client or httpx.Client(timeout=timeout_s)
        self._max_retries = max_retries

    def payload(self, state: Any, questions: dict[str, Question]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "state": state,
            "questions": {qid: question_payload(q) for qid, q in questions.items()},
        }
        gateway: dict[str, Any] = {}
        # A fallback conditioned on one question applies only to requests that ask it.
        cond_q = ((self._fallback or {}).get("when") or {}).get("question")
        if self._fallback and (cond_q is None or cond_q in questions):
            gateway["models"] = [self._fallback]
        if self._zdr:
            gateway["zeroDataRetention"] = True
        if gateway:
            payload["providerOptions"] = {"gateway": gateway}
        return payload

    def decide(self, state: Any, questions: dict[str, Question]) -> Decision:
        started = time.perf_counter()
        resp = post_json(
            self._client,
            self._url,
            headers=self._headers,
            payload=self.payload(state, questions),
            max_retries=self._max_retries,
        )
        latency = (time.perf_counter() - started) * 1000
        return parse_response(resp.json(), resp.headers, questions, self.model, latency)
