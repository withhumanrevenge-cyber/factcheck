import json

import httpx
import pytest

from factcheck.decider import (
    BooleanQuestion,
    ChoiceQuestion,
    DeciderError,
    JevDecider,
    LLMDecider,
    ScoreQuestion,
)
from factcheck.decider.jev import parse_response

QUESTIONS = {
    "ok": BooleanQuestion("Is it ok?"),
    "route": ChoiceQuestion("Where?", {"billing": "money", "tech": "bugs"}),
    "urgency": ScoreQuestion("How urgent?", ("low", "medium", "high")),
}

JEV_BODY = {
    "model": "typesafe-ai/jev",
    "answers": {
        "ok": {"type": "boolean", "probability": 0.98},
        "route": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.9, "tech": 0.1},
        },
        "urgency": {"type": "score", "score": 1.9, "probabilities": {"0": 0.0, "1": 0.1, "2": 0.9}},
    },
    "usage": {"inputTokens": 275, "outputTokens": 20},
    "providerMetadata": {"gateway": {"cost": "0.00001155"}},
}


def client_returning(*responses: httpx.Response, seen: list | None = None) -> httpx.Client:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return queue.pop(0)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_jev_parses_all_answer_types():
    seen: list = []
    jev = JevDecider("k", client=client_returning(httpx.Response(200, json=JEV_BODY), seen=seen))
    d = jev.decide({"claim": "x"}, QUESTIONS)
    assert seen[0]["questions"]["ok"] == {"type": "boolean", "instructions": "Is it ok?"}
    assert d.answers["ok"].probability == 0.98 and d.answers["ok"].value
    assert d.answers["route"].choice == "billing" and d.answers["route"].p() == 0.9
    assert d.answers["urgency"].level == 2 and d.answers["urgency"].p(2) == 0.9
    assert d.has_probabilities and not d.fallback_triggered
    assert d.cost_usd == pytest.approx(0.00001155) and d.input_tokens == 275


def test_jev_detects_language_model_fallback():
    body = json.loads(json.dumps(JEV_BODY))
    body["model"] = "openai/some-llm"
    body["answers"]["route"] = {
        "type": "choice",
        "choice": "tech",
        "probabilities": {},
        "confidence": 0,
    }
    d = parse_response(
        body, {"x-ai-gateway-decision-fallback-triggered": "true"}, QUESTIONS, "typesafe-ai/jev"
    )
    assert d.fallback_triggered and d.model_used == "openai/some-llm"
    assert d.answers["route"].probabilities is None
    assert not d.has_probabilities


def test_jev_missing_answer_is_an_error():
    body = {"model": "typesafe-ai/jev", "answers": {"ok": {"type": "boolean", "probability": 0.5}}}
    with pytest.raises(DeciderError, match="missing answers"):
        parse_response(body, {}, QUESTIONS, "typesafe-ai/jev")


def test_jev_retries_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    jev = JevDecider(
        "k",
        client=client_returning(
            httpx.Response(429, headers={"retry-after": "0"}), httpx.Response(200, json=JEV_BODY)
        ),
    )
    assert jev.decide("s", QUESTIONS).answers["ok"].probability == 0.98


def test_jev_client_error_not_retried():
    jev = JevDecider("k", client=client_returning(httpx.Response(422, json={"message": "bad"})))
    with pytest.raises(DeciderError, match="422"):
        jev.decide("s", QUESTIONS)


def test_jev_fallback_only_sent_when_its_question_is_asked():
    seen: list = []
    fb = {"model": "x/strong", "when": {"question": "verdict", "confidenceBelow": 0.6}}
    jev = JevDecider(
        "k", fallback=fb, client=client_returning(httpx.Response(200, json=JEV_BODY), seen=seen)
    )
    jev.decide("s", QUESTIONS)
    assert "providerOptions" not in seen[0]
    assert jev.payload("s", {"verdict": QUESTIONS["route"]})["providerOptions"] == {
        "gateway": {"models": [fb]}
    }


def _chat(content: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": json.dumps(content)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        },
    )


def test_llm_single_sample_has_no_probabilities():
    seen: list = []
    llm = LLMDecider(
        "k",
        "m",
        "https://x",
        client=client_returning(_chat({"ok": True, "route": "tech", "urgency": "high"}), seen=seen),
    )
    d = llm.decide({"claim": "x"}, QUESTIONS)
    schema = seen[0]["response_format"]["json_schema"]
    assert schema["strict"] and schema["schema"]["additionalProperties"] is False
    assert schema["schema"]["properties"]["urgency"]["enum"] == ["low", "medium", "high"]
    assert d.answers["route"].choice == "tech" and d.answers["route"].probabilities is None
    assert d.answers["urgency"].level == 2
    assert not d.has_probabilities


def test_llm_vote_shares_become_probabilities():
    llm = LLMDecider(
        "k",
        "m",
        "https://x",
        samples=4,
        client=client_returning(
            _chat({"ok": True, "route": "tech", "urgency": "high"}),
            _chat({"ok": True, "route": "tech", "urgency": "medium"}),
            _chat({"ok": False, "route": "billing", "urgency": "high"}),
            _chat({"ok": True, "route": "tech", "urgency": "high"}),
        ),
    )
    d = llm.decide("s", QUESTIONS)
    assert d.answers["ok"].probability == 0.75
    assert d.answers["route"].probabilities == {"billing": 0.25, "tech": 0.75}
    assert d.answers["urgency"].p(2) == 0.75
    assert d.has_probabilities and d.input_tokens == 400


def test_llm_skipped_question_is_an_error():
    llm = LLMDecider("k", "m", "https://x", client=client_returning(_chat({"ok": True})))
    with pytest.raises(DeciderError, match="skipped"):
        llm.decide("s", QUESTIONS)
