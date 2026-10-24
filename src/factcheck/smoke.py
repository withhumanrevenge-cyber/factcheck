"""Phase 0 smoke test: one real Jev request through the Gateway, plus the decision-model list.

    uv run python -m factcheck.smoke

Prints the model that answered, the answer, latency and cost, then the decision models your
key can use (look there for a versioned Jev slug to pin).
"""

from __future__ import annotations

import json

import httpx

from factcheck.config import GATEWAY_URL, load_settings, require
from factcheck.decider import JevDecider
from factcheck.questions import VERDICT
from factcheck.state import claim_state


def main() -> int:
    s = load_settings()
    key = require(s.ai_gateway_api_key, "AI_GATEWAY_API_KEY")
    jev = JevDecider(key, s.jev_model, zero_data_retention=s.zero_data_retention)
    state = claim_state("Water boils at 100 °C at sea level.", "2026-10-24", [])
    d = jev.decide(state, {"verdict": VERDICT})
    print(f"model requested: {d.model_requested}")
    print(f"model used:      {d.model_used} (fallback: {d.fallback_triggered})")
    print(f"answer:          {d.answers['verdict']}")
    print(f"latency:         {d.latency_ms:.0f} ms")
    print(f"tokens in/out:   {d.input_tokens}/{d.output_tokens}")
    print(f"cost:            ${d.cost_usd}")

    resp = httpx.get(
        f"{GATEWAY_URL}/typesafe/v1/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
    )
    print(f"\nGET /typesafe/v1/models -> {resp.status_code}")
    try:
        print(json.dumps(resp.json(), indent=2)[:4000])
    except ValueError:
        print(resp.text[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
