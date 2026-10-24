"""Settings read from the environment (and a local .env file, if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

GATEWAY_URL = "https://ai-gateway.vercel.sh"
GATEWAY_LLM_URL = "https://ai-gateway.vercel.sh/v1"
GROQ_URL = "https://api.groq.com/openai/v1"


@dataclass(frozen=True)
class Settings:
    ai_gateway_api_key: str | None
    groq_api_key: str | None
    jev_model: str
    groq_model: str
    strong_model: str | None
    llm_samples: int
    zero_data_retention: bool


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        ai_gateway_api_key=os.environ.get("AI_GATEWAY_API_KEY") or None,
        groq_api_key=os.environ.get("GROQ_API_KEY") or None,
        jev_model=os.environ.get("JEV_MODEL") or "typesafe-ai/jev",
        groq_model=os.environ.get("GROQ_MODEL") or "openai/gpt-oss-120b",
        strong_model=os.environ.get("STRONG_MODEL") or None,
        llm_samples=int(os.environ.get("LLM_SAMPLES") or 5),
        zero_data_retention=(os.environ.get("ZERO_DATA_RETENTION", "").lower() == "true"),
    )


def require(value: str | None, name: str) -> str:
    if not value:
        raise SystemExit(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value
