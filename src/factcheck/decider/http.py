"""POST with retries on rate limits, overload and server errors."""

from __future__ import annotations

import time
from typing import Any

import httpx

from factcheck.decider.base import DeciderError

RETRY_STATUSES = {429, 500, 502, 503, 504, 529}


def post_json(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    max_retries: int = 2,
    backoff_s: float = 1.0,
) -> httpx.Response:
    attempt = 0
    while True:
        try:
            resp = client.post(url, headers=headers, json=payload)
        except httpx.TransportError as e:
            if attempt >= max_retries:
                raise DeciderError(f"request to {url} failed: {e}") from e
            time.sleep(backoff_s * 2**attempt)
            attempt += 1
            continue
        if resp.status_code < 400:
            return resp
        if resp.status_code in RETRY_STATUSES and attempt < max_retries:
            wait = _retry_after(resp)
            time.sleep(wait if wait is not None else backoff_s * 2**attempt)
            attempt += 1
            continue
        raise DeciderError(f"{url} returned {resp.status_code}: {resp.text[:500]}")


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None
