"""Fetches evidence pages with SSRF guards and hard limits (review S6, C6).

Every URL goes through here, whether a user gave it or a model chose it:
- http and https only, on the default ports
- the hostname is resolved and every address must be public (no private, loopback,
  link-local, multicast, reserved or carrier-grade NAT ranges)
- redirects are followed by hand, and every hop is checked again
- the body is capped at `max_bytes`; content types other than HTML, plain text and PDF
  are refused

Not covered here: DNS rebinding between our check and httpx's own lookup. Production runs the
fetcher in an isolated egress network, as the design says; this module is the second line.
Failures come back as `FetchResult.error`, never as exceptions, so the probe can count them.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urljoin, urlsplit

import httpx

ALLOWED_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "application/pdf")
DEFAULT_PORTS = {"http": 80, "https": 443}
USER_AGENT = "factcheck-probe/0.1 (+https://github.com/withhumanrevenge-cyber/factcheck)"

Resolver = Callable[[str], list[str]]


def system_resolver(host: str) -> list[str]:
    return sorted({info[4][0] for info in socket.getaddrinfo(host, None)})


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    fetched_at: str
    status: int | None = None
    content_type: str | None = None
    charset: str | None = None
    body: bytes = b""
    truncated: bool = False
    error: str | None = None  # None on success, else a short reason code

    @property
    def ok(self) -> bool:
        return self.error is None


def check_url(url: str, resolve: Resolver = system_resolver) -> str | None:
    """Returns None if the URL may be fetched, else the reason it may not."""
    parts = urlsplit(url)
    if parts.scheme not in DEFAULT_PORTS:
        return "bad_scheme"
    if not parts.hostname:
        return "no_host"
    if parts.username or parts.password:
        return "credentials_in_url"
    try:
        port = parts.port
    except ValueError:
        return "bad_port"
    if port not in (None, DEFAULT_PORTS[parts.scheme]):
        return "bad_port"
    try:
        addresses = [ipaddress.ip_address(parts.hostname)]
    except ValueError:
        try:
            addresses = [ipaddress.ip_address(a.split("%")[0]) for a in resolve(parts.hostname)]
        except (OSError, ValueError):
            return "dns_failed"
    if not addresses:
        return "dns_failed"
    if not all(a.is_global for a in addresses):
        return "blocked_address"
    return None


class Fetcher:
    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        resolve: Resolver = system_resolver,
        max_bytes: int = 2_000_000,
        max_redirects: int = 3,
        timeout_s: float = 10.0,
    ):
        self._client = client or httpx.Client(timeout=timeout_s, follow_redirects=False)
        self._resolve = resolve
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects

    def fetch(self, url: str) -> FetchResult:
        fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
        current = url
        for _ in range(self.max_redirects + 1):
            reason = check_url(current, self._resolve)
            if reason:
                return FetchResult(url, current, fetched_at, error=reason)
            try:
                result = self._get(url, current, fetched_at)
            except httpx.TimeoutException:
                return FetchResult(url, current, fetched_at, error="timeout")
            except httpx.HTTPError:
                return FetchResult(url, current, fetched_at, error="network")
            if isinstance(result, FetchResult):
                return result
            current = result  # a redirect target, checked on the next pass
        return FetchResult(url, current, fetched_at, error="too_many_redirects")

    def _get(self, url: str, current: str, fetched_at: str) -> FetchResult | str:
        headers = {"User-Agent": USER_AGENT, "Accept": ", ".join(ALLOWED_TYPES)}
        with self._client.stream("GET", current, headers=headers) as resp:
            if resp.is_redirect:
                location = resp.headers.get("location")
                if not location:
                    return FetchResult(
                        url, current, fetched_at, resp.status_code, error="bad_redirect"
                    )
                return urljoin(current, location)
            if resp.status_code >= 400:
                return FetchResult(
                    url, current, fetched_at, resp.status_code, error=f"http_{resp.status_code}"
                )
            mime, _, params = resp.headers.get("content-type", "").partition(";")
            mime = mime.strip().lower()
            if mime not in ALLOWED_TYPES:
                return FetchResult(
                    url, current, fetched_at, resp.status_code, mime, error="unsupported_type"
                )
            charset = None
            for param in params.split(";"):
                key, _, value = param.partition("=")
                if key.strip().lower() == "charset":
                    charset = value.strip().strip('"') or None
            body = bytearray()
            truncated = False
            for chunk in resp.iter_bytes():
                body.extend(chunk)
                if len(body) > self.max_bytes:
                    del body[self.max_bytes :]
                    truncated = True
                    break
            return FetchResult(
                url, current, fetched_at, resp.status_code, mime, charset, bytes(body), truncated
            )
