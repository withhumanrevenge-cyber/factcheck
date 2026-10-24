import json
from datetime import date

import httpx
import pytest

from factcheck.retrieval.fetch import Fetcher, check_url
from factcheck.retrieval.pages import page_from_fetch, page_from_html, select_passages
from factcheck.retrieval.probe import probe_url, summarise
from factcheck.retrieval.quotes import QuoteExtractor, TokenPacer
from factcheck.retrieval.snapshots import SnapshotStore

PUBLIC = "93.184.215.14"


def resolver(host: str) -> list[str]:
    table = {
        "internal.example": ["10.0.0.5"],
        "localhost": ["127.0.0.1"],
        "metadata.example": ["169.254.169.254"],
    }
    if host == "nowhere.example":
        raise OSError("no such host")
    return table.get(host, [PUBLIC])


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("https://news.example/a", None),
        (f"http://{PUBLIC}/a", None),
        ("http://10.1.2.3/", "blocked_address"),
        ("http://localhost/admin", "blocked_address"),
        ("http://metadata.example/latest", "blocked_address"),
        ("http://[::1]/", "blocked_address"),
        ("http://100.64.0.1/", "blocked_address"),
        ("ftp://news.example/a", "bad_scheme"),
        ("file:///etc/passwd", "bad_scheme"),
        ("https://news.example:8443/a", "bad_port"),
        ("https://user:pw@news.example/a", "credentials_in_url"),
        ("https://nowhere.example/a", "dns_failed"),
    ],
)
def test_check_url(url, reason):
    assert check_url(url, resolver) == reason


def fetcher_for(handler, **kw) -> Fetcher:
    return Fetcher(
        client=httpx.Client(transport=httpx.MockTransport(handler)), resolve=resolver, **kw
    )


HTML = {"content-type": "text/html; charset=utf-8"}


def test_fetch_follows_redirect_and_rechecks_each_hop():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/start":
            return httpx.Response(302, headers={"location": "/next"})
        if req.url.path == "/next":
            return httpx.Response(301, headers={"location": "http://internal.example/secret"})
        raise AssertionError("must not reach the internal host")

    r = fetcher_for(handler).fetch("https://news.example/start")
    assert r.error == "blocked_address" and r.final_url == "http://internal.example/secret"


def test_fetch_success_and_limits():
    def handler(req: httpx.Request) -> httpx.Response:
        return {
            "/ok": httpx.Response(200, headers=HTML, content=b"<p>hello world</p>"),
            "/big": httpx.Response(200, headers=HTML, content=b"x" * 50),
            "/img": httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG"),
            "/gone": httpx.Response(404),
            "/loop": httpx.Response(302, headers={"location": "/loop"}),
        }[req.url.path]

    f = fetcher_for(handler, max_bytes=20)
    ok = f.fetch("https://news.example/ok")
    assert ok.ok and ok.body == b"<p>hello world</p>" and ok.charset == "utf-8"
    big = f.fetch("https://news.example/big")
    assert big.ok and big.truncated and len(big.body) == 20
    assert f.fetch("https://news.example/img").error == "unsupported_type"
    assert f.fetch("https://news.example/gone").error == "http_404"
    assert f.fetch("https://news.example/loop").error == "too_many_redirects"


ARTICLE = """<html><head><title>Bridge update</title>
<meta property="article:published_time" content="2026-09-04T08:00:00Z">
<script type="application/ld+json">{"@context":"https://schema.org","@graph":[
 {"@type":"ClaimReview","claimReviewed":"The bridge reopened in March","datePublished":"2026-09-20",
  "author":{"@type":"Organization","name":"Example Fact Desk"},"reviewRating":{"alternateName":"False"}}]}</script>
<script>var tracking = "ignore me";</script></head>
<body><nav>Home | News</nav><article><h1>Bridge still shut</h1>
<p>Officials confirmed the bridge has not reopened to cars.</p><p>Repairs run to November.</p></article>
<footer>Copyright</footer></body></html>"""


def test_page_from_html_extracts_text_dates_and_claimreview():
    page = page_from_html(ARTICLE)
    assert page.title == "Bridge update"
    assert "Officials confirmed the bridge has not reopened to cars." in page.text
    assert "ignore me" not in page.text and "Home | News" not in page.text
    assert "Copyright" not in page.text
    assert page.published_at == date(2026, 9, 20)  # JSON-LD first
    assert page.date_source == "page_metadata"
    assert page.claim_reviews == [
        {
            "claim": "The bridge reopened in March",
            "rating": "False",
            "author": "Example Fact Desk",
            "date": "2026-09-20",
        }
    ]


def test_page_date_falls_back_to_meta_then_time():
    meta = '<meta name="date" content="2025-01-02"><p>x</p>'
    assert page_from_html(meta).published_at == date(2025, 1, 2)
    time_tag = '<p>Posted <time datetime="2024-03-05">March 5</time></p>'
    assert page_from_html(time_tag).published_at == date(2024, 3, 5)
    assert page_from_html("<p>no date</p>").date_source == "unknown"


def test_pdf_is_stored_but_not_extracted_yet():
    from factcheck.retrieval.fetch import FetchResult

    r = FetchResult("u", "u", "t", 200, "application/pdf", None, b"%PDF")
    assert page_from_fetch(r).error == "pdf_not_supported_yet"


def test_select_passages_keeps_relevant_paragraphs_in_order():
    text = "\n".join(
        ["Weather was mild today."] * 5
        + ["The harbour bridge reopened to cars in March."]
        + ["Sports results follow."] * 5
    )
    out = select_passages(text, "Harbour bridge reopened in March", max_chars=80)
    assert "harbour bridge reopened" in out
    assert len(out) <= 80


def test_token_pacer_waits_for_the_window():
    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    pacer = TokenPacer(1000, clock=lambda: now[0], sleep=sleep)
    pacer.wait_for(600)
    pacer.record(600)
    now[0] = 10
    pacer.wait_for(600)  # would exceed 1000 within the minute
    assert slept and now[0] >= 60
    pacer.record(600)


def chat(content: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": json.dumps(content)}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 60},
        },
    )


def test_probe_url_verifies_quotes_against_the_snapshot(tmp_path):
    def page_handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=HTML, content=ARTICLE.encode())

    seen = []

    def groq_handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return chat({"quotes": [
            {"quote": "Officials confirmed the bridge has not reopened to cars.", "stance": "refutes"},
            {"quote": "Officials said the bridge is still closed to all cars.", "stance": "refutes"},
        ]})  # fmt: skip

    extractor = QuoteExtractor(
        "k",
        "openai/gpt-oss-120b",
        "https://groq.example",
        client=httpx.Client(transport=httpx.MockTransport(groq_handler)),
    )
    rec = probe_url(
        "The bridge reopened in March",
        "https://news.example/bridge",
        fetcher=fetcher_for(page_handler),
        extractor=extractor,
        store=SnapshotStore(tmp_path),
        tiers={"news.example": 2},
        max_chars=12000,
    )
    assert seen[0]["response_format"]["json_schema"]["strict"] is True
    assert rec["tier"] == 2 and rec["published_at"] == "2026-09-20"
    assert [q["verified"] for q in rec["quotes"]] == [True, False]
    assert rec["quotes"][1]["reason"] == "not_found"
    assert SnapshotStore(tmp_path).get(rec["content_hash"]).title == "Bridge update"

    s = summarise([{"claim_id": "c1", "claim": "x", "urls": [rec]}])
    assert s["span_pass_rate"] == 0.5
    assert s["claims_with_verified_evidence"] == 1
    assert s["tokens_per_claim_max"] == 960
    assert s["dropped_by_stance"] == {"refutes": 1}


def test_probe_url_records_fetch_failure_without_calling_groq(tmp_path):
    def groq_handler(req):
        raise AssertionError("Groq must not be called")

    extractor = QuoteExtractor(
        "k",
        "m",
        "https://groq.example",
        client=httpx.Client(transport=httpx.MockTransport(groq_handler)),
    )
    rec = probe_url(
        "c",
        "http://internal.example/x",
        fetcher=fetcher_for(lambda r: None),
        extractor=extractor,
        store=SnapshotStore(tmp_path),
        tiers={},
        max_chars=100,
    )
    assert rec["fetch_error"] == "blocked_address" and rec["quotes"] == []
    assert rec["page_error"] is None
