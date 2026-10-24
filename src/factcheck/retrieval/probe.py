"""Track B retrieval probe: does real evidence survive the span check, and what does it cost?

    uv run python -m factcheck.retrieval.probe data/probe/claims.jsonl

Input is JSONL, one claim per line, with the URLs to try:
    {"claim_id": "c1", "claim": "...", "as_of": "2026-10-24", "urls": ["https://..."]}
The URLs can come from a search API or be picked by hand; the search API is not chosen yet.

For each URL: fetch (with SSRF guards), extract text and date, store a snapshot, keep the
passages nearest the claim, ask Groq for quotes, then check every quote against the snapshot.
Writes records.jsonl and summary.md to runs/probe-<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from factcheck.config import GROQ_URL, load_settings, require
from factcheck.decider.base import DeciderError
from factcheck.evidence import tier_for_domain
from factcheck.policy import DEFAULT_POLICY
from factcheck.retrieval.fetch import Fetcher
from factcheck.retrieval.pages import page_from_fetch, select_passages
from factcheck.retrieval.quotes import QuoteExtractor, TokenPacer
from factcheck.retrieval.snapshots import Snapshot, SnapshotStore, content_hash
from factcheck.spancheck import check_span


def probe_url(
    claim: str,
    url: str,
    *,
    fetcher: Fetcher,
    extractor: QuoteExtractor,
    store: SnapshotStore,
    tiers: dict[str, int],
    max_chars: int,
) -> dict[str, Any]:
    domain = (urlsplit(url).hostname or "").lower()
    tier, tier_source = tier_for_domain(domain, tiers)
    rec: dict[str, Any] = {"url": url, "domain": domain, "tier": tier, "tier_source": tier_source}

    fetched = fetcher.fetch(url)
    rec.update(
        final_url=fetched.final_url,
        status=fetched.status,
        content_type=fetched.content_type,
        bytes=len(fetched.body),
        truncated=fetched.truncated,
        fetch_error=fetched.error,
    )
    page = page_from_fetch(fetched)
    rec.update(
        page_error=page.error if fetched.ok else None,  # a fetch failure is counted once
        chars=len(page.text),
        published_at=page.published_at.isoformat() if page.published_at else None,
        date_source=page.date_source,
        claim_reviews=page.claim_reviews,
        quotes=[],
    )
    if not page.text:
        if fetched.ok:
            rec["page_error"] = page.error or "empty_text"
        return rec

    snap = Snapshot(
        url=url,
        final_url=fetched.final_url,
        fetched_at=fetched.fetched_at,
        content_hash=content_hash(page.text),
        text=page.text,
        title=page.title,
        published_at=page.published_at,
        date_source=page.date_source,
        claim_reviews=page.claim_reviews,
        truncated=fetched.truncated,
    )
    store.save(snap)
    rec["content_hash"] = snap.content_hash

    passages = select_passages(page.text, claim, max_chars)
    rec["chars_sent"] = len(passages)
    try:
        ext = extractor.extract(claim, passages)
    except DeciderError as e:
        rec["extract_error"] = str(e)[:300]
        return rec
    rec.update(input_tokens=ext.input_tokens, output_tokens=ext.output_tokens)
    for q in ext.quotes:
        check = check_span(snap.text, q.quote, min_words=DEFAULT_POLICY.min_span_words)
        rec["quotes"].append(
            {
                "quote": q.quote,
                "sub_agent_stance": q.stance,
                "verified": check.verified,
                "reason": check.reason,
                "context": check.context,
            }
        )
    return rec


def summarise(claims: list[dict[str, Any]]) -> dict[str, Any]:
    urls = [u for c in claims for u in c["urls"]]
    quotes = [q for u in urls for q in u.get("quotes", [])]
    verified = [q for q in quotes if q["verified"]]
    tokens = [
        sum((u.get("input_tokens") or 0) + (u.get("output_tokens") or 0) for u in c["urls"])
        for c in claims
    ]
    with_evidence = [
        c
        for c in claims
        if any(
            q["verified"] and q["sub_agent_stance"] != "unrelated"
            for u in c["urls"]
            for q in u.get("quotes", [])
        )
    ]
    fetched = [u for u in urls if not u.get("fetch_error")]
    return {
        "claims": len(claims),
        "urls": len(urls),
        "fetch_ok": len(fetched),
        "fetch_errors": Counter(u["fetch_error"] for u in urls if u.get("fetch_error")),
        "page_errors": Counter(u["page_error"] for u in fetched if u.get("page_error")),
        "extract_errors": sum(1 for u in urls if u.get("extract_error")),
        "pages_with_date": sum(1 for u in fetched if u.get("published_at")),
        "pages_with_claimreview": sum(1 for u in fetched if u.get("claim_reviews")),
        "quotes": len(quotes),
        "quotes_verified": len(verified),
        "span_pass_rate": len(verified) / len(quotes) if quotes else None,
        "span_fail_reasons": Counter(q["reason"] for q in quotes if not q["verified"]),
        "dropped_by_stance": Counter(q["sub_agent_stance"] for q in quotes if not q["verified"]),
        "claims_with_verified_evidence": len(with_evidence),
        "tokens_per_claim_mean": sum(tokens) / len(tokens) if tokens else None,
        "tokens_per_claim_max": max(tokens) if tokens else None,
    }


def render(s: dict[str, Any]) -> str:
    def pct(n: int, d: int) -> str:
        return f"{n}/{d} ({100 * n / d:.0f}%)" if d else f"{n}/0"

    rows = [
        ("Claims with at least one verified, relevant quote", pct(s["claims_with_verified_evidence"], s["claims"])),
        ("Span pass rate", pct(s["quotes_verified"], s["quotes"])),
        ("Pages fetched", pct(s["fetch_ok"], s["urls"])),
        ("Fetched pages with a metadata date", pct(s["pages_with_date"], s["fetch_ok"])),
        ("Fetched pages with ClaimReview", pct(s["pages_with_claimreview"], s["fetch_ok"])),
        ("Groq tokens per claim, mean", f"{s['tokens_per_claim_mean']:.0f}" if s["tokens_per_claim_mean"] else "n/a"),
        ("Groq tokens per claim, max", str(s["tokens_per_claim_max"] or "n/a")),
        ("Extraction errors", str(s["extract_errors"])),
        ("Fetch errors", ", ".join(f"{k} {v}" for k, v in s["fetch_errors"].items()) or "none"),
        ("Page errors", ", ".join(f"{k} {v}" for k, v in s["page_errors"].items()) or "none"),
        ("Span failures", ", ".join(f"{k} {v}" for k, v in s["span_fail_reasons"].items()) or "none"),
        ("Dropped quotes by sub-agent stance", ", ".join(f"{k} {v}" for k, v in s["dropped_by_stance"].items()) or "none"),
    ]  # fmt: skip
    return "\n".join(
        ["| Measure | Result |", "| --- | --- |", *(f"| {a} | {b} |" for a, b in rows)]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("claims", type=Path, help="JSONL file of claims with URLs")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--tiers", type=Path, default=Path("data/source_tiers.json"))
    ap.add_argument("--max-urls", type=int, default=5, help="URLs tried per claim")
    ap.add_argument("--max-chars", type=int, default=12000, help="page text sent to Groq")
    ap.add_argument("--tokens-per-minute", type=int, default=8000, help="Groq budget to pace to")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    s = load_settings()
    extractor = QuoteExtractor(
        require(s.groq_api_key, "GROQ_API_KEY"),
        s.groq_model,
        GROQ_URL,
        pacer=TokenPacer(args.tokens_per_minute),
    )
    tiers = json.loads(args.tiers.read_text(encoding="utf-8")) if args.tiers.exists() else {}
    lines = [ln for ln in args.claims.read_text(encoding="utf-8").splitlines() if ln.strip()]
    claims_in = [json.loads(ln) for ln in lines][: args.limit]
    if not claims_in:
        print("no claims to probe", file=sys.stderr)
        return 1

    run_dir = args.out / f"probe-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    store = SnapshotStore(run_dir / "snapshots")
    fetcher = Fetcher()
    results = []
    with (run_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for c in claims_in:
            urls = []
            for url in c["urls"][: args.max_urls]:
                rec = probe_url(
                    c["claim"],
                    url,
                    fetcher=fetcher,
                    extractor=extractor,
                    store=store,
                    tiers=tiers,
                    max_chars=args.max_chars,
                )
                quotes = rec["quotes"]
                status = rec.get("fetch_error") or rec.get("page_error") or rec.get("extract_error")
                if not status:
                    status = f"{sum(q['verified'] for q in quotes)}/{len(quotes)} quotes verified"
                print(f"[{c['claim_id']}] {url}: {status}")
                urls.append(rec)
            claim_rec = {**c, "urls": urls}
            results.append(claim_rec)
            f.write(json.dumps(claim_rec, default=str) + "\n")

    summary = render(summarise(results))
    (run_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    print(f"\nwrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
