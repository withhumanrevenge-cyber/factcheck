from factcheck.spancheck import check_span, normalise

PAGE = (
    "The council met on Tuesday. The Harbour Bridge will stay closed to all vehicles until "
    "repairs are finished in November 2026. Detours are signposted. A walkway stays open."
)


def test_exact_quote_passes_with_context():
    r = check_span(PAGE, "will stay closed to all vehicles until repairs are finished")
    assert r.verified and r.reason == "ok"
    assert r.context.startswith("the council met on tuesday.")
    assert r.context.endswith("detours are signposted.")


def test_typography_differences_pass():
    page = "Officials said “the bridge’s deck is safe” — for now, at least."
    assert check_span(page, 'Officials said "the bridge\'s deck is safe" - for now').verified


def test_soft_hyphen_and_pdf_line_break_hyphenation_pass():
    page = "The pro­gramme ended. Its infor-\nmation was later re-\n  published in full by the city."
    assert check_span(page, "Its information was later republished in full by the city").verified


def test_ellipsis_fragments_must_appear_in_order():
    assert check_span(PAGE, "The Harbour Bridge will stay closed ... in November 2026").verified
    assert check_span(PAGE, "The Harbour Bridge will stay closed […] in November 2026").verified
    reordered = check_span(PAGE, "repairs are finished in November 2026 ... The council met on")
    assert not reordered.verified and reordered.reason == "not_found"


def test_altered_quote_fails():
    r = check_span(PAGE, "The Harbour Bridge reopened to all vehicles in March 2026")
    assert not r.verified and r.reason == "not_found"


def test_short_span_rejected():
    r = check_span(PAGE, "Detours are signposted")
    assert not r.verified and r.reason == "too_short"
    assert check_span(PAGE, "Detours are signposted", min_words=3).verified


def test_empty_span():
    assert check_span(PAGE, " … ").reason == "empty"


def test_normalise_collapses_whitespace_and_case():
    assert normalise("  A\tB\n\nC  ") == "a b c"
