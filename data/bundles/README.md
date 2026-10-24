# Spike bundles

One JSON file per claim. The schema is `factcheck.eval.bundle.Bundle`. The files in `examples/` show the format; they are invented and the runner skips them unless you pass `--include-synthetic`.

## How to build a bundle

1. Start from a published fact-check dated **after the models' training cutoffs**, and set `source_published`. Older fact-checks are probably in the models' training data, which inflates their scores.
2. Write the claim as a reader would meet it, not as the fact-check rephrased it.
3. Add 1–8 evidence items. Copy each quote **exactly** from the page, plus one sentence either side as `context`. Include some evidence the way real search returns it: weaker sources, items that repeat the claim, and items that don't help. Don't use only the fact-check's own quotes.
4. Set `tier` from the source, not from how useful the quote is: 1 = primary or official source, 2 = established outlet, 3 = everything else.
5. Label each item's `label_stance` and the bundle's `label_verdict` and `label_harm`. For `mixed` and `missing_context`, have a second person label a subset blind, and record how often the two agree.

## Coverage targets

Aim for 150–200 bundles: at least 20 per verdict, more for `mixed`, `missing_context` and `outdated`. The dev/test split is set automatically from `source_publisher` and `topic`, so keep `topic` the same for every bundle about one story.

## Labels

| Field | Values |
| --- | --- |
| `label_verdict` | supported, refuted, mixed, missing_context, outdated, unverifiable |
| `label_stance` | supports, refutes, mixed, context, superseded, unrelated |
| `label_harm` | low, medium, high |

An item that only repeats the claim without judging it (a debunk quoting the false claim, say) is `unrelated`.
