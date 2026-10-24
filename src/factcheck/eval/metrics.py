"""Spike metrics over runner records.

With ~100–200 bundles, numbers carry wide intervals: report the bootstrap interval with every
accuracy, and the count in every calibration bin.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

UNVERIFIABLE = "unverifiable"


def stance_pairs(records: Sequence[dict[str, Any]]) -> list[tuple[str, str]]:
    return [
        (s["label"], s["predicted"]) for r in records if not r.get("error") for s in r["stances"]
    ]


def verdict_confidences(records: Sequence[dict[str, Any]]) -> list[tuple[float, bool]]:
    """(P(model verdict), correct) for records whose decider gave a probability."""
    out = []
    for r in records:
        p = r.get("verdict_probability")
        if r.get("error") or r.get("model_verdict") is None or p is None:
            continue
        out.append((p, r["model_verdict"] == r["label_verdict"]))
    return out


@dataclass(frozen=True)
class CalibrationBin:
    lo: float
    hi: float
    n: int
    mean_confidence: float | None
    accuracy: float | None


def expected_calibration_error(
    pairs: Sequence[tuple[float, bool]], n_bins: int = 10
) -> tuple[float | None, list[CalibrationBin]]:
    if not pairs:
        return None, []
    bins = []
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        inside = [(p, c) for p, c in pairs if lo <= p < hi or (b == n_bins - 1 and p == 1.0)]
        if not inside:
            bins.append(CalibrationBin(lo, hi, 0, None, None))
            continue
        conf = sum(p for p, _ in inside) / len(inside)
        acc = sum(c for _, c in inside) / len(inside)
        ece += len(inside) / len(pairs) * abs(acc - conf)
        bins.append(CalibrationBin(lo, hi, len(inside), conf, acc))
    return ece, bins


def risk_coverage(
    pairs: Sequence[tuple[float, bool]],
) -> tuple[list[tuple[float, float]], float | None]:
    """Answer the most confident k first; returns (coverage, risk) points and their mean risk (AURC)."""
    if not pairs:
        return [], None
    ranked = sorted(pairs, key=lambda pc: -pc[0])
    points = []
    errors = 0
    for k, (_, correct) in enumerate(ranked, start=1):
        errors += not correct
        points.append((k / len(ranked), errors / k))
    return points, sum(r for _, r in points) / len(points)


def accuracy_at_coverage(pairs: Sequence[tuple[float, bool]], coverage: float) -> float | None:
    points, _ = risk_coverage(pairs)
    eligible = [risk for cov, risk in points if cov <= coverage + 1e-9]
    return 1 - eligible[-1] if eligible else None


def bootstrap_ci(
    values: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float | None],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(n_resamples):
        s = statistic([values[rng.randrange(len(values))] for _ in values])
        if s is not None:
            stats.append(s)
    if not stats:
        return None
    stats.sort()
    lo = stats[int(alpha / 2 * (len(stats) - 1))]
    hi = stats[int((1 - alpha / 2) * (len(stats) - 1))]
    return lo, hi


def mean_correct(pairs: Sequence[tuple[Any, Any]]) -> float | None:
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else None


def summarise(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if not r.get("error")]
    stance = stance_pairs(ok)
    raw = [(r["label_verdict"], r["model_verdict"]) for r in ok if r.get("model_verdict")]
    shown = [(r["label_verdict"], r["outcome"]["shown_verdict"]) for r in ok]
    decided = [(lab, v) for lab, v in shown if v != UNVERIFIABLE]
    conf = verdict_confidences(ok)
    ece, bins = expected_calibration_error(conf)
    _, aurc = risk_coverage(conf)
    return {
        "n": len(records),
        "errors": len(records) - len(ok),
        "stance_n": len(stance),
        "stance_accuracy": mean_correct(stance),
        "stance_accuracy_ci": bootstrap_ci(stance, mean_correct),
        "raw_verdict_n": len(raw),
        "raw_verdict_accuracy": mean_correct(raw),
        "raw_verdict_accuracy_ci": bootstrap_ci(raw, mean_correct),
        "coverage": len(decided) / len(ok) if ok else None,
        "selective_accuracy": mean_correct(decided),
        "selective_accuracy_ci": bootstrap_ci(decided, mean_correct),
        "with_probabilities_n": len(conf),
        "ece": ece,
        "calibration_bins": bins,
        "aurc": aurc,
        "accuracy_at_50pct_coverage": accuracy_at_coverage(conf, 0.5),
        "predicted_verdicts": Counter(v for _, v in raw),
        "label_verdicts": Counter(lab for lab, _ in raw),
        "withheld_reasons": Counter(
            r["outcome"]["reason"] for r in ok if r["outcome"]["shown_verdict"] == UNVERIFIABLE
        ),
        "median_latency_ms": _median([r["latency_ms"] for r in ok if r.get("latency_ms")]),
        "total_cost_usd": sum(r.get("cost_usd") or 0 for r in ok),
        "input_tokens": sum(r.get("input_tokens") or 0 for r in ok),
    }


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2
