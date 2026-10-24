"""Prints a Markdown comparison of the arms in one run directory.

uv run python -m factcheck.eval.report runs/20261024-120000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from factcheck.eval.metrics import summarise


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _ci(ci: tuple[float, float] | None) -> str:
    return "" if ci is None else f" ({100 * ci[0]:.0f}–{100 * ci[1]:.0f})"


def _num(x: float | None, fmt: str = "{:.3f}") -> str:
    return "n/a" if x is None else fmt.format(x)


def render(summaries: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Arm | Bundles | Errors | Stance acc. (95% CI) | Raw verdict acc. (95% CI) | Coverage "
        "| Selective acc. (95% CI) | ECE | AURC | Acc. @50% cov. | Median latency | Cost |",
        "| --- | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, s in summaries.items():
        lines.append(
            f"| {arm} | {s['n']} | {s['errors']} "
            f"| {_pct(s['stance_accuracy'])}{_ci(s['stance_accuracy_ci'])} "
            f"| {_pct(s['raw_verdict_accuracy'])}{_ci(s['raw_verdict_accuracy_ci'])} "
            f"| {_pct(s['coverage'])} "
            f"| {_pct(s['selective_accuracy'])}{_ci(s['selective_accuracy_ci'])} "
            f"| {_num(s['ece'])} | {_num(s['aurc'])} | {_pct(s['accuracy_at_50pct_coverage'])} "
            f"| {_num(s['median_latency_ms'], '{:.0f} ms')} | ${s['total_cost_usd']:.4f} |"
        )
    lines.append("")
    for arm, s in summaries.items():
        lines.append(
            f"**{arm}**: predicted {dict(s['predicted_verdicts'])}; "
            f"labels {dict(s['label_verdicts'])}; "
            f"withheld {dict(s['withheld_reasons'])}; "
            f"{s['with_probabilities_n']} verdicts with probabilities"
        )
        filled = [b for b in s["calibration_bins"] if b.n]
        if filled:
            lines.append(
                "  calibration: "
                + "; ".join(
                    f"{b.lo:.1f}–{b.hi:.1f}: n={b.n}, conf {b.mean_confidence:.2f}, acc {b.accuracy:.2f}"
                    for b in filled
                )
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    run_dir = Path(args[0])
    summaries = {}
    for path in sorted(run_dir.glob("*.jsonl")):
        records = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
        summaries[path.stem] = summarise(records)
    if not summaries:
        print(f"no .jsonl files in {run_dir}", file=sys.stderr)
        return 1
    print(render(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
