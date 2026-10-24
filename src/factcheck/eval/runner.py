"""Runs the decider spike: every bundle through every arm, one JSONL file per arm.

    uv run python -m factcheck.eval.runner --arms jev,groq --split dev

Arms:
  jev           A: Jev through AI Gateway
  groq          B: Groq LLM with a strict JSON schema (vote shares if LLM_SAMPLES > 1)
  strong        C: a strong LLM through the Gateway (STRONG_MODEL)
  jev+fallback  D: Jev, escalating the verdict to STRONG_MODEL below 0.6 confidence

Each bundle runs as production would: stance first, then the verdict on counted items only,
then the code gates. Records keep the raw answers too, so the report can score the decider
before and after the gates.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from factcheck.config import GATEWAY_LLM_URL, GROQ_URL, Settings, load_settings, require
from factcheck.decider import (
    BooleanAnswer,
    ChoiceAnswer,
    ChoiceQuestion,
    Decider,
    DeciderError,
    Decision,
    JevDecider,
    LLMDecider,
    Question,
    ScoreAnswer,
)
from factcheck.eval.bundle import Bundle, load_bundles
from factcheck.gates import counted_items, decide_outcome
from factcheck.policy import DEFAULT_POLICY, Policy
from factcheck.questions import stance_questions, verdict_questions
from factcheck.state import claim_state

ARMS = ("jev", "groq", "strong", "jev+fallback")


def build_arm(arm: str, s: Settings) -> Decider:
    if arm == "jev":
        return JevDecider(
            require(s.ai_gateway_api_key, "AI_GATEWAY_API_KEY"),
            s.jev_model,
            name="jev",
            zero_data_retention=s.zero_data_retention,
        )
    if arm == "groq":
        return LLMDecider(
            require(s.groq_api_key, "GROQ_API_KEY"),
            s.groq_model,
            GROQ_URL,
            name="groq",
            samples=s.llm_samples,
        )
    if arm == "strong":
        return LLMDecider(
            require(s.ai_gateway_api_key, "AI_GATEWAY_API_KEY"),
            require(s.strong_model, "STRONG_MODEL"),
            GATEWAY_LLM_URL,
            name="strong",
            samples=s.llm_samples,
        )
    if arm == "jev+fallback":
        return JevDecider(
            require(s.ai_gateway_api_key, "AI_GATEWAY_API_KEY"),
            s.jev_model,
            name="jev+fallback",
            fallback={
                "model": require(s.strong_model, "STRONG_MODEL"),
                "when": {"question": "verdict", "confidenceBelow": 0.6},
            },
            zero_data_retention=s.zero_data_retention,
        )
    raise SystemExit(f"unknown arm {arm!r}; choose from {', '.join(ARMS)}")


def permute(questions: dict[str, Question], rng: random.Random | None) -> dict[str, Question]:
    """Shuffle Choice options, to measure sensitivity to option order."""
    if rng is None:
        return questions
    out: dict[str, Question] = {}
    for qid, q in questions.items():
        if isinstance(q, ChoiceQuestion):
            keys = list(q.options)
            rng.shuffle(keys)
            q = replace(q, options={k: q.options[k] for k in keys})
        out[qid] = q
    return out


def _answer_dict(a: Any) -> dict[str, Any]:
    if isinstance(a, BooleanAnswer | ChoiceAnswer | ScoreAnswer):
        return asdict(a)
    return {}


def _add_usage(rec: dict[str, Any], d: Decision) -> None:
    for key, value in (
        ("latency_ms", d.latency_ms),
        ("input_tokens", d.input_tokens),
        ("cost_usd", d.cost_usd),
    ):
        if value is not None:
            rec[key] = (rec.get(key) or 0) + value
    rec["models_used"].append(d.model_used)
    rec["fallback_triggered"] = rec["fallback_triggered"] or d.fallback_triggered


def run_bundle(
    bundle: Bundle,
    decider: Decider,
    policy: Policy = DEFAULT_POLICY,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    items = bundle.evidence_items()
    as_of = bundle.as_of.isoformat()
    rec: dict[str, Any] = {
        "bundle_id": bundle.id,
        "split": bundle.split,
        "arm": decider.name,
        "policy_version": policy.version,
        "label_verdict": bundle.label_verdict,
        "label_harm": bundle.label_harm,
        "models_used": [],
        "fallback_triggered": False,
        "stances": [],
        "model_verdict": None,
        "verdict_probability": None,
    }
    try:
        stance_dec = decider.decide(
            claim_state(bundle.claim, as_of, items), permute(stance_questions(len(items)), rng)
        )
        _add_usage(rec, stance_dec)
        for i, e in enumerate(bundle.evidence):
            a = stance_dec.answers[f"stance_{i}"]
            assert isinstance(a, ChoiceAnswer)
            rec["stances"].append(
                {
                    "id": e.id,
                    "label": e.label_stance,
                    "predicted": a.choice,
                    "p": a.p(),
                    "probabilities": a.probabilities,
                }
            )

        counted = counted_items(items, stance_dec, policy)
        rec["counted_ids"] = [c.item.id for c in counted]
        verdict_dec = None
        if counted:
            verdict_dec = decider.decide(
                claim_state(bundle.claim, as_of, [c.item for c in counted]),
                permute(verdict_questions(), rng),
            )
            _add_usage(rec, verdict_dec)
            v = verdict_dec.answers["verdict"]
            assert isinstance(v, ChoiceAnswer)
            rec["model_verdict"] = v.choice
            rec["verdict_probability"] = v.p()
            rec["verdict_answers"] = {k: _answer_dict(a) for k, a in verdict_dec.answers.items()}
        rec["outcome"] = asdict(decide_outcome(verdict_dec, counted, policy, as_of=bundle.as_of))
    except DeciderError as e:
        rec["error"] = str(e)
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--bundles", type=Path, default=Path("data/bundles"))
    ap.add_argument("--arms", default="jev", help=f"comma-separated: {', '.join(ARMS)}")
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include-synthetic", action="store_true", help="also run format examples")
    ap.add_argument("--permute-seed", type=int, default=None, help="shuffle Choice options")
    ap.add_argument("--workers", type=int, default=1, help="parallel bundles per arm")
    args = ap.parse_args(argv)

    settings = load_settings()
    bundles = load_bundles(args.bundles, include_synthetic=args.include_synthetic)
    if args.split != "all":
        bundles = [b for b in bundles if b.split == args.split]
    bundles = bundles[: args.limit]
    if not bundles:
        print("no bundles to run", file=sys.stderr)
        return 1

    deciders = [build_arm(arm.strip(), settings) for arm in args.arms.split(",")]
    run_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    for decider in deciders:
        path = run_dir / f"{decider.name}.jsonl"

        def one(b: Bundle, decider: Decider = decider) -> dict[str, Any]:
            seed = None if args.permute_seed is None else f"{args.permute_seed}:{b.id}"
            return run_bundle(b, decider, DEFAULT_POLICY, random.Random(seed) if seed else None)

        with (
            ThreadPoolExecutor(max_workers=args.workers) as pool,
            path.open("w", encoding="utf-8") as f,
        ):
            for rec in pool.map(one, bundles):
                f.write(json.dumps(rec, default=str) + "\n")
                status = rec.get("error") or rec["outcome"]["shown_verdict"]
                print(f"[{decider.name}] {rec['bundle_id']}: {status}")
        print(f"wrote {path}")
    print(f"report: uv run python -m factcheck.eval.report {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
