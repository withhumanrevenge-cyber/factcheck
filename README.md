# factcheck

A fact-check engine. Groq-hosted LLMs do the language work, Jev (a decision model reached through Vercel AI Gateway) makes the typed decisions, and plain code checks citations and gates what users see.

The design, its architecture review and the phased work plan live in the team doc ("Jev and Groq Fact-Check Engine Design"). This repo is at **Phase 1**: the decider spike that settles whether Jev earns the verdict role.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in AI_GATEWAY_API_KEY and GROQ_API_KEY
uv run pytest
```

On a Windows machine where App Control blocks uv's own Python builds (`os error 4551`), create the venv from a python.org install first, then sync into it:

```bash
py -3.13 -m venv .venv
UV_PYTHON=.venv/Scripts/python.exe uv sync
```

## Commands

| What | Command |
| --- | --- |
| One real Jev request, plus the decision-model list (Phase 0) | `uv run python -m factcheck.smoke` |
| Run the spike on the dev split | `uv run python -m factcheck.eval.runner --arms jev,groq,strong,jev+fallback --split dev` |
| Same, with Choice options shuffled (order sensitivity) | add `--permute-seed 1` |
| Compare arms in a run | `uv run python -m factcheck.eval.report runs/<timestamp>` |

## Layout

| Path | What it holds |
| --- | --- |
| `src/factcheck/decider/` | The Decider interface. `jev.py` calls the Gateway Decision API (`POST /v1/evaluate`); `llm.py` is a chat LLM with a strict JSON schema, for Groq or the Gateway |
| `src/factcheck/questions.py` | The fixed question wording: stance, verdict, sufficient, harm, check-worthy, consistent |
| `src/factcheck/state.py` | Builds decider state from verified evidence only |
| `src/factcheck/spancheck.py` | Quoted-span check against a stored snapshot, with normalisation and a context window |
| `src/factcheck/gates.py` | Code gates between answers and what users see |
| `src/factcheck/policy.py` | Thresholds (placeholders until Phase 4) and the policy version |
| `src/factcheck/eval/` | Bundle format, runner, metrics and report for the spike |
| `data/bundles/` | Hand-built spike bundles; see its README for how to build them |
| `docs/phase0-findings.md` | What the provider docs confirmed, and what is still open |

## Rules the code enforces

- Only span-verified evidence reaches a decider (`state.claim_state` refuses anything else).
- Stance is asked first; the verdict sees only items whose stance counted.
- A verdict is shown only if code finds matching stances from two independent domains, including a tier-1 or tier-2 source, and no equal-or-better source disagrees.
- A Gateway fallback answer, or an LLM answer without probabilities, can only lower what is shown, never raise it.
- Code, not the decider, decides whether to search again.
