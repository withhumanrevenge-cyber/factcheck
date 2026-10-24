# Phase 0 findings

Checked 2026-10-24 against the providers' docs (sources at the end). Three answers change the design: the Gateway does report fallbacks, Jev should not do date reasoning, and Groq's base developer limits are tight enough to shape the whole retrieval design.

## Answered

| Question | Answer | What the code does |
| --- | --- | --- |
| Which Gateway API to call | Decision API, `POST https://ai-gateway.vercel.sh/v1/evaluate`, recommended for new code. The TypeSafe-compatible `/typesafe/v1/systemone` is the same capability with `noul` naming | `decider/jev.py` uses `/v1/evaluate`: `boolean` answers carry `probability`, Choice and Score carry `probabilities` |
| Does a fallback say which model answered? (review Q7) | Yes. Top-level `model` names the fallback model when it answered, and `x-ai-gateway-decision-fallback-triggered: true` is set. A language-model fallback returns `confidence: 0` and `probabilities: {}` for Choice and Score, meaning "unavailable" | Any fallback answer is a veto only: the claim is shown as Unverifiable (`gates.py`). Empty probabilities are read as "none" |
| Jev request limits | State plus all questions about 64k tokens; state plus any one question about 32k. A long-context header raises this but is "danger zone" | Fits 8 items with context easily; no long-context header |
| Jev weaknesses | Literal reading; unreliable counting, arithmetic and dates; Choice always returns an allowed option; text in state can sway answers | An outdated verdict is checked against dates in code (`gates.outdated_date_rule`), not Jev's reading of them. Every shown verdict is cross-checked against stances in code |
| Jev price and speed | $0.042 per million input tokens, output free; 70–500 ms, mostly about 100 ms | Cost read from `providerMetadata.gateway.cost` per call |
| Zero data retention | Gateway `providerOptions.gateway.zeroDataRetention: true`; needs a Pro or Enterprise plan | `ZERO_DATA_RETENTION=true` sets it |
| Direct TypeSafe access | New signups paused since September 22, 2026 | Use the Gateway only |
| Groq strict JSON output | Strict `json_schema` only on `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, `meta-models/muse-spark-open`; not the Llama models. **Not supported with tool use or streaming** | Arm B uses `openai/gpt-oss-120b`. Retrieval sub-agents will need a tool-calling turn and then a separate structured-extraction call |
| Groq tool use | All models support tools; gpt-oss models do not run tools in parallel, Llama and Qwen do | Matters for Track B's sub-agent design |
| Groq rate limits (Developer plan base) | gpt-oss-120b, gpt-oss-20b, qwen3.8-27b: 30 RPM, 1K RPD, **8K tokens/min**, 200K tokens/day. Whisper: 20 RPM, 2K RPD, 7,200 audio-seconds/hour | Runner defaults to one bundle at a time; the LLM decider honours `retry-after` on 429 |

## Risks this raises

1. **Groq's token limits may not carry a hosted product.** At 8K tokens a minute, one claim with a few fetched pages can use a whole minute's budget, and 200K tokens a day is a few dozen checks. Check your account's actual limits on the console's limits page and what higher tiers allow before Track B; otherwise plan a second LLM provider behind the same interface from the start.
2. **Gateway escalation (arm D) cannot produce shown verdicts.** Because a fallback answer is only a veto, arm D can only lower coverage in production. The spike still measures its raw accuracy, which is useful if the policy changes.
3. **Jev and dates.** The outdated verdict leans on date reasoning, which Jev is documented as weak at, so code now compares dates before an outdated verdict can be shown. The page date it relies on is `datePublished`, which on some sites (Wikipedia, for one) is the page's creation date, not its last edit; `dateModified` may need to count too.

## Still open

| Item | How to close it |
| --- | --- |
| A versioned Jev slug to pin | Run `uv run python -m factcheck.smoke` with a key; it lists the decision models available |
| Jev rate limits for our organisation | Shown in the TypeSafe or Gateway console, not published |
| Groq prices per token and Whisper per hour | The pricing page did not render for the doc reader; read it in a browser |
| Search API and its price per query | Choose before Track B |
| Monthly budget for the spend caps | Your call |
| Other decision models to add as arms | Gateway also lists Laya (free through October 31), Microsoft Decision-1, Liquid AI d1 and d1-ultra, Perplexity Decider V1.2 and Baseten Mammoth-1 / Elephant-1. Any can be an extra `JevDecider(model=...)` arm |

## Sources

- [TypeSafe API with AI Gateway](https://vercel.com/docs/ai-gateway/sdks-and-apis/typesafe), Vercel docs, updated 2026-10-05
- [Decision](https://vercel.com/docs/ai-gateway/modalities/decision), Vercel docs, updated 2026-10-07
- [How to build typed decisions with Jev](https://flaviocopes.com/jev/), Flavio Copes
- [How to use Jev with the Vercel AI SDK](https://flaviocopes.com/jev-vercel-ai-sdk/), Flavio Copes
- [Groq models](https://console.groq.com/docs/models), [structured outputs](https://console.groq.com/docs/structured-outputs), [tool use](https://console.groq.com/docs/tool-use), [rate limits](https://console.groq.com/docs/rate-limits)
