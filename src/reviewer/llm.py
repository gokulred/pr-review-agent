import os
import re

import groq
from langchain_core.runnables import Runnable, RunnableLambda

# Gemini's free tier allowed only 20 requests/day for gemini-3.8-flash (one agent review uses 3-9),
# but quotas are per model, so a list of models gives the fallback several separate quotas.
GEMINI_MODELS = "gemini-3.8-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite"


def chat_models() -> list:
    """Groq first (fast), then each Gemini model in GEMINI_MODEL (comma-separated) as fallbacks.

    Both clients already retry with backoff, so we only cap retries and add a timeout:
    an overloaded free tier (503/504) should fail over fast, not hang for minutes.
    """
    models = []
    if os.environ.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        models.append(ChatGroq(model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
                               temperature=0, max_retries=1, timeout=30))
    if os.environ.get("GEMINI_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        models += [ChatGoogleGenerativeAI(model=name.strip(), google_api_key=os.environ["GEMINI_API_KEY"],
                                          temperature=0, max_retries=1, timeout=60)
                   for name in os.environ.get("GEMINI_MODEL", GEMINI_MODELS).split(",") if name.strip()]
    if not models:
        raise RuntimeError("Set GEMINI_API_KEY or GROQ_API_KEY (see .env.example)")
    return models


def with_fallbacks(chains: list) -> Runnable:
    """Try each chain in order. If all fail, raise the first error with every other provider's
    error attached as a note, so a broken fallback is visible instead of hidden behind Groq's error."""
    # Rate limits (429) are worth waiting out: Groq's free tier is 8k tokens/min and the agent
    # resends its prompt every turn. Timeouts and 5xx are not retried here, so an outage still
    # fails over to the next provider quickly.
    chains = [c.with_retry(retry_if_exception_type=(groq.RateLimitError,), wait_exponential_jitter=True,
                           stop_after_attempt=4) for c in chains]
    if len(chains) == 1:
        return chains[0]

    # Providers that failed earlier in this review (an agent review makes up to 9 calls through one
    # chain) are skipped, so a used-up daily quota isn't re-tried, and waited on, at every turn.
    # If every provider has failed, all are tried again.
    failed: set[int] = set()

    def run(value, config):
        errors = []
        order = [i for i in range(len(chains)) if i not in failed] or range(len(chains))
        for i in order:
            try:
                result = chains[i].invoke(value, config)
                failed.discard(i)
                return result
            except Exception as e:  # noqa: BLE001 - any provider failure means: try the next one
                failed.add(i)
                errors.append(e)
        for e in errors[1:]:
            errors[0].add_note(f"Fallback also failed: {type(e).__name__}: {str(e)[:300]}")
        raise errors[0]

    return RunnableLambda(run)


def structured_llm(schema) -> Runnable:
    """invoke() returns {"raw": AIMessage, "parsed": schema | None, "parsing_error": ...}."""
    # Groq: json_schema, because gpt-oss in function-calling mode sometimes calls a
    # nonexistent "commentary" tool and Groq rejects the reply with a 400.
    return with_fallbacks([
        m.with_structured_output(schema, include_raw=True,
                                 **({"method": "json_schema"} if type(m).__name__ == "ChatGroq" else {}))
        for m in chat_models()
    ])


def describe(e: BaseException) -> str:
    """One-line error text including the notes with_fallbacks attaches for each failed fallback.
    Provider account IDs are masked: this text ends up in the UI and in committed eval results."""
    text = " | ".join([f"{type(e).__name__}: {str(e)[:300]}", *getattr(e, "__notes__", [])])
    return re.sub(r"\borg_[A-Za-z0-9]+", "org_<redacted>", text)
