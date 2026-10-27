"""LLM router: Groq first, automatic switch to a backup when Groq is slow or failing.

How it works
1. Every call tries the providers in order: Groq main model -> backups.
2. Groq gets a short timeout and no retries, so a stuck call fails fast instead of
   making the customer wait.
3. A provider that fails (timeout, rate limit 429, 5xx, network) is put on a
   "cooldown" and skipped for COOLDOWN_SECONDS. The next provider answers.
4. If Groq answers but takes longer than SLOW_SECONDS, the answer is used, but
   Groq goes on cooldown so the NEXT customers get the faster backup.
5. When the cooldown ends, Groq is tried first again automatically.

Backups (set in .env, all optional):
- FALLBACK_GROQ_MODEL   another Groq model, same key. Groq rate limits are per
                        model, so this helps when the main model hits its limit.
- FALLBACK_BASE_URL / FALLBACK_API_KEY / FALLBACK_MODEL
                        any OpenAI-compatible provider (e.g. Gemini). Helps when
                        Groq itself is slow or down. Needs `langchain-openai`.
"""

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

log = logging.getLogger("brim.router")


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


PRIMARY_TIMEOUT = _f("PRIMARY_TIMEOUT", 10)      # seconds before giving up on Groq
FALLBACK_TIMEOUT = _f("FALLBACK_TIMEOUT", 12)    # seconds for backups
SLOW_SECONDS = _f("SLOW_SECONDS", 6)             # a Groq call slower than this = "slow"
COOLDOWN_SECONDS = _f("COOLDOWN_SECONDS", 120)   # how long to skip a slow/failed provider
MAX_RATE_LIMIT_WAIT = _f("MAX_RATE_LIMIT_WAIT", 8)  # wait this long on a 429 before switching model
MAX_FINAL_WAIT = _f("MAX_FINAL_WAIT", 20)        # if ALL models are rate limited, wait up to this once
MODEL_MISSING_COOLDOWN = 3600                     # a model that doesn't exist (404) is skipped for 1 hour


def _retry_after_seconds(error: Exception) -> float | None:
    """Seconds a 429 asks us to wait ("Please try again in 2.5s" / "in 1m3s" / Retry-After)."""
    text = str(error)
    if "429" not in text and "rate limit" not in text.lower() and "rate_limit" not in text:
        return None
    resp = getattr(error, "response", None)
    try:
        header = resp.headers.get("retry-after") if resp is not None else None
        if header:
            return float(header)
    except (AttributeError, ValueError):
        pass
    m = re.search(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)\s*(ms|s)?", text, re.I)
    if not m:
        return None
    minutes, value, unit = m.group(1), float(m.group(2)), (m.group(3) or "s").lower()
    seconds = value / 1000 if unit == "ms" else value
    return seconds + (int(minutes) * 60 if minutes else 0)


def _is_model_missing(error: Exception) -> bool:
    text = str(error)
    return "404" in text and ("does not exist" in text or "model_not_found" in text or "not found" in text.lower())


def _is_tool_call_error(error: Exception) -> bool:
    text = str(error)
    return "tool_use_failed" in text or "Failed to call a function" in text


def flatten_tool_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Rewrite earlier tool calls and tool results as plain text.

    Gemini 3 rejects tool calls in the history that carry no Gemini "thought signature"
    (calls made by Groq, or its own calls after a LangChain round trip). As text, the
    model still sees what was looked up and can make new tool calls normally."""
    out: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            calls = "; ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in m.tool_calls)
            text = (str(m.content).strip() + "\n" if m.content else "") + f"[I looked this up with: {calls}]"
            out.append(AIMessage(content=text))
        elif isinstance(m, ToolMessage):
            out.append(HumanMessage(content=f"[Result of {m.name or 'tool'}]\n{m.content}"))
        else:
            out.append(m)
    return out


@dataclass
class Provider:
    name: str                 # label shown in logs / Opik, e.g. "groq:llama-3.3-70b-versatile"
    provider: str             # "groq" or "openai_compatible"
    model: str
    timeout: float
    build: callable           # () -> LangChain chat model
    flatten_tools: bool = False  # send earlier tool calls/results as plain text (needed for Gemini)
    _plain: object = None
    _tools: object = None
    down_until: float = 0.0
    last_error: str = ""
    stats: dict = field(default_factory=lambda: {"ok": 0, "failed": 0, "slow": 0})

    def runnable(self, tools, use_tools: bool):
        if self._plain is None:
            self._plain = self.build()
            self._tools = self._plain.bind_tools(tools)
        return self._tools if use_tools else self._plain


class AllProvidersFailed(RuntimeError):
    pass


class LLMRouter:
    def __init__(self, providers: list[Provider], tools: list):
        if not providers:
            raise ValueError("No LLM providers configured")
        self.providers = providers
        self.tools = tools
        self._lock = threading.Lock()

    # ------------------------------------------------------------ cooldowns
    def _cool(self, p: Provider, reason: str, seconds: float | None = None) -> None:
        seconds = COOLDOWN_SECONDS if seconds is None else seconds
        with self._lock:
            p.down_until = time.monotonic() + seconds
            p.last_error = reason
        log.warning("LLM %s on cooldown for %ss: %s", p.name, int(seconds), reason)

    def _order(self) -> list[Provider]:
        """Healthy providers first (in priority order), cooling ones last as a final resort."""
        now = time.monotonic()
        with self._lock:
            healthy = [p for p in self.providers if p.down_until <= now]
            cooling = [p for p in self.providers if p.down_until > now]
        return healthy + cooling

    # ------------------------------------------------------------ call
    def invoke(self, messages: list[BaseMessage], use_tools: bool = True) -> tuple[AIMessage, Provider, list[str]]:
        """Returns (response, provider_used, skipped_reasons)."""
        skipped: list[str] = []
        primary = self.providers[0]
        waits: list[tuple[float, Provider]] = []   # (seconds, provider) from 429 "try again in Xs"
        order = self._order()
        final_round = False
        i = 0
        while i < len(order):
            p = order[i]
            i += 1
            response, took, error = self._try(p, messages, use_tools)
            if error is not None:
                wait = _retry_after_seconds(error)
                if wait is not None:
                    waits.append((wait, p))
                p.stats["failed"] += 1
                if _is_tool_call_error(error):
                    # the model wrote a broken tool call: not a provider problem, no cooldown
                    reason = f"bad tool call: {str(error)[:120]}"
                    log.warning("LLM %s: %s", p.name, reason)
                elif _is_model_missing(error):
                    reason = (f"model '{p.model}' is not available on this account (404). "
                              f"Pick another model from the provider's model list.")
                    self._cool(p, reason, seconds=MODEL_MISSING_COOLDOWN)
                else:
                    reason = f"{type(error).__name__} after {took:.1f}s: {str(error)[:160]}"
                    self._cool(p, reason)
                skipped.append(f"{p.name}: {reason}")
                if i == len(order) and not final_round and waits:
                    # Every model failed, but some only need a short pause (rate limit).
                    # Waiting a bit is better than showing the customer an error.
                    wait, soonest = min(waits, key=lambda w: w[0])
                    if wait <= MAX_FINAL_WAIT:
                        log.info("All models busy; waiting %.1fs for %s, then one more try", wait, soonest.name)
                        time.sleep(wait + 0.3)
                        order.append(soonest)
                        final_round = True
                continue
            p.stats["ok"] += 1
            if p is primary and took > SLOW_SECONDS:
                p.stats["slow"] += 1
                self._cool(p, f"slow: {took:.1f}s (limit {SLOW_SECONDS:.1f}s)")
            elif p.down_until and p.down_until > time.monotonic() and took <= SLOW_SECONDS:
                # it was cooling but answered fast as a last resort -> healthy again
                with self._lock:
                    p.down_until = 0.0
            return response, p, skipped
        raise AllProvidersFailed("All LLM providers failed: " + " | ".join(skipped))

    def _try(self, p: Provider, messages, use_tools):
        """Call one provider, retrying ONCE on the same model when that is likely to work:
        - rate limit (429) that says "try again in Xs" with X <= MAX_RATE_LIMIT_WAIT -> wait, retry
        - broken tool call (Groq 400 tool_use_failed) -> retry right away
        Returns (response, seconds_taken, error_or_None)."""
        started = time.monotonic()
        error = None
        if p.flatten_tools:
            messages = flatten_tool_history(messages)
        for attempt in range(2):
            attempt_started = time.monotonic()
            try:
                response = p.runnable(self.tools, use_tools).invoke(messages)
                # only the successful attempt counts for "slow" (not our own rate-limit wait)
                return response, time.monotonic() - attempt_started, None
            except Exception as e:  # timeout, 429, 5xx, network, bad key, bad tool call
                error = e
                if attempt == 1:
                    break
                wait = _retry_after_seconds(e)
                if wait is not None and wait <= MAX_RATE_LIMIT_WAIT:
                    log.info("LLM %s rate limited, waiting %.1fs and retrying", p.name, wait)
                    time.sleep(wait + 0.3)
                    continue
                if wait is not None:
                    log.info("LLM %s rate limited for %.1fs (more than %.0fs), switching model", p.name, wait, MAX_RATE_LIMIT_WAIT)
                if _is_tool_call_error(e):
                    log.info("LLM %s wrote a broken tool call, retrying once", p.name)
                    continue
                break
        return None, time.monotonic() - started, error

    def status(self) -> list[dict]:
        now = time.monotonic()
        return [{
            "name": p.name,
            "healthy": p.down_until <= now,
            "cooldown_left_s": max(0, int(p.down_until - now)),
            "last_error": p.last_error,
            **p.stats,
        } for p in self.providers]


# ---------------------------------------------------------------------------
# Build providers from .env
# ---------------------------------------------------------------------------


def build_providers() -> list[Provider]:
    from langchain_groq import ChatGroq

    groq_key = os.getenv("GROQ_API_KEY") or None

    def groq(model: str):
        kwargs = dict(model=model, api_key=groq_key, temperature=0.2, timeout=PRIMARY_TIMEOUT, max_retries=0)
        if "gpt-oss" in model:
            # gpt-oss "thinks" before answering; those tokens count toward the rate limit.
            # Low effort is plenty for menu lookups and is much faster.
            kwargs["reasoning_effort"] = os.getenv("GROQ_REASONING_EFFORT", "low")
        return ChatGroq(**kwargs)

    main_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    providers = [Provider(
        name=f"groq:{main_model}", provider="groq", model=main_model, timeout=PRIMARY_TIMEOUT,
        build=lambda: groq(main_model),
    )]

    groq_backup = os.getenv("FALLBACK_GROQ_MODEL", "openai/gpt-oss-20b").strip()
    if groq_backup and groq_backup != main_model:
        providers.append(Provider(
            name=f"groq:{groq_backup}", provider="groq", model=groq_backup, timeout=PRIMARY_TIMEOUT,
            build=lambda: groq(groq_backup),
        ))

    base_url = os.getenv("FALLBACK_BASE_URL", "").strip()
    fb_model = os.getenv("FALLBACK_MODEL", "").strip()
    fb_key = os.getenv("FALLBACK_API_KEY", "").strip()
    if base_url and fb_model and fb_key:
        def build_openai():
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as e:
                raise RuntimeError("Install langchain-openai to use FALLBACK_BASE_URL") from e
            return ChatOpenAI(model=fb_model, api_key=fb_key, base_url=base_url, temperature=0.2,
                              timeout=FALLBACK_TIMEOUT, max_retries=0)

        host = base_url.split("//")[-1].split("/")[0]
        providers.append(Provider(
            name=f"{host}:{fb_model}", provider="openai_compatible", model=fb_model,
            timeout=FALLBACK_TIMEOUT, build=build_openai, flatten_tools=True,
        ))
    return providers