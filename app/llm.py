"""One OpenAI-compatible client for Groq, Gemini, DeepSeek, Grok (xAI) and local Ollama.

All of them expose OpenAI-compatible endpoints, so switching provider is just
LLM_PROVIDER=... in .env.

Fallback: set LLM_FALLBACK=ollama and, if the primary provider fails (no key, bad key,
rate limit / quota exceeded, network or server error), the call is retried on Ollama.
After a rate-limit or auth failure the primary is skipped for LLM_FALLBACK_COOLDOWN
seconds, so you don't pay the failed round-trip on every question.
"""
import logging
import threading
import time
from functools import lru_cache

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from app.config import settings

log = logging.getLogger(__name__)

PROVIDERS = {
    # Fast hosted inference with a free tier (no card): https://console.groq.com/keys
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key": lambda: settings.groq_api_key,
        "model": lambda: settings.groq_model,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key": lambda: settings.gemini_api_key,
        "model": lambda: settings.gemini_model,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "key": lambda: settings.deepseek_api_key,
        "model": lambda: settings.deepseek_model,
    },
    "grok": {  # xAI's Grok (not Groq)
        "base_url": "https://api.x.ai/v1",
        "key": lambda: settings.xai_api_key,
        "model": lambda: settings.xai_model,
    },
    # Local, free, no key. Install from https://ollama.com and `ollama pull gemma3:4b`
    "ollama": {
        "base_url": None,  # read from settings.ollama_base_url
        "key": lambda: "ollama",  # OpenAI client needs a non-empty key; Ollama ignores it
        "model": lambda: settings.ollama_model,
    },
}

LOCAL_PROVIDERS = {"ollama"}


class ProviderError(RuntimeError):
    """A provider failed in a way where trying the fallback makes sense."""


def _base_url(provider: str) -> str:
    if provider == "ollama":
        return settings.ollama_base_url
    return PROVIDERS[provider]["base_url"]


_lock = threading.Lock()
_last_call: dict[str, float] = {}
_cooldown_until: dict[str, float] = {}
_local = threading.local()  # remembers which provider answered, per thread


def last_provider() -> str | None:
    """Provider that produced the most recent answer in this thread (shown in the UI)."""
    return getattr(_local, "provider", None)


@lru_cache(maxsize=None)
def _client(provider: str) -> OpenAI:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider '{provider}'. Use one of: {', '.join(PROVIDERS)}")
    key = PROVIDERS[provider]["key"]()
    if not key:
        raise ProviderError(f"No API key set for provider '{provider}'. Fill it in .env")
    # local models can be slow on CPU, so allow a long timeout
    timeout = 300 if provider in LOCAL_PROVIDERS else 60
    return OpenAI(api_key=key, base_url=_base_url(provider), timeout=timeout)


def _throttle(provider: str) -> None:
    """Keep a minimum gap between calls so free-tier RPM limits aren't hit."""
    with _lock:
        wait = settings.llm_min_interval - (time.time() - _last_call.get(provider, 0.0))
        if wait > 0:
            time.sleep(wait)
        _last_call[provider] = time.time()


def _call(provider: str, system: str, user: str, temperature: float, max_retries: int) -> str:
    """Call one provider. Raises ProviderError for failures where a fallback could help."""
    client = _client(provider)
    model = PROVIDERS[provider]["model"]()
    delay = 2.0
    for attempt in range(max_retries):
        if provider not in LOCAL_PROVIDERS:  # no rate limits locally
            _throttle(provider)
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return (resp.choices[0].message.content or "").strip()
        except (APIConnectionError, APITimeoutError) as e:
            if provider == "ollama":
                raise ProviderError(
                    f"Can't reach Ollama at {settings.ollama_base_url}. Is the Ollama app running? "
                    f"Also check you've pulled the model: ollama pull {model}"
                ) from e
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise ProviderError(f"{provider}: network error ({e})") from e
        except APIStatusError as e:
            if e.status_code == 404:
                if provider == "ollama":
                    raise ProviderError(f"Ollama model '{model}' not found. Run: ollama pull {model}") from e
                # hosted providers retire models regularly (e.g. Groq retired llama-3.3-70b in 2026)
                raise ProviderError(f"{provider}: model '{model}' not found or retired. "
                                    f"Update the *_MODEL setting in .env") from e
            if isinstance(e, RateLimitError) or e.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise ProviderError(f"{provider}: rate limit / quota exceeded") from e
            if e.status_code in (401, 403):
                raise ProviderError(f"{provider}: API key rejected ({e.status_code})") from e
            if e.status_code >= 500:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise ProviderError(f"{provider}: server error {e.status_code}") from e
            raise  # 400 etc: a bug in the request, a fallback wouldn't help
    raise RuntimeError("unreachable")


def chat(system: str, user: str, temperature: float = 0.2, provider: str | None = None,
         max_retries: int = 4) -> str:
    primary = (provider or settings.llm_provider).lower()
    fallback = (settings.llm_fallback or "").lower()
    use_fallback = bool(fallback) and fallback != primary

    if use_fallback and time.time() < _cooldown_until.get(primary, 0.0):
        log.info("%s is cooling down after an error; using %s", primary, fallback)
    else:
        try:
            # with a fallback available, don't sit through long retry loops on the primary
            text = _call(primary, system, user, temperature, 2 if use_fallback else max_retries)
            _local.provider = primary
            return text
        except ProviderError as e:
            if not use_fallback:
                raise
            msg = str(e)
            if "rate limit" in msg or "key" in msg or "not found" in msg:
                _cooldown_until[primary] = time.time() + settings.llm_fallback_cooldown
            log.warning("%s failed (%s). Falling back to %s.", primary, msg, fallback)

    text = _call(fallback, system, user, temperature, max_retries)
    _local.provider = fallback
    return text
