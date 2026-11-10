"""OpenRouter request configuration."""

from dataclasses import dataclass

from src.config._env import env_float, env_int, env_str

DEFAULT_MODEL = "openai/gpt-6-luna"
DEFAULT_REASONING = "medium"
# Reasoning tokens count inside the cap; at 8192 reasoning can use the whole cap.
MAX_OUTPUT_TOKENS = 16384
MAX_TIMEOUT_SECONDS = 300
# Wall-clock limit of one request. OpenRouter sends keep-alive comments while a provider
# works, so the read timeout alone never ends a stalled request.
REQUEST_DEADLINE_SECONDS = 900
_UNBOUNDED = 2**31 - 1
_POSITIVE_REQUEST_LIMITS = "LLM_MAX_OUTPUT_TOKENS and LLM_TIMEOUT_SECONDS must be positive."


@dataclass(frozen=True)
class LLMSettings:
    api_key: str  # empty when unset; a request fails, but loading does not
    model: str
    reasoning_effort: str
    temperature: float | None  # None sends no temperature; the provider default applies
    max_output_tokens: int
    timeout_seconds: float
    deadline_seconds: float
    # OpenRouter groups requests with the same session_id. llm.client.session()
    # overrides this process-wide value for one block of requests.
    session_id: str | None


def llm_settings() -> LLMSettings:
    return LLMSettings(
        api_key=env_str("OPENROUTER_API_KEY"),
        model=env_str("LLM_MODEL", DEFAULT_MODEL),
        reasoning_effort=env_str("LLM_REASONING_EFFORT", DEFAULT_REASONING),
        temperature=(
            env_float("LLM_TEMPERATURE", 0, minimum=0, maximum=2)
            if env_str("LLM_TEMPERATURE")
            else None
        ),
        max_output_tokens=env_int(
            "LLM_MAX_OUTPUT_TOKENS",
            MAX_OUTPUT_TOKENS,
            minimum=1,
            maximum=_UNBOUNDED,
            message=_POSITIVE_REQUEST_LIMITS,
        ),
        timeout_seconds=env_float(
            "LLM_TIMEOUT_SECONDS",
            MAX_TIMEOUT_SECONDS,
            minimum=1e-9,
            maximum=float(_UNBOUNDED),
            message=_POSITIVE_REQUEST_LIMITS,
        ),
        deadline_seconds=env_float(
            "LLM_REQUEST_DEADLINE_SECONDS",
            REQUEST_DEADLINE_SECONDS,
            minimum=1e-9,
            maximum=float(_UNBOUNDED),
            message=_POSITIVE_REQUEST_LIMITS,
        ),
        session_id=env_str("LLM_SESSION_ID") or None,
    )
