"""Evaluation judge configuration."""

from dataclasses import dataclass

from src.config._env import env_str

DEFAULT_JUDGE_MODEL = "anthropic/claude-haiku-5.5"


@dataclass(frozen=True)
class JudgeSettings:
    model: str
    # The judge runs on OpenRouter with OPENROUTER_API_KEY. Empty when unset.
    api_key: str
    reasoning_effort: str | None
    # Only openrouter is supported.
    transport: str


def judge_settings() -> JudgeSettings:
    return JudgeSettings(
        model=env_str("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        api_key=env_str("OPENROUTER_API_KEY"),
        reasoning_effort=env_str("JUDGE_REASONING_EFFORT") or None,
        transport=env_str("JUDGE_TRANSPORT", "openrouter"),
    )
