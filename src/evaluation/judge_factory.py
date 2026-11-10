"""Select the configured judge transport."""

from src.config.evaluation import judge_settings
from src.evaluation.judge import Judge, JudgeError


def judge_from_env(*, require_key: bool = True) -> Judge:
    settings = judge_settings()
    if settings.transport == "openrouter":
        return Judge.from_env(require_key=require_key)
    raise JudgeError(f"Unknown judge transport: {settings.transport}")
