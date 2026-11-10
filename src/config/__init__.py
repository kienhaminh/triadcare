"""Environment configuration, one module per domain.

Each module exposes a loader that reads the environment on every call and returns a
frozen, validated settings object. Loaders do not cache: tests and the evaluation
runner change the environment at run time. Entry points load `.env` before they call
a loader. This package imports nothing else from `src`.
"""

from src.config._env import ConfigError
from src.config.agent import agent_settings
from src.config.database import database_settings
from src.config.evaluation import judge_settings
from src.config.fhir import fhir_settings
from src.config.imaging import imaging_settings
from src.config.llm import llm_settings

__all__ = ["ConfigError", "validate_all"]


def validate_all() -> None:
    """Load every settings object once, so an invalid value fails at startup.

    Unset optional services (FHIR, database, API keys) are not errors here.
    """
    for load in (
        agent_settings,
        database_settings,
        fhir_settings,
        imaging_settings,
        judge_settings,
        llm_settings,
    ):
        load()
