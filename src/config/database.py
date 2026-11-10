"""PostgreSQL connection and schema configuration.

The stores validate the URL scheme themselves, because their error types are part of
their contracts. src.db checks schema names.
"""

from dataclasses import dataclass

from src.config._env import env_str

DEFAULT_LEDGER_SCHEMA = "patient_knowledge_ledger"
DEFAULT_RECORD_SCHEMA = "record_governance"


@dataclass(frozen=True)
class DatabaseSettings:
    url: str | None  # None when DATABASE_URL is unset; stores that need it fail on use
    # LEDGER_SCHEMA isolates a process, for example one evaluation run.
    ledger_schema: str
    record_schema: str


def database_settings() -> DatabaseSettings:
    return DatabaseSettings(
        url=env_str("DATABASE_URL") or None,
        ledger_schema=env_str("LEDGER_SCHEMA", DEFAULT_LEDGER_SCHEMA),
        record_schema=env_str("RECORD_SCHEMA", DEFAULT_RECORD_SCHEMA),
    )
