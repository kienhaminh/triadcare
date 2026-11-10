"""PostgreSQL connections shared by the stores.

Each store checks its own DSN and maps connection errors to its own error type, because
those errors are part of its contract.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

_SCHEMA_NAME = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


def require_schema_name(name: str) -> str:
    if not _SCHEMA_NAME.fullmatch(name):
        raise ValueError("schema must be a valid lowercase PostgreSQL identifier.")
    return name


@contextmanager
def connect(dsn: str, schema: str | None = None) -> Iterator[psycopg.Connection[dict[str, Any]]]:
    """Open a connection that returns dict rows; with a schema, search only that schema."""
    with psycopg.connect(dsn, connect_timeout=5, row_factory=dict_row) as connection:
        if schema is not None:
            connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        yield connection
