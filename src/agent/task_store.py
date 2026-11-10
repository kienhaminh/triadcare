"""Storage for background specialist tasks: in memory for tests, PostgreSQL in service."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from src.config.database import database_settings

ACTIVE_STATUSES = ("queued", "running")
TERMINAL_STATUSES = ("completed", "failed", "cancelled", "interrupted")
INTERRUPTED_ERROR = "The backend restarted before the consultation finished."
STORAGE_ERROR = "Background task storage is unavailable."


class TaskStorageError(RuntimeError):
    """The task store could not be reached or initialized."""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


class MemoryTaskStore:
    """Process-local task store for tests."""

    def __init__(self) -> None:
        self._tasks: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, task: dict) -> dict:
        with self._lock:
            self._tasks[task["task_id"]] = deepcopy(task)
            return deepcopy(task)

    def claim(self, task_id: str) -> dict | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task["status"] != "queued":
                return None
            task.update(status="running", started_at=now())
            return deepcopy(task)

    def finish(
        self, task_id: str, status: str, *, answer: str | None = None, error: str | None = None
    ) -> dict | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task["status"] not in ACTIVE_STATUSES:
                return deepcopy(task) if task is not None else None
            task.update(status=status, answer=answer, error=error, finished_at=now())
            return deepcopy(task)

    def get(self, task_id: str) -> dict | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return deepcopy(task) if task is not None else None

    def list(self, session_id: str, *, limit: int) -> list[dict]:
        with self._lock:
            tasks = [task for task in self._tasks.values() if task["session_id"] == session_id]
            tasks.sort(key=lambda task: task["created_at"], reverse=True)
            return deepcopy(tasks[:limit])

    def claim_notifications(self, task_ids: list[str]) -> list[dict]:
        with self._lock:
            claimed = []
            for task_id in task_ids:
                task = self._tasks.get(task_id)
                if (
                    task is not None
                    and task["status"] in TERMINAL_STATUSES
                    and task["notified_at"] is None
                ):
                    task["notified_at"] = now()
                    claimed.append(deepcopy(task))
            return claimed

    def unnotified(self, session_id: str) -> list[dict]:
        with self._lock:
            tasks = [
                task
                for task in self._tasks.values()
                if task["session_id"] == session_id
                and task["status"] in TERMINAL_STATUSES
                and task["notified_at"] is None
            ]
            tasks.sort(key=lambda task: task["finished_at"] or task["created_at"])
            return deepcopy(tasks)

    def interrupt_orphans(self, session_id: str, live: set[str]) -> None:
        with self._lock:
            for task in self._tasks.values():
                if (
                    task["session_id"] == session_id
                    and task["status"] in ACTIVE_STATUSES
                    and task["task_id"] not in live
                ):
                    task.update(status="interrupted", error=INTERRUPTED_ERROR, finished_at=now())

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            for task_id in [
                key for key, task in self._tasks.items() if task["session_id"] == session_id
            ]:
                del self._tasks[task_id]


class PostgresTaskStore:
    """Durable task store in the agent PostgreSQL database."""

    def __init__(self, dsn: str | None = None, *, schema: str = "public") -> None:
        import psycopg
        from psycopg import sql

        self._psycopg = psycopg
        self._sql = sql
        self.dsn = dsn or database_settings().url
        if not self.dsn or not self.dsn.startswith(("postgres://", "postgresql://")):
            raise TaskStorageError("Background task storage requires a PostgreSQL DATABASE_URL.")
        self._schema = schema
        self._table = sql.Identifier(schema, "doctor_agent_tasks")
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        from src import db

        try:
            with db.connect(self.dsn) as connection:
                yield connection
        except (self._psycopg.Error, OSError) as error:
            raise TaskStorageError(STORAGE_ERROR) from error

    def _query(self, text: str) -> Any:
        return self._sql.SQL(text).format(table=self._table)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                self._sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                    self._sql.Identifier(self._schema)
                )
            )
            connection.execute(
                self._query(
                    """CREATE TABLE IF NOT EXISTS {table} (
                        task_id text PRIMARY KEY,
                        session_id text NOT NULL,
                        specialist text NOT NULL
                            CHECK (specialist IN ('ehr', 'radiologist', 'care')),
                        agent text NOT NULL,
                        question text NOT NULL,
                        context text,
                        status text NOT NULL CHECK (status IN (
                            'queued', 'running', 'completed', 'failed', 'cancelled',
                            'interrupted'
                        )),
                        answer text,
                        error text,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        started_at timestamptz,
                        finished_at timestamptz,
                        notified_at timestamptz
                    )"""
                )
            )
            connection.execute(
                self._query(
                    "CREATE INDEX IF NOT EXISTS doctor_agent_tasks_session "
                    "ON {table} (session_id, created_at DESC)"
                )
            )

    @staticmethod
    def _row(row: dict | None) -> dict | None:
        if row is None:
            return None
        task = dict(row)
        for key in ("created_at", "started_at", "finished_at", "notified_at"):
            if task[key] is not None:
                task[key] = (
                    task[key]
                    .astimezone(UTC)
                    .isoformat(timespec="microseconds")
                    .replace("+00:00", "Z")
                )
        return task

    def create(self, task: dict) -> dict:
        with self._connect() as connection:
            return self._row(
                connection.execute(
                    self._query(
                        "INSERT INTO {table} (task_id, session_id, specialist, agent, "
                        "question, context, status) VALUES (%s, %s, %s, %s, %s, %s, 'queued') "
                        "RETURNING *"
                    ),
                    (
                        task["task_id"],
                        task["session_id"],
                        task["specialist"],
                        task["agent"],
                        task["question"],
                        task["context"],
                    ),
                ).fetchone()
            )

    def claim(self, task_id: str) -> dict | None:
        with self._connect() as connection:
            return self._row(
                connection.execute(
                    self._query(
                        "UPDATE {table} SET status = 'running', started_at = now() "
                        "WHERE task_id = %s AND status = 'queued' RETURNING *"
                    ),
                    (task_id,),
                ).fetchone()
            )

    def finish(
        self, task_id: str, status: str, *, answer: str | None = None, error: str | None = None
    ) -> dict | None:
        with self._connect() as connection:
            connection.execute(
                self._query(
                    "UPDATE {table} SET status = %s, answer = %s, error = %s, "
                    "finished_at = now() WHERE task_id = %s "
                    "AND status IN ('queued', 'running')"
                ),
                (status, answer, error, task_id),
            )
            return self._row(
                connection.execute(
                    self._query("SELECT * FROM {table} WHERE task_id = %s"), (task_id,)
                ).fetchone()
            )

    def get(self, task_id: str) -> dict | None:
        with self._connect() as connection:
            return self._row(
                connection.execute(
                    self._query("SELECT * FROM {table} WHERE task_id = %s"), (task_id,)
                ).fetchone()
            )

    def list(self, session_id: str, *, limit: int) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                self._query(
                    "SELECT * FROM {table} WHERE session_id = %s "
                    "ORDER BY created_at DESC, task_id LIMIT %s"
                ),
                (session_id, limit),
            ).fetchall()
        return [self._row(row) for row in rows]

    def claim_notifications(self, task_ids: list[str]) -> list[dict]:
        if not task_ids:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                self._query(
                    "UPDATE {table} SET notified_at = now() WHERE task_id = ANY(%s) "
                    "AND notified_at IS NULL AND status IN "
                    "('completed', 'failed', 'cancelled', 'interrupted') RETURNING *"
                ),
                (task_ids,),
            ).fetchall()
        order = {task_id: index for index, task_id in enumerate(task_ids)}
        return sorted((self._row(row) for row in rows), key=lambda task: order[task["task_id"]])

    def unnotified(self, session_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                self._query(
                    "SELECT * FROM {table} WHERE session_id = %s AND notified_at IS NULL "
                    "AND status IN ('completed', 'failed', 'cancelled', 'interrupted') "
                    "ORDER BY coalesce(finished_at, created_at), task_id"
                ),
                (session_id,),
            ).fetchall()
        return [self._row(row) for row in rows]

    def interrupt_orphans(self, session_id: str, live: set[str]) -> None:
        with self._connect() as connection:
            connection.execute(
                self._query(
                    "UPDATE {table} SET status = 'interrupted', error = %s, "
                    "finished_at = now() WHERE session_id = %s "
                    "AND status IN ('queued', 'running') AND NOT (task_id = ANY(%s))"
                ),
                (INTERRUPTED_ERROR, session_id, list(live)),
            )

    def delete_session(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                self._query("DELETE FROM {table} WHERE session_id = %s"), (session_id,)
            )
