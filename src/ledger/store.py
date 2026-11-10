"""PostgreSQL storage for raw patient and caregiver conversation history.

The store keeps each source turn as supplied. It does not create summaries or
merge conflicting reports. Imported scenario transcripts are immutable; live
messages are append-only and keyed by a stable message ID.
"""

# All malformed source/filter values use the same caller-visible validation contract.

from __future__ import annotations

import hashlib
import json
import secrets
import uuid
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from src import db
from src.config.database import database_settings
from src.ledger.corpus import MAX_CONTENT_LENGTH, parse_source
from src.ledger.temporal import scope_as_of
from src.ledger.values import normalize_recorded_at, validate_text

_AUDIENCES = {"doctor", "patient", "caregiver"}
_MAX_PAGE_SIZE = 500


class LedgerStore:
    """Append-only raw conversation store backed by PostgreSQL."""

    def __init__(self, dsn: str | None = None, schema: str | None = None) -> None:
        settings = database_settings()
        schema = schema or settings.ledger_schema
        self.dsn = dsn or settings.url
        if not self.dsn:
            raise RuntimeError("Patient ledger PostgreSQL is not configured.")
        if not self.dsn.startswith(("postgres://", "postgresql://")):
            raise RuntimeError("Patient ledger storage requires a PostgreSQL URL.")
        self.schema = db.require_schema_name(schema)
        self._initialize()

    def _connect(self) -> AbstractContextManager[psycopg.Connection[dict[str, Any]]]:
        return db.connect(self.dsn, self.schema)

    def lock_schema_ddl(self, connection: psycopg.Connection) -> None:
        """Serialize CREATE ... IF NOT EXISTS on this schema until the transaction ends.

        PostgreSQL rejects two sessions that create the same schema or table at once, which
        happens when parallel tool calls open their stores on a new schema.
        """
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"{self.schema}:ddl",)
        )

    def _initialize(self) -> None:
        with db.connect(self.dsn) as connection:
            self.lock_schema_ddl(connection)
            connection.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema))
            )
            connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(self.schema)))
            statements = (
                """CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY
                )""",
                """CREATE TABLE IF NOT EXISTS imported_sources (
                    scenario_id TEXT PRIMARY KEY,
                    source_sha256 TEXT NOT NULL,
                    source_path TEXT NOT NULL UNIQUE,
                    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                    split TEXT NOT NULL,
                    imported_at TEXT NOT NULL
                )""",
                """CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                    audience TEXT NOT NULL CHECK (audience IN ('doctor', 'patient', 'caregiver')),
                    participant_id TEXT NOT NULL,
                    scenario_id TEXT,
                    split TEXT,
                    source_day INTEGER,
                    source_file TEXT,
                    source_sha256 TEXT,
                    token_hash TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE (scenario_id, source_day, audience)
                )""",
                """CREATE INDEX IF NOT EXISTS sessions_patient_scope
                   ON sessions(patient_id, audience, participant_id)""",
                """CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    speaker TEXT NOT NULL,
                    content TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    source_metadata JSONB,
                    inserted_at TEXT NOT NULL,
                    UNIQUE (session_id, sequence)
                )""",
                """CREATE INDEX IF NOT EXISTS messages_session_order
                   ON messages(session_id, recorded_at, sequence, message_id)""",
                """CREATE INDEX IF NOT EXISTS messages_recorded_order
                   ON messages(recorded_at, sequence, message_id)""",
            )
            for statement in statements:
                connection.execute(statement)

    @contextmanager
    def session_lock(self, session_id: str) -> Iterator[None]:
        """Serialize a full Care turn across application processes."""
        session_id = validate_text(session_id, "session_id")
        key = f"care-session:{self.schema}:{session_id}"
        with psycopg.connect(self.dsn, connect_timeout=5, autocommit=True) as connection:
            connection.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (key,))
            try:
                yield
            finally:
                connection.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (key,))

    @classmethod
    def _date_filter(cls, column: str, lower: Any, upper: Any) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if lower is not None:
            normalized = normalize_recorded_at(lower)
            clauses.append(f"{column} >= %s")
            parameters.append(normalized)
        if upper is not None:
            normalized = normalize_recorded_at(upper)
            if len(normalized) == 10:
                next_day = (date.fromisoformat(normalized) + timedelta(days=1)).isoformat()
                clauses.append(f"{column} < %s")
                parameters.append(next_day)
            else:
                clauses.append(f"({column} < %s OR (length({column}) > 10 AND {column} <= %s))")
                parameters.extend([normalized[:10], normalized])
        if lower is not None and upper is not None:
            lower_value = normalize_recorded_at(lower)
            upper_value = normalize_recorded_at(upper)
            if len(upper_value) == 10:
                upper_value = (date.fromisoformat(upper_value) + timedelta(days=1)).isoformat()
                if lower_value >= upper_value:
                    raise ValueError("date_from must not be after date_to.")
            elif lower_value > upper_value:
                raise ValueError("date_from must not be after date_to.")
        return clauses, parameters

    @staticmethod
    def _validate_audience(audience: str, participant_id: str | None) -> None:
        if not isinstance(audience, str) or audience not in _AUDIENCES:
            raise ValueError("audience must be 'doctor', 'patient', or 'caregiver'.")
        if audience in {"patient", "caregiver"} and participant_id is None:
            raise ValueError("participant_id is required for patient and caregiver access.")
        if participant_id is not None:
            validate_text(participant_id, "participant_id")

    @staticmethod
    def _escape_like(value: str) -> str:
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    @classmethod
    def _where(
        cls,
        patient_id: str,
        *,
        audience: str = "doctor",
        participant_id: str | None = None,
        session_id: str | None = None,
        speaker: str | None = None,
        date_from: Any = None,
        date_to: Any = None,
        as_of: Any = None,
        text: str | None = None,
        message_ids: list[str] | None = None,
    ) -> tuple[str, list[Any]]:
        validate_text(patient_id, "patient_id")
        cls._validate_audience(audience, participant_id)
        scope_upper = scope_as_of(patient_id)
        clauses = ["s.patient_id = %s"]
        parameters: list[Any] = [patient_id]
        if audience != "doctor":
            clauses.extend(["s.audience = %s", "s.participant_id = %s"])
            parameters.extend([audience, participant_id])
        elif participant_id is not None:
            clauses.append("s.participant_id = %s")
            parameters.append(participant_id)
        if session_id is not None:
            clauses.append("m.session_id = %s")
            parameters.append(validate_text(session_id, "session_id"))
        if speaker is not None:
            clauses.append("m.speaker = %s")
            parameters.append(validate_text(speaker, "speaker", max_length=128))
        date_clauses, date_parameters = cls._date_filter("m.recorded_at", date_from, date_to)
        clauses.extend(date_clauses)
        parameters.extend(date_parameters)
        # as_of is one more upper bound with the same day and timestamp rules as date_to.
        as_of_clauses, as_of_parameters = cls._date_filter("m.recorded_at", None, as_of)
        clauses.extend(as_of_clauses)
        parameters.extend(as_of_parameters)
        scope_clauses, scope_parameters = cls._date_filter("m.recorded_at", None, scope_upper)
        clauses.extend(scope_clauses)
        parameters.extend(scope_parameters)
        if text is not None:
            if not isinstance(text, str):
                raise ValueError("text must be a string.")
            # A message matches when it holds every word of text, in any order.
            for word in text.split() or [text]:
                clauses.append("m.content ILIKE %s ESCAPE '\\'")
                parameters.append("%" + cls._escape_like(word) + "%")
        if message_ids is not None:
            if not isinstance(message_ids, list) or len(message_ids) > 1000:
                raise ValueError("message_ids must be a list of at most 1000 IDs.")
            if not message_ids:
                clauses.append("0")
            else:
                for message_id in message_ids:
                    validate_text(message_id, "message_id")
                clauses.append(f"m.message_id IN ({','.join('%s' for _ in message_ids)})")
                parameters.extend(message_ids)
        return " AND ".join(clauses), parameters

    @staticmethod
    def _message_projection(row: dict[str, Any]) -> dict[str, Any]:
        metadata = row["source_metadata"]
        return {
            "message_id": row["message_id"],
            "patient_id": row["patient_id"],
            "session_id": row["session_id"],
            "speaker": row["speaker"],
            "participant_id": row["participant_id"],
            "audience": row["audience"],
            "content": row["content"],
            "recorded_at": row["recorded_at"],
            "reporting_precision": "date" if len(row["recorded_at"]) == 10 else "timestamp",
            "sequence": row["sequence"],
            "source_metadata": json.loads(metadata) if isinstance(metadata, str) else metadata,
        }

    @staticmethod
    def _message_select() -> str:
        return """
            SELECT m.message_id, s.patient_id, m.session_id, m.speaker,
                   s.participant_id, s.audience, m.content, m.recorded_at,
                   m.sequence, m.source_metadata, s.source_day
            FROM messages AS m
            JOIN sessions AS s ON s.session_id = m.session_id
        """

    def import_corpus(self, directory: Path) -> dict[str, int]:
        """Import scenario transcripts atomically and reject changed sources."""
        root = Path(directory)
        if not root.is_dir():
            raise ValueError(f"Corpus directory does not exist: {root}")
        parsed_sources = [parse_source(path, root) for path in sorted(root.rglob("*.json"))]
        scenario_ids = [source["scenario_id"] for source in parsed_sources]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("Corpus contains duplicate scenario_id values.")

        imported_scenarios = 0
        skipped_scenarios = 0
        imported_sessions = 0
        imported_messages = 0
        now = normalize_recorded_at(datetime.now(UTC))
        with self._connect() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"{self.schema}:corpus-import",),
            )
            for source in parsed_sources:
                existing = connection.execute(
                    "SELECT source_sha256 FROM imported_sources WHERE scenario_id = %s",
                    (source["scenario_id"],),
                ).fetchone()
                if existing is not None:
                    if existing["source_sha256"] != source["source_sha256"]:
                        raise ValueError(
                            f"Imported source changed for scenario {source['scenario_id']}; refusing to overwrite it."
                        )
                    skipped_scenarios += 1
                    continue
                path_owner = connection.execute(
                    "SELECT scenario_id FROM imported_sources WHERE source_path = %s",
                    (source["source_path"],),
                ).fetchone()
                if path_owner is not None:
                    raise ValueError(
                        f"Imported source path changed scenario identity at {source['source_path']}; refusing to overwrite it."
                    )

                if connection.execute(
                    "SELECT 1 FROM sessions WHERE scenario_id = %s LIMIT 1",
                    (source["scenario_id"],),
                ).fetchone():
                    raise ValueError(
                        f"Imported sessions already exist for scenario {source['scenario_id']}."
                    )

                connection.execute(
                    "INSERT INTO patients(patient_id) VALUES (%s) ON CONFLICT DO NOTHING",
                    (source["patient_id"],),
                )
                connection.execute(
                    """INSERT INTO imported_sources
                       (scenario_id, source_sha256, source_path, patient_id, split, imported_at)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (
                        source["scenario_id"],
                        source["source_sha256"],
                        source["source_path"],
                        source["patient_id"],
                        source["split"],
                        now,
                    ),
                )
                for session in source["sessions"]:
                    connection.execute(
                        """INSERT INTO sessions
                           (session_id, patient_id, audience, participant_id, scenario_id, split,
                            source_day, source_file, source_sha256, token_hash, created_at)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, %s)""",
                        (
                            session["session_id"],
                            source["patient_id"],
                            session["audience"],
                            session["participant_id"],
                            source["scenario_id"],
                            source["split"],
                            session["source_day"],
                            source["source_path"],
                            source["source_sha256"],
                            now,
                        ),
                    )
                    for message in session["messages"]:
                        connection.execute(
                            """INSERT INTO messages
                               (message_id, session_id, speaker, content, recorded_at, sequence,
                                source_metadata, inserted_at)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                            (
                                message["message_id"],
                                session["session_id"],
                                message["speaker"],
                                message["content"],
                                message["recorded_at"],
                                message["sequence"],
                                Jsonb(message["source_metadata"]),
                                now,
                            ),
                        )
                    imported_sessions += 1
                    imported_messages += len(session["messages"])
                imported_scenarios += 1
        return {
            "sources_imported": imported_scenarios,
            "sources_skipped": skipped_scenarios,
            "patients": len({source["patient_id"] for source in parsed_sources}),
            "sessions_imported": imported_sessions,
            "messages_imported": imported_messages,
        }

    def patients(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT p.patient_id,
                          COUNT(DISTINCT m.message_id) AS messages,
                          COUNT(DISTINCT s.session_id) AS sessions
                   FROM patients AS p
                   LEFT JOIN sessions AS s ON s.patient_id = p.patient_id
                   LEFT JOIN messages AS m ON m.session_id = s.session_id
                   GROUP BY p.patient_id ORDER BY p.patient_id"""
            ).fetchall()
        return [dict(row) for row in rows]

    def query_messages(
        self,
        patient_id: str,
        *,
        audience: str = "doctor",
        participant_id: str | None = None,
        session_id: str | None = None,
        speaker: str | None = None,
        date_from: Any = None,
        date_to: Any = None,
        as_of: Any = None,
        text: str | None = None,
        limit: int = 20,
        offset: int = 0,
        message_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= _MAX_PAGE_SIZE
        ):
            raise ValueError(f"limit must be an integer from 1 to {_MAX_PAGE_SIZE}.")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer.")
        where, parameters = self._where(
            patient_id,
            audience=audience,
            participant_id=participant_id,
            session_id=session_id,
            speaker=speaker,
            date_from=date_from,
            date_to=date_to,
            as_of=as_of,
            text=text,
            message_ids=message_ids,
        )
        with self._connect() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) AS total FROM messages AS m JOIN sessions AS s ON s.session_id=m.session_id WHERE {where}",
                parameters,
            ).fetchone()["total"]
            rows = connection.execute(
                self._message_select()
                + f" WHERE {where} ORDER BY m.recorded_at, m.sequence, m.message_id LIMIT %s OFFSET %s",
                [*parameters, limit + 1, offset],
            ).fetchall()
        has_more = len(rows) > limit
        messages = [self._message_projection(row) for row in rows[:limit]]
        return {
            "messages": messages,
            "total": total,
            "has_more": has_more,
            "next_offset": offset + len(messages) if has_more else None,
        }

    def counts(self, patient_id: str, **filters: Any) -> dict[str, int]:
        allowed = {
            "audience",
            "participant_id",
            "session_id",
            "speaker",
            "date_from",
            "date_to",
            "as_of",
            "text",
        }
        unknown = set(filters) - allowed
        if unknown:
            raise TypeError(f"Unexpected count filters: {', '.join(sorted(unknown))}")
        where, parameters = self._where(patient_id, **filters)
        with self._connect() as connection:
            row = connection.execute(
                f"""SELECT COUNT(*) AS messages, COUNT(DISTINCT m.session_id) AS sessions
                    FROM messages AS m JOIN sessions AS s ON s.session_id=m.session_id
                    WHERE {where}""",
                parameters,
            ).fetchone()
        return {"messages": row["messages"], "sessions": row["sessions"]}

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        session_id = validate_text(session_id, "session_id")
        with self._connect() as connection:
            row = connection.execute(
                """SELECT session_id, patient_id, audience, participant_id, scenario_id, split,
                          source_day, source_file, source_sha256, token_hash, created_at
                   FROM sessions WHERE session_id = %s""",
                (session_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def create_session(
        self,
        patient_id: str,
        audience: str,
        participant_id: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        patient_id = validate_text(patient_id, "patient_id")
        participant_id = validate_text(participant_id, "participant_id")
        self._validate_audience(audience, participant_id)
        if audience not in {"patient", "caregiver"}:
            raise ValueError("Live conversation sessions must use patient or caregiver audience.")
        session_id = (
            validate_text(session_id, "session_id")
            if session_id is not None
            else f"live:{uuid.uuid4()}"
        )
        access_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(access_token.encode("utf-8")).hexdigest()
        created_at = normalize_recorded_at(datetime.now(UTC))
        with self._connect() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM patients WHERE patient_id = %s", (patient_id,)
                ).fetchone()
                is None
            ):
                raise ValueError(f"Unknown patient_id: {patient_id}")
            if (
                connection.execute(
                    "SELECT 1 FROM sessions WHERE session_id = %s", (session_id,)
                ).fetchone()
                is not None
            ):
                raise ValueError(f"Session ID already exists: {session_id}")
            inserted = connection.execute(
                """INSERT INTO sessions
                   (session_id, patient_id, audience, participant_id, token_hash, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (session_id) DO NOTHING
                   RETURNING session_id""",
                (session_id, patient_id, audience, participant_id, token_hash, created_at),
            ).fetchone()
            if inserted is None:
                raise ValueError(f"Session ID already exists: {session_id}")
        session = self.get_session(session_id)
        assert session is not None
        session["access_token"] = access_token
        return session

    def append_message(
        self,
        session_id: str,
        speaker: str,
        content: str,
        *,
        message_id: str | None = None,
        recorded_at: Any = None,
    ) -> dict[str, Any]:
        session_id = validate_text(session_id, "session_id")
        speaker = validate_text(speaker, "speaker", max_length=128)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string.")
        if len(content) > MAX_CONTENT_LENGTH:
            raise ValueError(f"content must be at most {MAX_CONTENT_LENGTH} characters.")
        generated_recorded_at = recorded_at is None
        normalized_at = normalize_recorded_at(recorded_at, default_now=True)
        message_id = (
            validate_text(message_id, "message_id")
            if message_id is not None
            else f"live:{uuid.uuid4()}"
        )
        inserted_at = normalize_recorded_at(datetime.now(UTC))
        with self._connect() as connection:
            session = connection.execute(
                "SELECT patient_id, audience, scenario_id FROM sessions WHERE session_id = %s FOR UPDATE",
                (session_id,),
            ).fetchone()
            if session is None:
                raise ValueError(f"Unknown session_id: {session_id}")
            if session["scenario_id"] is not None:
                raise ValueError("Imported source sessions are immutable.")
            if speaker not in {session["audience"], "care_agent"}:
                raise ValueError(
                    f"speaker must be {session['audience']!r} or 'care_agent' for this session."
                )
            existing = connection.execute(
                """SELECT m.message_id, s.patient_id, m.session_id, m.speaker,
                          s.participant_id, s.audience, m.content, m.recorded_at,
                          m.sequence, m.source_metadata
                   FROM messages AS m JOIN sessions AS s ON s.session_id=m.session_id
                   WHERE m.message_id = %s""",
                (message_id,),
            ).fetchone()
            if existing is not None:
                same = (
                    existing["session_id"] == session_id
                    and existing["speaker"] == speaker
                    and existing["content"] == content
                    and (generated_recorded_at or existing["recorded_at"] == normalized_at)
                )
                if not same:
                    raise ValueError(f"Conflicting message_id: {message_id}")
                return self._message_projection(existing)
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence FROM messages WHERE session_id = %s",
                (session_id,),
            ).fetchone()["next_sequence"]
            inserted = connection.execute(
                """INSERT INTO messages
                   (message_id, session_id, speaker, content, recorded_at, sequence, source_metadata, inserted_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (message_id) DO NOTHING
                   RETURNING message_id""",
                (
                    message_id,
                    session_id,
                    speaker,
                    content,
                    normalized_at,
                    sequence,
                    Jsonb({"kind": "live"}),
                    inserted_at,
                ),
            ).fetchone()
            if inserted is None:
                existing = connection.execute(
                    """SELECT m.message_id, s.patient_id, m.session_id, m.speaker,
                              s.participant_id, s.audience, m.content, m.recorded_at,
                              m.sequence, m.source_metadata
                       FROM messages AS m JOIN sessions AS s ON s.session_id=m.session_id
                       WHERE m.message_id = %s""",
                    (message_id,),
                ).fetchone()
                if existing is None:
                    raise ValueError(f"Conflicting message_id: {message_id}")
                same = (
                    existing["session_id"] == session_id
                    and existing["speaker"] == speaker
                    and existing["content"] == content
                    and (generated_recorded_at or existing["recorded_at"] == normalized_at)
                )
                if not same:
                    raise ValueError(f"Conflicting message_id: {message_id}")
                return self._message_projection(existing)
            row = connection.execute(
                self._message_select() + " WHERE m.message_id = %s", (message_id,)
            ).fetchone()
        return self._message_projection(row)

    def session_history(self, session_id: str, limit: int = 40) -> list[dict[str, Any]]:
        session_id = validate_text(session_id, "session_id")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= _MAX_PAGE_SIZE
        ):
            raise ValueError(f"limit must be an integer from 1 to {_MAX_PAGE_SIZE}.")
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT * FROM (
                       {self._message_select()}
                       WHERE m.session_id = %s
                       ORDER BY m.recorded_at DESC, m.sequence DESC, m.message_id DESC
                       LIMIT %s
                   ) ORDER BY recorded_at, sequence, message_id""",
                (session_id, limit),
            ).fetchall()
        return [self._message_projection(row) for row in rows]

    def read_message(
        self,
        patient_id: str,
        message_id: str,
        *,
        audience: str = "doctor",
        participant_id: str | None = None,
        as_of: Any = None,
    ) -> dict[str, Any] | None:
        where, parameters = self._where(
            patient_id,
            audience=audience,
            participant_id=participant_id,
            as_of=as_of,
            message_ids=[message_id],
        )
        with self._connect() as connection:
            row = connection.execute(
                self._message_select() + f" WHERE {where} LIMIT 1", parameters
            ).fetchone()
        return self._message_projection(row) if row is not None else None

    def export_scenario(self, scenario_id: str) -> list[dict[str, Any]]:
        """Read back every stored message of one imported scenario, in source order."""
        scenario_id = validate_text(scenario_id, "scenario_id")
        with self._connect() as connection:
            rows = connection.execute(
                self._message_select()
                + " WHERE s.scenario_id = %s ORDER BY s.source_day, s.audience, m.sequence",
                (scenario_id,),
            ).fetchall()
        return [{**self._message_projection(row), "source_day": row["source_day"]} for row in rows]
