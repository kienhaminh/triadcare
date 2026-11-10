"""Radiologist analysis results stored in the patient ledger schema.

Each fresh analysis appends one row with the patient, study, modality, method, and
limitations; a reanalysis appends another row and leaves earlier results unchanged.
The results are supplementary model-generated records, not hospital records.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from psycopg.types.json import Jsonb

from src.ledger.store import LedgerStore

# An observed stored result has about 550 characters, so this count usually fits in one
# observation. read_tool_result gives the full text of a longer observation.
STORED_RESULT_LIMIT = 5


class ImagingResultStore:
    def __init__(self, ledger: LedgerStore | None = None) -> None:
        self.ledger = ledger or LedgerStore()
        with self.ledger._connect() as connection:
            self.ledger.lock_schema_ddl(connection)
            connection.execute(
                """CREATE TABLE IF NOT EXISTS imaging_results (
                    result_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    study_id TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    method TEXT,
                    result JSONB NOT NULL,
                    analyzed_at TEXT NOT NULL
                )"""
            )

    def append(self, result: dict[str, Any]) -> dict[str, Any]:
        row = {
            "result_id": f"imaging-result:{uuid.uuid4().hex}",
            "analyzed_at": datetime.now(UTC).isoformat(),
        }
        with self.ledger._connect() as connection:
            connection.execute(
                """INSERT INTO imaging_results(result_id, patient_id, study_id, modality, method,
                       result, analyzed_at) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    row["result_id"],
                    result["patient_id"],
                    result["study_id"],
                    result["modality"],
                    result.get("method"),
                    Jsonb(result),
                    row["analyzed_at"],
                ),
            )
        return {**result, **row}

    def results(self, patient_id: str, study_id: str) -> list[dict[str, Any]]:
        """The newest stored results of one study, newest first.
        The count limit keeps each result whole in the model observation."""
        with self.ledger._connect() as connection:
            rows = connection.execute(
                """SELECT result_id, analyzed_at, result FROM imaging_results
                   WHERE patient_id = %s AND study_id = %s ORDER BY analyzed_at DESC LIMIT %s""",
                (patient_id, study_id, STORED_RESULT_LIMIT),
            ).fetchall()
        return [
            {
                **_json(row["result"]),
                "result_id": row["result_id"],
                "analyzed_at": row["analyzed_at"],
            }
            for row in rows
        ]

    def export(self) -> list[dict[str, Any]]:
        with self.ledger._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM imaging_results ORDER BY analyzed_at"
            ).fetchall()
        return json.loads(json.dumps([dict(row) for row in rows], default=str))


def _json(value: Any) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else value
