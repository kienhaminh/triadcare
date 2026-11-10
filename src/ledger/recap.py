"""CareAgent recap of one stored check-in: its main content, changes, and unknowns.

The transcript stays stored and is the source; the recap only shows what a check-in
holds, so that Care Agent can choose which check-ins to read. One forced tool call per
check-in; a failure is recorded, and Care Agent then reads that check-in directly.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from src.ledger.store import LedgerStore
from src.ledger.temporal import scope_as_of
from src.ledger.values import recorded_weekday
from src.llm import client as llm

PROMPT = (Path(__file__).resolve().parents[1] / "prompt" / "care_recap.md").read_text(
    encoding="utf-8"
)
# One retry when the answer has no valid tool call.
MAX_ATTEMPTS = 2
# Each change or unknown names the message that states it, so an agent can cite that
# message. The message, not the recap item, is the source.
_ITEMS = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"text": {"type": "string"}, "message_id": {"type": "string"}},
        "required": ["text", "message_id"],
    },
}
TOOL = {
    "type": "function",
    "function": {
        "name": "record_recap",
        "description": "Record the recap of one check-in.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string"}, "changes": _ITEMS, "unknowns": _ITEMS},
            "required": ["summary", "changes", "unknowns"],
        },
    },
}

Complete = Callable[..., dict]


class RecapStore:
    """The recaps table in the ledger schema; one row per check-in session."""

    def __init__(self, ledger: LedgerStore) -> None:
        self.ledger = ledger
        with ledger._connect() as connection:
            ledger.lock_schema_ddl(connection)
            connection.execute(
                """CREATE TABLE IF NOT EXISTS recaps (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
                    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                    reporter TEXT NOT NULL CHECK (reporter IN ('patient', 'caregiver')),
                    participant_id TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    changes JSONB NOT NULL,
                    unknowns JSONB NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('stored', 'failed')),
                    model TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL
                )"""
            )

    def save(self, session: dict, recorded_at: str, recap: dict, **record: Any) -> dict:
        row = {
            "session_id": session["session_id"],
            "patient_id": session["patient_id"],
            "reporter": session["audience"],
            "participant_id": session["participant_id"],
            "recorded_at": recorded_at,
            "summary": recap.get("summary", ""),
            "changes": recap.get("changes", []),
            "unknowns": recap.get("unknowns", []),
            "created_at": datetime.now(UTC).isoformat(),
            **record,
        }
        columns = list(row)
        values = [Jsonb(v) if isinstance(v, list) else v for v in row.values()]
        with self.ledger._connect() as connection:
            connection.execute(
                f"""INSERT INTO recaps ({", ".join(columns)})
                    VALUES ({", ".join("%s" for _ in columns)})
                    ON CONFLICT (session_id) DO UPDATE SET
                    {", ".join(f"{c} = EXCLUDED.{c}" for c in columns[1:])}""",
                values,
            )
        return row

    def list(
        self,
        patient_id: str,
        *,
        reporter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        as_of: str | None = None,
        participant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """The patient's check-ins in recording order, each with its recap. A check-in
        without a recap (for example a live check-in) has status "no_recap". Dates are
        recording dates: the date of the check-in's last message."""
        clauses = ["s.patient_id = %s", "s.audience IN ('patient', 'caregiver')"]
        parameters: list[Any] = [patient_id]
        if reporter:
            clauses.append("s.audience = %s")
            parameters.append(reporter)
        if participant_id:
            clauses.append("s.participant_id = %s")
            parameters.append(participant_id)
        for lower, upper in ((date_from, date_to), (None, as_of), (None, scope_as_of(patient_id))):
            dates, values = LedgerStore._date_filter("m.recorded_at", lower, upper)
            clauses += dates
            parameters += values
        with self.ledger._connect() as connection:
            rows = connection.execute(
                f"""WITH m AS (
                        SELECT session_id, MAX(recorded_at) AS recorded_at
                        FROM messages GROUP BY session_id
                    )
                    SELECT s.session_id, s.audience AS reporter, m.recorded_at,
                           r.summary, r.changes, r.unknowns,
                           COALESCE(r.status, 'no_recap') AS status
                    FROM sessions AS s JOIN m USING (session_id)
                    LEFT JOIN recaps AS r USING (session_id)
                    WHERE {" AND ".join(clauses)}
                    ORDER BY m.recorded_at, s.audience, s.session_id""",
                parameters,
            ).fetchall()
        return [{k: v for k, v in row.items() if v is not None} for row in rows]

    def export_scenario(self, scenario_id: str) -> list[dict[str, Any]]:
        with self.ledger._connect() as connection:
            rows = connection.execute(
                """SELECT r.* FROM recaps AS r JOIN sessions AS s USING (session_id)
                   WHERE s.scenario_id = %s ORDER BY r.recorded_at, r.reporter""",
                (scenario_id,),
            ).fetchall()
        return [dict(row) for row in rows]


def _request(session: dict, messages: list[dict]) -> list[dict]:
    transcript = "\n".join(
        f"[{m['message_id']}] "
        f"{'Care Agent' if m['speaker'] == 'care_agent' else m['speaker']}: {m['content']}"
        for m in messages
    )
    user = (
        f"Reporter: {session['audience']}\nCheck-in date: {messages[-1]['recorded_at'][:10]}"
        f" ({recorded_weekday(messages[-1]['recorded_at'])})"
        f"\n\nTranscript:\n{transcript}"
    )
    return [{"role": "system", "content": PROMPT}, {"role": "user", "content": user}]


def _recap(answer: dict, message_ids: set[str]) -> dict:
    message = answer.get("message") or {}
    calls = [c for c in message.get("tool_calls") or [] if c["function"]["name"] == "record_recap"]
    text = calls[0]["function"]["arguments"] if len(calls) == 1 else message.get("content")
    if not isinstance(text, str) or "{" not in text:
        raise ValueError("no record_recap call")
    recap = json.loads(text[text.index("{") : text.rindex("}") + 1])
    if not isinstance(recap.get("summary"), str) or not recap["summary"].strip():
        raise ValueError("no summary")
    for key in ("changes", "unknowns"):
        items = recap.get(key) or []
        if not all(isinstance(item, dict) and isinstance(item.get("text"), str) for item in items):
            raise ValueError(f"{key} is not a list of items with text")
        # The transcript shows each ID in brackets; a model can copy them.
        for item in items:
            item["message_id"] = str(item.get("message_id") or "").strip().strip("[]").strip()
        unknown = [item["message_id"] for item in items if item["message_id"] not in message_ids]
        if unknown:
            raise ValueError(f"{key} names messages that are not in the check-in: {unknown}")
        recap[key] = [
            {"text": item["text"].strip(), "message_id": item["message_id"]}
            for item in items
            if item["text"].strip()
        ]
    return recap


def recap_session(
    recaps: RecapStore, session_id: str, *, complete: Complete | None = None
) -> dict[str, Any]:
    """Recap one stored check-in; a failure is stored as a failed recap, not raised."""
    session = recaps.ledger.get_session(session_id)
    if session is None or session["audience"] not in {"patient", "caregiver"}:
        raise ValueError(f"{session_id} is not a patient or caregiver check-in.")
    messages = recaps.ledger.query_messages(
        session["patient_id"], session_id=session_id, limit=100
    )["messages"]
    if not messages:
        raise ValueError(f"Session {session_id} has no messages.")
    request, errors = _request(session, messages), []
    for _ in range(MAX_ATTEMPTS):
        try:
            answer = (complete or llm.complete)(request, tools=[TOOL], tool_choice="required")
            recap = _recap(answer, {m["message_id"] for m in messages})
        except (ValueError, KeyError, llm.LLMError) as error:
            errors.append(f"{type(error).__name__}: {error}")
            continue
        return recaps.save(
            session,
            messages[-1]["recorded_at"],
            recap,
            status="stored",
            model=llm.settings().model,
            error="; ".join(errors) or None,
        )
    return recaps.save(
        session,
        messages[-1]["recorded_at"],
        {},
        status="failed",
        model=llm.settings().model,
        error="; ".join(errors),
    )
