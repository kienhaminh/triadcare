"""Record-update proposals, clinician decisions, governed FHIR writes, and audit history.

Agents can create, revise, and read proposals. Only trusted application code (the
backend API and the evaluation runner) can reject, authorize, and write. No agent tool
set contains those actions. Proposals, decisions, write attempts, and audit events are
stored in their own schema, separate from the Patient Knowledge Ledger.

Authorization checks run in a fixed order, and each check has its own denial reason:
role, confirmation, proposal state, proposal version, patient, content, target. The
role check runs first, so an unauthorized actor is denied for permission, not content.
"""

from __future__ import annotations

import base64
import contextvars
import copy
import hashlib
import json
import re
import uuid
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from typing import Any

import httpx
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from src import db
from src.config.database import database_settings
from src.fhir import server
from src.fhir.ids import ID_PATTERN

_TARGET_PATTERN = re.compile(r"^DocumentReference/" + ID_PATTERN + "$")
_FHIR_REFERENCE = re.compile(r"^[A-Z][A-Za-z]{1,63}/" + ID_PATTERN + "$")
MAX_CONTENT = 20_000

# The request (clinician message or evaluation task) that the current agent turn serves.
REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "record_request_id", default=None
)

CLINICIAN = "clinician"
INSUFFICIENT_PERMISSION = "insufficient_permission"
CONFIRMATION_REQUIRED = "confirmation_required"
PROPOSAL_NOT_PENDING = "proposal_not_pending"
STALE_PROPOSAL_VERSION = "stale_proposal_version"
PATIENT_MISMATCH = "patient_mismatch"
CONTENT_MISMATCH = "content_mismatch"
TARGET_MISMATCH = "target_mismatch"


class RecordUpdateError(ValueError):
    """A proposal request is invalid; the message is safe to show to an agent."""


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


# ---------------------------------------------------------------------------
# FHIR access for proposals and governed writes
# ---------------------------------------------------------------------------


class FhirRecords:
    """Read a target record and write an authorized version of it."""

    def __init__(self, transport: httpx.BaseTransport | None = None, timeout: float = 10.0):
        self.transport = transport
        self.timeout = timeout

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout, transport=self.transport, follow_redirects=False)

    @staticmethod
    def _url(target: str) -> str:
        return f"{server.root()}/{target}"

    @staticmethod
    def _headers() -> dict[str, str]:
        return server.headers()

    def read(self, target: str) -> dict[str, Any]:
        with self._client() as client:
            response = client.get(self._url(target), headers=self._headers())
        if response.status_code == 404:
            raise RecordUpdateError(f"Target record {target} does not exist.")
        if response.status_code != 200:
            raise RecordUpdateError(f"FHIR server returned HTTP {response.status_code}.")
        return response.json()

    def update(self, target: str, resource: dict[str, Any], version: str) -> httpx.Response:
        headers = {**self._headers(), "Content-Type": "application/fhir+json"}
        headers["If-Match"] = f'W/"{version}"'
        with self._client() as client:
            return client.put(self._url(target), headers=headers, json=resource)


def note_text(resource: dict[str, Any]) -> str | None:
    try:
        data = resource["content"][0]["attachment"]["data"]
        return base64.b64decode(data).decode("utf-8")
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _subject(resource: dict[str, Any]) -> str | None:
    reference = (resource.get("subject") or {}).get("reference")
    return reference.rsplit("/", 1)[-1] if isinstance(reference, str) else None


def fhir_reference(citation: str) -> str | None:
    """The Type/id a citation names: Type/id, the EHR citation form fhir:Type/id, or a URL
    that ends with Type/id."""
    candidate = citation.strip().removeprefix("fhir:")
    if _FHIR_REFERENCE.fullmatch(candidate):
        return candidate
    tail = "/".join(candidate.rstrip("/").split("/")[-2:])
    return tail if "://" in candidate and _FHIR_REFERENCE.fullmatch(tail) else None


def resource_version(resource: dict[str, Any]) -> str | None:
    value = (resource.get("meta") or {}).get("versionId")
    return str(value) if value is not None else None


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class RecordStore:
    """PostgreSQL tables for proposals, decisions, write attempts, and audit events."""

    def __init__(self, dsn: str | None = None, schema: str | None = None) -> None:
        settings = database_settings()
        self.dsn = dsn or settings.url
        if not self.dsn:
            raise RecordUpdateError("Record-update storage (DATABASE_URL) is not configured.")
        self.schema = db.require_schema_name(schema or settings.record_schema)
        self._initialize()

    def connect(self) -> AbstractContextManager[psycopg.Connection[dict[str, Any]]]:
        return db.connect(self.dsn, self.schema)

    def _initialize(self) -> None:
        with psycopg.connect(self.dsn, connect_timeout=5) as connection:
            connection.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema))
            )
            connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(self.schema)))
            for statement in (
                """CREATE TABLE IF NOT EXISTS proposals (
                    proposal_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    target TEXT NOT NULL,
                    target_version TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    sources JSONB NOT NULL,
                    parent_id TEXT REFERENCES proposals(proposal_id),
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'rejected', 'accepted')),
                    request_id TEXT,
                    created_at TEXT NOT NULL
                )""",
                """CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL REFERENCES proposals(proposal_id),
                    action TEXT NOT NULL CHECK (action IN ('reject', 'authorize_write')),
                    actor_role TEXT NOT NULL,
                    outcome TEXT NOT NULL CHECK (outcome IN ('recorded', 'denied')),
                    denial_reason TEXT,
                    reason TEXT,
                    request JSONB NOT NULL,
                    created_at TEXT NOT NULL
                )""",
                """CREATE TABLE IF NOT EXISTS write_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL REFERENCES decisions(decision_id),
                    proposal_id TEXT NOT NULL REFERENCES proposals(proposal_id),
                    target TEXT NOT NULL,
                    if_match_version TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('confirmed', 'failed', 'unconfirmed')),
                    written_version TEXT,
                    readback_matched BOOLEAN,
                    error TEXT,
                    created_at TEXT NOT NULL
                )""",
                """CREATE TABLE IF NOT EXISTS audit_events (
                    event_id BIGSERIAL PRIMARY KEY,
                    proposal_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    detail JSONB NOT NULL,
                    created_at TEXT NOT NULL
                )""",
            ):
                connection.execute(statement)


def _audit(connection, proposal_id: str, kind: str, detail: dict[str, Any]) -> None:
    connection.execute(
        "INSERT INTO audit_events(proposal_id, kind, detail, created_at) VALUES (%s, %s, %s, %s)",
        (proposal_id, kind, Jsonb(detail), _now()),
    )


def _public(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(row), default=str))


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class RecordUpdates:
    def __init__(self, store: RecordStore, fhir: FhirRecords) -> None:
        self.store = store
        self.fhir = fhir

    # -- Agent-facing: create, revise, read ------------------------------------

    def _checked_input(self, content: Any, rationale: Any, sources: Any) -> list[str]:
        if not isinstance(content, str) or not content.strip() or len(content) > MAX_CONTENT:
            raise RecordUpdateError(
                f"content must be nonblank text of at most {MAX_CONTENT} characters."
            )
        if not isinstance(rationale, str) or not rationale.strip():
            raise RecordUpdateError("rationale must be nonblank text.")
        if (
            not isinstance(sources, list)
            or not sources
            or any(not isinstance(item, str) or not item.strip() for item in sources)
        ):
            raise RecordUpdateError("sources must be a nonempty list of source references.")
        return [item.strip() for item in sources]

    def _target_version(self, patient_id: str, target: str) -> str:
        if not isinstance(target, str) or not _TARGET_PATTERN.fullmatch(target):
            raise RecordUpdateError("target must be DocumentReference/<id>.")
        resource = self.fhir.read(target)
        if _subject(resource) != patient_id:
            raise RecordUpdateError(f"{target} does not belong to patient {patient_id}.")
        if note_text(resource) is None:
            raise RecordUpdateError(f"{target} has no text note attachment to update.")
        version = resource_version(resource)
        if version is None:
            raise RecordUpdateError(f"{target} has no version; it cannot be updated safely.")
        return version

    def _create(
        self,
        connection,
        *,
        kind: str,
        patient_id: str,
        target: str,
        target_version: str,
        content: str,
        rationale: str,
        sources: list[str],
        parent_id: str | None,
        version: int,
        audit: dict[str, Any],
    ) -> dict[str, Any]:
        """Store a pending proposal and its audit event, whether first or a revision."""
        row = {
            "proposal_id": _new_id("proposal"),
            "patient_id": patient_id,
            "target": target,
            "target_version": target_version,
            "content": content,
            "content_sha256": content_sha256(content),
            "rationale": rationale,
            "sources": sources,
            "parent_id": parent_id,
            "version": version,
            "status": "pending",
            "request_id": REQUEST_ID.get(),
            "created_at": _now(),
        }
        columns = list(row)
        connection.execute(
            sql.SQL("INSERT INTO proposals({}) VALUES ({})").format(
                sql.SQL(", ").join(map(sql.Identifier, columns)),
                sql.SQL(", ").join(sql.Placeholder() * len(columns)),
            ),
            [Jsonb(row[c]) if c == "sources" else row[c] for c in columns],
        )
        _audit(
            connection,
            row["proposal_id"],
            kind,
            {
                "request_id": row["request_id"],
                "sources": sources,
                "target_version": target_version,
                "version": version,
                **audit,
            },
        )
        return _public(row)

    def propose(
        self, *, patient_id: str, target: str, content: str, rationale: str, sources: list[str]
    ) -> dict[str, Any]:
        sources = self._checked_input(content, rationale, sources)
        target_version = self._target_version(patient_id, target)
        with self.store.connect() as connection:
            pending = connection.execute(
                "SELECT proposal_id FROM proposals WHERE target = %s AND status = 'pending'",
                (target,),
            ).fetchone()
            if pending:
                raise RecordUpdateError(
                    f"Proposal {pending['proposal_id']} for {target} awaits a clinician decision."
                )
            return self._create(
                connection,
                kind="proposed",
                patient_id=patient_id,
                target=target,
                target_version=target_version,
                content=content,
                rationale=rationale,
                sources=sources,
                parent_id=None,
                version=1,
                audit={"target": target},
            )

    def revise(
        self, *, parent_id: str, content: str, rationale: str, sources: list[str]
    ) -> dict[str, Any]:
        sources = self._checked_input(content, rationale, sources)
        parent = self.get(parent_id)
        if parent["status"] != "rejected":
            raise RecordUpdateError(
                f"Only a rejected proposal can be revised; {parent_id} is {parent['status']}."
            )
        target_version = self._target_version(parent["patient_id"], parent["target"])
        with self.store.connect() as connection:
            existing = connection.execute(
                "SELECT proposal_id FROM proposals WHERE parent_id = %s", (parent_id,)
            ).fetchone()
            if existing:
                raise RecordUpdateError(
                    f"{parent_id} already has revision {existing['proposal_id']}."
                )
            return self._create(
                connection,
                kind="revised",
                patient_id=parent["patient_id"],
                target=parent["target"],
                target_version=target_version,
                content=content,
                rationale=rationale,
                sources=sources,
                parent_id=parent_id,
                version=parent["version"] + 1,
                audit={"parent_id": parent_id},
            )

    def get(self, proposal_id: str) -> dict[str, Any]:
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT * FROM proposals WHERE proposal_id = %s", (proposal_id,)
            ).fetchone()
        if row is None:
            raise RecordUpdateError(f"Proposal {proposal_id} does not exist.")
        return _public(row)

    def list(self, patient_id: str) -> list[dict[str, Any]]:
        """Proposals of one patient with their latest decision and write outcome."""
        with self.store.connect() as connection:
            proposals = connection.execute(
                "SELECT * FROM proposals WHERE patient_id = %s ORDER BY created_at", (patient_id,)
            ).fetchall()
            decisions = connection.execute(
                """SELECT d.* FROM decisions d JOIN proposals p USING (proposal_id)
                   WHERE p.patient_id = %s AND d.outcome = 'recorded' ORDER BY d.created_at""",
                (patient_id,),
            ).fetchall()
            writes = connection.execute(
                """SELECT w.* FROM write_attempts w JOIN proposals p USING (proposal_id)
                   WHERE p.patient_id = %s ORDER BY w.created_at""",
                (patient_id,),
            ).fetchall()
        result = []
        for row in proposals:
            item = _public(row)
            mine = [d for d in decisions if d["proposal_id"] == row["proposal_id"]]
            if mine:
                item["decision"] = {k: mine[-1][k] for k in ("action", "reason", "created_at")}
            attempts = [w for w in writes if w["proposal_id"] == row["proposal_id"]]
            if attempts:
                item["write"] = {k: attempts[-1][k] for k in ("status", "written_version", "error")}
            result.append(item)
        return result

    # -- Trusted application code only -----------------------------------------

    def _decide(
        self,
        connection,
        proposal_id: str,
        action: str,
        actor_role: str,
        request: dict[str, Any],
        *,
        denial: str | None,
        reason: str | None = None,
    ) -> str:
        decision_id = _new_id("decision")
        connection.execute(
            """INSERT INTO decisions(decision_id, proposal_id, action, actor_role, outcome,
                   denial_reason, reason, request, created_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                decision_id,
                proposal_id,
                action,
                actor_role,
                "denied" if denial else "recorded",
                denial,
                reason,
                Jsonb(request),
                _now(),
            ),
        )
        kind = (
            f"{action}_denied" if denial else ("rejected" if action == "reject" else "authorized")
        )
        _audit(
            connection,
            proposal_id,
            kind,
            {
                "decision_id": decision_id,
                "actor_role": actor_role,
                "denial_reason": denial,
                "reason": reason,
            },
        )
        return decision_id

    def _denial(
        self,
        proposal: dict[str, Any] | None,
        *,
        actor_role: str,
        confirmed: bool,
        proposal_version: int,
        patient_id: str,
        digest: str | None,
        target: str | None,
    ) -> str | None:
        if actor_role != CLINICIAN:
            return INSUFFICIENT_PERMISSION
        if not confirmed:
            return CONFIRMATION_REQUIRED
        if proposal is None or proposal["status"] != "pending":
            return PROPOSAL_NOT_PENDING
        if proposal["version"] != proposal_version:
            return STALE_PROPOSAL_VERSION
        if proposal["patient_id"] != patient_id:
            return PATIENT_MISMATCH
        if digest is not None and proposal["content_sha256"] != digest:
            return CONTENT_MISMATCH
        if target is not None and proposal["target"] != target:
            return TARGET_MISMATCH
        return None

    def _locked(self, connection, proposal_id: str) -> dict[str, Any] | None:
        return connection.execute(
            "SELECT * FROM proposals WHERE proposal_id = %s FOR UPDATE", (proposal_id,)
        ).fetchone()

    def reject(
        self,
        *,
        proposal_id: str,
        proposal_version: int,
        patient_id: str,
        actor_role: str,
        reason: str,
    ) -> dict[str, Any]:
        if not isinstance(reason, str) or not reason.strip():
            raise RecordUpdateError("A rejection needs a reason.")
        request = {"proposal_version": proposal_version, "patient_id": patient_id}
        with self.store.connect() as connection:
            proposal = self._locked(connection, proposal_id)
            if proposal is None:
                raise RecordUpdateError(f"Proposal {proposal_id} does not exist.")
            denial = self._denial(
                proposal,
                actor_role=actor_role,
                confirmed=True,
                proposal_version=proposal_version,
                patient_id=patient_id,
                digest=None,
                target=None,
            )
            decision_id = self._decide(
                connection, proposal_id, "reject", actor_role, request, denial=denial, reason=reason
            )
            if denial is None:
                connection.execute(
                    "UPDATE proposals SET status = 'rejected' WHERE proposal_id = %s",
                    (proposal_id,),
                )
        return {
            "decision_id": decision_id,
            "outcome": "denied" if denial else "recorded",
            "denial_reason": denial,
        }

    def authorize_and_write(
        self,
        *,
        proposal_id: str,
        proposal_version: int,
        patient_id: str,
        content_sha256: str,
        target: str,
        actor_role: str,
        confirmed: bool,
    ) -> dict[str, Any]:
        """Accept one reviewed version, authorize its write, write once, and read it back."""
        request = {
            "proposal_version": proposal_version,
            "patient_id": patient_id,
            "content_sha256": content_sha256,
            "target": target,
            "confirmed": confirmed,
        }
        with self.store.connect() as connection:
            proposal = self._locked(connection, proposal_id)
            if proposal is None:
                raise RecordUpdateError(f"Proposal {proposal_id} does not exist.")
            denial = self._denial(
                proposal,
                actor_role=actor_role,
                confirmed=confirmed,
                proposal_version=proposal_version,
                patient_id=patient_id,
                digest=content_sha256,
                target=target,
            )
            decision_id = self._decide(
                connection, proposal_id, "authorize_write", actor_role, request, denial=denial
            )
            if denial is not None:
                return {"decision_id": decision_id, "outcome": "denied", "denial_reason": denial}
            connection.execute(
                "UPDATE proposals SET status = 'accepted' WHERE proposal_id = %s", (proposal_id,)
            )
        write = self._write(decision_id, proposal)
        return {
            "decision_id": decision_id,
            "outcome": "recorded",
            "denial_reason": None,
            "write": write,
        }

    def _write(self, decision_id: str, proposal: dict[str, Any]) -> dict[str, Any]:
        """Write once with If-Match on the reviewed target version; never retry."""
        attempt = {
            "attempt_id": _new_id("write"),
            "decision_id": decision_id,
            "proposal_id": proposal["proposal_id"],
            "target": proposal["target"],
            "if_match_version": proposal["target_version"],
            "status": "failed",
            "written_version": None,
            "readback_matched": None,
            "error": None,
        }
        dispatched = False
        try:
            current = self.fhir.read(proposal["target"])
            resource = copy.deepcopy(current)
            resource.pop("meta", None)
            attachment = resource["content"][0]["attachment"]
            attachment["contentType"] = "text/plain"
            attachment["data"] = base64.b64encode(proposal["content"].encode("utf-8")).decode()
            # A stale description would give the note a second, conflicting text.
            if "description" in resource:
                resource["description"] = proposal["content"]
            dispatched = True
            response = self.fhir.update(proposal["target"], resource, proposal["target_version"])
            # FHIR specifies 412 for an If-Match conflict; HAPI returns 409.
            if response.status_code in (409, 412):
                attempt["error"] = "stale_target_version"
            elif response.status_code not in (200, 201):
                attempt["error"] = f"http_{response.status_code}"
            else:
                attempt["written_version"] = resource_version(response.json())
                saved = self.fhir.read(proposal["target"])
                matched = (
                    note_text(saved) == proposal["content"]
                    and resource_version(saved) == attempt["written_version"]
                )
                attempt["readback_matched"] = matched
                attempt["status"] = "confirmed" if matched else "failed"
                attempt["error"] = None if matched else "readback_mismatch"
        except httpx.TransportError as error:
            # After dispatch the outcome is unknown; it must not trigger a repeat write.
            attempt["status"] = "unconfirmed" if dispatched else "failed"
            attempt["error"] = f"{type(error).__name__}: {error}"
        except (RecordUpdateError, KeyError, IndexError, ValueError) as error:
            attempt["error"] = f"{type(error).__name__}: {error}"
        with self.store.connect() as connection:
            columns = list(attempt)
            connection.execute(
                sql.SQL("INSERT INTO write_attempts({}, created_at) VALUES ({}, %s)").format(
                    sql.SQL(", ").join(map(sql.Identifier, columns)),
                    sql.SQL(", ").join(sql.Placeholder() * len(columns)),
                ),
                [*attempt.values(), _now()],
            )
            _audit(
                connection,
                proposal["proposal_id"],
                f"write_{attempt['status']}",
                {
                    "decision_id": decision_id,
                    "attempt_id": attempt["attempt_id"],
                    "written_version": attempt["written_version"],
                    "readback_matched": attempt["readback_matched"],
                    "error": attempt["error"],
                },
            )
        return attempt

    def reviewed_note(self, proposal_id: str) -> dict[str, Any]:
        """The target note at the version the proposal was made against, and its current version.

        The review card compares the proposed text with this note, also after the decision.
        """
        proposal = self.get(proposal_id)
        current = self.fhir.read(proposal["target"])
        if _subject(current) != proposal["patient_id"]:
            raise RecordUpdateError("The target note belongs to another patient.")
        reviewed = self.fhir.read(f"{proposal['target']}/_history/{proposal['target_version']}")
        return {"resource": reviewed, "current_version": resource_version(current)}

    def open_citation(self, proposal_id: str, citation: str, ledger=None) -> dict[str, Any]:
        """Open one source that a proposal cites, as the review card does.

        Only a cited source opens. A FHIR citation returns the current resource with its
        identity and version; any other citation opens as a ledger message ID. The target
        note opens at the version the proposal was made against, so a later write does not
        change what the clinician reviews.
        """
        proposal = self.get(proposal_id)
        if citation not in proposal["sources"]:
            raise RecordUpdateError("The proposal does not cite this source.")
        reference = fhir_reference(citation)
        if reference is not None:
            if reference == proposal["target"]:
                reference = f"{reference}/_history/{proposal['target_version']}"
            resource = self.fhir.read(reference)
            subject = _subject(resource)
            if subject is not None and subject != proposal["patient_id"]:
                raise RecordUpdateError("The cited record belongs to another patient.")
            return {
                "citation": citation,
                "kind": "fhir",
                "reference": f"{resource.get('resourceType')}/{resource.get('id')}",
                "version": resource_version(resource),
                "resource": resource,
            }
        if ledger is None:
            from src.ledger.store import LedgerStore

            ledger = LedgerStore()
        message = ledger.read_message(proposal["patient_id"], citation)
        if message is None:
            raise RecordUpdateError("The cited source cannot be resolved for this patient.")
        return {"citation": citation, "kind": "ledger_message", "message": message}

    def export(self, patient_id: str) -> dict[str, list[dict[str, Any]]]:
        """Every stored governance row of one patient, for audit and scoring."""
        with self.store.connect() as connection:
            proposals = connection.execute(
                "SELECT * FROM proposals WHERE patient_id = %s ORDER BY created_at", (patient_id,)
            ).fetchall()
            ids = [row["proposal_id"] for row in proposals]
            tables = {"proposals": proposals}
            for table in ("decisions", "write_attempts", "audit_events"):
                order = "event_id" if table == "audit_events" else "created_at"
                tables[table] = connection.execute(
                    sql.SQL("SELECT * FROM {} WHERE proposal_id = ANY(%s) ORDER BY {}").format(
                        sql.Identifier(table), sql.Identifier(order)
                    ),
                    (ids,),
                ).fetchall()
        return json.loads(json.dumps(tables, default=str))


_DEFAULT: RecordUpdates | None = None


def default_record_updates() -> RecordUpdates:
    """Process-wide workflow bound to DATABASE_URL, RECORD_SCHEMA, and FHIR_BASE_URL."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = RecordUpdates(RecordStore(), FhirRecords())
    return _DEFAULT


def reset_default_record_updates() -> None:
    global _DEFAULT
    _DEFAULT = None
