"""Check-in tools of Care Agent: recaps and source messages, with server-owned audience scope."""

import asyncio
import re
from typing import Annotated, Literal

import psycopg
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.agent.react import Tool
from src.agent.registry import bind, register_tool
from src.ledger.store import LedgerStore
from src.ledger.values import recorded_weekday


class ScopeFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    patient_id: str | None = Field(default=None, min_length=1, max_length=128)
    session_id: str | None = Field(default=None, min_length=1, max_length=256)
    speaker: Literal["patient", "caregiver", "care_agent"] | None = None
    date_from: str | None = None
    date_to: str | None = None
    as_of: str | None = None


class Filters(ScopeFilters):
    text: str | None = Field(default=None, min_length=1, max_length=500)


class Page(Filters):
    session_ids: list[Annotated[str, Field(min_length=1, max_length=256)]] | None = Field(
        default=None, min_length=1, max_length=20
    )
    limit: int = Field(default=100, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class CheckIns(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    patient_id: str | None = Field(default=None, min_length=1, max_length=128)
    reporter: Literal["patient", "caregiver"] | None = None
    date_from: str | None = None
    date_to: str | None = None
    as_of: str | None = None


class Search(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    patient_id: str | None = Field(default=None, min_length=1, max_length=128)
    query: str = Field(min_length=3, max_length=500)
    speaker: Literal["patient", "caregiver"] | None = None
    date_from: str | None = None
    date_to: str | None = None
    as_of: str | None = None
    limit: int = Field(default=12, ge=1, le=20)


class Read(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    patient_id: str | None = Field(default=None, min_length=1, max_length=128)
    message_id: str = Field(min_length=1, max_length=256)
    as_of: str | None = None
    content_offset: int = Field(default=0, ge=0)


def _parse(model, args):
    # An explicit null would silently mean "no filter"; report it so the caller supplies
    # the value it intended or omits the field.
    nulls = sorted(key for key, value in args.items() if value is None)
    if nulls:
        raise ValueError(
            f"null is not a value for: {', '.join(nulls)}. Omit fields you do not use."
        )
    try:
        return model.model_validate(args).model_dump(exclude_none=True)
    except ValidationError as exc:
        # Do not echo an invalid payload or arbitrary private fields into traces.
        fields = ", ".join(".".join(map(str, e["loc"])) for e in exc.errors())
        raise ValueError(f"Invalid check-in tool arguments: {fields}.") from None


def _published_schema(model) -> dict:
    """JSON schema without null alternatives or null defaults: an optional field is
    omitted when unused, never sent as null."""
    schema = model.model_json_schema()
    for name, spec in schema["properties"].items():
        if "anyOf" in spec:
            (kept,) = [option for option in spec["anyOf"] if option.get("type") != "null"]
            schema["properties"][name] = {**kept, "title": spec["title"]}
    return schema


def _project(message: dict, *, size: int = 600, offset: int = 0) -> dict:
    """Keep source qualifiers; omit vectors, storage labels, technical metadata and credentials."""
    keys = (
        "message_id",
        "patient_id",
        "session_id",
        "speaker",
        "participant_id",
        "audience",
        "recorded_at",
        "reporting_precision",
        "sequence",
        "similarity",
    )
    result = {key: message[key] for key in keys if key in message}
    if "recorded_at" in result:
        result["recorded_weekday"] = recorded_weekday(result["recorded_at"])
    raw = message.get("content", "")
    if offset > len(raw):
        raise ValueError("content_offset exceeds the source message length.")
    end = min(len(raw), offset + size)
    result["content"] = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw[offset:end])
    # Offsets reference the unchanged original, even if display controls are removed.
    result["content_offset"] = offset
    source_truncated = bool(message.get("content_truncated") or message.get("truncated"))
    result["next_content_offset"] = end if end < len(raw) or source_truncated else None
    result["content_truncated"] = offset > 0 or end < len(raw) or source_truncated
    return result


# The fields of a check-in, given once for its messages.
_CHECK_IN = ("session_id", "audience", "recorded_at", "recorded_weekday", "reporting_precision")
# Fields the model does not need: the result names the patient, and the message_id holds
# the sequence.
_HIDDEN = ("patient_id", "participant_id", "sequence")
_OFFSETS = ("content_offset", "next_content_offset", "content_truncated")


def _by_check_in(messages: list[dict]) -> list[dict]:
    """Messages grouped by check-in, in the order of each check-in's first message. Each check-in gives its session_id,
    reporter (audience), and recording date once. A message gives its message_id,
    speaker, and text, and its content offsets only when it is cut."""
    check_ins: dict[str, dict] = {}
    for message in messages:
        check_in = check_ins.setdefault(
            message.get("session_id"),
            {k: message[k] for k in _CHECK_IN if k in message} | {"messages": []},
        )
        hidden = set(_CHECK_IN) | set(_HIDDEN)
        if not message.get("content_truncated"):
            hidden |= set(_OFFSETS)
        check_in["messages"].append({k: v for k, v in message.items() if k not in hidden})
    return list(check_ins.values())


# More than the messages of one check-in; a session_id query returns them all.
_CHECK_IN_MESSAGES = 100


def _search(store: LedgerStore, selected: str, access: dict, parameters: dict) -> dict:
    """Ranked matches, each with the message before and after it in its check-in."""
    from src.ledger.recap import RecapStore
    from src.ledger.search import MessageIndex

    query, limit = parameters.pop("query"), parameters.pop("limit", 12)
    ranked = MessageIndex(store).search(selected, query, limit=limit, **access, **parameters)
    rank = {row["message_id"]: position for position, row in enumerate(ranked, 1)}
    sessions = dict.fromkeys(row["session_id"] for row in ranked)
    shown = []
    for session_id in sessions:
        rows = store.query_messages(
            selected, **access, session_id=session_id, limit=_CHECK_IN_MESSAGES
        )["messages"]
        hits = [i for i, row in enumerate(rows) if row["message_id"] in rank]
        keep = sorted({j for i in hits for j in (i - 1, i, i + 1) if 0 <= j < len(rows)})
        for j in keep:
            message = _project(rows[j], size=2000)
            if rows[j]["message_id"] in rank:
                message["match_rank"] = rank[rows[j]["message_id"]]
            shown.append(message)
    # The other check-ins of the scope, without message IDs or text: a pointer, not a read.
    scope = {k: v for k, v in parameters.items() if k in ("date_from", "date_to", "as_of")}
    others = [
        {
            "session_id": row["session_id"],
            "reporter": row["reporter"],
            "recorded_at": row["recorded_at"],
        }
        for row in RecapStore(store).list(
            selected,
            reporter=parameters.get("speaker"),
            participant_id=access["participant_id"],
            **scope,
        )
        if row["session_id"] not in sessions
    ]
    return {
        "source": "stored_check_ins",
        "patient_id": selected,
        "query": query,
        "check_ins": _by_check_in(shown),
        "check_ins_without_match": others,
    }


DEFINITIONS = (
    (
        "count_conversation_records",
        "count",
        Filters,
        (
            "Count stored messages and distinct sessions for an explicit patient, optionally "
            "by speaker, session, reporting date, as_of date, and text (all words, any "
            "order). Counts are NOT counts of symptoms/events. No content is returned."
        ),
    ),
    (
        "query_conversations",
        "query",
        Page,
        (
            "Read the source messages of an explicit patient, grouped by check-in in "
            "recording order. One call with date_from and date_to returns the messages of "
            "all check-ins of the period (up to 100 messages; continue with next_offset and "
            "the same filters). Reading the period gives details that a recap leaves out, and "
            "reports in which a later check-in describes an earlier event. Filter by session_id for one check-in, "
            "by session_ids for several check-ins in one call, by speaker, or by "
            "text: a message matches when it holds all words of text, in any order. Date "
            "filters and as_of refer to the recording date, not to the date of the reported "
            "event. Each check-in gives its session_id, reporter (audience), recorded_at, and "
            "recorded_weekday once, then its messages with message_id, speaker, and text. "
            "When a reporter dates an event with time words ('yesterday', 'Sunday'), the event "
            "date follows from recorded_at and recorded_weekday; when the words do not fix a "
            "date, keep only the words. A "
            "message longer than 2000 characters has content_truncated and "
            "next_content_offset for read_conversation_message."
        ),
    ),
    (
        "search_conversations",
        "search",
        Search,
        (
            "Search the source messages of an explicit patient by meaning, for one topic "
            "across a period, or to find a report that a recap does not list. "
            "Give one topic in plain words as query, for example 'night wandering or "
            "leaving the house' or 'nausea after the dose increase'; send one call per "
            "topic. Returns up to limit best-matching patient and caregiver messages "
            "(match_rank 1 is the best), each with the message before and after it, "
            "grouped by check-in in recording order. check_ins_without_match lists the "
            "other check-ins of the scope, without their text. A search can miss a "
            "message: no match is not a negative and not a read of the period. Read a whole "
            "check-in with query_conversations and its session_id. Date filters and as_of "
            "refer to the recording date."
        ),
    ),
    (
        "read_conversation_message",
        "read",
        Read,
        (
            "Read a persistent source message by message_id within the permitted patient and "
            "audience. Returns up to 4000 characters; use next_content_offset as content_offset "
            "for the rest. To read surrounding messages, query its session_id. as_of restricts "
            "the report date. Cite the persistent message_id, not a result position."
        ),
    ),
    (
        "list_check_ins",
        "check_ins",
        CheckIns,
        (
            "List the patient's check-ins in recording order, each with its session_id, "
            "reporter, recording date, and a short recap: summary, changes, and unknowns. Each "
            "change and unknown names the message_id of the message that states it; the "
            "message is the source, not the recap. Read the messages with query_conversations "
            "and the session_id (session_ids for several), and cite message IDs. A session_id is a handle for "
            "query_conversations, never a citation. Date filters and as_of refer to the "
            "recording date. A check-in with a status (failed or no_recap) has no recap; read "
            "its messages."
        ),
    ),
)
for _public, _, _model, _description in DEFINITIONS:
    register_tool(_public, _description, _published_schema(_model))


def ledger_tools(
    *,
    store: LedgerStore | None = None,
    patient_id: str | None = None,
    audience: str = "doctor",
    participant_id: str | None = None,
) -> dict[str, Tool]:
    if audience not in {"doctor", "patient", "caregiver"}:
        raise ValueError("Invalid ledger audience.")
    if audience != "doctor" and (not patient_id or not participant_id):
        raise ValueError("Direct Care chat requires a bound patient and participant.")

    def scope(arguments):
        requested = arguments.pop("patient_id", None)
        if patient_id and requested and requested != patient_id:
            raise ValueError("The requested patient is outside this conversation.")
        selected = patient_id or requested
        if not selected or not selected.strip():
            raise ValueError("Supply the patient's explicit patient_id to query stored check-ins.")
        return selected, {"audience": audience, "participant_id": participant_id}

    def execute(name: str, args: dict) -> dict:
        nonlocal store
        model = {
            "count": Filters,
            "query": Page,
            "read": Read,
            "check_ins": CheckIns,
            "search": Search,
        }[name]
        parameters = _parse(model, args)
        selected, access = scope(parameters)
        if store is None:
            try:
                store = LedgerStore()
            except RuntimeError as exc:
                raise ValueError(
                    "The check-in store is not configured; this is not an empty record."
                ) from exc
        if name == "check_ins":
            from src.ledger.recap import RecapStore

            # A stored recap needs no status; a failed or missing recap shows its status.
            check_ins = [
                {
                    **{k: v for k, v in row.items() if (k, v) != ("status", "stored")},
                    "recorded_weekday": recorded_weekday(row["recorded_at"]),
                }
                for row in RecapStore(store).list(
                    selected, participant_id=access["participant_id"], **parameters
                )
            ]
            return {"source": "check_in_recaps", "patient_id": selected, "check_ins": check_ins}
        if name == "search":
            return _search(store, selected, access, parameters)
        if name == "count":
            result = store.counts(selected, **access, **parameters)
            return {
                "source": "stored_check_ins",
                "patient_id": selected,
                **result,
                "unit": "messages_and_sessions_not_clinical_events",
            }
        if name == "read":
            message_id = parameters.pop("message_id")
            offset = parameters.pop("content_offset", 0)
            row = store.read_message(selected, message_id, **access, **parameters)
            if row is None:
                raise ValueError("Source message not found within the permitted scope.")
            return {
                "source": "stored_check_ins",
                "patient_id": selected,
                "message": _project(row, size=4000, offset=offset),
            }
        session_ids = parameters.pop("session_ids", None)
        if session_ids is not None:
            if "session_id" in parameters:
                raise ValueError("Use session_id or session_ids, not both.")
            # Several whole check-ins in one call, in the order given.
            parameters.update(limit=_CHECK_IN_MESSAGES, offset=0)
            rows = [
                row
                for session_id in dict.fromkeys(session_ids)
                for row in store.query_messages(
                    selected, **access, **parameters, session_id=session_id
                )["messages"]
            ]
            result = {"messages": rows, "total": len(rows), "has_more": False, "next_offset": None}
        else:
            if "session_id" in parameters:
                # One call reads a whole check-in; Care Agent reads check-ins, not pages.
                parameters.update(limit=_CHECK_IN_MESSAGES, offset=0)
            result = store.query_messages(selected, **access, **parameters)
        check_ins = _by_check_in([_project(row, size=2000) for row in result.pop("messages")])
        result.pop("model", None)
        return {
            "source": "stored_check_ins",
            "patient_id": selected,
            "check_ins": check_ins,
            **result,
        }

    tools = {}
    for public, operation, model, _ in DEFINITIONS:

        async def call(args, operation=operation):
            try:
                return await asyncio.to_thread(execute, operation, args)
            except (psycopg.Error, OSError):
                raise ValueError(
                    "The check-in store is unavailable; this is not an empty record."
                ) from None

        schema = _published_schema(model)
        if patient_id is None:
            schema.setdefault("required", []).append("patient_id")
            schema["properties"]["patient_id"] = {"type": "string", "minLength": 1}
        tools[public] = bind(public, call, parameters=schema)
    return tools
