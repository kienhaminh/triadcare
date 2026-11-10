"""EHR tools: bounded, patient-scoped FHIR reads, counts, and paginated queries."""

import asyncio
import json
import secrets
from dataclasses import dataclass

from src.fhir import bundle, server
from src.fhir.arguments import (
    CLINICAL_DATE_FILTERS,
    COUNT_FHIR_PARAMETERS,
    FHIR_RESOURCE_TYPES,
    QUERY_FHIR_PARAMETERS,
    READ_FHIR_PARAMETERS,
    main_date_parameter,
    prepare_query_args,
    validate_count_args,
    validate_patient_id,
    validate_read_args,
    validate_search,
)
from src.fhir.projection import project_resource

__all__ = [
    "COUNT_FHIR_PARAMETERS",
    "FHIR_RESOURCE_TYPES",
    "QUERY_FHIR_PARAMETERS",
    "READ_FHIR_PARAMETERS",
    "count_fhir_records",
    "list_fhir_records",
    "query_fhir",
    "read_fhir_resource",
    "search_patients",
    "validate_patient_id",
]

_TOOL_OUTPUT_LIMIT = 64 * 1024
# Serialized characters per model-facing query page. This stays below the ReAct
# observation limit (12,000) so whole records reach the model without a cut.
MODEL_PAGE_CHARS = 10_000
RECORD_PAGE_OUTPUT_LIMIT = server.RESPONSE_LIMIT
_CURSOR_STORE_LIMIT = 32
_CURSOR_PAGE_LIMIT = 100
_PATIENT_LIST_COUNT = 50


async def search_patients(args: dict) -> dict:
    """Search a bounded Patient page by name or exact FHIR id."""
    if not isinstance(args, dict):
        raise ValueError("FHIR patient search arguments must be an object.")  # noqa: TRY004
    if set(args) - {"search", "patient_id"}:
        raise ValueError("FHIR patient search contains an unsupported argument.")

    patient_id = validate_patient_id(args["patient_id"]) if "patient_id" in args else None
    search = args.get("search")
    if search is not None:
        search = validate_search(search)
    if patient_id is not None and search is not None:
        raise ValueError("FHIR patient search accepts a name or exact id, not both.")

    root = server.root()
    params = [("_count", str(_PATIENT_LIST_COUNT))]
    if patient_id is not None:
        params.append(("_id", patient_id))
    elif search is not None:
        params.append(("name", search))

    payload, request_url = await server.request_json(f"{root}/Patient", params=params)
    resources, total, raw_next_url = bundle.validated_bundle(
        payload, "Patient", patient_id, _PATIENT_LIST_COUNT, root
    )
    next_url = None
    if raw_next_url is not None:
        next_url = bundle.validate_next_url(raw_next_url, request_url, root, "Patient")

    projected = _project_resources(resources)
    result = {
        "resources": projected,
        "total": total,
        "has_more": (
            next_url is not None
            or (total is not None and total > len(projected))
            or (total is None and len(projected) == _PATIENT_LIST_COUNT)
        ),
    }
    _check_tool_output(result)
    return result


def _check_tool_output(result: dict, limit: int = _TOOL_OUTPUT_LIMIT) -> None:
    try:
        encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        raise ValueError(
            "FHIR server returned a resource that could not be processed safely."
        ) from None
    if len(encoded) > limit:
        raise ValueError(
            f"FHIR tool output exceeded {limit // 1024} KiB; use a smaller query or lower count."
        )


def _project_resources(resources: list[dict]) -> list[dict]:
    return [project_resource(resource) for resource in resources]


def _model_view(resource: dict, patient_id: str | None) -> dict:
    """A record as a model-facing page shows it: without the reference to the queried
    patient, and with _projection only when it reports omitted or cut clinical content."""
    view = dict(resource)
    for key in ("subject", "patient"):
        if (view.get(key) or {}).get("reference") == f"Patient/{patient_id}":
            del view[key]
    notes = view.get("_projection") or {}
    if not (
        notes.get("clinical_text_omitted") or notes.get("source_subsetted") or notes.get("truncated")
    ):
        view.pop("_projection", None)
    return view


def _fit_page(result: dict, resources: list[dict], budget: int | None) -> int:
    """Return how many leading resources fit in ``budget`` serialized characters.

    Characters are counted as the ReAct loop serializes observations. At least one
    resource is returned so a single large record still makes progress.
    """
    if budget is None or not resources:
        return len(resources)
    used = len(json.dumps({**result, "resources": []}, ensure_ascii=False))
    for index, resource in enumerate(resources):
        # Each later item also adds the ", " list separator.
        used += len(json.dumps(resource, ensure_ascii=False)) + (2 if index else 0)
        if used > budget:
            return max(1, index)
    return len(resources)


@dataclass(frozen=True)
class _Cursor:
    """Where a paginated search continues. Only query_fhir creates one."""

    root: str
    patient_id: str | None
    resource_type: str
    count: int
    next_url: str | None
    seen_urls: tuple[str, ...]  # canonical URLs of the pages fetched so far
    consumed: int  # records already returned to the caller
    total: int | None
    buffered: tuple[dict, ...]  # fetched and projected records not yet returned
    arguments: dict  # the first call without cursor and scope


def _resume(
    cursors: dict | None, cursor: str, root: str, resource_type: str, patient_id: str | None
) -> _Cursor:
    if cursors is None:
        raise ValueError("FHIR cursor is unavailable outside its original Doctor turn.")
    state = cursors.get(cursor)
    if state is None:
        raise ValueError("FHIR cursor is unknown or expired.")
    if (state.root, state.patient_id, state.resource_type) != (root, patient_id, resource_type):
        raise ValueError("FHIR cursor does not match this server, patient, or resource type.")
    if not state.buffered and len(state.seen_urls) >= _CURSOR_PAGE_LIMIT:
        raise ValueError("FHIR cursor state is invalid or expired.")
    return state


async def _fetch_page(
    endpoint: str,
    params: list[tuple[str, str]] | None,
    resource_type: str,
    patient_id: str | None,
    count: int,
    root: str,
    seen_urls: tuple[str, ...],
) -> tuple[list[dict], int | None, str | None, tuple[str, ...]]:
    """Fetch one upstream page: its projected records, total, next URL, and seen URLs."""
    payload, request_url = await server.request_json(endpoint, params=params)
    resources, total, raw_next_url = bundle.validated_bundle(
        payload, resource_type, patient_id, count, root
    )
    next_url = None
    if raw_next_url is not None:
        next_url = bundle.validate_next_url(raw_next_url, request_url, root, resource_type)
    seen_urls = (*seen_urls, bundle.canonical_url(request_url))
    if next_url is not None and bundle.canonical_url(next_url) in seen_urls:
        raise ValueError("FHIR pagination repeated a page and was stopped safely.")
    if len(seen_urls) > _CURSOR_PAGE_LIMIT:
        raise ValueError("FHIR pagination exceeded the safe page limit.")
    return _project_resources(resources), total, next_url, seen_urls


def _search_params(
    resource_type: str, patient_id: str | None, count: int, filters: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    # HAPI leaves out Bundle.total for a result of more than one page unless asked.
    params = [("_count", str(count)), ("_total", "accurate")]
    if resource_type == "Patient":
        if patient_id is not None:
            params.append(("_id", patient_id))
    else:
        params.append(("patient", f"Patient/{patient_id}"))
    return [*params, *filters]


# A text search or an index reads all records of a search in pages of this size, up to
# the scan limit. A patient record here has at most about 900 records of one type.
_SCAN_COUNT = 200
_SCAN_LIMIT = 2_000


async def _scan(
    root: str, resource_type: str, patient_id: str | None, filters: list[tuple[str, str]]
) -> tuple[list[dict], int | None, bool]:
    """All projected records of a search in its order: the records, the total, and
    whether the scan read them all."""
    records: list[dict] = []
    endpoint = f"{root}/{resource_type}"
    params = _search_params(resource_type, patient_id, _SCAN_COUNT, filters)
    seen: tuple[str, ...] = ()
    total = None
    while True:
        page, page_total, next_url, seen = await _fetch_page(
            endpoint, params, resource_type, patient_id, _SCAN_COUNT, root, seen
        )
        total = page_total if total is None else total
        records += page
        if next_url is None:
            return records, total, True
        if len(records) >= _SCAN_LIMIT:
            return records, total, False
        endpoint, params = next_url, None


# Fields that hold identifiers or technical values, not words of the record.
_NOT_TEXT = frozenset(
    {"id", "reference", "system", "url", "resourceType", "meta", "_projection", "contentType"}
)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in _NOT_TEXT:
                yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _matches(resource: dict, words: tuple[str, ...]) -> bool:
    """True when the record text holds every word, case-insensitive."""
    text = " ".join(_strings(resource)).lower()
    return all(word in text for word in words)


async def query_fhir(
    args: dict,
    *,
    cursors: dict[str, _Cursor] | None = None,
    output_limit: int = _TOOL_OUTPUT_LIMIT,
    page_chars: int | None = MODEL_PAGE_CHARS,
) -> dict:
    """Search patient resources; runtime callers may inject per-agent pagination state.

    Calls without ``cursors`` remain first-page-only helpers and return no continuation token.
    ``page_chars`` keeps model-facing pages below the ReAct observation limit: records
    that do not fit stay in the cursor and are returned by the next call without a new
    upstream request. ``output_limit`` is the hard size cap; the read-only records API
    passes the larger upstream cap and no ``page_chars`` because its output does not
    enter a prompt.
    """
    query = prepare_query_args(args)
    resource_type, patient_id, count = query.resource_type, query.patient_id, query.count
    filters, cursor = query.filters, query.cursor
    root = server.root()

    previous = None
    scan_complete = True
    if cursor is None and query.words:
        # Text is matched here, not by the server: FHIR token search matches codes only,
        # and many records have a code text without a code.
        consumed = 0
        scanned, _, scan_complete = await _scan(root, resource_type, patient_id, filters)
        projected = [record for record in scanned if _matches(record, query.words)]
        total = len(projected) if scan_complete else None
        next_url, seen_urls = None, ()
    elif cursor is None:
        consumed = 0
        projected, total, next_url, seen_urls = await _fetch_page(
            f"{root}/{resource_type}",
            _search_params(resource_type, patient_id, count, filters),
            resource_type,
            patient_id,
            count,
            root,
            (),
        )
    else:
        previous = _resume(cursors, cursor, root, resource_type, patient_id)
        if any(previous.arguments.get(k) != v for k, v in query.arguments.items()):
            raise ValueError(
                "A continuation must repeat the first call's arguments unchanged or leave "
                "them out. Start a new query to change them."
            )
        count, consumed = previous.count, previous.consumed
        # Records already fetched for an earlier page come first. When they are fewer
        # than one upstream page, the next upstream page fills the model page.
        projected = list(previous.buffered)
        total, next_url, seen_urls = previous.total, previous.next_url, previous.seen_urls
        if next_url is not None and len(projected) < count:
            endpoint = bundle.validate_next_url(next_url, root, root, resource_type)
            if bundle.canonical_url(endpoint) in seen_urls:
                raise ValueError("FHIR pagination repeated a page and was stopped safely.")
            page, total, next_url, seen_urls = await _fetch_page(
                endpoint, None, resource_type, patient_id, count, root, seen_urls
            )
            projected += page

    result = {
        "source": "fhir",
        "resource_type": resource_type,
        "patient_id": patient_id,
        "resources": [],
        "total": total,
        "returned": 0,
        "has_more": False,
        "next_cursor": None,
    }
    if not scan_complete:
        result["text_scan"] = f"Only the first {len(scanned)} records in sort order were searched."
    if cursor is None and not query.words and any(n in CLINICAL_DATE_FILTERS for n, _ in filters):
        # A date filter leaves out records dated outside it and records without that date;
        # the count without it keeps an empty page from reading as none in the record.
        # None tells the agent that the count is unknown, so the gap stays visible.
        result["outside_date_filter"] = None
        kept = [(n, v) for n, v in filters if n not in CLINICAL_DATE_FILTERS and n != "_sort"]
        try:
            if total is not None:
                result["outside_date_filter"] = max(
                    await _count(root, resource_type, patient_id, kept) - total, 0
                )
        except ValueError:
            pass
    if page_chars is not None:
        projected = [_model_view(resource, patient_id) for resource in projected]
    fitted = _fit_page(result, projected, page_chars)
    page, remaining = projected[:fitted], projected[fitted:]
    returned = consumed + len(page)
    result.update(
        resources=page,
        returned=len(page),
        has_more=bool(remaining)
        or next_url is not None
        or (total is not None and total > returned),
    )
    next_state = None
    if (remaining or next_url is not None) and cursors is not None:
        next_state = _Cursor(
            root,
            patient_id,
            resource_type,
            count,
            next_url,
            seen_urls,
            returned,
            total,
            tuple(remaining),
            previous.arguments if previous is not None else query.arguments,
        )
        result["next_cursor"] = secrets.token_urlsafe(32)
        while result["next_cursor"] in cursors:
            result["next_cursor"] = secrets.token_urlsafe(32)

    _check_tool_output(result, output_limit)
    if previous is not None:
        cursors.pop(cursor, None)
    if next_state is not None:
        cursors[result["next_cursor"]] = next_state
        while len(cursors) > _CURSOR_STORE_LIMIT:
            del cursors[next(iter(cursors))]
    return result


async def _count(
    root: str, resource_type: str, patient_id: str | None, filters: list[tuple[str, str]]
) -> int:
    params = [("_summary", "count"), ("_total", "accurate")]
    if patient_id is not None:
        params.append(
            ("_id", patient_id)
            if resource_type == "Patient"
            else ("patient", f"Patient/{patient_id}")
        )
    params.extend(filters)
    payload, _ = await server.request_json(f"{root}/{resource_type}", params=params)
    _, total, _ = bundle.validated_bundle(payload, resource_type, patient_id, 50, root)
    if total is None:
        raise ValueError("FHIR server did not report a total.")
    return total


_INDEX_DATES = (
    "recordedDate", "onsetDateTime", "effectiveDateTime", "effectivePeriod", "authoredOn",
    "period", "performedDateTime", "performedPeriod", "started", "date", "issued",
)
_INDEX_LABELS = ("code", "medicationCodeableConcept", "type", "description", "title", "category")


def _first_text(value) -> str | None:
    """The first text of a field: a string, a "text" or "start" member, or a list item."""
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = value.get("text") or value.get("start") or value.get("display")
    return value if isinstance(value, str) and value else None


def _index_entry(resource: dict) -> dict:
    """A record as one index line: its reference, clinical date, label, and status. The
    line has no values and no note text, so it is not a read of the record."""
    entry = {"reference": f"{resource['resourceType']}/{resource['id']}"}
    for name, keys in (("date", _INDEX_DATES), ("label", _INDEX_LABELS)):
        found = next((_first_text(resource[k]) for k in keys if _first_text(resource.get(k))), None)
        if found:
            entry[name] = found
    status = _first_text(resource.get("clinicalStatus")) or resource.get("status")
    if isinstance(status, str):
        entry["status"] = status
    return entry


def _index_rows(entries: list[dict]) -> list[str]:
    """One line per label, newest group first. A label with one record gives its date,
    status, and reference; a label with more gives the count, the date range, and the
    newest record."""
    groups: dict[str, list[dict]] = {}
    for entry in sorted(entries, key=lambda e: e.get("date", ""), reverse=True):
        groups.setdefault((entry.get("label") or "").lower(), []).append(entry)
    rows = []
    for members in groups.values():
        newest = members[0]
        label = newest.get("label") or "(no label)"
        date = newest.get("date", "no date")[:10]
        status = f" ({newest['status']})" if newest.get("status") else ""
        if len(members) == 1:
            rows.append(f"{label} | {date} | {newest['reference']}{status}")
        else:
            oldest = members[-1].get("date", "no date")[:10]
            rows.append(
                f"{label} | {len(members)} records {oldest}..{date} | newest "
                f"{newest['reference']}{status}"
            )
    return rows


# Notes, problems, and medicines first; long series of results and visits last.
_INDEX_ORDER = (
    "DocumentReference", "Condition", "MedicationRequest", "MedicationStatement",
    "AllergyIntolerance", "CarePlan", "ImagingStudy", "DiagnosticReport", "Encounter",
    "Observation", "Procedure",
)


async def list_fhir_records(args: dict) -> dict:
    """An index of a patient's whole record: for each resource type, the total and one
    line per label with its count, dates, and newest reference."""
    if not isinstance(args, dict) or set(args) != {"patient_id"}:
        raise ValueError("list_fhir_records accepts only patient_id.")
    patient_id = validate_patient_id(args["patient_id"])
    root = server.root()

    async def one(resource_type: str) -> dict:
        sort = [("_sort", f"-{main_date_parameter(resource_type)}")]
        records, total, complete = await _scan(root, resource_type, patient_id, sort)
        listed = {"resource_type": resource_type, "total": total}
        if not complete:
            listed["note"] = f"Only the {len(records)} newest records are indexed."
        listed["labels"] = _index_rows([_index_entry(r) for r in records])
        return listed

    indexed = await asyncio.gather(*(one(t) for t in _INDEX_ORDER))
    result = {
        "source": "fhir_index",
        "patient_id": patient_id,
        "empty_types": [t["resource_type"] for t in indexed if not t["total"]],
        "types": [t for t in indexed if t["total"]],
    }
    _check_tool_output(result)
    return result


async def count_fhir_records(args: dict) -> dict:
    """Count patient-scoped records, or all accessible Patients when no ID is supplied."""
    patient_id, queries = validate_count_args(args)
    root = server.root()
    counts = {}
    errors = {}
    for resource_type, filters in queries.items():
        try:
            counts[resource_type] = await _count(root, resource_type, patient_id, filters)
        except ValueError as error:
            counts[resource_type] = None
            errors[resource_type] = str(error)
    return {
        "source": "fhir",
        "patient_id": patient_id,
        "counts": counts,
        "errors": errors,
        "complete": not errors,
    }


async def read_fhir_resource(args: dict) -> dict:
    """Read one patient-owned resource, then project it before returning it."""
    resource_type, patient_id, resource_id, version_id = validate_read_args(args)
    root = server.root()
    endpoint = f"{root}/{resource_type}/{resource_id}"
    if version_id is not None:
        endpoint = f"{endpoint}/_history/{version_id}"
    payload, _ = await server.request_json(endpoint, not_found=True)
    if not isinstance(payload, dict):
        raise ValueError("FHIR server returned an invalid resource.")  # noqa: TRY004
    if payload.get("resourceType") == "OperationOutcome":
        raise ValueError("FHIR server returned an OperationOutcome.")
    if payload.get("resourceType") != resource_type:
        raise ValueError("FHIR server returned a resource of the wrong type.")
    if payload.get("id") != resource_id:
        raise ValueError("FHIR server returned a resource with the wrong id.")
    if version_id is not None:
        meta = payload.get("meta")
        if not isinstance(meta, dict) or meta.get("versionId") != version_id:
            raise ValueError("FHIR server returned a different resource version.")
    if resource_type == "Patient":
        if resource_id != patient_id:
            raise ValueError("FHIR server returned a different patient.")
    elif not bundle.patient_reference_matches(
        bundle.read_reference(payload, resource_type), patient_id, root
    ):
        raise ValueError("FHIR server returned a resource with an unresolved or different patient.")
    projected = project_resource(payload)
    result = {
        "source": "fhir",
        "resource_type": resource_type,
        "patient_id": patient_id,
        "resource": projected,
    }
    _check_tool_output(result)
    return result
