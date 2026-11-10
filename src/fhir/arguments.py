"""Tool schemas and argument checks for the read-only FHIR tools."""

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, ValidationError

from src.fhir.ids import SCHEMA_PATTERN, valid_fhir_id
from src.llm.tool_schema import ToolArguments, parameters

FHIR_RESOURCE_TYPES = (
    "Patient",
    "Condition",
    "Observation",
    "MedicationStatement",
    "MedicationRequest",
    "AllergyIntolerance",
    "Encounter",
    "Procedure",
    "DiagnosticReport",
    "CarePlan",
    "ImagingStudy",
    "DocumentReference",
)

_FILTER_NAMES = (
    "_id",
    "name",
    "code",
    "category",
    "status",
    "clinical-status",
    "date",
    "onset-date",
    "recorded-date",
    "effective",
    "started",
    "authoredon",
    "birthdate",
)
# The filters that the tool schema shows. The type-specific date parameters stay valid,
# but date_from and date_to (or "date") cover them for every type.
_SCHEMA_FILTER_NAMES = ("_id", "code", "category", "status", "clinical-status", "date", "name")
# The clinical date parameters of each type on the server; others fail with a bare HTTP 400.
_DATE_PARAMETERS = {
    "Patient": ("birthdate",),
    # The first parameter is the one that "date" means for the type.
    "Condition": ("recorded-date", "onset-date"),
    "MedicationStatement": ("effective",),
    # MedicationRequest "date" is the dosage timing, not the date of the order.
    "MedicationRequest": ("authoredon",),
    "ImagingStudy": ("started",),
}
def main_date_parameter(resource_type: str) -> str:
    """The search parameter of the main clinical date of a resource type."""
    return _DATE_PARAMETERS.get(resource_type, ("date",))[0]


CLINICAL_DATE_FILTERS = frozenset({"date", *(p for ps in _DATE_PARAMETERS.values() for p in ps)})
_DATE_FILTERS = CLINICAL_DATE_FILTERS
_FILTERS_DESCRIPTION = (
    "Optional FHIR search filters. code, category, and status match coded values only; "
    "many records have text without a code, so use text to find a topic. A list matches "
    "any of its values. _id takes a list of record ids to read several records in one "
    "call. date is the main clinical date as prefixed FHIR dates, e.g. "
    '["ge2026-03-02", "le2026-03-11"]; date_from and date_to are simpler.'
)
_PLAIN_DATE = re.compile(r"\d{4}(-\d{2}(-\d{2})?)?")
_TEXT_LIMIT = 200
SORTS = ("newest", "oldest")
# FHIR date search: an optional prefix, then a year, month, day, or instant.
_FHIR_DATE = re.compile(
    r"(eq|ne|gt|lt|ge|le|sa|eb|ap)?\d{4}(-\d{2}(-\d{2}(T\d{2}:\d{2}(:\d{2}(\.\d+)?)?"
    r"(Z|[+-]\d{2}:\d{2})?)?)?)?"
)
_FILTER_VALUE_LIMIT = 200
_FILTER_VALUES_LIMIT = 50
PATIENT_SEARCH_LIMIT = 200

# Model-facing shape only. Value lengths and counts are enforced by _validate_filters;
# repeating those bounds for all nine filters in two tools cost about 750 tokens per request.
_FILTER_VALUE_SCHEMA = {
    "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]
}

QUERY_FHIR_PARAMETERS = {
    "type": "object",
    "properties": {
        "resource_type": {
            "type": "string",
            "enum": list(FHIR_RESOURCE_TYPES),
        },
        "patient_id": {
            "type": "string",
            "pattern": SCHEMA_PATTERN,
            "description": "Required for clinical resource types. Optional when resource_type is Patient.",
        },
        "search": {
            "type": "string",
            "minLength": 1,
            "maxLength": PATIENT_SEARCH_LIMIT,
            "description": "Optional patient name to search for when resource_type is Patient.",
        },
        "date_from": {
            "type": "string",
            "description": "First clinical date to include: YYYY, YYYY-MM, or YYYY-MM-DD.",
        },
        "date_to": {
            "type": "string",
            "description": "Last clinical date to include: YYYY, YYYY-MM, or YYYY-MM-DD. "
            "2026-06 includes all of June.",
        },
        "text": {
            "type": "string",
            "minLength": 1,
            "maxLength": _TEXT_LIMIT,
            "description": "Words to find in the record text (name, code text, values, notes, "
            "document text); a record matches when it holds all words, in any order, "
            "case-insensitive. Example: 'weight', 'memantine', 'MMSE', 'fall'.",
        },
        "sort": {
            "type": "string",
            "enum": list(SORTS),
            "description": "Order by main clinical date. Default newest.",
        },
        "count": {"type": "integer", "minimum": 1, "maximum": 50},
        "filters": {
            "type": "object",
            "properties": {name: _FILTER_VALUE_SCHEMA for name in _SCHEMA_FILTER_NAMES},
            "additionalProperties": False,
            "description": _FILTERS_DESCRIPTION,
        },
        "cursor": {
            "type": "string",
            "minLength": 24,
            "maxLength": 128,
            "description": "next_cursor of the previous page. Send it with resource_type and "
            "patient_id; other arguments may be repeated unchanged or left out.",
        },
    },
    "required": ["resource_type"],
    "additionalProperties": False,
}

COUNT_FHIR_PARAMETERS = {
    "type": "object",
    "properties": {
        "patient_id": {
            "type": "string",
            "pattern": SCHEMA_PATTERN,
            "description": "Omit to count all accessible Patient records. Required for other types.",
        },
        "queries": {
            "type": "array",
            "minItems": 1,
            "maxItems": len(FHIR_RESOURCE_TYPES),
            "description": "One query per resource type, with optional filters for each type.",
            "items": {
                "type": "object",
                "properties": {
                    "resource_type": {"type": "string", "enum": list(FHIR_RESOURCE_TYPES)},
                    "filters": QUERY_FHIR_PARAMETERS["properties"]["filters"],
                },
                "required": ["resource_type"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["queries"],
    "additionalProperties": False,
}


def _normalize_patient_id(value: object) -> object:
    """Normalize common variations of patient ID like p-001, p001, 1 to canonical P-001."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    match = re.match(r"^[pP]-?(\d{1,3})$", stripped)
    if match:
        return f"P-{int(match.group(1)):03d}"
    if stripped.isdigit() and 1 <= int(stripped) <= 999:
        return f"P-{int(stripped):03d}"
    return stripped


@dataclass(frozen=True)
class Query:
    """A validated query_fhir call."""

    resource_type: str
    patient_id: str | None
    count: int
    filters: list[tuple[str, str]]  # FHIR search parameters, with the date range and sort
    words: tuple[str, ...]  # lowercased words that a record must hold; empty for no text
    cursor: str | None
    arguments: dict  # the call without cursor and scope, to check a continuation


_QUERY_ARGUMENTS = {
    "resource_type", "patient_id", "count", "filters", "search", "date_from", "date_to",
    "text", "sort", "cursor",
}


def _date_bound(name: str, value: object) -> str:
    """A FHIR date search value for date_from or date_to. A year or month covers all its
    days. The upper bound is "lt" the next day: HAPI reads "le2025-11" as the first day
    of the month, and "lt" keeps every time on the last day."""
    if not isinstance(value, str) or not _PLAIN_DATE.fullmatch(value.strip()):
        raise ValueError(f"FHIR {name} must be YYYY, YYYY-MM, or YYYY-MM-DD.")
    parts = [int(part) for part in value.strip().split("-")]
    try:
        first = date(parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1)
    except ValueError:
        raise ValueError(f"FHIR {name} is not a valid date.") from None
    if name == "date_from":
        return f"ge{first.isoformat()}"
    if len(parts) == 3:
        end = first + timedelta(days=1)
    elif len(parts) == 2:
        end = date(first.year + first.month // 12, first.month % 12 + 1, 1)
    else:
        end = date(first.year + 1, 1, 1)
    return f"lt{end.isoformat()}"


def _validate_args(args: dict) -> tuple[str, str | None, int, list[tuple[str, str]], tuple]:
    resource_type = args.get("resource_type")
    if resource_type not in FHIR_RESOURCE_TYPES:
        raise ValueError("FHIR resource_type is not supported.")

    patient_id = args.get("patient_id")
    if patient_id is not None:
        patient_id = validate_patient_id(patient_id)
    elif resource_type != "Patient":
        raise ValueError("FHIR patient_id is required for clinical resource types.")

    search = args.get("search")
    if search is not None:
        if resource_type != "Patient":
            raise ValueError("search parameter is only supported when resource_type is Patient.")
        search = validate_search(search)

    # One upstream page fills several model pages; the rest stays in the cursor.
    count = args.get("count", 50)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 50:
        raise ValueError("FHIR count must be an integer from 1 to 50.")

    filters = _validate_filters(args.get("filters", {}), resource_type, patient_id)
    if search is not None:
        filters.append(("name", search))

    main_date = main_date_parameter(resource_type)
    bounds = [name for name in ("date_from", "date_to") if name in args]
    if bounds and any(n in CLINICAL_DATE_FILTERS for n, _ in filters):
        raise ValueError("Use date_from and date_to or a date filter, not both.")
    filters += [(main_date, _date_bound(name, args[name])) for name in bounds]
    sort = args.get("sort", "newest")
    if sort not in SORTS:
        raise ValueError("FHIR sort must be newest or oldest.")
    if resource_type != "Patient":
        filters.append(("_sort", ("-" if sort == "newest" else "") + main_date))

    text = args.get("text")
    words: tuple[str, ...] = ()
    if text is not None:
        if not isinstance(text, str) or not text.strip() or len(text) > _TEXT_LIMIT:
            raise ValueError("FHIR text must be non-blank words of at most 200 characters.")
        words = tuple(text.lower().split())

    return resource_type, patient_id, count, filters, words


def validate_search(search: object) -> str:
    """A patient name to search for, stripped."""
    if not isinstance(search, str) or not search.strip() or len(search) > PATIENT_SEARCH_LIMIT:
        raise ValueError("FHIR patient search must be a non-blank name of at most 200 characters.")
    return search.strip()


def _validate_filters(
    filters: object, resource_type: str, patient_id: str | None
) -> list[tuple[str, str]]:
    if not isinstance(filters, dict):
        raise ValueError("FHIR filters must be an object.")  # noqa: TRY004
    unsupported = set(filters) - set(_FILTER_NAMES)
    if unsupported:
        raise ValueError(
            f"FHIR filters do not support {', '.join(sorted(unsupported))}. Use sort for the "
            "order, date_from and date_to for clinical dates, and text for words."
        )
    if resource_type == "Patient" and patient_id is not None and "_id" in filters:
        raise ValueError("Patient search cannot override its patient id.")

    dates = _DATE_PARAMETERS.get(resource_type, ("date",))
    if "date" in filters and dates[0] != "date" and dates[0] not in filters:
        # "date" means the main clinical date of every type, so one name works for all.
        filters = {(dates[0] if name == "date" else name): value for name, value in filters.items()}
    for name in filters:
        if name in CLINICAL_DATE_FILTERS and name not in dates:
            raise ValueError(
                f"FHIR {resource_type} has no {name} filter. "
                f"{resource_type} dates: {', '.join(dates)}."
            )

    query_filters: list[tuple[str, str]] = []
    total_values = 0
    for name, values in filters.items():
        if isinstance(values, str):
            candidates = [values]
        elif isinstance(values, list) and values:
            candidates = values
        else:
            raise ValueError("FHIR filter values must be strings or non-empty string lists.")
        if len(candidates) > 20:
            raise ValueError("FHIR filters allow at most 20 values per parameter.")
        total_values += len(candidates)
        for value in candidates:
            if not isinstance(value, str) or not value.strip() or len(value) > _FILTER_VALUE_LIMIT:
                raise ValueError(
                    "FHIR filter values must be non-blank strings of at most 200 characters."
                )
            # A comma means OR, and HAPI rejects OR on date parameters with HTTP 400. Other
            # non-FHIR forms ("a..b", "a to b") reach the server and fail with a bare 400.
            if name in _DATE_FILTERS and not _FHIR_DATE.fullmatch(value.strip()):
                problem = "does not accept commas" if "," in value else "needs FHIR date syntax"
                raise ValueError(
                    f"FHIR {name} filter {problem}. Use one prefixed date per value and a list "
                    'for a date range, e.g. ["ge2026-03-02", "le2026-03-11"].'
                )
        # Repeated date parameters combine with AND to make a range. Other lists mean OR,
        # which FHIR writes as one comma-joined value.
        if name in _DATE_FILTERS:
            query_filters += [(name, value.strip()) for value in candidates]
        else:
            query_filters.append((name, ",".join(value.strip() for value in candidates)))
    if total_values > _FILTER_VALUES_LIMIT:
        raise ValueError("FHIR filters allow at most 50 values in total.")
    return query_filters


def _validate_resource_type(value: object) -> str:
    if value not in FHIR_RESOURCE_TYPES:
        raise ValueError("FHIR resource_type is not supported.")
    return value


def validate_patient_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("FHIR patient_id must be a valid FHIR id.")  # noqa: TRY004
    normalized = _normalize_patient_id(value)
    if not valid_fhir_id(normalized):
        raise ValueError("FHIR patient_id must be a valid FHIR id.")
    return normalized


def validate_count_args(args: object) -> tuple[str | None, dict[str, list[tuple[str, str]]]]:
    if not isinstance(args, dict):
        raise ValueError("FHIR count arguments must be an object.")  # noqa: TRY004
    if set(args) - {"patient_id", "queries"}:
        raise ValueError("FHIR count contains an unsupported argument.")
    patient_id = validate_patient_id(args["patient_id"]) if "patient_id" in args else None
    queries = args.get("queries")
    if not isinstance(queries, list) or not 1 <= len(queries) <= len(FHIR_RESOURCE_TYPES):
        raise ValueError("FHIR queries must contain from 1 to 12 resource queries.")
    validated = {}
    for query in queries:
        if not isinstance(query, dict) or set(query) - {"resource_type", "filters"}:
            raise ValueError("Each FHIR count query accepts only resource_type and filters.")
        resource_type = _validate_resource_type(query.get("resource_type"))
        if patient_id is None and resource_type != "Patient":
            raise ValueError("FHIR patient_id is required to count clinical resource types.")
        filters = _validate_filters(query.get("filters", {}), resource_type, patient_id)
        if resource_type in validated:
            raise ValueError("FHIR count queries must use unique resource types.")
        validated[resource_type] = filters
    return patient_id, validated


def prepare_query_args(args: object) -> Query:
    if not isinstance(args, dict):
        raise ValueError("FHIR query arguments must be an object.")  # noqa: TRY004
    unsupported = set(args) - _QUERY_ARGUMENTS
    if unsupported:
        raise ValueError(f"FHIR query does not support {', '.join(sorted(unsupported))}.")
    rest = {k: v for k, v in args.items() if k not in {"cursor", "resource_type", "patient_id"}}
    cursor = args.get("cursor")
    if cursor is None:
        resource_type, patient_id, count, filters, words = _validate_args(args)
        if words and patient_id is None:
            raise ValueError("FHIR text search needs a patient_id.")
        return Query(resource_type, patient_id, count, filters, words, None, rest)
    if not isinstance(cursor, str) or not 24 <= len(cursor) <= 128:
        raise ValueError("FHIR cursor is invalid.")
    resource_type = _validate_resource_type(args.get("resource_type"))
    patient_id = args.get("patient_id")
    if patient_id is not None:
        patient_id = validate_patient_id(patient_id)
    elif resource_type != "Patient":
        raise ValueError("FHIR patient_id is required for clinical resource types.")
    # A continuation may repeat any of the first call's other arguments, unchanged.
    return Query(resource_type, patient_id, 0, [], (), cursor, rest)


_ID_SCHEMA = {"pattern": SCHEMA_PATTERN}


def _fhir_id(value: str) -> str:
    if not valid_fhir_id(value):
        raise ValueError("invalid FHIR id")
    return value


ResourceType = Literal[FHIR_RESOURCE_TYPES]
PatientId = Annotated[str, AfterValidator(validate_patient_id), Field(json_schema_extra=_ID_SCHEMA)]
FhirId = Annotated[str, AfterValidator(_fhir_id), Field(json_schema_extra=_ID_SCHEMA)]


class ReadArguments(ToolArguments):
    resource_type: ResourceType
    patient_id: PatientId
    resource_id: FhirId
    version_id: FhirId | None = None


READ_FHIR_PARAMETERS = parameters(ReadArguments)

_READ_FIELD_ERRORS = {
    "resource_type": "FHIR resource_type is not supported.",
    "patient_id": "FHIR patient_id must be a valid FHIR id.",
    "resource_id": "FHIR resource_id must be a valid FHIR id.",
    "version_id": "FHIR version_id must be a valid FHIR id.",
}


def validate_read_args(args: object) -> tuple[str, str, str, str | None]:
    """Validate a read call. With several problems, the error names the first one in the
    order argument shape, argument names, then fields in declaration order."""
    try:
        call = ReadArguments.model_validate(args)
    except ValidationError as error:
        problems = error.errors()
    else:
        return call.resource_type, call.patient_id, call.resource_id, call.version_id
    if problems[0]["type"] == "model_type":
        raise ValueError("FHIR read arguments must be an object.")
    if any(problem["type"] == "extra_forbidden" for problem in problems):
        raise ValueError("FHIR read contains an unsupported argument.")
    raise ValueError(_READ_FIELD_ERRORS[problems[0]["loc"][0]])
