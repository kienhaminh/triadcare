"""FHIR resource ids that are safe to use as one URL path segment."""

from __future__ import annotations

import re

ID_PATTERN = r"[A-Za-z0-9.-]{1,64}"  # FHIR id characters and length
# valid_fhir_id as a JSON Schema pattern for tool parameters.
SCHEMA_PATTERN = r"^(?!\.{1,2}$)" + ID_PATTERN + "$"
_ID = re.compile(ID_PATTERN)
_PATH_SEGMENTS = frozenset({".", ".."})  # valid FHIR characters, but not resource ids


def valid_fhir_id(value: object) -> bool:
    return isinstance(value, str) and bool(_ID.fullmatch(value)) and value not in _PATH_SEGMENTS
