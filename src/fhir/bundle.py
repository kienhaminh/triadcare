"""Checks on what the FHIR server returns: Bundles, patient ownership, and next links.

The server is outside the trust boundary. A resource reaches an agent only after its
type, id, and patient reference match the request.
"""

import re
from urllib.parse import unquote, urljoin, urlsplit

import httpx

from src.fhir.ids import valid_fhir_id

_NEXT_URL_LIMIT = 8192


def _same_origin(left, right) -> bool:
    def origin(parts):
        scheme = parts.scheme.lower()
        host = (parts.hostname or "").lower()
        try:
            port = parts.port
        except ValueError:
            return None
        if port is None:
            port = 443 if scheme == "https" else 80
        return scheme, host, port

    return origin(left) is not None and origin(left) == origin(right)


def patient_reference_matches(reference: object, patient_id: str, root: str) -> bool:
    if not isinstance(reference, str) or not reference or any(c.isspace() for c in reference):
        return False

    try:
        reference_parts = urlsplit(reference)
    except ValueError:
        return False
    patient_path = f"Patient/{patient_id}"
    history_suffix = r"/_history/[A-Za-z0-9.-]{1,64}"
    relative_pattern = re.compile(re.escape(patient_path) + rf"(?:{history_suffix})?\Z")
    if not reference_parts.scheme and not reference_parts.netloc:
        return bool(relative_pattern.fullmatch(reference))

    if reference_parts.scheme.lower() not in {"http", "https"} or not reference_parts.netloc:
        return False
    try:
        root_parts = urlsplit(root)
        root_hostname = root_parts.hostname
    except ValueError:
        return False
    if (
        not _same_origin(reference_parts, root_parts)
        or reference_parts.query
        or reference_parts.fragment
        or reference_parts.username is not None
        or reference_parts.password is not None
    ):
        return False
    root_path = root_parts.path.rstrip("/")
    escaped_patient_path = f"{root_path}/{patient_path}"
    absolute_pattern = re.compile(re.escape(escaped_patient_path) + rf"(?:{history_suffix})?\Z")
    return bool(root_hostname and absolute_pattern.fullmatch(reference_parts.path))


def read_reference(resource: dict, resource_type: str) -> object:
    reference_field = "patient" if resource_type == "AllergyIntolerance" else "subject"
    reference = resource.get(reference_field)
    if not isinstance(reference, dict):
        return None
    return reference.get("reference")


def _validate_bundle(
    payload: object,
    resource_type: str,
    patient_id: str | None,
    count: int,
    root: str,
) -> dict:
    if not isinstance(payload, dict):
        raise TypeError("FHIR server returned an invalid Bundle.")
    if payload.get("resourceType") == "OperationOutcome":
        raise ValueError("FHIR server returned an OperationOutcome.")
    if payload.get("resourceType") != "Bundle" or payload.get("type") != "searchset":
        raise ValueError("FHIR server returned an invalid search Bundle.")

    total = payload.get("total")
    if total is not None and (isinstance(total, bool) or not isinstance(total, int) or total < 0):
        raise ValueError("FHIR server returned an invalid Bundle total.")

    entries = payload.get("entry", [])
    if not isinstance(entries, list):
        raise TypeError("FHIR server returned invalid Bundle entries.")
    resources = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("FHIR server returned invalid Bundle entries.")
        search = entry.get("search")
        if search is not None:
            if not isinstance(search, dict):
                raise TypeError("FHIR server returned invalid Bundle entries.")
            if search.get("mode") == "outcome":
                raise ValueError("FHIR server returned an error entry.")
        entry_response = entry.get("response")
        if entry_response is not None:
            if not isinstance(entry_response, dict):
                raise TypeError("FHIR server returned invalid Bundle entries.")
            status = entry_response.get("status")
            if not isinstance(status, str) or not status[:3].isdigit():
                raise TypeError("FHIR server returned invalid Bundle entries.")
            if not 200 <= int(status[:3]) < 300:
                raise ValueError("FHIR server returned an error entry.")
        resource = entry.get("resource")
        if not isinstance(resource, dict):
            raise TypeError("FHIR server returned an invalid resource entry.")
        returned_type = resource.get("resourceType")
        if returned_type == "OperationOutcome":
            raise ValueError("FHIR server returned an OperationOutcome.")
        if returned_type != resource_type:
            raise ValueError("FHIR server returned a resource of the wrong type.")
        if not valid_fhir_id(resource.get("id")):
            raise ValueError("FHIR server returned a resource with an invalid id.")
        if resource_type == "Patient":
            if patient_id is not None and resource.get("id") != patient_id:
                raise ValueError("FHIR server returned a different patient.")
        elif not patient_reference_matches(
            read_reference(resource, resource_type), patient_id, root
        ):
            raise ValueError(
                "FHIR server returned a resource with an unresolved or different patient."
            )
        resources.append(resource)

    if total is not None and total < len(resources):
        raise ValueError("FHIR server returned a total smaller than its matching records.")

    links = payload.get("link", [])
    if not isinstance(links, list):
        raise TypeError("FHIR server returned invalid Bundle links.")
    next_links = []
    for link in links:
        if not isinstance(link, dict) or not isinstance(link.get("relation"), str):
            raise TypeError("FHIR server returned invalid Bundle links.")
        if link["relation"] == "next":
            next_links.append(link.get("url"))
    if len(next_links) > 1:
        raise ValueError("FHIR server returned multiple next-page links.")
    if len(resources) > count:
        raise ValueError(
            "FHIR server returned more records than requested; retry with a smaller count."
        )
    return {
        "total": total,
        "resources": resources,
        "next_url": next_links[0] if next_links else None,
    }


def validated_bundle(
    payload: object,
    resource_type: str,
    patient_id: str | None,
    count: int,
    root: str,
) -> tuple[list[dict], int | None, object]:
    try:
        checked = _validate_bundle(payload, resource_type, patient_id, count, root)
    except TypeError:
        raise ValueError("FHIR server returned an invalid Bundle.") from None
    return checked["resources"], checked["total"], checked["next_url"]


def canonical_url(url: str) -> str:
    return str(httpx.URL(url))


def validate_next_url(raw_url: object, base_url: str, root: str, resource_type: str) -> str:
    if (
        not isinstance(raw_url, str)
        or not raw_url
        or len(raw_url) > _NEXT_URL_LIMIT
        or any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in raw_url
        )
        or "\\" in raw_url
        or "#" in raw_url
    ):
        raise ValueError("FHIR server returned an unsafe next-page link.")

    try:
        raw_parts = urlsplit(raw_url)
        # Reject literal and encoded traversal, separators, and backslashes before urljoin
        # can normalize them into an apparently safe namespace path.
        for segment in raw_parts.path.split("/"):
            decoded_segment = unquote(segment)
            if (
                decoded_segment in {".", ".."}
                or "/" in decoded_segment
                or "\\" in decoded_segment
                or re.search(r"%(?![0-9A-Fa-f]{2})", segment)
            ):
                raise ValueError("FHIR server returned an unsafe next-page link.")
        resolved = urljoin(base_url, raw_url)
        parts = urlsplit(resolved)
        root_parts = urlsplit(root)
        # Trigger validation for malformed ports and bracketed host syntax.
        _ = parts.port
        root_path = root_parts.path.rstrip("/") or "/"
        allowed_paths = {root_path, f"{root_path.rstrip('/')}/"}
        allowed_paths.add(f"{root_path.rstrip('/')}/{resource_type}")
    except ValueError as error:
        if str(error) == "FHIR server returned an unsafe next-page link.":
            raise
        raise ValueError("FHIR server returned an unsafe next-page link.") from None

    if (
        parts.scheme.lower() not in {"http", "https"}
        or not parts.netloc
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
        or not _same_origin(parts, root_parts)
        or parts.path not in allowed_paths
    ):
        raise ValueError("FHIR server returned an unsafe next-page link.")
    return resolved
