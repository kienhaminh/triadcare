"""Deterministic allowlist projection for FHIR resources sent to the model."""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
from html.parser import HTMLParser
from typing import Any, ClassVar

from src.fhir.ids import valid_fhir_id
from src.fhir.projection_schema import (
    ATTACHMENT_METADATA,
    BOOLEAN,
    COMMON_META,
    NUMBER,
    REFERENCE,
    RESOURCE_SCHEMAS,
    STRING,
    TEXT,
)

POLICY = "clinical-v2"
MAX_STRING_LENGTH = 2_000
MAX_ARRAY_LENGTH = 20
MAX_RESOURCE_BYTES = 16 * 1024
_TRUNCATION_MARKER = "…[truncated]"
_OMITTED_LIMIT = 100


class _PlainTextParser(HTMLParser):
    """Extract visible text without executing or retaining HTML content."""

    _BLOCK_TAGS: ClassVar[set[str]] = {
        "address",
        "article",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "li",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in {"script", "style"}:
            self._hidden_depth += 1
        elif not self._hidden_depth and tag.lower() in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style"} and self._hidden_depth:
            self._hidden_depth -= 1
        elif not self._hidden_depth and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.parts.append(data)


def _plain_text(value: str) -> str:
    parser = _PlainTextParser()
    parser.feed(value)
    parser.close()
    text = "".join(parser.parts)
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _nonempty(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _attachment_text(attachment: dict) -> str | None:
    """The inline body of a text/plain attachment, or None when it is not readable text."""
    content_type = attachment.get("contentType")
    data = attachment.get("data")
    if not isinstance(content_type, str) or not isinstance(data, str):
        return None
    if content_type.split(";", 1)[0].strip().lower() != "text/plain":
        return None
    try:
        return base64.b64decode(data, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None


def _raw_marker_fields(resource: dict) -> list[str]:
    """Return known omitted clinical payload paths without reading their values."""
    omitted = []
    if _nonempty(resource.get("contained")):
        omitted.append("contained")
    if _nonempty(resource.get("extension")):
        omitted.append("extension")
    if _nonempty(resource.get("text")):
        narrative = resource.get("text")
        if isinstance(narrative, dict) and _nonempty(narrative.get("div")):
            omitted.append("text.div")
    resource_type = resource.get("resourceType")
    if resource_type == "DocumentReference":
        content = resource.get("content")
        if isinstance(content, list):
            for index, item in enumerate(content[:MAX_ARRAY_LENGTH]):
                attachment = item.get("attachment") if isinstance(item, dict) else None
                if isinstance(attachment, dict):
                    for key in ("data", "url"):
                        if _nonempty(attachment.get(key)) and not (
                            key == "data" and _attachment_text(attachment) is not None
                        ):
                            omitted.append(f"content[{index}].attachment.{key}")
    elif resource_type == "DiagnosticReport":
        forms = resource.get("presentedForm")
        if isinstance(forms, list):
            for index, attachment in enumerate(forms[:MAX_ARRAY_LENGTH]):
                if isinstance(attachment, dict):
                    for key in ("data", "url"):
                        if _nonempty(attachment.get(key)) and not (
                            key == "data" and _attachment_text(attachment) is not None
                        ):
                            omitted.append(f"presentedForm[{index}].{key}")
    elif resource_type == "ImagingStudy":
        series = resource.get("series")
        if isinstance(series, list):
            for index, item in enumerate(series[:MAX_ARRAY_LENGTH]):
                if isinstance(item, dict):
                    if _nonempty(item.get("instance")):
                        omitted.append(f"series[{index}].instance")
                    if _nonempty(item.get("endpoint")):
                        omitted.append(f"series[{index}].endpoint")
    return omitted


def _contains_modifier_extension(value: Any) -> bool:
    pending = [(value, 0)]
    seen: set[int] = set()
    while pending:
        current, depth = pending.pop()
        if depth > 10:
            raise ValueError("FHIR resource exceeds the safe nesting limit.")
        if isinstance(current, (dict, list)):
            identity = id(current)
            if identity in seen:
                raise ValueError("FHIR resource contains invalid nested data.")
            seen.add(identity)
        if isinstance(current, dict):
            if _nonempty(current.get("modifierExtension")):
                return True
            pending.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            pending.extend((child, depth + 1) for child in current)
    return False


class _Projector:
    def __init__(self) -> None:
        self.omitted: list[str] = []
        self.truncated: list[str] = []

    def omit(self, path: str) -> None:
        if len(path) > 256:
            path = path[:240] + "…[path truncated]"
        if path not in self.omitted:
            if len(self.omitted) < _OMITTED_LIMIT:
                self.omitted.append(path)
            elif "[additional omitted fields]" not in self.omitted:
                self.omitted.append("[additional omitted fields]")

    def _record_unknown(self, value: dict, schema: dict, path: str) -> None:
        root_fields = {"resourceType", "id", "meta", "patient", "subject", "text"}
        for key, child in value.items():
            if key not in schema and not (not path and key in root_fields) and _nonempty(child):
                self.omit(f"{path}.{key}" if path else key)

    def project(self, value: Any, schema: Any, path: str, depth: int = 0) -> Any:
        if depth > 10:
            self.truncated.append(path)
            return {"_truncated": True}
        if isinstance(schema, dict) and "_array" in schema:
            if not isinstance(value, list):
                raise ValueError(f"FHIR resource has an invalid clinical field at {path}.")
            output = []
            source_values = value
            source_was_limited = False
            if schema.get("_deduplicate"):
                unique = []
                scanned = source_values[:400]
                for item in scanned:
                    if not any(item == existing for existing in unique):
                        unique.append(item)
                    if len(unique) > MAX_ARRAY_LENGTH:
                        break
                source_values = unique
                source_was_limited = len(value) > len(scanned) and len(unique) <= MAX_ARRAY_LENGTH
            has_more = len(source_values) > MAX_ARRAY_LENGTH or source_was_limited
            item_limit = MAX_ARRAY_LENGTH - 1 if has_more else MAX_ARRAY_LENGTH
            limit = min(len(source_values), item_limit)
            for index, item in enumerate(source_values[:limit]):
                projected = self.project(item, schema["_array"], f"{path}[{index}]", depth + 1)
                if projected is not None:
                    output.append(projected)
            if has_more:
                output.append({"_truncated": True})
                self.truncated.append(path)
            return output
        if isinstance(schema, dict):
            if not isinstance(value, dict):
                raise ValueError(  # noqa: TRY004
                    f"FHIR resource has an invalid clinical field at {path}."
                )
            result = {}
            # A text/plain body is clinical text; other bodies and URLs stay omitted.
            text = _attachment_text(value) if schema is ATTACHMENT_METADATA else None
            if text is not None:
                value = {key: child for key, child in value.items() if key != "data"}
            self._record_unknown(value, schema, path)
            for key, child_schema in schema.items():
                if key not in value:
                    continue
                child_path = f"{path}.{key}" if path else key
                projected = self.project(value[key], child_schema, child_path, depth + 1)
                if projected is not None:
                    result[key] = projected
            if text is not None:
                result["text"] = self._bounded_string(text, f"{path}.text")
            return result or None
        if schema == STRING:
            if not isinstance(value, str):
                raise ValueError(f"FHIR resource has an invalid clinical field at {path}.")
            if len(value) > MAX_STRING_LENGTH:
                raise ValueError(f"FHIR resource has an oversized clinical field at {path}.")
            return value
        if schema == TEXT:
            if not isinstance(value, str):
                raise ValueError(f"FHIR resource has an invalid clinical field at {path}.")
            return self._bounded_string(_plain_text(value), path)
        if schema == BOOLEAN:
            if not isinstance(value, bool):
                raise ValueError(f"FHIR resource has an invalid clinical field at {path}.")
            return value
        if schema == NUMBER:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"FHIR resource has an invalid clinical field at {path}.")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"FHIR resource has an invalid clinical field at {path}.")
            return value
        return None

    def _bounded_string(self, value: str, path: str) -> str:
        if len(value) <= MAX_STRING_LENGTH:
            return value
        self.truncated.append(path)
        return value[: MAX_STRING_LENGTH - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


def project_resource(resource: dict) -> dict:
    """Return a compact, deterministic clinical projection of one FHIR resource.

    Unknown fields and nested objects are never copied through. The source mapping is
    read only and is not modified.
    """
    if not isinstance(resource, dict):
        raise ValueError("FHIR resource must be an object.")  # noqa: TRY004
    resource_type = resource.get("resourceType")
    if not isinstance(resource_type, str):
        raise ValueError("FHIR resource type is missing or invalid.")  # noqa: TRY004
    schema = RESOURCE_SCHEMAS.get(resource_type)
    if schema is None:
        raise ValueError("FHIR resource type is not supported by the clinical projection.")
    resource_id = resource.get("id")
    if not valid_fhir_id(resource_id):
        raise ValueError("FHIR resource is missing a safe source id.")
    if _contains_modifier_extension(resource):
        raise ValueError("FHIR resource has an unsupported modifierExtension.")

    projector = _Projector()
    for path in _raw_marker_fields(resource):
        projector.omit(path)

    result = {"resourceType": resource_type, "id": resource_id}
    meta = resource.get("meta")
    if meta is not None and not isinstance(meta, dict):
        raise ValueError("FHIR resource has invalid source metadata.")
    projected_meta = projector.project(meta or {}, COMMON_META, "meta") or {}
    for key in ("versionId",):
        value = (meta or {}).get(key)
        if key in (meta or {}) and (
            not isinstance(value, str) or not value or len(value) > MAX_STRING_LENGTH
        ):
            raise ValueError("FHIR resource has invalid source metadata.")
    if projected_meta:
        result["meta"] = projected_meta

    reference_key = "patient" if resource_type == "AllergyIntolerance" else "subject"
    if resource_type != "Patient" and reference_key in resource:
        reference = resource[reference_key]
        if (
            not isinstance(reference, dict)
            or not isinstance(reference.get("reference"), str)
            or not reference["reference"]
            or len(reference["reference"]) > MAX_STRING_LENGTH
        ):
            raise ValueError("FHIR resource has an invalid patient reference.")
        projected_reference = projector.project(reference, REFERENCE, reference_key)
        if projected_reference:
            result[reference_key] = projected_reference

    narrative = resource.get("text")
    if narrative is not None and (
        not isinstance(narrative, dict)
        or ("status" in narrative and not isinstance(narrative["status"], str))
        or ("div" in narrative and not isinstance(narrative["div"], str))
    ):
        raise ValueError("FHIR resource has invalid narrative text.")

    projected_body = projector.project(resource, schema, "") or {}
    for key, value in projected_body.items():
        if key not in {"resourceType", "id", "meta", "patient", "subject", "text", "_projection"}:
            result[key] = value

    narrative_text = narrative.get("div") if isinstance(narrative, dict) else None
    narrative_status = narrative.get("status") if isinstance(narrative, dict) else None
    if isinstance(narrative_text, str):
        clean_narrative = _plain_text(narrative_text)
        has_structured_data = any(
            key not in {"resourceType", "id", "meta", "patient", "subject"} and _nonempty(value)
            for key, value in result.items()
        )
        retain_narrative = (
            narrative_status in {None, "additional", "extensions"} or not has_structured_data
        )
        if retain_narrative and clean_narrative:
            projected_text = {}
            if isinstance(narrative_status, str):
                if len(narrative_status) > MAX_STRING_LENGTH:
                    raise ValueError("FHIR resource has an oversized narrative status.")
                projected_text["status"] = narrative_status
            projected_text["div"] = projector._bounded_string(clean_narrative, "text.div")
            result["text"] = projected_text
            if "text.div" in projector.omitted:
                projector.omitted.remove("text.div")
        elif clean_narrative:
            projector.omit("text.div")

    projection = {
        "policy": POLICY,
        "omitted_fields": projector.omitted,
        "clinical_text_omitted": "text.div" in projector.omitted,
        "source_subsetted": _has_subsetted_tag(meta),
        "truncated": bool(projector.truncated),
        "truncated_fields": projector.truncated,
    }
    result["_projection"] = projection
    if _json_size(result) > MAX_RESOURCE_BYTES:
        raise ValueError("FHIR resource exceeds the safe projection size limit.")
    return result


def _has_subsetted_tag(meta: Any) -> bool:
    if not isinstance(meta, dict) or not isinstance(meta.get("tag"), list):
        return False
    for tag in meta["tag"]:
        if isinstance(tag, dict) and tag.get("code") == "SUBSETTED":
            return True
    return False
