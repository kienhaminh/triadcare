"""Read retrieval evidence and consultations from a saved Doctor trace.

A trace is the list of tool records that one Doctor turn and its specialists produced
(`RunState.trace`). Each record has agent, tool, arguments, observation and error.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from src.agent.care_context import written_text

DOCTOR = "Doctor Agent"
# The manuscript's specialist agents; Care Agent retrieval is PKL context, not a specialist.
SPECIALIST_TOOLS = {"consult_ehr": "ehr", "consult_radiologist": "radiologist"}
CARE_TOOL = "consult_care"
ANALYSIS_TOOLS = {"analyze_mri", "analyze_pet", "analyze_dti"}
STORED_IMAGING_TOOL = "read_stored_imaging_results"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _dicts(value: Any) -> Iterator[dict]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _dicts(item)


def consultations(trace: list[dict], tool: str | None = None) -> list[dict]:
    """Doctor-level consultations, optionally of one tool, with question and answer text."""
    result = []
    for record in trace:
        name = record.get("tool", "")
        if record.get("agent") != DOCTOR or not name.startswith("consult_"):
            continue
        if tool is not None and name != tool:
            continue
        observation = (
            record.get("observation") if isinstance(record.get("observation"), dict) else {}
        )
        arguments = record.get("arguments") or {}
        context = arguments.get("context", "")
        result.append(
            {
                "tool": name,
                "question": arguments.get("question", ""),
                # A JSON request carries earlier statements as a list; judges read text.
                "context": context
                if isinstance(context, str)
                else json.dumps(context, ensure_ascii=False),
                "answer": written_text(observation),
                # Stored rows a Care answer cites, attached to the result by the runtime.
                "cited_observations": observation.get("cited_observations") or [],
                # The whole result: a JSON answer keeps the fields its agent wrote.
                "result": observation,
                "error": record.get("error"),
            }
        )
    return result


@dataclass
class Retrieval:
    """What the agent team actually read during one task."""

    ledger_text: dict[str, str] = field(default_factory=dict)  # message_id -> text read
    fhir: set[str] = field(default_factory=set)  # "Type/id"
    imaging_studies: set[str] = field(default_factory=set)  # analyzed or stored-result study IDs
    catalog_patients: set[str] = field(default_factory=set)
    imaging_patients: set[str] = field(default_factory=set)  # patients of those results

    def ledger_session_text(self, session_id: str) -> str:
        """Text read from one check-in session, in turn order (message IDs end in seq-N)."""
        prefix = f"{session_id}:"
        found = [
            (int(message_id.rsplit("seq-", 1)[-1]) if "seq-" in message_id else 0, text)
            for message_id, text in self.ledger_text.items()
            if message_id.startswith(prefix)
        ]
        return "\n".join(text for _, text in sorted(found))


def _merge(chunks: list[tuple[int, str]]) -> str:
    """Join content pages by character offset; overlapping pages extend each other."""
    segments: list[tuple[int, int, str]] = []
    for start, text in sorted(chunks):
        end = start + len(text)
        if segments and start <= segments[-1][1]:
            first, last, joined = segments[-1]
            if end > last:
                segments[-1] = (first, end, joined[: start - first] + text)
        else:
            segments.append((start, end, text))
    return "\n".join(text for _, _, text in segments)


def retrieval(trace: list[dict]) -> Retrieval:
    evidence = Retrieval()
    pages: dict[str, list[tuple[int, str]]] = {}
    for record in trace:
        if record.get("error") is not None:
            continue
        observation = record.get("observation")
        tool = record.get("tool", "")
        for item in _dicts(observation):
            message_id = item.get("message_id")
            if isinstance(message_id, str) and isinstance(item.get("content"), str):
                offset = int(item.get("content_offset") or 0)
                pages.setdefault(message_id, []).append((offset, item["content"]))
            if isinstance(item.get("resourceType"), str) and isinstance(item.get("id"), str):
                evidence.fhir.add(f"{item['resourceType']}/{item['id']}")
        if isinstance(observation, dict):
            analyzed = tool in ANALYSIS_TOOLS and isinstance(observation.get("study_id"), str)
            stored = tool == STORED_IMAGING_TOOL and observation.get("stored_results")
            if analyzed or stored:
                evidence.imaging_studies.add(observation["study_id"])
                if isinstance(observation.get("patient_id"), str):
                    evidence.imaging_patients.add(observation["patient_id"])
            if tool == "list_imaging_studies" and isinstance(observation.get("patient_id"), str):
                evidence.catalog_patients.add(observation["patient_id"])
    for message_id, chunks in pages.items():
        evidence.ledger_text[message_id] = _merge(chunks)
    return evidence


def source_retrieved(
    source_id: str,
    source_text: str,
    quote: str,
    patient_id: str,
    evidence: Retrieval,
) -> bool:
    """Whether the team read the anchored source of one reference unit.

    A check-in counts only when the anchored quote appears in the message text read from
    that check-in session; a recap is not a source. A FHIR resource counts when it was
    returned. An imaging study counts when it was analyzed or a stored result of it was
    returned; catalog identity or scope counts when the task patient's studies were
    listed, analyzed, or read.
    """
    kind, _, rest = source_id.partition(":")
    if kind == "raw-check-in":
        # raw-check-in:<scenario>:day-N:<role> is stored as session imported:<scenario>:day-N:<role>.
        text = normalize(evidence.ledger_session_text(f"imported:{rest}"))
        return bool(text) and normalize(quote) in text
    if kind == "fhir":
        return rest in evidence.fhir
    if kind == "imaging-fixture":
        try:
            studies = json.loads(source_text).get("image_ids", [])
        except (json.JSONDecodeError, AttributeError):
            studies = []
        if studies:
            return any(study in evidence.imaging_studies for study in studies)
        return patient_id in evidence.catalog_patients | evidence.imaging_patients
    return False
