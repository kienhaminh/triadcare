"""Source IDs of a turn: the IDs a text cites and the IDs the tools of a turn returned."""

from __future__ import annotations

import json
import re

from src.agent.care_context import PLACEHOLDER, plain

# Results of these tools hold earlier answers, whose IDs this turn did not retrieve.
HISTORY_TOOLS = frozenset({"search_history", "list_history", "read_history"})


# Ledger message and observation IDs, FHIR references with or without the fhir: prefix,
# and imaging results. A reference ends before punctuation that closes a sentence.
_FHIR_TYPES = (
    "Patient|Observation|Condition|MedicationRequest|MedicationStatement|"
    "MedicationAdministration|Medication|Procedure|Encounter|DiagnosticReport|"
    "DocumentReference|AllergyIntolerance|Immunization|CarePlan|CareTeam|Goal|"
    "ServiceRequest|Composition|Practitioner|PractitionerRole|Organization|ImagingStudy|"
    "ClinicalImpression|FamilyMemberHistory|RiskAssessment|Consent|Communication|"
    "Appointment|Task|Provenance|Binary|Media|Device|Specimen|List|Flag|RelatedPerson|"
    "Coverage|EpisodeOfCare|NutritionOrder|QuestionnaireResponse|Questionnaire|Basic|"
    "DetectedIssue|Location|Bundle|AdverseEvent|Person"
)
_REFERENCE = re.compile(
    r"(?:obs:)?imported:[\w.-]+(?::[\w.-]+)*"
    rf"|(?:fhir:)?\b(?:{_FHIR_TYPES})/[A-Za-z0-9][\w.-]*"
    r"|imaging-result:[\w-]+"
)
_AGENT = r"(?:Care|EHR|Radiologist) Agent"
# An agent named where a source belongs: in a citation parenthesis or after "per".
_SPECIALIST = r"(?:Care|EHR|Radiologist)(?: Agent)?"
_AGENT_SOURCE = re.compile(
    rf"\([^()]*\b{_AGENT}\b[^()]*\)|\b(?:per|according to|source:|via)\s+(?:the\s+)?{_AGENT}"
    rf"|\[[^\[\]]*\b{_AGENT}\b[^\[\]]*\]"
    rf"|\[[^\[\]]*\b{_SPECIALIST} (?:report|consultation)\b[^\[\]]*\]"
    rf"|\b{_SPECIALIST} (?:report|consultation)\b",
    re.IGNORECASE,
)


# More messages of the same check-in after a message ID: ", :seq-4", "; seq-6", "and seq-8".
_MORE_SEQUENCES = re.compile(r"(?:\s*(?:,|;|and|&)\s*:?seq-\d+\b)+")


def cited_ids(text: str) -> list[str]:
    """Source references in text, first seen first, without the fhir: prefix. A record ID
    has a digit, so a word after a type ("Medication/Health") is prose, and a ledger ID
    with a template part ("day-N") is not a citation. A list of "seq-K" after a message ID
    names more messages of its check-in."""
    text = plain(text)
    found = []
    for match in _REFERENCE.finditer(text):
        reference = match.group(0).rstrip(".-").removeprefix("fhir:")
        if "/" in reference:
            if not re.search(r"\d", reference.split("/", 1)[1]):
                continue
        elif PLACEHOLDER.search(reference):
            continue
        references = [reference]
        head, _, last = reference.rpartition(":")
        more = _MORE_SEQUENCES.match(text, match.start() + len(reference))
        if reference.startswith("imported:") and last.startswith("seq-") and more:
            references += [f"{head}:seq-{n}" for n in re.findall(r"seq-(\d+)", more.group(0))]
        found += [r for r in references if r not in found]
    return found


def _dicts(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _dicts(item)


def returned_ids(trace: list[dict]) -> set[str]:
    """References the tool observations of a trace returned, plus the FHIR resources they
    returned as resourceType and id. Results of the conversation history tools do not count."""
    known: set[str] = set()
    for step in trace:
        observation = step.get("observation")
        if observation is None or step.get("tool") in HISTORY_TOOLS:
            continue
        known.update(cited_ids(json.dumps(observation, ensure_ascii=False)))
        for item in _dicts(observation):
            if isinstance(item.get("resourceType"), str) and isinstance(item.get("id"), str):
                known.add(f"{item['resourceType']}/{item['id']}")
    return known


def retrieved_ids(trace: list[dict]) -> set[str]:
    """The IDs that the record tools of a trace returned. A consultation result is text
    that a specialist wrote, and read_conversation returns conversation text; neither
    is retrieved evidence."""
    return returned_ids(
        [
            step
            for step in trace
            if not str(step.get("tool", "")).startswith("consult_")
            and step.get("tool") != "read_conversation"
        ]
    )


def _unresolved(text: str, known: set[str]) -> dict:
    cited = cited_ids(text)
    return {"cited": len(cited), "unresolved": [c for c in cited if c not in known]}


_MESSAGE_ID = re.compile(r"imported:[\w.-]+:day-\d+:(?:patient|caregiver):seq-\d+")


def unresolved_citations(text: str, known: set[str]) -> list[str]:
    """Cited IDs that are not in known, imported check-in IDs that do not name one
    message (a session ID or a malformed ID), and agents named where a source belongs."""
    return [
        c
        for c in cited_ids(text)
        if c not in known or (c.startswith("imported:") and not _MESSAGE_ID.fullmatch(c))
    ] + agent_sources(text)


def known_ids(trace: list[dict], agent: str) -> set[str]:
    """The IDs an agent can cite: for the Doctor Agent, the IDs all record tools of the
    turn returned; for a specialist, the IDs its own record tools returned."""
    if agent == "Doctor Agent":
        return retrieved_ids(trace)
    return retrieved_ids([s for s in trace if s.get("agent") == agent])


def citation_check(answer: str, trace: list[dict]) -> dict:
    """Code check of cited source IDs, with no model repair. The Doctor answer is checked
    against the IDs that the record tools of the trace returned. Each specialist's reports
    are checked against the IDs that its own record tools returned."""
    specialists: dict[str, dict] = {}
    for step in trace:
        result = step.get("observation")
        if not str(step.get("tool", "")).startswith("consult_") or not isinstance(result, dict):
            continue
        name = result.get("agent")
        if not name:
            continue
        known = retrieved_ids([s for s in trace if s.get("agent") == name])
        check = _unresolved(str(result.get("answer") or ""), known)
        total = specialists.setdefault(name, {"cited": 0, "unresolved": []})
        total["cited"] += check["cited"]
        total["unresolved"] += [c for c in check["unresolved"] if c not in total["unresolved"]]
    return {"answer": _unresolved(answer, retrieved_ids(trace)), "specialists": specialists}


def agent_sources(text: str) -> list[str]:
    """Passages that give a specialist agent as the source of a statement."""
    return [match.group(0) for match in _AGENT_SOURCE.finditer(text)]
