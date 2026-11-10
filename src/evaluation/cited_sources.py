"""Resolve the sources cited in one output."""

import re

from src.agent.care_context import PLACEHOLDER
from src.agent.citations import _MORE_SEQUENCES, _REFERENCE
from src.evaluation.judge import _STATEMENT_CITATION

_CITATION = re.compile(rf"(?:{_STATEMENT_CITATION.pattern})|(?:{_REFERENCE.pattern})")


def select_cited_sources(
    output: str, sources: dict[str, str]
) -> tuple[dict[str, str], list[str]]:
    """Return cited source blocks and unresolved IDs, without reading source text as output.

    Check-in citations resolve only when the exact message is present in a source block.
    Returned blocks follow source order; unresolved IDs follow output order.
    """
    message_sources: dict[str, list[str]] = {}
    for source_id, text in sources.items():
        if not source_id.startswith("raw-check-in:"):
            continue
        prefix = source_id.replace("raw-check-in:", "imported:", 1) + ":seq-"
        for line in text.splitlines():
            message_id, separator, _ = line.partition(": ")
            if separator and message_id.startswith(prefix) and message_id[len(prefix):].isdigit():
                message_sources.setdefault(message_id, []).append(source_id)

    selected: set[str] = set()
    missing: list[str] = []
    seen: set[str] = set()
    for match in _CITATION.finditer(output):
        citation = match.group().strip("[]()").removeprefix("obs:")
        # Prefer an exact ID, since punctuation can also be part of an ID.
        known = sources.keys() | message_sources.keys()
        while (
            citation not in known
            and f"fhir:{citation}" not in sources
            and citation.endswith((".", ":"))
        ):
            citation = citation[:-1]
        if citation.startswith("imported:") and PLACEHOLDER.search(citation):
            continue
        if (
            not citation.startswith(("fhir:", "imported:", "imaging:", "imaging-result:", "query:"))
            and "/" in citation
            and f"fhir:{citation}" not in sources
            and not re.search(r"\d", citation.split("/", 1)[1])
        ):
            continue
        citations = [citation]
        if citation.startswith("imported:") and re.search(r":seq-\d+$", citation):
            more = _MORE_SEQUENCES.match(output, match.end())
            if more:
                head = citation.rpartition(":")[0]
                citations.extend(f"{head}:seq-{n}" for n in re.findall(r"seq-(\d+)", more.group()))
        for cite in citations:
            if cite in seen:
                continue
            seen.add(cite)
            source_key = f"fhir:{cite}" if not cite.startswith(
                ("fhir:", "imported:", "imaging:", "imaging-result:", "query:")
            ) else cite
            if source_key in sources:
                selected.add(source_key)
            elif cite.startswith("imported:") and len(message_sources.get(cite, [])) == 1:
                selected.add(message_sources[cite][0])
            else:
                missing.append(cite)
    return {key: value for key, value in sources.items() if key in selected}, missing
