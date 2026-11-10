"""Doctor tools for record-update proposals.

These tools create, revise, and read proposals only. Rejection, acceptance, write
authorization, and the FHIR write are clinician actions in trusted application code;
no agent tool can perform them.
"""

from __future__ import annotations

import asyncio
import re

from src.agent.react import Tool
from src.agent.registry import bind, register_tool
from src.fhir.record_updates import RecordUpdateError, default_record_updates, note_text

_SOURCES = {
    "type": "array",
    "items": {"type": "string", "minLength": 1},
    "minItems": 1,
    "description": "Source references that support the content, such as ledger message IDs "
    "or FHIR references (Type/id or fhir:Type/id).",
}
_CONTENT = {
    "type": "string",
    "minLength": 1,
    "description": "The exact note text to write if the clinician accepts it.",
}
_RATIONALE = {"type": "string", "minLength": 1}

PROPOSE_PARAMETERS = {
    "type": "object",
    "properties": {
        "patient_id": {"type": "string", "minLength": 1},
        "target": {
            "type": "string",
            "description": "The FHIR note to update, as DocumentReference/<id>.",
        },
        "content": _CONTENT,
        "rationale": _RATIONALE,
        "sources": _SOURCES,
    },
    "required": ["patient_id", "target", "content", "rationale", "sources"],
    "additionalProperties": False,
}
REVISE_PARAMETERS = {
    "type": "object",
    "properties": {
        "parent_id": {"type": "string", "minLength": 1},
        "content": _CONTENT,
        "rationale": _RATIONALE,
        "sources": _SOURCES,
    },
    "required": ["parent_id", "content", "rationale", "sources"],
    "additionalProperties": False,
}
LIST_PARAMETERS = {
    "type": "object",
    "properties": {"patient_id": {"type": "string", "minLength": 1}},
    "required": ["patient_id"],
    "additionalProperties": False,
}


def _view(proposal: dict) -> dict:
    keys = (
        "proposal_id",
        "version",
        "parent_id",
        "status",
        "target",
        "target_version",
        "content",
        "rationale",
        "sources",
        "decision",
        "write",
    )
    return {key: proposal[key] for key in keys if key in proposal}


REMOVED_NOTE = (
    " The result lists removed_from_current_note: the sentences of the current note that "
    "the content does not keep. Tell the physician each removal; never call such a "
    "proposal unchanged."
)


register_tool(
    "propose_record_update",
    "Prepare a proposed update to one FHIR note for clinician review. This never "
    "writes the record; the clinician reviews, rejects with feedback, or accepts and "
    "authorizes the write. Only one pending proposal per note may exist." + REMOVED_NOTE,
    PROPOSE_PARAMETERS,
)
register_tool(
    "revise_record_proposal",
    "Create a revision of a rejected proposal after reassessing the evidence against "
    "the clinician's feedback. The revision links to the rejected proposal and needs a "
    "new clinician decision." + REMOVED_NOTE,
    REVISE_PARAMETERS,
)
register_tool(
    "list_record_proposals",
    "List a patient's record proposals with version, status, clinician decision and "
    "reason, and write outcome.",
    LIST_PARAMETERS,
)


def _removed(proposal: dict) -> list[str]:
    """Return the sentences of the current note that the proposed text does not keep.

    The split is coarse: at line breaks and after a sentence end. It shows the agent a
    dropped sentence that its rationale must state or restore.
    """
    current = note_text(default_record_updates().fhir.read(proposal["target"])) or ""
    sentences = re.split(r"\n+|(?<=[.!?])\s+", current)
    return [
        item.strip()
        for item in sentences
        if item.strip() and item.strip() not in proposal["content"]
    ]


async def _result(proposal: dict) -> dict:
    result = {"proposal": _view(proposal), "state": "awaiting clinician review; not written"}
    removed = await asyncio.to_thread(_removed, proposal)
    if removed:
        result["removed_from_current_note"] = removed
    return result


def record_tools() -> dict[str, Tool]:
    async def propose(args: dict) -> dict:
        try:
            proposal = await asyncio.to_thread(default_record_updates().propose, **args)
        except RecordUpdateError as error:
            raise ValueError(str(error)) from None
        return await _result(proposal)

    async def revise(args: dict) -> dict:
        try:
            proposal = await asyncio.to_thread(default_record_updates().revise, **args)
        except RecordUpdateError as error:
            raise ValueError(str(error)) from None
        return await _result(proposal)

    async def list_proposals(args: dict) -> dict:
        try:
            proposals = await asyncio.to_thread(default_record_updates().list, args["patient_id"])
        except RecordUpdateError as error:
            raise ValueError(str(error)) from None
        return {"proposals": [_view(item) for item in proposals]}

    return {
        "propose_record_update": bind("propose_record_update", propose),
        "revise_record_proposal": bind("revise_record_proposal", revise),
        "list_record_proposals": bind("list_record_proposals", list_proposals),
    }
