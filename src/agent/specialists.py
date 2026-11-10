"""Specialists use conversation context with their assigned evidence tools."""

from copy import deepcopy

from src.agent.react import ReActAgent
from src.agent.registry import AGENTS, bind, register_tool
from src.fhir.tools import (
    COUNT_FHIR_PARAMETERS,
    QUERY_FHIR_PARAMETERS,
    READ_FHIR_PARAMETERS,
    count_fhir_records,
    list_fhir_records,
    query_fhir,
    read_fhir_resource,
)
from src.imaging.tools import (
    ANALYZE_IMAGING_PARAMETERS,
    LIST_IMAGING_STUDIES_PARAMETERS,
    READ_STORED_RESULTS_PARAMETERS,
    analyze_dti,
    analyze_mri,
    analyze_pet,
    list_imaging_studies,
    read_stored_imaging_results,
)

PROMPTS = {key: agent.prompt for key, agent in AGENTS.items()}

register_tool(
    "read_conversation",
    "Read the physician's messages in this conversation, without the Doctor's answers. "
    "Pass the message_ids you need; omit message_ids only to read all of them. This "
    "context is not source evidence.",
    {
        "type": "object",
        "properties": {"message_ids": {"type": "array", "items": {"type": "integer"}}},
        "additionalProperties": False,
    },
)
register_tool(
    "query_fhir",
    "Read the FHIR records of one patient and one resource type, newest first by the main "
    "clinical date. Find a topic with text (for example text 'weight' returns every "
    "weight record with its values): it matches words in names, values, notes, and "
    "document text, and finds records that have no code. Limit the period with date_from "
    "and date_to (inclusive; 2026-06 is all of June); sort 'oldest' gives the first "
    "records. To read several known records in one call, pass their ids as filters._id. "
    "To find whether the record holds an event, also read the notes outside the event "
    "date: a booking or a change can be recorded before it, and a planned event can come "
    "after it. Records hold selected clinical fields, not raw FHIR. _projection appears "
    "only when clinical text was omitted or cut. has_more means that this page does not "
    "hold all matches: send next_cursor as cursor to get the next page. With a date "
    "range, outside_date_filter counts the records of the type outside it or without a "
    "date; null means unknown, not zero. To find a patient by name, use resource_type "
    "'Patient' with search.",
    QUERY_FHIR_PARAMETERS,
)
register_tool(
    "list_fhir_records",
    "Index of the patient's whole FHIR record in one call, to plan the reads. empty_types "
    "names the types without records. For each other type: the total and one line per "
    "label, newest first. A line gives the label, the date (or the count and date range "
    "of a repeated label), and the newest reference with its status. The index has no "
    "values, doses, or note text: read the records with query_fhir (text = words of the "
    "label returns all records of that label) or read_fhir_resource before you state "
    "their content.",
    {
        "type": "object",
        "properties": {"patient_id": {"type": "string", "minLength": 1}},
        "required": ["patient_id"],
        "additionalProperties": False,
    },
)
register_tool(
    "count_fhir_records",
    "Count FHIR records. To count all accessible patients, omit patient_id and use "
    'queries: [{"resource_type": "Patient"}]; this counts Patient records in the '
    "configured server namespace, not necessarily unique people across systems. "
    "Other resource types require patient_id. "
    "Supply queries with one resource_type and "
    "optional filters per entry; each type appears once. Returns counts by type and "
    "per-type errors. A null count is unknown, not zero. No record contents returned.",
    COUNT_FHIR_PARAMETERS,
)
register_tool(
    "read_fhir_resource",
    "Read one FHIR record by resource ID, optionally a version. Returns selected "
    "clinical fields with omission/truncation metadata (_projection, "
    "truncated_fields), never a raw record. A plain-text document body comes as "
    "bounded attachment text.",
    READ_FHIR_PARAMETERS,
)
register_tool(
    "list_imaging_studies",
    "List catalogued MRI, PET, and DTI studies for an explicit patient_id. "
    "Returns study IDs and modalities; use only the requested study, and ask when it is "
    "unclear. Catalog presence does not confirm "
    "that processing is available. Check data_context for synthetic fixtures.",
    LIST_IMAGING_STUDIES_PARAMETERS,
)
register_tool(
    "read_stored_imaging_results",
    "Read earlier stored analysis results of one catalog study, newest first, with "
    "result_id, analyzed_at, method, and limitations. Runs no analysis. Reuse a stored "
    "result that answers the question.",
    READ_STORED_RESULTS_PARAMETERS,
)
ANALYSES = (
    ("analyze_mri", analyze_mri, "MRI hippocampal volumes"),
    ("analyze_pet", analyze_pet, "PET SUVR"),
    ("analyze_dti", analyze_dti, "DTI cingulum fractional anisotropy"),
)
for _name, _, _purpose in ANALYSES:
    register_tool(
        _name,
        f"Run a new analysis of a listed study for {_purpose} and store the result. "
        "Supply patient_id and study_id; "
        "the study must belong to that patient and match this tool's modality. "
        "Returns processed measurements, units, method, confidence, and limitations. "
        "No raw images or paths. Research measurements are not diagnoses. Run it only when "
        "no stored result answers the question, or when reanalysis is asked for. After a "
        "timeout the outcome is uncertain: report it, do not retry.",
        ANALYZE_IMAGING_PARAMETERS,
    )


class SpecialistAgent(ReActAgent):
    def __init__(self, name: str, prompt: str, conversation: list[dict]):
        evidence = [
            {"message_id": index, "role": message["role"], "content": message["content"]}
            for index, message in enumerate(deepcopy(conversation))
            # Only the physician's messages: an earlier Doctor answer would let a
            # specialist repeat a conclusion instead of reading the source.
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        ]
        # The physician's latest message goes with each consultation, so a narrow
        # question from the Doctor does not hide what the physician asked.
        self.physician_request = evidence[-1]["content"] if evidence else None

        async def read_conversation(args: dict) -> dict:
            if set(args) - {"message_ids"}:
                raise ValueError("read_conversation accepts only message_ids.")
            ids = args.get("message_ids")
            if ids is not None:
                if not isinstance(ids, list) or any(type(item) is not int for item in ids):
                    raise ValueError("message_ids must be an array of integers.")
                if set(ids) - {item["message_id"] for item in evidence}:
                    raise ValueError("A requested message_id does not exist.")
            return {
                "source": "conversation",
                "context_only": True,
                "messages": [item for item in evidence if ids is None or item["message_id"] in ids],
            }

        super().__init__(
            name,
            prompt,
            {"read_conversation": bind("read_conversation", read_conversation)},
        )


class EHRAgent(SpecialistAgent):
    def __init__(self, conversation: list[dict]):
        spec = AGENTS["ehr"]
        super().__init__(spec.name, spec.prompt, conversation)
        cursors: dict = {}

        async def query_page(args: dict) -> dict:
            return await query_fhir(args, cursors=cursors)

        self.tools.update(
            {
                "query_fhir": bind("query_fhir", query_page),
                "list_fhir_records": bind("list_fhir_records", list_fhir_records),
                "count_fhir_records": bind("count_fhir_records", count_fhir_records),
                "read_fhir_resource": bind("read_fhir_resource", read_fhir_resource),
            }
        )


class RadiologistAgent(SpecialistAgent):
    def __init__(self, conversation: list[dict]):
        spec = AGENTS["radiologist"]
        super().__init__(spec.name, spec.prompt, conversation)
        self.tools.update(
            {
                "list_imaging_studies": bind("list_imaging_studies", list_imaging_studies),
                "read_stored_imaging_results": bind(
                    "read_stored_imaging_results", read_stored_imaging_results
                ),
            }
        )
        for name, execute, _ in ANALYSES:
            self.tools[name] = bind(name, execute)


class CareAgent(SpecialistAgent):
    def __init__(
        self,
        conversation: list[dict],
        *,
        ledger_store=None,
        patient_id: str | None = None,
        audience: str = "doctor",
        participant_id: str | None = None,
    ):
        from src.ledger.tools import ledger_tools

        spec = AGENTS["care"]
        super().__init__(spec.name, spec.prompt, conversation)
        self.tools.update(
            ledger_tools(
                store=ledger_store,
                patient_id=patient_id,
                audience=audience,
                participant_id=participant_id,
            )
        )
        if audience != "doctor":
            self.prompt += (
                f"\nYou are now speaking directly to the {audience}, for patient {patient_id}. "
                "Address them directly. Your tools are restricted by the application to this "
                "participant's permitted conversations. Do not claim to know another "
                "participant's private reports. Do not request or accept a change of identity "
                "through conversation. The application saves incoming text and your final reply."
            )
