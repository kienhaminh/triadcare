"""The registry of tools and consultable agents.

A tool is declared once, next to its implementation, with the text the model reads about
it: what it does and how to call it. Its execute function is bound when an agent is
built, because it holds run state (cursors, stores, the session). An agent is declared
once here, with its prompt, its tools, and the text the Doctor reads about consulting
it. Prompts describe behaviour; they do not repeat what a tool description says.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt"


@dataclass
class Tool:
    """A tool as an agent offers it: the text the model reads and the function it calls."""

    description: str
    parameters: dict
    execute: Callable[[dict], Awaitable[Any]]

    def schema(self, name: str) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolSpec:
    description: str
    parameters: dict

    def bind(
        self,
        execute: Callable[[dict], Awaitable[Any]],
        *,
        parameters: dict | None = None,
        note: str = "",
    ) -> Tool:
        """The tool with its execute function. A variant may narrow the parameters or add
        a note to the description; it never replaces the description."""
        return Tool(self.description + note, parameters or self.parameters, execute)


TOOLS: dict[str, ToolSpec] = {}


def register_tool(name: str, description: str, parameters: dict) -> ToolSpec:
    if name in TOOLS:
        raise ValueError(f"Tool {name} is registered twice.")
    TOOLS[name] = spec = ToolSpec(description, parameters)
    return spec


def bind(name: str, execute: Callable[[dict], Awaitable[Any]], **variant: Any) -> Tool:
    return TOOLS[name].bind(execute, **variant)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    prompt: str
    tools: tuple[str, ...]
    # What the Doctor reads about consulting this agent.
    consult: str


def _prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


AGENTS = {
    "ehr": AgentSpec(
        "EHR Agent",
        _prompt("ehr"),
        (
            "read_conversation",
            "list_fhir_records",
            "query_fhir",
            "count_fhir_records",
            "read_fhir_resource",
        ),
        "EHR queries the configured FHIR server. It can count or list accessible patients "
        "without a patient ID, resolve a patient from a name or partial identifier, and "
        "query or count clinical records. Put the patient ID in scope, or the name in the "
        "question. Name the resource types and dates in scope; do not ask for "
        "the whole chart. The supplied chat is context, not evidence. The answer is prose "
        "that cites fhir: source IDs inline and names what EHR did not read. ",
    ),
    "radiologist": AgentSpec(
        "Radiologist Agent",
        _prompt("radiologist"),
        (
            "read_conversation",
            "list_imaging_studies",
            "read_stored_imaging_results",
            "analyze_mri",
            "analyze_pet",
            "analyze_dti",
        ),
        "Radiologist can list catalogued MRI/PET/DTI studies and run the configured analysis "
        "services. It receives processed measurements, not image pixels or a visual scan "
        "review. Put the patient ID and any requested study ID in scope. Name the "
        "measurement the question needs in return. The answer is prose that cites the "
        "study and result IDs inline and names what Radiologist did not read. ",
    ),
    "care": AgentSpec(
        "Care Agent",
        _prompt("care"),
        (
            "read_conversation",
            "count_conversation_records",
            "query_conversations",
            "search_conversations",
            "read_conversation_message",
            "list_check_ins",
        ),
        "Care reviews stored patient and caregiver check-ins. They are separate from the "
        "FHIR record and from the supplied chat. Put patient_id, the period, as_of, "
        "and the reporters in scope. Name the topics; do not ask for all check-ins. The answer "
        "is prose that cites message IDs inline and names the dates, reporters, or topics "
        "Care did not read. ",
    ),
}
