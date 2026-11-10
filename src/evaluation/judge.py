"""Model-judge questions and strict validation of their JSON answers.

The judge never receives the generating model's identity. Every answer is validated;
an invalid or failed answer raises JudgeError, and the scorer marks the item unscored.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx

from src.agent.care_context import PLACEHOLDER
from src.agent.citations import _REFERENCE
from src.config.evaluation import judge_settings
from src.evaluation.judge_usage import record_usage
from src.fhir.arguments import FHIR_RESOURCE_TYPES

# (system, user) -> answer; an optional schema= constrains the answer JSON.
Ask = Callable[..., dict]

STATUSES = ("retained", "omitted", "incorrect")
LABELS = ("source_link", "attribution", "temporal", "uncertainty", "detail")
OPENROUTER_URL = "https://openrouter.ai/api/v1"
# Route only to the model's own provider, so OpenRouter uses the BYOK provider key, and
# only to endpoints that honor the strict response schema.
PROVIDER = {"only": ["anthropic"], "allow_fallbacks": False, "require_parameters": True}


class JudgeError(RuntimeError):
    """The judge call failed or returned an answer that does not meet its contract."""


@dataclass(frozen=True)
class Judge:
    model: str
    api_key: str
    reasoning_effort: str | None = None
    timeout: float = 600.0
    provider: dict = field(default_factory=lambda: PROVIDER)

    @classmethod
    def from_env(cls, *, require_key: bool = True) -> Judge:
        """The judge runs on OpenRouter with OPENROUTER_API_KEY."""
        settings = judge_settings()
        if require_key and not settings.api_key:
            raise JudgeError("Set OPENROUTER_API_KEY for the judge.")
        return cls(
            model=settings.model,
            api_key=settings.api_key,
            reasoning_effort=settings.reasoning_effort,
            # Pin Anthropic models to Anthropic; another judge model takes any provider
            # that supports the strict schema.
            provider=PROVIDER
            if settings.model.startswith("anthropic/")
            else {"require_parameters": True},
        )

    def identity(self) -> dict[str, Any]:
        """Scoring version: judge settings plus a hash of this module's prompts and rules."""
        return {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "judge_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def request_body(self, system: str, user: str, schema: dict | None = None) -> dict:
        """The chat/completions body of one question; the batch grading sends it too."""
        # json_object mode does not enforce value types; a strict schema does.
        response_format: dict[str, Any] = {"type": "json_object"}
        if schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": schema},
            }
        body: dict[str, Any] = {
            "model": self.model,
            # The votes of one question send the same prompt; the provider caches it.
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user, "cache_control": {"type": "ephemeral"}}
                    ],
                },
            ],
            "response_format": response_format,
            "provider": self.provider,
            "usage": {"include": True},
        }
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        return body

    def __call__(self, system: str, user: str, schema: dict | None = None) -> dict:
        try:
            response = httpx.post(
                f"{OPENROUTER_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=self.request_body(system, user, schema),
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            choice = result["choices"][0]
            record_usage(self.model, result.get("usage"), choice.get("finish_reason"))
            content = choice["message"]["content"]
            answer = json.loads(content)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            raise JudgeError(f"Judge call failed: {type(error).__name__}: {error}") from error
        if not isinstance(answer, dict):
            raise JudgeError("Judge answer is not a JSON object.")
        return answer


def question_key(
    identity: dict[str, Any], system: str, user: str, schema: dict | None = None
) -> str:
    """Cache key: a cached answer is reused only for the same judge settings, the same
    question, and the same answer schema. The code hash stays out of the key: an edit of
    this module invalidates only the questions it changes."""
    judge = {key: value for key, value in identity.items() if key != "judge_code_sha256"}
    scope = json.dumps([judge, schema], sort_keys=True)
    return hashlib.sha256(f"{scope}\n\x00\n{system}\n\x00\n{user}".encode()).hexdigest()


def _imported_messages(sources: dict[str, str]) -> dict[str, str]:
    """Read exact message IDs from the available check-in source blocks."""
    messages = {}
    for source_id, text in sources.items():
        if not source_id.startswith("raw-check-in:"):
            continue
        prefix = source_id.replace("raw-check-in:", "imported:", 1) + ":seq-"
        for line in text.splitlines():
            message_id, separator, words = line.partition(": ")
            if separator and message_id.startswith(prefix) and message_id[len(prefix) :].isdigit():
                messages[message_id] = words
    return messages


def _sources_block(sources: dict[str, str], required: list[str] | None = None) -> str:
    marked = set(required or ())
    blocks = []
    for source_id, text in sources.items():
        if source_id.startswith("raw-check-in:"):
            messages = _imported_messages({source_id: text})
            text = "\n".join(
                f"[{message_id}] {words}" if message_id in messages else line
                for line in text.splitlines()
                for message_id, _, words in [line.partition(": ")]
            )
        blocks.append(f"[{source_id}]{' (required)' if source_id in marked else ''}\n{text}")
    return "\n\n".join(blocks) or "(none)"


def _citation_check(output: str, sources: dict[str, str]) -> str:
    """Render the code-resolved message and imaging IDs for the judge."""
    return json.dumps(source_citation_resolution(output, sources))


def imported_citation_resolution(output: str, sources: dict[str, str]) -> dict[str, list[str]]:
    """Resolve imported citation IDs by exact string equality before model scoring."""
    cited = {item.rstrip(".:") for item in re.findall(r"\bimported:[A-Za-z0-9_.:-]+", output)}
    available = _imported_messages(sources)
    return {
        "available": sorted(cited & available.keys()),
        "unavailable": sorted(cited - available.keys()),
    }


def source_citation_resolution(output: str, sources: dict[str, str]) -> dict[str, list[str]]:
    """Resolve message and imaging source identities independently of claim support."""
    resolved = imported_citation_resolution(output, sources)
    cited = {
        item.rstrip(".:") for item in re.findall(r"\bimaging(?:-result)?:[A-Za-z0-9_.:-]+", output)
    }
    available = set(sources)
    return {
        "available": sorted(set(resolved["available"]) | (cited & available)),
        "unavailable": sorted(set(resolved["unavailable"]) | (cited - available)),
    }


def unit_output_candidates(
    output: str, units: list[dict], sources: dict[str, str]
) -> dict[str, list[int]]:
    """Find output lines linked to unit anchors without deciding semantic presence."""
    lines = output.splitlines()
    cited = [
        {token.rstrip(".:") for token in re.findall(
            r"\b(?:fhir:|imported:|imaging:|imaging-result:)[A-Za-z0-9_.:/-]+", line
        )}
        for line in lines
    ]
    messages = _imported_messages(sources)
    candidates = {}
    for unit in units:
        ids, quotes = set(), []
        for anchor in unit.get("anchors", []):
            source_id, quote = anchor["source_id"], anchor.get("quote", "")
            if quote:
                quotes.append(quote)
            if source_id.startswith("raw-check-in:"):
                prefix = source_id.replace("raw-check-in:", "imported:", 1) + ":seq-"
                ids.update(
                    message_id for message_id, text in messages.items()
                    if message_id.startswith(prefix) and (not quote or quote in text)
                )
            else:
                ids.add(source_id)
        candidates[unit["id"]] = [
            number for number, (line, tokens) in enumerate(zip(lines, cited), 1)
            if ids & tokens or any(quote in line for quote in quotes)
        ]
    return candidates


def _unit_view(unit: dict) -> dict:
    keys = (
        "id",
        "content",
        "reporter",
        "subject",
        "event_time",
        "reported_at",
        "uncertainty",
        "correction_of",
        "allowed_transformations",
    )
    view = {key: unit[key] for key in keys if key in unit}
    view["anchors"] = [
        {"source_id": anchor["source_id"], "quote": anchor.get("quote", "")}
        for anchor in unit.get("anchors", [])
    ]
    return view


def _coverage_view(unit: dict) -> dict:
    """The unit with the scenario_v4 fields that decide its coverage."""
    keys = ("priority", "must_state", "essential_limit")
    return {**_unit_view(unit), **{key: unit[key] for key in keys if key in unit}}


def coverage_score(units: list[dict], findings: dict[str, dict]) -> int | None:
    """Coverage from the findings of scenario_v4 units; None for units without priority.

    2: every essential unit is present and no unit is incorrect. 0: an essential unit is
    incorrect. 1: otherwise. An absent supporting unit does not lower the score. An answer
    that misses the task fails under task_completion, not here.
    """
    essential = [unit["id"] for unit in units if unit.get("priority") == "essential"]
    if not essential:
        return None
    status = {unit_id: findings[unit_id]["status"] for unit_id in findings}
    if any(status[unit_id] == "incorrect" for unit_id in essential):
        return 0
    if any(status[unit_id] == "absent" for unit_id in essential) or "incorrect" in status.values():
        return 1
    return 2


# ---------------------------------------------------------------------------
# Information-unit preservation and error labels
# ---------------------------------------------------------------------------

UNITS_SYSTEM = """You judge information preservation for a clinical documentation study.
You receive required information units, the full sources that the units anchor, and one
text. Classify each unit exactly once, by what a reader of the text learns about it.
Each unit lists must_state: the short facts that any statement of the unit must carry.
The unit's core content is its must_state facts with its reporter and subject. When
essential_limit is true, the unit's uncertainty is also core. The other content of the
unit is secondary, and an event date or a report date is core only when a must_state
fact names it. Within a must_state item, only the main fact is core; a secondary detail in
it (a duration, a date, a dose history, a reason) is secondary.
- retained: the text states the core content and states no core fact more strongly than
  the unit. Paraphrase and summary within the unit's allowed_transformations count. A
  missing or wrong secondary detail leaves the unit retained; the labels record it.
- omitted: the text does not state the unit, or states only part of its core content.
- incorrect: the text states the unit but contradicts a core fact, assigns the unit to
  another reporter or subject, reverses its correction relation, or states a core fact
  more strongly than the unit: an unknown, unseen, or unremembered matter as absent or
  negative; one episode or one reporter's observation as a whole period or as both
  reporters' account; a report, a plan, or a listing as a confirmed fact; an open
  question as resolved.
A missing part makes a unit retained or omitted; incorrect needs a wrong or stronger
statement of core content. A left-out limit is not a stronger statement unless the text
states its opposite. A statement under a reporter's heading, or cited to that reporter's
message, is that reporter's account, not a confirmed fact. For incorrect, the evidence quotes the statement of the text
that is wrong or stronger and names the unit fact that it contradicts. A wrong or missing
source link alone leaves the status as it is.
An explicit requirement in allowed_transformations overrides these allowances: when it
requires a qualifier, a text without it states only part of the unit. A must_state fact
is a meaning, not a wording. Accept equivalent wording ("daily" for "once daily").
Judge as a physician who reads the text: a statement that tells the physician the same
fact is retained, even when it is shorter, less exact (an approximate time or amount), or
uses other words.
Read the whole text before you classify a unit. A core fact can be stated across several
sentences, lines, or sections, including the attached records; combine them: for
example, a place named in one line and an event named in the next line. A fact is
stated when the text's statements, read together, directly entail it. Do not accept a
fact that the reader must guess.
A person's name and their reporter role are the same reporter ("Arthur" for the
caregiver). A separately supported account of another reporter does not contradict the
required account; when the required account is absent, the unit is omitted.
Judge the text against the unit and its full sources. A statement that the sources
support is not wrong or stronger than the unit, even when the unit's quotes do not show
it. An absent phenomenon does not meet a required positive finding. A text can report a
different statement of the same source, about another time or event; classify the unit
by whether the text also states the unit's own content.
A text can have two parts: the agent's own text, and records the system attached as
recorded (for example under "Evidence" or "Cited observations"). The reader receives
both, so a unit stated in either part is stated. When the agent's own text misstates a
unit that an attached record states correctly, classify the misstatement: the reader
receives it.
Return only JSON."""

LABELS_RULES = """For each unit the text states (retained or incorrect), also set five
labels; they may overlap. Every label of an omitted unit is false.
- source_link: the text gives no source link for the unit, or links it to the wrong source.
- attribution: the text assigns the unit to the wrong reporter or source.
- temporal: the text states a wrong event date or report date, or merges them.
- uncertainty: the text states the unit with more certainty than the unit or its sources.
- detail: the text leaves out a secondary qualifier of the unit and states nothing
  stronger than the unit.
The attribution label goes with the status incorrect; the detail label goes with the
status retained. The temporal and uncertainty labels go with incorrect when they concern
core content, and with retained otherwise."""


def units_question(
    stage: str, text: str, units: list[dict], sources: dict[str, str], *, labels: bool
) -> tuple[str, str]:
    schema = {"units": [{"unit_id": "string", "status": "|".join(STATUSES), "evidence": "string"}]}
    if labels:
        schema["units"][0]["labels"] = {label: "boolean" for label in LABELS}
    system = UNITS_SYSTEM + ("\n\n" + LABELS_RULES if labels else "")
    user = (
        f"Comparison: {stage}\n\n"
        f"Required units:\n{json.dumps([_coverage_view(u) for u in units], indent=1, ensure_ascii=False)}\n\n"
        f"Sources of the units:\n{_sources_block(sources)}\n\n"
        f"Text under assessment:\n<<<\n{text}\n>>>\n\n"
        f"Answer with this JSON shape:\n{json.dumps(schema)}"
    )
    return system, user


def parse_units(answer: dict, units: list[dict], *, labels: bool) -> dict[str, dict]:
    expected = [unit["id"] for unit in units]
    items = answer.get("units")
    if not isinstance(items, list):
        raise JudgeError("Answer has no units list.")
    result: dict[str, dict] = {}
    for item in items:
        unit_id = item.get("unit_id") if isinstance(item, dict) else None
        if unit_id not in expected or unit_id in result:
            raise JudgeError(f"Unexpected or repeated unit {unit_id!r}.")
        status = item.get("status")
        if status not in STATUSES:
            raise JudgeError(f"Invalid status {status!r} for {unit_id}.")
        entry = {"status": status, "evidence": str(item.get("evidence", ""))}
        if status == "incorrect" and not entry["evidence"].strip():
            raise JudgeError(f"Incorrect unit {unit_id} has no evidence.")
        if labels:
            raw = item.get("labels")
            if not isinstance(raw, dict) or any(not isinstance(raw.get(k), bool) for k in LABELS):
                raise JudgeError(f"Invalid labels for {unit_id}.")
            entry["labels"] = {label: raw[label] for label in LABELS}
            if status == "omitted" and any(entry["labels"].values()):
                raise JudgeError(f"Omitted unit {unit_id} has error labels.")
            if entry["labels"]["attribution"] and status != "incorrect":
                raise JudgeError(f"Unit {unit_id} has the attribution label but is {status}.")
            if entry["labels"]["detail"] and status != "retained":
                raise JudgeError(f"Unit {unit_id} has the detail label but is {status}.")
        result[unit_id] = entry
    if set(result) != set(expected):
        raise JudgeError(f"Answer misses units: {sorted(set(expected) - set(result))}.")
    return result


# ---------------------------------------------------------------------------
# Output quality rubric
# ---------------------------------------------------------------------------

# Days of the rubric calendar before and after the task date.
CALENDAR_DAYS_BEFORE = 42
CALENDAR_DAYS_AFTER = 7


def calendar_text(as_of: str) -> str:
    """List the dates and weekdays around the task date. The judge uses the list to check
    the conversion of a relative day to an absolute date."""
    today = date.fromisoformat(as_of[:10])
    days = range(-CALENDAR_DAYS_BEFORE, CALENDAR_DAYS_AFTER + 1)
    return "\n".join(f"{today + timedelta(n):%Y-%m-%d %A}" for n in days)


def rubric_question(
    rubric: dict,
    request: str,
    output: str,
    sources: dict[str, str],
    *,
    required: list[str],
    required_units: list[dict] | None = None,
    previous: str | None = None,
    feedback: str | None = None,
    runtime_errors: list[str] | None = None,
    coverage_by_code: bool = False,
    calendar: str | None = None,
    cited_only: bool = False,
    citation_resolution: dict[str, list[str]] | None = None,
) -> tuple[str, str]:
    # Code computes the coverage of a final answer from the separate unit votes. Then the
    # rubric judge gets no units, and it must not score missing facts in other dimensions.
    coverage_rule = (
        "\n- Code scores coverage from separate unit judgments. Return coverage 2. Do not "
        "lower task_completion or another dimension because a fact is missing; lower "
        "task_completion only for a requested part that the output does not answer."
        if coverage_by_code
        else ""
    )
    system = (
        "You score one clinical task output with a fixed rubric. Return only JSON.\n\n"
        f"Scale: {json.dumps(rubric['scale'])}\n"
        f"Dimensions and anchors: {json.dumps(rubric['dimension_anchors'])}\n"
        f"Critical errors (use these exact strings): {json.dumps(rubric['critical_errors'])}\n"
        f"Rules: {json.dumps(rubric['judge_rules'])}\n"
        "Sources:\n"
        "- The reference sources are every source available at the review date. Assess source "
        "support against all of them, and coverage against the sources marked (required).\n"
        "- A raw-check-in:<scenario>:day-N:<reporter> block names a runtime session. Each "
        "imported:<scenario>:day-N:<reporter>:seq-N prefix inside that block identifies the "
        "primary source message on that line. A citation to that exact listed imported ID "
        "links to its message, recorded date, and reporter. Accept it as a valid source, even "
        "when the enclosing block is labeled raw-check-in. Only imported IDs listed in the "
        "available source blocks are valid; an unlisted or unknown ID is unavailable. A "
        "conversation:N token is context-only and is never a primary source ID.\n"
        "- A Care Agent question, hypothesis, or safety recommendation in that session "
        "is not a patient finding or a completed clinical action. Preserve the speaker. "
        "Do not turn advice to seek assessment into evidence that assessment occurred. "
        "A patient or caregiver account of follow-through remains a reported account.\n"
        "- The source citation check resolves imported and imaging IDs against the available "
        "source blocks. IDs in its available list exist. Do not mark those IDs as "
        "unavailable or malformed. Still assess whether their message supports the claim, "
        "reporter, and date. IDs in its unavailable list are not available sources.\n"
        "- Return source_findings for every source-support penalty and every proposition "
        "counted as unsupported. Each finding has kind, citation, proposition, and "
        "material_effect. Use unavailable_source only for a citation absent from the "
        "available sources; use unsupported_claim when a cited source exists but does not "
        "support the proposition; use wrong_attribution when a valid source is assigned "
        'to the wrong reporter or source. For a missing citation, set citation to "none". '
        "Use one exact source ID per citation field, without brackets or joined IDs. "
        "Select the relevant ID from the output. A valid source ID that "
        "does not support the claim is not unavailable. Keep one finding per affected "
        "proposition or fabricated citation. Use an empty array when there is no source "
        "support penalty.\n"
        "- The fhir: sources are the complete hospital record of the patient. A claim that the "
        "record holds no resource of a type is supported when no fhir: source of that type "
        "exists, for example no MedicationRequest. A claim that the patient does not have a "
        "condition, allergy, or event needs a source that states it.\n"
        f"- The record tools read only these FHIR resource types: {', '.join(FHIR_RESOURCE_TYPES)}. "
        "A statement that the review did not read or could not query another type is "
        "supported by this tool scope.\n"
        "- Preserve the scope of source descriptions. FHIR metadata with no inference "
        "result can coexist with a separately stored runtime analysis. A statement about "
        "the FHIR record lacking results does not deny the runtime result, especially when "
        "the output separately reports that result. Read section headings, citations, and "
        "the whole answer before broadening a source-scoped statement. FHIR "
        "ImagingStudy.started is the acquisition date of the study; an output can state it "
        "as the date of the study.\n"
        "- An imaging:<study> source contains the patient's complete frozen catalog view "
        "and ordered runtime events through the current task. It supports study membership, "
        "modality, data_context, and catalog-scoped absence claims. Missing catalog dates "
        "mean this catalog provides no acquisition date. FHIR ImagingStudy.started is a "
        "separate record field: it does not contradict a claim specifically about missing "
        "catalog dates. A broader claim that no source supplies dates must also consider FHIR.\n"
        "- An imaging-result:<id> source contains a model-generated analysis and its "
        "availability events. A successful analysis stores its result immediately. A later "
        "read reuses that stored result and does not run a new analysis. current_task marks "
        "which events occurred in the task being scored; earlier events are history. An "
        "empty read is a snapshot at that event, not proof that later analyses do not exist. "
        "A read lists at most five results, so it is not a complete lifetime count.\n"
        "- analyzed_at is the wall-clock time of the evaluation's analysis execution. It is "
        "not the study acquisition date or a clinical event date. These runtime analyses "
        "are allowed evidence in this evaluation even when executed after the scenario "
        "review date. Penalize analyzed_at only when the output presents it as the date of "
        "a clinical event, such as the study acquisition. Do not penalize another use of "
        "a runtime result, its events, or its execution time.\n"
        "- A query:query_fhir/<type> source lists the record searches of this task that "
        "returned no records, with their filters. It supports a statement that a search of "
        "that type, filters, and dates found nothing; it does not show that the patient "
        "has no such condition or event.\n"
        "- Runtime tool errors are system events of this task, not patient evidence. A claim "
        "that reports one as a system limitation is supported; a patient fact inferred from "
        "one is unsupported.\n"
        "Output:\n"
        "- The output can end with records the system attached as recorded (for example under "
        '"Evidence"). They are part of the output: assess them for source support and '
        "coverage. When the agent's own text misstates a record, score the misstatement.\n"
        "- claims_assessed counts factual claims of the agent's own text, before the "
        "attached records; claims_unsupported counts factual propositions those sources do "
        "not support. Count propositions, not citation tokens. Assess source identity, "
        "provenance, and attribution separately: an unavailable citation can itself assert "
        "a fabricated source, even when another source supports the proposition.\n"
        "- Correct general clinical knowledge needs no patient source when the output "
        "presents it as general knowledge and not as a finding about this patient: for "
        "example a known effect of a drug class or the purpose of a standard test. Do not "
        "count it as unsupported. Every claim about this patient still needs a source.\n"
        "- Treat a conversation:N token as a redundant internal alias only when a valid "
        "primary citation also supports the same proposition and the alias does not assign "
        "it to a different reporter, source, or author. In this narrow case, the proposition "
        "is supported, and the alias is at most a noncritical citation imprecision.\n"
        "- A different valid citation does not excuse a fabricated material source or a "
        "materially wrong source attribution, including a citation-only fabrication. Select "
        "the critical error 'Invented or contradicted material event, dose, date or source' "
        "when that false source assertion or attribution meets the rubric's materiality "
        "rule. Keep material unsupported or contradicted propositions and the other "
        "critical errors.\n"
        "Scoring:\n"
        "- Score as a physician who reads the output to act on it. The standard is an output "
        "that is correct and gives enough information for the request, not a perfect one. "
        "Lower a dimension only for a defect that would mislead the physician or leave them "
        "without information the request needs. A defect that does not change what the "
        "physician understands or does is not a defect: score 2.\n"
        "- A defect is material when it changes a task-essential account, a clinical "
        "interpretation, or a supported action. A harmless difference of terms (for example "
        '"clinical note" for a cited document) is not material.\n'
        "- In source_support, attribution, temporal_fidelity, and uncertainty, a defect is "
        "only a hallucination or an overclaim: a statement that no source supports, that a "
        "source contradicts, that names the wrong person or source, that puts an event on a "
        "wrong date or in a wrong order, or that is more certain than its sources. Before you "
        "lower one of these scores, find the source text that the statement conflicts with or "
        "lacks, read the Care question before a short reply, and quote both. These are not "
        "defects: a missing fact or a missing qualifier, when the output does not state the "
        "opposite; a reasonable inference that the output presents as its reading of the "
        "cited sources; the output's own advice or next step; a report date that the output "
        "gives as the date of the report and not of the event; a statement that a search or "
        "a tool view found nothing, when that matches the tool output; a citation label or "
        "format that is imprecise, when an available source supports the statement; wording "
        "that a fair reading of the sources allows. When you are unsure, score 2.\n"
        "- source_support: a citation to a nearby message of the same session (the same day "
        "and reporter) supports a proposition when another listed message of that session "
        "states it. Do not count it as unsupported and do not lower the score for it. Lower "
        "source_support only for a proposition that no available source supports, or that a "
        "source contradicts.\n"
        "- uncertainty: lower it only when the output states a matter materially more "
        "certainly than its sources: an unknown as known, a report as confirmed, a plan as "
        "done, an open question as resolved. Paraphrase of a hedge ('might be' for 'I think') "
        "or of a minor qualifier is not a defect.\n"
        "- attribution: lower it only when the output gives an account to the wrong person "
        "or the wrong kind of source. A group term that includes the reporter ('the family' "
        "for the caregiver) is not a defect when no other reporter disagrees. Radiologist "
        "Agent is the system's imaging specialist: calling its analysis the radiologist's "
        "or the imaging review's result is correct attribution. Only a claim of a formal "
        "clinical radiology report that no source shows is a wrong source.\n"
        "- task_completion: lower it only for a part of the request that the output does not "
        "answer, or answers wrongly. Do not lower it for context, background, or a next step "
        "that the request did not ask for, even when it would be useful.\n"
        "- For each score below 2 and each critical error, quote the offending or missing "
        "proposition, name its source, and state its material effect. When the rationale "
        "names no defect, the score is 2. Select a critical error only when its stated "
        "condition is met.\n"
        "- Score each defect once, under the one dimension that fits it best: a claim that "
        "its sources do not support under source_support, a wrong reporter or source under "
        "attribution, a wrong date or sequence under temporal_fidelity, a statement more "
        "certain than its sources under uncertainty, and an unanswered part of the request "
        "under task_completion. Do not lower another dimension for the same defect.\n"
        "- A relative day or date (a weekday, yesterday, last week) stated as an absolute "
        "date is supported when the date arithmetic from the report date is correct. Use the "
        "supplied calendar to check weekdays.\n"
        "- Before you call a proposition omitted, check the agent's text and the attached "
        "records. A required source does not make each of its sentences required; assess a "
        "missing comparison or synthesis under task_completion.\n"
        "- When required information units are supplied, return one coverage finding for "
        "every unit. Judge the full output, including its Evidence section. Use present "
        "when the output correctly tells the physician the main point of the unit, in any "
        "words. Use absent when it does not give that main point, and incorrect when it "
        "states the unit with wrong or stronger content. A left-out limit or detail is not "
        "incorrect unless the output states its opposite. Before you mark a unit incorrect, "
        "quote the output statement and the source text that it contradicts or overstates. "
        "For present and incorrect findings, select evidence_lines from the numbered output. "
        "Code extracts their exact text. Select the lines that state the finding and its "
        "reporter or date context. Source-block lines are not output lines. For absent, "
        "use an empty evidence_lines array and explain what is missing. A source quotation "
        "in Evidence does not repair conflicting prose: mark the unit incorrect and select "
        "the conflicting output lines. For incorrect, quote in the explanation the source "
        "statement that the selected lines contradict. A supported account of another "
        "person is not a contradiction. Read the whole output and combine statements in "
        "different lines before you call a unit absent. Judge synthesis under "
        "task_completion, apart "
        "from whether each fact appears in coverage. An incorrect date or attribution can "
        "coexist with complete proposition coverage; score each dimension independently. "
        "Do not require every source sentence.\n"
        "- When a unit has must_state, its main point is the main fact of each must_state "
        "item. A secondary detail in a must_state item (a duration, a date, a dose history, "
        "a reason) is optional: its absence does not make the unit absent. A secondary "
        "detail that the output states wrongly makes the unit incorrect. The other content "
        "of the unit is also optional detail. When essential_limit is true, the unit's "
        "uncertainty is part of the main point. Code computes the coverage score from these findings and the unit "
        "priorities.\n"
        "- Score task_completion on the parts of the request: each question answered, and "
        "each requested comparison, synthesis, verdict, or change made. Coverage already "
        "scores each required unit, so an absent or incorrect unit does not lower "
        "task_completion. Lower task_completion only for a part of the request that the "
        "output does not answer or answers in a way the sources do not support.\n"
        "- A supported event day stays supported when its clock time, onset, or duration is "
        "unknown.\n"
        "- A statement about one tool view is assessed in that view: fhir: dates do not show "
        "that an imaging catalog lists dates. A runtime error shows that one attempt failed, "
        "not that its retries or other queries failed. Older or undated resources do not "
        "contradict an empty search of a date range."
        f"{coverage_rule}"
    )
    if cited_only:
        system = (
            "Score the clinical task output with the supplied rubric. Return only JSON.\n"
            f"Scale: {json.dumps(rubric['scale'])}\n"
            f"Dimensions and anchors: {json.dumps(rubric['dimension_anchors'])}\n"
            f"Critical errors (use exact strings): {json.dumps(rubric['critical_errors'])}\n"
            f"Rules: {json.dumps(rubric['judge_rules'])}\n"
            "The reference sources contain only sources cited in the output. They are not "
            "the complete patient record. Check each cited proposition against its own "
            "cited source, including the full cited check-in session. Check reporter, time, "
            "uncertainty, and source identity. Do not infer that a fact or source does not "
            "exist elsewhere. An uncited patient claim is unverified, not proven false. "
            "An unavailable citation is not proof that its claim is false. The source "
            "citation check is computed against the full source catalog.\n"
            "For each source-support penalty, return source_findings with kind, citation, "
            "proposition, and material_effect. Use unavailable_source for a citation "
            "marked unavailable; use unsupported_claim when its cited source does not "
            "support the claim; use wrong_attribution for the wrong reporter or source. "
            "Use one exact ID per finding. Provide a rationale for each score below 2. "
            "Assess attached output evidence as part of the output. Do not treat runtime "
            "tool errors as patient facts. Follow the supplied calendar for relative dates.\n"
            "When required units are supplied, return one coverage finding per unit. "
            "Use present, absent, or incorrect; select evidence_lines from the numbered "
            "output for present or incorrect, and explain each finding. "
            "Do not infer a missing fact is false.\n"
            f"{coverage_rule}"
        )
    revision = ""
    if previous is not None:
        revision = (
            f"\n\nPrevious version:\n<<<\n{previous}\n>>>\n\n"
            f"Clinician feedback on the previous version:\n<<<\n{feedback}\n>>>"
        )
    schema = {
        "scores": {dimension: "integer 0, 1, or 2" for dimension in rubric["dimensions"]},
        "critical_errors": ["exact string from the list"],
        "claims_assessed": "integer",
        "claims_unsupported": "integer",
        "source_findings": [
            {
                "kind": "unavailable_source | unsupported_claim | wrong_attribution",
                "citation": "exact cited source ID, or none",
                "proposition": "affected factual proposition",
                "material_effect": "effect on the task or patient record",
            }
        ],
        "rationale": {dimension: "string" for dimension in rubric["dimensions"]},
    }
    if required_units:
        schema["coverage_findings"] = [
            {
                "unit_id": "exact required unit ID",
                "status": "present | absent | incorrect",
                "evidence_lines": ["integer output line number; empty array when absent"],
                "explanation": "brief reason for the status",
            }
        ]
    units = (
        ""
        if not required_units
        else "Required information units for coverage:\n"
        f"{json.dumps([_coverage_view(unit) for unit in required_units], indent=1, ensure_ascii=False)}\n\n"
    )
    assessed_output = (
        "\n".join(f"L{number}: {line}" for number, line in enumerate(output.splitlines(), 1))
        if required_units
        else output
    )
    navigation = ""
    if required_units:
        navigation = (
            "\n\nCandidate output lines by required unit (computed from source IDs or "
            "exact anchor text):\n"
            f"{json.dumps(unit_output_candidates(output, required_units, sources))}\n"
            "These are navigation candidates, not presence decisions. Read these lines "
            "before calling a unit absent. A citation can concern another proposition; "
            "an uncited paraphrase can occur elsewhere. Check the full output and required "
            "qualifiers. Do not infer correctness from a citation.\n"
        )
    calendar_block = f"\n\nCalendar:\n{calendar}" if calendar else ""
    user = (
        f"Request:\n<<<\n{request}\n>>>\n\n"
        f"Reference sources:\n{_sources_block(sources, required)}\n\n"
        f"Source citation check (exact IDs, computed by code):\n"
        f"{json.dumps(citation_resolution) if citation_resolution is not None else _citation_check(output, sources)}\n\n"
        f"{units}"
        f"Runtime tool errors:\n{chr(10).join(runtime_errors or []) or '(none)'}"
        f"{calendar_block}"
        f"{revision}\n\nOutput under assessment (response-1):\n<<<\n{assessed_output}\n>>>\n\n"
        f"{navigation}"
        f"Answer with this JSON shape:\n{json.dumps(schema)}"
    )
    return system, user


def rubric_schema(rubric: dict, required_units: list[dict] | None = None) -> dict:
    """Strict answer schema of a rubric question: integer scores and listed critical errors."""

    def closed(properties: dict) -> dict:
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    dimensions = rubric["dimensions"]
    properties = {
        "scores": closed({d: {"type": "integer", "enum": [0, 1, 2]} for d in dimensions}),
        "critical_errors": {
            "type": "array",
            "items": {"type": "string", "enum": list(rubric["critical_errors"])},
        },
        "claims_assessed": {"type": "integer"},
        "claims_unsupported": {"type": "integer"},
        "source_findings": {
            "type": "array",
            "items": closed(
                {
                    "kind": {
                        "type": "string",
                        "enum": [
                            "unavailable_source",
                            "unsupported_claim",
                            "wrong_attribution",
                        ],
                    },
                    "citation": {"type": "string"},
                    "proposition": {"type": "string"},
                    "material_effect": {"type": "string"},
                }
            ),
        },
        "rationale": closed({d: {"type": "string"} for d in dimensions}),
    }
    if required_units:
        properties["coverage_findings"] = {
            "type": "array",
            "items": closed(
                {
                    "unit_id": {
                        "type": "string",
                        "enum": [unit["id"] for unit in required_units],
                    },
                    "status": {"type": "string", "enum": ["present", "absent", "incorrect"]},
                    "evidence_lines": {"type": "array", "items": {"type": "integer"}},
                    "explanation": {"type": "string"},
                }
            ),
        }
    return closed(properties)


def _normalized_quote(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_citations(findings: list) -> list:
    """Keep one source finding for each cited ID. The judge sometimes joins several IDs
    in one citation field; the checks then apply to each ID."""
    result = []
    for finding in findings:
        citation = finding.get("citation") if isinstance(finding, dict) else None
        parts = [p for p in re.split(r"\s*[;,]\s*", citation) if p] if isinstance(citation, str) else []
        if len(parts) > 1:
            result.extend({**finding, "citation": part} for part in parts)
        else:
            result.append(finding)
    return result


def parse_rubric(
    answer: dict,
    rubric: dict,
    *,
    source_citations: dict[str, list[str]],
    required_units: list[dict] | None = None,
    output: str | None = None,
) -> dict:
    scores = answer.get("scores")
    if not isinstance(scores, dict):
        raise JudgeError("Answer has no scores.")
    parsed = {}
    for dimension in rubric["dimensions"]:
        value = scores.get(dimension)
        if type(value) is not int or value not in (0, 1, 2):
            raise JudgeError(f"Invalid score for {dimension}.")
        parsed[dimension] = value
    critical = answer.get("critical_errors")
    if not isinstance(critical, list) or any(
        item not in rubric["critical_errors"] for item in critical
    ):
        raise JudgeError("Critical errors must come from the fixed list.")
    assessed, unsupported = answer.get("claims_assessed"), answer.get("claims_unsupported")
    if (
        type(assessed) is not int
        or type(unsupported) is not int
        or not 0 <= unsupported <= assessed
    ):
        raise JudgeError("Claim counts are invalid.")
    findings = answer.get("source_findings")
    if not isinstance(findings, list):
        raise JudgeError("Source findings must be a list.")
    findings = _split_citations(findings)
    finding_kinds = {"unavailable_source", "unsupported_claim", "wrong_attribution"}
    for finding in findings:
        if not isinstance(finding, dict) or set(finding) != {
            "kind",
            "citation",
            "proposition",
            "material_effect",
        }:
            raise JudgeError("A source finding has an invalid shape.")
        if (
            not isinstance(finding["kind"], str)
            or finding["kind"] not in finding_kinds
            or any(
                not isinstance(finding[field], str) or not finding[field].strip()
                for field in ("citation", "proposition", "material_effect")
            )
        ):
            raise JudgeError("A source finding has invalid values.")
        citation = finding["citation"]
        checked_id = re.search(r"\b(?:fhir|imported|imaging(?:-result)?|query):", citation)
        if (
            checked_id
            and re.fullmatch(
                r"(?:fhir|imported|imaging(?:-result)?|query):[A-Za-z0-9_.:/-]+", citation
            ) is None
        ):
            raise JudgeError("A citation must contain one exact source ID.")
        if (
            finding["kind"] == "unavailable_source"
            and checked_id
            and citation not in source_citations.get("unavailable", [])
        ):
            raise JudgeError(
                "An available or uncited message or imaging source was labeled unavailable."
            )
    unsupported_propositions = {
        _normalized_quote(finding["proposition"]).casefold()
        for finding in findings
        if finding["kind"] in {"unsupported_claim", "wrong_attribution"}
    }
    # The count is the distinct propositions the findings name, so each counted claim has
    # evidence; the judge's own count only bounds claims_assessed.
    unsupported = len(unsupported_propositions)
    assessed = max(assessed, unsupported)
    source_critical_error = any(
        any(term in item.lower() for term in ("source", "unsupported", "unknown"))
        for item in critical
    )
    # An attribution score below 2 can come from an omitted reporter account, which has no
    # cited source to name, so only source_support needs a structured finding.
    source_support_penalty = parsed.get("source_support", 2) < 2 or source_critical_error
    if source_support_penalty and not findings:
        raise JudgeError("A source-support penalty has no structured source finding.")
    # A score below 2 needs a stated defect. Code cannot check the content of the reason.
    rationale = answer.get("rationale")
    if not isinstance(rationale, dict):
        raise JudgeError("Answer has no rationale.")
    for dimension, value in parsed.items():
        reason = rationale.get(dimension)
        if value < 2 and (not isinstance(reason, str) or not reason.strip()):
            raise JudgeError(f"Score {value} for {dimension} has no rationale.")
    coverage_findings = None
    if required_units:
        if not isinstance(output, str):
            raise JudgeError("Coverage findings require the assessed output.")
        expected_units = [unit["id"] for unit in required_units]
        if len(expected_units) != len(set(expected_units)):
            raise JudgeError("Required coverage units contain duplicate IDs.")
        raw_coverage = answer.get("coverage_findings")
        if not isinstance(raw_coverage, list):
            raise JudgeError("Coverage findings must be a list.")
        coverage_findings = {}
        output_lines = output.splitlines()
        for finding in raw_coverage:
            if not isinstance(finding, dict) or set(finding) != {
                "unit_id",
                "status",
                "evidence_lines",
                "explanation",
            }:
                raise JudgeError("A coverage finding has an invalid shape.")
            unit_id = finding["unit_id"]
            status = finding["status"]
            evidence_lines = finding["evidence_lines"]
            explanation = finding["explanation"]
            if not isinstance(unit_id, str) or not isinstance(status, str):
                raise JudgeError("Coverage unit IDs and statuses must be strings.")
            if unit_id not in expected_units or unit_id in coverage_findings:
                raise JudgeError(f"Unexpected or repeated coverage unit {unit_id!r}.")
            if status not in {"present", "absent", "incorrect"}:
                raise JudgeError(f"Invalid coverage status for {unit_id}.")
            if not isinstance(evidence_lines, list) or any(
                type(number) is not int or not 1 <= number <= len(output_lines)
                for number in evidence_lines
            ):
                raise JudgeError(f"Invalid output line numbers for {unit_id}.")
            if len(evidence_lines) != len(set(evidence_lines)):
                raise JudgeError(f"Repeated output line numbers for {unit_id}.")
            if not isinstance(explanation, str):
                raise JudgeError(f"Invalid explanation for {unit_id}.")
            if not explanation.strip():
                raise JudgeError(f"Coverage unit {unit_id} has no explanation.")
            if status == "absent":
                if evidence_lines:
                    raise JudgeError(f"Absent coverage unit {unit_id} cites output lines.")
            elif not evidence_lines or not any(
                output_lines[number - 1].strip() for number in evidence_lines
            ):
                raise JudgeError(f"Coverage unit {unit_id} has no output evidence.")
            coverage_findings[unit_id] = {
                **finding,
                "quotation": "\n".join(output_lines[number - 1] for number in evidence_lines),
            }
        if set(coverage_findings) != set(expected_units):
            missing = sorted(set(expected_units) - set(coverage_findings))
            raise JudgeError(f"Coverage findings miss units: {missing}.")
        computed = coverage_score(required_units, coverage_findings)
        if computed is not None and "coverage" in parsed:
            parsed["coverage"] = computed
        elif parsed.get("coverage") == 2 and any(
            finding["status"] == "absent" for finding in coverage_findings.values()
        ):
            raise JudgeError("Coverage score 2 conflicts with absent units.")
        if parsed.get("coverage") == 0 and all(
            finding["status"] == "present" for finding in coverage_findings.values()
        ):
            raise JudgeError("Coverage score 0 conflicts with all required units present.")
    return {
        "scores": parsed,
        "critical_errors": critical,
        "claims_assessed": assessed,
        "claims_unsupported": unsupported,
        "source_findings": findings,
        **({"coverage_findings": coverage_findings} if coverage_findings is not None else {}),
        "decision": decision(parsed, critical),
        "rationale": rationale,
    }


def decision(scores: dict[str, int], critical_errors: list[str]) -> str:
    """Frozen decision rule: full pass, partial pass, or failure."""
    values = list(scores.values())
    if critical_errors or 0 in values:
        return "failure"
    return "full_pass" if all(value == 2 for value in values) else "partial_pass"


# Groundedness: code splits the output into statements, and the judge compares each
# statement with its sources. An omission is not a grounding defect; the physician can ask
# again or open the cited data.
STATEMENT_DEFECTS = (
    "unsupported", "contradicted", "overclaim", "wrong_attribution", "wrong_date",
)
STATEMENT_UNVERIFIED = ("uncited", "unavailable_citation")
STATEMENT_VERDICTS = ("supported", "not_a_claim", *STATEMENT_DEFECTS, *STATEMENT_UNVERIFIED)
_STATEMENT_CITATION = re.compile(
    r"\[?\(?\b(?:fhir:|imported:|imaging:|imaging-result:|query:)[A-Za-z0-9_.:/-]+\)?\]?"
    rf"|(?:{_REFERENCE.pattern})"
)

STATEMENTS_SYSTEM = """You check if each statement of a clinical output is true to its
sources. Do not judge completeness or clinical quality. Return only JSON.

Ask one question for each statement: do the sources say this, about the same person, on the
same date, with the same certainty? Give one verdict:
- supported: yes. Paraphrase, summary, a reasonable inference, and a left-out detail are
  supported, unless the statement says the opposite of a source.
- not_a_claim: the statement gives no fact about the patient: a heading, the output's own
  advice or plan, general clinical knowledge, a reference to the request, or a note about
  what the system read, searched, or could not find.
- unsupported: no source says it.
- contradicted: a source says the opposite or a different value.
- overclaim: the statement is more certain than its sources: a report as a confirmed fact,
  a possibility as a diagnosis, a plan as done, a relayed account as first-hand.
- wrong_attribution: the statement gives the account to the wrong person or source.
- wrong_date: the statement gives the wrong date or order of an event.

How to read the sources:
- Check a statement against the sources it cites and the other messages of the same
  check-in session. Check a statement with no citation against all sources.
- An imported:<scenario>:day-N:<reporter>:seq-K message is what that reporter said on that
  day. A message about "yesterday" or a weekday describes an earlier day; use the calendar.
- The fhir: sources are the complete hospital record. The record tools read only these
  types: {types}. query: sources are searches that found nothing. Runtime tool errors are
  system failures, not patient facts.
- The imaging catalog and FHIR ImagingStudy.started are separate: a catalog entry with no
  date supports "the catalogued study has no date". imaging-result sources are model
  analyses by the system; analyzed_at is the run time, so an analysis run after the review
  date was not available at that review.

For each verdict other than supported and not_a_claim, quote the wrong words and the source
text, or say that no source has them. When you are unsure, use supported.""".replace(
    "{types}", ", ".join(FHIR_RESOURCE_TYPES)
)

CITED_STATEMENTS_SYSTEM = """Check the statements against their cited sources only.
The supplied sources do not contain the complete patient record. For each statement return
supported, not_a_claim, unsupported, contradicted, overclaim, wrong_attribution, wrong_date,
uncited, or unavailable_citation. Use uncited for an unreferenced patient claim. Use
unavailable_citation when a cited source is missing. Neither means the claim is false.
Use not_a_claim for headings, general knowledge, advice and system limitations.
For cited claims, assess the cited source and other messages of the same check-in session.
Do not infer anything about sources not supplied. For a defect, quote the statement and
source. Use the calendar for relative dates. Return only JSON."""

def split_statements(output: str, sources: dict[str, str] | None = None) -> list[dict]:
    """Statements of an output: its non-empty lines without their citation tokens, each
    with the IDs it cites. The Doctor cites at the end of a line, so a line is one unit."""
    statements = []
    previous_statement = False
    for line in output.splitlines():
        cites = []

        def remove_citation(match: re.Match, found: list[str] = cites) -> str:
            cite = match.group().strip("[]().,;:")
            plain = cite.removeprefix("obs:")
            if plain.startswith("imported:") and PLACEHOLDER.search(plain):
                return match.group()
            if (
                not plain.startswith(("fhir:", "imported:", "imaging:", "imaging-result:", "query:"))
                and "/" in plain
                and not re.search(r"\d", plain.split("/", 1)[1])
                and f"fhir:{plain}" not in (sources or {})
            ):
                return match.group()
            found.append(cite)
            return ""

        text = _STATEMENT_CITATION.sub(remove_citation, line).strip(" -*#>|\t")
        if any(character.isalnum() for character in text):
            statements.append({"text": text, "cites": list(dict.fromkeys(cites))})
            previous_statement = True
        elif cites and not text and previous_statement:
            statements[-1]["cites"] = list(dict.fromkeys([*statements[-1]["cites"], *cites]))
        else:
            previous_statement = False
    return statements


def statements_question(
    request: str,
    statements: list[dict],
    sources: dict[str, str],
    *,
    runtime_errors: list[str] | None = None,
    calendar: str | None = None,
    cited_only: bool = False,
) -> tuple[str, str]:
    numbered = "\n".join(
        f"{n}. {s['text']}" + (f"  [cites: {', '.join(s['cites'])}]" if s["cites"] else "")
        for n, s in enumerate(statements, 1)
    )
    schema = {"statements": [{"n": "integer", "verdict": "|".join(STATEMENT_VERDICTS), "quote": "string"}]}
    user = (
        f"Request:\n{request}\n\n"
        f"Sources:\n{_sources_block(sources)}\n\n"
        f"Runtime tool errors:\n{chr(10).join(runtime_errors or []) or '(none)'}\n\n"
        f"Calendar:\n{calendar or '(none)'}\n\n"
        f"Statements:\n{numbered}\n\n"
        f"Answer with this JSON shape:\n{json.dumps(schema)}"
    )
    return CITED_STATEMENTS_SYSTEM if cited_only else STATEMENTS_SYSTEM, user


def statements_schema() -> dict:
    item = {
        "type": "object",
        "properties": {
            "n": {"type": "integer"},
            "verdict": {"type": "string", "enum": list(STATEMENT_VERDICTS)},
            "quote": {"type": "string"},
        },
        "required": ["n", "verdict", "quote"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"statements": {"type": "array", "items": item}},
        "required": ["statements"],
        "additionalProperties": False,
    }


def parse_statements(answer: dict, count: int) -> dict:
    items = answer.get("statements")
    if not isinstance(items, list):
        raise JudgeError("Answer has no statements list.")
    verdicts: dict[int, dict] = {}
    for item in items:
        n = item.get("n") if isinstance(item, dict) else None
        if not isinstance(n, int) or not 1 <= n <= count or n in verdicts:
            raise JudgeError(f"Unexpected or repeated statement {n!r}.")
        verdict, quote = item.get("verdict"), str(item.get("quote", ""))
        if verdict not in STATEMENT_VERDICTS:
            raise JudgeError(f"Invalid verdict {verdict!r} for statement {n}.")
        if verdict in STATEMENT_DEFECTS and not quote.strip():
            raise JudgeError(f"Statement {n} has a defect but no quote.")
        verdicts[n] = {"verdict": verdict, "quote": quote}
    if len(verdicts) != count:
        raise JudgeError(f"Answer misses statements: {sorted(set(range(1, count + 1)) - set(verdicts))}.")
    return {"verdicts": [verdicts[n] for n in range(1, count + 1)]}


def merge_statement_votes(statements: list[dict], votes: list[dict]) -> dict:
    """A statement is ungrounded when most scored votes give it a defect of any kind. A
    citation to a missing source is a defect. A statement that most votes find uncited is
    unverified: its ungrounded value is None, and the answer stays scored."""
    scored = [vote["verdicts"] for vote in votes if vote.get("status") == "scored"]
    if not scored:
        return {"status": "unscored", "error": "; ".join(v.get("error", "") for v in votes)}
    merged = []
    for n, statement in enumerate(statements):
        given = [vote[n] for vote in scored]
        defects = [v for v in given if v["verdict"] in (*STATEMENT_DEFECTS, "unavailable_citation")]
        uncited = sum(v["verdict"] == "uncited" for v in given)
        merged.append({
            **statement,
            "votes": [v["verdict"] for v in given],
            "ungrounded": None if 2 * uncited > len(given) else 2 * len(defects) > len(given),
            "quotes": [v["quote"] for v in defects],
        })
    return {"status": "scored", "votes": len(scored), "statements": merged}


# ---------------------------------------------------------------------------
# Specialist-result incorporation and specialist-call necessity
# ---------------------------------------------------------------------------


def incorporation_question(
    unit: dict, specialist_answers: list[str], output: str
) -> tuple[str, str]:
    system = (
        "You judge whether a clinical answer correctly incorporates a required specialist-agent "
        "result. Judge the meaning, as a physician who reads the answer: does the physician "
        "learn the result's core facts, without a misstatement? The core facts are the "
        "result's must_state facts; a secondary detail inside a must_state item (a duration, a "
        "date, a dose history, a reason) is optional, but it must not be stated wrongly. When essential_limit is true, the result's uncertainty is "
        "also core. The rest of the result is optional detail: its absence does not make the "
        "result not incorporated. Accept any wording, summary, or approximation that carries "
        "the same core fact. The result is not incorporated only when the answer leaves out a "
        "core fact, or states one wrongly or more strongly than the sources (an unknown as "
        "known, a report as confirmed). Assess the answer's own text and its attached "
        "Evidence together. Return only JSON."
    )
    answers = "\n\n".join(f"<<<\n{text}\n>>>" for text in specialist_answers)
    user = (
        f"Required specialist result:\n{json.dumps(_coverage_view(unit), indent=1, ensure_ascii=False)}\n\n"
        f"Specialist answers received:\n{answers}\n\nAnswer under assessment:\n<<<\n{output}\n>>>\n\n"
        'Answer with this JSON shape: {"incorporated": true, "evidence": "string"}'
    )
    return system, user


def parse_flag(answer: dict, key: str) -> dict:
    value = answer.get(key)
    if not isinstance(value, bool):
        raise JudgeError(f"Answer has no boolean {key}.")
    reason = answer.get("evidence", answer.get("reason", ""))
    return {key: value, "reason": str(reason)}
