"""Score saved run artifacts of one (model, scenario) against the frozen reference.

    poetry run python -m src.evaluation.score output/evaluation/dev-smoke

Structural checks are deterministic. Content checks use the model judge; each accepted
judge answer is cached under <run>/judgments, so rescoring reuses it. A failed or
invalid judge answer leaves the item unscored; it never becomes a failure.

With --votes K, the unit questions of the preservation stages are asked K times, each
vote under its own judge identity, and each unit takes the majority status. The other
questions are asked once.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.agent.care_context import consultation_text
from src.agent.citations import HISTORY_TOOLS, citation_check, cited_ids
from src.evaluation import judge as J
from src.evaluation.cited_sources import select_cited_sources
from src.evaluation.judge_factory import judge_from_env
from src.evaluation.judgments import Judgments, vote_identities, voted
from src.evaluation.release import (
    PRESERVATION_STAGES,
    ROOT,
    SOURCE_NEED_STAGE,
    SPECIALIST_STAGE,
    Case,
    Release,
)
from src.evaluation.rq3 import score_rq3
from src.evaluation.run import dump
from src.evaluation.trace import (
    ANALYSIS_TOOLS,
    CARE_TOOL,
    SPECIALIST_TOOLS,
    STORED_IMAGING_TOOL,
    consultations,
    normalize,
    retrieval,
    source_retrieved,
)
from src.ledger.corpus import IMPORTED_SPEAKERS

# Which run artifact each reference stage compares. None: the runtime has no such artifact.
STAGE_ARTIFACTS: dict[str, str | None] = {
    "conversation_to_derived": "recaps",
    "ledger_to_context": "care_context",
    "source_to_specialist": "specialist_answers",
    "sources_to_proposal": "stored_proposal",
    "proposal_to_revision": "stored_proposal",
    "source_to_final": "final_answer",
}
ARTIFACT_NOTES = {
    "recaps": "Recaps CareAgent wrote of the stored check-ins recorded up to the task date: "
    "reporter, recording date, summary, changes, and unknowns.",
    "care_context": "Care Agent answers the Doctor received in the task, with the check-in "
    "messages they cite, plus earlier Doctor answers of the same session.",
    "specialist_answers": "EHR and Radiologist Agent answers the Doctor received in the task.",
    "final_answer": "The Doctor's final answer text.",
    "stored_proposal": "Content of the latest record-update proposal the Doctor stored during "
    "the task; empty when it stored none.",
}

REVISION_GROUPS = {"follow_up", "record_revision"}

# Judge questions are independent, and the cache keeps one file per question, so threads
# can ask them at the same time. The order of the results stays the order of the items.
JUDGE_WORKERS = 16


def parallel(function, items: list) -> list:
    with ThreadPoolExecutor(JUDGE_WORKERS) as pool:
        return list(pool.map(function, items))


def load_tasks(run_dir: Path, case: Case) -> dict[str, dict]:
    tasks = {}
    for task in case.tasks:
        path = run_dir / "tasks" / f"{task['question_id']}.json"
        tasks[task["id"]] = (
            json.loads(path.read_text())
            if path.exists()
            else {"task_id": task["id"], "status": "not_run"}
        )
    return tasks


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------


def score_storage(case: Case, readback: list[dict]) -> list[dict]:
    """Compare each stored check-in with its corpus transcript, field by field."""
    corpus = json.loads(case.conversation_path.read_text())
    stored: dict[str, list[dict]] = {}
    for row in readback:
        stored.setdefault(row["session_id"], []).append(row)
    days = {day["day"]: day for day in corpus["days"]}
    results = []
    for opportunity in case.opportunities("storage"):
        # storage:<scenario>:day-N:<role>
        _, _, day_label, role = opportunity["id"].split(":")
        day = days[int(day_label.removeprefix("day-"))]
        transcript = day["check_ins"][role]["transcript"]
        rows = sorted(
            stored.get(f"imported:{case.scenario_id}:{day_label}:{role}", []),
            key=lambda row: row["sequence"],
        )
        mismatches = []
        if len(rows) != len(transcript):
            mismatches.append(f"{len(rows)} stored messages for {len(transcript)} turns")
        for index, (turn, row) in enumerate(zip(transcript, rows, strict=False), start=1):
            expected = {
                "patient_id": corpus["patient_id"],
                "audience": role,
                "source_day": day["day"],
                "recorded_at": day["check_in_date"],
                "sequence": turn.get("sequence", index),
                "speaker": IMPORTED_SPEAKERS.get(turn["speaker"]),
                "content": turn["text"],
            }
            for field, value in expected.items():
                if row.get(field) != value:
                    mismatches.append(f"turn {index} {field} differs")
        results.append(
            {"id": opportunity["id"], "passed": not mismatches, "mismatches": mismatches}
        )
    return results


def score_source_needs(case: Case, tasks: dict[str, dict]) -> list[dict]:
    results = []
    for task_id in case.required_tasks(SOURCE_NEED_STAGE):
        record = tasks[task_id]
        evidence = retrieval(record.get("trace", [])) if record["status"] == "completed" else None
        for unit in case.required(SOURCE_NEED_STAGE, task_id):
            entry = {"unit_id": unit["id"], "task_id": task_id}
            if evidence is None:
                entry["status"] = "not_executed"
            else:
                missing = [
                    anchor["source_id"]
                    for anchor in unit["anchors"]
                    if not source_retrieved(
                        anchor["source_id"],
                        case.sources[anchor["source_id"]],
                        anchor.get("quote", ""),
                        case.patient_id,
                        evidence,
                    )
                ]
                entry.update(
                    status="unfulfilled" if missing else "fulfilled", missing_sources=missing
                )
            results.append(entry)
    return results


def _analyses(record: dict) -> list[str]:
    """Study IDs of the fresh analyses of a task: its successful analysis tool calls."""
    return [
        (item.get("arguments") or {}).get("study_id")
        for item in record.get("trace", [])
        if item.get("tool") in ANALYSIS_TOOLS and item.get("error") is None
    ]


def score_imaging(case: Case, tasks: dict[str, dict], specialist: list[dict]) -> list[dict]:
    """Score result reuse by availability, access, citation, and incorporation."""
    results = []
    for metric in ("imaging_reuse", "imaging_rerun"):
        for opportunity in case.opportunities(metric):
            record = tasks[opportunity["task_id"]]
            entry = {"id": opportunity["id"], "metric": metric, "task_id": opportunity["task_id"]}
            if record["status"] != "completed":
                entry["status"] = "not_executed"
                results.append(entry)
                continue
            fresh = _analyses(record)
            if metric == "imaging_rerun":
                wanted = set(re.findall(r"CASE-[A-Z0-9-]+", opportunity.get("reason", "")))
                hits = [study for study in fresh if not wanted or study in wanted]
                entry.update(status="pass" if hits else "fail", fresh_analyses=len(hits))
            else:
                wanted = set(re.findall(r"CASE-[A-Z0-9-]+", opportunity.get("reason", "")))
                first_analysis = next(
                    (i for i, step in enumerate(record.get("trace", []))
                     if step.get("tool") in ANALYSIS_TOOLS),
                    len(record.get("trace", [])),
                )
                sources = imaging_sources(case, tasks, opportunity["task_id"])
                eligible = {
                    key: source
                    for key, value in sources.items()
                    if key.startswith("imaging-result:")
                    for source in [json.loads(value)]
                    if (not wanted or source["result"]["study_id"] in wanted)
                    and any(
                        not event["current_task"]
                        or (event["tool"] == STORED_IMAGING_TOOL
                            and event["trace_index"] < first_analysis)
                        for event in source["availability_events"]
                    )
                }
                cited = set(J.source_citation_resolution(record.get("answer") or "", eligible)[
                    "available"
                ])
                history_cited = set(J.source_citation_resolution(
                    "\n".join(record.get("history_answers", [])), eligible
                )["available"])
                accessed = {
                    key for key, source in eligible.items()
                    if key in history_cited or any(
                        event["current_task"] and event["tool"] == STORED_IMAGING_TOOL
                        and event["trace_index"] < first_analysis
                        for event in source["availability_events"]
                    )
                }
                reused = bool(cited & accessed)
                incorporated = [
                    item
                    for item in specialist
                    if item["task_id"] == opportunity["task_id"]
                    and item["specialist"] == "radiologist"
                ]
                states = {item["status"] for item in incorporated}
                if not eligible:
                    entry.update(status="not_available", reason="no prior result available for reuse")
                elif fresh:
                    entry.update(
                        status="fail", reason=f"{len(fresh)} fresh analyses instead of reuse"
                    )
                elif "unscored" in states:
                    entry["status"] = "unscored"
                elif reused and (not incorporated or states == {"incorporated"}):
                    entry["status"] = "pass"
                else:
                    entry.update(status="fail", reason="stored result not incorporated")
            results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Judge checks
# ---------------------------------------------------------------------------


PROVENANCE_STAGE = "source_to_final"


def _record_steps(trace: list[dict]) -> list[dict]:
    """Steps of the record tools: no consultations, conversation text, or history tools."""
    return [
        step
        for step in trace
        if not str(step.get("tool", "")).startswith("consult_")
        and step.get("tool") != "read_conversation"
        and step.get("tool") not in HISTORY_TOOLS
    ]


def anchor_messages(source_id: str, quote: str, readback: list[dict]) -> tuple[str, list[str]]:
    """The check-in session of an anchor and the message IDs that hold its quote. A quote
    that spans turns resolves to every message of the session whose text is part of it."""
    session = "imported:" + source_id.partition(":")[2]
    quote = normalize(quote)
    rows = [m for m in readback if m["session_id"] == session]
    single = [m["message_id"] for m in rows if quote and quote in normalize(m["content"])]
    if single:
        return session, single
    return session, [
        m["message_id"]
        for m in rows
        if len(normalize(m["content"])) > 15 and normalize(m["content"]) in quote
    ]


def _provenance_targets(case: Case, task_id: str) -> list[dict]:
    """Required sources of one task output: the anchors of its required final units, and the
    imaging study of each imaging opportunity of the task."""
    targets = [
        {"unit_id": unit["id"], "priority": unit.get("priority"), **anchor}
        for unit in case.required(PROVENANCE_STAGE, task_id)
        for anchor in unit["anchors"]
    ]
    for metric in ("imaging_reuse", "imaging_rerun"):
        for opportunity in case.opportunities(metric):
            if opportunity["task_id"] == task_id:
                for study in sorted(set(re.findall(r"CASE-[A-Z0-9-]+", opportunity["reason"]))):
                    targets.append(
                        {"unit_id": opportunity["id"], "priority": "essential", "study": study}
                    )
    return targets


def score_provenance(
    case: Case,
    tasks: dict[str, dict],
    readback: list[dict],
    records: dict | None,
    imaging_results: list[dict],
    consults: bool,
) -> list[dict]:
    """Trace each required source ID through three stages: the record tools returned it, a
    specialist answer cited it, and the task output cited it. A check-in counts as returned
    only when the text of the anchored message was read; a recap is not a source. An imaging
    study counts as cited when the output cites an imaging result of that study. The output
    of a proposal task is its stored proposal."""
    study_of = {r["result_id"]: r["study_id"] for r in imaging_results}
    proposal_tasks = (
        {records["proposal_task"], records["revision_task"]} if records is not None else set()
    )
    results = []
    for task in case.tasks:
        targets = _provenance_targets(case, task["id"])
        if not targets:
            continue
        record = tasks[task["id"]]
        if record["status"] != "completed":
            results += [{"task_id": task["id"], "unit_id": t["unit_id"], "status": "not_executed"}
                        for t in targets]
            continue
        trace = record.get("trace", [])
        evidence = retrieval(_record_steps(trace))
        specialist_text = "\n".join(c["answer"] for c in consultations(trace) if c["error"] is None)
        output = record["answer"]
        if task["id"] in proposal_tasks:
            # A proposal cites its sources in its text or in its sources field.
            mine = [p for p in records["stored"]["proposals"] if p["request_id"] == task["id"]]
            output = "\n".join([mine[-1]["content"], *mine[-1]["sources"]]) if mine else ""
        cited = {"specialist": set(cited_ids(specialist_text)), "final": set(cited_ids(output))}
        for target in targets:
            source_id = target.get("source_id", "imaging-fixture")
            kind = source_id.split(":", 1)[0]
            entry = {
                "task_id": task["id"],
                "unit_id": target["unit_id"],
                "priority": target["priority"],
                "source_id": source_id,
                "kind": {"raw-check-in": "check-in", "fhir": "fhir"}.get(kind, "imaging"),
                "status": "scored",
            }
            near: dict[str, bool] = {}
            if kind == "raw-check-in":
                session, ids = anchor_messages(source_id, target.get("quote", ""), readback)
                returned = any(i in evidence.ledger_text for i in ids)
                hits = {stage: any(i in found for i in ids) for stage, found in cited.items()}
                near = {
                    stage: not hits[stage] and any(i.startswith(session + ":") for i in found)
                    for stage, found in cited.items()
                }
            elif kind == "fhir":
                reference = source_id.removeprefix("fhir:")
                returned = reference in evidence.fhir
                hits = {stage: reference in found for stage, found in cited.items()}
            else:
                studies = (
                    [target["study"]]
                    if "study" in target
                    else json.loads(case.sources[source_id]).get("image_ids", [])
                )
                entry["source_id"] = ",".join(studies)
                returned = any(s in evidence.imaging_studies for s in studies)
                hits = {
                    stage: any(study_of.get(i) in studies for i in found)
                    for stage, found in cited.items()
                }
            entry.update(
                returned=returned,
                specialist_cited=hits["specialist"] if consults else None,
                final_cited=hits["final"],
                final_near_miss=near.get("final", False),
            )
            results.append(entry)
    return results


def _session_consultation_answers(
    tasks: dict[str, dict], case: Case, task_id: str, tool: str
) -> list[str]:
    """Answers of one consultation tool in this task and in earlier tasks of its session."""
    session = tasks[task_id].get("session_id")
    answers = []
    for task in case.tasks:
        record = tasks[task["id"]]
        if record.get("session_id") == session and record["status"] == "completed":
            answers += [
                text
                for c in consultations(record.get("trace", []), tool)
                if (text := consultation_text(c["result"]))
            ]
        if task["id"] == task_id:
            break
    return answers


def score_specialists(case: Case, tasks: dict[str, dict], judged: Judgments) -> list[dict]:
    tools = {specialist: tool for tool, specialist in SPECIALIST_TOOLS.items()}

    def score(job: tuple[str, dict]) -> dict:
        task_id, unit = job
        record = tasks[task_id]
        specialist = case.specialist_of(unit["id"])
        entry = {"unit_id": unit["id"], "task_id": task_id, "specialist": specialist}
        if record["status"] != "completed":
            entry["status"] = "not_executed"
        else:
            answers = _session_consultation_answers(
                tasks, case, task_id, tools.get(specialist, "")
            )
            if not answers:
                entry.update(status="not_incorporated", reason="no answer from this specialist")
            else:
                verdict = judged(
                    J.incorporation_question(unit, answers, record["answer"]),
                    lambda answer: J.parse_flag(answer, "incorporated"),
                )
                if verdict["status"] == "unscored":
                    entry.update(status="unscored", error=verdict["error"])
                else:
                    entry.update(
                        status="incorporated"
                        if verdict["incorporated"]
                        else "not_incorporated",
                        reason=verdict["reason"],
                    )
        return entry

    jobs = [
        (task_id, unit)
        for task_id in case.required_tasks(SPECIALIST_STAGE)
        for unit in case.required(SPECIALIST_STAGE, task_id)
    ]
    return parallel(score, jobs)


def score_calls(case: Case, tasks: dict[str, dict]) -> list[dict]:
    """A specialist call is necessary when the task has a required unit that this
    specialist owns. Code decides; no judge."""
    results = []
    for opportunity in case.opportunities("unnecessary_calls"):
        task_id = opportunity["task_id"]
        record = tasks[task_id]
        entry: dict[str, Any] = {"id": opportunity["id"], "task_id": task_id}
        if record["status"] != "completed":
            entry["status"] = "not_executed"
            results.append(entry)
            continue
        needs = case.required(SOURCE_NEED_STAGE, task_id) + case.required(SPECIALIST_STAGE, task_id)
        owners = {case.specialist_of(unit["id"]) for unit in needs}
        calls = [
            {
                "tool": call["tool"],
                "question": call["question"],
                "status": "necessary" if SPECIALIST_TOOLS[call["tool"]] in owners else "unnecessary",
            }
            for call in consultations(record.get("trace", []))
            if call["tool"] in SPECIALIST_TOOLS
        ]
        entry.update(status="exercised", calls=calls)
        results.append(entry)
    return results


def _artifact(kind: str, record: dict, records: dict | None) -> str:
    trace = record.get("trace", [])
    if kind == "final_answer":
        return record["answer"]
    if kind == "stored_proposal":
        stored = [p for p in records["stored"]["proposals"] if p["request_id"] == record["task_id"]]
        return stored[-1]["content"] if stored else ""
    if kind == "specialist_answers":
        parts = [
            consultation_text(c["result"])
            for tool in SPECIALIST_TOOLS
            for c in consultations(trace, tool)
        ]
    else:
        parts = [
            consultation_text(c["result"]) for c in consultations(trace, CARE_TOOL)
        ] + record.get("history_answers", [])
    return "\n\n---\n\n".join(part for part in parts if part)


def recap_text(recaps: list[dict], as_of: str) -> str:
    """The recaps of the check-ins recorded up to the task date."""
    parts = []
    for recap in recaps:
        if recap["recorded_at"][:10] > as_of:
            continue
        lines = [
            f"[{recap['session_id']}] {recap['reporter']} check-in, recorded {recap['recorded_at']}"
        ]
        lines.append(recap["summary"] if recap["status"] == "stored" else "(no recap: failed)")
        lines += [f"Change: {item['text']}" for item in recap["changes"]]
        lines += [f"Unknown: {item['text']}" for item in recap["unknowns"]]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def score_preservation(
    case: Case,
    tasks: dict[str, dict],
    judged: Judgments,
    records: dict | None,
    recaps: list[dict] | None = None,
    consults: bool = True,
) -> dict[str, dict]:
    as_of = {task["id"]: task["as_of"] for task in case.tasks}

    def compare(job: tuple[str, str]) -> dict[str, Any]:
        stage, task_id = job
        artifact = STAGE_ARTIFACTS[stage]
        units = case.required(stage, task_id)
        record = tasks[task_id]
        comparison: dict[str, Any] = {"task_id": task_id, "required": len(units)}
        missing = {"stored_proposal": records, "recaps": recaps}
        # The single-agent baseline makes no consultations, so it has no such answers.
        consultation = artifact in {"care_context", "specialist_answers"}
        if (
            artifact is None
            or (artifact in missing and missing[artifact] is None)
            or (consultation and not consults)
        ):
            comparison["status"] = "runtime_unsupported"
        # Recaps are made from the stored check-ins, independent of the task's execution.
        elif artifact != "recaps" and record["status"] != "completed":
            comparison["status"] = "not_executed"
        else:
            text = (
                recap_text(recaps, as_of[task_id])
                if artifact == "recaps"
                else _artifact(artifact, record, records)
            )
            labels = stage == "source_to_final"
            if not text.strip():
                verdict = {
                    "status": "scored",
                    "units": {
                        unit["id"]: {
                            "status": "omitted",
                            "evidence": "empty artifact",
                            **({"labels": dict.fromkeys(J.LABELS, False)} if labels else {}),
                        }
                        for unit in units
                    },
                }
            else:
                system, user = J.units_question(
                    stage, text, units, case.anchor_sources(units), labels=labels
                )
                verdict = judged(
                    (system, user),
                    lambda answer, units=units, labels=labels: {
                        "units": J.parse_units(answer, units, labels=labels)
                    },
                )
            comparison["status"] = verdict["status"]
            comparison["units"] = verdict.get("units")
            comparison["error"] = verdict.get("error")
        return comparison

    jobs = [(stage, task_id) for stage in PRESERVATION_STAGES for task_id in case.required_tasks(stage)]
    done = dict(zip(jobs, parallel(compare, jobs), strict=True))
    return {
        stage: {
            "artifact": STAGE_ARTIFACTS[stage],
            "comparisons": [done[(stage, task_id)] for task_id in case.required_tasks(stage)],
        }
        for stage in PRESERVATION_STAGES
    }


def imaging_sources(case: Case, tasks: dict[str, dict], task_id: str) -> dict[str, str]:
    """Keep imaging evidence through this task, with ordered reads and analysis events."""
    studies = {
        study["study_id"]: {
            "study_id": study["study_id"],
            "catalog": case.imaging_catalog,
            "runtime_events": [],
        }
        for study in case.imaging_catalog["studies"]
    }
    results: dict[str, dict] = {}
    for task in case.tasks:
        record = tasks[task["id"]]
        for index, step in enumerate(record.get("trace", [])):
            observation = step.get("observation")
            tool = step.get("tool")
            if (
                step.get("error") is not None
                or not isinstance(observation, dict)
                or observation.get("patient_id") != case.patient_id
                or observation.get("study_id") not in studies
                or tool not in ANALYSIS_TOOLS | {STORED_IMAGING_TOOL}
            ):
                continue
            study_id = observation["study_id"]
            event = {
                "task_id": task["id"],
                "current_task": task["id"] == task_id,
                "trace_index": index,
                "tool": tool,
            }
            stored = (
                observation.get("stored_results", [])
                if tool == STORED_IMAGING_TOOL
                else [observation]
            )
            result_ids = []
            for result in stored:
                if (
                    result.get("patient_id") != case.patient_id
                    or result.get("study_id") != study_id
                    or not result.get("result_id")
                ):
                    continue
                result_id = result["result_id"]
                result_ids.append(result_id)
                entry = results.setdefault(result_id, {"result": result, "availability_events": []})
                entry["availability_events"].append(event)
            if tool == STORED_IMAGING_TOOL:
                detail = {k: v for k, v in observation.items() if k != "stored_results"}
                detail["stored_result_ids"] = result_ids
            else:
                detail = {"result_id": observation.get("result_id"), "stored_on_success": True}
            studies[study_id]["runtime_events"].append({**event, "observation": detail})
        if task["id"] == task_id:
            break
    return {
        **{
            f"imaging:{study}": json.dumps(value, sort_keys=True)
            for study, value in studies.items()
        },
        **{result_id: json.dumps(value, sort_keys=True) for result_id, value in results.items()},
    }


def _query_source_events(
    traces: list[tuple[str | None, list[dict]]], patient_id: str | None
) -> dict[str, str]:
    """Summarize query metadata without copying returned FHIR resources."""
    found: dict[str, list[str]] = {}

    def add(tool: str, resource_type: str, value: dict) -> None:
        source_id = f"query:{tool}/{resource_type}"
        found.setdefault(source_id, []).append(json.dumps(value, sort_keys=True))

    for task_id, trace in traces:
        for step in trace:
            tool = step.get("tool")
            if tool == "list_imaging_studies":
                # The radiologist prompt cites an empty catalog as query:list_imaging_studies.
                observation = step.get("observation")
                observed = observation.get("patient_id") if isinstance(observation, dict) else None
                if patient_id is None or observed in (None, patient_id):
                    event = {"tool": tool, "task_id": task_id, "result": observation}
                    found.setdefault(f"query:{tool}", []).append(json.dumps(event, sort_keys=True))
                continue
            if tool not in {"query_fhir", "count_fhir_records"}:
                continue
            arguments = step.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            observation = step.get("observation")
            if not isinstance(observation, dict):
                observation = {}
            patient_ids = (arguments.get("patient_id"), observation.get("patient_id"))
            if patient_id is not None and any(
                observed_id is not None and observed_id != patient_id
                for observed_id in patient_ids
            ):
                continue

            event: dict[str, Any] = {"tool": tool}
            if task_id:
                event["task_id"] = task_id
            error = step.get("error")

            if tool == "query_fhir":
                resource_type = arguments.get("resource_type") or observation.get("resource_type")
                if not isinstance(resource_type, str) or not resource_type:
                    continue
                event["arguments"] = {
                    key: value for key, value in arguments.items() if key != "cursor"
                }
                if "cursor" in arguments:
                    event["continuation_page"] = True
                if error:
                    event["error"] = error
                elif not observation:
                    event["error"] = "No structured query result was returned."
                else:
                    observed_type = observation.get("resource_type")
                    if observed_type and observed_type != resource_type:
                        event["error"] = "Query result resource type did not match its arguments."
                    else:
                        metadata = {
                            key: observation[key]
                            for key in (
                                "patient_id",
                                "total",
                                "returned",
                                "has_more",
                                "outside_date_filter",
                            )
                            if key in observation
                        }
                        resource_ids = []
                        resources = observation.get("resources")
                        if isinstance(resources, list):
                            for resource in resources:
                                if not isinstance(resource, dict):
                                    continue
                                resource_id = resource.get("id")
                                resource_kind = resource.get("resourceType") or resource_type
                                if not isinstance(resource_id, str) or not resource_id:
                                    continue
                                resource_ids.append(f"{resource_kind}/{resource_id}")
                        metadata["resource_ids"] = resource_ids
                        partial_errors = observation.get("errors")
                        if isinstance(partial_errors, dict) and partial_errors:
                            metadata["errors"] = partial_errors
                        event["result"] = metadata
                add(tool, resource_type, event)
                continue

            query_specs = arguments.get("queries")
            if not isinstance(query_specs, list):
                query_specs = []
            counts = observation.get("counts")
            if not isinstance(counts, dict):
                counts = {}
            errors = observation.get("errors")
            if not isinstance(errors, dict):
                errors = {}
            for query in query_specs:
                if not isinstance(query, dict):
                    continue
                resource_type = query.get("resource_type")
                if not isinstance(resource_type, str) or not resource_type:
                    continue
                item: dict[str, Any] = {
                    **event,
                    "arguments": {
                        **{key: value for key, value in arguments.items() if key != "queries"},
                        "query": query,
                    },
                }
                component_error = error or errors.get(resource_type)
                count = counts.get(resource_type)
                if component_error:
                    item["error"] = component_error
                elif type(count) is int:
                    item["count"] = count
                elif count is None and resource_type in counts:
                    item["error"] = "Count was unavailable; no zero count was reported."
                else:
                    item["error"] = "No count was returned for this resource type."
                add(tool, resource_type, item)
    return {source: "\n".join(lines) for source, lines in found.items()}


def query_results(trace: list[dict]) -> dict[str, str]:
    """Summarize query evidence from one task, grouped by exact tool and resource type."""
    return _query_source_events([(None, trace)], patient_id=None)


def _session_traces(
    tasks: dict[str, dict], case: Case, task_id: str
) -> list[tuple[str | None, list[dict]]]:
    """Get this task and completed earlier tasks from its session."""
    session_id = tasks[task_id].get("session_id")
    traces: list[tuple[str | None, list[dict]]] = []
    for task in case.tasks:
        record = tasks[task["id"]]
        if task["id"] == task_id:
            if record.get("status") == "completed":
                traces.append((task_id, record.get("trace", [])))
            return traces
        if (
            session_id is not None
            and record.get("session_id") == session_id
            and record.get("status") == "completed"
        ):
            traces.append((task["id"], record.get("trace", [])))
    raise ValueError(f"Task is not in the case plan: {task_id}")


def session_query_results(
    tasks: dict[str, dict], case: Case, task_id: str
) -> dict[str, str]:
    """Include query evidence from this task and completed earlier tasks in its session."""
    return _query_source_events(_session_traces(tasks, case, task_id), patient_id=case.patient_id)


def session_runtime_errors(tasks: dict[str, dict], case: Case, task_id: str) -> list[str]:
    """Keep current errors and identify earlier errors available in the same session."""
    errors = runtime_errors(tasks[task_id].get("trace", []))
    for earlier_id, trace in _session_traces(tasks, case, task_id):
        if earlier_id != task_id:
            errors.extend(f"Earlier task {earlier_id}: {error}" for error in runtime_errors(trace))
    return errors


def runtime_errors(trace: list[dict]) -> list[str]:
    """Failed tool calls, and failed parts of a call that returned partial results."""
    errors = []
    for step in trace:
        name = f"{step.get('agent')} {step.get('tool')}"
        if step.get("error"):
            errors.append(f"{name}: {step['error']}")
        observation = step.get("observation")
        partial = observation.get("errors") if isinstance(observation, dict) else None
        if isinstance(partial, dict):
            errors.extend(f"{name} {part}: {error}" for part, error in partial.items())
    return errors


def score_quality(
    case: Case,
    tasks: dict[str, dict],
    judged: Judgments,
    rubric: dict,
    groups: dict[str, str],
    preservation: dict,
    records: dict | None = None,
    statement_judges: list[Judgments] | None = None,
) -> list[dict]:
    """Judge each task output. The output of a proposal or revision task is the stored
    proposal; a revision is judged with its stored parent and the rejection reason."""
    finals = _comparisons(preservation, "source_to_final")
    proposal_tasks = (
        {records["proposal_task"], records["revision_task"]} if records is not None else set()
    )
    stored = {p["proposal_id"]: p for p in records["stored"]["proposals"]} if records else {}

    def score(task: dict) -> dict[str, Any]:
        record = tasks[task["id"]]
        entry: dict[str, Any] = {"task_id": task["id"], "question_id": task["question_id"]}
        if record["status"] != "completed":
            entry["status"] = "not_executed"
            return entry
        output, previous, feedback = record["answer"], None, None
        if groups.get(task["question_id"]) in REVISION_GROUPS and task["depends_on"]:
            previous = tasks[task["depends_on"]].get("answer")
            feedback = task["prompt"]
        if task["id"] in proposal_tasks:
            mine = [p for p in stored.values() if p["request_id"] == task["id"]]
            if not mine:
                entry["status"] = "no_output"
                return entry
            output = mine[-1]["content"]
            parent = stored.get(mine[-1]["parent_id"])
            previous, feedback = (parent["content"], record.get("effective_feedback", task["prompt"])) if parent else (None, None)
        sources, required = case.judge_sources(task["id"])
        required_units = case.required("source_to_final", task["id"]) or None
        # The unit votes decide the coverage of a final answer. The rubric judge gets the
        # units only for a stored proposal, where its own findings decide coverage.
        coverage_by_code = task["id"] not in proposal_tasks and any(
            unit.get("priority") == "essential" for unit in required_units or []
        )
        rubric_units = None if coverage_by_code else required_units
        sources = {
            **sources,
            **imaging_sources(case, tasks, task["id"]),
            **session_query_results(tasks, case, task["id"]),
        }
        full_sources = sources
        sources, missing_citations = select_cited_sources(output, full_sources)
        source_citations = J.source_citation_resolution(output, full_sources)
        source_citations["available"] = sorted(
            set(source_citations["available"])
            | {key for key in sources if not key.startswith("raw-check-in:")}
        )
        source_citations["unavailable"] = sorted(set(missing_citations))
        question = J.rubric_question(
            rubric,
            record["request"],
            output,
            sources,
            required=required,
            required_units=rubric_units,
            previous=previous,
            feedback=feedback,
            runtime_errors=session_runtime_errors(tasks, case, task["id"]),
            coverage_by_code=coverage_by_code,
            calendar=J.calendar_text(task["as_of"]),
            cited_only=True,
            citation_resolution=source_citations,
        )
        entry.update(
            judged(
                question,
                lambda answer, source_citations=source_citations, rubric_units=rubric_units, output=output: J.parse_rubric(
                    answer,
                    rubric,
                    source_citations=source_citations,
                    required_units=rubric_units,
                    output=output,
                ),
                J.rubric_schema(rubric, rubric_units),
            )
        )
        if coverage_by_code:
            unit_coverage(entry, required_units, finals.get(task["id"]))
        statements = J.split_statements(output, sources=full_sources)
        for statement in statements:
            statement_sources, missing = select_cited_sources(
                " ".join(statement["cites"]), full_sources
            )
            if not statement_sources and not missing:
                statement["cites"] = []
            statement["missing_citations"] = missing
        statement_question = J.statements_question(
            record["request"],
            statements,
            sources,
            runtime_errors=session_runtime_errors(tasks, case, task["id"]),
            calendar=J.calendar_text(task["as_of"]),
            cited_only=True,
        )

        def parse_cited_statements(answer: dict) -> dict:
            parsed = J.parse_statements(statement_list(answer), len(statements))
            for statement, verdict in zip(statements, parsed["verdicts"], strict=True):
                if statement["missing_citations"]:
                    verdict["verdict"] = "unavailable_citation"
                elif not statement["cites"] and verdict["verdict"] != "not_a_claim":
                    verdict["verdict"] = "uncited"
            return parsed

        votes = parallel(
            lambda judge: judge(
                statement_question,
                parse_cited_statements,
                statement_schema(len(statements)),
            ),
            statement_judges or [judged],
        )
        entry["grounding"] = J.merge_statement_votes(statements, votes)
        entry["citation_scope"] = {
            "kind": "cited_sources", "missing_citations": missing_citations,
            "selected_sources": list(sources),
        }
        return entry

    return parallel(score, case.tasks)


def statement_schema(count: int) -> dict:
    """The statements schema with one required key per numbered statement. The judge
    sometimes splits a long line into sentences and returns more statements than it got;
    a fixed key set makes the answer hold exactly the numbered statements."""
    item = J.statements_schema()["properties"]["statements"]["items"]
    verdict = {
        **item,
        "properties": {k: v for k, v in item["properties"].items() if k != "n"},
        "required": [k for k in item["required"] if k != "n"],
    }
    keys = [str(n) for n in range(1, count + 1)]
    return {
        "type": "object",
        "properties": {
            "statements": {
                "type": "object",
                "properties": {key: verdict for key in keys},
                "required": keys,
                "additionalProperties": False,
            }
        },
        "required": ["statements"],
        "additionalProperties": False,
    }


def statement_list(answer: dict) -> dict:
    """Turn the keyed statements of statement_schema into the list parse_statements reads."""
    items = answer.get("statements")
    if not isinstance(items, dict):
        return answer
    return {"statements": [
        {"n": int(key), **value} if key.isdigit() and isinstance(value, dict) else {"n": key}
        for key, value in items.items()
    ]}


UNIT_COVERAGE = {"retained": "present", "omitted": "absent", "incorrect": "incorrect"}


def unit_coverage(entry: dict, units: list[dict], final: dict | None) -> None:
    """Set the coverage score from the source_to_final unit judgments, so that one judge
    decides each unit. The rubric judge gets no units for this task. A proposal or
    revision task keeps the rubric coverage: its scored output is the stored proposal,
    not the final answer."""
    if entry.get("status") != "scored" or "coverage" not in entry["scores"]:
        return
    # The rubric judge returned a placeholder coverage, so a task without unit votes has
    # no coverage score.
    if not final or final["status"] != "scored":
        task_ids = {key: entry[key] for key in ("task_id", "question_id")}
        entry.clear()
        entry.update(
            task_ids, status="unscored", error="The source_to_final unit votes are unscored."
        )
        return
    findings = {u: {"status": UNIT_COVERAGE[f["status"]]} for u, f in final["units"].items()}
    computed = J.coverage_score(units, findings)
    entry["scores"]["coverage"] = computed
    entry["coverage_findings"] = findings
    entry["coverage_source"] = "source_to_final"
    not_present = [u for u, f in findings.items() if f["status"] != "present"]
    entry["rationale"]["coverage"] = (
        f"Code score {computed} from the source_to_final unit votes. "
        f"Units not present: {', '.join(not_present) or 'none'}."
    )
    entry["decision"] = J.decision(entry["scores"], entry["critical_errors"])


def _comparisons(preservation: dict, stage: str) -> dict[str, dict]:
    return {c["task_id"]: c for c in preservation[stage]["comparisons"]}


# A proposal or revision task delivers the stored proposal; its chat answer only points to it.
ESSENTIAL_STAGES = {"proposal": "sources_to_proposal", "revision": "proposal_to_revision"}


def score_essential(case: Case, preservation: dict) -> list[dict]:
    """Essential units retained in each final answer, with the task kind. For a proposal or
    revision task, the final answer is the stored proposal.

    The primary output measure: a unit rate does not fall with the number of units a task
    requires, as the full pass does.
    """
    rows = []
    for task in case.tasks:
        stage = ESSENTIAL_STAGES.get(task["kind"], "source_to_final")
        essential = [
            unit["id"]
            for unit in case.required(stage, task["id"])
            if unit.get("priority") == "essential"
        ]
        if not essential:
            continue
        comparison = _comparisons(preservation, stage).get(task["id"]) or {}
        units = comparison.get("units") if comparison.get("status") == "scored" else None
        rows.append(
            {
                "task_id": task["id"],
                "kind": task["kind"],
                "essential": len(essential),
                "status": comparison.get("status", "not_executed"),
                "retained": (
                    sum(units[u]["status"] == "retained" for u in essential if u in units)
                    if units is not None
                    else None
                ),
            }
        )
    return rows


def score_revision_units(case: Case, preservation: dict) -> dict | None:
    """Split W2-only reference units from retained earlier content.

    W2-only units are required in the revision reference but not the proposal reference. Kept content:
    units the proposal retained that the revision also requires. A unit the proposal missed
    counts in neither, so a proposal error stays in
    sources_to_proposal.
    """
    kinds = {task["kind"]: task["id"] for task in case.tasks}
    if "proposal" not in kinds or "revision" not in kinds:
        return None
    proposal = _comparisons(preservation, "sources_to_proposal").get(kinds["proposal"]) or {}
    revision = _comparisons(preservation, "proposal_to_revision").get(kinds["revision"]) or {}
    first = {u["id"] for u in case.required("sources_to_proposal", kinds["proposal"])}
    second = [u["id"] for u in case.required("proposal_to_revision", kinds["revision"])]
    if proposal.get("status") != "scored" or revision.get("status") != "scored":
        return {"status": "not_scored"}
    kept_before = [
        u for u in second if u in first and proposal["units"][u]["status"] == "retained"
    ]
    changes = [u for u in second if u not in first]

    def retained(units: list[str]) -> int:
        return sum(revision["units"][u]["status"] == "retained" for u in units)

    return {
        "status": "scored",
        "changes": {"count": retained(changes), "total": len(changes)},
        "kept": {"count": retained(kept_before), "total": len(kept_before)},
    }


def score_run(
    run_dir: Path,
    release: Release,
    ask: J.Ask | None,
    identity: dict[str, Any],
    votes: int = 1,
) -> dict:
    manifest = json.loads((run_dir / "run.json").read_text())
    case = release.case(manifest["split"], manifest["scenario_id"])
    if manifest["release_manifest_sha256"] != release.manifest_sha256:
        raise ValueError(f"{run_dir} was run against another reference release.")
    tasks = load_tasks(run_dir, case)
    judged = Judgments(run_dir / "judgments", ask, identity)
    readback_path = run_dir / "ledger-readback.json"
    readback = json.loads(readback_path.read_text()) if readback_path.exists() else []
    records_path = run_dir / "records.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else None
    recaps_path = run_dir / "recaps.json"
    recaps = json.loads(recaps_path.read_text()) if recaps_path.exists() else None
    imaging_path = run_dir / "imaging-results.json"
    imaging_results = json.loads(imaging_path.read_text()) if imaging_path.exists() else []
    condition = manifest.get("condition", "full")
    # The single-agent system consults no specialist, so these measures do not apply.
    consults = condition != "single_agent"
    rubric = release.rubric()
    # Incorporation and every preservation vote ask independent questions; run them together.
    with ThreadPoolExecutor() as pool:
        specialist_job = pool.submit(score_specialists, case, tasks, judged) if consults else None
        votes_jobs = [
            pool.submit(
                score_preservation,
                case,
                tasks,
                Judgments(run_dir / "judgments", ask, vote),
                records,
                recaps,
                consults,
            )
            for vote in vote_identities(identity, votes)
        ]
        specialist = specialist_job.result() if specialist_job else []
        preservation = voted([job.result() for job in votes_jobs])
    quality_identity = {**identity, "source_scope": "cited_sources_v1"}
    quality_judged = Judgments(run_dir / "judgments", ask, quality_identity)
    statement_judges = [
        Judgments(run_dir / "judgments", ask, vote)
        for vote in vote_identities(quality_identity, votes)
    ]
    quality = score_quality(
        case, tasks, quality_judged, rubric, release.question_groups(), preservation, records,
        statement_judges,
    )
    judge = identity if votes == 1 else {**identity, "votes": votes}
    scores = {
        "model": manifest["model"],
        "condition": condition,
        "split": case.split,
        "scenario_id": case.scenario_id,
        "release_manifest_sha256": release.manifest_sha256,
        "run_status": manifest.get("status"),
        "system": manifest.get("git"),
        "judge": judge,
        "quality_source_scope": "cited_sources_v1",
        "tasks": {task_id: record["status"] for task_id, record in tasks.items()},
        "storage": score_storage(case, readback),
        "source_needs": score_source_needs(case, tasks),
        "specialist_results": specialist,
        "specialist_calls": score_calls(case, tasks) if consults else [],
        "imaging": score_imaging(case, tasks, specialist),
        "provenance": score_provenance(
            case, tasks, readback, records, imaging_results, consults
        ),
        "preservation": preservation,
        "quality": quality,
        "essential": score_essential(case, preservation),
        "rq3": score_rq3(case, tasks, records),
        "citations": [
            {"task_id": task_id, **citation_check(record["answer"], record.get("trace", []))}
            for task_id, record in tasks.items()
            if record["status"] == "completed"
        ],
        "revision_units": score_revision_units(case, preservation),
    }
    dump(run_dir / "scores.json", scores)
    return scores


def run_directories(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("run.json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("out", type=Path, help="output directory of src.evaluation.run")
    parser.add_argument("--cached-only", action="store_true", help="use cached judge answers only")
    parser.add_argument(
        "--votes", type=int, default=3, help="judge votes per unit question (majority decides)"
    )
    args = parser.parse_args(argv)
    if args.votes < 1:
        parser.error("--votes must be at least 1.")

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    release = Release.load()
    judge = judge_from_env(require_key=not args.cached_only)
    ask = None if args.cached_only else judge
    for run_dir in run_directories(args.out):
        scores = score_run(run_dir, release, ask, judge.identity(), votes=args.votes)
        print(f"scored {scores['model']} {scores['scenario_id']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
