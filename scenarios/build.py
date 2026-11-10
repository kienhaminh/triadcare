"""Build the conversations, FHIR fixtures, imaging catalog entries, and reference release
of the authored scenario_v4 cases.

A scenario_v4 case (scenarios/cases/<id>.yaml) holds everything in one file: the people,
the record, the imaging studies, the check-in transcripts, the reference units, and the
tasks. This script derives the frozen artifacts of both splits from it and fails on any
inconsistency. The question bank is copied byte for byte from the source release.

Each unit has a priority. An essential unit is a fact the clinician needs to act; a
supporting unit is context. must_state lists the one or two facts a statement of the unit
must carry; the rest of content is optional detail. essential_limit marks an uncertainty
that a statement must keep. Each task has a kind; an update task asks what is new since an
earlier review date, and a decision task asks for a conditional verdict on a
non-prescribing step. A task can set, for itself only, the must_state, the priority, or the
essential_limit of a unit: {id: <unit>, must_state: [...], priority: ..., essential_limit: ...}
in its units list. A unit is essential for one question and context for another.

    poetry run python scenarios/build.py                                  # all cases
    poetry run python scenarios/build.py --check scenarios/cases/<id>.yaml  # one case
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
from datetime import date, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"
RELEASES = ROOT / "eval" / "reference_units" / "releases"
CATALOG = SCENARIOS / "fixtures" / "imaging-workflow.json"
DLBS_ROOT = ROOT / "data" / "dlbs"
CATALOG_INPUT = "scenarios/fixtures/ten_day_v2/imaging-workflow.json"  # key in the release inputs

BANK = RELEASES.parent / "held_out.bank.json"  # the frozen rubric and question bank
TARGET_RELEASE = "triadcare"
FROZEN_AT = "2026-11-08T15:06:06.866300+00:00"  # release timestamp; part of the manifest digest
REVIEWED_BY = "AI-assisted dialogue revision against guidelines; no independent clinician adjudication"
MAX_TURN_LENGTH = 16_000
# Reporter text must not contain authoring language or study-day labels.
FORBIDDEN = re.compile(
    r"\bday[- ]?\d+\b|synthetic|omitted from|first reported today|"
    r"I do not know about that from today",
    re.IGNORECASE,
)
ALLOWED_TRANSFORMATIONS = (
    "Equivalent paraphrase is allowed. Preserve the stated reporter, temporal scope, "
    "uncertainty and source relationship; do not add clinical inference."
)
QUESTION_IDS = ("T1", "C1", "C2", "C3", "C4", "S1", "S2", "F1", "W1", "W2")
# Fixed kinds; C2 is always an update; C3..S2 take one of OPEN_KINDS.
FIXED_KINDS = {
    "T1": "decision", "C1": "brief", "C2": "update", "F1": "followup",
    "W1": "proposal", "W2": "revision",
}
OPEN_KINDS = ("question", "reconcile", "not_established", "brief", "decision")
PRIORITIES = ("essential", "supporting")
SPLIT_NAMES = {"dev": "development", "held_out": "test"}
SYNTHETIC_TAG = {
    "system": "https://triadcare.local/fhir/CodeSystem/data-classification",
    "code": "synthetic-scenario",
}
LOINC = {"mmse": ("72106-8", "MMSE total score"), "moca": ("72172-0", "MoCA total score")}
SCALES = {
    "cdr_global": "Clinical Dementia Rating global score",
    "cdr_sb": "Clinical Dementia Rating sum of boxes",
    "faq": "Functional Activities Questionnaire total score",
}


class BuildError(ValueError):
    pass


class CaseLoader(yaml.SafeLoader):
    """Reject duplicate fields instead of silently replacing authored evidence."""


def unique_mapping(loader: CaseLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        require(key not in seen, f"Duplicate YAML field {key!r} at line {key_node.start_mark.line + 1}")
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


CaseLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def load_case(path: Path) -> dict:
    """Read one authored case. Every date must be a quoted string in the YAML."""
    case = yaml.load(path.read_text(), Loader=CaseLoader)
    require(case.get("schema") == "scenario_v4", f"{path.name} is not a scenario_v4 case")
    require(case.get("split") in SPLIT_NAMES, f"{path.name}: split is dev or held_out")

    def check(value) -> None:
        require(not isinstance(value, date), f"{path.name}: quote the date {value}")
        for item in (
            value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
        ):
            check(item)

    check(case)
    return case


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


def day_dates(case: dict) -> dict[int, str]:
    start = date.fromisoformat(case["start_date"])
    return {day: (start + timedelta(days=day - 1)).isoformat() for day in range(1, 11)}


def build_conversation(case: dict) -> dict:
    dates = day_dates(case)
    days = []
    require([d["day"] for d in case["days"]] == list(range(1, 11)), "days must be 1..10")
    for item in case["days"]:
        check_ins = {}
        for role in ("patient", "caregiver"):
            transcript, sequence = [], 0
            for turn, pair in enumerate(item[role], start=1):
                require(1 <= len(pair) <= 2, f"day {item['day']} {role}: a turn has 1 or 2 lines")
                for speaker, text in zip(("care_agent", f"{role}_sim"), pair, strict=False):
                    require(bool(text.strip()) and len(text) < MAX_TURN_LENGTH, "bad turn text")
                    if speaker != "care_agent":
                        found = FORBIDDEN.search(text)
                        require(not found, f"day {item['day']} {role}: forbidden {found!r}")
                    sequence += 1
                    transcript.append(
                        {
                            "sequence": sequence,
                            "day": item["day"],
                            "turn": turn,
                            "speaker": speaker,
                            "text": text.strip(),
                        }
                    )
            check_ins[role] = {
                "check_in_id": f"{case['id']}:day-{item['day']}:{role}",
                "transcript": transcript,
            }
        days.append(
            {"day": item["day"], "check_in_date": dates[item["day"]], "check_ins": check_ins}
        )
    return {
        "schema": "scenario_conversation",
        "scenario_id": case["id"],
        "patient_id": case["patient"]["id"],
        "patient_name": case["patient"]["name"],
        "caregiver_name": case["caregiver"]["name"],
        "split": case["split"],
        "title": case["title"],
        "protocol": "check-in",
        "model": "scripted_fixture",
        "sim_model": "scripted_fixture",
        "authorship": "AI-assisted script editing; not a live model simulation",
        "dialogue_version": "clinical_dialogue_v5",
        "artifact_version": "scenario_v4",
        "days": days,
    }


def check_in_texts(conversation: dict, scenario: str) -> dict[str, str]:
    """The text of each check-in as the release stores it: every turn, one per line."""
    return {
        f"raw-check-in:{scenario}:day-{day['day']}:{role}": "\n".join(
            turn["text"] for turn in check_in["transcript"]
        )
        for day in conversation["days"]
        for role, check_in in day["check_ins"].items()
    }


def reporter_texts(conversation: dict, scenario: str) -> dict[str, str]:
    return {
        f"raw-check-in:{scenario}:day-{day['day']}:{role}": "\n".join(
            turn["text"] for turn in check_in["transcript"] if turn["speaker"] != "care_agent"
        )
        for day in conversation["days"]
        for role, check_in in day["check_ins"].items()
    }


# ---------------------------------------------------------------------------
# FHIR fixture
# ---------------------------------------------------------------------------


def resource_id(case: dict, local: str) -> str:
    return f"{case['patient']['id'].lower().replace('-', '')}-{local}"


def _base(case: dict, kind: str, local: str) -> dict:
    return {
        "resourceType": kind,
        "id": resource_id(case, local),
        "meta": {"tag": [SYNTHETIC_TAG]},
        "subject": {"reference": f"Patient/{case['patient']['id']}"},
    }


def _text(value: str) -> dict:
    return {"text": value}


def patient_resource(case: dict) -> dict:
    person = case["patient"]
    return {
        "resourceType": "Patient",
        "id": person["id"],
        "meta": {"tag": [SYNTHETIC_TAG]},
        "active": True,
        "name": [{"use": "usual", "text": person["name"]}],
        "gender": person["gender"],
        "birthDate": str(person["birth_year"]),
    }


def baseline_resources(case: dict) -> list[dict]:
    baseline = case["baseline"]
    condition = _base(case, "Condition", "condition-cognitive")
    condition.update(
        clinicalStatus={"coding": [{"code": "active"}], "text": "active"},
        verificationStatus={"coding": [{"code": "confirmed"}], "text": "confirmed"},
        code=_text(baseline["diagnosis"]),
        recordedDate=baseline["date"],
    )
    resources = [condition]
    for key, (code, display) in LOINC.items():
        if key in baseline:
            item = _base(case, "Observation", f"observation-{key}")
            item.update(
                status="final",
                category=[_text("cognitive-baseline")],
                code={"coding": [{"system": "http://loinc.org", "code": code}], "text": display},
                effectiveDateTime=baseline["date"],
                valueQuantity={"value": baseline[key], "unit": "score"},
            )
            resources.append(item)
    for key, display in SCALES.items():
        if key in baseline:
            item = _base(case, "Observation", f"observation-{key.replace('_', '-')}")
            item.update(
                status="final",
                category=[_text("cognitive-baseline")],
                code=_text(display),
                effectiveDateTime=baseline["date"],
                valueQuantity={"value": baseline[key], "unit": "score"},
            )
            resources.append(item)
    return resources


def record_resource(case: dict, item: dict) -> dict:
    kind = item["type"]
    resource = _base(case, kind, item["id"])
    note = [{"text": item["note"]}] if item.get("note") else None
    if kind == "MedicationRequest":
        resource.update(
            status=item["status"],
            intent="order",
            medicationCodeableConcept=_text(item["medication"]),
            authoredOn=item["date"],
            dosageInstruction=[_text(item["dosage"])],
        )
        if item.get("status_reason"):
            resource["statusReason"] = _text(item["status_reason"])
        if note:
            resource["note"] = note
    elif kind == "Observation":
        resource.update(
            status="final",
            category=[_text(item["category"])],
            code=_text(item["code"]),
            effectiveDateTime=item["date"],
        )
        if "quantity" in item:
            resource["valueQuantity"] = {"value": item["quantity"], "unit": item["unit"]}
        else:
            resource["valueString"] = item["value"]
        if note:
            resource["note"] = note
    elif kind == "Encounter":
        # FHIR date search reads a period without an end as ongoing, so a closed visit
        # with only a start matches every later date range. A closed visit ends on its day.
        period = {"start": item["date"]}
        if item.get("end") or item["status"] in {"finished", "cancelled"}:
            period["end"] = item.get("end") or item["date"]
        resource.update(
            status=item["status"],
            **{
                "class": {
                    "code": item.get("class", "AMB"),
                    "display": item.get("class_display", "ambulatory"),
                }
            },
            type=[_text(item["kind"])],
            period=period,
            reasonCode=[_text(item["reason"])],
        )
    elif kind == "DocumentReference":
        resource.pop("subject")
        resource.update(
            status="current",
            docStatus=item.get("doc_status", "final"),
            type=_text(item["title"]),
            subject={"reference": f"Patient/{case['patient']['id']}"},
            date=item["date"],
            author=[{"display": item.get("author", "Clinic record")}],
            description=item["text"],
            content=[
                {
                    "attachment": {
                        "contentType": "text/plain",
                        "title": item["title"],
                        "creation": item["date"],
                        "data": base64.b64encode(item["text"].encode()).decode(),
                    }
                }
            ],
        )
    elif kind == "Condition":
        resource.update(
            clinicalStatus=_text(item["status"]),
            code=_text(item["code"]),
            recordedDate=item["date"],
        )
        if note:
            resource["note"] = note
    else:
        raise BuildError(f"Unsupported record type {kind}")
    return resource


def imaging_resource(case: dict, index: int, study: dict) -> dict:
    resource = _base(case, "ImagingStudy", f"imaging-{index + 1:02d}")
    resource.update(
        status="available",
        identifier=[
            {"system": "https://triadcare.local/imaging-study", "value": study["study_id"]}
        ],
        started=study["date"],
        modality=[{"code": study["modality"].upper()}],
        description=study["description"],
    )
    return resource


def background_resources(case: dict) -> list[dict]:
    """Synthea background history from synthea_background.py, or none."""
    path = SCENARIOS / "fixtures" / "background" / f"{case['id']}.json"
    return json.loads(path.read_text())["resources"] if path.exists() else []


def build_fixture(case: dict) -> dict:
    resources = [patient_resource(case), *baseline_resources(case)]
    resources += [record_resource(case, item) for item in case.get("record", [])]
    resources += [imaging_resource(case, i, s) for i, s in enumerate(case.get("imaging", []))]
    resources += background_resources(case)
    ids = [f"{r['resourceType']}/{r['id']}" for r in resources]
    require(len(ids) == len(set(ids)), "duplicate FHIR resource ID")
    targets = []
    if case.get("record_target"):
        target = f"DocumentReference/{resource_id(case, case['record_target'])}"
        require(target in ids, "record_target names no DocumentReference")
        resource = resources[ids.index(target)]
        note = base64.b64decode(resource["content"][0]["attachment"]["data"]).decode()
        require(note == resource["description"], "record_target note does not read back")
        targets.append(
            {"resource": target, "source_id": f"{case['id']}:note-target", "version": "1"}
        )
    return {
        "schema": "scenario_fhir_fixture",
        "schema_version": 3,
        "scenario_id": case["id"],
        "patient_id": case["patient"]["id"],
        "split": case["split"],
        "synthetic": True,
        "no_real_patient_data": True,
        "agent_visible": {"resources": resources},
        "record_targets": targets,
    }


# ---------------------------------------------------------------------------
# Imaging catalog
# ---------------------------------------------------------------------------


def series_validation(series_id: str) -> dict:
    """Hash the files of one DLBS series in data/dlbs (scenarios/fetch_dlbs.py downloads them)."""
    series_dir = DLBS_ROOT / series_id
    files = sorted(p for p in series_dir.glob("*") if p.is_file()) if series_dir.is_dir() else []
    require(files, f"missing DLBS series {series_id}; run scenarios/fetch_dlbs.py")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return {"digest_algorithm": "sha256", "artifact_sha256": digest.hexdigest(), "file_count": len(files)}


def catalog_entry(case: dict) -> dict:
    studies = [
        {
            "modality": study["modality"],
            "image_ids": [study["study_id"]],
            "source_series": study["source_series"],
            "artifact_validation": series_validation(study["source_series"]),
        }
        for study in case.get("imaging", [])
    ]
    return {
        "case_id": case["id"],
        "split": case["split"],
        "patient_id": case["patient"]["id"],
        "studies": studies,
        "source_input_integrity_checked": True,
    }


CATALOG_HEADER = {
    "schema": "scenario_imaging_workflow_v2",
    "version": "scenario_v4",
    "evaluation_scope": "workflow_capability",
    "inference_status": "not_run",
    "diagnostic_accuracy_evaluated": False,
    "coverage": {},
    "source": {
        "dataset": "Dallas Lifespan Brain Study (OpenNeuro ds004856 v1.3.0)",
        "license": "CC0",
        "statement": (
            "Each synthetic case uses the wave-1 T1-weighted series of a different DLBS "
            "subject, matched by sex and approximate age."
        ),
        "source_subject_count": 11,
    },
    "synthetic_profile": {
        "is_real_patient": False,
        "warning": (
            "Case and patient identifiers are synthetic bindings; this fixture is not a "
            "real patient record."
        ),
    },
}


def update_catalog(cases: list[dict]) -> dict:
    catalog = {**CATALOG_HEADER, "cases": []}
    entries = [catalog_entry(case) for case in cases]
    entries = [e for e in entries if e["studies"]]
    study_ids = [i for e in entries for s in e["studies"] for i in s["image_ids"]]
    require(len(study_ids) == len(set(study_ids)), "duplicate study ID in the imaging catalog")
    patients = [e["patient_id"] for e in entries]
    require(len(patients) == len(set(patients)), "duplicate patient ID in the imaging catalog")
    catalog["cases"] = entries
    catalog["version"] = "scenario_v4"
    count = len(study_ids)
    catalog["coverage"] = {
        "expected_case_count": len(entries),
        "covered_case_count": len(entries),
        "expected_study_count": count,
        "covered_study_count": count,
        "all_cases_covered": True,
    }
    return catalog


# ---------------------------------------------------------------------------
# Reference release
# ---------------------------------------------------------------------------


def source_kind(source_id: str) -> str:
    return source_id.split(":", 1)[0]


def build_sources(case: dict, conversation: dict, conv_hash: str, fixture: dict, fhir_hash: str):
    """Reference sources and a map from authored anchor names to source IDs."""
    scenario, patient = case["id"], case["patient"]["id"]
    dates = day_dates(case)
    sources, names = [], {}
    for source_id, text in check_in_texts(conversation, scenario).items():
        day, role = source_id.split(":")[2:4]
        names[f"{day}:{role}"] = source_id
        sources.append(
            {
                "id": source_id,
                "text": text,
                "version": conv_hash,
                "patient_id": patient,
                "reporter": role,
                "reported_at": dates[int(day.removeprefix("day-"))],
            }
        )
    for resource in fixture["agent_visible"]["resources"]:
        key = f"{resource['resourceType']}/{resource['id']}"
        names[key] = f"fhir:{key}"
        sources.append(
            {
                "id": f"fhir:{key}",
                "text": json.dumps(resource, sort_keys=True),
                "version": fhir_hash,
                "patient_id": patient,
                "reporter": "FHIR record",
                # The seeded record is available at baseline, including planned events.
                "reported_at": dates[1],
            }
        )
    for index, study in enumerate(case.get("imaging", [])):
        source_id = f"imaging-fixture:{scenario}:{index}"
        names[f"imaging:{study['study_id']}"] = source_id
        text = json.dumps(
            {"image_ids": [study["study_id"]], "modality": study["modality"]}, sort_keys=True
        )
        sources.append(
            {
                "id": source_id,
                "text": text,
                "version": sha256_file(CATALOG),
                "patient_id": patient,
                "reporter": "imaging catalog",
                "reported_at": dates[1],
            }
        )
    return sources, names


def build_units(
    case: dict, names: dict, sources: list[dict], reporter: dict[str, str]
) -> list[dict]:
    texts = {s["id"]: s["text"] for s in sources}
    units = []
    for item in case["units"]:
        anchors = []
        for anchor in item["anchors"]:
            source_id = names.get(anchor["source"]) or names.get(
                f"{anchor['source'].split('/')[0]}/{resource_id(case, anchor['source'].split('/', 1)[-1])}"
            )
            require(source_id is not None, f"{item['id']}: unknown source {anchor['source']}")
            quote = anchor["quote"]
            body = (
                reporter[source_id]
                if source_kind(source_id) == "raw-check-in"
                else texts[source_id]
            )
            if source_kind(source_id) == "fhir":
                body = json.dumps(json.loads(body), ensure_ascii=False)
            require(normalize(quote) in normalize(body), f"{item['id']}: quote not in {source_id}")
            anchors.append({"source_id": source_id, "quote": quote})
        must_state = item["must_state"]
        require(
            item["priority"] in PRIORITIES, f"{item['id']}: priority is essential or supporting"
        )
        require(
            isinstance(must_state, list) and 1 <= len(must_state) <= 2 and all(must_state),
            f"{item['id']}: must_state lists one or two facts",
        )
        units.append(
            {
                "id": f"{case['id']}:{item['id']}",
                "content": item["content"],
                "anchors": anchors,
                "reporter": item["reporter"],
                "subject": case["patient"]["id"],
                "event_time": item["event_time"],
                "reported_at": item["reported_at"],
                "uncertainty": item["uncertainty"],
                "priority": item["priority"],
                "must_state": must_state,
                "essential_limit": bool(item.get("essential_limit", False)),
                "correction_of": [f"{case['id']}:{u}" for u in item.get("correction_of", [])],
                "allowed_transformations": ALLOWED_TRANSFORMATIONS,
            }
        )
    ids = [u["id"] for u in units]
    require(len(ids) == len(set(ids)), "duplicate unit ID")
    return units


def task_as_of(case: dict, task: dict) -> str:
    return day_dates(case)[task["as_of_day"]]


def build_tasks(case: dict) -> list[dict]:
    scenario, tasks = case["id"], []
    for task in case["tasks"]:
        qid = task["qid"]
        depends = {"F1": "C1", "W1": "F1", "W2": "W1"}.get(qid)
        session = f"{scenario}:C1" if qid in {"C1", "F1", "W1", "W2"} else f"{scenario}:{qid}"
        row = {
            "id": f"{scenario}:{qid}",
            "scenario_id": scenario,
            "question_id": qid,
            "as_of": task_as_of(case, task),
            "prompt": task["prompt"].strip(),
            "depends_on": f"{scenario}:{depends}" if depends else None,
            "session_id": session,
            "kind": FIXED_KINDS.get(qid) or task["kind"],
        }
        require(
            qid in FIXED_KINDS or task["kind"] in OPEN_KINDS,
            f"{row['id']}: kind is one of {OPEN_KINDS}",
        )
        if qid == "C2":
            require(
                1 <= task["since_day"] < task["as_of_day"], f"{row['id']}: since_day before as_of"
            )
            row["since"] = day_dates(case)[task["since_day"]]
        criteria = task.get("clinical_criteria")
        if criteria is not None:
            require(
                isinstance(criteria, dict)
                and set(criteria) == {"required_actions", "forbidden_actions", "rationale", "sources"},
                f"{row['id']}: invalid clinical_criteria fields",
            )
            for field in ("required_actions", "forbidden_actions", "sources"):
                require(
                    isinstance(criteria[field], list) and bool(criteria[field])
                    and all(isinstance(s, str) and s.strip() for s in criteria[field]),
                    f"{row['id']}: {field} must contain nonempty strings",
                )
            require(
                isinstance(criteria["rationale"], str) and bool(criteria["rationale"].strip())
                and all(s.startswith("https://") for s in criteria["sources"]),
                f"{row['id']}: clinical criteria need rationale and HTTPS sources",
            )
            row["clinical_criteria"] = criteria
        require(qid != "T1" or criteria is not None, "T1 needs clinical safety criteria")
        if qid == "W2":
            row["revision_policy"] = {
                "primary_prompt": row["prompt"],
                "fallback_prompt": task["fallback_prompt"].strip(),
                "selection_rule": (
                    "Use primary when it requires a source-supported material change; otherwise use "
                    "the single predefined payload-scope fallback. If neither requires a change, mark "
                    "revision_not_exercised and stop before approval."
                ),
                "primary_and_fallback_are_alternatives": True,
                "scheduled_W2_responses": 1,
            }
        tasks.append(row)
    ids = [t["question_id"] for t in tasks]
    require("C1" in ids and len(ids) == len(set(ids)), "tasks include C1 and have unique IDs")
    require(set(ids) <= set(QUESTION_IDS), f"unknown question ID in {ids}")
    require(
        [q for q in QUESTION_IDS if q in ids] == ids, "tasks follow T1, C1..S2, F1, W1, W2"
    )
    for task in tasks:
        if task["question_id"] == "T1":
            require(task["as_of"] < tasks[ids.index("C1")]["as_of"], "T1 precedes final review")
        if task["question_id"] in {"F1", "W1", "W2"}:
            require(
                task["as_of"] == tasks[ids.index("C1")]["as_of"],
                f"{task['id']}: as_of differs from C1",
            )
    for task in tasks:
        require(
            not task["depends_on"] or task["depends_on"] in {t["id"] for t in tasks},
            "bad dependency",
        )
    return tasks


def unit_stages(unit: dict, qid: str) -> list[str]:
    """Stages of one required unit. W1 and W2 replace synthesis with the proposal or revision."""
    kinds = {source_kind(a["source_id"]) for a in unit["anchors"]}
    output = {"W1": "sources_to_proposal", "W2": "proposal_to_revision"}.get(
        qid, "sources_to_synthesis"
    )
    if kinds == {"raw-check-in"}:
        return ["ledger_to_context", output, "source_to_final"]
    if "raw-check-in" not in kinds:  # one specialist owns every anchor
        owners = {"ehr" if k == "fhir" else "radiologist" for k in kinds}
        require(len(owners) == 1, f"{unit['id']}: anchors span two specialists")
        return [
            "source_requirement",
            "source_to_specialist",
            "specialist_integration",
            output,
            "source_to_final",
        ]
    return ["source_requirement", output, "source_to_final"]


TASK_OVERRIDES = ("must_state", "priority", "essential_limit")


def task_units(authored: dict) -> list[tuple[str, dict]]:
    """The units of one authored task, each with the fields that the task sets for itself."""
    entries = []
    for entry in authored["units"]:
        if isinstance(entry, str):
            entries.append((entry, {}))
            continue
        name = f"{authored['qid']}: {entry['id']}"
        require(set(entry) - {"id"} <= set(TASK_OVERRIDES), f"{name} sets an unknown field")
        overrides = {key: entry[key] for key in TASK_OVERRIDES if key in entry}
        require(overrides, f"{name} sets no field; write the unit ID alone")
        focus = overrides.get("must_state", ["x"])
        require(
            isinstance(focus, list) and 1 <= len(focus) <= 2 and all(focus),
            f"{name} narrows must_state to one or two facts",
        )
        require(overrides.get("priority", "essential") in PRIORITIES, f"{name}: bad priority")
        require(
            isinstance(overrides.get("essential_limit", True), bool),
            f"{name}: essential_limit is true or false",
        )
        entries.append((entry["id"], overrides))
    return entries


def build_requirements(case: dict, tasks: list[dict], units: list[dict], sources: list[dict]):
    by_id = {u["id"]: u for u in units}
    reported = {s["id"]: s["reported_at"] for s in sources}
    rows = []
    first = next(task for task in tasks if task["question_id"] == "C1")
    derived = []
    for authored, task in zip(case["tasks"], tasks, strict=True):
        require(
            any(
                {**by_id.get(f"{case['id']}:{u}", {}), **overrides}.get("priority") == "essential"
                for u, overrides in task_units(authored)
            )
            or task["kind"] == "not_established",
            f"{task['id']}: a task needs an essential unit",
        )
        for local, overrides in task_units(authored):
            unit = by_id.get(f"{case['id']}:{local}")
            require(unit is not None, f"{task['id']}: unknown unit {local}")
            require(unit["reported_at"] <= task["as_of"], f"{task['id']}: {local} is reported after as_of")
            for anchor in unit["anchors"]:
                day = reported.get(anchor["source_id"])
                require(
                    day is None or day <= task["as_of"], f"{task['id']}: {local} is after as_of"
                )
            if "since" in task:
                days = [reported.get(a["source_id"]) for a in unit["anchors"]]
                require(
                    any(day and day > task["since"] for day in days),
                    f"{task['id']}: {local} has no report after the earlier review",
                )
            for stage in unit_stages(unit, task["question_id"]):
                rows.append(
                    {
                        "task_id": task["id"],
                        "stage": stage,
                        "unit_id": unit["id"],
                        "required": True,
                        "exclusion_reason": "",
                        **overrides,
                    }
                )
            if (
                {source_kind(a["source_id"]) for a in unit["anchors"]} == {"raw-check-in"}
                and all(reported[a["source_id"]] <= first["as_of"] for a in unit["anchors"])
                and unit["id"] not in derived
            ):
                derived.append(unit["id"])
    rows = [
        {
            "task_id": first["id"],
            "stage": "conversation_to_derived",
            "unit_id": unit_id,
            "required": True,
            "exclusion_reason": "",
        }
        for unit_id in derived
    ] + rows
    used = {row["unit_id"] for row in rows}
    unused = [u["id"] for u in units if u["id"] not in used]
    require(not unused, f"units with no task: {unused}")
    return rows


def build_opportunities(case: dict, tasks: list[dict]) -> list[dict]:
    scenario = case["id"]
    by_qid = {t["question_id"]: t["id"] for t in tasks}
    items = [
        {"metric": "unnecessary_calls", "id": f"unnecessary_calls:{t['id']}", "task_id": t["id"]}
        for t in tasks
    ]
    items += [
        {"metric": "storage", "id": f"storage:{scenario}:day-{day}:{role}", "task_id": None}
        for day in range(1, 11)
        for role in ("patient", "caregiver")
    ]
    for item in case.get("imaging_opportunities", []):
        row = {
            "metric": item["metric"],
            "id": f"{item['metric']}:{scenario}:{item['qid']}",
            "task_id": by_qid[item["qid"]],
            "reason": item["reason"],
        }
        if item["metric"] == "imaging_rerun":
            require(
                re.search(r"CASE-[A-Z0-9-]+", item["reason"]) is not None,
                "rerun reason names no study",
            )
        items.append(row)
    if "W1" in by_qid:
        source = f"DocumentReference/{resource_id(case, case['source_open'])}"
        items.append(
            {
                "metric": "source_open",
                "id": f"source_open:{scenario}",
                "task_id": by_qid["W1"],
                "ui_contract": {
                    "source_reference": source,
                    "source_version": "captured current FHIR version before the W1 review",
                    "proposal": "sole pending proposal persisted by W1, bound by ID and version",
                    "actions": [
                        "open_source_citation",
                        "inspect_source_identity_and_version",
                        "open_review_confirmation",
                        "cancel_confirmation",
                    ],
                    "timing": "after W1 persistence and before rejection or approval",
                },
            }
        )
        items.append(
            {
                "metric": "caregiver_denial",
                "id": f"caregiver_denial:{scenario}",
                "task_id": by_qid["W1"],
            }
        )
        for metric in (
            "rejection",
            "revision_link",
            "revision_quality",
            "revision_safety",
            "content_gate",
            "approval_binding",
            "fhir_write",
            "full_workflow",
            "audit",
        ):
            items.append({"metric": metric, "id": f"{metric}:{scenario}", "task_id": by_qid["W2"]})
    return items


def build_controls(case: dict, tasks: list[dict]) -> list[dict]:
    scenario = case["id"]
    controls = [
        {"id": f"{scenario}:storage", "task_id": tasks[0]["id"], "kind": "storage_integrity"}
    ]
    if any(t["question_id"] == "W2" for t in tasks):
        for kind in ("revision_eligibility", "authorization", "write_readback", "audit_links"):
            controls.append({"id": f"{scenario}:{kind}", "task_id": f"{scenario}:W2", "kind": kind})
    return controls


def build_release_case(case: dict, conv_path: Path, fhir_path: Path, frozen_at: str):
    split = SPLIT_NAMES[case["split"]]
    conversation = json.loads(conv_path.read_text())
    fixture = json.loads(fhir_path.read_text())
    conv_hash, fhir_hash = sha256_file(conv_path), sha256_file(fhir_path)
    sources, names = build_sources(case, conversation, conv_hash, fixture, fhir_hash)
    units = build_units(case, names, sources, reporter_texts(conversation, case["id"]))
    tasks = build_tasks(case)
    if "W1" in {t["question_id"] for t in tasks}:
        require(bool(fixture["record_targets"]), "W1 needs a record_target")
        require(case.get("source_open"), "W1 needs source_open")
    reference = {
        "schema_version": 1,
        "split": split,
        "patient_id": case["patient"]["id"],
        "tasks": [t["id"] for t in tasks],
        "sources": sources,
        "units": units,
        "requirements": build_requirements(case, tasks, units, sources),
        "controls": build_controls(case, tasks),
        "reviewed_by": REVIEWED_BY,
        "frozen_at": frozen_at,
    }
    plan = {
        "schema_version": 1,
        "split": split,
        "tasks": tasks,
        "opportunities": build_opportunities(case, tasks),
        "clinical_accuracy_evaluated": False,
        "independent_clinical_reasoning_evaluated": False,
        "scripted_care_advice_may_be_visible": True,
        "clinical_safety_criteria": "authored checklist; scored separately from preservation",
        "independent_clinician_adjudication": False,
        "source_open_scope": "scripted_ui",
        "transport_attempts": 1,
        "selection_rule": "first_completed_output",
    }
    rel_conv = conv_path.relative_to(ROOT).as_posix()
    rel_fhir = fhir_path.relative_to(ROOT).as_posix()
    origins = {
        "sources": {
            s["id"]: {"path": rel_conv, "sha256": conv_hash, "pointer": s["id"]}
            if source_kind(s["id"]) == "raw-check-in"
            else {"path": rel_fhir, "sha256": fhir_hash, "resource": s["id"].removeprefix("fhir:")}
            if source_kind(s["id"]) == "fhir"
            else {"path": CATALOG.relative_to(ROOT).as_posix(), "sha256": s["version"]}
            for s in sources
        },
        "authored_source": f"scenarios/cases/{case['id']}.yaml",
        "test_outputs_used": False,
    }
    return reference, plan, origins


def write_release(
    cases: list[dict], case_paths: dict[str, tuple[Path, Path]], frozen_at: str
) -> Path:
    target = RELEASES / TARGET_RELEASE
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copy2(BANK, target / "held_out.bank.json")
    inputs = {CATALOG_INPUT: sha256_file(CATALOG)}
    scenarios = []
    for case in sorted(cases, key=lambda c: c["split"]):  # dev before held_out
        conv_path, fhir_path = case_paths[case["id"]]
        reference, plan, origins = build_release_case(case, conv_path, fhir_path, frozen_at)
        split = SPLIT_NAMES[case["split"]]
        stem = target / split / case["id"]
        write_json(Path(f"{stem}.reference.json"), reference)
        write_json(Path(f"{stem}.plan.json"), plan)
        write_json(Path(f"{stem}.origins.json"), origins)
        for path in (conv_path, fhir_path, SCENARIOS / "cases" / f"{case['id']}.yaml"):
            inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        scenarios.append(
            {
                "scenario_id": case["id"],
                "split": split,
                "patient_id": case["patient"]["id"],
                "tasks": len(plan["tasks"]),
                "reference_hash": sha256_file(Path(f"{stem}.reference.json")),
                "plan_hash": sha256_file(Path(f"{stem}.plan.json")),
            }
        )
    files = {
        path.relative_to(target).as_posix(): sha256_file(path)
        for path in sorted(target.rglob("*.json"))
    }
    manifest = {
        "condition": "manuscript_source_grounded_v1",
        "frozen_at": frozen_at,
        "inputs": inputs,
        "scenarios": scenarios,
        "splits": ["development", "test"],
        "files": files,
        "test_outputs_used": False,
        "review": "Both splits built from scenario_v4 cases; question bank copied unchanged.",
        "derived_from": {"bank": "eval/reference_units/held_out.bank.json", "cases": "scenarios/build.py"},
    }
    write_json(target / "manifest.json", manifest)
    write_json(
        RELEASES.parent / "CURRENT.json",
        {
            "release": f"releases/{TARGET_RELEASE}",
            "manifest_sha256": sha256_file(target / "manifest.json"),
        },
    )
    return target


def write_case_files(case: dict) -> tuple[Path, Path]:
    conv_path = SCENARIOS / "generated" / "conversations" / case["split"] / f"{case['id']}.json"
    fhir_path = SCENARIOS / "fixtures" / "fhir" / case["split"] / f"{case['id']}.json"
    write_json(conv_path, build_conversation(case))
    write_json(fhir_path, build_fixture(case))
    return conv_path, fhir_path


def check(path: Path) -> int:
    """Build one case without touching the catalog or the release, and report its size."""
    case = load_case(path)
    require(not case["patient"]["id"].startswith("P-0"), "P-0xx patient IDs belong to old cases")
    catalog_entry(case)
    conv_path, fhir_path = write_case_files(case)
    reference, plan, _ = build_release_case(case, conv_path, fhir_path, "check")
    stages: dict[str, int] = {}
    for row in reference["requirements"]:
        stages[row["stage"]] = stages.get(row["stage"], 0) + 1
    print(f"{case['id']}: {len(reference['units'])} units, {len(plan['tasks'])} tasks, {stages}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", type=Path, help="validate one case YAML only")
    args = parser.parse_args()
    if args.check:
        return check(args.check.resolve())
    manifest = yaml.safe_load((SCENARIOS / "manifest.yaml").read_text())
    cases = []
    for entry in manifest["cases"]:
        case = load_case(SCENARIOS / entry["file"])
        require(case["id"] == entry["id"], f"{entry['file']}: id differs from the manifest")
        require(
            case["split"] == entry["split"], f"{entry['file']}: split differs from the manifest"
        )
        cases.append(case)
    patients = [c["patient"]["id"] for c in cases]
    require(len(patients) == len(set(patients)), "duplicate patient ID")
    write_json(CATALOG, update_catalog(cases))
    paths = {case["id"]: write_case_files(case) for case in cases}
    target = write_release(cases, paths, FROZEN_AT)
    print(f"Built {len(cases)} cases into {target.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
