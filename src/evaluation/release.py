"""Load the frozen reference release and refuse to use it when any file changed."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any

from src.fhir.projection import project_resource
from src.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = ROOT / "eval" / "reference_units"

# Reference split names and the corpus directories that hold their inputs.
SPLITS = {"development": "dev", "test": "held_out"}

# Reference stage names are canonical. Rows of the manuscript tables map to them in report.py.
PRESERVATION_STAGES = (
    "conversation_to_derived",
    "ledger_to_context",
    "source_to_specialist",
    "sources_to_proposal",
    "proposal_to_revision",
    "source_to_final",
)
SOURCE_NEED_STAGE = "source_requirement"
SPECIALIST_STAGE = "specialist_integration"
# Development origins do not name specialists; the anchor source type identifies the owner.
SOURCE_OWNERS = {"fhir": "ehr", "imaging-fixture": "radiologist"}


TASK_OVERRIDES = ("must_state", "priority", "essential_limit")
# Tasks that the experiment leaves out: the note-update tasks (proposal and revision).
# They are the last tasks of each scenario, so the earlier tasks do not change.
DROPPED_TASKS = ("W1", "W2")


class ReleaseError(ValueError):
    """The frozen release is missing, changed, or internally inconsistent."""


def _read(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _current_input(frozen_path: str) -> Path:
    # The corpus moved out of its ten_day_v2 directories after the freeze; content is unchanged.
    return ROOT / frozen_path.replace("ten_day_v2/", "")


@dataclass(frozen=True)
class Release:
    directory: Path
    manifest: dict[str, Any]

    @classmethod
    def load(cls) -> Release:
        pointer = _read(REFERENCE_ROOT / "CURRENT.json")
        directory = REFERENCE_ROOT / pointer["release"]
        manifest_path = directory / "manifest.json"
        if sha256_file(manifest_path) != pointer["manifest_sha256"]:
            raise ReleaseError("Release manifest does not match CURRENT.json.")
        manifest = _read(manifest_path)
        for name, digest in manifest["files"].items():
            path = directory / name
            if not path.is_file() or sha256_file(path) != digest:
                raise ReleaseError(f"Frozen release file changed or is missing: {name}")
        return cls(directory, manifest)

    @property
    def manifest_sha256(self) -> str:
        return sha256_file(self.directory / "manifest.json")

    def scenarios(self, split: str) -> list[str]:
        if split not in SPLITS:
            raise ReleaseError(f"Unknown split {split!r}; use one of {sorted(SPLITS)}.")
        return [
            item["scenario_id"] for item in self.manifest["scenarios"] if item["split"] == split
        ]

    def frozen_input(self, relative: str) -> Path:
        """Return a frozen input file after checking its content hash."""
        for frozen_path, digest in self.manifest["inputs"].items():
            path = _current_input(frozen_path)
            if path == ROOT / relative:
                if not path.is_file() or sha256_file(path) != digest:
                    raise ReleaseError(f"Frozen input changed or is missing: {relative}")
                return path
        raise ReleaseError(f"{relative} is not a frozen input of this release.")

    def case(self, split: str, scenario_id: str) -> Case:
        if scenario_id not in self.scenarios(split):
            raise ReleaseError(f"{scenario_id} is not in the {split} split.")
        directory = self.directory / split
        corpus = SPLITS[split]
        case = Case(
            split=split,
            scenario_id=scenario_id,
            reference=_read(directory / f"{scenario_id}.reference.json"),
            plan=_read(directory / f"{scenario_id}.plan.json"),
            origins=_read(directory / f"{scenario_id}.origins.json"),
            conversation_path=self.frozen_input(
                f"scenarios/generated/conversations/{corpus}/{scenario_id}.json"
            ),
            fhir_path=self.frozen_input(f"scenarios/fixtures/fhir/{corpus}/{scenario_id}.json"),
            imaging_catalog_path=self.frozen_input("scenarios/fixtures/imaging-workflow.json"),
        )
        case.validate()
        return _without_dropped_tasks(case)

    def question_groups(self) -> dict[str, str]:
        """Return the frozen task group of each question ID, for example F1 -> follow_up."""
        bank = _read(self.directory / "held_out.bank.json")
        return {question: item["group"] for question, item in bank["questions"].items()}

    def rubric(self) -> dict[str, Any]:
        return _read(self.directory / "held_out.bank.json")["rubric"]


def _dropped(task_id: str | None) -> bool:
    return task_id is not None and task_id.rsplit(":", 1)[-1] in DROPPED_TASKS


def _without_dropped_tasks(case: Case) -> Case:
    """The case without the DROPPED_TASKS, their requirements, and their opportunities."""
    plan = {
        **case.plan,
        "tasks": [t for t in case.plan["tasks"] if not _dropped(t["id"])],
        "opportunities": [o for o in case.plan["opportunities"] if not _dropped(o["task_id"])],
    }
    requirements = [r for r in case.reference["requirements"] if not _dropped(r["task_id"])]
    return replace(case, plan=plan, reference={**case.reference, "requirements": requirements})


@dataclass(frozen=True)
class Case:
    split: str
    scenario_id: str
    reference: dict[str, Any]
    plan: dict[str, Any]
    origins: dict[str, Any]
    conversation_path: Path
    fhir_path: Path
    imaging_catalog_path: Path

    @property
    def patient_id(self) -> str:
        return self.reference["patient_id"]

    @property
    def tasks(self) -> list[dict[str, Any]]:
        """Tasks in their frozen execution order."""
        return self.plan["tasks"]

    @cached_property
    def units(self) -> dict[str, dict[str, Any]]:
        return {unit["id"]: unit for unit in self.reference["units"]}

    @cached_property
    def sources(self) -> dict[str, str]:
        return {source["id"]: source["text"] for source in self.reference["sources"]}

    def required(self, stage: str, task_id: str | None = None) -> list[dict[str, Any]]:
        """Required units of one stage, optionally for one task, in reference order. A row
        can set the unit's must_state, priority, or essential_limit for its task."""
        return [
            {
                **self.units[row["unit_id"]],
                **{key: row[key] for key in TASK_OVERRIDES if key in row},
            }
            for row in self.reference["requirements"]
            if row["required"]
            and row["stage"] == stage
            and (task_id is None or row["task_id"] == task_id)
        ]

    def required_tasks(self, stage: str) -> list[str]:
        """Task IDs that have required units in one stage, in task order."""
        wanted = {
            row["task_id"]
            for row in self.reference["requirements"]
            if row["required"] and row["stage"] == stage
        }
        return [task["id"] for task in self.tasks if task["id"] in wanted]

    def opportunities(self, metric: str) -> list[dict[str, Any]]:
        return [item for item in self.plan["opportunities"] if item["metric"] == metric]

    def specialist_of(self, unit_id: str) -> str | None:
        """The specialist named in origins, else the one that owns the unit's anchor sources."""
        for need in self.origins.get("specialist_needs", []):
            if need["unit_id"] == unit_id:
                return need["specialist"]
        owners = {
            SOURCE_OWNERS.get(anchor["source_id"].split(":", 1)[0])
            for anchor in self.units[unit_id].get("anchors", [])
        }
        return owners.pop() if len(owners) == 1 else None

    def task_sources(self, task_id: str) -> dict[str, str]:
        """Source texts anchored by any required unit of one task."""
        source_ids: list[str] = []
        for row in self.reference["requirements"]:
            if row["required"] and row["task_id"] == task_id:
                for anchor in self.units[row["unit_id"]].get("anchors", []):
                    if anchor["source_id"] not in source_ids:
                        source_ids.append(anchor["source_id"])
        return {source_id: self.sources[source_id] for source_id in source_ids}

    def anchor_sources(self, units: list[dict[str, Any]]) -> dict[str, str]:
        """Judge views of the full sources that the units anchor."""
        source_ids = dict.fromkeys(a["source_id"] for u in units for a in u.get("anchors", []))
        return {sid: self._judge_view(sid, self.sources[sid]) for sid in source_ids}

    @cached_property
    def check_in_dates(self) -> dict[int, str]:
        """Check-in date of each scenario day in the corpus."""
        days = json.loads(self.conversation_path.read_text())["days"]
        return {day["day"]: day["check_in_date"] for day in days}

    @cached_property
    def imaging_catalog(self) -> dict:
        """Project the frozen catalog into the patient-scoped runtime view."""
        catalog = _read(self.imaging_catalog_path)
        studies = [
            {"study_id": study_id, "modality": study["modality"]}
            for case in catalog["cases"]
            if case["patient_id"] == self.patient_id
            for study in case["studies"]
            for study_id in study["image_ids"]
        ]
        return {
            "source": "imaging_catalog",
            "patient_id": self.patient_id,
            "data_context": "synthetic_fixture"
            if catalog.get("synthetic_profile", {}).get("is_real_patient") is False
            else "configured_catalog",
            "studies": studies,
            "availability": "not_checked",
        }

    def judge_sources(self, task_id: str) -> tuple[dict[str, str], list[str]]:
        """Sources available at the task's review date, with the required ones first.

        Unsupported claims are judged against every available source: all record and
        imaging fixtures, and the check-ins recorded on or before the review date.
        Coverage is judged against the required sources, which the second value names.
        """
        as_of = next(task["as_of"] for task in self.tasks if task["id"] == task_id)[:10]
        recorded = {item["id"]: item["reported_at"][:10] for item in self.reference["sources"]}
        required = self.task_sources(task_id)
        available = dict(required)
        for source_id, text in self.sources.items():
            kind, _, rest = source_id.partition(":")
            if source_id in available:
                continue
            if recorded[source_id] > as_of:
                continue
            if kind == "raw-check-in":
                day = int(rest.split(":")[1].removeprefix("day-"))
                if self.check_in_dates[day] > as_of:
                    continue
            available[source_id] = text
        sources = {sid: self._judge_view(sid, text) for sid, text in available.items()}
        for study in self.imaging_catalog["studies"]:
            sources[f"imaging:{study['study_id']}"] = json.dumps(
                {"study_id": study["study_id"], "catalog": self.imaging_catalog}, sort_keys=True
            )
        return sources, list(required)

    def _judge_view(self, source_id: str, text: str) -> str:
        """A check-in as the runtime stores it: recorded date, reporter, and the ledger
        message ID of each line (line K is message seq-K), so the judge can resolve the
        dates and message IDs the agents cite. Other sources are unchanged."""
        kind, _, rest = source_id.partition(":")
        if kind == "fhir":
            return self._fhir_judge_view(text)
        if kind != "raw-check-in":
            return text
        day, role = rest.split(":")[1:3]
        header = (
            f"(recorded {self.check_in_dates[int(day.removeprefix('day-'))]}; reporter: {role})"
        )
        lines = [f"imported:{rest}:seq-{k}: {line}" for k, line in enumerate(text.split("\n"), 1)]
        return "\n".join([header, *lines])

    @staticmethod
    def _fhir_judge_view(text: str) -> str:
        """A record with a note naming the attachment bodies the runtime's EHR projection
        omits, so a claim that such a body was not readable counts as a system limit."""
        resource, _ = json.JSONDecoder().raw_decode(text)
        if resource.get("resourceType") not in {"DocumentReference", "DiagnosticReport"}:
            return text
        omitted = project_resource(resource)["_projection"]["omitted_fields"]
        hidden = [path for path in omitted if path.endswith((".data", ".url"))]
        if not hidden:
            return text
        return (
            f"{text}\n(Runtime view: the EHR tools omit {', '.join(hidden)}. The agents "
            "cannot read this body; an output that says it was not readable reports a "
            "system limitation.)"
        )

    def validate(self) -> None:
        task_ids = {task["id"] for task in self.tasks}
        for row in self.reference["requirements"]:
            if row["unit_id"] not in self.units:
                raise ReleaseError(f"Requirement names unknown unit {row['unit_id']}.")
            if row["task_id"] not in task_ids:
                raise ReleaseError(f"Requirement names unknown task {row['task_id']}.")
        for unit in self.units.values():
            for anchor in unit.get("anchors", []):
                if anchor["source_id"] not in self.sources:
                    raise ReleaseError(f"Unit {unit['id']} anchors unknown source.")
        for need in self.origins.get("specialist_needs", []):
            kinds = {
                a["source_id"].split(":", 1)[0] for a in self.units[need["unit_id"]]["anchors"]
            }
            if {SOURCE_OWNERS.get(kind) for kind in kinds} != {need["specialist"]}:
                raise ReleaseError(f"Specialist of {need['unit_id']} disagrees with its sources.")
        for unit in self.required(SPECIALIST_STAGE):
            if self.specialist_of(unit["id"]) is None:
                raise ReleaseError(f"Specialist unit {unit['id']} names no specialist.")
        for task in self.tasks:
            if task["depends_on"] and task["depends_on"] not in task_ids:
                raise ReleaseError(f"Task {task['id']} depends on an unknown task.")
