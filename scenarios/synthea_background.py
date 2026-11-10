"""Make a long synthetic background history for each scenario patient with Synthea.

The authored case record stays the source of the scenario facts. This script adds older,
unrelated history (laboratory results, vital signs, procedures, visits, resolved conditions,
and finished medication orders) so that each FHIR record has the size of a real
longitudinal record. Each history ends 30 days before the baseline cognitive assessment and
keeps the BACKGROUND_SIZE most recent resources.

Usage: poetry run python scenarios/synthea_background.py
The script writes scenarios/fixtures/background/<case>.json. build.py adds these resources
to the FHIR fixtures. Synthea (Apache-2.0) runs from data/synthea/synthea.jar.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import zlib
from datetime import date, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"
OUT = SCENARIOS / "fixtures" / "background"
SYNTHEA_DIR = ROOT / "data" / "synthea"
JAVA = os.environ.get("JAVA", "java")  # Java 17 or later

GAP_DAYS = 30
YEARS_OF_HISTORY = 20
BACKGROUND_SIZE = 1000
KEEP_TYPES = {"Observation", "Procedure", "Encounter", "Condition", "MedicationRequest"}
# The authored records hold these measures; background values would compete with them.
DROP_LOINC = {
    "29463-7", "39156-5", "85354-9", "8480-6", "8462-4", "8867-4",  # weight, BMI, BP, HR
    "33914-3", "62238-1", "98979-8", "72106-8", "72172-0",  # eGFR, MMSE, MoCA
}
DROP_OBSERVATION_CATEGORIES = {"survey", "social-history"}
# Topics that would change a scenario's clinical or social story.
DROP_TEXT = re.compile(
    r"alzheimer|dementia|cognit|donepezil|memantine|galantamine|rivastigmine|mmse|moca"
    r"|abuse|violence|overdose|opioid|drug|alcohol|substance|pregnan|miscarr|labor force"
    r"|employ|educat|housing|income|insurance|prison|refugee|isolation|stress|depress|anxiety"
    r"|hospice|palliative|terminal|death",
    re.IGNORECASE,
)
SYNTHETIC_TAG = {
    "system": "https://triadcare.local/fhir/CodeSystem/data-classification",
    "code": "synthea-background",
}


def history_end(case: dict) -> date:
    """Return the day the background history ends: GAP_DAYS before the baseline assessment."""
    return date.fromisoformat(str(case["baseline"]["date"])) - timedelta(days=GAP_DAYS)


def run_synthea(case: dict, end: date) -> dict:
    """Run Synthea for one living patient of the case's sex and age. Return its bundle."""
    out = SYNTHEA_DIR / "out" / case["id"]
    shutil.rmtree(out, ignore_errors=True)
    seed = str(zlib.crc32(case["id"].encode()) % 2**31)
    age = end.year - int(case["patient"]["birth_year"])
    gender = "F" if case["patient"]["gender"] == "female" else "M"
    command = [
        JAVA, "-jar", str(SYNTHEA_DIR / "synthea.jar"),
        "-s", seed, "-cs", seed, "-p", "1", "-g", gender, "-a", f"{age}-{age}",
        "-r", end.strftime("%Y%m%d"), "-e", end.strftime("%Y%m%d"),
        f"--exporter.baseDirectory={out}/",
        "--exporter.fhir.export=true",
        "--exporter.hospital.fhir.export=false",
        "--exporter.practitioner.fhir.export=false",
        f"--exporter.years_of_history={YEARS_OF_HISTORY}",
    ]
    subprocess.run(command, cwd=SYNTHEA_DIR, check=True, capture_output=True)
    for path in sorted((out / "fhir").glob("*.json")):
        bundle = json.loads(path.read_text())
        patient = bundle["entry"][0]["resource"]
        if patient["resourceType"] == "Patient" and "deceasedDateTime" not in patient:
            return bundle
    raise RuntimeError(f"{case['id']}: Synthea made no living patient")


def strip_references(value):
    """Remove every reference to another resource. The subject is set again later."""
    if isinstance(value, dict):
        if "reference" in value:
            return None
        kept = {k: strip_references(v) for k, v in value.items()}
        return {k: v for k, v in kept.items() if v not in (None, {}, [])}
    if isinstance(value, list):
        kept = [strip_references(v) for v in value]
        return [v for v in kept if v not in (None, {}, [])]
    return value


def codes(resource: dict) -> set[str]:
    concepts = [resource.get("code", {}), *(c.get("code", {}) for c in resource.get("component", []))]
    return {c.get("code") for concept in concepts for c in concept.get("coding", [])}


def keep(resource: dict) -> bool:
    kind = resource["resourceType"]
    if kind not in KEEP_TYPES or DROP_TEXT.search(json.dumps(resource)):
        return False
    if kind == "Condition":
        status = resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code")
        return status == "resolved" and "(disorder)" in resource.get("code", {}).get("text", "")
    if kind == "MedicationRequest":
        return resource.get("status") in {"completed", "stopped"} and "medicationCodeableConcept" in resource
    if kind == "Observation":
        categories = {
            c.get("code") for cat in resource.get("category", []) for c in cat.get("coding", [])
        }
        return not (codes(resource) & DROP_LOINC or categories & DROP_OBSERVATION_CATEGORIES)
    return True


def case_end(case: dict) -> str:
    return history_end(case).isoformat()


def resource_date(resource: dict) -> str:
    for key in ("effectiveDateTime", "performedDateTime", "onsetDateTime", "authoredOn", "recordedDate"):
        if key in resource:
            return resource[key]
    for key in ("performedPeriod", "period", "effectivePeriod"):
        if key in resource:
            return resource[key].get("start", "")
    return ""


def background(case: dict, bundle: dict) -> list[dict]:
    prefix = case["patient"]["id"].lower().replace("-", "")
    resources = [e["resource"] for e in bundle["entry"] if keep(e["resource"])]
    resources = [r for r in resources if resource_date(r)[:10] <= case_end(case)]
    resources = sorted(resources, key=resource_date)[-BACKGROUND_SIZE:]
    out = []
    for index, resource in enumerate(resources, start=1):
        item = strip_references({k: v for k, v in resource.items() if k not in {"meta", "text", "identifier"}})
        item["id"] = f"{prefix}-bg-{index:04d}"
        item["meta"] = {"tag": [SYNTHETIC_TAG]}
        item["subject"] = {"reference": f"Patient/{case['patient']['id']}"}
        out.append(item)
    return out


def main() -> int:
    manifest = yaml.safe_load((SCENARIOS / "manifest.yaml").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for entry in manifest["cases"]:
        case = yaml.safe_load((SCENARIOS / entry["file"]).read_text())
        end = history_end(case)
        resources = background(case, run_synthea(case, end))
        document = {"scenario_id": case["id"], "history_end": end.isoformat(), "resources": resources}
        (OUT / f"{case['id']}.json").write_text(json.dumps(document, indent=1) + "\n")
        print(f"{case['id']}: {len(resources)} resources, history ends {end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
