"""Seed canonical synthetic FHIR fixtures into the configured FHIR server."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from src.fhir import server

_DEPENDENCY_ORDER: dict[str, int] = {
    "Patient": 0,
    "Condition": 1,
    "MedicationStatement": 2,
    "Observation": 3,
    "CarePlan": 4,
    "DocumentReference": 5,
    "ImagingStudy": 6,
    "Provenance": 7,
}


def load_fixtures(
    directory: Path, patient_ids: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    """Collect unique FHIR resources from fixture files in dependency order."""
    if not directory.is_dir():
        raise ValueError(f"Fixtures directory does not exist: {directory}")

    patients = set(patient_ids) if patient_ids is not None else None
    unique_resources: dict[tuple[str, str], dict[str, Any]] = {}

    for path in sorted(directory.rglob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        resources = data.get("agent_visible", {}).get("resources", [])
        for resource in resources:
            resource_type = resource.get("resourceType")
            resource_id = resource.get("id")
            if not resource_type or not resource_id:
                continue

            if patients is not None:
                # Check if resource belongs to requested patients
                subject_ref = resource.get("subject", {}).get("reference") or resource.get(
                    "patient", {}
                ).get("reference")
                is_target = False
                if (resource_type == "Patient" and resource_id in patients) or (
                    subject_ref and any(subject_ref.endswith(f"/{pid}") for pid in patients)
                ):
                    is_target = True
                elif resource_type == "Provenance":
                    targets = [t.get("reference", "") for t in resource.get("target", [])]
                    if any(
                        any(target.endswith(f"/{pid}") for pid in patients) for target in targets
                    ):
                        is_target = True

                if not is_target:
                    continue

            key = (resource_type, resource_id)
            unique_resources[key] = resource

    return sorted(
        unique_resources.values(),
        key=lambda res: (_DEPENDENCY_ORDER.get(res["resourceType"], 99), res["id"]),
    )


def seed_fhir(
    directory: Path,
    patient_ids: Iterable[str] | None = None,
    timeout: float = 15.0,
) -> dict[str, int]:
    """Upsert canonical resources into the FHIR server and return counts by type."""
    root = server.root()
    headers = server.headers()
    resources = load_fixtures(directory, patient_ids)

    counts: dict[str, int] = {}
    with httpx.Client(timeout=timeout) as client:
        for resource in resources:
            resource_type = resource["resourceType"]
            resource_id = resource["id"]
            url = f"{root}/{resource_type}/{resource_id}"
            response = client.put(url, json=resource, headers=headers)
            if response.status_code not in (200, 201):
                raise RuntimeError(
                    f"Failed to seed {resource_type}/{resource_id}: "
                    f"HTTP {response.status_code} - {response.text[:200]}"
                )
            counts[resource_type] = counts.get(resource_type, 0) + 1

    return counts


def main() -> int:
    repository_root = Path(__file__).resolve().parents[2]
    load_dotenv(repository_root / ".env")
    parser = argparse.ArgumentParser(
        description="Seed canonical synthetic fixtures into the FHIR server."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=repository_root / "scenarios" / "fixtures" / "fhir",
        help="fixtures directory (default: scenarios/fixtures/fhir)",
    )
    parser.add_argument(
        "--patient",
        action="append",
        dest="patients",
        help="Seed only this patient ID (e.g. P-001); repeatable",
    )
    args = parser.parse_args()
    results = seed_fhir(args.directory, args.patients)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
