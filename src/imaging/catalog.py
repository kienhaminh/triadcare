"""The imaging catalog: which studies exist, who owns them, and their modality."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.fhir.ids import ID_PATTERN, valid_fhir_id
from src.imaging import ImagingError, settings
from src.imaging.modalities import MODALITIES

PATIENT_ID_PATTERN = "^" + ID_PATTERN + "$"  # FHIR Patient id
STUDY_ID_PATTERN = r"^[A-Z0-9_-]{1,64}$"
_STUDY_ID = re.compile(STUDY_ID_PATTERN)
_MAX_CATALOG_BYTES = 1024 * 1024


def valid_study_id(value: str) -> bool:
    return _STUDY_ID.fullmatch(value) is not None


@dataclass(frozen=True)
class Study:
    patient_id: str
    study_id: str  # also the case ID of the inference gateway
    modality: str


@dataclass(frozen=True)
class Catalog:
    studies: dict[str, Study]  # by study ID
    # "synthetic_fixture" when the catalog declares no real patients.
    data_context: str

    def patient_studies(self, patient_id: str) -> list[Study]:
        return [study for study in self.studies.values() if study.patient_id == patient_id]


def load() -> Catalog:
    """Read and validate the configured catalog. Every study ID is unique."""
    try:
        with settings().catalog_path.open("rb") as file:
            raw = file.read(_MAX_CATALOG_BYTES + 1)
    except OSError:
        raise ImagingError("Imaging catalog is unavailable.") from None
    if len(raw) > _MAX_CATALOG_BYTES:
        raise ImagingError("Imaging catalog exceeds the size limit.")
    try:
        catalog = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        raise ImagingError("Imaging catalog is invalid.") from None
    if not isinstance(catalog, dict) or not isinstance(catalog.get("cases"), list):
        raise ImagingError("Imaging catalog is invalid.")

    studies: dict[str, Study] = {}
    for case in catalog["cases"]:
        if not isinstance(case, dict):
            raise ImagingError("Imaging catalog contains an invalid case.")
        patient_id, entries = case.get("patient_id"), case.get("studies")
        if (
            not isinstance(patient_id, str)
            or not valid_fhir_id(patient_id)
            or not isinstance(entries, list)
        ):
            raise ImagingError("Imaging catalog contains an invalid case binding.")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ImagingError("Imaging catalog contains an invalid study.")
            modality, image_ids = entry.get("modality"), entry.get("image_ids")
            if (
                not isinstance(modality, str)
                or modality not in MODALITIES
                or not isinstance(image_ids, list)
                or not image_ids
            ):
                raise ImagingError("Imaging catalog contains an invalid study binding.")
            for study_id in image_ids:
                if (
                    not isinstance(study_id, str)
                    or not valid_study_id(study_id)
                    or study_id in studies
                ):
                    raise ImagingError(
                        "Imaging catalog contains a duplicate or invalid study binding."
                    )
                studies[study_id] = Study(patient_id, study_id, modality)

    profile = catalog.get("synthetic_profile")
    data_context = (
        "synthetic_fixture"
        if isinstance(profile, dict) and profile.get("is_real_patient") is False
        else "configured_catalog"
    )
    return Catalog(studies, data_context)
