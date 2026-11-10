"""Radiologist tools: list catalog studies, run analyses, and read stored results.

One Pydantic model per tool validates the call and generates the JSON schema that
the model sees (src.llm.tool_schema).
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

import psycopg
from pydantic import AfterValidator, Field, ValidationError
from pydantic_core import PydanticCustomError

from src.fhir.ids import valid_fhir_id
from src.imaging import ImagingError, catalog, gateway
from src.imaging.catalog import PATIENT_ID_PATTERN, STUDY_ID_PATTERN, Catalog, Study
from src.imaging.projection import check_size, project
from src.imaging.results import STORED_RESULT_LIMIT, ImagingResultStore
from src.llm.tool_schema import ToolArguments, parameters


def _valid(check):
    def validate(value: str) -> str:
        if not check(value):
            raise PydanticCustomError("invalid_id", "invalid identifier")
        return value

    return AfterValidator(validate)


PatientId = Annotated[
    str,
    _valid(valid_fhir_id),
    Field(json_schema_extra={"pattern": PATIENT_ID_PATTERN, "not": {"enum": [".", ".."]}}),
]
StudyId = Annotated[
    str, _valid(catalog.valid_study_id), Field(json_schema_extra={"pattern": STUDY_ID_PATTERN})
]


class ListStudiesArguments(ToolArguments):
    patient_id: PatientId


class StudyArguments(ToolArguments):
    patient_id: PatientId
    study_id: StudyId


LIST_IMAGING_STUDIES_PARAMETERS = parameters(ListStudiesArguments)
ANALYZE_IMAGING_PARAMETERS = parameters(StudyArguments)
READ_STORED_RESULTS_PARAMETERS = parameters(StudyArguments)

_SHAPE_ERRORS = {"model_type", "missing", "extra_forbidden"}


def _arguments[T: ToolArguments](
    model: type[T], args: Any, *, id_error: str = "Imaging identifiers are invalid."
) -> T:
    """Validate a call. With several problems, the error follows the order the tools
    always checked: argument names, identifier types, identifier values.
    """
    try:
        return model.model_validate(args)
    except ValidationError as error:
        problems = [(item["type"], item["loc"][:1]) for item in error.errors()]
    if any(kind in _SHAPE_ERRORS for kind, _ in problems):
        raise ImagingError("Imaging arguments are invalid.") from None
    if any(kind == "string_type" for kind, _ in problems):
        raise ImagingError("Imaging identifiers are invalid.") from None
    raise ImagingError(id_error) from None


def _study(args: StudyArguments, modality: str | None) -> tuple[Study, Catalog]:
    """The catalog study of this patient; with a modality, the study must match it."""
    studies = catalog.load()
    study = studies.studies.get(args.study_id)
    if study is None or study.patient_id != args.patient_id:
        raise ImagingError("Study was not found for this patient in the imaging catalog.")
    if modality is not None and study.modality != modality:
        raise ImagingError(f"Study modality does not match the {modality.upper()} analysis tool.")
    return study, studies


def _result_store() -> ImagingResultStore:
    """The PKL imaging-result store of this process (LEDGER_SCHEMA)."""
    try:
        return ImagingResultStore()
    except (RuntimeError, OSError) as error:
        raise ImagingError("The imaging result store is unavailable.") from error


async def list_imaging_studies(args: dict) -> dict:
    """List catalog-authorized studies without contacting an inference service."""
    patient_id = _arguments(
        ListStudiesArguments, args, id_error="Patient identifier is invalid."
    ).patient_id
    studies = catalog.load()
    matches = [
        {"study_id": study.study_id, "modality": study.modality}
        for study in studies.patient_studies(patient_id)
    ]
    return check_size(
        {
            "source": "imaging_catalog",
            "patient_id": patient_id,
            "data_context": studies.data_context,
            "studies": matches,
            "availability": "not_checked",
            # The catalog lists analyzable studies only. Without this note, agents read a
            # missing date or report as an absent one.
            "note": "The catalog has no acquisition date and no clinical report. The FHIR "
            "record (ImagingStudy, DiagnosticReport) holds them. An empty list means no "
            "analyzable study in the catalog; cite it as query:list_imaging_studies.",
        }
    )


async def _analyze_study(args: dict, modality: str) -> dict:
    """Run a fresh analysis of a catalog-authorized study and store the result in PKL.

    A failed store is a tool error, so every successful call has one stored result.
    """
    call = _arguments(StudyArguments, args)
    study, studies = _study(call, modality)
    observation = project(await gateway.analyze(study), study, studies.data_context)
    try:
        return await asyncio.to_thread(lambda: _result_store().append(observation))
    except psycopg.Error as error:
        raise ImagingError("The analysis ran but its result could not be stored.") from error


async def read_stored_imaging_results(args: dict) -> dict:
    """Stored analysis results of one catalog study, newest first; no analysis runs."""
    study, studies = _study(_arguments(StudyArguments, args), None)
    try:
        results = await asyncio.to_thread(
            lambda: _result_store().results(study.patient_id, study.study_id)
        )
    except psycopg.Error as error:
        raise ImagingError("The imaging result store is unavailable.") from error
    return {
        "source": "stored_imaging_results",
        "patient_id": study.patient_id,
        "study_id": study.study_id,
        "modality": study.modality,
        "data_context": studies.data_context,
        "stored_results": results,
        "note": "Stored results are earlier model-generated analyses, not hospital records. "
        f"The read shows at most the {STORED_RESULT_LIMIT} newest results."
        if results
        else "No stored result for this study; an analysis is needed to obtain one.",
    }


async def analyze_mri(args: dict) -> dict:
    return await _analyze_study(args, "mri")


async def analyze_pet(args: dict) -> dict:
    return await _analyze_study(args, "pet")


async def analyze_dti(args: dict) -> dict:
    return await _analyze_study(args, "dti")
