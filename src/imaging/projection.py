"""Turn a raw service result into the validated observation an agent receives."""

from __future__ import annotations

import json
import re

from src.imaging import ImagingError
from src.imaging.catalog import Study
from src.imaging.modalities import COMMON_LIMITATIONS, MODALITIES, number

_MODEL_VERSION = re.compile(r"[A-Za-z0-9._+-]{1,64}\Z")
_MAX_OUTPUT_BYTES = 16 * 1024


def check_size(observation: dict) -> dict:
    if len(json.dumps(observation, ensure_ascii=False).encode("utf-8")) > _MAX_OUTPUT_BYTES:
        raise ImagingError("Imaging result exceeds the size limit.")
    return observation


def project(result: dict, study: Study, data_context: str) -> dict:
    """Keep only validated measurements and metadata. Raw paths, images, and free text
    from the service never reach the agent."""
    spec = MODALITIES[study.modality]
    if result.get("modality") != f"brain-{spec.name}":
        raise ImagingError("Imaging service returned a mismatched modality.")
    if "case_id" in result and result["case_id"] != study.study_id:
        raise ImagingError("Imaging service returned a mismatched study.")
    stats = result.get("volume_stats")
    if not isinstance(stats, dict):
        raise ImagingError("Imaging service returned an invalid result.")
    confidence = number(result.get("confidence"), maximum=1)
    measurements = spec.measurements(stats)

    model_version = result.get("model_version")
    if model_version is not None and (
        not isinstance(model_version, str) or _MODEL_VERSION.fullmatch(model_version) is None
    ):
        raise ImagingError("Imaging service returned an invalid model version.")
    method, method_limitation = spec.method(result.get("summary"))
    limitations = [*COMMON_LIMITATIONS, *spec.limitations]
    if method_limitation:
        limitations.append(method_limitation)
    return check_size(
        {
            "source": "imaging",
            "patient_id": study.patient_id,
            "study_id": study.study_id,
            "modality": spec.name,
            "data_context": data_context,
            "measurements": measurements,
            "confidence": confidence,
            "method": method,
            "model_version": model_version,
            "limitations": limitations,
        }
    )
