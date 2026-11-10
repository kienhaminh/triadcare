"""What differs per modality: reported methods, measurements, and limitations."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from src.imaging import ImagingError

COMMON_LIMITATIONS = (
    "Research estimate; not clinically validated.",
    "Confidence describes pipeline reliability, not diagnosis.",
)
UNREPORTED_METHOD = "unreported"
UNREPORTED_METHOD_LIMITATION = "Service did not report a recognized processing method."


def number(value: object, *, minimum: float = 0, maximum: float | None = None) -> int | float:
    """A finite service number in [minimum, maximum]; booleans are not numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ImagingError("Imaging service returned an invalid measurement.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ImagingError("Imaging service returned an invalid measurement.")
    if value < minimum or (maximum is not None and value > maximum):
        raise ImagingError("Imaging service returned an invalid measurement.")
    return value


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    # (marker in the service summary, reported method name); the first match wins.
    methods: tuple[tuple[str, str], ...]
    limitations: tuple[str, ...]
    # Validates the service's volume_stats and returns only the published measurements.
    measurements: Callable[[dict], dict]

    def method(self, summary: object) -> tuple[str, str | None]:
        """The reported method, and a limitation when the service named none we know."""
        text = summary.casefold() if isinstance(summary, str) else ""
        for marker, method in self.methods:
            if marker in text:
                return method, None
        return UNREPORTED_METHOD, UNREPORTED_METHOD_LIMITATION


def _mri(stats: dict) -> dict:
    return {
        "hippocampus_left_ml": {"value": number(stats.get("hippocampus_left_ml")), "unit": "mL"},
        "hippocampus_right_ml": {"value": number(stats.get("hippocampus_right_ml")), "unit": "mL"},
    }


def _pet(stats: dict) -> dict:
    return {"suvr": {"value": number(stats.get("suvr")), "unit": "1"}}


def _dti(stats: dict) -> dict:
    measurements = {
        "fa_cingulum": {"value": number(stats.get("fa_cingulum"), maximum=1), "unit": "1"}
    }
    if "cingulum_roi_voxel_count" in stats:
        count = number(stats["cingulum_roi_voxel_count"], minimum=1)
        if int(count) != count:
            raise ImagingError("Imaging service returned an invalid measurement.")
        measurements["cingulum_roi_voxel_count"] = {"value": int(count), "unit": "voxels"}
    return measurements


MODALITIES = {
    spec.name: spec
    for spec in (
        ModalitySpec(
            name="mri",
            methods=(("hippodeep", "hippodeep"),),
            limitations=("No normative comparison is available.",),
            measurements=_mri,
        ),
        ModalitySpec(
            name="pet",
            methods=(
                (
                    "fallback mni-to-native aal3 registration",
                    "fallback_mni_to_native_aal3_registration",
                ),
                (
                    "fastsurfer deep-learning native-space segmentation",
                    "fastsurfer_deep_learning_native_space_segmentation",
                ),
            ),
            limitations=(
                "Not partial-volume-corrected.",
                "No published amyloid-positivity cutoffs apply to this pipeline.",
            ),
            measurements=_pet,
        ),
        ModalitySpec(
            name="dti",
            methods=(
                (
                    "fallback mni-to-native search window projection",
                    "fallback_mni_to_native_search_window_projection",
                ),
                (
                    "tractseg deep-learning native-space segmentation",
                    "tractseg_deep_learning_native_space_segmentation",
                ),
            ),
            limitations=(
                "Uses a single-tensor model and does not model crossing fibers.",
                "The ROI uses an FA > 0.3 threshold; no normative high/low claims are supported.",
            ),
            measurements=_dti,
        ),
    )
}
