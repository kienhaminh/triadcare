"""MRI hippocampus-volume analysis of one DLBS T1-weighted series.

Reads the series from the read-only data mount.
"""

from __future__ import annotations

import os
from pathlib import Path

from hippodeep_runner import run_hippodeep
from result_contract import agent_facing_error, agent_facing_result
from series import SeriesNotFoundError, copy_series, find_series_dir

_DATA_ROOT = Path(os.environ.get("MRI_DATA_ROOT", "/data/dlbs"))
# hippodeep with fixed TTA shifts gives the same volumes for the same series, so the
# service keeps each series result. A repeat request then does not wait for the run lock.
_MEASURED: dict[Path, dict[str, float]] = {}


def _measure(series_dir: Path) -> dict[str, float]:
    if series_dir not in _MEASURED:
        _MEASURED[series_dir] = run_hippodeep(copy_series(series_dir, prefix="mri_"))
    return _MEASURED[series_dir]


def analyze_mri(patient_id: str, case_id: str, modality: str = "brain-mri") -> dict:
    """Hippocampus-volume finding for one T1-weighted series.

    ``case_id`` is a case ID or a series ID (``DLBS-2133-W1-T1W``).
    Raises on failure; never fabricate a result.
    """
    # never forward a series ID or a path through the error channel
    try:
        series_dir = find_series_dir(case_id, _DATA_ROOT)
    except SeriesNotFoundError as exc:
        raise ValueError(agent_facing_error("source_not_found")) from exc
    except Exception as exc:
        raise RuntimeError(agent_facing_error("source_unavailable")) from exc

    try:
        measured = _measure(series_dir)
    except Exception as exc:
        raise RuntimeError(agent_facing_error("analysis_failed")) from exc

    left_ml = measured["hippocampus_left_ml"]
    right_ml = measured["hippocampus_right_ml"]
    private_result = {
        "case_id": case_id,
        "modality": modality,
        "region_labels": ["hippocampus_left", "hippocampus_right"],
        "volume_stats": {
            "hippocampus_left_ml": left_ml,
            "hippocampus_right_ml": right_ml,
        },
        "slice_dim": 2,
        "representative_slice_index": 0,
        "summary": (
            f"Hippocampus volume (research estimate, not clinically validated): "
            f"left {left_ml:.2f} mL, right {right_ml:.2f} mL. Computed via a "
            f"vendored open-source model (hippodeep) from a T1-weighted MRI "
            f"series; no normative comparison is available for this value."
        ),
        "confidence": measured["confidence"],
        "patient_id": patient_id,
        "requested_modality": modality,
    }
    return agent_facing_result(private_result)

