"""Hippocampus volume via the ``hippodeep`` model (fetched at image build).

hippodeep has no native confidence. TTA uses a fixed (not random) shift set.
Confidence is ensemble CoV, not a validated accuracy score.

hippodeep loads weights at import from ``sys.argv``. Import once, then call
``main()`` with real argv so TTA does not reload weights.

hippodeep reads its input from the process-wide ``sys.argv``. The gateway runs requests
in threads, so one lock holds each whole run: two runs at the same time could otherwise
read each other's image and report a wrong volume.
"""

from __future__ import annotations

import csv
import sys
import tempfile
import threading
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import shift as nd_shift

_THIRD_PARTY_DIR = Path(__file__).resolve().parent / "third_party" / "hippodeep"

_TTA_SHIFTS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),  # reported volumes; other shifts feed confidence only
    (0.5, 0.0, 0.0),
    (-0.5, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, 0.5),
)

_hippodeep_module = None  # process-wide; hippodeep loads weights at import
_RUN_LOCK = threading.Lock()


class HippodeepError(RuntimeError):
    """Raised when the hippodeep run fails or produces no CSV output."""


def _hippodeep():
    global _hippodeep_module
    if _hippodeep_module is None:
        sys.path.insert(0, str(_THIRD_PARTY_DIR))
        sys.argv = ["hippodeep.py", "__init_placeholder__.nii.gz"]
        import hippodeep  # deferred: hippodeep loads weights at import

        _hippodeep_module = hippodeep
    return _hippodeep_module


def _read_volumes_csv(nifti_path: Path) -> tuple[float, float]:
    """Read (hippoL_mm3, hippoR_mm3) from the CSV hippodeep writes next to its input."""
    csv_path = nifti_path.with_name(
        nifti_path.name.replace(".nii.gz", "_hippoLR_volumes.csv")
    )
    if not csv_path.exists():
        raise HippodeepError(f"hippodeep did not produce {csv_path} for {nifti_path}")
    with open(csv_path, newline="") as f:
        row = next(csv.DictReader(f))
    return float(row["hippoL"]), float(row["hippoR"])


def _make_shifted_copy(src: Path, dst: Path, shift: tuple[float, float, float]) -> None:
    img = nib.load(src)
    data = img.get_fdata()
    shifted = nd_shift(data, shift=shift, order=1, mode="nearest")
    nib.save(nib.Nifti1Image(shifted.astype(np.float32), img.affine, img.header), dst)


def run_hippodeep(nifti_path: Path) -> dict[str, float]:
    """Run hippodeep on ``nifti_path`` plus a small TTA ensemble.

    Returns ``{"hippocampus_left_ml", "hippocampus_right_ml", "confidence"}``.
    The reported volumes come from the ORIGINAL unperturbed image; the
    perturbed copies only feed the confidence estimate.
    """
    with _RUN_LOCK:
        return _run_locked(Path(nifti_path))


def _run_locked(nifti_path: Path) -> dict[str, float]:
    hippodeep = _hippodeep()
    with tempfile.TemporaryDirectory(prefix="hippodeep_tta_") as tmp:
        tmp_dir = Path(tmp)
        pass_volumes_mm3: list[float] = []
        reported_left_mm3: float | None = None
        reported_right_mm3: float | None = None

        for i, shift_vec in enumerate(_TTA_SHIFTS):
            if shift_vec == (0.0, 0.0, 0.0):
                pass_input = nifti_path
            else:
                pass_input = tmp_dir / f"tta_{i}.nii.gz"
                _make_shifted_copy(nifti_path, pass_input, shift_vec)

            sys.argv = ["hippodeep.py", str(pass_input)]
            hippodeep.main()
            left_mm3, right_mm3 = _read_volumes_csv(pass_input)
            pass_volumes_mm3.append(left_mm3 + right_mm3)
            if shift_vec == (0.0, 0.0, 0.0):
                reported_left_mm3, reported_right_mm3 = left_mm3, right_mm3

    if reported_left_mm3 is None or reported_right_mm3 is None:
        raise HippodeepError(f"TTA ensemble never ran the unperturbed pass for {nifti_path}")

    volumes = np.array(pass_volumes_mm3, dtype=np.float64)
    mean = float(volumes.mean())
    std = float(volumes.std())
    coefficient_of_variation = (std / mean) if mean > 0 else 1.0
    confidence = 1.0 / (1.0 + coefficient_of_variation)
    confidence = max(0.0, min(1.0, confidence))

    return {
        "hippocampus_left_ml": reported_left_mm3 / 1000.0,
        "hippocampus_right_ml": reported_right_mm3 / 1000.0,
        "confidence": confidence,
    }
