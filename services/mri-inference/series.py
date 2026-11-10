"""Series lookup for the DLBS NIfTI mount.

Each series is one directory, ``<data root>/<series ID>/``, that holds one BIDS NIfTI
file and its sidecars. ``IMAGING_CASE_ALIASES`` maps a case ID (``CASE-P104-MRI``)
to a series ID (``DLBS-2133-W1-T1W``).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


class SeriesNotFoundError(LookupError):
    pass


def resolve_series_id(case_id: str) -> str:
    """Apply ``IMAGING_CASE_ALIASES`` to a case ID. An unknown ID stays as it is."""
    normalized = case_id.strip().upper()
    aliases = {}
    for item in os.environ.get("IMAGING_CASE_ALIASES", "").split(","):
        if ":" not in item:
            continue
        alias, series_id = (part.strip().upper() for part in item.split(":", 1))
        if alias and series_id:
            aliases[alias] = series_id
    return aliases.get(normalized, normalized)


def find_series_dir(case_id: str, data_root: Path) -> Path:
    """Return the series directory for a case ID. Raise if it does not exist."""
    series_id = resolve_series_id(case_id)
    series_dir = Path(data_root) / series_id
    if not series_dir.is_dir():
        raise SeriesNotFoundError(f"Series {series_id!r} was not found under {data_root}")
    return series_dir


def series_nifti(series_dir: Path) -> Path:
    """Return the one ``.nii.gz`` file of a series. Raise if there is not exactly one."""
    niftis = sorted(series_dir.glob("*.nii.gz"))
    if len(niftis) != 1:
        raise SeriesNotFoundError(
            f"Series {series_dir.name!r} has {len(niftis)} NIfTI files; expected exactly one"
        )
    return niftis[0]


def copy_series(series_dir: Path, *, prefix: str, out_dir: Path | None = None) -> Path:
    """Copy the NIfTI to a work directory as ``series.nii.gz``.

    The data mount is read-only, and the analysis steps write files next to their input.
    Returns the path of the copied ``series.nii.gz``.
    """
    nifti = series_nifti(series_dir)
    work_dir = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix=prefix))
    work_dir.mkdir(parents=True, exist_ok=True)
    target = work_dir / "series.nii.gz"
    shutil.copyfile(nifti, target)
    return target
