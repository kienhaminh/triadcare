"""Download the DLBS image series that the scenarios use.

The Dallas Lifespan Brain Study (OpenNeuro ds004856, CC0) is the image source. Each
series goes to data/dlbs/<series ID>/ with its BIDS sidecars, and the MRI service reads
it there. Only wave-1 T1-weighted series are used.

    poetry run python scenarios/fetch_dlbs.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

DATASET = "ds004856"
VERSION = "1.3.0"
BASE_URL = f"https://s3.amazonaws.com/openneuro.org/{DATASET}"
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "dlbs"

# Source subjects. Scenario patients match each subject by sex and approximate age.
SUBJECTS = [
    "sub-833", "sub-4540", "sub-3319", "sub-823", "sub-1600",  # female, test
    "sub-935", "sub-867", "sub-408", "sub-4869",  # male, test
    "sub-1023", "sub-2133",  # female, development
]  # fmt: skip

# Series kind -> (BIDS folder, file stem after the subject and session, file suffixes)
KINDS = {
    "T1W": ("anat", "acq-MPRAGE_run-1_T1w", (".nii.gz", ".json")),
}


def series_id(subject: str, kind: str) -> str:
    """The series ID of one wave-1 series: sub-833, T1W -> DLBS-833-W1-T1W."""
    return f"DLBS-{subject.removeprefix('sub-')}-W1-{kind}"


def series_files(subject: str, kind: str) -> list[str]:
    folder, stem, suffixes = KINDS[kind]
    name = f"{subject}_ses-wave1_{stem}"
    return [f"{subject}/ses-wave1/{folder}/{name}{suffix}" for suffix in suffixes]


def fetch(subject: str, kind: str) -> None:
    target = DATA_ROOT / series_id(subject, kind)
    target.mkdir(parents=True, exist_ok=True)
    for remote in series_files(subject, kind):
        path = target / Path(remote).name
        if path.exists():
            continue
        partial = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(f"{BASE_URL}/{remote}", partial)
        partial.rename(path)
    print(target.name)


if __name__ == "__main__":
    for subject in SUBJECTS:
        for kind in KINDS:
            fetch(subject, kind)
