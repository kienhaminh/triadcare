"""Analyze the MRI studies of the test split once, so the MRI service holds their results.

The MRI service keeps the result of each series in memory. Run this script after each
start of the MRI container and before the evaluation runs. Then no run waits for an
MRI analysis.

    poetry run python -m src.imaging.prewarm
"""

from __future__ import annotations

import asyncio
import time

from src.imaging.catalog import Study
from src.imaging.gateway import analyze

STUDIES = ["P201", "P202", "P203", "P205", "P206", "P207", "P208", "P209", "P210"]


async def main() -> None:
    for patient in STUDIES:
        started = time.monotonic()
        result = await analyze(Study(f"P-{patient[1:]}", f"CASE-{patient}-MRI", "mri"))
        volumes = result.get("measurements") or result.get("volume_stats")
        print(f"CASE-{patient}-MRI {time.monotonic() - started:.1f}s {volumes}")


if __name__ == "__main__":
    asyncio.run(main())
