"""Radiologist imaging tools over the local MRI, PET, and DTI inference gateways.

Modules: `tools` (agent tools), `catalog` (authorized studies), `gateway` (inference
HTTP client), `projection` and `modalities` (result validation), `results` (stored
results).
"""

from src.config import ConfigError
from src.config.imaging import ImagingSettings, imaging_settings


class ImagingError(ValueError):
    """A safe error suitable for returning to an agent or user."""


def settings() -> ImagingSettings:
    """The imaging configuration; an invalid value raises ImagingError."""
    try:
        return imaging_settings()
    except ConfigError as error:
        raise ImagingError(str(error)) from None
