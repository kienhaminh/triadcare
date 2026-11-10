"""FHIR server configuration."""

import os
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from src.config._env import ConfigError, env_str

_SCOPE_PATTERN = re.compile(r"[A-Za-z0-9._~-]{1,128}\Z")


@dataclass(frozen=True)
class FHIRSettings:
    base_url: str | None  # None when FHIR_BASE_URL is unset; FHIR tools then fail on use
    scope: str  # one path segment after the base URL; empty for none
    bearer_token: str

    def root(self) -> str:
        """The server root URL for requests."""
        if self.base_url is None:
            raise ConfigError("Set FHIR_BASE_URL to configure the FHIR server.")
        root = self.base_url.rstrip("/")
        return f"{root}/{self.scope}" if self.scope else root


def _base_url() -> str | None:
    base_url = env_str("FHIR_BASE_URL")
    if not base_url:
        return None
    if any(character.isspace() for character in base_url) or "\\" in base_url:
        raise ConfigError("FHIR_BASE_URL must be a valid HTTP or HTTPS URL.")
    try:
        parsed = urlsplit(base_url)
        # Accessing these properties validates malformed host and port syntax.
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        raise ConfigError("FHIR_BASE_URL must be a valid HTTP or HTTPS URL.") from None
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "?" in base_url
        or "#" in base_url
    ):
        raise ConfigError(
            "FHIR_BASE_URL must be a valid HTTP or HTTPS URL without credentials, query, or fragment."
        )
    return base_url


def fhir_settings() -> FHIRSettings:
    # An empty FHIR_SCOPE means no scope segment, so a blank value is not replaced
    # by the default.
    scope = os.environ.get("FHIR_SCOPE", "DEFAULT")
    if scope and (not _SCOPE_PATTERN.fullmatch(scope) or scope in {".", ".."}):
        raise ConfigError("FHIR_SCOPE must be one safe path segment.")
    return FHIRSettings(
        base_url=_base_url(), scope=scope, bearer_token=env_str("FHIR_BEARER_TOKEN")
    )
