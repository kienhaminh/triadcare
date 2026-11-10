"""Imaging gateway configuration: catalog, inference service URLs, and limits."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit

from src.config._env import ConfigError, env_float, env_int, env_str

DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "scenarios/fixtures/imaging-workflow.json"
DEFAULT_BASE_URLS = {
    "mri": ("MRI_BASE_URL", "http://127.0.0.1:8101"),
    "pet": ("PET_BASE_URL", "http://127.0.0.1:8102"),
    "dti": ("DTI_BASE_URL", "http://127.0.0.1:8103"),
}


@dataclass(frozen=True)
class ImagingSettings:
    catalog_path: Path
    base_urls: Mapping[str, str]  # modality -> base URL without a trailing slash
    timeout_seconds: float
    concurrency: int  # parallel inference requests per event loop; the services are GPU-bound
    bearer_token: str


def _base_url(env_name: str, default: str) -> str:
    base_url = env_str(env_name, default).rstrip("/")
    try:
        parts = urlsplit(base_url)
        port = parts.port
    except ValueError:
        raise ConfigError("Imaging service configuration is invalid.") from None
    if (
        parts.scheme not in {"http", "https"}
        or not parts.netloc
        or (port is not None and not 1 <= port <= 65535)
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ConfigError("Imaging service configuration is invalid.")
    return base_url


def imaging_settings() -> ImagingSettings:
    token = env_str("IMAGING_BEARER_TOKEN")
    if any(not 33 <= ord(char) <= 126 for char in token):
        raise ConfigError("Imaging authorization configuration is invalid.")
    return ImagingSettings(
        catalog_path=Path(env_str("IMAGING_CATALOG_PATH") or DEFAULT_CATALOG),
        base_urls=MappingProxyType(
            {
                modality: _base_url(env_name, default)
                for modality, (env_name, default) in DEFAULT_BASE_URLS.items()
            }
        ),
        timeout_seconds=env_float(
            "IMAGING_TIMEOUT_SECONDS",
            300,
            minimum=1,
            maximum=3600,
            message="Imaging timeout configuration is invalid.",
        ),
        concurrency=env_int(
            "IMAGING_CONCURRENCY",
            2,
            minimum=1,
            maximum=16,
            message="Imaging concurrency configuration is invalid.",
        ),
        bearer_token=token,
    )
