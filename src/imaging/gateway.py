"""HTTP client of the MRI, PET, and DTI inference gateways."""

from __future__ import annotations

import asyncio
import json
import weakref
from urllib.parse import quote

import httpx

from src.imaging import ImagingError, settings
from src.imaging.catalog import Study

_MAX_RESPONSE_BYTES = 1024 * 1024
_MAX_DETAIL_CHARACTERS = 200

# Inference gateways are GPU-bound. Parallel consultations and background tasks
# share this limit on each event loop.
_LIMITS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _limit() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    limit = _LIMITS.get(loop)
    if limit is None:
        limit = _LIMITS[loop] = asyncio.Semaphore(settings().concurrency)
    return limit


async def analyze(study: Study) -> dict:
    """Run the study's modality analysis and return the service's raw JSON object."""
    async with _limit():
        return await _request(study)


async def _read(response: httpx.Response) -> bytes:
    if (
        response.headers.get("content-length", "").isdigit()
        and int(response.headers["content-length"]) > _MAX_RESPONSE_BYTES
    ):
        raise ImagingError("Imaging service response exceeds the size limit.")
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > _MAX_RESPONSE_BYTES:
            raise ImagingError("Imaging service response exceeds the size limit.")
    return bytes(body)


def _failure(status: int, raw: bytes) -> str:
    """Error text of a failed request with the HTTP status and the service detail.
    Only a JSON "detail" string goes to the agent. Other body text can hold private data."""
    try:
        body = json.loads(raw)
        detail = body.get("detail") if isinstance(body, dict) else None
    except (UnicodeDecodeError, ValueError):
        detail = None
    detail = " ".join(detail.split())[:_MAX_DETAIL_CHARACTERS] if isinstance(detail, str) else ""
    message = f"Imaging service request failed with HTTP {status}"
    return f"{message}: {detail}" if detail else f"{message}."


async def _request(study: Study) -> dict:
    config = settings()
    headers = {"Authorization": f"Bearer {config.bearer_token}"} if config.bearer_token else {}
    url = f"{config.base_urls[study.modality]}/v1/studies/{quote(study.study_id, safe='')}/analyze"
    try:
        async with (
            httpx.AsyncClient(timeout=config.timeout_seconds, follow_redirects=False) as client,
            client.stream("POST", url, headers=headers) as response,
        ):
            if not 200 <= response.status_code < 300:
                try:
                    raw = await _read(response)
                except ImagingError:
                    raw = b""
                raise ImagingError(_failure(response.status_code, raw))
            raw = await _read(response)
    except ImagingError:
        raise
    except httpx.TimeoutException:
        raise ImagingError("Imaging service request timed out; the outcome is unknown.") from None
    except httpx.InvalidURL:
        raise ImagingError("Imaging service configuration is invalid.") from None
    except httpx.HTTPError:
        raise ImagingError("Imaging service request failed.") from None
    try:
        result = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        raise ImagingError("Imaging service returned an invalid response.") from None
    if not isinstance(result, dict):
        raise ImagingError("Imaging service returned an invalid response.")
    return result
