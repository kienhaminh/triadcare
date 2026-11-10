"""The configured FHIR server: root URL, request headers, and bounded GET requests."""

import json

import httpx

from src.config.fhir import fhir_settings

RESPONSE_LIMIT = 1024 * 1024


def root() -> str:
    """The server root URL: FHIR_BASE_URL, then the FHIR_SCOPE segment."""
    return fhir_settings().root()


async def _read_limited(response: httpx.Response) -> bytes:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > RESPONSE_LIMIT:
            raise ValueError("FHIR response exceeded the 1 MiB size limit.")
        body.extend(chunk)
    return bytes(body)


def headers() -> dict[str, str]:
    token = fhir_settings().bearer_token
    request_headers = {
        "Accept": "application/fhir+json",
        "Prefer": "handling=strict",
    }
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    return request_headers


async def request_json(
    url: str,
    *,
    params: list[tuple[str, str]] | None = None,
    not_found: bool = False,
) -> tuple[object, str]:
    """GET one bounded response and return its payload and request URL."""
    try:
        async with (
            httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client,
            client.stream("GET", url, params=params, headers=headers()) as response,
        ):
            if not_found and response.status_code == 404:
                raise ValueError("FHIR resource not found.")
            if response.status_code != 200:
                raise ValueError(f"FHIR server returned HTTP {response.status_code}.")
            content = await _read_limited(response)
            request_url = str(response.request.url)
    except httpx.TimeoutException:
        raise ValueError("FHIR request timed out.") from None
    except httpx.RequestError:
        raise ValueError("FHIR server could not be reached.") from None

    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("FHIR server returned invalid JSON.") from None
    return payload, request_url
