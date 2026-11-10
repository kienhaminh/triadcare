"""Authenticated MRI gateway. Clients get only this URL and a token, never a data path.

Routes: ``GET /health`` (no token) and ``POST /v1/studies/{case_id}/analyze``.
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from analysis import analyze_mri
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

_CASE_ID_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
_DATA_ROOT = Path(os.environ.get("MRI_DATA_ROOT", "/data/dlbs"))
_AUTH_TOKEN = os.environ.get("IMAGING_GATEWAY_TOKEN", "").strip()
_ALLOWED_CASE_IDS = frozenset(
    value.strip().upper()
    for value in os.environ.get("IMAGING_ALLOWED_CASE_IDS", "").split(",")
    if value.strip()
)


def require_allowed_case(case_id: str) -> str:
    normalized = case_id.strip().upper()
    if not normalized or len(normalized) > 64 or any(
        char not in _CASE_ID_CHARS for char in normalized
    ):
        raise HTTPException(status_code=400, detail="invalid study id")
    if normalized not in _ALLOWED_CASE_IDS:
        raise HTTPException(status_code=404, detail="study not available")
    return normalized


class GatewayAuthMiddleware(BaseHTTPMiddleware):
    """Protect the data routes while leaving the health check probeable."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health" or not _AUTH_TOKEN:
            return await call_next(request)
        if request.headers.get("authorization") != f"Bearer {_AUTH_TOKEN}":
            return JSONResponse(
                status_code=401,
                content={"detail": "missing or invalid gateway token"},
            )
        return await call_next(request)


app = FastAPI(title="Local MRI Imaging Gateway", version="0.1.0")
app.add_middleware(GatewayAuthMiddleware)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "data_root_mounted": _DATA_ROOT.is_dir(),
        "allowlist_configured": bool(_ALLOWED_CASE_IDS),
        "alias_configured": bool(os.environ.get("IMAGING_CASE_ALIASES", "").strip()),
    }


@app.post("/v1/studies/{case_id}/analyze")
def analyze(case_id: str) -> dict:
    allowed_case_id = require_allowed_case(case_id)
    try:
        return analyze_mri(
            patient_id="local-gateway", case_id=allowed_case_id, modality="brain-mri"
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="study analysis failed") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="study analysis unavailable") from exc


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("MRI_HOST", "0.0.0.0"),
        port=int(os.environ.get("MRI_PORT", "8101")),
    )
