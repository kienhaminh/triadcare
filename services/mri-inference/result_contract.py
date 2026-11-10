"""Boundary between the MRI service result and the agent-facing gateway output."""

from __future__ import annotations

from typing import Any

_AGENT_FACING_KEYS = (
    "modality",
    "region_labels",
    "volume_stats",
    "slice_dim",
    "representative_slice_index",
    "summary",
    "confidence",
)
_ERROR_MESSAGES = {
    "source_not_found": "MRI source series was not found",
    "source_unavailable": "MRI source series could not be resolved",
    "analysis_failed": "MRI analysis failed",
}


def agent_facing_result(private_result: dict[str, Any]) -> dict[str, Any]:
    """Drop the case ID and the request echo fields. They are not part of the agent-facing contract."""
    return {key: private_result[key] for key in _AGENT_FACING_KEYS if key in private_result}


def agent_facing_error(kind: str) -> str:
    """Map all container errors to a stable, identifier-free message."""
    return _ERROR_MESSAGES.get(kind, "MRI analysis could not be completed")
