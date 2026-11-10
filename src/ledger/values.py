"""Checks on the values the ledger stores."""

# All malformed source values use the same caller-visible validation contract.
# ruff: noqa: TRY004

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any


def validate_text(value: Any, name: str, *, max_length: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise ValueError(f"{name} must be a non-empty string of at most {max_length} characters.")
    return value


def normalize_recorded_at(value: Any, *, default_now: bool = False) -> str:
    if value is None:
        if not default_now:
            raise ValueError("recorded_at is required.")
        value = datetime.now(UTC)
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        return value.isoformat()
    elif isinstance(value, str):
        if len(value) == 10:
            try:
                return date.fromisoformat(value).isoformat()
            except ValueError as exc:
                raise ValueError(
                    "recorded_at must be an ISO date or timezone-aware timestamp."
                ) from exc
        try:
            parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
        except ValueError as exc:
            raise ValueError(
                "recorded_at must be an ISO date or timezone-aware timestamp."
            ) from exc
    else:
        raise ValueError("recorded_at must be an ISO date or timezone-aware timestamp.")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamp values must include a timezone.")
    parsed = parsed.astimezone(UTC)
    return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")


def recorded_weekday(recorded_at: str) -> str:
    """Weekday of the recording date, so that time words such as "Sunday" or "last night"
    in a check-in resolve to a date."""
    return date.fromisoformat(recorded_at[:10]).strftime("%A")
