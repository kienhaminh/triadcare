"""Typed readers for environment variables."""

import math
import os


class ConfigError(ValueError):
    """An environment variable has an invalid value; the message is safe to show."""


def env_str(name: str, default: str = "") -> str:
    """The stripped value, or the default when the variable is unset or blank."""
    return os.environ.get(name, "").strip() or default


def env_float(
    name: str, default: float, *, minimum: float, maximum: float, message: str | None = None
) -> float:
    """A finite number in [minimum, maximum], or the default when unset or blank."""
    raw = os.environ.get(name, "").strip()
    error = message or f"{name} configuration is invalid."
    try:
        value = float(raw) if raw else default
    except ValueError:
        raise ConfigError(error) from None
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ConfigError(error)
    return value


def env_int(
    name: str, default: int, *, minimum: int, maximum: int, message: str | None = None
) -> int:
    """An integer in [minimum, maximum], or the default when unset or blank."""
    raw = os.environ.get(name, "").strip()
    error = message or f"{name} configuration is invalid."
    try:
        value = int(raw) if raw else default
    except ValueError:
        raise ConfigError(error) from None
    if not minimum <= value <= maximum:
        raise ConfigError(error)
    return value
