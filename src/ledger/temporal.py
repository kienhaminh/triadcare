"""Server-owned ledger scope for one evaluation task."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from src.ledger.values import normalize_recorded_at, validate_text

_SCOPE: ContextVar[tuple[str, str] | None] = ContextVar("ledger_scope", default=None)


@contextmanager
def ledger_scope(patient_id: str, as_of: str) -> Iterator[None]:
    patient_id = validate_text(patient_id, "patient_id")
    as_of = normalize_recorded_at(as_of)
    token = _SCOPE.set((patient_id, as_of))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def scope_as_of(patient_id: str) -> str | None:
    scope = _SCOPE.get()
    if scope is None:
        return None
    if patient_id != scope[0]:
        raise ValueError("The requested patient is outside this evaluation task.")
    return scope[1]
