"""Record provider usage without storing prompts or credentials."""

import json
import logging
import os
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

usage_target: ContextVar[tuple[Path, str] | None] = ContextVar("judge_usage_target", default=None)


def record_usage(model: str, usage: object, stop_reason: object) -> None:
    target = usage_target.get()
    if target is None:
        return
    path, question_key = target
    entry = {
        "time": datetime.now(UTC).isoformat(), "question_key": question_key,
        "model": model, "usage": usage, "stop_reason": stop_reason,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(entry, ensure_ascii=False) + "\n").encode()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
    except OSError:
        logging.getLogger(__name__).warning("Could not record judge usage in %s", path)

