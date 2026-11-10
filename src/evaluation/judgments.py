"""Ask the judge once per question and cache accepted answers; merge repeated votes.

When no attempt succeeds and at least one answer was invalid, the unscored outcome is
cached as terminal, and a resumed run does not ask again. When every attempt failed with a
call error, nothing is cached, and a resumed run asks again. Use a new directory for an independent run.

Each accepted judge answer is saved under <directory> by a key of the judge identity and
the question, so rescoring reuses it. A failed or invalid answer leaves the item unscored;
it never becomes a failure. With several votes, each unit takes the majority status.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.evaluation import judge as J
from src.evaluation.judge_usage import usage_target
from src.evaluation.run import dump

MAX_JUDGE_ATTEMPTS = 2

# The votes of one question send the same prompt. The first vote asks alone and the
# others wait for it, so they read the provider's prompt cache.
_first_asks: dict[str, threading.Event] = {}
_first_asks_lock = threading.Lock()


def _first_ask(prompt_key: str) -> threading.Event | None:
    """The event to set when the first ask of this prompt ends, or None after waiting."""
    with _first_asks_lock:
        event = _first_asks.get(prompt_key)
        if event is None:
            event = _first_asks[prompt_key] = threading.Event()
            return event
    event.wait()
    return None


def _write_rejection(path: Path, key: str, answer: Any, reason: str) -> str | None:
    """Append a rejected raw answer and its parser reason beside the cache."""
    rejected_path = path.with_suffix(".rejected.jsonl")
    entry = {"question_key": key, "answer": answer, "reason": reason}
    try:
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        with rejected_path.open("a") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except OSError as error:
        return f"Rejected answer could not be saved ({type(error).__name__})."
    return None


class Judgments:
    """Ask the judge, validate, and cache accepted answers per judge identity."""

    def __init__(self, directory: Path, ask: J.Ask | None, identity: dict[str, Any]) -> None:
        self.directory = directory
        self.ask = ask
        self.identity = identity

    def __call__(
        self,
        question: tuple[str, str],
        parse: Callable[[dict], dict],
        schema: dict | None = None,
    ) -> dict:
        system, user = question
        key = J.question_key(self.identity, system, user, schema)
        path = self.directory / f"{key}.json"
        terminal_path = path.with_suffix(".unscored.json")
        if terminal_path.exists():
            try:
                terminal = json.loads(terminal_path.read_text())["result"]
                if terminal.get("status") == "unscored" and isinstance(terminal.get("error"), str):
                    return terminal
            except (ValueError, TypeError, KeyError, AttributeError, OSError):
                pass
            return {"status": "unscored", "error": "Saved terminal judgment could not be read."}
        errors = []
        if path.exists():
            answer = None
            try:
                cached = json.loads(path.read_text())
                if not isinstance(cached, dict):
                    raise J.JudgeError("Cached judgment is not a JSON object.")
                answer = cached.get("answer")
                if not isinstance(answer, dict):
                    raise J.JudgeError("Cached answer is not a JSON object.")
                parsed = parse(answer)
            except J.JudgeError as error:
                reason = str(error)
                save_error = _write_rejection(path, key, answer, reason)
                if save_error:
                    reason = f"{reason}; {save_error}"
                errors.append(reason)
            except (ValueError, TypeError, OSError):
                reason = "Cached judgment could not be read or parsed."
                save_error = _write_rejection(path, key, answer, reason)
                if save_error:
                    reason = f"{reason}; {save_error}"
                errors.append(reason)
            else:
                return {"status": "scored", **parsed}
        if self.ask is None:
            errors.append("No judge configured.")
            return {"status": "unscored", "error": "; ".join(errors)}
        first = _first_ask(J.question_key({}, system, user, schema))
        try:
            return self._ask(system, user, parse, schema, key, path, terminal_path, errors)
        finally:
            if first is not None:
                first.set()

    def _ask(self, system, user, parse, schema, key, path, terminal_path, errors) -> dict:
        rejected = False
        for _ in range(MAX_JUDGE_ATTEMPTS):
            token = usage_target.set((self.directory / "usage.jsonl", key))
            try:
                answer = (
                    self.ask(system, user)
                    if schema is None
                    else self.ask(system, user, schema=schema)
                )
            except J.JudgeError as error:
                errors.append(str(error))
                continue
            finally:
                usage_target.reset(token)
            try:
                if not isinstance(answer, dict):
                    raise J.JudgeError("Judge answer is not a JSON object.")
                parsed = parse(answer)
            except J.JudgeError as error:
                rejected = True
                reason = str(error)
                save_error = _write_rejection(path, key, answer, reason)
                if save_error:
                    reason = f"{reason}; {save_error}"
                errors.append(reason)
                continue
            dump(path, {"judge": self.identity, "system": system, "user": user, "answer": answer})
            return {"status": "scored", **parsed}
        result = {"status": "unscored", "error": "; ".join(errors)}
        # Only a rejected answer is terminal. A call error (HTTP error, empty or non-JSON
        # content) is not a judgment, so a later scoring run asks again.
        if rejected:
            dump(terminal_path, {
                "judge": self.identity, "system": system, "user": user,
                "attempt_limit": MAX_JUDGE_ATTEMPTS, "result": result,
            })
        return result


def vote_identities(identity: dict[str, Any], votes: int) -> list[dict[str, Any]]:
    """Judge identities of the votes. The first vote is the plain identity, so earlier
    single-vote judgments are its cache; later votes are numbered replicates."""
    return [identity, *({**identity, "replicate": vote} for vote in range(2, votes + 1))]


def _majority(units: list[dict]) -> dict:
    """The status most votes gave; among tied leaders, the earliest vote. A label holds
    when most votes set it, and never on an omitted unit."""
    statuses = [unit["status"] for unit in units]
    counts = Counter(statuses)
    status = next(s for s in statuses if counts[s] == max(counts.values()))
    entry = {
        "status": status,
        "evidence": next(u["evidence"] for u in units if u["status"] == status),
        "votes": statuses,
        "agreement": counts[status],
    }
    if "labels" in units[0]:
        entry["labels"] = {
            label: status != "omitted" and 2 * sum(u["labels"][label] for u in units) > len(units)
            for label in J.LABELS
        }
    return entry


def voted(blocks: list[dict[str, dict]]) -> dict[str, dict]:
    """Merge the preservation results of the votes: each unit takes the majority of its
    scored votes. A comparison no vote scored keeps the first vote's status."""
    result: dict[str, dict] = {}
    for stage, block in blocks[0].items():
        comparisons = []
        for index, comparison in enumerate(block["comparisons"]):
            scored = [
                vote[stage]["comparisons"][index]
                for vote in blocks
                if vote[stage]["comparisons"][index]["status"] == "scored"
            ]
            merged = dict(comparison)
            if scored:
                merged.pop("error", None)
                merged.update(
                    status="scored",
                    votes=len(scored),
                    units={
                        unit_id: _majority([vote["units"][unit_id] for vote in scored])
                        for unit_id in scored[0]["units"]
                    },
                )
            comparisons.append(merged)
        result[stage] = {**block, "comparisons": comparisons}
    return result
