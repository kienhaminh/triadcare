"""Read a scenario conversation file into sessions of messages, without a database."""

# All malformed source values use the same caller-visible validation contract.
# ruff: noqa: TRY004

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.ledger.values import normalize_recorded_at, validate_text

IMPORTED_SPEAKERS = {
    "patient_sim": "patient",
    "caregiver_sim": "caregiver",
    "care_agent": "care_agent",
}
MAX_CONTENT_LENGTH = 16_000


def parse_source(path: Path, root: Path) -> dict[str, Any]:
    try:
        payload_bytes = path.read_bytes()
        payload = json.loads(payload_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read scenario source {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "scenario_conversation":
        raise ValueError(f"Unsupported scenario conversation source: {path}")
    scenario_id = validate_text(payload.get("scenario_id"), "scenario_id")
    patient_id = validate_text(payload.get("patient_id"), "patient_id")
    split = validate_text(payload.get("split"), "split", max_length=32)
    if split not in {"dev", "held_out"}:
        raise ValueError(f"Invalid split in {path}: {split}")
    days = payload.get("days")
    if not isinstance(days, list):
        raise ValueError(f"Scenario {scenario_id} must contain a days list.")
    try:
        source_path = path.relative_to(root).as_posix()
    except ValueError:
        source_path = path.name
    source_hash = hashlib.sha256(payload_bytes).hexdigest()
    sessions: list[dict[str, Any]] = []
    seen_days: set[int] = set()
    for day_data in days:
        if not isinstance(day_data, dict):
            raise ValueError(f"Scenario {scenario_id} contains an invalid day.")
        source_day = day_data.get("day")
        if isinstance(source_day, bool) or not isinstance(source_day, int) or source_day < 1:
            raise ValueError(f"Scenario {scenario_id} contains an invalid source day.")
        if source_day in seen_days:
            raise ValueError(f"Scenario {scenario_id} contains duplicate source day {source_day}.")
        seen_days.add(source_day)
        recorded_at = normalize_recorded_at(day_data.get("check_in_date"))
        if len(recorded_at) != 10:
            raise ValueError(f"Scenario {scenario_id} check_in_date must be a date without a time.")
        check_ins = day_data.get("check_ins")
        if not isinstance(check_ins, dict):
            raise ValueError(f"Scenario {scenario_id} day {source_day} has no check_ins map.")
        for role in ("patient", "caregiver"):
            check_in = check_ins.get(role)
            if not isinstance(check_in, dict):
                raise ValueError(
                    f"Scenario {scenario_id} day {source_day} is missing {role} transcript."
                )
            transcript = check_in.get("transcript")
            if not isinstance(transcript, list):
                raise ValueError(
                    f"Scenario {scenario_id} day {source_day} has an invalid {role} transcript."
                )
            session_id = f"imported:{scenario_id}:day-{source_day}:{role}"
            participant_id = f"{role}:{patient_id}"
            messages: list[dict[str, Any]] = []
            seen_sequences: set[int] = set()
            for index, turn in enumerate(transcript, start=1):
                if not isinstance(turn, dict):
                    raise ValueError(f"Scenario {scenario_id} contains an invalid transcript turn.")
                speaker_source = turn.get("speaker")
                if not isinstance(speaker_source, str) or speaker_source not in IMPORTED_SPEAKERS:
                    raise ValueError(
                        f"Unknown imported speaker {speaker_source!r} in {scenario_id}."
                    )
                expected_speaker = f"{role}_sim"
                if speaker_source not in {expected_speaker, "care_agent"}:
                    raise ValueError(
                        f"Speaker {speaker_source!r} does not belong in the {role} transcript for {scenario_id}."
                    )
                sequence = turn.get("sequence", index)
                if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
                    raise ValueError(f"Scenario {scenario_id} contains an invalid turn sequence.")
                if sequence in seen_sequences:
                    raise ValueError(
                        f"Scenario {scenario_id} repeats sequence {sequence} in {role} day {source_day}."
                    )
                seen_sequences.add(sequence)
                content = turn.get("text")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"Scenario {scenario_id} contains a turn with empty text.")
                if len(content) > MAX_CONTENT_LENGTH:
                    raise ValueError(
                        f"Scenario {scenario_id} contains a turn above the content length cap."
                    )
                message_id = f"imported:{scenario_id}:day-{source_day}:{role}:seq-{sequence}"
                metadata = {
                    "scenario_id": scenario_id,
                    "split": split,
                    "source_day": source_day,
                    "source_sequence": sequence,
                    "source_check_in_id": check_in.get("check_in_id"),
                    "source_file": source_path,
                    "source_sha256": source_hash,
                }
                messages.append(
                    {
                        "message_id": message_id,
                        "speaker": IMPORTED_SPEAKERS[speaker_source],
                        "content": content,
                        "recorded_at": recorded_at,
                        "sequence": sequence,
                        "source_metadata": metadata,
                    }
                )
            sessions.append(
                {
                    "session_id": session_id,
                    "audience": role,
                    "participant_id": participant_id,
                    "source_day": source_day,
                    "messages": messages,
                }
            )
    return {
        "scenario_id": scenario_id,
        "patient_id": patient_id,
        "split": split,
        "source_path": source_path,
        "source_sha256": source_hash,
        "sessions": sessions,
    }
