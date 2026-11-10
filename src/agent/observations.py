"""Cut a tool result to fit model input, and page back through the rest."""

import json
from typing import Any

# Characters of one tool observation that enter model input. Longer results are
# cut and point to read_tool_result; events and traces keep the full value.
OBSERVATION_LIMIT = 12_000
# The Doctor sees the whole specialist answer, so a consultation result is never cut.
CONSULTATION_PREFIX = "consult_"
RESULT_PAGE_LIMIT = 8_000
READ_TOOL_RESULT = "read_tool_result"
# A nested specialist's tool events reach the Doctor's stream with observations
# cut to this many characters. The trace keeps the full value.
NESTED_OBSERVATION_LIMIT = 1_000


def _encoded_prefix(text: str, budget: int) -> str:
    """Return the longest prefix whose JSON string encoding fits in ``budget`` characters."""
    used = 0
    for index, character in enumerate(text):
        # Quotes, backslashes, and control characters grow when JSON-encoded.
        used += len(json.dumps(character, ensure_ascii=False)) - 2
        if used > budget:
            return text[:index]
    return text


def read_page(tool_results: dict[str, str], args: dict) -> dict:
    """One page of a stored tool result, by character offset."""
    if set(args) - {"result_id", "offset", "limit"}:
        raise ValueError(f"{READ_TOOL_RESULT} accepts only result_id, offset, and limit.")
    result_id = args.get("result_id")
    offset = args.get("offset", 0)
    limit = args.get("limit", RESULT_PAGE_LIMIT)
    if not isinstance(result_id, str) or result_id not in tool_results:
        raise ValueError("Tool result not found in this turn.")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a nonnegative integer.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= RESULT_PAGE_LIMIT:
        raise ValueError(f"limit must be an integer from 1 to {RESULT_PAGE_LIMIT}.")
    text = tool_results[result_id]
    if offset > len(text):
        raise ValueError("offset exceeds the tool result length.")
    content = _encoded_prefix(text[offset : offset + limit], RESULT_PAGE_LIMIT)
    end = offset + len(content)
    return {
        "result_id": result_id,
        "content": content,
        "content_offset": offset,
        "total_characters": len(text),
        "next_offset": end if end < len(text) else None,
    }


# A check-in list holds the recaps of the whole period, and a period query holds its
# messages; a cut result hides check-ins.
TOOL_OBSERVATION_LIMITS = {
    # The FHIR index is the plan of all record reads; a cut index hides record types.
    "list_fhir_records": 32_000,
    "list_check_ins": 32_000,
    "query_conversations": 40_000,
    "search_conversations": 40_000,
}


def limit_of(name: str) -> float:
    """Characters of a tool's observation that enter model input."""
    if name.startswith(CONSULTATION_PREFIX):
        return float("inf")
    return TOOL_OBSERVATION_LIMITS.get(name, OBSERVATION_LIMIT)


def model_observation(
    name: str, observation: Any, call_id: str, tool_results: dict[str, str]
) -> str:
    """Serialize an observation for model input, cutting it with a pointer if too long.
    Fields that start with "_" are for code only and stay out."""
    if isinstance(observation, dict):
        observation = {k: v for k, v in observation.items() if not k.startswith("_")}
    text = json.dumps(observation, ensure_ascii=False)
    limit = limit_of(name)
    # Pages of read_tool_result are already bounded; cutting them again would nest pointers.
    if len(text) <= limit or name == READ_TOOL_RESULT:
        return text
    tool_results[call_id] = text
    shaped = _with_cut_answer(observation, call_id, len(text), limit)
    if shaped is not None:
        return shaped
    wrapper = {
        "truncated": True,
        "result_id": call_id,
        "total_characters": len(text),
        "content": "",
        "next_offset": len(text),
        "read_with": READ_TOOL_RESULT,
    }
    overhead = len(json.dumps(wrapper, ensure_ascii=False))
    content = _encoded_prefix(text, limit - overhead)
    wrapper.update(content=content, next_offset=len(content))
    return json.dumps(wrapper, ensure_ascii=False)


def _with_cut_answer(observation: Any, call_id: str, total: int, limit: int) -> str | None:
    """A result with its other fields whole and its answer prose cut to fit, with a
    pointer to the full result. None when the other fields do not fit."""
    if not isinstance(observation, dict) or not isinstance(observation.get("answer"), str):
        return None
    pointer = {"result_id": call_id, "total_characters": total, "read_with": READ_TOOL_RESULT}
    shaped = {**observation, "answer": "", "answer_cut": pointer}
    room = limit - len(json.dumps(shaped, ensure_ascii=False))
    if room <= 0:
        return None
    shaped["answer"] = _encoded_prefix(observation["answer"], room)
    return json.dumps(shaped, ensure_ascii=False)


def nested_event(event: dict) -> dict:
    """Bound a forwarded tool event so saved workspace traces stay small."""
    if event["type"] != "tool_end" or event.get("observation") is None:
        return event
    text = json.dumps(event["observation"], ensure_ascii=False)
    if len(text) <= NESTED_OBSERVATION_LIMIT:
        return event
    preview = {
        "truncated": True,
        "total_characters": len(text),
        "preview": text[:NESTED_OBSERVATION_LIMIT],
    }
    return {**event, "observation": preview}
