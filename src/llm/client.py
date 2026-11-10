"""OpenRouter requests. The caller owns the conversation history."""

import json
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import httpx

from src.config import ConfigError
from src.config.llm import DEFAULT_MODEL, LLMSettings, llm_settings

__all__ = [
    "DEFAULT_MODEL",
    "LLMContextError",
    "LLMEmptyAnswer",
    "LLMError",
    "LLMOutputLimit",
    "LLMProviderError",
    "LLMTransientError",
    "complete",
    "session",
    "stream",
]

API_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMError(RuntimeError):
    """A configuration, transport, or provider response error."""


class LLMEmptyAnswer(LLMError):
    """An answer with neither text nor tool calls, or one that the provider ended with an
    error or did not end within the deadline. The caller can ask again."""


class LLMOutputLimit(LLMError):
    """A response that reached its output token limit before completion."""

    def __init__(self, response: dict, max_tokens: int, *, has_tool_calls: bool = False) -> None:
        self.response = response
        self.max_tokens = max_tokens
        self.has_tool_calls = has_tool_calls
        super().__init__("OpenRouter reached the output token limit before completing the answer.")


class LLMTransientError(LLMError):
    """A provider failure that a later request can avoid: a timeout, no connection, a rate
    limit or server error, or an answer that the provider ended with an error."""


class LLMProviderError(LLMEmptyAnswer, LLMTransientError):
    """An answer that the provider ended with an error."""


# complete() asks again after a transient error, after these waits in seconds.
TRANSIENT_WAITS = (10, 30, 60)


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise LLMEmptyAnswer("OpenRouter did not complete the answer within the request deadline.")


class LLMContextError(LLMError):
    """The provider rejected the request because it exceeds the model's context window."""

    def __init__(self, message: str, code: str = "context_limit_exceeded") -> None:
        super().__init__(message)
        self.code = code


def settings() -> LLMSettings:
    """The current LLM configuration; an invalid value raises LLMError."""
    try:
        return llm_settings()
    except ConfigError as error:
        raise LLMError(str(error)) from None


def _error_detail(data: object) -> str | None:
    """The provider's error message in a response body, or None."""
    error = data.get("error") if isinstance(data, dict) else None
    detail = error.get("message") if isinstance(error, dict) else None
    return detail if isinstance(detail, str) and detail.strip() else None


def _status_error(exc: httpx.HTTPStatusError) -> LLMError:
    """Map a provider rejection to an error; a context-length rejection is recoverable."""
    status = exc.response.status_code
    try:
        detail = _error_detail(json.loads(exc.response.content))
    except (ValueError, httpx.StreamError):
        detail = None
    if status == 400 and detail and "context length" in detail.lower():
        return LLMContextError(detail)
    error = LLMTransientError if status == 429 or status >= 500 else LLMError
    return error(f"OpenRouter returned HTTP {status}" + (f": {detail}" if detail else "."))


SESSION_ID_LIMIT = 256  # OpenRouter's session_id length limit
_SESSION_ID: ContextVar[str | None] = ContextVar("llm_session_id", default=None)


@contextmanager
def session(session_id: str) -> Iterator[None]:
    """Group the requests made in this block under one OpenRouter session_id.

    The value follows the context into tasks and threads started in the block. It
    takes precedence over LLM_SESSION_ID.
    """
    token = _SESSION_ID.set(session_id)
    try:
        yield
    finally:
        _SESSION_ID.reset(token)


# Some providers of a model ignore tool_choice and answer in text; a request with
# tool_choice goes only to providers that honor every parameter of the request.
# No provider is pinned, so the provider can change between requests; each response
# records its provider.
PROVIDER_WITH_TOOL_CHOICE = {"require_parameters": True}


def request_profile() -> dict:
    """Default generation settings. Individual requests can record explicit overrides."""
    config = settings()
    return {
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "temperature": config.temperature,
        "seed": None,
        "provider": {"pinned": False, "with_tool_choice": PROVIDER_WITH_TOOL_CHOICE},
        "max_tokens": config.max_output_tokens,
        "timeout_seconds": config.timeout_seconds,
        "deadline_seconds": config.deadline_seconds,
    }


def _request_options(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
    max_tokens: int | None = None,
    reasoning: dict | None = None,
) -> tuple[str, float, dict]:
    config = settings()
    if not config.api_key:
        raise LLMError("Set OPENROUTER_API_KEY in .env.")
    payload = {
        "model": config.model,
        "messages": messages,
        "max_tokens": max_tokens or config.max_output_tokens,
        "reasoning": reasoning if reasoning is not None else {"effort": config.reasoning_effort},
    }
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    # OpenRouter groups requests with the same session_id (Logs > Sessions).
    session_id = _SESSION_ID.get() or config.session_id
    if session_id:
        payload["session_id"] = session_id[:SESSION_ID_LIMIT]
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
        payload["provider"] = dict(PROVIDER_WITH_TOOL_CHOICE)
    return config.api_key, config.timeout_seconds, payload


def _empty_answer(finish_reason: str | None) -> str:
    return (
        f"OpenRouter returned no valid text answer (finish_reason: {finish_reason}). "
        "Check the model and output token limit."
    )


def _validate_tool_calls(message: dict) -> bool:
    """Validate a completed assistant tool-call array, if present."""
    if "tool_calls" not in message:
        return False
    tool_calls = message["tool_calls"]
    if tool_calls is None or tool_calls == []:
        return False
    if not isinstance(tool_calls, list):
        raise TypeError

    ids: set[str] = set()
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            raise TypeError
        call_id = tool_call.get("id")
        if not isinstance(call_id, str) or not call_id.strip() or call_id in ids:
            raise ValueError
        ids.add(call_id)
        if tool_call.get("type") != "function":
            raise ValueError
        function = tool_call.get("function")
        if not isinstance(function, dict):
            raise TypeError
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name.strip() or not isinstance(arguments, str):
            raise ValueError
    return True


def complete(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
    max_tokens: int | None = None,
    reasoning: dict | None = None,
) -> dict:
    """One answer. Token and reasoning overrides apply only to this request. A transient provider
    error is asked again after each wait of TRANSIENT_WAITS."""
    for wait in (*TRANSIENT_WAITS, None):
        try:
            return _complete_once(messages, tools, tool_choice, max_tokens, reasoning)
        except LLMTransientError:
            if wait is None:
                raise
            time.sleep(wait)
    raise AssertionError("unreachable")


def _complete_once(
    messages: list[dict],
    tools: list[dict] | None,
    tool_choice: str | None,
    max_tokens: int | None,
    reasoning: dict | None = None,
) -> dict:
    """One request."""
    key, timeout, payload = _request_options(
        messages, tools=tools, tool_choice=tool_choice, max_tokens=max_tokens,
        reasoning=reasoning,
    )
    deadline = time.monotonic() + llm_settings().deadline_seconds
    try:
        with httpx.stream(
            "POST",
            API_URL,
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
            timeout=timeout,
        ) as response:
            if response.is_error:
                response.read()  # the body holds the provider's error message
            response.raise_for_status()
            body = bytearray()
            for part in response.iter_bytes():
                _check_deadline(deadline)
                body += part
    except httpx.HTTPStatusError as exc:
        raise _status_error(exc) from None
    except httpx.TimeoutException:
        raise LLMTransientError("The OpenRouter request timed out.") from None
    except httpx.RequestError as exc:
        raise LLMTransientError(f"Cannot reach OpenRouter: {type(exc).__name__}.") from None

    data: dict = {}
    try:
        data = json.loads(body)
        if isinstance(data, dict) and data.get("error"):
            # Like a streamed error object, the caller can ask again.
            raise LLMProviderError(
                f"OpenRouter failed: {_error_detail(data) or 'provider error'}." + _usage(data)
            )
        choice = data["choices"][0]
        if not isinstance(choice, dict):
            raise TypeError("the choice is not an object")
        finish_reason = choice.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason.strip():
            raise ValueError("no finish_reason")
        if finish_reason == "length":
            message = choice.get("message")
            if isinstance(message, dict):
                message = _without_encrypted(message)
            raise LLMOutputLimit(
                {
                    "message": message,
                    "model": data.get("model", payload["model"]),
                    "usage": data.get("usage"),
                    "finish_reason": finish_reason,
                },
                payload["max_tokens"],
                has_tool_calls=bool(message.get("tool_calls")) if isinstance(message, dict) else False,
            )
        message = choice["message"]
        if not isinstance(message, dict) or message["role"] != "assistant":
            raise ValueError("the message is not an assistant message")
        content = message.get("content")
        if "content" not in message or (content is not None and not isinstance(content, str)):
            raise ValueError("the content is not text")
        has_tool_calls = _validate_tool_calls(message)
        if finish_reason == "error":
            detail = _error_detail(data) or _error_detail(choice)
            raise LLMProviderError(
                "OpenRouter ended the answer with an error before it was complete"
                + (f": {detail}." if detail else ".")
                + _usage(data)
            )
        if has_tool_calls and finish_reason != "tool_calls":
            raise ValueError(f"tool calls with finish_reason {finish_reason}")
        if finish_reason == "tool_calls" and not has_tool_calls:
            raise ValueError("finish_reason tool_calls without tool calls")
        if not has_tool_calls and (content is None or not content.strip()):
            raise LLMEmptyAnswer(_empty_answer(finish_reason) + _usage(data))
        return {
            "message": _without_encrypted(message),
            "model": data.get("model", payload["model"]),
            "provider": data.get("provider"),
            "usage": data.get("usage"),
            "finish_reason": choice.get("finish_reason"),
        }
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        reason = f" Reason: {str(exc) or 'invalid tool calls'} ({type(exc).__name__})."
        raise LLMError(
            "OpenRouter returned no valid text answer. Check the model and output token limit."
            + reason
            + _usage(data)
        ) from None


def _usage(data: dict) -> str:
    """The token counts of a response, for an error message."""
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return ""
    details = usage.get("completion_tokens_details") or {}
    return (
        f" Tokens: prompt {usage.get('prompt_tokens')}, completion "
        f"{usage.get('completion_tokens')}, reasoning {details.get('reasoning_tokens')}."
    )


_READABLE_REASONING = {"reasoning.text", "reasoning.summary"}


def _without_encrypted(message: dict) -> dict:
    """The message without encrypted reasoning, which a caller does not replay to the model."""
    details = message.get("reasoning_details")
    if not isinstance(details, list):
        return message
    readable = [
        detail
        for detail in details
        if isinstance(detail, dict) and detail.get("type") in _READABLE_REASONING
    ]
    kept = {key: value for key, value in message.items() if key != "reasoning_details"}
    if readable:
        kept["reasoning_details"] = readable
    return kept


def _same_reasoning_detail(left: dict, right: dict) -> bool:
    if "index" in left and "index" in right:
        return left["index"] == right["index"] and (
            left.get("type") is None
            or right.get("type") is None
            or left.get("type") == right.get("type")
        )
    return left.get("id") is not None and left.get("id") == right.get("id")


def _merge_reasoning_details(details: list[dict], updates: list) -> None:
    for update in updates:
        if not isinstance(update, dict):
            raise TypeError
        existing = next(
            (detail for detail in details if _same_reasoning_detail(detail, update)),
            None,
        )
        if existing is None:
            details.append(dict(update))
            continue
        for key, value in update.items():
            previous = existing.get(key)
            if key in {"text", "summary"} and isinstance(previous, str) and isinstance(value, str):
                existing[key] = previous + value
            else:
                existing[key] = value


def _merge_tool_call_deltas(tool_calls: dict[int, dict], updates: list) -> None:
    if not isinstance(updates, list):
        raise TypeError
    for update in updates:
        if not isinstance(update, dict):
            raise TypeError
        index = update.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError
        partial = tool_calls.setdefault(
            index,
            {"id": "", "type": None, "name": "", "arguments": "", "has_arguments": False},
        )
        if "id" in update:
            call_id = update["id"]
            if not isinstance(call_id, str):
                raise ValueError
            partial["id"] += call_id
        if "type" in update:
            call_type = update["type"]
            if call_type != "function":
                raise ValueError
            partial["type"] = call_type

        if "function" in update:
            function = update["function"]
            if not isinstance(function, dict):
                raise TypeError
            if "name" in function:
                name = function["name"]
                if not isinstance(name, str):
                    raise ValueError
                partial["name"] += name
            if "arguments" in function:
                arguments = function["arguments"]
                if not isinstance(arguments, str):
                    raise ValueError
                partial["arguments"] += arguments
                partial["has_arguments"] = True


def _finalize_tool_call_deltas(tool_calls: dict[int, dict]) -> list[dict]:
    finalized = []
    ids: set[str] = set()
    for index in sorted(tool_calls):
        partial = tool_calls[index]
        call_id = partial["id"]
        name = partial["name"]
        if (
            not call_id.strip()
            or call_id in ids
            or partial["type"] != "function"
            or not name.strip()
            or not partial["has_arguments"]
        ):
            raise ValueError
        ids.add(call_id)
        finalized.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": partial["arguments"]},
            }
        )
    return finalized


async def stream(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
) -> AsyncIterator[dict]:
    """Yield answer chunks and a final response envelope from OpenRouter SSE."""
    key, timeout, payload = _request_options(messages, tools=tools, tool_choice=tool_choice)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    deadline = time.monotonic() + llm_settings().deadline_seconds

    content_parts: list[str] = []
    has_content_string = False
    tool_call_parts: dict[int, dict] = {}
    saw_tool_call_delta = False
    reasoning_fields: dict[str, str] = {}
    reasoning_details: list[dict] = []
    model = payload["model"]
    provider = None
    usage = None
    finish_reason = None
    done = False

    def consume_event(event_data: str) -> tuple[str | None, bool, bool]:
        nonlocal model, provider, usage, finish_reason, done
        nonlocal has_content_string, saw_tool_call_delta
        if event_data == "[DONE]":
            done = True
            return None, False, True
        try:
            data = json.loads(event_data)
        except (json.JSONDecodeError, TypeError):
            raise ValueError from None
        if not isinstance(data, dict):
            raise TypeError
        if data.get("error"):
            # Like finish_reason "error", the caller can ask again.
            detail = _error_detail(data) or "provider error"
            raise LLMEmptyAnswer(f"OpenRouter stream failed: {detail}.")
        if "model" in data:
            model = data["model"]
        if "provider" in data:
            provider = data["provider"]
        if "usage" in data:
            usage = data["usage"]

        choices = data.get("choices")
        if choices is None and "usage" in data:
            choices = []
        if not isinstance(choices, list):
            raise TypeError
        if not choices:
            return None, False, False

        choice = choices[0]
        if not isinstance(choice, dict):
            raise TypeError
        new_finish_reason = choice.get("finish_reason")
        if new_finish_reason is not None:
            if not isinstance(new_finish_reason, str):
                raise ValueError
            if new_finish_reason == "error":
                raise LLMEmptyAnswer("OpenRouter stream failed before completing the answer.")
            finish_reason = new_finish_reason
        delta = choice.get("delta", {})
        if isinstance(delta, dict) and delta.get("tool_calls"):
            saw_tool_call_delta = True
        if finish_reason == "length":
            chunk_content = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(chunk_content, str):
                has_content_string = True
                return chunk_content, False, False
            return None, False, False
        if not isinstance(delta, dict):
            raise TypeError
        chunk_content = delta.get("content")
        if chunk_content is not None and not isinstance(chunk_content, str):
            raise ValueError
        if isinstance(chunk_content, str):
            has_content_string = True
        if "tool_calls" in delta:
            updates = delta["tool_calls"]
            if updates is not None:
                _merge_tool_call_deltas(tool_call_parts, updates)

        has_reasoning = False
        for field in ("reasoning", "reasoning_content"):
            part = delta.get(field)
            if part is not None:
                if not isinstance(part, str):
                    raise ValueError
                reasoning_fields[field] = reasoning_fields.get(field, "") + part
                has_reasoning = has_reasoning or bool(part)
        details = delta.get("reasoning_details")
        if details is not None:
            if not isinstance(details, list):
                raise TypeError
            _merge_reasoning_details(reasoning_details, details)
            has_reasoning = has_reasoning or bool(details)

        return chunk_content, has_reasoning, False

    try:
        async with (
            httpx.AsyncClient(timeout=timeout) as client,
            client.stream(
                "POST",
                API_URL,
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
            ) as response,
        ):
            if response.is_error:
                await response.aread()  # the body holds the provider's error message
            response.raise_for_status()
            data_lines: list[str] = []

            async def dispatch_event() -> AsyncIterator[dict]:
                nonlocal data_lines
                if not data_lines:
                    return
                event_data = "\n".join(data_lines)
                data_lines = []
                chunk_content, has_reasoning, is_done = consume_event(event_data)
                if has_reasoning:
                    yield {"type": "reasoning"}
                if chunk_content:
                    content_parts.append(chunk_content)
                    yield {"type": "content", "text": chunk_content}
                if is_done:
                    return

            async for line in response.aiter_lines():
                _check_deadline(deadline)
                if line.startswith(":"):
                    continue
                if line == "":
                    async for event in dispatch_event():
                        yield event
                    if done:
                        break
                elif line.startswith("data:"):
                    value = line[5:]
                    data_lines.append(value.removeprefix(" "))
            if data_lines and not done:
                async for event in dispatch_event():
                    yield event
    except httpx.HTTPStatusError as exc:
        raise _status_error(exc) from None
    except httpx.TimeoutException:
        raise LLMTransientError("The OpenRouter request timed out.") from None
    except httpx.RequestError as exc:
        raise LLMTransientError(f"Cannot reach OpenRouter: {type(exc).__name__}.") from None
    except (ValueError, TypeError, KeyError, IndexError):
        raise LLMError("OpenRouter returned an invalid streaming response.") from None

    if finish_reason == "length":
        message = {
            "role": "assistant",
            "content": "".join(content_parts) if has_content_string else None,
        }
        if tool_call_parts:
            message["tool_calls"] = [
                {
                    "id": partial["id"],
                    "type": partial["type"],
                    "function": {
                        "name": partial["name"],
                        "arguments": partial["arguments"],
                    },
                }
                for _, partial in sorted(tool_call_parts.items())
            ]
        elif saw_tool_call_delta:
            message["tool_calls"] = []
        raise LLMOutputLimit(
            {
                "message": message,
                "model": model,
                "usage": usage,
                "finish_reason": finish_reason,
            },
            payload["max_tokens"],
            has_tool_calls=bool(tool_call_parts) or saw_tool_call_delta,
        )
    if not done or finish_reason is None:
        raise LLMError("OpenRouter ended the stream before the answer was complete.")
    try:
        tool_calls = _finalize_tool_call_deltas(tool_call_parts)
    except (ValueError, TypeError, KeyError):
        raise LLMError("OpenRouter returned an invalid streaming response.") from None
    if tool_calls and finish_reason == "length":
        raise LLMError("OpenRouter truncated the response before completing tool calls.")
    if tool_calls and finish_reason != "tool_calls":
        raise LLMError("OpenRouter returned an invalid streaming response.")
    if finish_reason == "tool_calls" and not tool_calls:
        raise LLMError("OpenRouter returned an invalid streaming response.")
    content = "".join(content_parts)
    if not content.strip() and not tool_calls:
        raise LLMEmptyAnswer(_empty_answer(finish_reason))

    message = {
        "role": "assistant",
        "content": content if has_content_string else None,
        **reasoning_fields,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_details:
        message["reasoning_details"] = reasoning_details
    yield {
        "type": "result",
        "result": {
            "message": _without_encrypted(message),
            "model": model,
            "provider": provider,
            "usage": usage,
            "finish_reason": finish_reason,
        },
    }
