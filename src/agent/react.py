"""A bounded ReAct loop using native function calls and tool observations."""

import asyncio
import json
import math
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from contextvars import ContextVar, copy_context
from copy import deepcopy
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from src.agent.citations import known_ids, unresolved_citations
from src.agent.observations import (
    READ_TOOL_RESULT,
    RESULT_PAGE_LIMIT,
    model_observation,
    nested_event,
    read_page,
)
from src.agent.registry import Tool, bind, register_tool
from src.config.agent import agent_settings
from src.llm import client as llm


class RunLimitError(llm.LLMError):
    """The shared request budget was reached."""


# Receives tool lifecycle events of a nested agent run inside a tool call.
_EVENT_SINK: ContextVar[Callable[[dict], None] | None] = ContextVar(
    "react_event_sink", default=None
)


def _sum_usage(total: dict, usage: dict) -> bool:
    for key, value in usage.items():
        if isinstance(value, dict):
            nested = total.setdefault(key, {})
            if not isinstance(nested, dict) or not _sum_usage(nested, value):
                return False
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            previous = total.get(key, 0)
            if not isinstance(previous, (int, float)):
                return False
            total[key] = previous + value
    return True


@dataclass
class RunState:
    """Per-turn state shared by Doctor and its specialists; never stored globally."""

    requests: int = 0
    responses: int = 0
    max_requests: int = field(default_factory=lambda: agent_settings().max_requests)
    usage: dict = field(default_factory=dict)
    usage_complete: bool = True
    usage_samples: list[dict | None] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    tool_call_count: int = 0
    tool_results: dict[str, str] = field(default_factory=dict)
    # Agents of this turn that are running and will still send a closing request.
    active_agents: int = 0
    # False in the governance ablation: no citation repair and no attached sources.
    governed: bool = True

    def can_step(self) -> bool:
        """Whether one more tool step leaves a closing request for every running agent."""
        return self.max_requests - self.requests - 1 >= self.active_agents

    def reserve_request(self) -> None:
        if self.requests >= self.max_requests:
            raise RunLimitError("The agent team reached its request limit.")
        self.requests += 1

    def new_tool_call_id(self) -> str:
        """Return an ID unique to this tool execution across the shared team run."""
        self.tool_call_count += 1
        return f"tool-{self.tool_call_count}"

    def record_usage(self, usage: dict | None) -> None:
        self.responses += 1
        self.usage_samples.append(deepcopy(usage) if isinstance(usage, dict) else None)
        if not isinstance(usage, dict) or not usage or not _sum_usage(self.usage, usage):
            self.usage_complete = False

    def summed_usage_metric(self, *path: str, integer: bool = False) -> int | float | None:
        """Sum one usage field only when every provider response reported it."""
        if not self.usage_samples or self.responses != self.requests:
            return None

        values = []
        for usage in self.usage_samples:
            value = usage
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
                or (integer and not isinstance(value, int))
            ):
                return None
            values.append(value)
        return sum(values)


register_tool(
    READ_TOOL_RESULT,
    "Read more of a tool result that was cut to fit model input. The content is "
    "serialized JSON text paged by character offset; use next_offset until it is null.",
    {
        "type": "object",
        "properties": {
            "result_id": {"type": "string", "minLength": 1},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": RESULT_PAGE_LIMIT},
        },
        "required": ["result_id"],
        "additionalProperties": False,
    },
)


def _read_tool_result(state: RunState) -> Tool:
    async def read(args: dict) -> dict:
        return read_page(state.tool_results, args)

    return bind(READ_TOOL_RESULT, read)


async def _invoke(tool: Tool, args: dict) -> tuple[Any, str | None, RunLimitError | None, float]:
    started = monotonic()
    try:
        return await tool.execute(args), None, None, monotonic() - started
    except RunLimitError as exc:
        return None, str(exc), exc, monotonic() - started
    except (ValueError, llm.LLMError) as exc:
        return None, str(exc), None, monotonic() - started


REQUEST_LIMIT_NOTICE = (
    "Request limit reached: no more tool calls are available. Answer now from the "
    "evidence you already retrieved, and state which parts of the request you could not review."
)
OUTPUT_LIMIT_NOTICE = (
    "Your last response reached the output limit. Give a complete, shorter answer. "
    "Do not repeat source text. Cite source lines when available. Do not call tools."
)
CITATION_NOTICE = (
    "Code check of your answer: these citations are not IDs that your tools returned, do "
    "not name one check-in message, or name an agent instead of a source: {ids}. For each "
    "claim, cite the message_id or record ID exactly as a tool returned it, or remove the "
    "claim. Do not cite a session_id or an agent. Then give the full corrected answer."
)
MAX_CALLS_PER_STEP = 16
# A model answer with neither text nor tool calls is requested this many times in all.
EMPTY_ANSWER_ATTEMPTS = 2
def _replayable(message: dict) -> dict:
    """The assistant message for the history, with unparsable tool arguments replaced by {}.

    The tool result reports the parse error; the unparsable text itself would make every
    later request of this run invalid.
    """
    calls = []
    for call in message.get("tool_calls") or []:
        arguments = call.get("function", {}).get("arguments")
        try:
            valid = isinstance(arguments, dict) or isinstance(json.loads(arguments), dict)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            call = {**call, "function": {**call["function"], "arguments": "{}"}}
        calls.append(call)
    return {**message, "tool_calls": calls}


class AnswerText:
    """All text of one agent run. The last request holds only the text after the last
    tool call, but text before a tool call is answer too. Text after tool events starts a
    new paragraph."""

    def __init__(self) -> None:
        self.text = ""
        self._new_paragraph = False

    def add(self, event: dict) -> None:
        if event["type"] == "answer_reset":
            self.text = ""
            self._new_paragraph = False
        elif event["type"] == "content":
            self.text += ("\n\n" if self._new_paragraph and self.text else "") + event["text"]
            self._new_paragraph = False
        elif event["type"] in {"tool_start", "tool_end"}:
            self._new_paragraph = True

    def answer(self, result: dict) -> dict:
        """The result with all text of the run as its message content."""
        if not self.text:
            return result
        return {**result, "message": {**result["message"], "content": self.text}}


class ReActAgent:
    def __init__(
        self,
        name: str,
        prompt: str,
        tools: dict[str, Tool],
    ):
        self.name = name
        self.prompt = prompt
        self.tools = tools

    async def run(
        self,
        messages: list[dict],
        *,
        state: RunState,
        sync: bool = False,
        transcript: list[dict] | None = None,
    ) -> dict:
        # Inside a tool call, forward tool lifecycle events to the calling agent's
        # stream. Prose and provider events stay private to this run.
        sink = _EVENT_SINK.get()
        text = AnswerText()
        stream = self.events(messages, state=state, sync=sync, transcript=transcript)
        async with aclosing(stream) as events:
            async for event in events:
                if event["type"] == "result":
                    return text.answer(event["result"])
                text.add(event)
                if sink is not None and event["type"] in {"tool_start", "tool_end"}:
                    sink(nested_event(event))
        raise llm.LLMError("The agent ended without an answer.")

    async def events(
        self,
        messages: list[dict],
        *,
        state: RunState,
        sync: bool = False,
        transcript: list[dict] | None = None,
    ) -> AsyncIterator[dict]:
        """The events of one run. When transcript is a list, it receives the run's messages
        without the system prompt, ending with the final answer, so a caller can continue
        the same run later."""
        history = deepcopy(messages)
        if self.prompt and (not history or history[0].get("role") != "system"):
            history.insert(0, {"role": "system", "content": self.prompt})
        tools = dict(self.tools)
        schemas = [tool.schema(name) for name, tool in tools.items()]
        # The agent takes tool steps until it answers. When the team's request budget
        # holds only the closing requests of the running agents, one closing request without tools
        # returns what was found instead of failing the caller's whole turn.
        state.active_agents += 1
        try:
            async with aclosing(self._steps(history, tools, schemas, state, sync)) as steps:
                async for event in steps:
                    if event["type"] == "result" and transcript is not None:
                        transcript[:] = [m for m in history if m.get("role") != "system"]
                    yield event
        finally:
            state.active_agents -= 1

    async def _steps(
        self, history: list[dict], tools: dict, schemas: list[dict], state: RunState, sync: bool
    ) -> AsyncIterator[dict]:
        step = 0
        force_closing = False
        citations_checked = False
        while True:
            closing = force_closing or not state.can_step()
            step += 1
            options: dict = {}
            if closing:
                history.append({"role": "user", "content": REQUEST_LIMIT_NOTICE})
                options["tool_choice"] = "none"
            state.reserve_request()
            request_messages = history
            request_options = options
            recovery_used = False
            result = None
            started = monotonic()
            for attempt in range(EMPTY_ANSWER_ATTEMPTS):
                streamed = False
                try:
                    if sync:
                        # Off the event loop: a blocking request would stall parallel
                        # agents and expire their tool timeouts.
                        result = await asyncio.to_thread(
                            llm.complete, request_messages, tools=schemas, **request_options
                        )
                    else:
                        stream = llm.stream(request_messages, tools=schemas, **request_options)
                        async with aclosing(stream) as events:
                            async for event in events:
                                if event["type"] == "result":
                                    result = event["result"]
                                else:
                                    if event["type"] == "content":
                                        streamed = True
                                    yield event
                    break
                except llm.LLMOutputLimit as error:
                    response = error.response
                    message = response.get("message")
                    draft = message.get("content") if isinstance(message, dict) else None
                    draft = draft if isinstance(draft, str) else ""
                    usage = response.get("usage")
                    state.record_usage(usage if isinstance(usage, dict) else None)
                    trace_record = {
                        "agent": self.name,
                        "tool": "output_limit",
                        "arguments": {"requested_max_tokens": error.max_tokens},
                        "observation": {
                            "finish_reason": response.get("finish_reason"),
                            "usage": usage,
                            "partial_content_length": len(draft),
                            "partial_draft": draft,
                            "recovery": "unavailable",
                        },
                        "error": None,
                    }
                    state.trace.append(trace_record)
                    has_tool_calls = error.has_tool_calls
                    free = state.max_requests - state.requests
                    can_recover = (
                        attempt + 1 < EMPTY_ANSWER_ATTEMPTS
                        and not recovery_used
                        and not streamed
                        and not has_tool_calls
                        and free >= state.active_agents
                    )
                    if not can_recover:
                        trace_record["error"] = str(error)
                        raise
                    recovery_used = True
                    force_closing = True
                    closing = True
                    trace_record["observation"]["recovery"] = "attempted"
                    request_messages = [
                        *history,
                        {"role": "user", "content": OUTPUT_LIMIT_NOTICE},
                    ]
                    request_options = {**options, "tool_choice": "none"}
                    state.reserve_request()
                except llm.LLMEmptyAnswer:
                    # An empty answer is asked again once, from the shared budget, only when
                    # the budget still holds the closing requests of the running agents.
                    # A closing step already used its own closing request, so it needs one
                    # free request beyond the closing requests of the other agents.
                    free = state.max_requests - state.requests
                    room = free >= state.active_agents if closing else state.can_step()
                    if attempt + 1 == EMPTY_ANSWER_ATTEMPTS or streamed or not room:
                        raise
                    state.reserve_request()
            if result is None:
                raise llm.LLMError("The response stream ended before completion.")
            state.record_usage(result.get("usage"))
            message = result["message"]
            calls = message.get("tool_calls") or []
            # One record for each answered step, so cost and latency are known per agent.
            state.trace.append(
                {
                    "agent": self.name,
                    "tool": "llm_request",
                    "step": step,
                    "model": result.get("model"),
                    "provider": result.get("provider"),
                    "usage": result.get("usage"),
                    "finish_reason": result.get("finish_reason"),
                    "latency": round(monotonic() - started, 3),
                    "tool_calls": len(calls),
                    "text_length": len(message.get("content") or ""),
                }
            )
            if not calls:
                history.append({"role": "assistant", "content": message["content"]})
                unresolved = (
                    []
                    if closing or citations_checked or not state.governed
                    else unresolved_citations(message["content"], known_ids(state.trace, self.name))
                )
                if unresolved and state.can_step():
                    # One repair round: the agent sees the IDs the code could not resolve.
                    citations_checked = True
                    state.trace.append(
                        {
                            "agent": self.name,
                            "tool": "citation_check",
                            "arguments": {"unresolved": unresolved},
                            "observation": None,
                            "error": None,
                        }
                    )
                    history.append(
                        {"role": "user", "content": CITATION_NOTICE.format(ids=", ".join(unresolved))}
                    )
                    yield {"type": "answer_reset"}
                    continue
                if sync or not streamed:
                    yield {"type": "content", "text": message["content"]}
                yield {"type": "result", "result": result}
                return
            if sync and message.get("content"):
                # A streamed run already sent this text; a sync run sends it here, so the
                # caller can keep answer text that came with tool calls.
                yield {"type": "content", "text": message["content"]}
            if closing:
                raise RunLimitError(
                    f"{self.name} reached the request limit without a final answer."
                )
            if result.get("finish_reason") != "tool_calls":
                raise llm.LLMError("The provider did not complete its tool calls.")
            history.append(_replayable(message))
            items = []
            for index, call in enumerate(calls):
                item = self._prepare(call, tools, state)
                item["step"] = step
                if index >= MAX_CALLS_PER_STEP and item["error"] is None:
                    # The first calls run; the agent requests the rest in its next step.
                    item["error"] = (
                        f"Not run: {len(calls)} tool calls requested in one step and only the "
                        f"first {MAX_CALLS_PER_STEP} run. Request this call again if needed."
                    )
                items.append(item)
                yield item["start_event"]
            async for event in self._run_tools(items, tools, state):
                yield event
            limit_errors = [item["limit_error"] for item in items if item["limit_error"]]
            if limit_errors:
                raise limit_errors[0]
            # Tool messages follow the provider's call order, whatever order calls finished in.
            for item in items:
                content = model_observation(
                    item["name"], item["observation"], item["call_id"], state.tool_results
                )
                if item["call_id"] in state.tool_results and READ_TOOL_RESULT not in tools:
                    # Offer the reader only after a cut, so short turns keep their tool set.
                    tools[READ_TOOL_RESULT] = _read_tool_result(state)
                    schemas.append(tools[READ_TOOL_RESULT].schema(READ_TOOL_RESULT))
                history.append(
                    {"role": "tool", "tool_call_id": item["call"]["id"], "content": content}
                )

    def _prepare(self, call: dict, tools: dict[str, Tool], state: RunState) -> dict:
        name = call["function"]["name"]
        call_id = state.new_tool_call_id()
        raw_arguments = call["function"].get("arguments")
        args = None
        error = None
        if isinstance(raw_arguments, str):
            try:
                parsed_arguments = json.loads(raw_arguments)
                if not isinstance(parsed_arguments, dict):
                    raise TypeError("Tool arguments must be a JSON object.")
                args = parsed_arguments
            except (ValueError, TypeError) as exc:
                error = str(exc)
        else:
            error = "Tool arguments must be JSON text."
        if error is None and name not in tools:
            error = f"Unknown tool: {name}"
        if error is None:
            parameters = tools[name].parameters
            missing = [key for key in parameters.get("required", []) if key not in args]
            unknown = [key for key in args if key not in parameters.get("properties", {})]
            if missing:
                error = f"Missing required arguments: {', '.join(missing)}."
            elif unknown and parameters.get("additionalProperties") is False:
                error = f"Unknown arguments: {', '.join(unknown)}."
        start_event = {
            "type": "tool_start",
            "call_id": call_id,
            "agent": self.name,
            "name": name,
            "arguments": args,
        }
        if args is None and isinstance(raw_arguments, str):
            start_event["raw_arguments"] = raw_arguments
        return {
            "call": call,
            "name": name,
            "call_id": call_id,
            "raw_arguments": raw_arguments,
            "args": args,
            "error": error,
            "observation": None,
            "limit_error": None,
            "duration": 0.0,
            "start_event": start_event,
        }

    async def _run_tools(
        self, items: list[dict], tools: dict[str, Tool], state: RunState
    ) -> AsyncIterator[dict]:
        """Run one step's tool calls concurrently and yield events as they happen.

        Nested tool events from a specialist run arrive through a queue while
        the calls run. Each call's tool_end is yielded when that call finishes.
        """
        queue: asyncio.Queue[dict] = asyncio.Queue()
        tasks: dict[asyncio.Task, dict] = {}
        for item in items:
            if item["error"] is not None:
                continue
            context = copy_context()
            context.run(_EVENT_SINK.set, queue.put_nowait)
            task = asyncio.create_task(_invoke(tools[item["name"]], item["args"]), context=context)
            tasks[task] = item
        for item in items:
            if item["error"] is not None:
                yield self._finish(item, state)
        pending = set(tasks)
        try:
            while pending:
                getter = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait(
                    pending | {getter}, return_when=asyncio.FIRST_COMPLETED
                )
                if getter in done:
                    yield getter.result()
                else:
                    getter.cancel()
                while not queue.empty():
                    yield queue.get_nowait()
                for task in done - {getter}:
                    pending.discard(task)
                    item = tasks[task]
                    observation, error, limit_error, duration = task.result()
                    item.update(
                        observation=observation,
                        error=error,
                        limit_error=limit_error,
                        duration=duration,
                    )
                    yield self._finish(item, state)
        finally:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    def _finish(self, item: dict, state: RunState) -> dict:
        """Record a finished call in the trace and return its tool_end."""
        name, args, error = item["name"], item["args"], item["error"]
        if error is not None:
            item["observation"] = {"error": error}
        observation = item["observation"]
        record = {
            "agent": self.name,
            "tool": name,
            "step": item["step"],
            "call_id": item["call_id"],
            "duration": round(item["duration"], 3),
            "arguments": args,
            "observation": observation,
            "error": error,
        }
        if args is None and isinstance(item["raw_arguments"], str):
            record["raw_arguments"] = item["raw_arguments"]
        state.trace.append(record)
        return {
            "type": "tool_end",
            "call_id": item["call_id"],
            "agent": self.name,
            "name": name,
            "duration": item["duration"],
            "error": error,
            "observation": observation,
        }
