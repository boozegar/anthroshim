from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional


Json = Dict[str, Any]


@dataclass
class _ToolCallState:
    item_id: str
    name: str
    call_id: str
    # OpenAI streams function_call arguments as string deltas.
    partial_json: str = ""
    emitted_chars: int = 0
    done: bool = False


@dataclass
class _StreamState:
    message_id: str
    model: str
    started: bool = False
    content_index: int = 0
    active_block: Optional[str] = None  # "text" or "tool:<item_id>"
    active_index: Optional[int] = None
    tool_calls: Dict[str, _ToolCallState] = field(default_factory=dict)
    tool_queue: List[str] = field(
        default_factory=list
    )  # item_ids in the order they should appear
    pending_text: List[str] = field(default_factory=list)
    last_emitted_block_type: Optional[str] = None
    response_usage: Optional[Json] = None
    stop_reason: Optional[str] = None
    reasoning_summary: str = ""
    reasoning_emitted: bool = False


def iter_anthropic_events(
    openai_events: Iterable[Json],
    *,
    model: str = "unknown",
    message_id: Optional[str] = None,
    keep_reasoning_summary: bool = False,
) -> Iterator[Json]:
    """Convert OpenAI Responses streaming events into Anthropic Messages streaming events.

    Input: iterable of OpenAI SSE event payloads (already JSON-decoded dicts).
    Output: iterable of Anthropic SSE event payloads (dicts with `type`).

    Notes:
    - OpenAI `reasoning_*` stream events are dropped (except reasoning_summary when enabled).
    - Content blocks are emitted sequentially (no interleaving deltas across blocks).
    - Tool call inputs are streamed via `input_json_delta` partial JSON.
    """

    st = _StreamState(
        message_id=message_id or f"msg_{uuid.uuid4().hex}",
        model=model,
    )

    for ev in openai_events:
        if not isinstance(ev, dict):
            continue
        et = ev.get("type")

        if isinstance(et, str) and et.startswith("response.reasoning_summary"):
            if not keep_reasoning_summary:
                continue
            if et.endswith(".delta"):
                delta = ev.get("delta")
                if delta:
                    st.reasoning_summary += str(delta)
            elif et.endswith(".done"):
                summary = ev.get("summary") or ev.get("text") or ev.get("delta")
                if summary:
                    st.reasoning_summary = str(summary)
            else:
                summary = ev.get("summary") or ev.get("text")
                if summary:
                    st.reasoning_summary = str(summary)
            continue

        # Drop other reasoning stream events by default.
        if isinstance(et, str) and et.startswith("response.reasoning"):
            continue

        if et == "response.created":
            resp = ev.get("response") or {}
            m = resp.get("model") if isinstance(resp, dict) else None
            if isinstance(m, str) and m:
                st.model = m
            # Don't emit anything yet; wait for actual content.
            continue

        if et == "response.output_item.added":
            item = ev.get("item") or {}
            itype = item.get("type")
            if itype == "reasoning" and keep_reasoning_summary:
                summary = item.get("summary") or item.get("text")
                if summary:
                    st.reasoning_summary = str(summary)
                continue
            if itype == "function_call":
                item_id = str(item.get("id") or ev.get("item_id") or "")
                st.tool_calls[item_id] = _ToolCallState(
                    item_id=item_id,
                    name=str(item.get("name") or ""),
                    call_id=str(item.get("call_id") or item.get("id") or item_id),
                )
                st.tool_queue.append(item_id)
            elif itype == "custom_tool_call":
                # We'll treat custom tool input as a JSON object {"input": "..."}.
                item_id = str(item.get("id") or ev.get("item_id") or "")
                st.tool_calls[item_id] = _ToolCallState(
                    item_id=item_id,
                    name=str(item.get("name") or ""),
                    call_id=str(item.get("call_id") or item.get("id") or item_id),
                )
                st.tool_queue.append(item_id)
            continue

        if et == "response.output_text.delta":
            yield from _ensure_message_started(st)
            delta = ev.get("delta") or ""
            if delta:
                # If a tool block is active, buffer text until tool finishes.
                if st.active_block and st.active_block.startswith("tool:"):
                    st.pending_text.append(str(delta))
                else:
                    yield from _ensure_text_block(st)
                    yield {
                        "type": "content_block_delta",
                        "index": st.active_index,
                        "delta": {"type": "text_delta", "text": str(delta)},
                    }
            continue

        if et == "response.output_text.done":
            # End of a text content part; do nothing. We'll stop blocks on item.done/response.completed.
            continue

        if et == "response.refusal.delta":
            # Represent refusal as text (it is effectively a visible assistant output).
            yield from _ensure_message_started(st)
            delta = ev.get("delta") or ""
            if delta:
                if st.active_block and st.active_block.startswith("tool:"):
                    st.pending_text.append(str(delta))
                else:
                    yield from _ensure_text_block(st)
                    yield {
                        "type": "content_block_delta",
                        "index": st.active_index,
                        "delta": {"type": "text_delta", "text": str(delta)},
                    }
            continue

        if et == "response.function_call_arguments.delta":
            item_id = str(ev.get("item_id") or "")
            delta = ev.get("delta")
            tc = st.tool_calls.get(item_id)
            if tc is None:
                # Unknown tool call; ignore.
                continue
            delta_str = str(delta or "")
            tc.partial_json += delta_str
            yield from _ensure_message_started(st)
            # Only stream deltas if this tool is the active block.
            yield from _ensure_tool_block(st, item_id, emit_buffered=False)
            if st.active_block == f"tool:{item_id}" and delta is not None:
                # If we buffered earlier chunks before the tool block became active, flush them first.
                buffered_len = len(tc.partial_json) - len(delta_str)
                if tc.emitted_chars < buffered_len:
                    prefix = tc.partial_json[tc.emitted_chars:buffered_len]
                    if prefix:
                        yield {
                            "type": "content_block_delta",
                            "index": st.active_index,
                            "delta": {"type": "input_json_delta", "partial_json": prefix},
                        }
                        tc.emitted_chars = buffered_len
                yield {
                    "type": "content_block_delta",
                    "index": st.active_index,
                    "delta": {"type": "input_json_delta", "partial_json": delta_str},
                }
                tc.emitted_chars += len(delta_str)
            continue

        if et == "response.function_call_arguments.done":
            item_id = str(ev.get("item_id") or "")
            tc = st.tool_calls.get(item_id)
            if tc is None:
                continue
            tc.done = True
            # Ensure tool block exists and (if not yet streamed) push any remaining json.
            yield from _ensure_message_started(st)
            yield from _ensure_tool_block(st, item_id, emit_buffered=True)
            # Some OpenAI streams may only emit done with full arguments; emit as a delta if we never streamed.
            args = ev.get("arguments")
            if isinstance(args, str):
                # If we didn't stream anything yet, send the full arguments.
                if tc.partial_json == "":
                    tc.partial_json = args
                # If tool block is active, flush any un-emitted suffix.
                if st.active_block == f"tool:{item_id}" and tc.emitted_chars < len(
                    tc.partial_json
                ):
                    suffix = tc.partial_json[tc.emitted_chars :]
                    if suffix:
                        yield {
                            "type": "content_block_delta",
                            "index": st.active_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": suffix,
                            },
                        }
                        tc.emitted_chars = len(tc.partial_json)
            continue

        if et == "response.custom_tool_call_input.delta":
            # Custom tool input is not JSON; buffer and emit at .done.
            item_id = str(ev.get("item_id") or "")
            delta = ev.get("delta")
            tc = st.tool_calls.get(item_id)
            if tc is None:
                continue
            tc.partial_json += str(delta or "")
            continue

        if et == "response.custom_tool_call_input.done":
            item_id = str(ev.get("item_id") or "")
            tc = st.tool_calls.get(item_id)
            if tc is None:
                continue
            tc.done = True
            # Convert accumulated raw input into JSON object.
            raw = ev.get("input")
            if raw is None:
                raw = tc.partial_json
            tc.partial_json = json.dumps({"input": str(raw)}, ensure_ascii=True)
            yield from _ensure_message_started(st)
            yield from _ensure_tool_block(st, item_id, emit_buffered=True)
            yield {
                "type": "content_block_delta",
                "index": st.active_index,
                "delta": {"type": "input_json_delta", "partial_json": tc.partial_json},
            }
            continue

        if et == "response.output_item.done":
            item = ev.get("item") or {}
            itype = item.get("type")

            if itype == "message":
                # Close any active text block.
                yield from _close_active_block(st)
                continue

            if itype in {"function_call", "custom_tool_call"}:
                item_id = str(item.get("id") or "")
                # Ensure tool block is open and then close it.
                yield from _ensure_message_started(st)
                yield from _ensure_tool_block(st, item_id, emit_buffered=True)
                yield from _close_active_block(st)
                # After closing a tool block, flush buffered text (if any).
                if st.pending_text:
                    yield from _ensure_text_block(st)
                    while st.pending_text:
                        chunk = st.pending_text.pop(0)
                        yield {
                            "type": "content_block_delta",
                            "index": st.active_index,
                            "delta": {"type": "text_delta", "text": chunk},
                        }
                continue

            continue

        if et in {"response.completed", "response.incomplete", "response.failed"}:
            resp = ev.get("response") or {}
            if isinstance(resp, dict):
                st.response_usage = resp.get("usage")
                if keep_reasoning_summary and not st.reasoning_summary:
                    summary = resp.get("reasoning_summary")
                    if isinstance(summary, str) and summary.strip():
                        st.reasoning_summary = summary
                inc = resp.get("incomplete_details")
                if isinstance(inc, dict) and inc.get("reason") == "max_tokens":
                    st.stop_reason = "max_tokens"

                # If the response has a terminal tool call as output, match Anthropic's stop_reason.
                out = resp.get("output")
                if isinstance(out, list) and out:
                    last = out[-1]
                    if isinstance(last, dict) and last.get("type") in {
                        "function_call",
                        "custom_tool_call",
                    }:
                        st.stop_reason = st.stop_reason or "tool_use"
                st.stop_reason = st.stop_reason or "end_turn"

            # Close any active block and end message.
            yield from _close_active_block(st)
            if keep_reasoning_summary and st.reasoning_summary and not st.reasoning_emitted:
                yield from _emit_thinking_block(st, st.reasoning_summary)
                st.reasoning_emitted = True
            yield from _ensure_message_started(st)
            usage = {}
            if (
                isinstance(st.response_usage, dict)
                and "output_tokens" in st.response_usage
            ):
                usage = {"output_tokens": st.response_usage.get("output_tokens")}
            yield {
                "type": "message_delta",
                "delta": {"stop_reason": st.stop_reason, "stop_sequence": None},
                "usage": usage,
            }
            yield {"type": "message_stop"}
            return

    # If stream ends without an explicit completed event, close best-effort.
    if st.started:
        yield from _close_active_block(st)
        if keep_reasoning_summary and st.reasoning_summary and not st.reasoning_emitted:
            yield from _emit_thinking_block(st, st.reasoning_summary)
            st.reasoning_emitted = True
        yield {
            "type": "message_delta",
            "delta": {
                "stop_reason": st.stop_reason or "end_turn",
                "stop_sequence": None,
            },
            "usage": {},
        }
        yield {"type": "message_stop"}


def iter_openai_sse_json_events(lines: Iterable[str]) -> Iterator[Json]:
    """Parse OpenAI SSE (text lines) into JSON event dicts.

    This is intentionally minimal:
    - Collects one SSE event at a time (separated by blank line)
    - Joins all `data:` lines (without the `data:` prefix)
    - Ignores `data: [DONE]`

    Use this when you have raw SSE text from `responses.create(stream=True)`.
    """

    data_buf: List[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if line == "":
            if not data_buf:
                continue
            payload = "\n".join(data_buf).strip()
            data_buf = []
            if payload == "[DONE]":
                continue
            try:
                ev = json.loads(payload)
            except Exception:
                continue
            if isinstance(ev, dict):
                yield ev
            continue

        if line.startswith("data:"):
            data_buf.append(line[len("data:") :].lstrip())

    # Flush trailing buffer if stream ended without a blank line.
    if data_buf:
        payload = "\n".join(data_buf).strip()
        if payload and payload != "[DONE]":
            try:
                ev = json.loads(payload)
            except Exception:
                return
            if isinstance(ev, dict):
                yield ev


def iter_anthropic_sse_lines(anthropic_events: Iterable[Json]) -> Iterator[str]:
    """Convert Anthropic event dicts into SSE text lines."""

    for ev in anthropic_events:
        if not isinstance(ev, dict) or "type" not in ev:
            continue
        et = ev.get("type")
        if not isinstance(et, str):
            continue
        yield f"event: {et}\n"
        yield "data: " + json.dumps(ev, ensure_ascii=True) + "\n\n"


def _ensure_message_started(st: _StreamState) -> List[Json]:
    if st.started:
        return []
    st.started = True
    st.last_emitted_block_type = None
    return [
        {
            "type": "message_start",
            "message": {
                "id": st.message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": st.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
    ]


def _ensure_text_block(st: _StreamState) -> List[Json]:
    if st.active_block == "text":
        return []
    out: List[Json] = []
    out.extend(_close_active_block(st))
    st.active_block = "text"
    st.active_index = st.content_index
    st.content_index += 1
    st.last_emitted_block_type = "text"
    out.append(
        {
            "type": "content_block_start",
            "index": st.active_index,
            "content_block": {"type": "text", "text": ""},
        }
    )
    return out


def _ensure_tool_block(
    st: _StreamState,
    item_id: str,
    *,
    emit_buffered: bool = True,
) -> List[Json]:
    if st.active_block == f"tool:{item_id}":
        return []

    # Only allow tool blocks in the order they were added.
    if st.tool_queue and st.tool_queue[0] != item_id:
        return []

    tc = st.tool_calls.get(item_id)
    if tc is None:
        return []

    out: List[Json] = []
    out.extend(_close_active_block(st))
    st.active_block = f"tool:{item_id}"
    st.active_index = st.content_index
    st.content_index += 1
    st.last_emitted_block_type = "tool_use"

    out.append(
        {
            "type": "content_block_start",
            "index": st.active_index,
            "content_block": {
                "type": "tool_use",
                "id": tc.call_id,
                "name": tc.name,
                "input": {},
            },
        }
    )

    # Anthropic examples emit an empty input_json_delta first; do that for compatibility.
    out.append(
        {
            "type": "content_block_delta",
            "index": st.active_index,
            "delta": {"type": "input_json_delta", "partial_json": ""},
        }
    )

    if emit_buffered:
        # If we buffered JSON before this block became active, emit the suffix now.
        if tc.partial_json and tc.emitted_chars < len(tc.partial_json):
            suffix = tc.partial_json[tc.emitted_chars :]
            if suffix:
                out.append(
                    {
                        "type": "content_block_delta",
                        "index": st.active_index,
                        "delta": {"type": "input_json_delta", "partial_json": suffix},
                    }
                )
                tc.emitted_chars = len(tc.partial_json)
    return out


def _close_active_block(st: _StreamState) -> List[Json]:
    if st.active_block is None:
        return []
    idx = st.active_index
    st.active_block = None
    st.active_index = None
    out = [{"type": "content_block_stop", "index": idx}]

    # If we just closed a tool block, advance the queue.
    if st.tool_queue and st.last_emitted_block_type == "tool_use":
        st.tool_queue.pop(0)

    return out


def _emit_thinking_block(st: _StreamState, summary: str) -> List[Json]:
    text = str(summary or "").strip()
    if not text:
        return []
    out: List[Json] = []
    out.extend(_ensure_message_started(st))
    out.extend(_close_active_block(st))
    idx = st.content_index
    st.content_index += 1
    st.last_emitted_block_type = "thinking"
    out.append(
        {
            "type": "content_block_start",
            "index": idx,
            "content_block": {
                "type": "thinking",
                "thinking": "",
                "signature": "",
            },
        }
    )
    out.append(
        {
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "thinking_delta", "thinking": text},
        }
    )
    out.append({"type": "content_block_stop", "index": idx})
    return out
