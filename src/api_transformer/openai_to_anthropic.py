from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


Json = Dict[str, Any]


def convert_openai_to_anthropic(
    data: Any,
    *,
    mode: str = "auto",
    keep_reasoning: bool = False,
    keep_unknown: bool = False,
    keep_reasoning_summary: bool = False,
) -> Json:
    """Convert OpenAI Responses API payloads into Anthropic Messages request shape.

    Supports:
    - responses.create(input=...) payloads (list/dict of messages)
    - response objects (expects a top-level "output" list)
    - raw output item lists (response.output)
    """

    instructions: Optional[str] = None
    items: List[Json]

    if mode not in {"auto", "input", "response", "output"}:
        raise ValueError("mode must be one of: auto, input, response, output")

    if mode == "auto":
        mode = _auto_detect_mode(data)

    if mode == "response":
        if not isinstance(data, dict) or not isinstance(data.get("output"), list):
            raise TypeError("mode=response expects a dict response object with an 'output' list")
        instructions = data.get("instructions")
        items = list(data.get("output") or [])
    elif mode == "output":
        if not isinstance(data, list):
            raise TypeError("mode=output expects a list (response.output)")
        items = list(data)
    elif mode == "input":
        items = _normalize_openai_input_to_items(data)
        # OpenAI input may be accompanied by separate instructions in your app; keep None here.
        instructions = None
    else:
        raise AssertionError("unreachable")

    system, messages = openai_items_to_anthropic_messages(
        items,
        instructions=instructions,
        keep_reasoning=keep_reasoning,
        keep_unknown=keep_unknown,
        keep_reasoning_summary=keep_reasoning_summary,
    )

    out: Json = {"messages": messages}
    if system is not None:
        out["system"] = system
    return out


def openai_items_to_anthropic_messages(
    items: Iterable[Json],
    *,
    instructions: Optional[str] = None,
    keep_reasoning: bool = False,
    keep_unknown: bool = False,
    keep_reasoning_summary: bool = False,
) -> Tuple[Optional[Union[str, List[Json]]], List[Json]]:
    """Convert OpenAI item list to Anthropic Messages `messages` array.

    Returns (system, messages).
    """

    system: Optional[Union[str, List[Json]]] = instructions
    messages: List[Json] = []

    def ensure_message(role: str) -> Json:
        if messages and messages[-1].get("role") == role:
            return messages[-1]
        msg = {"role": role, "content": []}
        messages.append(msg)
        return msg

    for item in items:
        if not isinstance(item, dict):
            if keep_unknown:
                ensure_message("assistant")["content"].append({"type": "text", "text": str(item)})
            continue

        t = item.get("type")

        if t == "reasoning":
            summary = _extract_reasoning_summary(item)
            if keep_reasoning_summary and summary:
                ensure_message("assistant")["content"].append(_thinking_block(summary))
            elif keep_reasoning:
                ensure_message("assistant")["content"].append(
                    {"type": "text", "text": "[openai_reasoning]"}
                )
            continue

        if t == "message" or (t is None and "role" in item and "content" in item):
            role = item.get("role")
            if role not in {"user", "assistant"}:
                role = "user" if role == "system" else "assistant"

            # If OpenAI injected system via role=system, hoist it.
            if item.get("role") == "system":
                sys_text = _extract_openai_text(item.get("content"))
                if sys_text:
                    system = sys_text
                continue

            msg = ensure_message(role)
            msg["content"].extend(_openai_content_to_anthropic_blocks(item.get("content"), keep_unknown=keep_unknown))
            continue

        if t == "function_call":
            # Convert to Anthropic tool_use block (assistant content)
            call_id = item.get("call_id") or item.get("id")
            name = item.get("name")
            args = item.get("arguments")
            tool_input: Any = {}
            if isinstance(args, str) and args.strip():
                try:
                    tool_input = json.loads(args)
                except Exception:
                    tool_input = {"_raw": args}
            ensure_message("assistant")["content"].append(
                {"type": "tool_use", "id": str(call_id), "name": str(name), "input": tool_input}
            )
            continue

        if t == "custom_tool_call":
            call_id = item.get("call_id") or item.get("id")
            name = item.get("name")
            tool_input = {"input": item.get("input", "")}
            ensure_message("assistant")["content"].append(
                {"type": "tool_use", "id": str(call_id), "name": str(name), "input": tool_input}
            )
            continue

        if t == "function_call_output":
            # Anthropic expects tool_result blocks in a user message
            call_id = item.get("call_id")
            output = item.get("output")
            if isinstance(output, (dict, list)):
                output = json.dumps(output, ensure_ascii=True)
            ensure_message("user")["content"].append(
                {"type": "tool_result", "tool_use_id": str(call_id), "content": str(output or "")}
            )
            continue

        if keep_unknown:
            ensure_message("assistant")["content"].append({"type": "text", "text": json.dumps(item, ensure_ascii=True)})

    # Anthropic allows content to be a string shorthand; keep it as blocks.
    # If any message has empty content, drop it.
    messages = [m for m in messages if m.get("content")]
    return system, messages


def _auto_detect_mode(data: Any) -> str:
    if isinstance(data, dict) and isinstance(data.get("output"), list):
        return "response"
    if isinstance(data, list):
        # A raw output items list OR an input message list
        if data and isinstance(data[0], dict) and "type" in data[0] and data[0].get("type") in {
            "message",
            "function_call",
            "reasoning",
            "custom_tool_call",
        }:
            return "output"
        if data and isinstance(data[0], dict) and "role" in data[0]:
            return "input"
        return "output"
    if isinstance(data, dict) and "role" in data and "content" in data:
        return "input"
    raise TypeError("Could not auto-detect mode for provided data")


def _normalize_openai_input_to_items(data: Any) -> List[Json]:
    # OpenAI Responses input can be a string, or a list of {role, content}, or a single object.
    if isinstance(data, str):
        return [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": data}]}]

    if isinstance(data, dict) and "role" in data and "content" in data:
        return [{"type": "message", "role": data.get("role"), "content": _normalize_openai_content(data.get("content"))}]

    if isinstance(data, list):
        out: List[Json] = []
        for m in data:
            if isinstance(m, dict) and "role" in m and "content" in m:
                out.append({"type": "message", "role": m.get("role"), "content": _normalize_openai_content(m.get("content"))})
            elif isinstance(m, str):
                out.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": m}]})
        return out

    raise TypeError("Unsupported OpenAI input shape")


def _normalize_openai_content(content: Any) -> List[Json]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if isinstance(content, list):
        return [c for c in content if isinstance(c, dict)]
    return [{"type": "input_text", "text": str(content)}]


def _openai_content_to_anthropic_blocks(content: Any, *, keep_unknown: bool) -> List[Json]:
    blocks: List[Json] = []
    if content is None:
        return blocks

    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                if keep_unknown:
                    blocks.append({"type": "text", "text": str(part)})
                continue
            ptype = part.get("type")
            if ptype in {"input_text", "output_text"}:
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif ptype in {"input_image", "image"}:
                # OpenAI input_image can be {type: input_image, image_url: ...} or {image_url:{url:...}}
                src = _openai_image_part_to_anthropic_source(part)
                if src is not None:
                    blocks.append({"type": "image", "source": src})
            else:
                if keep_unknown:
                    blocks.append({"type": "text", "text": json.dumps(part, ensure_ascii=True)})
        return blocks

    # Fallback
    return [{"type": "text", "text": str(content)}]


def _openai_image_part_to_anthropic_source(part: Json) -> Optional[Json]:
    # OpenAI message.input_image.image_url includes urls optionally.
    # Accept common shapes: {"image_url": "https://..."} or {"image_url": {"url": "..."}}
    # or {"url": "..."}
    url = None
    if isinstance(part.get("image_url"), str):
        url = part.get("image_url")
    elif isinstance(part.get("image_url"), dict):
        url = part.get("image_url", {}).get("url")
    elif isinstance(part.get("url"), str):
        url = part.get("url")
    if url:
        return {"type": "url", "url": url}

    # Base64 shape not standardized here; leave None.
    return None


def _extract_openai_text(content: Any) -> Optional[str]:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: List[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") in {"input_text", "output_text"}:
                texts.append(str(p.get("text") or ""))
        joined = "".join(texts).strip()
        return joined or None
    return str(content)


def _extract_reasoning_summary(item: Json) -> Optional[str]:
    summary = item.get("summary") or item.get("text")
    if isinstance(summary, str) and summary.strip():
        return summary
    return None


def _thinking_block(summary: str) -> Json:
    return {
        "type": "thinking",
        "thinking": str(summary),
        "signature": "",
    }
