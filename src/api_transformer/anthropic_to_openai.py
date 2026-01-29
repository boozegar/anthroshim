from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, Iterable, List, Optional

Json = Dict[str, Any]


def convert_anthropic_to_openai_request(
    payload: Json,
    *,
    thinking_enabled: bool = False,
    thinking_config: Optional[Json] = None,
) -> Json:
    """Convert Anthropic Messages request into OpenAI Responses request payload."""

    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")

    model = payload.get("model")
    messages = payload.get("messages") or []
    system = payload.get("system")

    items = anthropic_messages_to_openai_items(messages)

    out: Json = {"model": model, "input": items}
    instructions = _anthropic_system_to_text(system)
    if instructions:
        out["instructions"] = instructions

    max_tokens = payload.get("max_tokens")
    if isinstance(max_tokens, int):
        out["max_output_tokens"] = max_tokens

    if "temperature" in payload:
        out["temperature"] = payload.get("temperature")
    if "top_p" in payload:
        out["top_p"] = payload.get("top_p")

    tools = payload.get("tools")
    if isinstance(tools, list):
        out["tools"] = _anthropic_tools_to_openai_tools(tools)

    tool_choice = payload.get("tool_choice")
    if tool_choice is not None:
        out["tool_choice"] = _anthropic_tool_choice_to_openai(tool_choice)

    stream = payload.get("stream")
    if isinstance(stream, bool):
        out["stream"] = stream

    if thinking_enabled and isinstance(thinking_config, dict) and thinking_config:
        out["reasoning"] = _merge_reasoning(out.get("reasoning"), thinking_config)

    return out


def anthropic_messages_to_openai_items(messages: Iterable[Json]) -> List[Json]:
    items: List[Json] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue

        blocks = _normalize_anthropic_content(msg.get("content"))
        text_part_type = "input_text" if role == "user" else "output_text"
        cur_parts: List[Json] = []

        def flush_message() -> None:
            nonlocal cur_parts
            if not cur_parts:
                return
            items.append({"type": "message", "role": role, "content": cur_parts})
            cur_parts = []

        for block in blocks:
            btype = block.get("type")
            if btype == "text":
                cur_parts.append(
                    {"type": text_part_type, "text": str(block.get("text") or "")}
                )
                continue

            if btype == "image":
                image_part_type = (
                    "output_image" if role == "assistant" else "input_image"
                )
                part = _anthropic_image_to_openai_part(block, image_part_type)
                if part is not None:
                    cur_parts.append(part)
                continue

            if btype == "tool_use":
                flush_message()
                items.append(_anthropic_tool_use_to_openai_item(block))
                continue

            if btype == "tool_result":
                flush_message()
                items.append(_anthropic_tool_result_to_openai_item(block))
                continue

            # Unknown block types become text for safety.
            cur_parts.append(
                {"type": text_part_type, "text": json.dumps(block, ensure_ascii=True)}
            )

        flush_message()

    return items


def _normalize_anthropic_content(content: Any) -> List[Json]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        out: List[Json] = []
        for c in content:
            if isinstance(c, dict):
                out.append(c)
                continue
            d = _maybe_block_to_dict(c)
            if d is not None:
                out.append(d)
        return out
    return [{"type": "text", "text": str(content)}]


def _maybe_block_to_dict(obj: Any) -> Optional[Json]:
    if isinstance(obj, dict):
        return obj
    for meth in ("model_dump", "dict"):
        fn = getattr(obj, meth, None)
        if callable(fn):
            try:
                data = fn()
            except Exception:
                data = None
            if isinstance(data, dict) and "type" in data:
                return data
    data = getattr(obj, "__dict__", None)
    if isinstance(data, dict) and "type" in data:
        return data
    obj_type = getattr(obj, "type", None)
    if obj_type:
        out: Json = {"type": obj_type}
        for key in ("text", "source", "name", "id", "input", "tool_use_id", "content"):
            if hasattr(obj, key):
                out[key] = getattr(obj, key)
        return out
    return None


def _anthropic_system_to_text(system: Any) -> Optional[str]:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for b in system:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text") or ""))
        text = "".join(parts).strip()
        return text or None
    return str(system)


def _anthropic_tools_to_openai_tools(tools: List[Json]) -> List[Json]:
    out: List[Json] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        if not name:
            continue
        out.append(
            {
                "type": "function",
                "name": name,
                "description": t.get("description"),
                "parameters": t.get("input_schema") or {},
            }
        )
    return out


def _anthropic_tool_choice_to_openai(choice: Any) -> Any:
    if isinstance(choice, str):
        return choice
    if isinstance(choice, dict):
        if choice.get("type") == "tool" and choice.get("name"):
            return {"type": "function", "name": choice.get("name")}
    return choice


def _anthropic_tool_use_to_openai_item(block: Json) -> Json:
    call_id = str(block.get("id") or f"call_{uuid.uuid4().hex}")
    name = str(block.get("name") or "")
    tool_input = block.get("input")
    if isinstance(tool_input, str):
        args = tool_input
    else:
        args = json.dumps(tool_input or {}, ensure_ascii=True)
    return {
        "type": "function_call",
        "id": f"fc_{uuid.uuid4().hex}",
        "call_id": call_id,
        "name": name,
        "arguments": args,
    }


def _anthropic_tool_result_to_openai_item(block: Json) -> Json:
    call_id = str(block.get("tool_use_id") or "")
    content = block.get("content")
    if isinstance(content, list):
        content = _anthropic_blocks_to_text(content)
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": "" if content is None else str(content),
    }


def _anthropic_blocks_to_text(blocks: Iterable[Json]) -> str:
    parts: List[str] = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(str(b.get("text") or ""))
    return "".join(parts)


def _anthropic_image_to_openai_part(
    block: Json, image_part_type: str = "input_image"
) -> Optional[Json]:
    source = _normalize_anthropic_image_source(block.get("source"))
    if not source:
        return None
    use_object = _openai_image_url_object()
    stype = source.get("type")
    if stype == "url":
        url = source.get("url")
        if isinstance(url, str) and url:
            if use_object:
                return {"type": image_part_type, "image_url": {"url": url}}
            return {"type": image_part_type, "image_url": url}
    if stype == "base64":
        data = source.get("data")
        media_type = source.get("media_type") or "application/octet-stream"
        if isinstance(data, str) and data:
            data_url = f"data:{media_type};base64,{data}"
            if use_object:
                return {"type": image_part_type, "image_url": {"url": data_url}}
            return {"type": image_part_type, "image_url": data_url}
    return None


def _normalize_anthropic_image_source(source: Any) -> Optional[Json]:
    if isinstance(source, dict):
        return source
    for meth in ("model_dump", "dict"):
        fn = getattr(source, meth, None)
        if callable(fn):
            try:
                data = fn()
            except Exception:
                data = None
            if isinstance(data, dict) and "type" in data:
                return data
    data = getattr(source, "__dict__", None)
    if isinstance(data, dict) and "type" in data:
        return data
    stype = getattr(source, "type", None)
    if stype:
        out: Json = {"type": stype}
        for key in ("url", "data", "media_type"):
            if hasattr(source, key):
                out[key] = getattr(source, key)
        return out
    return None


def _openai_image_url_object() -> bool:
    # Some upstreams expect image_url as a string, not {"url": "..."}.
    raw = os.getenv("OPENAI_IMAGE_URL_OBJECT", "false").lower()
    return raw in {"1", "true", "yes", "on"}



def _merge_include(existing: Any, additions: Iterable[str]) -> List[str]:
    seen = set()
    merged: List[str] = []

    def append_unique(value: Optional[str]) -> None:
        if not isinstance(value, str):
            return
        if value in seen:
            return
        seen.add(value)
        merged.append(value)

    if isinstance(existing, list):
        for item in existing:
            append_unique(item)

    for item in additions:
        append_unique(item)

    return merged


def _merge_reasoning(existing: Any, updates: Any) -> Json:
    if not isinstance(updates, dict) or not updates:
        return existing if isinstance(existing, dict) else {}
    if not isinstance(existing, dict) or not existing:
        return dict(updates)
    merged = dict(existing)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _merge_reasoning(merged[k], v)
        else:
            merged[k] = v
    return merged
