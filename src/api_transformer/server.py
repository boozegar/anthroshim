from __future__ import annotations

import json
import logging
import os
import uuid
import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import httpx
import yaml
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from dotenv import load_dotenv

from .anthropic_to_openai import convert_anthropic_to_openai_request
from .openai_stream_to_anthropic_stream import (
    iter_anthropic_events,
    iter_anthropic_sse_lines,
    iter_openai_sse_json_events,
)
from .openai_to_anthropic import openai_items_to_anthropic_messages


Json = Dict[str, Any]

load_dotenv()

_MODEL_MAP_CACHE: Optional[Dict[str, Any]] = None


def _model_map_path() -> Path:
    return Path("model-map.yml")


def _get_model_map() -> Dict[str, Any]:
    global _MODEL_MAP_CACHE
    if _MODEL_MAP_CACHE is not None:
        return _MODEL_MAP_CACHE

    path = _model_map_path()
    if not path.exists():
        _MODEL_MAP_CACHE = {}
        return _MODEL_MAP_CACHE

    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.exception("failed to read model map path=%s", str(path))
        _MODEL_MAP_CACHE = {}
        return _MODEL_MAP_CACHE

    try:
        data = yaml.safe_load(raw) if raw.strip() else None
    except Exception as exc:
        logger.exception("failed to parse model map yaml path=%s", str(path))
        _MODEL_MAP_CACHE = {}
        return _MODEL_MAP_CACHE

    if not isinstance(data, dict):
        _MODEL_MAP_CACHE = {}
        return _MODEL_MAP_CACHE

    # Allow optional wrapper for compatibility, but default is a bare mapping.
    if isinstance(data.get("model_map"), dict):
        data = data.get("model_map")
    elif isinstance(data.get("api_transformer_config"), dict):
        inner = data.get("api_transformer_config")
        if isinstance(inner, dict) and isinstance(inner.get("model_map"), dict):
            data = inner.get("model_map")

    if not isinstance(data, dict):
        _MODEL_MAP_CACHE = {}
        return _MODEL_MAP_CACHE

    out: Dict[str, Any] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not k:
            continue
        if isinstance(v, str) and v:
            out[k] = v
            continue
        if isinstance(v, dict) and v:
            out[k] = v

    _MODEL_MAP_CACHE = out
    return _MODEL_MAP_CACHE


def _get_log_level() -> int:
    raw = os.getenv("TRANSFORMER_LOG_LEVEL", "INFO").upper()
    return getattr(logging, raw, logging.INFO)


def _configure_logging() -> logging.Logger:
    level = _get_log_level()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("api_transformer")
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        stream = logging.StreamHandler()
        stream.setLevel(level)
        stream.setFormatter(fmt)
        logger.addHandler(stream)

        log_file = os.getenv("TRANSFORMER_LOG_FILE")
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

    return logger


logger = _configure_logging()
LOG_PAYLOADS = os.getenv("TRANSFORMER_LOG_PAYLOADS", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LOG_MAX_CHARS = int(os.getenv("TRANSFORMER_LOG_MAX_CHARS", "4000"))


app = FastAPI(title="api-transformer", version="0.1.0")


@app.post("/v1/messages")
@app.post("/v1/message")
async def create_message(
    request: Request,
    x_openai_api_key: Optional[str] = Header(default=None),
    x_openai_api_url: Optional[str] = Header(default=None),
) -> Any:
    try:
        payload = await request.json()
    except Exception as exc:
        logger.warning("invalid json body", exc_info=exc)
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    _log_json("anthropic.request", _scrub_payload(payload))

    key, base_url = _get_openai_config(x_openai_api_key, x_openai_api_url)
    openai_req = convert_anthropic_to_openai_request(payload)
    mapped_model, mapped_extra = _map_model_and_extras(openai_req.get("model"))
    openai_req["model"] = mapped_model
    if isinstance(mapped_extra, dict) and mapped_extra:
        _deep_merge_inplace(openai_req, mapped_extra)
    if not openai_req.get("model"):
        raise HTTPException(status_code=400, detail="Missing model")

    # Force stateless Responses API mode (equivalent to stateful=false).
    openai_req["store"] = False

    if _force_stream():
        openai_req["stream"] = True

    _log_json("openai.request", _scrub_payload(openai_req))

    url = _responses_url(base_url)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    client_stream = payload.get("stream") is True
    if client_stream:
        return await _stream_openai_to_anthropic(url, headers, openai_req)

    if openai_req.get("stream") is True:
        data = await _stream_openai_to_anthropic_message(url, headers, openai_req)
    else:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=openai_req)
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            data = resp.json()

    _log_json("openai.response", _scrub_payload(data))
    out = _openai_response_to_anthropic_message(data)
    _log_json("anthropic.response", _scrub_payload(out))
    return JSONResponse(out)


async def _stream_openai_to_anthropic(
    url: str, headers: Json, openai_req: Json
) -> StreamingResponse:
    openai_events = await _fetch_openai_stream_events(url, headers, openai_req)
    _log_json("openai.stream.events", openai_events)
    anth_events = list(
        iter_anthropic_events(openai_events, model=openai_req.get("model") or "unknown")
    )
    _log_json("anthropic.stream.events", anth_events)
    return StreamingResponse(
        iter_anthropic_sse_lines(anth_events), media_type="text/event-stream"
    )


async def _stream_openai_to_anthropic_message(
    url: str, headers: Json, openai_req: Json
) -> Json:
    openai_events = await _fetch_openai_stream_events(url, headers, openai_req)
    _log_json("openai.stream.events", openai_events)
    for ev in reversed(openai_events):
        if ev.get("type") in {
            "response.completed",
            "response.incomplete",
            "response.failed",
        }:
            resp = ev.get("response")
            if isinstance(resp, dict):
                return resp
    raise HTTPException(
        status_code=502, detail="Upstream stream did not include response object"
    )


async def _fetch_openai_stream_events(
    url: str, headers: Json, openai_req: Json
) -> list[Json]:
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            resp = await client.post(url, headers=headers, json=openai_req)
        except httpx.TimeoutException as exc:
            logger.exception("openai.request timeout url=%s", url)
            raise HTTPException(
                status_code=504, detail=f"Upstream timeout: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            logger.exception("openai.request error url=%s", url)
            raise HTTPException(
                status_code=502, detail=f"Upstream connection error: {exc}"
            ) from exc
        if resp.status_code >= 400:
            text = await resp.aread()
            logger.warning(
                "openai.response error status=%s url=%s body=%s",
                resp.status_code,
                url,
                text[:500].decode("utf-8", errors="replace"),
            )
            raise HTTPException(
                status_code=resp.status_code,
                detail=text.decode("utf-8", errors="replace"),
            )
        raw = await resp.aread()
    lines = raw.decode("utf-8", errors="replace").splitlines()
    return list(iter_openai_sse_json_events(lines))


def _get_openai_config(
    header_key: Optional[str],
    header_url: Optional[str],
) -> tuple[str, str]:
    key = header_key or os.getenv("OPENAI_API_KEY")
    if not key:
        logger.error("Missing OPENAI_API_KEY")
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")
    base_url = header_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    return key, base_url.rstrip("/")


def _responses_url(base_url: str) -> str:
    if base_url.endswith("/responses"):
        return base_url
    return f"{base_url}/responses"


def _force_stream() -> bool:
    return os.getenv("OPENAI_FORCE_STREAM", "").lower() in {"1", "true", "yes", "on"}


def _map_model_and_extras(model: Optional[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    if not model:
        return None, {}
    mapping: Dict[str, Any] = dict(_get_model_map())

    # 1) Exact match
    if model in mapping:
        return _normalize_model_map_value(model, mapping[model])

    # 2) Best wildcard match (excluding the catch-all "*")
    best_key: Optional[str] = None
    best_score: Tuple[int, int] = (-1, -1)
    for k in mapping.keys():
        if not isinstance(k, str) or not k or k == "*":
            continue
        if "*" not in k and "?" not in k:
            continue
        if fnmatch.fnmatchcase(model, k):
            # Prefer more specific patterns (more non-wildcard chars, then longer pattern).
            non_wild = len(k.replace("*", "").replace("?", ""))
            score = (non_wild, len(k))
            if score > best_score:
                best_key = k
                best_score = score

    if best_key is not None:
        return _normalize_model_map_value(model, mapping[best_key])

    # 3) Catch-all
    if "*" in mapping:
        return _normalize_model_map_value(model, mapping["*"])

    return model, {}


def _normalize_model_map_value(
    requested_model: str, value: Any
) -> Tuple[str, Dict[str, Any]]:
    # String value means: just replace the model.
    if isinstance(value, str) and value:
        return value, {}

    # Dict value means: {"model": "...", ...additional openai fields...}
    if isinstance(value, dict):
        out_model = value.get("model")
        model_str = (
            out_model if isinstance(out_model, str) and out_model else requested_model
        )
        extra = {k: v for k, v in value.items() if k != "model"}
        return model_str, extra

    return requested_model, {}


def _deep_merge_inplace(base: Dict[str, Any], updates: Dict[str, Any]) -> None:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge_inplace(base[k], v)
        else:
            base[k] = v


def _parse_model_map_pairs(raw: str) -> Dict[str, str]:
    # Deprecated: env-based model map parsing was removed.
    return {}


def _openai_response_to_anthropic_message(resp: Json) -> Json:
    items = resp.get("output") if isinstance(resp, dict) else None
    if not isinstance(items, list):
        items = []
    _, messages = openai_items_to_anthropic_messages(items, instructions=None)

    content = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content.extend(msg.get("content") or [])

    stop_reason = _openai_stop_reason(resp)
    usage = {}
    if isinstance(resp, dict) and isinstance(resp.get("usage"), dict):
        usage = {
            "input_tokens": resp.get("usage", {}).get("input_tokens"),
            "output_tokens": resp.get("usage", {}).get("output_tokens"),
        }

    return {
        "id": resp.get("id")
        if isinstance(resp, dict) and resp.get("id")
        else f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": resp.get("model") if isinstance(resp, dict) else "unknown",
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


def _openai_stop_reason(resp: Json) -> str:
    if isinstance(resp, dict):
        inc = resp.get("incomplete_details")
        if isinstance(inc, dict) and inc.get("reason") == "max_tokens":
            return "max_tokens"

        out = resp.get("output")
        if isinstance(out, list) and out:
            last = out[-1]
            if isinstance(last, dict) and last.get("type") in {
                "function_call",
                "custom_tool_call",
            }:
                return "tool_use"
    return "end_turn"


def _scrub_payload(data: Any) -> Any:
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if k.lower() in {"authorization", "api_key", "x-openai-api-key"}:
                out[k] = "***"
            else:
                out[k] = _scrub_payload(v)
        return out
    if isinstance(data, list):
        return [_scrub_payload(v) for v in data]
    return data


def _log_json(label: str, data: Any) -> None:
    if not (LOG_PAYLOADS or logger.isEnabledFor(logging.DEBUG)):
        return
    try:
        text = json.dumps(data, ensure_ascii=True)
    except Exception:
        text = str(data)
    if len(text) > LOG_MAX_CHARS:
        text = text[:LOG_MAX_CHARS] + "...(truncated)"
    if LOG_PAYLOADS:
        logger.info("%s %s", label, text)
    else:
        logger.debug("%s %s", label, text)
