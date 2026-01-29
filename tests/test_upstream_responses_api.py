import json
import os

import httpx
import pytest

from api_transformer.openai_stream_to_anthropic_stream import (
    iter_openai_sse_json_events,
)


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def _responses_url(base_url: str) -> str:
    if base_url.endswith("/responses"):
        return base_url
    return f"{base_url}/responses"


def _require_upstream_env() -> tuple[str, str, str]:
    if not _env_bool("RUN_UPSTREAM_TESTS"):
        pytest.skip("RUN_UPSTREAM_TESTS not enabled")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    model = os.getenv("OPENAI_TEST_MODEL")
    if not model:
        pytest.skip("OPENAI_TEST_MODEL not set")
    base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    return key, base_url.rstrip("/"), model


def _build_request(model: str) -> dict:
    req = {
        "model": model,
        "input": "Return a one-sentence greeting.",
    }
    include_raw = os.getenv("OPENAI_TEST_INCLUDE")
    if include_raw:
        include = [s.strip() for s in include_raw.split(",") if s.strip()]
        if include:
            req["include"] = include
    effort = os.getenv("OPENAI_TEST_REASONING_EFFORT")
    if effort:
        req["reasoning"] = {"effort": effort}
    return req


def test_upstream_responses_basic():
    key, base_url, model = _require_upstream_env()
    url = _responses_url(base_url)
    req = _build_request(model)

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, headers=headers, json=req)
    assert resp.status_code < 400, resp.text
    data = resp.json()

    summary = {
        "id": data.get("id"),
        "model": data.get("model"),
        "status": data.get("status"),
        "output_types": [
            item.get("type")
            for item in (data.get("output") or [])
            if isinstance(item, dict)
        ],
        "usage_keys": list((data.get("usage") or {}).keys())
        if isinstance(data.get("usage"), dict)
        else [],
        "has_reasoning_summary": bool(data.get("reasoning_summary")),
    }
    print(json.dumps(summary, ensure_ascii=True))


def test_upstream_responses_stream():
    if not _env_bool("RUN_UPSTREAM_STREAM_TESTS"):
        pytest.skip("RUN_UPSTREAM_STREAM_TESTS not enabled")
    key, base_url, model = _require_upstream_env()
    url = _responses_url(base_url)
    req = _build_request(model)
    req["stream"] = True

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", url, headers=headers, json=req) as resp:
            assert resp.status_code < 400, resp.read().decode("utf-8", errors="replace")

            def _lines():
                for line in resp.iter_lines():
                    if isinstance(line, bytes):
                        yield line.decode("utf-8", errors="replace")
                    else:
                        yield str(line)

            events = list(iter_openai_sse_json_events(_lines()))

    types = [ev.get("type") for ev in events if isinstance(ev, dict)]
    summary = {
        "event_types": sorted({t for t in types if isinstance(t, str)}),
        "count": len(types),
    }
    print(json.dumps(summary, ensure_ascii=True))
