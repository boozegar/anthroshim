import json
from pathlib import Path

from api_transformer.openai_stream_to_anthropic_stream import iter_anthropic_events


def _write_model_map(tmp_path: Path, text: str) -> None:
    (tmp_path / "model-map.yml").write_text(text, encoding="utf-8")


def test_model_map_value_object_and_store_false(monkeypatch, tmp_path: Path):
    from api_transformer import server

    monkeypatch.chdir(tmp_path)
    server._MODEL_MAP_CACHE = None
    _write_model_map(
        tmp_path,
        json.dumps(
            {
                "claude-sonnet-*": {
                    "model": "gpt-5.2-codex",
                    "reasoning": {"effort": "medium"},
                },
                "*": "gpt-4.1",
            },
            ensure_ascii=True,
        ),
    )

    model, extra = server._map_model_and_extras("claude-sonnet-4-5")
    assert model == "gpt-5.2-codex"
    assert extra.get("reasoning") == {"effort": "medium"}

    req: dict[str, object] = {"model": "claude-sonnet-4-5"}
    server._deep_merge_inplace(req, extra)
    req["store"] = False
    assert req["store"] is False


def test_text_stream_basic():
    openai_events = [
        {"type": "response.created", "response": {"model": "gpt-5.2"}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hel",
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "lo",
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [],
            },
        },
        {
            "type": "response.completed",
            "response": {"usage": {"output_tokens": 2}, "output": []},
        },
    ]

    out = list(iter_anthropic_events(openai_events, model="claude-sonnet-4-5"))
    assert out[0]["type"] == "message_start"
    assert out[1]["type"] == "content_block_start"
    assert out[2]["type"] == "content_block_delta"
    assert out[2]["delta"]["type"] == "text_delta"
    assert out[-1]["type"] == "message_stop"


def test_tool_call_arguments_stream():
    openai_events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 0,
            "delta": '{"location":',
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 0,
            "delta": ' "SF"}',
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": "fc_1",
            "name": "get_weather",
            "output_index": 0,
            "arguments": '{"location": "SF"}',
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"location": "SF"}',
            },
        },
        {
            "type": "response.completed",
            "response": {
                "usage": {"output_tokens": 1},
                "output": [{"type": "function_call"}],
            },
        },
    ]

    out = list(iter_anthropic_events(openai_events, model="claude-sonnet-4-5"))
    # message_start then tool_use block
    assert any(
        e.get("type") == "content_block_start"
        and e.get("content_block", {}).get("type") == "tool_use"
        for e in out
    )
    assert any(
        e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "input_json_delta"
        for e in out
    )


def test_reasoning_summary_stream_emits_thinking_block():
    openai_events = [
        {"type": "response.reasoning_summary.delta", "delta": "First"},
        {"type": "response.reasoning_summary.delta", "delta": " Second"},
        {"type": "response.completed", "response": {"usage": {"output_tokens": 1}, "output": []}},
    ]

    out = list(
        iter_anthropic_events(
            openai_events,
            model="claude-sonnet-4-5",
            keep_reasoning_summary=True,
        )
    )
    assert any(
        e.get("type") == "content_block_start"
        and e.get("content_block", {}).get("type") == "thinking"
        for e in out
    )
    assert any(
        e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "thinking_delta"
        for e in out
    )


def test_reasoning_summary_stream_dropped_when_disabled():
    openai_events = [
        {"type": "response.reasoning_summary.delta", "delta": "Hidden"},
        {"type": "response.completed", "response": {"usage": {"output_tokens": 1}, "output": []}},
    ]

    out = list(iter_anthropic_events(openai_events, model="claude-sonnet-4-5"))
    assert not any(
        e.get("type") == "content_block_start"
        and e.get("content_block", {}).get("type") == "thinking"
        for e in out
    )


def test_response_reasoning_summary_to_thinking_block():
    from api_transformer import server

    resp = {
        "id": "resp_1",
        "model": "gpt-5.2",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi"}],
            }
        ],
        "reasoning_summary": "Because.",
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }

    out = server._openai_response_to_anthropic_message(
        resp, thinking_enabled=True
    )
    assert any(
        b.get("type") == "thinking" and b.get("thinking") == "Because."
        for b in out.get("content", [])
    )
