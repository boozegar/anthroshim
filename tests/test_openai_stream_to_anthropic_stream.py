import json

from api_transformer.openai_stream_to_anthropic_stream import iter_anthropic_events


def test_text_stream_basic():
    openai_events = [
        {"type": "response.created", "response": {"model": "gpt-5.2"}},
        {"type": "response.output_item.added", "output_index": 0, "item": {"id": "msg_1", "type": "message", "role": "assistant", "content": []}},
        {"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "Hel"},
        {"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "lo"},
        {"type": "response.output_item.done", "output_index": 0, "item": {"id": "msg_1", "type": "message", "role": "assistant", "content": []}},
        {"type": "response.completed", "response": {"usage": {"output_tokens": 2}, "output": []}},
    ]

    out = list(iter_anthropic_events(openai_events, model="claude-sonnet-4-5"))
    assert out[0]["type"] == "message_start"
    assert out[1]["type"] == "content_block_start"
    assert out[2]["type"] == "content_block_delta"
    assert out[2]["delta"]["type"] == "text_delta"
    assert out[-1]["type"] == "message_stop"


def test_tool_call_arguments_stream():
    openai_events = [
        {"type": "response.output_item.added", "output_index": 0, "item": {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0, "delta": "{\"location\":"},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0, "delta": " \"SF\"}"},
        {"type": "response.function_call_arguments.done", "item_id": "fc_1", "name": "get_weather", "output_index": 0, "arguments": "{\"location\": \"SF\"}"},
        {"type": "response.output_item.done", "output_index": 0, "item": {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"location\": \"SF\"}"}},
        {"type": "response.completed", "response": {"usage": {"output_tokens": 1}, "output": [{"type": "function_call"}]}},
    ]

    out = list(iter_anthropic_events(openai_events, model="claude-sonnet-4-5"))
    # message_start then tool_use block
    assert any(e.get("type") == "content_block_start" and e.get("content_block", {}).get("type") == "tool_use" for e in out)
    assert any(e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "input_json_delta" for e in out)
