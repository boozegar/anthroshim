from __future__ import annotations

from pathlib import Path

from api_transformer import anthropic_to_openai, server


def _write_model_map(tmp_path: Path, text: str) -> None:
    (tmp_path / "model-map.yml").write_text(text, encoding="utf-8")


def test_model_map_wildcard_and_catch_all(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    server._MODEL_MAP_CACHE = None
    _write_model_map(
        tmp_path,
        """
claude-*-4-5:
  model: gpt-5.2-codex
  reasoning:
    effort: low
"*": gpt-4o-mini
""".lstrip(),
    )

    mapped_model, mapped_extra = server._map_model_and_extras("claude-sonnet-4-5")

    assert mapped_model == "gpt-5.2-codex"
    assert mapped_extra == {"reasoning": {"effort": "low"}}


def test_request_level_model_mapping(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    server._MODEL_MAP_CACHE = None
    _write_model_map(
        tmp_path,
        """
claude-opus-3-0:
  model: gpt-4.1
  reasoning:
    effort: high
"*": gpt-4o
""".lstrip(),
    )

    anthropic_req = {
        "model": "claude-opus-3-0",
        "messages": [{"role": "user", "content": "hi"}],
    }
    openai_req = anthropic_to_openai.convert_anthropic_to_openai_request(
        anthropic_req
    )
    mapped_model, mapped_extra = server._map_model_and_extras(openai_req.get("model"))
    openai_req["model"] = mapped_model
    if mapped_extra:
        server._deep_merge_inplace(openai_req, mapped_extra)

    assert openai_req["model"] == "gpt-4.1"
    assert openai_req["reasoning"] == {"effort": "high"}
