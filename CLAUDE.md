# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

- Install deps (uv): `uv sync`
- Run tests: `uv run pytest`
- Run a single test: `uv run pytest tests/test_openai_stream_to_anthropic_stream.py -k test_text_stream_basic`
- Run server (Anthropic-compatible): `uv run uvicorn api_transformer.server:app --host 0.0.0.0 --port 8000`

## Architecture overview

- `src/api_transformer/openai_to_anthropic.py` converts OpenAI Responses payloads (input, response, or output items) into Anthropic Messages request shape.
- `src/api_transformer/openai_stream_to_anthropic_stream.py` converts OpenAI streaming events to Anthropic streaming events, including SSE parsing/formatting helpers.
- `src/api_transformer/anthropic_to_openai.py` converts Anthropic Messages requests into OpenAI Responses payloads, including tools/tool_choice mapping.
- `src/api_transformer/server.py` exposes a FastAPI `/v1/messages` endpoint that proxies to OpenAI Responses, mapping models and streaming modes; it uses the conversion modules above.
- Tests in `tests/` focus on stream conversion behavior; `tests/conftest.py` ensures `src/` is on `sys.path` for src-layout imports.

## Runtime configuration (server)

- `.env` is used by `api_transformer.server` (via `dotenv`) for OpenAI auth, model mapping, and streaming behavior.
- Key env vars: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ANTHROPIC_MODEL_MAP`, `ANTHROPIC_MODEL_DEFAULT`, `OPENAI_FORCE_STREAM`, `TRANSFORMER_LOG_LEVEL`, `TRANSFORMER_LOG_FILE`, `TRANSFORMER_LOG_PAYLOADS`.
