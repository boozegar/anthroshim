# api-transformer

[English](README.md) | [中文](README.zh-CN.md)

Convert OpenAI Responses API payloads/streams into Anthropic Messages format.

## Install (uv)

```bash
uv sync
```

## Server (Anthropic-compatible)

Expose an Anthropic-compatible `POST /v1/messages` endpoint backed by OpenAI Responses.

Create a `.env` file:

```ini
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
```

Non-secret configuration lives in `model-map.yml`.

Then run:

```bash
uv run uvicorn api_transformer.server:app --host 0.0.0.0 --port 8000
```

Optional request headers to override config:

- `x-openai-api-key`
- `x-openai-api-url`

## Docker (local)

```bash
docker build -t api-transformer:local .
docker run --env-file .env -p 8000:8000 api-transformer:local
```

## Docker Compose (server)

On the server, only `compose.yml` and `.env` are needed. Update flow:

```bash
docker compose pull
docker compose up -d
```

## GitHub Container Registry (GHCR)

A public image is built and pushed on every push to `main`:

- `ghcr.io/<owner>/<repo>:latest`
- `ghcr.io/<owner>/<repo>:<sha>`


## Library

```python
from api_transformer.openai_to_anthropic import convert_openai_to_anthropic
from api_transformer.openai_stream_to_anthropic_stream import iter_anthropic_events

anthropic_req = convert_openai_to_anthropic(openai_data)

for event in iter_anthropic_events(openai_events_iterable, model="claude-sonnet-4-5"):
    handle(event)
```
