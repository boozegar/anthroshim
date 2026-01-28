# api-transformer

Convert OpenAI Responses API payloads/streams into Anthropic Messages format.

## Install (uv)

```bash
uv sync
```


## Server (Anthropic-compatible)

Expose an Anthropic-compatible `POST /v1/messages` endpoint backed by OpenAI Responses:

Create a `.env` file:

```ini
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
ANTHROPIC_MODEL_MAP={"claude-sonnet-4-5":"gpt-5.2-codex"}
ANTHROPIC_MODEL_DEFAULT=gpt-5.2-codex
OPENAI_FORCE_STREAM=true
```

Then run:

```bash
uv run uvicorn api_transformer.server:app --host 0.0.0.0 --port 8000
```

Optional request headers to override config:

- `x-openai-api-key`
- `x-openai-api-url`



## CLI [ready to remove]

Convert a saved OpenAI response (or OpenAI `responses.create(input=...)` payload) to Anthropic `{system, messages}`:

```bash
uv run api-transformer openai-to-anthropic --in openai.json --out anthropic.json
```

Convert a newline-delimited JSON stream of OpenAI Responses streaming events into a newline-delimited JSON stream of Anthropic Messages streaming events:

```bash
uv run api-transformer openai-stream-to-anthropic-stream --in openai_events.ndjson --out anthropic_events.ndjson
```

## Library

```python
from api_transformer.openai_to_anthropic import convert_openai_to_anthropic
from api_transformer.openai_stream_to_anthropic_stream import iter_anthropic_events

anthropic_req = convert_openai_to_anthropic(openai_data)

for event in iter_anthropic_events(openai_events_iterable, model="claude-sonnet-4-5"):
    handle(event)
```
