# api-transformer

[English](README.md) | [中文](README.zh-CN.md)

将 OpenAI Responses API 的 payload/流转换为 Anthropic Messages 格式。

## 安装（uv）

```bash
uv sync
```

## 服务器（Anthropic 兼容）

提供与 Anthropic 兼容的 `POST /v1/messages` 接口，后端转发到 OpenAI Responses。

创建 `.env` 文件：

```ini
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
ANTHROPIC_MODEL_MAP={"claude-sonnet-4-5":"gpt-5.2-codex"}
ANTHROPIC_MODEL_DEFAULT=gpt-5.2-codex
OPENAI_FORCE_STREAM=true
```

启动：

```bash
uv run uvicorn api_transformer.server:app --host 0.0.0.0 --port 8000
```

可选请求头（覆盖配置）：

- `x-openai-api-key`
- `x-openai-api-url`

## Docker（本地）

```bash
docker build -t api-transformer:local .
docker run --env-file .env -p 8000:8000 api-transformer:local
```

## Docker Compose（服务器）

服务器只需要 `compose.yml` 和 `.env`。更新流程：

```bash
docker compose pull
docker compose up -d
```

## GitHub 容器镜像（GHCR）

每次 push 到 `main` 会自动构建并推送公开镜像：

- `ghcr.io/<owner>/<repo>:latest`
- `ghcr.io/<owner>/<repo>:<sha>`

## CLI [ready to remove]

将保存的 OpenAI response（或 OpenAI `responses.create(input=...)` payload）转换为 Anthropic `{system, messages}`：

```bash
uv run api-transformer openai-to-anthropic --in openai.json --out anthropic.json
```

将 OpenAI Responses 的 NDJSON 流转换为 Anthropic Messages 的 NDJSON 流：

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
