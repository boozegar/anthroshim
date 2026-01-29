# anthroshim

[English](README.md) | [中文](README.zh-CN.md)

Convert upstream OpenAI Responses API payloads/streams into Anthropic Messages format.
Provide an Anthropic-compatible `POST /v1/messages` endpoint, proxying to OpenAI Responses.

## Server Deployment

```bash
mkdir anthroshim
cd anthroshim
```

Create [.env](.template.env), [model-map.yml](model-map.yml), and [compose.yml](compose.yml)

> Fill in `.env` yourself; copy the others as-is.
> Choose the port in `compose.yml`.

```bash
docker compose pull
docker compose up -d
```

Update
```bash
docker compose pull
docker compose down
docker compose up -d
```
