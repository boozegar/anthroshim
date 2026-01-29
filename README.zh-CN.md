# anthroshim

[English](README.md) | [中文](README.zh-CN.md)

将 上游OpenAI Responses API 的 payload/流转换为 Anthropic Messages 格式。
提供与 Anthropic 兼容的 `POST /v1/messages` 接口，后端转发到 OpenAI Responses。

## 服务器部署 

```bash
mkdir anthroshim
cd anthroshim
```


创建 [.env](.template.env) 、 [model-map.yml](model-map.yml)和  [compose.yml](compose.yml)

> .env需自己填写，其余复制即可
> compose.yml 端口自己选择 

 


```bash
docker compose pull
docker compose up -d
```

更新 
```bash
docker compose pull
docker compose down
docker compose up -d

```



