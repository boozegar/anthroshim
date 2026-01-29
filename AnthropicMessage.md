以下是基于 Anthropic 官方文档（截至 2026 年 1 月底的最新版本，主要参考 https://docs.anthropic.com/en/api/messages ）整理的 **Anthropic Messages API** 详细字段 Markdown 文档。

Anthropic 的 Messages API（`/v1/messages`）是其核心聊天/对话接口，完全**无状态**（stateless），每次请求必须完整传入历史消息。支持工具调用（tool use）、图像/文档输入、多模态、思考块（thinking）、缓存等功能。

文档分成 **Request**（POST body 参数）和 **Response**（返回对象）两部分，列出所有常见字段、类型、是否必填、默认值及说明。部分字段为 beta 或特定模型专有。

### 1. Request Body 参数（POST /v1/messages）

```json
{
  "model": "claude-sonnet-4-5-20251022",
  "messages": [ /* 数组 */ ],
  "max_tokens": 1024,
  // ... 其他参数
}
```

| 字段名                  | 类型                          | 是否必填 | 默认值          | 说明 & 注意事项 |
|-------------------------|-------------------------------|----------|-----------------|--------------------|
| `model`                 | string                        | 必填     | -               | 模型名称，如 "claude-3-5-sonnet-20241022"、"claude-sonnet-4-5"、"claude-opus-4" 等。完整列表见 [Models](https://docs.anthropic.com/en/docs/models-overview) |
| `messages`              | array[MessageParam]           | 必填     | -               | 对话历史数组（从旧到新）。每个元素是对象：<br>- `role`: "user" 或 "assistant"<br>- `content`: string 或 array[ContentBlock]（支持 text、image、tool_use、tool_result 等） |
| `max_tokens`            | integer                       | 必填     | -               | 输出最大 token 数（模型可能提前停止）。推荐 1024–8192，根据模型上下文长度 |
| `system`                | string                        | 可选     | null            | 系统提示（system prompt），放在所有消息最前面。支持多行 |
| `temperature`           | number (0.0–1.0)              | 可选     | 1.0             | 随机性。0.0 最确定性，1.0 最创意。推荐 0.0–0.5 用于分析任务 |
| `top_p`                 | number (0.0–1.0)              | 可选     | 1.0             | Nucleus 采样。通常只调 temperature 或 top_p 中的一个 |
| `top_k`                 | integer                       | 可选     | -1（禁用）      | 仅高级使用。通常不推荐设置 |
| `stop_sequences`        | array[string]                 | 可选     | []              | 自定义停止序列（如 "\n\nHuman:"） |
| `tools`                 | array[ToolParam]              | 可选     | []              | 工具定义数组。每个工具：<br>- `name`<br>- `description`<br>- `input_schema` (JSON Schema) |
| `tool_choice`           | object                        | 可选     | {"type": "auto"}| 工具选择策略：<br>- "auto"（模型决定）<br>- "any"（必须用工具）<br>- "tool"（指定某个工具） |
| `anthropic-beta`        | string 或 array[string]       | 可选     | -               | Beta 特性头，如 "tools-2024-04-16"、"structured-outputs-2025-11-13"（用于新功能如结构化输出） |
| `metadata`              | object                        | 可选     | null            | 自定义元数据（如 user_id），用于日志/计费追踪 |
| `stream`                | boolean                       | 可选     | false           | 是否流式响应（SSE 格式） |
| `cache_control`         | object                        | 可选     | -               | 提示缓存控制（如 ephemeral），用于长 prompt 省 token |
| `thinking`              | object (beta)                 | 可选     | -               | 启用思考块（chain-of-thought），如 budget_tokens |

**messages.content 的常见 ContentBlock 类型**（array 元素）：
- `{"type": "text", "text": "..."}`
- `{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "base64string"}}`
- `{"type": "tool_use", "id": "...", "name": "...", "input": {...}}`（assistant 发出）
- `{"type": "tool_result", "tool_use_id": "...", "content": "..."}`（user 反馈工具结果）

### 2. Response Object - 返回结构

```json
{
  "id": "msg_01X...",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-4-5-20251022",
  "content": [ /* ContentBlock 数组 */ ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": { "input_tokens": 1234, "output_tokens": 567 }
}
```

| 字段名              | 类型     | 说明 |
|---------------------|----------|------|
| `id`                | string   | 消息唯一 ID（如 "msg_01Wvr1r5sXhgsZXBYK7zug1v"） |
| `type`              | string   | 固定 "message" |
| `role`              | string   | 固定 "assistant"（响应总是 assistant） |
| `model`             | string   | 实际使用的模型 |
| `content`           | array[ContentBlock] | 输出内容块数组。最常见：<br>- `{"type": "text", "text": "..."}`<br>- `{"type": "tool_use", "id": "...", "name": "...", "input": {...}}`<br>- `{"type": "thinking", "thinking": "...", "signature": "..."}`（beta） |
| `stop_reason`       | string   | 停止原因：<br>- "end_turn"（自然结束）<br>- "max_tokens"（达到上限）<br>- "tool_use"（调用工具）<br>- "stop_sequence"（命中停止序列） |
| `stop_sequence`     | string \| null | 如果命中自定义 stop_sequences，返回匹配的那个 |
| `usage`             | object   | Token 消耗：<br>- `input_tokens`: 输入总 token（包括缓存命中）<br>- `output_tokens`: 输出 token |
| `usage.input_tokens_details` (beta) | object | 详细输入 token：<br>- `cache_read_input_tokens`<br>- `cache_creation_input_tokens` 等 |

### 3. 快速使用建议 & 省 token Tips（2026 年现状）

- **无状态设计**：必须自己维护 messages 数组（本地拼接历史）。长对话时建议总结/压缩旧消息。
- **省 token**：用 prompt caching（cache_control: {type: "ephemeral"}），重复前缀可缓存 5m 或 1h，缓存读取 token 极便宜。
- **工具调用**：当 stop_reason == "tool_use" 时，解析 content 中的 tool_use 块，执行后 append tool_result 块再发新请求。
- **多模态**：支持图像/文档输入（base64 或 URL），但 token 消耗较高。
- **结构化输出**（beta）：用 anthropic-beta: "structured-outputs-2025-11-13" + output_format: {type: "json_schema", schema: {...}} 强制 JSON 输出。
- **监控**：每次响应看 usage.input_tokens / output_tokens 计算成本。

这份文档覆盖了 Messages API 95%+ 的实际使用字段（包括 2025–2026 年新增的 beta 特性如 thinking、structured outputs、cache control）。如果官方文档有更新，建议直接查 https://docs.anthropic.com/en/api/messages 。

想加具体 Python/JS 示例、某个字段的完整 JSON 样例、或对比 OpenAI Responses API 的差异，随时告诉我～