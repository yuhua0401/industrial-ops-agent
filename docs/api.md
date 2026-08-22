# API 文档

所有业务接口挂载在 `/api/v1` 前缀下，完整 OpenAPI 文档在服务启动后访问 `/docs`。

## 健康检查

### GET /health
服务健康检查（不在 `/api/v1` 下）。

```json
{ "status": "ok", "env": "local" }
```

## 认证

### POST /api/v1/login
登录签发 JWT。

### GET /api/v1/me
获取当前用户信息（需 `Authorization: Bearer <token>`）。

## 统一对话接口

### POST /api/v1/chat/stream
统一对话入口（SSE 流式）。流程：规则前置拦截 → LLM 路由 → 按路由分发到对应 Agent 流式执行。

**请求体：**
```json
{
  "session_id": "session_001",
  "message": "XH-2000 报错 E023，屏幕不亮",
  "customer_id": "CUS001",
  "device_model": "XH-2000"
}
```

**SSE 事件序列：**
| 事件 | 说明 |
|---|---|
| `routing_decision` | 路由决策结果（agent_type / confidence / reason / execution_mode） |
| `progress` | Agent 执行进度提示 |
| `token` | 流式回答 token |
| `guidance` | 需引导跳转的意图（如"请上传故障图片"） |
| `pipeline_plan` | 多 Agent 协同计划（故障报修全流程） |
| `meta` | 回答结束元数据（sources / confidence 等） |
| `done` | 流结束信号 |
| `error` | 异常 |

## 知识问答接口

### POST /api/v1/knowledge/chat
非流式 RAG 问答。

### POST /api/v1/knowledge/chat/stream
SSE 流式 RAG 问答。

**请求体：**
```json
{
  "session_id": "session_001",
  "message": "XH-2000 的冷却液如何更换？",
  "device_model": "XH-2000",
  "top_k": 5
}
```

### GET /api/v1/knowledge/sessions/{session_id}/history
查询知识问答会话历史。

## 工单接口

### POST /api/v1/tickets
创建维修工单（当前为占位实现，未写入数据库）。

### GET /api/v1/tickets/{ticket_id}
查询工单详情（当前 501 待实现）。

### GET /api/v1/tickets
工单列表（当前返回空）。

## 未实现接口

- `GET /api/v1/admin/stats`（管理后台统计）—— admin 模块尚未实现
- `POST /api/v1/diagnosis/*`、`POST /api/v1/after_sale/*` —— 对应 API 文件为空壳，未注册
