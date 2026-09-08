# 系统架构设计

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    客户触点层                              │
│  企业微信 / 微信公众号 / Web 浮窗 / API 接入               │
│  演示前端：backend/static/（零构建，SSE 聊天 + JWT 登录）   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    API 接入层                              │
│  FastAPI /api/v1（chat / knowledge / tickets /            │
│  diagnosis / after-sale / auth）+ SSE 流式                │
│  （admin 管理后台规划中，尚未实现）                         │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  Agent 编排层（LangGraph）                  │
│                                                          │
│   ┌─────────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐ │
│   │ 意图识别与路由 │→ │ 知识库检索 │  │ 故障诊断  │  │ 工单 │ │
│   │ (规则拦截+LLM)│  │ (RAG)    │  │ (诊断树)  │  │ 管理 │ │
│   └─────────────┘  └──────────┘  └──────────┘  └──────┘ │
│                                        ┌──────────┐    │
│                                        │ 售后协调   │    │
│                                        │ (ToolCall)│    │
│                                        └──────────┘    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   支撑层                                   │
│  LLM工厂 │ Milvus │ PostgreSQL(9表含parts) │ BGE-M3      │
│  Checkpointer(memory|postgres 可切换) │ Reranker         │
└─────────────────────────────────────────────────────────┘
```

> **说明**：
> - **意图识别**当前为「规则拦截（`api/chat.py::_pre_filter`）+ LLM 路由（`_llm_route`）」两级实现，尚无独立 Agent 目录（`backend/agents/intent/` 规划中）。
> - **工单 API** 前缀为 `/api/v1/tickets`（由 `api/router.py` 统一挂载，`ticket.py` 不再自带 `/api/tickets` 前缀）。
> - **管理后台**（admin API + Vue 前端）为规划中能力，尚未实现。
> - **售后配件查询**已落库：`parts` 备件表 + `check_part_stock` 真查库（在库/缺货/查无/DB 异常四态降级）。
> - **诊断追问状态**：`CHECKPOINTER_BACKEND=memory|postgres` 可切换（postgres 不可用自动降级 memory）。

## Agent 协作流程

| 场景 | 流程 |
|---|---|
| 知识问答 | 意图路由 → 知识库 Agent（RAG 检索 → 生成） |
| 故障诊断 | 意图路由 → 故障诊断 Agent（诊断树三轨匹配 → 追问 → 报告） |
| 报修全流程 | 意图路由 → 故障诊断 → 工单管理 → 售后协调（pipeline 串联） |

各 Agent 的 LangGraph 图已独立实现（`build_*_graph()`），由 `backend/supervisor.py`（Supervisor 服务类）编排；统一对话入口 `api/chat.py` 通过「规则拦截 + LLM 路由 + 流式执行器」调用 Supervisor 驱动各 Agent Graph，pipeline 模式串联 诊断 → 工单 → 售后。

## 目录结构与职责

```
backend/
├── agents/          # Agent 逻辑（每 Agent 独立的 State/Node/Graph/Prompt）
│   ├── diagnosis/   #   故障诊断 Agent（最完整：诊断树 + 三轨匹配 + 追问循环）
│   ├── knowledge/   #   知识库 Agent（RAG 检索 → 生成）
│   ├── ticket/      #   工单管理 Agent（结构化生成）
│   └── after_sale/  #   售后协调 Agent（Tool 定义）
├── core/            # 核心基础设施（LLM 工厂/日志/重试/异常/意图分类）
├── api/             # HTTP API（chat / knowledge / tickets / auth）
├── config.py        # 配置中心（读 .env.local）
├── db/              # 数据库（模型 / 幂等迁移）
└── knowledge_base/  # 知识库引擎（加载/分块/嵌入/检索/精排/Contextual RAG）
```
