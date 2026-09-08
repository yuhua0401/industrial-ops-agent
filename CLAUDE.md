# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个面向**工业设备制造商与使用方**的多 Agent 智能运维平台，面向钢铁/流程行业的设备全生命周期管理场景（设备管理、AI 点巡检、设备故障诊断、智慧设备现场管理）。

系统用 AI 智能体重塑设备运维的 **点检 → 诊断 → 报修 → 备件 → 售后** 全链路，目标是从「被动响应」走向「主动服务」——把「人找事」变成「事找人」。

> ⚠️ **注意**：当前为**新项目起步阶段**，已实现诊断/知识库/工单/售后/主编排等核心模块；「设备台账 / 点检巡检」等仍为规划中。动手前先读「当前实现状态」与「当前已知问题」。

## 技术栈

- **语言**: Python 3.11（推荐用 conda 环境 `industrial_agent`：`conda activate industrial_agent`，已装齐全部依赖）
- **AI 编排框架**: LangGraph 1.0+ (State Graph + ToolNode + Checkpointer)
- **大模型**: DeepSeek V4 Flash（主力）/ Pro（复杂推理），兼容 OpenAI 接口，GPT 可作兜底
- **结构化输出**: Pydantic + with_structured_output (function_calling)
- **向量数据库**: Milvus 2.4+（Dense + Sparse 混合检索，WeightedRanker 0.7/0.3）
- **嵌入模型**: BGE-M3（本地部署，dense + sparse 双向量）
- **精排模型**: BGE-Reranker（本地部署，置信度阈值 0.75）
- **意图分类**: MiniLM-L6-v2（`core/query_classifier.py`，尚未接入请求链路）
- **后端框架**: FastAPI 0.117+（异步 + SSE 流式响应）
- **数据库**: PostgreSQL + asyncpg + SQLAlchemy 2.0
- **认证**: JWT（python-jose + bcrypt）
- **部署**: Docker Compose（私有化部署，PG/Milvus/etcd/MinIO/backend/nginx）
- **其他**: Contextual RAG、MCP 协议支持（预留）

## 常用命令

- **启动 PostgreSQL（Docker）**: `docker compose -f deploy/docker-compose.yml up -d postgres`（映射宿主机 5432，库 `equipment_cs`，用户 `postgres/postgres`）
- **启动全部依赖（PG/Milvus/MinIO/etcd）**: `docker compose -f deploy/docker-compose.yml up -d`
- **数据库迁移（建表 + 索引）**: `python scripts/migrate.py`（服务启动时也会自动执行幂等迁移）
- **开发种子数据（用户/客户/设备，幂等）**: `python scripts/seed_dev_data.py`（账号 `engineer/admin123`、`customer/admin123`）
- **启动开发服务器**: `uvicorn backend.main:app --reload --port 8000`
- **运行测试**: `pytest tests/ -v`
- **运行诊断线专项测试**: `pytest tests/test_diagnosis.py -v`
- **故障诊断离线演示**（无需 API Key，注入 FakeLLM）: `python scripts/demo_diagnosis.py` 或 `python backend/agents/diagnosis/graph.py`
- **知识库建库**: `python scripts/build_knowledge_base.py <文档路径> [--course-id 设备型号] [--no-context]`（完整四步：读取→分块→BGE-M3嵌入→写Milvus）
- **Docker 构建**: `docker compose -f deploy/docker-compose.yml up -d --build`
- **配置**: 复制 `deploy/.env.example` 为根目录 `.env.local` 并填写

## 开发规范与约定

- **代码风格**: 遵循 Ruff 和 mypy 配置，提交前必须通过 `ruff check .` 和 `mypy backend/`。
- **提交信息**: 使用约定式提交 (Conventional Commits)，格式为 `<type>(<scope>): <description>`，例如 `feat(agent-diagnosis): 完成故障诊断线实现与企业级加固`。
- **类型安全**: 所有函数必须标注类型，禁止使用 `Any`（确有必要时用 `# type: ignore` 注释并说明理由）。
- **错误处理**: Agent 节点统一返回 `{"error": null, "data": ...}` 或 `{"error": "错误信息", "data": null}` 格式；LLM 调用失败走 `retry → fallback → raise` 链路（`core/retry.py` 已封装 `with_retry` 装饰器与 `AgentFallbackHandler`）。
- **LLM 获取入口**: 所有 Agent 必须通过 `backend/core/llm_factory.py` 的 `get_llm(agent_type)` / `get_structured_llm(agent_type, schema)` 获取模型，禁止直接调用 `init_chat_model`。agent_type 路由表见 `_AGENT_MODEL_ROUTING`；新增 Agent 时在此注册。
- **Prompt 管理**: 各 Agent 的 Prompt 放在**各自 Agent 目录下的 `prompts.py`**（不存在 `backend/prompts/` 目录），禁止在代码中硬编码长段 Prompt。
- **结构化日志**: 用 `get_logger(__name__)` 获取日志器，采用 `logger.info("事件名", key=value)` 结构化写法。
- **配置**: 所有环境配置通过 `backend/config.py` 的 `get_settings()` 读取（读根目录 `.env.local`），涉及 API Key / Base URL / 模型名必须用环境变量。

## 项目结构

```
backend/
├── config.py                  # 配置中心（Pydantic Settings，读 .env.local）
├── dependencies.py            # FastAPI 依赖注入（get_db / get_current_user）
├── main.py                    # FastAPI 入口（lifespan：幂等迁移 + 本地模型预热）
├── agents/                    # 各 Agent 实现（每 Agent 独立 State/Node/Graph/Prompt）
│   ├── diagnosis/             #   故障诊断 Agent（当前最完整，唯一闭环）
│   │   ├── graph.py           #     9 节点图：parse→load→diag_tracks→context→report→sufficiency
│   │   │                      #                └─(追问 interrupt)→apply→diag_tracks 循环
│   │   ├── nodes.py           #     节点函数 + 纯函数（故障码/现象提取、三轨匹配）
│   │   ├── diag_tree.py       #     诊断树：DiagTree/DiagNode，故障码精确 + 现象 Jaccard 模糊匹配
│   │   ├── kb_client.py       #     知识库检索适配层（委托 retriever.hybrid_retrieve）
│   │   ├── state.py           #     DiagnosisState + DiagnosisReport 等 Schema
│   │   └── prompts.py
│   ├── knowledge/             #   运维知识库 Agent（RAG 检索→生成）
│   │   ├── graph.py           #     retrieve → generate → END
│   │   ├── nodes.py           #     retrieve_node / generate_node（重试 2 次 + 转人工兜底）
│   │   ├── state.py           #     KnowledgeState + KnowledgeResult
│   │   └── prompts.py
│   ├── ticket/                #   工单管理 Agent（真实落库 + 审计）
│   │   ├── graph.py           #     create_ticket → END
│   │   ├── nodes.py           #     create_ticket_node（结构化归一化 → 真实落库）
│   │   ├── repo.py            #     数据访问层：create/get/list/update_status + ticket_logs 审计
│   │   ├── state.py           #     TicketState + TicketSchema（⚠️ Schema 在此，不在 schemas.py）
│   │   └── schemas.py         #     仅 TICKET_STATUS_FLOW 状态机常量
│   ├── after_sale/            #   备件与售后协调 Agent（保修查库 / 配件桩 / 预约落库）
│   │   ├── graph.py           #     route_request_type → (query_warranty | order_part | prepare_appointment→tools)
│   │   ├── nodes.py           #     route_request_type_node + query_warranty/order_part/prepare_appointment
│   │   ├── tools.py           #     @tool query_warranty(查 devices 表) / check_part_stock(查 parts 表) / create_appointment(写库)
│   │   ├── state.py           #     AfterSaleState + WarrantyInfo/PartOrder
│   │   └── prompts.py
│   ├── asset/                 #   设备台账 Agent（⬜ 规划中，Device 表已建）
│   └── inspection/            #   点检巡检 Agent（⬜ 规划中，对标宝信 AI 点巡检）
├── core/                      # 核心基础设施
│   ├── llm_factory.py         #   LLM 工厂：get_llm / get_structured_llm / clear_cache
│   ├── retry.py               #   with_retry：重试2次→AgentFallbackHandler→系统兜底
│   ├── exceptions.py          #   EqcsAgentBaseError + 8 个子类
│   ├── logger.py              #   结构化日志（_Logger 包装，configure_logging 压噪）
│   └── query_classifier.py    #   MiniLM general/specialized 二分类（⚠️ 未接入链路）
├── api/                       # 对外 API（挂载到 /api/v1）
│   ├── router.py              #   路由聚合：auth / chat / knowledge / ticket / diagnosis / after_sale
│   ├── chat.py                #   统一对话 SSE 入口：规则拦截 + LLM 路由(6类) + 流式执行器（接 Supervisor）
│   ├── auth.py                #   POST /login、GET /me（JWT）
│   ├── knowledge.py           #   POST /knowledge/chat、/chat/stream、会话历史
│   ├── ticket.py              #   工单 CRUD + 状态机流转 + 审计日志（真实落库 PostgreSQL）
│   ├── diagnosis.py           #   POST /diagnosis 非流式诊断（支持 interrupt 追问 + resume）
│   └── after_sale.py          #   POST /after_sale 售后协调（保修/配件/预约）
├── db/                        # 数据库
│   ├── models.py              #   User/Customer/Device/Ticket/TicketLog/Conversation/
│   │                          #   KnowledgeDocument/AfterSaleAppointment（8 张表）
│   └── migrations.py          #   run_migrations()：create_all + 幂等 CREATE INDEX
├── knowledge_base/            # 知识库引擎（RAG 管线）
│   ├── loader.py              #   load_pdf/markdown/docx/xlsx + load_document 分发
│   ├── splitter.py            #   split_documents（PDF/MD 标题/DOCX 启发式/XLSX 每50行）
│   ├── embedder.py            #   BGEMEmbedder 单例（encode 输出 dense+sparse）
│   ├── writer.py              #   KnowledgeBaseClient：建 collection + 写入 Milvus（写端）
│   ├── retriever.py           #   hybrid_retrieve(query, device_model, top_k) 模块级函数
│   ├── reranker.py            #   BGEReranker 单例（rerank_with_confidence）
│   └── contextual.py          #   Contextual RAG：LLM 生成定位描述拼到 chunk 前
├── session_state.py           # 会话状态注册表（进程级，记录 pipeline 中断后的续跑标记）
└── supervisor.py              # ✅ Supervisor 主编排服务类（run_*/stream_*，编排 4 个 Agent 图）

data/                          # 知识库数据文件（不要直接修改，走脚本/后台）
├── diag_tree_full.yaml        #   诊断树 245 节点（D001-D013 手工 + D101-D334 自动转换）
└── diag_examples.json         #   数控机床维修实例 234 条（含 OCR 乱码）

deploy/                        # docker-compose / Dockerfile / nginx.conf / .env.example
docs/                          # architecture.md / api.md / deployment.md / diagnosis-flow.html
scripts/                       # migrate / seed_dev_data / demo_diagnosis / build_knowledge_base / evaluate 等
tests/                         # test_diagnosis.py（50+ 用例）等
```

## Agent 图结构说明

### 目标主编排图（规划中，尚未实现）

```
[Start] → ① 意图识别与路由
               │
        ┌──────┼──────┬───────────┐
        ▼      ▼      ▼           ▼
    ② 台账  ③ 知识库 ④ 故障诊断  ⑦ 备件售后
               │      │           │
               │      ▼           │
               │  ⑤ 点检巡检      │
               │      │           │
               └──┬───┘           │
                  ▼               │
              ⑥ 工单管理 ◄────────┘
                  │
                 ⬇
           [End / 转人工]
```

**当前实现**：各 Agent 的 LangGraph 图已独立实现（`build_*_graph()`），并由 `backend/supervisor.py`（服务类，非巨型 StateGraph）编排。`api/chat.py` 的 4 个流式执行器已接入各 Agent Graph（`astream updates/values`），pipeline 模式串联诊断 → 工单 → 售后。诊断 interrupt 追问通过 `thread_id=session_id` + `graph.get_state(config).next` 检测 + `Command(resume=...)` 恢复。

### 每个 Agent 内部节点模式

遵循 Serial Pipeline + Retry/Fallback 模式（以故障诊断 Agent 为完整范例）：

```python
def build_diagnosis_graph():
    """故障诊断 Agent 的 LangGraph 图（最完整实现，含追问循环）。"""
    builder = StateGraph(DiagnosisState)
    builder.add_node("parse_input",          parse_input_node)
    builder.add_node("load_diag_tree",       load_diag_tree_node)
    builder.add_node("run_diag_tracks",      run_diag_tracks_node)   # 三轨并行：故障码精确/现象模糊/LLM假设
    builder.add_node("assemble_context",     assemble_context_node)
    builder.add_node("generate_report",      generate_report_node)
    builder.add_node("check_sufficiency",    check_sufficiency_node) # 低置信度 → 追问
    builder.add_node("ask_clarify",          ask_clarify_node)       # interrupt 暂停等用户回答
    builder.add_node("apply_clarify_answer", apply_clarify_answer_node)
    builder.add_node("route_next",           route_next_node)
    builder.add_edge(START, "parse_input")
    # ... 线性链 + 条件路由 + 追问循环
    return builder.compile(checkpointer=_get_checkpointer())  # 进程级 MemorySaver 单例
```

**重要**：诊断 Agent 的 `MemorySaver` 必须是**进程级单例**（`_get_checkpointer()`），否则 interrupt 追问跨请求 resume 会失败。生产环境应切换 `AsyncPostgresSaver`。

### 意图路由（当前实现）

`api/chat.py` 实现了两级意图识别：
1. **`_pre_filter()` 规则拦截**：五类社交/元场景（问候/感谢/道别/身份/功能）零 Token 直接返回模板。
2. **`_llm_route()` LLM 路由**：归类到 6 类——knowledge / diagnosis / ticket / after_sale / pipeline / clarify，降级默认 knowledge。

## 当前实现状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 统一对话入口（SSE） | ✅ 已实现 | 规则拦截 + LLM 路由（6 类意图）→ Supervisor 分发流式执行器（接 Agent Graph） |
| 故障诊断 Agent | ✅ 完整闭环 | 9 节点 LangGraph + 追问循环 + 245 节点诊断树 + 置信度/转人工 + 离线 demo |
| 知识库 RAG 管线 | ✅ 完整闭环 | 加载/分块/BGE-M3 嵌入/Milvus 写入+混合检索/BGE 精排/Contextual RAG，示例库 `CNC-1000` 已建（14 chunk） |
| 知识库 Agent | ✅ 已实现 | retrieve → generate，结构化输出 |
| 数据库层 | ✅ 已实现 | 8 张表 ORM + 幂等迁移 + 工单审计日志 |
| 工单管理 Agent | ✅ 已实现 | 结构化生成 + 真实落库（tickets）+ 状态机流转 + ticket_logs 审计 |
| 售后协调 Agent | ✅ 已实现 | 保修查 devices 表 / 配件桩 / 预约写 after_sale_appointments；图拓扑修复 |
| 工单 API | ✅ 已实现 | POST/GET/PATCH 全套 + JWT 鉴权 + 状态机校验（非法流转 400） |
| 故障诊断 API | ✅ 已实现 | POST /diagnosis 非流式 + interrupt 追问 + resume |
| 售后 API | ✅ 已实现 | POST /after_sale 保修/配件/预约 |
| 意图识别 | 🟡 散落实现 | 规则 + LLM 路由已工作；独立 Agent 目录与 MiniLM 分类器待接入 |
| 设备台账 Agent | ⬜ 规划中 | Device 表已建，Agent 待实现 |
| 点检巡检 Agent | ⬜ 规划中 | 对标 AI 点巡检智能体 |
| Supervisor 主编排 | ✅ 已实现 | `backend/supervisor.py` 服务类：run_*/stream_*，pipeline 串联诊断→工单→售后 |

## 当前已知问题与待办（务必对照）

以下是**当前已知缺陷与待办**，修改代码前先确认这些状态：

1. **`backend/agents/intent/` 目录不存在**。意图路由散落在 `api/chat.py` + `core/query_classifier.py`。若要做独立意图 Agent，需新建该目录。
2. **`backend/supervisor.py` 已实现**（服务类，非巨型 StateGraph）。`chat.py` 已接入 Supervisor。✅
3. **`backend/core/config.py` 与 `backend/core/console.py` 不存在**。配置在 `backend/config.py`；无 console 模块。
4. **`backend/api/admin.py` 不存在**（router.py 中已注释掉）。`backend/api/diagnosis.py`、`after_sale.py` **已补齐**（非空壳）。✅
5. **`backend/db/session.py` 不存在**。会话管理在 `dependencies.py` 的 `get_db`。
6. **`backend/prompts/` 目录不存在**。Prompt 在各 Agent 目录下的 `prompts.py`。
7. **工单状态机**已统一为 `pending→dispatched→processing→waiting_parts→resolved→closed`（`ticket/schemas.py` TICKET_STATUS_FLOW 为权威定义，`state.py`/`db/models.py` 已对齐）。✅
8. **Milvus 写入端已实现**（`backend/knowledge_base/writer.py` 的 `KnowledgeBaseClient`）。`scripts/build_knowledge_base.py` 已完成四步闭环，示例建库：`python scripts/build_knowledge_base.py data/knowledge_sample.md --course-id CNC-1000 --no-context`。✅
9. **`knowledge_base/contextual.py` 调 `get_llm("qa", ...)`** —— 已改为 `"knowledge"`（`_AGENT_MODEL_ROUTING` 有该键）。✅
10. **`api/knowledge.py` 用 `KnowledgeBaseRetriever().hybrid_retrieve(...)` 实例调用** —— 已改为模块级调用。✅
11. **工单创建已真实落库**（ticket_id `TK-YYYYMMDD-XXXXXX` 顺序号，消除 hash 随机化 bug）+ `ticket_logs` 审计。售后保修查 `devices` 表、预约写 `after_sale_appointments` 表；`check_part_stock` 已接 `parts` 备件表真查库（`query_part_stock_from_db`，含在库/缺货/查无/DB 异常四态）。✅
12. **`tests/` 已补齐**：`test_ticket`/`test_after_sale`/`test_supervisor`/`test_knowledge`/`test_intent` 均为真实用例（114 个总用例），离线可跑（FakeLLM/mock DB）。
13. **`scripts/seed_knowledge.py`、`scripts/evaluate.py` 已实装**。✅
    - `seed_knowledge.py`：委托 build_knowledge_base 四步流水线（默认 knowledge_sample.md → CNC-1000）。
    - `evaluate.py`：诊断树匹配层离线评估（examples ↔ 转换节点 roundtrip，零 LLM 依赖）；改现象词表/阈值/停用词后必跑。
    - `preflight_check.py`：演示环境预检（Docker/PG 表/种子数据/Milvus collection/LLM Key）。
14. **演示前端已上线**：`backend/static/`（零构建），`main.py` 挂载 `/`，SSE 全事件渲染 + JWT 登录。
15. **Checkpointer 可切换**：`CHECKPOINTER_BACKEND=memory|postgres`（`diagnosis/graph.py::_get_checkpointer`，postgres 不可用时自动降级 memory 并告警）。

## 注意事项

- 不要直接修改 `data/` 下的知识库数据文件（`diag_tree_full.yaml` / `diag_examples.json`），应通过 `scripts/` 下脚本或管理后台操作。
- 新增 Agent 时：先在 `docs/architecture.md` 更新架构图，遵循已有 State / Node / Graph / Prompt 四层结构，并在 `core/llm_factory.py` 的 `_AGENT_MODEL_ROUTING` 注册模型路由。
- LLM Prompt 修改后，务必在测试集上跑评估（`python scripts/evaluate.py`，离线零 LLM），确认准确率没有下降。
- 故障诊断 Agent 的诊断结论必须标注置信度（高/中/低），置信度为「低」时自动触发追问或转人工流程。
- 工单系统状态变更必须记录操作日志（谁、什么时间、从什么状态变成什么状态、原因），对应 `TicketLog` 审计表，用于后续审计。
- 所有外部 API 调用（ERP、CRM、WMS）必须设置超时（默认 10 秒），超时后走降级路径（参考 `core/retry.py`）。
- 知识库数据属于客户核心资产，本地开发测试一律使用脱敏的示例数据。
- 涉及模型配置（API Key、Base URL、模型名称）必须使用环境变量（`.env.local`），参考 `deploy/.env.example`，禁止硬编码。
