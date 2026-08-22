# industrial-ops-agent — 工业设备智能运维多 Agent 平台

面向 **工业设备制造商与使用方** 的多 Agent 智能运维平台，以宝信软件在钢铁/流程行业「设备全生命周期管理」的业务实践为蓝本，融合「工厂运维智能 Agent」与「多 Agent 编排框架」两类项目的设计与实现经验，用大模型智能体重塑设备运维的 **点检 → 诊断 → 报修 → 备件 → 售后** 全链路。

> **参考业务对象**：宝信软件（[iEQMS 设备管理系统](https://product.baosight.com/erp/4285)、AI 点巡检智能体、智慧设备现场管理、故障诊断与预测性维护）
> **参考项目**：[factory-operation-agent](https://gitee.com/zllyws/factory-operation-agent)（工厂运维智能 Agent，SpringBoot + 百炼大模型）、[edu-agent](https://gitee.com/czq401/edu-agent)（LangGraph 多 Agent 编排）

## 为什么做这个

传统设备运维存在三个痛点：

1. **知识沉淀难**——老师傅的经验、维修手册、设备档案散落在文档和脑海里，无法规模化复用；
2. **故障响应慢**——从客户报障、人工判断、创建工单到派单上门，链路长、反复确认信息；
3. **管理靠人盯**——点检计划、设备状态、保修期、备件库存全凭人工记忆和纸质台账，异常发现滞后。

本平台借鉴宝信软件的思路：把工艺与维修经验固化为 **知识库 + 诊断树**，用 **AI 智能体** 把「人找事」变成「事找人」，让设备运维从**被动响应**走向**主动服务**。

## 核心能力

| 能力 | 说明 | 对应宝信实践 |
|---|---|---|
| 设备台账与生命周期 | 设备档案、安装位置、保修期、运行状态一站式管理 | iEQMS 资产管理/状态管理 |
| 运维知识库问答 | RAG 检索设备手册、维修案例，7×24 智能问答 | 运维资料智能问答 |
| 故障诊断推理 | 诊断树 + 大模型推理双引擎，多轮追问，输出带置信度的根因与方案 | 设备故障分析智能体 |
| 智能点检巡检 | 点检任务自动生成、路线规划、异常重点提醒（规划中） | AI 点巡检智能体 |
| 维修工单闭环 | 诊断结论自动生成工单，流转、催单、审计日志全记录 | 点巡检异常→工单闭环 |
| 备件与售后协调 | 保修查询、备件库存核对、上门预约（对接 ERP/WMS） | 智慧设备现场管理 |

## 系统架构

### 主编排图（目标架构）

```
                        ┌──────────────────────────────┐
  客户 / 运维人员 ─────▶ │   ① 意图识别与路由 Agent       │
   (微信/Web/API)        │   规则拦截 + LLM 路由 + 分类器  │
                        └──────────────┬───────────────┘
                                       │
              ┌────────────┬───────────┼───────────┬──────────────┐
              ▼            ▼           ▼           ▼              ▼
        ② 设备台账    ③ 运维知识库  ④ 故障诊断   ⑤ 点检巡检   ⑦ 备件与售后
         Agent          Agent        Agent        Agent         Agent
         (EAM)          (RAG)      (诊断树+LLM)  (AI点巡检)    (工具调用)
              │            │           │            │              │
              │            │           ▼            │              │
              │            │      ⑥ 工单管理 Agent ◀┘              │
              │            │          (闭环+审计)                  │
              └────────────┴───────────┴─────────────────────────────┘
                                            │
                                            ▼
                                     [End / 转人工]
```

### 每个 Agent 内部模式

每个 Agent 内部遵循 **Serial Pipeline + Retry/Fallback** 模式：

```
extract_input → retrieve/reason → generate → check_quality ─┬─ pass → generate → END
                                                            └─ fail → fallback → END
```

- LLM 调用统一走 `retry → fallback → raise` 链路；
- 结构化输出用 Pydantic + function calling；
- 所有外部系统调用（ERP/CRM/WMS）设置超时并带降级路径；
- 诊断结论标注置信度，低置信度自动追问或转人工。

## 技术选型

| 领域 | 选型 | 参考来源 |
|---|---|---|
| 语言 / 编排 | Python 3.11+ · LangGraph 1.0+ · LangChain 1.2+ | edu-agent |
| 大模型 | DeepSeek V4（Flash 主力 / Pro 推理），兼容 OpenAI 接口 | 当前栈 |
| 结构化输出 | Pydantic + `with_structured_output`（function calling） | edu-agent |
| 向量检索 | Milvus 2.4+（Dense + Sparse 混合检索）· BGE-M3 · BGE-Reranker | 当前栈 |
| 后端 | FastAPI + SSE 流式响应 | 当前栈 |
| 数据库 | PostgreSQL + asyncpg + SQLAlchemy 2.0 | 当前栈 |
| 认证 | JWT（python-jose + bcrypt） | 当前栈 |
| 部署 | Docker Compose（私有化部署）· Nginx（反代 + SSE） | 当前栈 |
| 监控（规划） | Langfuse（LLM 链路追踪与评估） | edu-agent |

> 注：参考项目 factory-operation-agent 采用 **Java/SpringBoot + MyBatis-Plus + 阿里云百炼大模型**。若团队更擅长 Java 体系，可将本方案的 Python/LangGraph 部分替换为 SpringBoot + Spring AI 或百炼智能体平台，架构思路（Agent 划分、诊断树、工单闭环）完全通用。

## 路线图

### Phase 1 — 诊断闭环（MVP）
- [ ] 设备台账 Agent（设备档案/保修期/状态查询）
- [ ] 故障诊断 Agent（诊断树 + LLM 推理，带置信度与追问）
- [ ] 工单管理 Agent（诊断结论自动生成工单，状态机 + 审计日志）
- [ ] 统一对话入口（规则拦截 + LLM 路由 + 流式响应）

### Phase 2 — 知识底座
- [ ] 运维知识库（RAG：手册/维修案例入库，混合检索 + 精排）
- [ ] 售后协调 Agent（保修/备件/预约，对接 ERP/WMS）

### Phase 3 — 主动运维
- [ ] 点检巡检 Agent（任务自动生成、路线规划、异常提醒）
- [ ] Supervisor 主编排（多 Agent pipeline 全流程）
- [ ] 管理后台（知识库管理 / Agent 配置 / 对话评估）

### Phase 4 — 预测性维护（远期）
- [ ] 设备状态监测数据接入，预警模型 + 诊断模型

## 项目结构

```
backend/
├── agents/                 # 各 Agent 实现（每 Agent 独立 State/Node/Graph/Prompt）
│   ├── asset/              #   设备台账 Agent
│   ├── knowledge/          #   运维知识库 Agent（RAG）
│   ├── diagnosis/          #   故障诊断 Agent（诊断树 + 追问循环）
│   ├── inspection/         #   点检巡检 Agent（规划中）
│   ├── ticket/             #   工单管理 Agent
│   └── after_sale/         #   备件与售后协调 Agent（Tool）
├── core/                   # 核心基础设施
│   ├── llm_factory.py      #   LLM 工厂（按 Agent 路由 / 结构化输出 / 缓存）
│   ├── retry.py            #   重试与降级（retry → fallback → raise）
│   ├── exceptions.py       #   统一异常体系
│   ├── logger.py           #   结构化日志
│   └── config.py           #   配置中心
├── api/                    # 对外 API（chat / ticket / admin）
├── db/                     # 数据库（models / migrations）
├── knowledge_base/         # RAG 管线（loader/splitter/embedder/retriever/reranker）
├── supervisor.py           # Supervisor 主编排图
└── main.py                 # FastAPI 入口

data/                       # 知识库数据（诊断树 / 维修实例，脱敏）
deploy/                     # docker-compose / Dockerfile / nginx / .env.example
docs/                       # 架构 / API / 部署文档
scripts/                    # 建库 / 迁移 / 评估 / 演示
tests/                      # 测试
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量（填写 DEEPSEEK_API_KEY 等）
cp deploy/.env.example .env.local

# 3. 初始化数据库
python scripts/migrate.py

# 4. 启动服务
uvicorn backend.main:app --reload --port 8000
```

- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

```bash
# 测试
pytest tests/ -v

# 故障诊断离线演示（无需 API Key）
python scripts/demo_diagnosis.py
```

## 相关文档

- [系统架构设计](docs/architecture.md)
- [API 文档](docs/api.md)
- [部署指南](docs/deployment.md)

## License

私有化部署产品，商业授权请联系项目方。
