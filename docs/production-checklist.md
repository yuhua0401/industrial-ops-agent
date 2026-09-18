# 生产上线检查清单（Production Readiness Checklist）

> 状态标记：✅ 已完成 · 🟡 部分完成 · ⬜ 未做（上线前需评估）
> 最近更新：2026-09（企业级加固批次）

## 1. 安全

- ✅ 全部业务 API 要求 JWT 鉴权（chat / knowledge / tickets / diagnosis / after-sale；`/login`、`/health`、`/readyz`、静态页公开）
- ✅ CORS 来源配置化（`ALLOWED_ORIGINS`），默认仅本地开发端口
- ✅ 应用层限流：单 IP 滑动窗口（`RATE_LIMIT_PER_MINUTE`，默认 120/min，超限 429）
- ✅ 网关层限流：nginx `limit_req` 20r/s burst 40（`deploy/nginx.conf`）
- ✅ 全局异常处理：未捕获异常返回统一 500（`INTERNAL_ERROR` + request_id），堆栈仅入日志
- ✅ JWT 弱密钥启动告警（占位密钥时 warning）
- ✅ 容器非 root 运行 + HEALTHCHECK（`deploy/Dockerfile`）
- ⬜ 密钥管理：生产应使用密钥管理服务（Vault / 云 KMS）而非 .env 文件；`JWT_SECRET_KEY` 必须为随机强密钥
- ⬜ RBAC 角色校验：`get_current_user` 已返回 role，但尚未按角色限制端点（如管理接口）；当前仅登录校验
- ⬜ HTTPS/TLS 终结：需在 nginx 或负载均衡配置证书（nginx.conf 未含 443 server 块）
- ⬜ 生产关闭 `/docs` `/redoc`（nginx 注释掉对应 location，或 FastAPI `docs_url=None`）

## 2. 数据

- ✅ 幂等自动迁移（启动时 `run_migrations`），种子脚本幂等
- ✅ 工单状态机 + ticket_logs 审计日志
- ✅ 备件库存 / 售后预约 / 点检（若启用）落库
- ⬜ **Alembic 版本化迁移**：当前为"比对建表"式幂等迁移，无版本链；schema 变更审计与回滚缺失
- ⬜ 数据库备份策略（每日 pg_dump + 异地存储）与恢复演练
- ⬜ Milvus 数据备份（collection snapshot）与重建 SOP（建库脚本已具备）
- ⬜ 敏感数据加密：客户联系方式等 PII 字段落盘加密

## 3. 可观测性

- ✅ 结构化日志（`logger.info("event", key=value)` 风格）
- ✅ 每请求 X-Request-ID（响应头回传 + 访问日志），支持网关透传
- ✅ /health（存活，轻量）与 /readyz（就绪，探测 PG + Milvus）
- ✅ 优雅关闭：释放 LLM 缓存与数据库连接池
- ⬜ 指标暴露（Prometheus /metrics：QPS、时延分位、Agent 成功率、LLM token 消耗）
- ⬜ 链路追踪（Langfuse / OpenTelemetry，覆盖 LLM 调用链与 Agent 编排）
- ⬜ 告警规则（readyz 5xx、限流命中率、LLM 错误率）

## 4. 可靠性

- ✅ LLM 调用统一 retry → fallback → raise；各 Agent 有降级路径（诊断模板兜底、视觉描述置空、售后查库不抛异常）
- ✅ 诊断 Checkpointer 可切换 `CHECKPOINTER_BACKEND=memory|postgres`（生产用 postgres，失败自动降级 memory）
- ✅ 意图路由置信度真实化（LLM 自评 + 夹取 + 降级压低）
- ⬜ 会话状态 `session_state.py` 为进程内注册表——多副本部署需迁移到 Redis（pipeline 续跑标记）
- ⬜ 限流为单实例内存实现——多副本部署需迁移网关或 Redis
- ⬜ 对外系统（ERP/CRM/WMS）对接与超时演练（当前备件为本地 parts 表）

## 5. 质量门禁

- ✅ ruff 全仓零告警（`ruff check .`，CI 阻断）
- ✅ pytest 离线测试（FakeLLM / mock DB / 内存 SQLite，CI 不依赖外部服务）
- ✅ GitHub Actions CI（lint + test，`.github/workflows/ci.yml`）
- 🟡 mypy：宽松模式运行，存量 6 个错误未清零（CI 暂不含 mypy；清零后加入门禁）
- ⬜ 评估体系：`scripts/evaluate.py` 覆盖诊断树匹配层；LLM 生成质量评估（结论准确率）未建
- ⬜ HTTP 层集成测试扩充（当前仅安全相关；业务流仍以单元测试为主）
- ⬜ 压测（SSE 并发长连接下的连接池与事件循环表现）

## 6. 部署

- ✅ docker-compose 数据面（PG/Milvus/MinIO/etcd）healthcheck 齐全（etcd 除外）
- ✅ backend 容器 healthcheck + `depends_on: service_healthy`
- ✅ 环境预检脚本（`scripts/preflight_check.py`）
- ⬜ 生产编排评审：backend 直接暴露 `${APP_PORT}`，生产应仅经 nginx（移除 ports 映射或绑定 127.0.0.1）
- ⬜ 横向扩容前提：上方的会话状态与限流迁移完成后，backend 方可 `--scale`
- ⬜ 日志采集：容器 stdout → Loki/ELK（日志格式当前为 key=value 文本，非 JSON）
- ⬜ 资源配额：compose 未设 cpu/memory limits；本地模型预热内存峰值需实测后设定

## 7. 上线前动作（按序执行）

```bash
# 1. 生成强密钥并配置 .env.local（对照 deploy/.env.example）
python -c "import secrets; print(secrets.token_urlsafe(48))"   # → JWT_SECRET_KEY

# 2. 拉起数据面并初始化
docker compose -f deploy/docker-compose.yml up -d postgres etcd minio milvus
python scripts/migrate.py && python scripts/seed_dev_data.py
python scripts/build_knowledge_base.py data/knowledge_sample.md --course-id CNC-1000 --no-context

# 3. 预检 + 门禁
python scripts/preflight_check.py   # 必须全绿
pytest tests/ -q && ruff check .

# 4. 生产启动（CHECKPOINTER_BACKEND=postgres）
docker compose -f deploy/docker-compose.yml up -d --build backend nginx

# 5. 部署后验证
curl -s http://<host>/health && curl -s http://<host>/readyz
curl -s -o /dev/null -w "%{http_code}" -X POST http://<host>/api/v1/chat/stream   # 期望 401
```
