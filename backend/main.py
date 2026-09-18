"""
main - 设备智能客服系统 FastAPI 入口

多 Agent 智能客服系统 API 服务，基于 FastAPI + Uvicorn。
"""

import asyncio
import os
import sys

# Windows conda 专用：把 base 的 Library/bin 加入 DLL 搜索路径，
# 让 _lzma.pyd 能找到 liblzma.dll（非 Windows 跳过，不影响 Linux/Mac）
if sys.platform == "win32":
    _conda_base_lib_bin = os.path.normpath(
        os.path.join(os.path.dirname(sys.executable), "..", "..", "Library", "bin")
    )
    if os.path.isdir(_conda_base_lib_bin):
        os.add_dll_directory(_conda_base_lib_bin)

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.router import api_router
from backend.config import get_settings
from backend.core.logger import configure_logging, get_logger

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动预热 → 运行 → 关闭清理。"""
    configure_logging()
    logger = get_logger(__name__)
    logger.info("app.starting", env=settings.app_env, port=settings.app_port)

    # ── ⓪ 弱密钥告警（生产必须更换）────────────────────────
    if settings.jwt_secret_key in ("", "change_me_to_a_random_secret_key"):
        logger.warning("app.weak_jwt_secret",
                       hint="JWT_SECRET_KEY 仍为占位值，生产环境必须更换为随机强密钥")

    # ── ① 数据库 Schema 自动迁移（幂等，每次启动执行）──────────
    try:
        from backend.db.migrations import run_migrations
        await run_migrations()
    except Exception as e:
        logger.warning("app.migrations_failed", error=str(e))

    # ── ② 并行预热三个本地模型（首次加载慢，提前热好）──────────
    try:
        from backend.core.query_classifier import QueryClassifier
        from backend.knowledge_base.embedder import BGEMEmbedder
        from backend.knowledge_base.reranker import BGEReranker

        loop = asyncio.get_running_loop()
        await asyncio.gather(
            loop.run_in_executor(None, BGEReranker.get_instance),
            loop.run_in_executor(None, QueryClassifier.get_instance),
            loop.run_in_executor(None, BGEMEmbedder.get_instance),
        )
        logger.info("app.local_models_warmed_up")
    except Exception as e:
        logger.warning("app.local_models_warmup_failed", error=str(e))

    logger.info("app.started")

    yield  # ← 应用运行期间停在这里

    # ── 关闭时执行 ────────────────────────────────────────────
    logger.info("app.shutting_down")
    from backend.core.llm_factory import LLMFactory
    LLMFactory.clear_cache()
    # 释放数据库连接池（优雅关闭，避免 asyncpg 连接泄漏）
    from backend.dependencies import engine
    await engine.dispose()
    logger.info("app.shutdown_complete")


app = FastAPI(
    title="设备智能客服系统",
    description="工业设备 Multi-Agent 智能运维服务平台 — 产品知识问答 / 故障诊断 / 工单管理 / 售后协调",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── 中间件（add_middleware 后添加者在最外层）────────────────────
# 生效顺序（外→内）：CORS → 访问日志/Request-ID → 限流 → 业务路由
from backend.core.middleware import RateLimitMiddleware, RequestLoggingMiddleware  # noqa: E402

app.add_middleware(RateLimitMiddleware, limit_per_minute=settings.rate_limit_per_minute)
app.add_middleware(RequestLoggingMiddleware)

# ── CORS：来源从配置读取（ALLOWED_ORIGINS，逗号分隔），生产改为实际域名 ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 全局异常处理：未捕获异常不向客户端泄漏堆栈 ────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.scope, "request_id", "-")
    logger.error(
        "app.unhandled_exception",
        path=request.url.path,
        request_id=request_id,
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"code": "INTERNAL_ERROR", "message": "服务内部错误，请稍后重试",
                 "request_id": request_id},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求体校验失败：统一错误体，仅保留字段错误摘要（不含提交值）。"""
    fields = [
        f"{'.'.join(str(loc) for loc in err.get('loc', []))}: {err.get('msg', '')}"
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"code": "VALIDATION_ERROR", "message": "请求参数不合法",
                 "detail": fields},
    )


# ── 挂载总路由 ─────────────────────────────────────────────────
app.include_router(api_router, prefix="/api/v1")


# ── 健康检查：/health 存活（轻量）；/readyz 就绪（依赖探测）──────
@app.get("/health", tags=["系统"])
async def health_check():
    return {"status": "ok", "env": settings.app_env}


async def _probe_db() -> None:
    """DB 连通性探针（失败抛异常）。"""
    from sqlalchemy import text

    from backend.dependencies import AsyncSessionLocal
    async with AsyncSessionLocal() as session:
        await session.execute(text("SELECT 1"))


async def _probe_milvus() -> None:
    """Milvus 连通性探针（失败抛异常）。"""
    from pymilvus import MilvusClient
    client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
    client.list_collections()


@app.get("/readyz", tags=["系统"])
async def readiness_check():
    """就绪探针：核心依赖（PostgreSQL / Milvus）可用才返回 200。"""
    checks = {}
    ok = True
    for name, probe in (("postgres", _probe_db), ("milvus", _probe_milvus)):
        try:
            await asyncio.wait_for(probe(), timeout=3.0)
            checks[name] = "ok"
        except Exception as e:
            ok = False
            checks[name] = f"fail: {type(e).__name__}"
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ok" if ok else "degraded", "checks": checks},
    )


# ── 演示前端（零构建静态页）─────────────────────────────────────
# 挂载在最后：/api/v1、/docs、/health 等已注册路由优先匹配，
# 其余路径回落到 static 目录（html=True 时 "/" 返回 index.html）。
from pathlib import Path  # noqa: E402

from fastapi.staticfiles import StaticFiles  # noqa: E402


class NoCacheStaticFiles(StaticFiles):
    """静态资源禁用强缓存：始终协商（ETag 相同返回 304，不重复下载）。

    避免发版后浏览器还用旧 JS/CSS 调新接口（如带鉴权前的旧前端打新后端）。
    """

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.is_dir():
    app.mount("/", NoCacheStaticFiles(directory=str(_static_dir), html=True), name="static")
