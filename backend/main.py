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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.core.logger import configure_logging, get_logger
from backend.api.router import api_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动预热 → 运行 → 关闭清理。"""
    configure_logging()
    logger = get_logger(__name__)
    logger.info("app.starting", env=settings.app_env, port=settings.app_port)

    # ── ① 数据库 Schema 自动迁移（幂等，每次启动执行）──────────
    try:
        from backend.db.migrations import run_migrations
        await run_migrations()
    except Exception as e:
        logger.warning("app.migrations_failed", error=str(e))

    # ── ② 并行预热三个本地模型（首次加载慢，提前热好）──────────
    try:
        from backend.knowledge_base.reranker import BGEReranker
        from backend.core.query_classifier import QueryClassifier
        from backend.knowledge_base.embedder import BGEMEmbedder

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
    logger.info("app.shutdown_complete")


app = FastAPI(
    title="设备智能客服系统",
    description="工业设备 Multi-Agent 智能运维服务平台 — 产品知识问答 / 故障诊断 / 工单管理 / 售后协调",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS：允许前端开发端口跨域访问 ──────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://localhost:5173", "http://localhost:8080",
        "http://127.0.0.1:3000", "http://127.0.0.1:5173", "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 挂载总路由 ─────────────────────────────────────────────────
app.include_router(api_router, prefix="/api/v1")


# ── 健康检查（运维探活用）───────────────────────────────────────
@app.get("/health", tags=["系统"])
async def health_check():
    return {"status": "ok", "env": settings.app_env}


# ── 演示前端（零构建静态页）─────────────────────────────────────
# 挂载在最后：/api/v1、/docs、/health 等已注册路由优先匹配，
# 其余路径回落到 static 目录（html=True 时 "/" 返回 index.html）。
from pathlib import Path  # noqa: E402

from fastapi.staticfiles import StaticFiles  # noqa: E402

_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
