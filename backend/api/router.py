"""
api_router - API 聚合路由

所有业务接口通过此模块统一注册，挂载到 /api/v1 下。
"""

from fastapi import APIRouter, Depends

api_router = APIRouter()

# ── 认证（公开：签发 Token，不能要求鉴权）─────────────────────
from backend.api.auth import router as auth_router

api_router.include_router(auth_router, tags=["认证"])

# ── 统一对话入口（鉴权：SSE 可触发全部 Agent 并创建工单）──────
from backend.api.chat import router as chat_router  # noqa: E402
from backend.dependencies import get_current_user  # noqa: E402

api_router.include_router(
    chat_router, prefix="/chat", tags=["对话"],
    dependencies=[Depends(get_current_user)],
)

# ── 产品知识问答 ──────────────────────────────────────────────
from backend.api.knowledge import router as knowledge_router

api_router.include_router(knowledge_router, prefix="/knowledge", tags=["知识问答"])

# ── 工单管理 ──────────────────────────────────────────────────
# ticket router 不再自带 /api/tickets 前缀，统一在此挂载，最终路径 /api/v1/tickets
from backend.api.ticket import router as ticket_router

api_router.include_router(ticket_router, prefix="/tickets", tags=["工单"])

# ── 故障诊断 ──────────────────────────────────────────────────
from backend.api.diagnosis import router as diagnosis_router

api_router.include_router(diagnosis_router, prefix="/diagnosis", tags=["故障诊断"])

# ── 售后协调 ──────────────────────────────────────────────────
from backend.api.after_sale import router as after_sale_router

api_router.include_router(after_sale_router, prefix="/after-sale", tags=["售后协调"])

# ── 管理后台 ──────────────────────────────────────────────────
# admin.py 尚未实现，暂时注释，待完成后取消注释
# from backend.api.admin import router as admin_router
# api_router.include_router(admin_router, prefix="/admin", tags=["管理"])
