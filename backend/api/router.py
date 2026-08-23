"""
api_router - API 聚合路由

所有业务接口通过此模块统一注册，挂载到 /api/v1 下。
"""

from fastapi import APIRouter

api_router = APIRouter()

# ── 认证 ──────────────────────────────────────────────────────
from backend.api.auth import router as auth_router

api_router.include_router(auth_router, tags=["认证"])

# ── 统一对话入口 ──────────────────────────────────────────────
from backend.api.chat import router as chat_router

api_router.include_router(chat_router, prefix="/chat", tags=["对话"])

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
