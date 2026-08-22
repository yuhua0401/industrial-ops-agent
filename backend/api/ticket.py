"""
ticket - 工单 API

工单 CRUD 接口，供管理后台和 Agent 调用。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# 前缀由 api/router.py 挂载时统一添加（prefix="/tickets"，最终路径 /api/v1/tickets）。
# 这里不写 prefix，避免与 /api/v1 叠加成 /api/v1/api/tickets 的前缀嵌套。
router = APIRouter(tags=["tickets"])


class TicketCreate(BaseModel):
    customer_name: str
    customer_contact: str
    device_model: str
    device_sn: str = ""
    fault_description: str
    severity: str = "medium"
    category: str = "repair"


class TicketResponse(BaseModel):
    ticket_id: str
    status: str
    created_at: str
    assigned_engineer: str = ""


@router.post("")
async def create_ticket(ticket: TicketCreate) -> TicketResponse:
    """创建工单。"""
    # TODO: 写入 PostgreSQL
    return TicketResponse(
        ticket_id=f"TK-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        status="pending",
        created_at=datetime.now().isoformat(),
    )


@router.get("/{ticket_id}")
async def get_ticket(ticket_id: str) -> TicketResponse:
    """查询工单详情。"""
    # TODO: 从 PostgreSQL 查询
    raise HTTPException(status_code=501, detail="工单查询接口待实现")


@router.get("")
async def list_tickets(page: int = 1, size: int = 20) -> list[TicketResponse]:
    """工单列表。"""
    return []
