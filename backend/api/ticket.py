"""
ticket - 工单 API

工单 CRUD 接口，供管理后台和 Agent 调用。
全部端点要求 JWT 鉴权；落库 + 状态机流转 + 审计日志见 agents/ticket/repo.py。
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.ticket.repo import (
    create_ticket_record,
    get_ticket_record,
    list_ticket_records,
    update_ticket_status_record,
)
from backend.agents.ticket.schemas import TICKET_STATUS_FLOW
from backend.core.logger import get_logger
from backend.db.models import Ticket
from backend.dependencies import get_current_user, get_db

# 前缀由 api/router.py 挂载时统一添加（prefix="/tickets"，最终路径 /api/v1/tickets）。
router = APIRouter(tags=["tickets"])
logger = get_logger(__name__)


class TicketCreate(BaseModel):
    """创建工单请求。"""
    customer_id:      str  = Field(default="", description="客户 ID")
    customer_name:    str  = Field(default="", description="客户姓名/企业名")
    customer_contact: str  = Field(default="", description="联系方式")
    device_model:     str  = Field(default="", description="设备型号")
    device_sn:        str  = Field(default="", description="设备序列号")
    fault_description: str = Field(..., min_length=1, max_length=4000, description="故障描述")
    diagnosis_result: str  = Field(default="", description="Agent 诊断结论")
    severity:         str  = Field(default="medium", description="low/medium/high/critical")
    category:         str  = Field(
        default="repair", description="repair/maintenance/inspection/inquiry",
    )
    notes:            str  = Field(default="", description="备注")


class TicketResponse(BaseModel):
    """工单响应。"""
    ticket_id:        str
    customer_id:      str = ""
    customer_name:    str = ""
    customer_contact: str = ""
    device_model:     str = ""
    device_sn:        str = ""
    fault_description: str = ""
    diagnosis_result: str = ""
    severity:         str = "medium"
    category:         str = "repair"
    status:           str = "pending"
    assigned_engineer: str = ""
    resolution:       str = ""
    notes:            str = ""
    created_at:       str = ""
    updated_at:       str = ""


class TicketListResponse(BaseModel):
    """工单分页列表响应。"""
    total: int
    page:  int
    size:  int
    items: list[TicketResponse]


class StatusUpdateRequest(BaseModel):
    """工单状态流转请求。"""
    to_status: str = Field(..., description="目标状态，须符合 TICKET_STATUS_FLOW")
    comment:   str = Field(default="", description="变更原因/备注")


def _to_response(t: Ticket) -> TicketResponse:
    """Ticket ORM → TicketResponse。"""
    return TicketResponse(
        ticket_id=t.ticket_id,
        customer_id=t.customer_id or "",
        customer_name=t.customer_name or "",
        customer_contact=t.customer_contact or "",
        device_model=t.device_model or "",
        device_sn=t.device_sn or "",
        fault_description=t.fault_description or "",
        diagnosis_result=t.diagnosis_result or "",
        severity=t.severity or "medium",
        category=t.category or "repair",
        status=t.status or "pending",
        assigned_engineer=t.assigned_engineer or "",
        resolution=t.resolution or "",
        notes=t.notes or "",
        created_at=t.created_at.isoformat() if t.created_at else "",
        updated_at=t.updated_at.isoformat() if t.updated_at else "",
    )


@router.post("", response_model=TicketResponse, status_code=status.HTTP_201_CREATED)
async def create_ticket(
    ticket: TicketCreate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """创建工单（真实落库 + 写 created 审计日志）。"""
    data = ticket.model_dump()
    data.setdefault("tenant_id", current_user.get("tenant_id", "tenant_default"))
    data.setdefault("customer_id", ticket.customer_id or current_user.get("user_id", ""))

    try:
        record = await create_ticket_record(
            db, data, operator=current_user.get("user_id", "customer"),
        )
    except Exception as e:
        logger.error("ticket.api_create_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "TICKET_CREATE_ERROR", "message": str(e)},
        ) from e
    return _to_response(record)


@router.get("/{ticket_id}", response_model=TicketResponse)
async def get_ticket(
    ticket_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """查询工单详情。"""
    record = await get_ticket_record(db, ticket_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="工单不存在")
    return _to_response(record)


@router.get("", response_model=TicketListResponse)
async def list_tickets(
    page: int = 1,
    size: int = 20,
    status_filter: str | None = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """工单分页列表（按租户隔离 + 可选状态过滤）。"""
    records, total = await list_ticket_records(
        db,
        tenant_id=current_user.get("tenant_id", "tenant_default"),
        page=page,
        size=size,
        status_filter=status_filter,
    )
    return TicketListResponse(
        total=total, page=page, size=size,
        items=[_to_response(r) for r in records],
    )


@router.patch("/{ticket_id}/status", response_model=TicketResponse)
async def update_ticket_status(
    ticket_id: str,
    req: StatusUpdateRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """工单状态流转（校验状态机 + 写 status_changed 审计日志）。"""
    record = await get_ticket_record(db, ticket_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="工单不存在")

    ok = await update_ticket_status_record(
        db,
        record,
        to_status=req.to_status,
        comment=req.comment,
        operator=current_user.get("user_id", "customer"),
    )
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "INVALID_TICKET_TRANSITION",
                "message": f"工单当前状态 {record.status} 不允许流转到 {req.to_status}",
                "allowed": TICKET_STATUS_FLOW.get(record.status, []),
            },
        )
    await db.commit()
    return _to_response(record)
