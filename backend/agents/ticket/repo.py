"""
repo - 工单数据访问层

工单落库 + 状态机流转 + 审计日志（ticket_logs）。
API 端点（FastAPI DI session）与 Agent 节点（AsyncSessionLocal）共用。

约定：
- 每个函数接收一个 AsyncSession（调用方负责 commit），保持事务边界清晰。
- ticket_id 生成：TK-YYYYMMDD-XXXXXX，按当天最大序号自增；并发冲突由调用方捕获
  IntegrityError 重试。
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.ticket.schemas import TICKET_STATUS_FLOW
from backend.core.logger import get_logger
from backend.db.models import Ticket, TicketLog

logger = get_logger(__name__)


async def _next_ticket_id(session: AsyncSession, now: datetime | None = None) -> str:
    """生成当天递增的工单号 TK-YYYYMMDD-XXXXXX。

    查询当天已有最大序号 +1；无记录则从 000001 开始。
    """
    now = now or datetime.now()
    prefix = f"TK-{now.strftime('%Y%m%d')}-"
    stmt = (
        select(Ticket.ticket_id)
        .where(Ticket.ticket_id.like(f"{prefix}%"))
        .order_by(text("ticket_id DESC"))
        .limit(1)
    )
    result = await session.execute(stmt)
    last_id = result.scalar_one_or_none()
    if last_id is None:
        seq = 1
    else:
        try:
            seq = int(last_id.rsplit("-", 1)[-1]) + 1
        except (ValueError, IndexError):
            seq = 1
    return f"{prefix}{seq:06d}"


def _normalize_ticket_data(data: dict) -> dict:
    """把外部传入的 ticket_data 映射为 Ticket ORM 字段（丢弃无关键）。

    customer_id 为空串时置 None：customers 外键列 nullable=True，
    空串会触发外键校验失败（与 after_sale_appointments.ticket_id 同理）。
    匿名工单仍保留 customer_name / customer_contact 冗余字段记录客户信息。
    """
    return {
        "customer_id":       str(data.get("customer_id", "") or "") or None,
        "customer_name":     str(data.get("customer_name", "") or ""),
        "customer_contact":  str(data.get("customer_contact", "") or ""),
        "device_model":      str(data.get("device_model", "") or ""),
        "device_sn":         str(data.get("device_sn", "") or ""),
        "fault_description": str(data.get("fault_description", "") or ""),
        "diagnosis_result":  str(data.get("diagnosis_result", "") or ""),
        "severity":          str(data.get("severity", "medium") or "medium"),
        "category":          str(data.get("category", "repair") or "repair"),
        "status":            "pending",
        "assigned_engineer": str(data.get("assigned_engineer", "") or ""),
        "resolution":        "",
        "notes":             str(data.get("notes", "") or ""),
        "tenant_id":         str(data.get("tenant_id", "tenant_default") or "tenant_default"),
    }


async def create_ticket_record(
    session: AsyncSession, data: dict, operator: str = "system",
) -> Ticket:
    """创建工单 + 写创建审计日志。返回 Ticket ORM 对象。"""
    ticket_id = await _next_ticket_id(session)
    fields = _normalize_ticket_data(data)
    fields["ticket_id"] = ticket_id

    ticket = Ticket(**fields)
    session.add(ticket)
    await session.flush()  # 拿到 ticket_id（数据库层再保证唯一）

    session.add(TicketLog(
        ticket_id=ticket_id,
        operator=operator,
        action="created",
        from_status="",
        to_status="pending",
        comment="工单由 Agent 自动创建",
    ))

    logger.info("ticket.created", ticket_id=ticket_id, operator=operator)
    return ticket


async def get_ticket_record(session: AsyncSession, ticket_id: str) -> Ticket | None:
    """按工单号查询，不存在返回 None。"""
    stmt = select(Ticket).where(Ticket.ticket_id == ticket_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_ticket_records(
    session: AsyncSession,
    tenant_id: str,
    page: int = 1,
    size: int = 20,
    status_filter: str | None = None,
) -> tuple[list[Ticket], int]:
    """分页查询工单（按租户隔离 + 可选状态过滤）。返回 (list, total)。"""
    stmt = select(Ticket).where(Ticket.tenant_id == tenant_id)
    if status_filter:
        stmt = stmt.where(Ticket.status == status_filter)

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await session.execute(count_stmt)).scalar_one()

    stmt = (
        stmt
        .order_by(Ticket.created_at.desc())
        .offset((page - 1) * size)
        .limit(size)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all()), total


async def update_ticket_status_record(
    session: AsyncSession,
    ticket: Ticket,
    to_status: str,
    comment: str = "",
    operator: str = "system",
) -> bool:
    """校验并执行工单状态流转，成功写审计日志。非法流转返回 False（不抛）。"""
    allowed = TICKET_STATUS_FLOW.get(ticket.status, [])
    if to_status not in allowed:
        logger.warning(
            "ticket.invalid_transition", ticket_id=ticket.ticket_id,
            from_status=ticket.status, to_status=to_status, allowed=allowed,
        )
        return False

    from_status = ticket.status
    ticket.status = to_status
    session.add(TicketLog(
        ticket_id=ticket.ticket_id,
        operator=operator,
        action="status_changed",
        from_status=from_status,
        to_status=to_status,
        comment=comment,
    ))
    logger.info(
        "ticket.status_changed", ticket_id=ticket.ticket_id,
        from_status=from_status, to_status=to_status, operator=operator,
    )
    return True


# ── 模块级 session 工厂（懒加载，供 Agent 节点使用；测试可 monkeypatch）──
_session_factory = None


def _get_session_factory():
    """懒加载 AsyncSessionLocal（避免离线导入时触发 dependencies → config 必填校验）。"""
    global _session_factory
    if _session_factory is None:
        from backend.dependencies import AsyncSessionLocal
        _session_factory = AsyncSessionLocal
    return _session_factory


async def create_ticket_via_factory(data: dict, operator: str = "system") -> dict:
    """Agent 节点落库入口：自行开短生命周期 session 并 commit。

    返回 {ticket_id, created, error}。DB 异常降级 created=False（不抛）。
    """
    try:
        async with _get_session_factory()() as session:
            ticket = await create_ticket_record(session, data, operator=operator)
            await session.commit()
            return {"ticket_id": ticket.ticket_id, "created": True, "error": ""}
    except Exception as e:
        logger.error("ticket.create_db_failed", error=str(e))
        return {"ticket_id": "", "created": False, "error": str(e)}
