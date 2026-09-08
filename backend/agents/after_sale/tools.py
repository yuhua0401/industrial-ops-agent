"""
tools - 售后协调 Agent 的 Tool 定义

用于 LangChain Tool Calling，LLM 根据客户需求调用对应的工具。
保修查询、备件库存与预约创建均接 PostgreSQL 真实数据。
"""
from datetime import datetime

from langchain_core.tools import tool
from sqlalchemy import select

from backend.core.logger import get_logger
from backend.db.models import AfterSaleAppointment, Device, Part

logger = get_logger(__name__)

# 售后工具以 AsyncSessionLocal 开短生命周期会话（被 Supervisor/chat.py 调用时无 DI 上下文）。
# 懒加载避免模块导入即触发 dependencies → config 的必填校验（离线测试无 .env.local 时也能 import）。
# 测试通过 monkeypatch tools._session_factory 替换为 fake async context manager。
_session_factory = None


def _get_session_factory():
    """懒加载 AsyncSessionLocal 并缓存（测试可 monkeypatch tools._session_factory）。"""
    global _session_factory
    if _session_factory is None:
        from backend.dependencies import AsyncSessionLocal
        _session_factory = AsyncSessionLocal
    return _session_factory


# ──────────────────────────────────────────────────────────────
# 共享 repo 函数：Tool 与 Node 共用，避免重复 SQL
# ──────────────────────────────────────────────────────────────

async def query_warranty_from_db(device_sn: str) -> dict:
    """查询设备保修信息（devices 表）。

    返回 {device_sn, device_model, purchase_date, warranty_end, status, coverage}；
    设备不存在或 DB 异常时返回 {status: "unknown", error: ...}，**不抛异常**（保持售后降级语义）。
    """
    try:
        async with _get_session_factory()() as session:
            stmt = select(Device).where(Device.device_sn == device_sn)
            result = await session.execute(stmt)
            device = result.scalar_one_or_none()
    except Exception as e:
        logger.warning("after_sale.warranty_db_error", device_sn=device_sn, error=str(e))
        return {"status": "unknown", "error": "保修信息查询服务暂不可用"}

    if device is None:
        return {"status": "unknown", "error": f"未找到设备 {device_sn} 的保修档案"}

    now = datetime.now()
    warranty_end = device.warranty_end
    if warranty_end is None:
        status = "unknown"
    elif warranty_end >= now:
        status = "in_warranty"
    else:
        status = "out_of_warranty"

    return {
        "device_sn":       device.device_sn,
        "device_model":    device.device_model,
        "purchase_date":   (
            device.purchase_date.strftime("%Y-%m-%d") if device.purchase_date else ""
        ),
        "warranty_end":    warranty_end.strftime("%Y-%m-%d") if warranty_end else "",
        "status":          status,
        "coverage":        "整机保修",
    }


async def query_part_stock_from_db(part_no: str) -> dict:
    """查询备件库存（parts 表）。

    返回 {status: in_stock/out_of_stock/not_found/unknown, part_no, name,
    stock_qty, lead_time_days, price, device_models}；
    DB 异常或查无此件**不抛异常**（保持售后降级语义）。
    """
    code = (part_no or "").strip().upper()
    try:
        async with _get_session_factory()() as session:
            stmt = select(Part).where(Part.part_no == code)
            result = await session.execute(stmt)
            part = result.scalar_one_or_none()
    except Exception as e:
        logger.warning("after_sale.part_db_error", part_no=code, error=str(e))
        return {"status": "unknown", "error": "配件库存查询服务暂不可用"}

    if part is None:
        return {"status": "not_found", "error": f"未找到配件 {code} 的库存档案"}

    return {
        "status": "in_stock" if part.stock_qty > 0 else "out_of_stock",
        "part_no": part.part_no,
        "name": part.name,
        "stock_qty": part.stock_qty,
        "lead_time_days": part.lead_time_days,
        "price": part.price,
        "device_models": part.device_models,
    }


def _next_appointment_id() -> str:
    """生成预约号 APPT-YYYYMMDD-XXXXXX（时间戳 + 随机后缀，够用即可）。"""
    import random

    return f"APPT-{datetime.now().strftime('%Y%m%d')}-{random.randint(0, 999999):06d}"


async def create_appointment_record(
    device_sn: str,
    customer_id: str,
    scheduled_time: str,
    address: str = "",
    ticket_id: str = "",
    engineer_name: str = "",
) -> dict:
    """写入 after_sale_appointments 表，返回预约记录 dict。

    scheduled_time 支持 "YYYY-MM-DD HH:MM" 字符串；解析失败降级为当前时间+3天。
    """
    try:
        scheduled = datetime.fromisoformat(scheduled_time)
    except (ValueError, TypeError):
        # 兜底：默认 3 天后同一时刻
        from datetime import timedelta

        scheduled = datetime.now() + timedelta(days=3)
        scheduled_time = scheduled.isoformat(timespec="minutes")

    appointment_id = _next_appointment_id()
    record = {
        "appointment_id": appointment_id,
        "ticket_id":      ticket_id,
        "customer_id":    customer_id,
        "device_sn":      device_sn,
        "engineer_name":  engineer_name,
        "scheduled_time": scheduled_time,
        "address":        address,
        "status":         "scheduled",
    }
    # ORM 插入需要 datetime 对象（asyncpg 拒绝字符串传入 TIMESTAMP 列）；
    # ticket_id 为空串时置 None（外键列 nullable=True，空串会触发外键校验失败）
    orm_record = {
        **record,
        "scheduled_time": scheduled,
        "ticket_id":      ticket_id or None,
    }
    try:
        async with _get_session_factory()() as session:
            session.add(AfterSaleAppointment(**orm_record))
            await session.commit()
    except Exception as e:
        logger.warning("after_sale.appointment_db_error", device_sn=device_sn, error=str(e))
        return {"error": "预约创建失败，请稍后重试或联系人工客服", **record}

    logger.info(
        "after_sale.appointment_created",
        appointment_id=appointment_id, device_sn=device_sn,
    )
    return record


# ──────────────────────────────────────────────────────────────
# LangChain Tool 定义
# ──────────────────────────────────────────────────────────────

@tool
async def query_warranty(device_sn: str) -> str:
    """查询设备保修信息。传入设备序列号，返回保修状态和截止日期。"""
    info = await query_warranty_from_db(device_sn)
    if info.get("status") == "unknown":
        return f"设备 {device_sn}：{info.get('error', '保修信息查询失败')}"
    status_text = {
        "in_warranty": "保修期内",
        "out_of_warranty": "已过保修期",
    }.get(info["status"], "保修状态未知")
    return (
        f"设备 {info['device_sn']}（{info['device_model'] or '未知型号'}）保修状态："
        f"{status_text}，保修截止 {info['warranty_end']}，覆盖范围：{info['coverage']}。"
    )


@tool
async def check_part_stock(part_no: str) -> str:
    """查询配件库存。传入配件编号，返回库存数量和预计发货时间。"""
    info = await query_part_stock_from_db(part_no)
    if info["status"] == "unknown":
        return f"配件 {part_no}：{info.get('error', '配件库存查询失败')}"

    if info["status"] == "not_found":
        return f"未找到配件 {part_no} 的库存档案，请核对配件编号后重试。"

    name = info["name"]
    qty = info["stock_qty"]
    lead = info["lead_time_days"]
    if info["status"] == "out_of_stock":
        return (
            f"配件 {name}（{part_no}）当前缺货，补货周期约 {lead} 天，"
            f"可先创建上门服务预约或联系人工客服。"
        )
    price_text = f"，单价 {info['price']:.2f} 元" if info.get("price") else ""
    return (
        f"配件 {name}（{part_no}）库存 {qty} 件，预计 {lead} 天内发货{price_text}。"
    )


@tool
async def create_appointment(device_sn: str, customer_address: str, prefer_time: str) -> str:
    """创建上门服务预约。返回预约确认信息和工程师预计到达时间。"""
    record = await create_appointment_record(
        device_sn=device_sn,
        customer_id="",
        scheduled_time=prefer_time,
        address=customer_address,
    )
    if "error" in record:
        return record["error"]
    return (
        f"已为您预约上门服务，预约号 {record['appointment_id']}，"
        f"设备 {record['device_sn']}，地址 {record['address'] or '待确认'}，"
        f"时间 {record['scheduled_time']}。"
    )
