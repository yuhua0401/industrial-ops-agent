"""
models - 数据库 SQLAlchemy ORM 模型

设备智能客服系统的全部数据表定义。
"""

from datetime import datetime

from sqlalchemy import (
    Column, String, Text, DateTime, Integer, Float, Boolean, Enum, JSON, ForeignKey,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """ORM 基类。"""
    pass


# ═══════════════════════════════════════════════════════════════
# 用户与认证
# ═══════════════════════════════════════════════════════════════

class User(Base):
    """系统用户表（管理后台 + 客户端登录）。"""
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    username      = Column(String(64), unique=True, nullable=False, index=True)
    email         = Column(String(128), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    role          = Column(String(32), nullable=False, default="customer")  # admin / engineer / customer
    tenant_id     = Column(String(64), nullable=False, default="tenant_default")
    display_name  = Column(String(64), default="")
    phone         = Column(String(32), default="")
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime, default=datetime.now)
    updated_at    = Column(DateTime, default=datetime.now, onupdate=datetime.now)


# ═══════════════════════════════════════════════════════════════
# 客户与设备
# ═══════════════════════════════════════════════════════════════

class Customer(Base):
    """客户（设备使用方）信息表。"""
    __tablename__ = "customers"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    customer_id    = Column(String(32), unique=True, nullable=False, index=True)
    name           = Column(String(128), nullable=False)
    contact_person = Column(String(64), default="")
    contact_phone  = Column(String(32), default="")
    company        = Column(String(128), default="")
    address        = Column(Text, default="")
    tenant_id      = Column(String(64), nullable=False, default="tenant_default")
    created_at     = Column(DateTime, default=datetime.now)
    updated_at     = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class Device(Base):
    """设备台账表（每台已售设备一条记录）。"""
    __tablename__ = "devices"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    device_sn       = Column(String(64), unique=True, nullable=False, index=True)
    device_model    = Column(String(64), nullable=False, index=True)
    customer_id     = Column(String(32), ForeignKey("customers.customer_id"), nullable=False, index=True)
    purchase_date   = Column(DateTime, default=datetime.now)
    warranty_end    = Column(DateTime, nullable=True)
    warranty_status = Column(String(32), default="in_warranty")  # in_warranty / out_of_warranty / grace_period
    status          = Column(String(32), default="active")        # active / retired / scrapped
    installation_site = Column(String(256), default="")
    notes           = Column(Text, default="")
    tenant_id       = Column(String(64), nullable=False, default="tenant_default")
    created_at      = Column(DateTime, default=datetime.now)
    updated_at      = Column(DateTime, default=datetime.now, onupdate=datetime.now)


# ═══════════════════════════════════════════════════════════════
# 工单
# ═══════════════════════════════════════════════════════════════

class Ticket(Base):
    """工单记录表。"""
    __tablename__ = "tickets"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id         = Column(String(32), unique=True, nullable=False, index=True)
    customer_id       = Column(String(32), ForeignKey("customers.customer_id"), index=True)
    customer_name     = Column(String(128))
    customer_contact  = Column(String(64), default="")
    device_model      = Column(String(64))
    device_sn         = Column(String(64), index=True)
    fault_description = Column(Text)
    diagnosis_result  = Column(Text, default="")          # Agent③ 诊断结论
    severity          = Column(String(16), default="medium")  # low / medium / high / critical
    category          = Column(String(32), default="repair")  # repair / maintenance / inspection / inquiry
    status            = Column(String(16), default="pending")  # pending / dispatched / processing / waiting_parts / resolved / closed / cancelled（对齐 ticket/schemas.py TICKET_STATUS_FLOW）
    assigned_engineer = Column(String(64), default="")
    resolution        = Column(Text, default="")           # 最终解决方案
    notes             = Column(Text, default="")
    tenant_id         = Column(String(64), nullable=False, default="tenant_default")
    created_at        = Column(DateTime, default=datetime.now)
    updated_at        = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class TicketLog(Base):
    """工单操作日志表（审计追溯）。"""
    __tablename__ = "ticket_logs"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id   = Column(String(32), ForeignKey("tickets.ticket_id"), nullable=False, index=True)
    operator    = Column(String(64), nullable=False)          # 操作人（用户ID或系统Agent名）
    action      = Column(String(32), nullable=False)          # created / status_changed / assigned / note_added / resolved
    from_status = Column(String(16), default="")              # 变更前状态
    to_status   = Column(String(16), default="")              # 变更后状态
    comment     = Column(Text, default="")                    # 操作备注/原因
    created_at  = Column(DateTime, default=datetime.now)


# ═══════════════════════════════════════════════════════════════
# 对话记录
# ═══════════════════════════════════════════════════════════════

class Conversation(Base):
    """对话记录表。"""
    __tablename__ = "conversations"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_id   = Column(String(64), nullable=False, index=True)
    customer_id  = Column(String(32), index=True)
    message      = Column(Text)
    reply        = Column(Text)
    agent_chain  = Column(JSON, default=list)       # 经过的 Agent 链路 ["intent", "diagnosis", "ticket"]
    intent       = Column(String(32), default="")   # 路由意图
    duration_ms  = Column(Integer, default=0)        # 单轮耗时（毫秒）
    confidence   = Column(Float, default=0.0)        # Agent 置信度
    tenant_id    = Column(String(64), nullable=False, default="tenant_default")
    created_at   = Column(DateTime, default=datetime.now)


# ═══════════════════════════════════════════════════════════════
# 知识库文档
# ═══════════════════════════════════════════════════════════════

class KnowledgeDocument(Base):
    """知识库文档元数据表。"""
    __tablename__ = "knowledge_documents"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    document_id   = Column(String(64), unique=True, nullable=False, index=True)
    file_name     = Column(String(256), nullable=False)
    file_type     = Column(String(16), nullable=False)       # pdf / docx / xlsx / md
    file_size     = Column(Integer, default=0)               # 字节
    chunk_count   = Column(Integer, default=0)               # 分块数
    index_status  = Column(String(32), default="pending")    # pending / indexing / indexed / failed
    device_model  = Column(String(64), default="")           # 关联设备型号（限定检索范围）
    version       = Column(String(16), default="1.0")
    uploaded_by   = Column(String(64), default="")
    error_message = Column(Text, default="")                 # 索引失败原因
    tenant_id     = Column(String(64), nullable=False, default="tenant_default")
    created_at    = Column(DateTime, default=datetime.now)
    updated_at    = Column(DateTime, default=datetime.now, onupdate=datetime.now)


# ═══════════════════════════════════════════════════════════════
# 备件库存
# ═══════════════════════════════════════════════════════════════

class Part(Base):
    """备件库存表（售后配件查询；后续可扩展 WMS 对接）。"""
    __tablename__ = "parts"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    part_no        = Column(String(64), unique=True, nullable=False, index=True)
    name           = Column(String(128), nullable=False)
    category       = Column(String(32), default="")           # 机械 / 电气 / 润滑 / 耗材
    stock_qty      = Column(Integer, default=0)
    lead_time_days = Column(Integer, default=7)               # 补货/发货周期（天）
    price          = Column(Float, default=0.0)
    device_models  = Column(String(256), default="")          # 适用设备型号（逗号分隔）
    tenant_id      = Column(String(64), nullable=False, default="tenant_default")
    created_at     = Column(DateTime, default=datetime.now)
    updated_at     = Column(DateTime, default=datetime.now, onupdate=datetime.now)


# ═══════════════════════════════════════════════════════════════
# 售后预约
# ═══════════════════════════════════════════════════════════════

class AfterSaleAppointment(Base):
    """售后上门服务预约表。"""
    __tablename__ = "after_sale_appointments"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    appointment_id  = Column(String(32), unique=True, nullable=False, index=True)
    ticket_id       = Column(String(32), ForeignKey("tickets.ticket_id"), index=True)
    customer_id     = Column(String(32), nullable=False)
    device_sn       = Column(String(64), nullable=False)
    engineer_name   = Column(String(64), default="")
    scheduled_time  = Column(DateTime, nullable=False)
    address         = Column(String(256), default="")
    status          = Column(String(32), default="scheduled")  # scheduled / en_route / arrived / completed / cancelled
    notes           = Column(Text, default="")
    tenant_id       = Column(String(64), nullable=False, default="tenant_default")
    created_at      = Column(DateTime, default=datetime.now)
    updated_at      = Column(DateTime, default=datetime.now, onupdate=datetime.now)
