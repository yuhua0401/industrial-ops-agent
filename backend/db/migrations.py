"""
migrations - 数据库迁移

启动时自动执行的 Schema 管理（全部幂等，可重复运行）。
规则：
  - 只写 ADD COLUMN IF NOT EXISTS / CREATE INDEX IF NOT EXISTS 等幂等 DDL
  - 禁止写 DROP / TRUNCATE 等破坏性变更
  - ORM 新模型通过 create_all 自动建表，后续字段追加在 _MIGRATIONS 中补丁
"""

from sqlalchemy import text

from backend.dependencies import AsyncSessionLocal
from backend.core.logger import get_logger

logger = get_logger(__name__)

# ── Schema 补丁列表（按时间顺序追加，每条 SQL 必须幂等）──────────
_MIGRATIONS: list[tuple[str, str]] = [
    # ── tickets 索引 ─────────────────────────────────────────
    (
        "idx_tickets_status_created",
        "CREATE INDEX IF NOT EXISTS idx_tickets_status_created "
        "ON tickets (status, created_at DESC)",
    ),
    (
        "idx_tickets_customer_id",
        "CREATE INDEX IF NOT EXISTS idx_tickets_customer_id "
        "ON tickets (customer_id)",
    ),
    (
        "idx_tickets_device_sn",
        "CREATE INDEX IF NOT EXISTS idx_tickets_device_sn "
        "ON tickets (device_sn)",
    ),

    # ── conversations 索引 ───────────────────────────────────
    (
        "idx_conversations_session_created",
        "CREATE INDEX IF NOT EXISTS idx_conversations_session_created "
        "ON conversations (session_id, created_at DESC)",
    ),
    (
        "idx_conversations_customer_id",
        "CREATE INDEX IF NOT EXISTS idx_conversations_customer_id "
        "ON conversations (customer_id)",
    ),

    # ── users 索引 ───────────────────────────────────────────
    (
        "idx_users_email",
        "CREATE INDEX IF NOT EXISTS idx_users_email "
        "ON users (email)",
    ),

    # ── devices 索引 ─────────────────────────────────────────
    (
        "idx_devices_customer_id",
        "CREATE INDEX IF NOT EXISTS idx_devices_customer_id "
        "ON devices (customer_id)",
    ),
    (
        "idx_devices_model",
        "CREATE INDEX IF NOT EXISTS idx_devices_model "
        "ON devices (device_model)",
    ),

    # ── ticket_logs 索引 ─────────────────────────────────────
    (
        "idx_ticket_logs_ticket_created",
        "CREATE INDEX IF NOT EXISTS idx_ticket_logs_ticket_created "
        "ON ticket_logs (ticket_id, created_at DESC)",
    ),

    # ── after_sale_appointments 索引 ──────────────────────────
    (
        "idx_appointments_ticket_id",
        "CREATE INDEX IF NOT EXISTS idx_appointments_ticket_id "
        "ON after_sale_appointments (ticket_id)",
    ),
    (
        "idx_appointments_status",
        "CREATE INDEX IF NOT EXISTS idx_appointments_status "
        "ON after_sale_appointments (status)",
    ),
]


async def run_migrations() -> None:
    """
    应用启动时执行：先建 ORM 模型对应的表，再跑增量 Schema 补丁。

    流程：
        1. create_all — 创建 ORM 模型定义的所有表（幂等，已存在则跳过）
        2. _MIGRATIONS — 逐个执行增量 DDL 补丁
    单条失败只记录警告，不阻断启动。
    """
    from backend.db.models import Base
    from backend.dependencies import engine

    # ── Step 1：ORM 模型 → 建表（幂等）────────────────────────
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("db.create_all_done")
    except Exception as e:
        logger.warning("db.create_all_failed", error=str(e))

    # ── Step 2：增量 Schema 补丁 ──────────────────────────────
    async with AsyncSessionLocal() as session:
        for desc, sql in _MIGRATIONS:
            try:
                await session.execute(text(sql))
                await session.commit()
                logger.debug("db.migration_applied", column=desc)
            except Exception as e:
                await session.rollback()
                err = str(e)
                if "already exists" not in err:
                    logger.warning("db.migration_failed", column=desc, error=err)

    logger.info("db.migrations_done", count=len(_MIGRATIONS))
