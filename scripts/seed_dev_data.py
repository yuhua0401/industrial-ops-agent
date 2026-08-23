"""
seed_dev_data.py - 开发环境种子数据

插入基础数据（用户 / 客户 / 设备），供本地开发调试。
使用脱敏的示例数据；生产环境请勿执行。

用法：
    python scripts/seed_dev_data.py
"""
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bcrypt  # noqa: E402

from backend.core.logger import get_logger  # noqa: E402
from backend.db.models import Customer, Device, User  # noqa: E402
from backend.dependencies import AsyncSessionLocal  # noqa: E402

logger = get_logger(__name__)


async def seed() -> None:
    """插入缺失的基础数据（幂等：已存在的跳过）。"""
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        # ── 用户（工程师 / 客服）──
        users = [
            dict(
                username="engineer", email="engineer@eqcs.com", role="engineer",
                display_name="张工程师", phone="13800000001",
            ),
            dict(
                username="customer", email="customer@eqcs.com", role="customer",
                display_name="李客户", phone="13800000002",
            ),
        ]
        pw_hash = bcrypt.hashpw(b"admin123", bcrypt.gensalt()).decode()
        for u in users:
            exists = (await session.execute(
                select(User).where(User.username == u["username"]),
            )).scalar_one_or_none()
            if exists:
                logger.info("seed.user_exists", username=u["username"])
                continue
            session.add(User(
                username=u["username"], email=u["email"],
                password_hash=pw_hash, role=u["role"],
                tenant_id="tenant_default", display_name=u["display_name"],
                phone=u["phone"], is_active=True,
            ))
            logger.info("seed.user_created", username=u["username"])

        # ── 客户 ──
        customers = [
            dict(
                customer_id="C-1001", name="华东钢铁集团", contact_person="张工",
                contact_phone="13800000001", company="华东钢铁集团",
                address="上海市宝山区友谊路1号",
            ),
            dict(
                customer_id="C-1002", name="中南设备制造", contact_person="李工",
                contact_phone="13800000002", company="中南设备制造有限公司",
                address="武汉市青山区冶金大道1号",
            ),
        ]
        for c in customers:
            exists = (await session.execute(
                select(Customer).where(Customer.customer_id == c["customer_id"]),
            )).scalar_one_or_none()
            if exists:
                logger.info("seed.customer_exists", customer_id=c["customer_id"])
                continue
            session.add(Customer(**c))
            logger.info("seed.customer_created", customer_id=c["customer_id"])

        # ── 设备（保修期内 / 已过保）──
        now = datetime.now()
        devices = [
            dict(
                device_sn="SN-AC-0001", device_model="CNC-1000",
                customer_id="C-1001", purchase_date=now - timedelta(days=200),
                warranty_end=now + timedelta(days=365), warranty_status="in_warranty",
                status="active", installation_site="三号车间",
            ),
            dict(
                device_sn="SN-AC-0002", device_model="CNC-2000",
                customer_id="C-1002", purchase_date=now - timedelta(days=900),
                warranty_end=now - timedelta(days=200), warranty_status="out_of_warranty",
                status="active", installation_site="一车间",
            ),
        ]
        for d in devices:
            exists = (await session.execute(
                select(Device).where(Device.device_sn == d["device_sn"]),
            )).scalar_one_or_none()
            if exists:
                logger.info("seed.device_exists", device_sn=d["device_sn"])
                continue
            session.add(Device(**d))
            logger.info("seed.device_created", device_sn=d["device_sn"])

        await session.commit()
        logger.info("seed.done")


if __name__ == "__main__":
    asyncio.run(seed())
    print("开发种子数据写入完成。")
